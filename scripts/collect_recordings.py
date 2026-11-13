"""Prayer frames from recorded broadcasts of the Quran channel's Makkah stream on YouTube (user request 2026-10-29:
more prayer frames for a later v4, recent days plus a few dense ones such as Ramadan Taraweeh; no training yet).

Slow on purpose, at the user's request (YouTube's bot block): one video at a time, every yt-dlp call through the same
gate as the live services (edge/frame_grabber: at least MIN_LOOKUP_SPACING_S apart, back-off after a failure, counted
per day, nothing while YouTube is paused), the download rate-limited, a pause between videos. A bot check stops every
recording download for the rest of the day (state "recordings_blocked_day"; user, 2026-10-29). 1080p video only (no
audio). Every --every seconds of the video one frame goes through the live analysis (edge/shot.ShotAnalyzer with the
original DM-Count, as scripts/finetune_data.py places frames); only frames on a calibrated camera are kept, as JPEG in
data/kept_frames/recordings/<video id>/ with index.csv (housekeeping never deletes data/kept_frames). The video file is
deleted afterwards. Everything stays on this PC (data/ is gitignored); frames are never committed or uploaded.

  .venv/Scripts/python scripts/collect_recordings.py list "ytsearch20:<words>" [--channel-id UC...]
  .venv/Scripts/python scripts/collect_recordings.py fetch <video id> [<video id> ...] [--every 10] [--tag "..."]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge import frame_grabber as fg  # noqa: E402
from edge.kept_frames import KEPT  # noqa: E402

OUT = KEPT / "recordings"
TMP = ROOT / "data" / "recordings_tmp"
RATE_LIMIT = 3 * 2**20  # bytes per second while downloading
PAUSE_S = (10 * 60, 15 * 60)  # between two videos


def blocked_today() -> bool:
    """A bot check today stops every recording download until tomorrow (user, 2026-10-29: the live stream stays safe)."""
    return fg.load_state().get("recordings_blocked_day") == date.today().isoformat()


def gate(reason: str) -> None:
    """Wait until a yt-dlp call is allowed (spacing / back-off / pause shared with the live services), then count it."""
    if blocked_today():
        raise SystemExit("recordings stopped for today (YouTube bot check)")
    while True:
        st = fg.load_state()
        wait = max(fg.youtube_paused(st), st.get("next_lookup_after", 0) - time.time())
        if wait <= 0:
            break
        print(f"  waiting {wait / 60:.1f} min for the yt-dlp gate ({reason})", flush=True)
        time.sleep(min(wait + 5, 600))
    st = fg.load_state()
    today = date.today().isoformat()
    st.setdefault("calls", {})[today] = st.get("calls", {}).get(today, 0) + 1
    st["next_lookup_after"] = time.time() + fg.MIN_LOOKUP_SPACING_S
    fg.save_state(st)
    print(f"  yt-dlp call #{st['calls'][today]} today ({reason})", flush=True)


def failed(e: Exception) -> None:
    """A failed call: the same back-off as the live services (a bot check waits longest)."""
    st = fg.load_state()
    st["failures"] = st.get("failures", 0) + 1
    bot = "not a bot" in str(e) or "Sign in to confirm" in str(e)
    delay = min(fg.LOOKUP_BACKOFF_S["bot" if bot else "error"] * 2 ** (st["failures"] - 1), fg.MAX_LOOKUP_BACKOFF_S)
    st["next_lookup_after"] = time.time() + delay
    if bot:
        st["recordings_blocked_day"] = date.today().isoformat()
    fg.save_state(st)
    print(f"  yt-dlp failed{' (bot check)' if bot else ''}: {str(e)[:200]}; next call in {delay / 60:.0f} min", flush=True)
    if bot:
        raise SystemExit("bot check: recordings stopped for today")


def ok() -> None:
    st = fg.load_state()
    st["failures"] = 0
    fg.save_state(st)


def ydl_opts(**extra) -> dict:
    return {"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}, "deno": {}}, **fg.cookie_options(), **extra}


def cmd_list(query: str, channel_id: str | None) -> None:
    import yt_dlp

    gate(f"list {query[:40]}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts(extract_flat="in_playlist")) as ydl:
            info = ydl.extract_info(query, download=False)
    except Exception as e:  # noqa: BLE001
        failed(e)
        raise SystemExit(1)
    ok()
    for e in info.get("entries") or []:
        if channel_id and e.get("channel_id") not in (channel_id, None):
            continue
        dur = f"{(e.get('duration') or 0) / 60:.0f} min"
        print(f"{e.get('id')}  {dur:>7}  {e.get('channel') or e.get('uploader') or ''} ({e.get('channel_id')})  {e.get('title')}")


def broadcast_start(info: dict) -> datetime:
    """When the recording started (a past live stream's own start; else its upload time)."""
    t = info.get("release_timestamp") or info.get("timestamp")
    if t:
        return datetime.fromtimestamp(t, timezone.utc)
    return datetime.strptime(info.get("upload_date", "19700101"), "%Y%m%d").replace(tzinfo=timezone.utc)


def download(video_id: str) -> tuple[Path, dict]:
    import yt_dlp

    TMP.mkdir(parents=True, exist_ok=True)
    gate(f"download {video_id}")
    opts = ydl_opts(format="bv*[height=1080][ext=mp4]/bv*[height=1080]", outtmpl=str(TMP / "%(id)s.%(ext)s"),
                    ratelimit=RATE_LIMIT, noplaylist=True, sleep_interval_requests=2, retries=3, noprogress=True)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:  # noqa: BLE001
        failed(e)
        raise
    ok()
    return path, info


def frames_of(path: Path, info: dict, every: float, tag: str) -> list[dict]:
    """Every `every` s one frame through the live analysis; frames on a calibrated camera saved, the rest dropped."""
    from edge import shot as shot_mod
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import load_config
    from finetune_data import Clock

    shot_mod.time = Clock
    an = ShotAnalyzer(load_config(), CrowdModel(), Bank(ROOT / "data" / "vloc" / "bank.npz"))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    t0 = broadcast_start(info)
    out_dir = OUT / info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], 0
    for i in range(0, n, max(int(round(every * fps)), 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        okf, img = cap.read()
        if not okf or img is None:
            continue
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        seen += 1
        off = i / fps
        ts = t0 + timedelta(seconds=off)
        Clock.now = ts.timestamp()
        sr = an.analyse(img, ts)
        if sr.cam is None:
            continue
        stem = ts.strftime("%Y%m%dT%H%M%SZ")
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"stem": stem, "video": info["id"], "offset_s": round(off, 1), "camera": sr.camera,
                     "method": sr.method, "how": sr.how, "rows": "+rows" in (sr.how or ""), "tag": tag})
    cap.release()
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "video", "offset_s", "camera", "method", "how", "rows", "tag"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "video.json").write_text(json.dumps({k: info.get(k) for k in (
        "id", "title", "channel", "channel_id", "upload_date", "release_timestamp", "timestamp", "duration", "width",
        "height", "fps")} | {"tag": tag, "every_s": every, "frames_checked": seen, "frames_kept": len(rows),
                             "broadcast_start_utc": t0.isoformat()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def cmd_fetch(ids: list[str], every: float, tag: str) -> None:
    from collections import Counter

    for j, vid in enumerate(ids):
        if j:
            pause = random.uniform(*PAUSE_S)
            print(f"pause {pause / 60:.0f} min before the next video", flush=True)
            time.sleep(pause)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} UTC  {vid}: downloading (1080p, video only, "
              f"{RATE_LIMIT / 2**20:.0f} MB/s max)", flush=True)
        try:
            path, info = download(vid)
        except Exception:  # noqa: BLE001
            continue
        print(f"  {info.get('title')} ({(info.get('duration') or 0) / 60:.0f} min, {path.stat().st_size / 2**30:.2f} GB); "
              f"analysing every {every:g} s", flush=True)
        try:
            rows = frames_of(path, info, every, tag)
        finally:
            path.unlink(missing_ok=True)  # the video itself is not kept
        cams = Counter(r["camera"] for r in rows)
        print(f"  kept {len(rows)} frames on calibrated cameras {dict(cams)}, {sum(r['rows'] for r in rows)} with prayer "
              f"rows -> {OUT / vid}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Arabic titles on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("query", help='"ytsearch20:<words>" or a channel / playlist URL')
    a.add_argument("--channel-id")
    b = sub.add_parser("fetch")
    b.add_argument("ids", nargs="+")
    b.add_argument("--every", type=float, default=10.0, help="seconds of video between analysed frames")
    b.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args.query, args.channel_id)
    else:
        cmd_fetch(args.ids, args.every, args.tag)


if __name__ == "__main__":
    main()
