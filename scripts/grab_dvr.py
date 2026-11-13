"""Save broadcast frames of a past time window from the YouTube live stream's DVR (user request 2026-10-26: raw frames
around prayer times, for ground truth and for testing; frames to data/samples/dvr/ only, gitignored, never uploaded).

The live HLS playlist lists the last ~hour of 5 s segments with their broadcast time (#EXT-X-PROGRAM-DATE-TIME). This
reuses the HLS URL the crowd collector already looked up (data/.ytdlp_state.json: no extra YouTube lookup) and fetches
only the start of each chosen segment (an HTTP range: the segment opens with a key frame), decoded by FFmpeg in memory.
While YouTube is paused (edge/frame_grabber.py --pause-youtube; 2026-10-27) the daemon records the prayer windows live
from SBA's stream instead (SBA keeps only ~30 s, no DVR): one frame every --every s, scaled to 1920x1080 as the live
services see it, named by its arrival time, tag "<day> <prayer> sba".
After each prayer saved from YouTube the daemon places it (scripts/finetune_data.py --windows <day>-<prayer>, in the
background): frames on a calibrated camera are copied to data/kept_frames, which housekeeping never deletes (user,
2026-10-29: collect prayer frames for a later v4).

  .venv/Scripts/python scripts/grab_dvr.py --from 17:45 --to 18:25 --every 10      # Makkah times, today
  .venv/Scripts/python scripts/grab_dvr.py --prayers                             # daemon: every prayer, -10..+35 min
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.frame_grabber import GrabError, ffmpeg_exe, grab_frame, sba_url, youtube_paused  # noqa: E402
from edge.runtime import setup_logging, single_instance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / ".ytdlp_state.json"
OUT = ROOT / "data" / "samples" / "dvr"
DB = ROOT / "data" / "haram.db"
MAKKAH = timezone(timedelta(hours=3))
RANGE_BYTES = 1_500_000  # the segment's first key frame fits in this (1080p)
KEEP_GB = 6.0  # the daemon deletes the oldest frames beyond this
PRAYER_WINDOW = (-10, 35)  # minutes around each prayer time
PRAYERS = ("fajr", "dhuhr", "asr", "maghrib", "isha")


def playlist() -> list[tuple[datetime, float, str]]:
    """(broadcast time UTC, duration s, URL) of every segment in the live playlist."""
    url = json.loads(STATE.read_text())["stream"]["url"]
    lines = urllib.request.urlopen(url, timeout=30).read().decode().splitlines()
    out, t, dur = [], None, 5.0
    for line in lines:
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            t = datetime.fromisoformat(line.split(":", 1)[1].replace("Z", "+00:00"))
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and t is not None:
            out.append((t, dur, line))
            t = t + timedelta(seconds=dur)
    return out


def frame(url: str) -> np.ndarray | None:
    """The segment's first frame, from its first RANGE_BYTES (the whole segment if that is not enough)."""
    for rng in (RANGE_BYTES, None):
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{rng - 1}"} if rng else {})
        data = urllib.request.urlopen(req, timeout=60).read()
        p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "pipe:1"], input=data, capture_output=True, timeout=60)
        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR) if p.stdout else None
        if img is not None:
            return img
    return None


def grab(t0: datetime, t1: datetime, every: float, tag: str = "") -> int:
    segs = [s for s in playlist() if t0 <= s[0] < t1]
    chosen, last = [], None
    for s in segs:
        if last is None or (s[0] - last).total_seconds() >= every - 0.1:
            chosen.append(s)
            last = s[0]
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.csv"
    new = not index.exists()

    def one(s):
        name = s[0].strftime("%Y%m%dT%H%M%SZ") + ".jpg"
        path = OUT / name
        if path.exists():
            return name, None
        try:
            img = frame(s[2])
        except Exception as e:  # noqa: BLE001
            return name, f"error {str(e)[:80]}"
        if img is None:
            return name, "no frame"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return name, f"{img.shape[1]}x{img.shape[0]}"

    n = 0
    with ThreadPoolExecutor(4) as ex, index.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "utc", "makkah", "tag", "result"])
        for s, (name, res) in zip(chosen, ex.map(one, chosen)):
            if res is None:
                continue
            w.writerow([name, s[0].isoformat(timespec="seconds"), s[0].astimezone(MAKKAH).strftime("%H:%M:%S"), tag, res])
            n += res[0].isdigit()
    if segs:
        say(f"{tag or 'window'}: {len(segs)} segments {segs[0][0].astimezone(MAKKAH):%H:%M:%S}-"
              f"{segs[-1][0].astimezone(MAKKAH):%H:%M:%S} Makkah, saved {n} of {len(chosen)} frames")
    else:
        say(f"{tag or 'window'}: no segments in the window (the playlist holds about the last hour)")
    return n


def say(msg: str) -> None:
    import logging

    print(msg, flush=True)
    logging.getLogger("grab_dvr").info(msg)


def prune() -> None:
    files = sorted(OUT.glob("*.jpg"))
    total = sum(f.stat().st_size for f in files)
    while files and total > KEEP_GB * 1e9:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def prayer_times(day: str) -> dict[str, datetime]:
    row = sqlite3.connect(DB).execute(f"SELECT {', '.join(PRAYERS)} FROM prayer_times WHERE date = ?", (day,)).fetchone()
    return {p: datetime.fromisoformat(v).replace(tzinfo=timezone.utc) for p, v in zip(PRAYERS, row or []) if v}


def live_sba(tag: str) -> bool:
    """One frame from SBA's live stream now, saved like a DVR frame (1920x1080, arrival time). True if saved."""
    now = datetime.now(timezone.utc)
    try:
        img = grab_frame(sba_url())
    except GrabError as e:
        say(f"{tag}: SBA frame failed: {str(e)[:120]}")
        return False
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    OUT.mkdir(parents=True, exist_ok=True)
    name = now.strftime("%Y%m%dT%H%M%SZ") + ".jpg"
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    index = OUT / "index.csv"
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "utc", "makkah", "tag", "result"])
        w.writerow([name, now.isoformat(timespec="seconds"), now.astimezone(MAKKAH).strftime("%H:%M:%S"), tag,
                    "1920x1080 sba"])
    return True


def place(window: str) -> None:
    """Place a saved prayer window in the background (its calibrated frames are kept for training; log data/logs/place.log)."""
    with (ROOT / "data" / "logs" / "place.log").open("a", encoding="utf-8") as log_file:
        subprocess.Popen([sys.executable, str(ROOT / "scripts" / "finetune_data.py"), "--windows", window], cwd=ROOT,
                         stdout=log_file, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    say(f"{window}: placing in the background (data/logs/place.log)")


def daemon(every: float) -> None:
    """For every prayer: once the window has ended, save its frames (the DVR keeps an hour, the window is 45 min).
    While YouTube is paused: frames from SBA's live stream during the window instead."""
    done: set = set()
    live: dict = {}  # (day, prayer) -> [frames saved live, last grab (monotonic)]
    while True:
        now = datetime.now(timezone.utc)
        paused = youtube_paused() > 0
        for day in ((now - timedelta(days=1)).date().isoformat(), now.date().isoformat()):
            for p, t in prayer_times(day).items():
                a, b = t + timedelta(minutes=PRAYER_WINDOW[0]), t + timedelta(minutes=PRAYER_WINDOW[1])
                if paused and a <= now < b:
                    st = live.setdefault((day, p), [0, -1e9])
                    if time.monotonic() - st[1] >= every:
                        st[1] = time.monotonic()
                        st[0] += live_sba(f"{day} {p} sba")
                if (day, p) not in done and b <= now < b + timedelta(minutes=15):
                    if (day, p) in live:
                        say(f"{day} {p}: saved {live[(day, p)][0]} frames live from SBA (YouTube paused)")
                        prune()
                    elif paused:
                        say(f"{day} {p}: YouTube paused, window not recorded")
                    else:
                        try:
                            grab(a, b, every, f"{day} {p}")
                            prune()
                            place(f"{day}-{p}")
                        except Exception as e:  # noqa: BLE001
                            say(f"{day} {p}: {e}")
                    done.add((day, p))
        time.sleep(1 if paused else 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="t0", help="Makkah HH:MM today")
    ap.add_argument("--to", dest="t1", help="Makkah HH:MM today")
    ap.add_argument("--every", type=float, default=10.0, help="seconds between saved frames")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prayers", action="store_true", help="run until stopped: save every prayer window")
    args = ap.parse_args()
    if args.prayers:
        if not single_instance("grab_dvr"):
            sys.exit(3)
        setup_logging("grab_dvr")
        daemon(args.every)
        return
    today = datetime.now(MAKKAH).date()

    def at(hm: str) -> datetime:
        h, m = map(int, hm.split(":"))
        return datetime(today.year, today.month, today.day, h, m, tzinfo=MAKKAH).astimezone(timezone.utc)

    grab(at(args.t0), at(args.t1), args.every, args.tag)


if __name__ == "__main__":
    main()
