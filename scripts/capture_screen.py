"""Prayer frames from recorded broadcasts the user plays in their own browser (user request 2026-10-29, after YouTube's
bot checks stopped scripts/collect_recordings.py): no yt-dlp, nothing opened on YouTube by this script; it only reads
the screen.

Two steps (user, 2026-10-29: "don't analyse while capturing", so no frame is missed):
  grab    the user plays one video per monitor, fullscreen. Every --every seconds each monitor is grabbed, cropped to
          the centred 16:9 picture (a 16:10 monitor has black bars top and bottom) and scaled to 1920x1080; every grab
          is saved to data/recordings_tmp/screen_<start>_<display>/<UTC stamp>.jpg. No analysis. The screen and the
          PC stay awake while it runs (SetThreadExecutionState). Stop: --minutes, Ctrl+C, or the file
          data/recordings_tmp/capture.stop.
  filter  after the videos end: each temp folder in time order through the live analysis (edge/shot.ShotAnalyzer
          with the original DM-Count, one analyser per folder, time = the grab's stamp, as collect_recordings.py
          places frames). Frames on a calibrated camera move to data/kept_frames/recordings/<folder>/ with index.csv
          and run.json. Also kept there (user, 2026-10-29): frames of an uncalibrated shot worth calibrating later,
          given as a reference picture (--like data/kept_frames/recordings/refs/wide_rows.jpg: the high wide shot of
          the prayer rows), matched by ORB features and a homography outside the broadcast graphics; camera
          "new:<reference name>". The rest move to data/recordings_rejected/<folder>/ (user: not deleted until v4 is
          trained, then the user decides; housekeeping does not touch it), then the temp folder is removed.
          Every frame is labelled with the prayer and time of day on screen (--prayer / --time-of-day; --label for
          a stretch where the browser played another video, e.g. YouTube autoplay; user, 2026-10-29), in index.csv
          of both folders. --drop "DISPLAY:HHMMSS-HHMMSS" sends a stretch straight to the rejected folder (method
          "dropped"), e.g. a broadcast with subtitles burnt into the picture, which would be read as floor (user,
          2026-10-29: drop rather than mask). --mask "DISPLAY:x0,y0,x1,y1" (1920x1080 px) adds a box to the
          broadcast graphics mask for that screen's analysis, e.g. a sign-language interpreter (user, 2026-10-29); it
          is written to run.json ("extra_graphics_mask") so later steps can mask it too. config/views.json is not
          changed. Dropped before the analysis (user, 2026-10-29): grabs showing YouTube's player controls (their
          red progress bar) and grabs of a paused video (same as the grab before). Not detected: the YouTube page
          when the browser leaves fullscreen, the mouse pointer alone, burnt-in subtitles (a whole video with
          subtitles is dropped with --drop); those stretches are found by eye and given as --drop.
Everything stays on this PC (data/ is gitignored); frames are never committed or uploaded.

  .venv/Scripts/python scripts/capture_screen.py list
  .venv/Scripts/python scripts/capture_screen.py grab [--displays DISPLAY7 DISPLAY1] [--every 1] [--minutes 5]
  .venv/Scripts/python scripts/capture_screen.py filter [screen_<start>_<display> ...] [--like REF.jpg ...] [--tag "..."]
      [--prayer Fajr --time-of-day dawn] [--label "DISPLAY7:102108-102704=Isha,night" ...]
      [--drop "DISPLAY1:102045-102704" ...] [--mask "DISPLAY1:100,700,480,1080" ...]
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import shutil
import sys
import time
from collections import Counter
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge.kept_frames import KEPT  # noqa: E402

OUT = KEPT / "recordings"
TMP = ROOT / "data" / "recordings_tmp"
STOP = TMP / "capture.stop"
REJECTED = ROOT / "data" / "recordings_rejected"
NEW_SHOTS = KEPT / "new_shots"  # --shot NAME=REF,...: uncalibrated shots kept apart, never calibrated or trained on
LIKE_SIZE = (640, 360)  # reference matching on small pictures (the reference may come from a small screen grab)
LIKE_MIN_INLIERS = 40  # homography inliers to count a frame as the reference's shot
RED_RUN_MIN = 150  # px: a straight red line this long = YouTube's progress bar (controls on screen)
PAUSED_DIFF = 0.3  # mean abs difference to the previous grab below this = the video is paused


def youtube_controls(img: np.ndarray) -> bool:
    """YouTube's player controls on a grab: a straight red line (the progress bar) of at least RED_RUN_MIN px."""
    b, g, r = (img[:, :, i] for i in range(3))
    red = ((r > 170) & (g < 70) & (b < 70)).astype(np.int8)
    if not red.any():
        return False
    d = np.diff(np.pad(red, ((0, 0), (1, 1))), axis=1)  # longest horizontal run of red pixels in any row
    starts, ends = np.nonzero(d == 1), np.nonzero(d == -1)
    return bool(len(starts[0]) and (ends[1] - starts[1]).max() >= RED_RUN_MIN)
STAMP = "%Y%m%dT%H%M%SZ"
FIELDS = ["stem", "display", "captured_utc", "video", "prayer", "time_of_day", "camera", "method", "how", "rows", "tag"]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]


def monitors() -> dict[str, tuple[int, int, int, int]]:
    """Display name (DISPLAY7) -> (left, top, right, bottom) in physical pixels."""
    u = ctypes.windll.user32
    out = {}
    proc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                              wintypes.LPARAM)

    def cb(h, dc, r, lp):
        mi = _MonitorInfo()
        mi.cbSize = ctypes.sizeof(_MonitorInfo)
        u.GetMonitorInfoW(h, ctypes.byref(mi))
        rc = mi.rcMonitor
        out[mi.szDevice.replace("\\\\.\\", "")] = (rc.left, rc.top, rc.right, rc.bottom)
        return 1

    u.EnumDisplayMonitors(None, None, proc(cb), 0)
    return out


def picture(bgr: np.ndarray) -> np.ndarray:
    """The centred 16:9 part of a monitor grab, scaled to 1920x1080."""
    h, w = bgr.shape[:2]
    if w * 9 > h * 16:
        cw = h * 16 // 9
        bgr = bgr[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = w * 9 // 16
        bgr = bgr[(h - ch) // 2:(h - ch) // 2 + ch]
    if bgr.shape[:2] != (1080, 1920):
        bgr = cv2.resize(bgr, (1920, 1080), interpolation=cv2.INTER_AREA if bgr.shape[1] > 1920 else cv2.INTER_CUBIC)
    return bgr


def keep_awake(on: bool) -> None:
    es_continuous, es_system, es_display = 0x80000000, 0x00000001, 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system | es_display if on else 0))


def cmd_grab(displays: list[str], every: float, minutes: float, info: dict | None = None) -> None:
    from PIL import ImageGrab

    mons = monitors()
    missing = [d for d in displays if d not in mons]
    if missing:
        sys.exit(f"no such display: {missing} (have {sorted(mons)})")
    start = datetime.now(timezone.utc)
    dirs = {d: TMP / f"screen_{start:{STAMP}}_{d}" for d in displays}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    stops = [STOP] + [TMP / f"capture_{d}.stop" for d in displays]  # all grabs, or one screen's (user: each screen
    for s in stops:                                                  # starts and stops on its own)
        s.unlink(missing_ok=True)
    keep_awake(True)
    end = time.monotonic() + minutes * 60 if minutes > 0 else float("inf")
    print(f"grabbing {', '.join(displays)} every {every:g} s{f' for {minutes:g} min' if minutes > 0 else ''} -> "
          f"{TMP}", flush=True)
    n, late, failed, last_report = 0, 0, 0, time.monotonic()
    nxt = time.monotonic()
    try:
        while time.monotonic() < end and not any(s.exists() for s in stops):
            ts = datetime.now(timezone.utc)
            for d in displays:
                try:
                    im = ImageGrab.grab(bbox=mons[d], all_screens=True)
                except OSError as e:  # Windows refused one grab ("screen grab failed"): skip this second, go on
                    failed += 1
                    print(f"{ts:%H:%M:%S} UTC  {d}: {e} (skipped)", flush=True)
                    continue
                img = picture(cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(dirs[d] / f"{ts:{STAMP}}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            if time.monotonic() - last_report >= 60:
                last_report = time.monotonic()
                print(f"{datetime.now(timezone.utc):%H:%M:%S} UTC  {n} grabs per display"
                      + (f" ({late} late)" if late else ""), flush=True)
            nxt += every
            wait = nxt - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                late += 1
                nxt = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        keep_awake(False)
        for d, p in dirs.items():
            (p / "grab.json").write_text(json.dumps({
                "display": d, "size": [mons[d][2] - mons[d][0], mons[d][3] - mons[d][1]], "every_s": every,
                "started_utc": start.isoformat(), "ended_utc": datetime.now(timezone.utc).isoformat(), "grabs": n,
                "late": late, "failed_grabs": failed} | (info or {}), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{n} grabs per display ({late} late) in {', '.join(p.name for p in dirs.values())}", flush=True)


class Like:
    """Is a frame the same shot as a reference picture? ORB features outside the broadcast graphics, a RANSAC
    homography, at least LIKE_MIN_INLIERS inliers."""

    def __init__(self, path: Path, graphics: np.ndarray):
        self.name = path.stem
        self.orb = cv2.ORB_create(3000)
        self.mask = (cv2.resize(graphics, LIKE_SIZE, interpolation=cv2.INTER_NEAREST) == 0).astype(np.uint8) * 255
        self.kp, self.des = self.features(cv2.imread(str(path)))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, bgr: np.ndarray):
        gray = cv2.cvtColor(cv2.resize(bgr, LIKE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, self.mask)

    def inliers(self, bgr: np.ndarray) -> int:
        kp, des = self.features(bgr)
        if des is None or self.des is None or len(kp) < 10:
            return 0
        good = [m for m, *n in self.bf.knnMatch(self.des, des, k=2) if n and m.distance < 0.75 * n[0].distance]
        if len(good) < 10:
            return 0
        a = np.float32([self.kp[m.queryIdx].pt for m in good])
        b = np.float32([kp[m.trainIdx].pt for m in good])
        _, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
        return int(inl.sum()) if inl is not None else 0


def labeller(prayer: str, time_of_day: str, labels: list[str]):
    """(display, stem) -> (prayer, time of day); a --label "DISPLAY:HHMMSS-HHMMSS=prayer,time" (UTC, inclusive)
    overrides the default for its stretch."""
    spans = []
    for s in labels:
        where, what = s.split("=", 1)
        display, span = where.split(":", 1)
        a, b = span.split("-", 1)
        p, t = (what.split(",", 1) + [time_of_day])[:2]
        spans.append((display, a, b, p.strip(), t.strip()))

    def label(display: str, stem: str) -> tuple[str, str]:
        hms = stem[9:15]
        for d, a, b, p, t in spans:
            if d == display and a <= hms <= b:
                return p, t
        return prayer, time_of_day

    return label


def dropped(drops: list[str], display: str, stem: str) -> bool:
    """Is this grab inside a --drop "DISPLAY:HHMMSS-HHMMSS" stretch (UTC, inclusive)?"""
    for s in drops:
        d, span = s.split(":", 1)
        a, b = span.split("-", 1)
        if d == display and a <= stem[9:15] <= b:
            return True
    return False


def extra_masks(masks: list[str], display: str, ref_size) -> list[list[list[int]]]:
    """--mask "DISPLAY:x0,y0,x1,y1" (1920x1080 px) boxes for this display, as graphics_mask polygons (ref_size px)."""
    out = []
    for s in masks:
        d, box = s.split(":", 1)
        if d != display:
            continue
        x0, y0, x1, y1 = (float(v) for v in box.split(","))
        sx, sy = ref_size[0] / 1920, ref_size[1] / 1080
        out.append([[round(x0 * sx), round(y0 * sy)], [round(x1 * sx), round(y0 * sy)],
                    [round(x1 * sx), round(y1 * sy)], [round(x0 * sx), round(y1 * sy)]])
    return out


def cmd_filter(names: list[str], tag: str, like_paths: list[str], prayer: str = "", time_of_day: str = "",
               labels: list[str] | None = None, drops: list[str] | None = None,
               masks: list[str] | None = None, shots: list[str] | None = None) -> None:
    from edge import shot as shot_mod
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import graphics_mask, load_config
    from finetune_data import Clock

    folders = [TMP / n for n in names] if names else sorted(p for p in TMP.glob("screen_*") if p.is_dir())
    if not folders:
        sys.exit(f"nothing to filter in {TMP}")
    shot_mod.time = Clock
    cfg, model = load_config(), CrowdModel()
    bank = Bank(ROOT / "data" / "vloc" / "bank.npz")
    for src in folders:
        frames = sorted(src.glob("*.jpg"))
        grab = json.loads((src / "grab.json").read_text(encoding="utf-8")) if (src / "grab.json").exists() else {}
        display = grab.get("display") or src.name.rsplit("_", 1)[-1]
        label = labeller(prayer or grab.get("prayer", ""), time_of_day or grab.get("time_of_day", ""), labels or [])
        video_id = (grab.get("video") or "").split(" ", 1)[0]  # grab --video starts with the YouTube id (user:
        extra = extra_masks(masks or [], display, cfg["ref_size"])  # videos with the same title are told apart by id)
        scfg = cfg | {"graphics_mask": list(cfg.get("graphics_mask", [])) + extra}  # this screen's graphics
        likes = [Like(Path(p), graphics_mask(scfg, (1920, 1080))) for p in like_paths]
        shot_likes = [(name, [Like(Path(r), graphics_mask(scfg, (1920, 1080))) for r in refs.split(",")])
                      for name, refs in (s.split("=", 1) for s in shots or [])]
        shot_rows: dict[str, list[dict]] = {name: [] for name, _ in shot_likes}
        dst, rej = OUT / src.name, REJECTED / src.name
        dst.mkdir(parents=True, exist_ok=True)
        rej.mkdir(parents=True, exist_ok=True)
        an = ShotAnalyzer(scfg, model, bank)
        rows, rejected, cams, t0, prev = [], [], Counter(), time.monotonic(), None
        why = Counter()
        print(f"{src.name}: {len(frames)} grabs", flush=True)
        for i, f in enumerate(frames):
            ts = datetime.strptime(f.stem, STAMP).replace(tzinfo=timezone.utc)
            p, tod = label(display, f.stem)
            base = {"stem": f.stem, "display": display, "captured_utc": ts.isoformat(), "video": video_id, "prayer": p,
                    "time_of_day": tod, "tag": tag}
            if dropped(drops or [], display, f.stem):
                shutil.move(str(f), str(rej / f.name))
                rejected.append(base | {"method": "dropped"})
                why["--drop"] += 1
                continue
            img = cv2.imread(str(f))
            if img is None:
                shutil.move(str(f), str(rej / f.name))
                rejected.append(base | {"method": "unreadable"})
                continue
            paused = prev is not None and np.abs(img.astype(np.int16) - prev).mean() < PAUSED_DIFF
            prev = img.astype(np.int16)
            skip = ("youtube ui", "controls") if youtube_controls(img) else \
                ("paused", "same as the grab before") if paused else None
            if skip:  # user, 2026-10-29: drop frames with YouTube's controls, or of a paused video
                shutil.move(str(f), str(rej / f.name))
                rejected.append(base | {"method": "dropped", "how": f"{skip[0]}: {skip[1]}"})
                why[skip[0]] += 1
                continue
            hit = next(((name, n) for name, lks in shot_likes for n in [max(lk.inliers(img) for lk in lks)]
                        if n >= LIKE_MIN_INLIERS), None)
            if hit:  # a --shot, before any camera analysis: its own folder, never calibrated or trained on (user)
                d = NEW_SHOTS / hit[0] / src.name
                d.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(d / f.name))
                shot_rows[hit[0]].append(base | {"camera": f"new:{hit[0]}", "method": "uncalibrated",
                                                 "how": f"like {hit[1]} inliers"})
                continue
            Clock.now = ts.timestamp()
            sr = an.analyse(img, ts)
            camera, method, how = sr.camera, sr.method, sr.how
            if sr.cam is None:
                camera = None
                for lk in likes:
                    n = lk.inliers(img)
                    if n >= LIKE_MIN_INLIERS:
                        camera, method, how = f"new:{lk.name}", "uncalibrated", f"like {n} inliers"
                        break
            if camera is None:
                shutil.move(str(f), str(rej / f.name))
                rejected.append(base | {"method": method, "how": how})
            else:
                shutil.move(str(f), str(dst / f.name))
                cams[str(camera)] += 1
                rows.append(base | {"camera": camera, "method": method, "how": how, "rows": "+rows" in (how or "")})
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(frames)} checked, {len(rows)} kept ({time.monotonic() - t0:.0f} s)", flush=True)
        for name, items in shot_rows.items():
            if items:
                with (NEW_SHOTS / name / src.name / "index.csv").open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(items)
                print(f"  {len(items)} frames of the shot {name} -> {NEW_SHOTS / name / src.name}", flush=True)
        for folder, items in ((dst, rows), (rej, rejected)):
            with (folder / "index.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(items)
        (dst / "run.json").write_text(json.dumps(grab | {
            "tag": tag, "extra_graphics_mask": extra, "frames_checked": len(frames), "frames_kept": len(rows),
            "dropped": dict(why), "cameras": dict(cams),
            "with_rows": sum(r["rows"] for r in rows)}, indent=1), encoding="utf-8")
        if (src / "grab.json").exists():
            shutil.copy2(src / "grab.json", rej / "grab.json")
        shutil.rmtree(src)
        print(f"  kept {len(rows)} of {len(frames)} {dict(cams)}, {sum(r['rows'] for r in rows)} with prayer rows -> "
              f"{dst}; the other {len(frames) - len(rows)} -> {rej} (dropped before analysis: {dict(why)})", flush=True)


def main() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # monitor rectangles and grabs in physical pixels
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("grab")
    g.add_argument("--displays", nargs="+", default=["DISPLAY7", "DISPLAY1"])
    g.add_argument("--every", type=float, default=1.0, help="seconds between two grabs of each monitor")
    g.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0: until stopped)")
    g.add_argument("--video", default="", help="what the screen plays (YouTube id / title), kept in grab.json")
    g.add_argument("--prayer", default="", help="the prayer on screen (the filter's default label)")
    g.add_argument("--time-of-day", default="", help="dawn / day / night (the filter's default label)")
    f = sub.add_parser("filter")
    f.add_argument("folders", nargs="*", help="temp folders (default: every screen_* in data/recordings_tmp)")
    f.add_argument("--like", nargs="*", default=[], help="reference pictures of uncalibrated shots to keep too")
    f.add_argument("--tag", default="")
    f.add_argument("--prayer", default="", help="the prayer the videos show (default label)")
    f.add_argument("--time-of-day", default="", help="dawn / day / night (default label)")
    f.add_argument("--label", action="append", default=[],
                   help='"DISPLAY:HHMMSS-HHMMSS=prayer,time_of_day" (UTC) for a stretch showing another video')
    f.add_argument("--shot", action="append", default=[],
                   help='"NAME=REF.jpg,REF2.jpg": uncalibrated frames of that shot go to data/kept_frames/new_shots/NAME')
    f.add_argument("--mask", action="append", default=[],
                   help='"DISPLAY:x0,y0,x1,y1" (1920x1080 px): extra broadcast graphics, e.g. a sign-language box')
    f.add_argument("--drop", action="append", default=[],
                   help='"DISPLAY:HHMMSS-HHMMSS" (UTC): straight to the rejected folder (e.g. burnt-in subtitles)')
    args = ap.parse_args()
    if args.cmd == "list":
        for name, (l, t, r, b) in sorted(monitors().items(), key=lambda kv: kv[1][0]):
            print(f"{name}: {r - l}x{b - t} at ({l}, {t})")
    elif args.cmd == "grab":
        cmd_grab(args.displays, args.every, args.minutes,
                 {k: v for k, v in (("video", args.video), ("prayer", args.prayer),
                                    ("time_of_day", args.time_of_day)) if v})
    else:
        cmd_filter(args.folders, args.tag, args.like, args.prayer, args.time_of_day, args.label, args.drop, args.mask,
                   args.shot)


if __name__ == "__main__":
    main()
