"""Live shot service (2026-10-26, user request): analyse the broadcast frame on air continuously, as often as the GPU
allows (target every 1-2 s), instead of every 15 s and only for specific cameras. Every frame gets people per m^2 by
the first method that works (edge/shot.py): a calibrated camera, registration against the panorama bank, tracking of
the same shot, or the approximate head-scale estimate; prayer rows give the reading when the frame shows them.

Writes (frames stay in memory; only numbers and small density heatmaps leave this process):
  crowd_m2_frame / crowd_m2_minute   the Mataf metric (path = the method), at most one frame row per camera per
                                     DB_EVERY_S, minutes pooled from every analysed frame; this replaces the crowd
                                     collector's own m2 path (config/m2.json "enabled": false there)
  data/live/heatmap_<camera>.png/.json   the latest heatmap per camera (every frame) and data/live/m2/<camera>_<UTC>
                                     every HEAT_EVERY_S, for "What the AI sees" and the twin's floor map
  data/live/shot.json + heatmap_shot.png   the current shot whatever its method (the approximate one too)
  data/.shot_status.json             frames, rate, methods, GPU memory
  data/samples/pairs/                ground-truth pairs for the inner-ring check (PairCapture; local only, capped)
No GPU memory cap (user, 2026-10-28: no limits).
GPU load (user request 2026-10-26, frame drops on this PC): a shot that stays on air is analysed every 2nd frame (every
frame for HALF_AFTER_CUT_S after a cut); DM-Count is also served to the crowd collector (edge/model_share.py, one copy
on the GPU); one ALIKED + LightGlue (fp16 attention); PyTorch's cache freed after a full search.

  .venv/Scripts/python scripts/live_shot.py [--fps 1]
  .venv/Scripts/python scripts/live_shot.py --frames data/samples/dvr --since 20261026T1500   (offline test, no DB)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# Intel OpenMP (PyTorch's CPU thread pool): idle workers sleep at once instead of spinning 200 ms after every parallel
# region (user request 2026-11-06, a quieter laptop). Same 16 threads, same work split, so the same results; offline
# test at one frame a second: 3.21 -> 0.20 CPU cores, outputs bit-identical. Read when torch loads, so set it first.
os.environ.setdefault("KMP_BLOCKTIME", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge import db  # noqa: E402
from edge.heatmap import density_png, heat_scale  # noqa: E402
from edge.runtime import setup_logging, single_instance  # noqa: E402

log = logging.getLogger("live_shot")
LIVE_DIR = ROOT / "data" / "live"
LIVE_WEIGHTS: str | None = None  # the live model when it is a fine-tuned one (config/m2.json "live_model")
LIVE_NAME: str | None = None  # its name for the readings' "model" column (config/m2.json live_model "use", e.g. "v4")
SPEED_FILE = Path(__file__).resolve().parents[1] / "data" / "live" / "speed.json"  # the tawaf speed sensor's last minute
HIST_DIR = LIVE_DIR / "m2"
STATUS = ROOT / "data" / ".shot_status.json"
BANK = ROOT / "data" / "vloc" / "bank.npz"
DB_EVERY_S = 5.0
HEAT_EVERY_S = 5.0
HEAT_KEEP_S = 600
HALF_AFTER_CUT_S = 10.0  # every frame this long after a cut, then every 2nd frame while the shot stays on air


def atomic(path: Path, data: bytes | str) -> None:
    """Write via a temporary file and a rename, so readers never see half a file. On Windows the rename fails while
    the API has the old file open: retry briefly, then skip this update (the next frame writes it again)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data) if isinstance(data, bytes) else tmp.write_text(data)
    for i in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    log.warning("%s is in use; skipped this update", path.name)


def cam_json(cam) -> dict:
    return {"f": round(float(cam.f), 2), "rvec": [round(float(x), 7) for x in np.ravel(cam.rvec)],
            "tvec": [round(float(x), 5) for x in np.ravel(cam.tvec)], "size": list(cam.size)}


class PairCapture:
    """Ground-truth pairs for the inner ring (user request 2026-10-26): frames where a low or wide camera shows the floor
    1-10 m from the Kaaba with heads >= TRUTH_HEAD px, or a roof view zoomed to heads >= ROOF_TRUTH_HEAD px (a steeper
    truth), and the roof views within +-WINDOW_S of them (the ones before come from a short in-memory buffer). Full frames,
    JPEG, local only (data/samples/pairs, at most MAX_GB; scripts/housekeeping.py keeps it under that), one frame per
    camera per EVERY_S, each with a sidecar (camera, rings, sectors). Never changes a reading; errors are logged only."""
    TRUTH_CAMS = ("cam0_mataf_low", "cam1_mataf_wide")
    ROOF_CAMS = ("cam5_roof", "cam8_roof_b")
    TRUTH_HEAD, ROOF_TRUTH_HEAD = 9.0, 6.0
    WINDOW_S, EVERY_S, MAX_GB = 120.0, 10.0, 2.0
    DIR = ROOT / "data" / "samples" / "pairs"

    def __init__(self):
        self.buf: deque = deque()
        self.last_truth = -1e18
        self.last_saved: dict = defaultdict(lambda: -1e18)
        self.full = False
        self.checked = 0.0

    def offer(self, img, sr) -> None:
        try:
            self._offer(img, sr)
        except Exception:
            log.exception("pair capture failed (the reading is not affected)")

    def _offer(self, img, sr) -> None:
        if sr.reading is None or sr.cam is None or sr.camera not in self.TRUTH_CAMS + self.ROOF_CAMS:
            return
        t = sr.ts.timestamp()
        if t - self.last_saved[sr.camera] < self.EVERY_S:
            return
        head, area = sr.reading.head_px.get("inner"), sr.reading.area.get("inner") or 0.0
        roof = sr.camera in self.ROOF_CAMS
        truth = area >= 20 and head is not None and head >= (self.ROOF_TRUTH_HEAD if roof else self.TRUTH_HEAD)
        if not (truth or roof):
            return
        self.last_saved[sr.camera] = t
        item = (t, sr.camera, truth, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), self._side(sr, truth))
        while self.buf and t - self.buf[0][0] > self.WINDOW_S:
            self.buf.popleft()
        if truth:
            self.last_truth = t
            for b in self.buf:
                self._save(b)
            self.buf.clear()
            self._save(item)
        elif t - self.last_truth <= self.WINDOW_S:
            self._save(item)
        else:
            self.buf.append(item)

    @staticmethod
    def _side(sr, truth: bool) -> dict:
        rd = sr.reading
        return {"ts": sr.ts.isoformat(timespec="seconds"), "camera": sr.camera, "truth": truth, "method": sr.method,
                "how": sr.how, "cam": cam_json(sr.cam),
                "rings": {k: {"raw": rd.density.get(k), "corrected": rd.density_corr.get(k), "area_m2": rd.area.get(k),
                              "count": rd.count.get(k), "head_px": rd.head_px.get(k)} for k in rd.area},
                "sectors": rd.sectors, "rows": sr.rows.__dict__ if sr.rows else None}

    def _save(self, item) -> None:
        if time.monotonic() - self.checked > 300:
            self.checked = time.monotonic()
            size = sum(f.stat().st_size for f in self.DIR.glob("*.jpg")) if self.DIR.exists() else 0
            if size > self.MAX_GB * 1e9 and not self.full:
                log.warning("pair capture: %s is full (%.1f GB); paused until housekeeping makes room", self.DIR, size / 1e9)
            self.full = size > self.MAX_GB * 1e9
        if self.full:
            return
        t, cam, truth, jpg, side = item
        self.DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.fromtimestamp(t, timezone.utc):%Y%m%dT%H%M%SZ}_{cam}{'_truth' if truth else ''}"
        (self.DIR / f"{stem}.jpg").write_bytes(jpg)
        (self.DIR / f"{stem}.json").write_text(json.dumps(side, default=float))


class Writer:
    def __init__(self, to_db: bool = True):
        self.to_db = to_db
        self.minute: datetime | None = None
        self.rows: list[dict] = []
        self.last_db: dict = defaultdict(float)
        self.last_heat: dict = defaultdict(float)
        self.methods: Counter = Counter()
        self.frames = 0
        self.source = "youtube"
        self.speed_fn = None  # the tawaf speed sensor's sectors (LiveSpeed.sectors), asked at every minute's flush

    def frame(self, sr, size: tuple[int, int], source: str = "youtube") -> None:
        w, h = size
        ts = sr.ts
        self.source = db_source(source)
        self.frames += 1
        self.methods[sr.method] += 1
        base = {"ts": ts.isoformat(timespec="seconds"), "method": sr.method, "how": sr.how, "camera": sr.camera,
                "people_m2": sr.people_m2, "source": source, "frame_size": [w, h], "ms": sr.ms}
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        heat = sr.heat if sr.heat is not None else sr.density
        png = density_png(heat, (w, h)) if heat is not None else None
        top = round(heat_scale(heat, (w, h)), 6) if heat is not None else None
        shot = base | {"density_top": top, "rings": sr.reading.density_corr if sr.reading else None,
                       "rows": sr.rows.__dict__ if sr.rows else None, "approx": sr.approx,
                       "cam": cam_json(sr.cam) if sr.cam is not None else None}
        if png:
            atomic(LIVE_DIR / "heatmap_shot.png", png)
        est = getattr(sr, "est", None)
        atomic(LIVE_DIR / "shot.json", json.dumps(shot, default=float))
        if sr.cam is None or not sr.camera or png is None:
            return
        meta = base | {"view_id": sr.camera, "accepted": True, "density_top": top, "from": "m2", "cam": cam_json(sr.cam),
                       "live_model": LIVE_WEIGHTS}  # a fine-tuned live model: the heatmap needs no correction (backend)
        # cells the fixes estimated (edge/m2_fill.LiveFixes): hatched as "estimated" by the AI view and the twin
        est_png = LIVE_DIR / f"heatmap_{sr.camera}_est.png"
        if est is not None and est.any():
            meta["est"] = est_png.name
            ok, buf = cv2.imencode(".png", est.astype(np.uint8) * 255)
            atomic(est_png, buf.tobytes())
        else:
            try:
                est_png.unlink(missing_ok=True)
            except OSError:  # Windows: the API is reading it; meta has no "est", so it is not used
                pass
        atomic(LIVE_DIR / f"heatmap_{sr.camera}.png", png)
        atomic(LIVE_DIR / f"heatmap_{sr.camera}.json", json.dumps(meta, default=float))
        now = time.monotonic()
        if now - self.last_heat[sr.camera] >= HEAT_EVERY_S:
            self.last_heat[sr.camera] = now
            HIST_DIR.mkdir(parents=True, exist_ok=True)
            stamp = ts.strftime("%Y%m%dT%H%M%SZ")
            atomic(HIST_DIR / f"{sr.camera}_{stamp}.png", png)
            atomic(HIST_DIR / f"{sr.camera}_{stamp}.json", json.dumps(meta, default=float))
            cutoff = time.time() - HEAT_KEEP_S
            for f in HIST_DIR.iterdir():
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        if self.to_db and sr.reading and sr.reading.density_corr:
            self.add(sr)

    def add(self, sr) -> None:
        minute = sr.ts.replace(second=0, microsecond=0)
        if self.minute is not None and minute != self.minute:
            self.flush()
        self.minute = minute
        # every analysed frame counts in the minute; one frame row per camera per DB_EVERY_S is stored
        keep = sr.ts.timestamp() - self.last_db[sr.camera] >= DB_EVERY_S
        if keep:
            self.last_db[sr.camera] = sr.ts.timestamp()
        rd, c = sr.reading, sr.cam.centre
        for ring, v in rd.density_corr.items():
            self.rows.append({"ts_utc": sr.ts, "ring": ring, "source": self.source, "path": sr.how, "camera": sr.camera,
                              "people_m2": v, "people_m2_raw": rd.density.get(ring), "count": rd.count.get(ring),
                              "area_m2": rd.area.get(ring), "head_px": rd.head_px.get(ring), "zoom": None,
                              "focal_px": round(float(sr.cam.f), 1), "ptz_err": None, "cam_x": round(float(c[0]), 2),
                              "cam_y": round(float(c[1]), 2), "cam_z": round(float(c[2]), 2),
                              "head_bins": json.dumps(rd.bins) if ring == "all" and rd.bins else None,
                              "sectors": json.dumps(rd.sectors, separators=(",", ":")) if ring == "all" and rd.sectors else None,
                              **(far_cols(rd.far) if ring == "all" else {}), "model": LIVE_NAME, "_keep": keep})

    def flush(self, speed: tuple | None = None) -> str:
        if not self.rows:
            return ""
        if speed is None and self.speed_fn is not None:  # add() flushes when a new minute starts: ask here
            try:
                speed = self.speed_fn()
            except Exception:
                log.exception("tawaf speed sectors failed")
        groups: dict = defaultdict(list)
        for r in self.rows:
            groups[r["ring"]].append(r)
        minutes = []
        for ring, rs in groups.items():
            area = sum(r["area_m2"] for r in rs)
            if area <= 0:
                continue
            # one row per minute and ring, under the source most of its frames came from (both are the main series)
            minutes.append({"minute_utc": self.minute, "ring": ring,
                            "source": Counter(r["source"] for r in rs).most_common(1)[0][0],
                            "people_m2": round(sum(r["people_m2"] * r["area_m2"] for r in rs) / area, 4),
                            "people_m2_raw": round(sum((r["people_m2_raw"] or 0) * r["area_m2"] for r in rs) / area, 4),
                            "area_m2": round(area / len(rs), 1), "n_frames": len(rs),
                            "cameras": ",".join(sorted({r["camera"] or "" for r in rs}))[:200], "model": LIVE_NAME})
            fr = [r for r in rs if r.get("area_m2_far")]
            if ring == "all" and fr:  # far Mataf: pooled the same way, from the frames that saw it
                fa = sum(r["area_m2_far"] for r in fr)
                minutes[-1].update(people_m2_far=round(sum(r["people_m2_far"] * r["area_m2_far"] for r in fr) / fa, 4),
                                   people_m2_far_raw=round(sum(r["people_m2_far_raw"] * r["area_m2_far"] for r in fr) / fa, 4),
                                   area_m2_far=round(fa / len(fr), 1), n_frames_far=len(fr))
        speed_note = blend_speed(minutes, self.rows, speed) if speed else ""
        frames = [{k: v for k, v in r.items() if k != "_keep"} for r in self.rows if r["_keep"]]
        db.insert_m2(same_keys(frames), same_keys(minutes))
        self.rows = []
        return "".join(f" | Mataf {m['ring']} {m['people_m2']:.2f}/m2 ({m['n_frames']} frames)" for m in minutes
                       if m["ring"] == "all") + speed_note


def same_keys(rows: list[dict]) -> list[dict]:
    """Every row gets every key (None where it has none): a batch insert takes its columns from the first row, so a key
    only some rows carry (people_m2_count where the speed changed one ring, the far Mataf on ring "all") failed the
    batch or was dropped."""
    keys = {k for r in rows for k in r}
    return [{k: r.get(k) for k in keys} for r in rows]


RINGS_M = {"inner": (1.0, 10.0), "outer": (10.0, 25.0)}  # edge/m2_metric.RINGS


def blend_speed(minutes: list[dict], rows: list[dict], speed: tuple) -> str:
    """Tawaf speed sensor (user 2026-10-30): per ring, the walking sectors (edge/tawaf_speed.LiveSpeed, the last 2 min)
    take the speed's people per m^2, the stopped ones the count model's (its 30 deg sectors pooled over the minute's
    frames, else the ring's value); the ring's reading is the mean over its sectors, "all" the area-weighted rings.
    Changes the minute rows in place (people_m2_count keeps the count model's value, model "<live>+speed") and writes
    data/live/speed.json for the API and the twin. Returns a log note."""
    from edge.tawaf_speed import SECTOR_DEG, ring_area

    cam, n_pairs, secs = speed
    by = {m["ring"]: m for m in minutes}
    sec_sum: dict = defaultdict(lambda: [0.0, 0.0])  # count model: (corrected count, m^2) per "i3" / "o3"
    for r in rows:
        if r["ring"] == "all" and r.get("sectors"):
            for k, v in json.loads(r["sectors"]).items():
                sec_sum[k][0] += v[1]
                sec_sum[k][1] += v[2]
    out, used = {}, False
    for ring, letter in (("inner", "i"), ("outer", "o")):
        m = by.get(ring)
        if m is None or ring not in secs:
            continue
        vals, speeds = [], []
        for s in secs[ring]:
            if s["rho"] is not None:
                vals.append(s["rho"])
                if s["how"] in ("walking", "slow", "fast"):
                    speeds.append(s["speed"])
            else:  # stopped, or no floor seen: the count model's sector (edge/m2_metric sector numbering)
                a = (s["deg"] + 180) % 360 - 180
                c = sec_sum.get(f"{letter}{int((a + 180) // 30) % 12}")
                vals.append(c[0] / c[1] if c and c[1] > 1 else m["people_m2"])
        info = {"count": m["people_m2"], "speed_ms": round(float(np.median(speeds)), 3) if speeds else None,
                **{k: round(sum(s["how"] == k for s in secs[ring]) / len(secs[ring]), 3)
                   for k in ("walking", "slow", "fast", "estimated", "stopped")}}
        if speeds:
            used = True
            m["people_m2_count"], m["people_m2"] = m["people_m2"], round(float(np.mean(vals)), 4)
            m["model"] = f"{LIVE_NAME}+speed"
        info["value"] = m["people_m2"]
        out[ring] = info
    if used and {"inner", "outer"} <= by.keys() and "all" in by:
        ai, ao = ring_area(*RINGS_M["inner"]), ring_area(*RINGS_M["outer"])
        a = by["all"]
        a["people_m2_count"] = a["people_m2"]
        a["people_m2"] = round((by["inner"]["people_m2"] * ai + by["outer"]["people_m2"] * ao) / (ai + ao), 4)
        a["model"] = f"{LIVE_NAME}+speed"
    atomic(SPEED_FILE, json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "camera": cam,
                                   "pairs": n_pairs, "used": used, "sector_deg": SECTOR_DEG, "rings": out,
                                   "sectors": {r: [{"deg": s["deg"], "how": s["how"], "rho": s["rho"]} for s in ss]
                                               for r, ss in secs.items()}}))
    if not used:
        return ""
    return f" | speed {cam} {n_pairs} pairs: " + ", ".join(
        f"{r} {v['speed_ms']} m/s -> {v['value']:.2f}" for r, v in out.items() if v["speed_ms"] is not None)


def db_source(source: str) -> str:
    """The source a reading is stored under: "youtube", or another source that has its own head-size curve
    (config/m2.json correction "source"); without one, "<source>_raw" (e.g. SBA's 720p reads ~30 % low on YouTube's
    curve: kept out of the main series, edge/db.MAIN_SOURCES)."""
    from edge.m2_metric import live_model_config
    from edge.m2_pipeline import Correction

    return source if source == "youtube" or Correction(live_model_config()).has_curve(source) else f"{source}_raw"


def far_cols(far: dict | None) -> dict:
    """The far Mataf's columns of a ring "all" frame row (edge/m2_metric.M2Reading.far)."""
    if not far:
        return {"people_m2_far": None, "people_m2_far_raw": None, "area_m2_far": None, "head_px_far": None,
                "sectors_far": None}
    return {"people_m2_far": far["corr"], "people_m2_far_raw": far["raw"], "area_m2_far": far["area"],
            "head_px_far": far["head_px"], "sectors_far": json.dumps(far["sectors"], separators=(",", ":"))}


def gpu_mb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"allocated_mb": round(torch.cuda.memory_allocated() / 2**20), "reserved_mb": round(torch.cuda.memory_reserved() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=1.0, help="frames decoded per second")
    ap.add_argument("--no-anchor", action="store_true", help="no Kaaba segmenter (OWLv2 + SAM): ptz without the check")
    ap.add_argument("--frames", type=Path, help="offline test on saved frames (no DB, no live files)")
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="99999999T9999")
    args = ap.parse_args()
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import load_config

    if args.frames:
        logging.basicConfig(level=logging.WARNING)
        offline(args)
        return
    setup_logging("shot")
    if not single_instance("shot"):
        log.warning("another live_shot.py is already running; exiting")
        sys.exit(3)
    db.init_db()
    cfg = load_config()
    from edge.m2_metric import live_model_config
    from edge.model_share import LockedModel, serve

    global LIVE_WEIGHTS
    live_cfg = live_model_config()  # the Mataf metric's model (config/m2.json "live_model"; v3 since 2026-10-29)
    LIVE_WEIGHTS = live_cfg["weights"] if live_cfg else None
    global LIVE_NAME
    LIVE_NAME = live_cfg.get("name") if live_cfg else None
    share = LockedModel(CrowdModel())  # the crowd collector's per-camera series stays on the original DM-Count
    model = LockedModel(CrowdModel(LIVE_WEIGHTS)) if live_cfg else share
    try:
        serve(share)  # the crowd collector's DM-Count (one copy on the GPU)
    except OSError:
        log.exception("model share unavailable (the crowd collector runs DM-Count on the CPU)")
    bank = Bank(BANK) if BANK.exists() else None
    fixes = None
    fx = (live_cfg or {}).get("fixes")
    if fx:  # edge/m2_fill.LiveFixes: the near-wall model runs on a crop around the Kaaba only
        from edge.m2_fill import LiveFixes

        near = LockedModel(CrowdModel(fx["near_wall_weights"])) if fx.get("near_wall_weights") else None
        fixes = LiveFixes(near, float(fx.get("near_wall_m", 4.0)), bool(fx.get("far_in_prayer", True)),
                          float(fx.get("stopped_rows_m2") or 0.0))
        log.warning("fixes on: near the walls (%s m) max with %s, far side in prayer %s, stopped rows in prayer %s",
                    fx.get("near_wall_m"), fx.get("near_wall_weights"), fx.get("far_in_prayer"),
                    f"{fixes.stopped_rho}/m2" if fixes.stopped_rho else "off")
    an = ShotAnalyzer(cfg, model, bank, verify=not args.no_anchor, anchor=not args.no_anchor, live_model=live_cfg,
                      fixes=fixes)
    speed = None
    if (live_cfg or {}).get("speed"):
        from edge.tawaf_speed import LiveSpeed

        speed = LiveSpeed(args.fps)
        log.warning("tawaf speed sensor on (edge/tawaf_speed.py)")
        if fixes is not None and fixes.stopped_rho:  # the stopped-rows fix reads the sensor's sectors
            fixes.speed_fn = speed.sectors
    if fixes is not None and fixes.stopped_rho and fixes.speed_fn is None:
        log.warning("stopped rows fix needs the speed sensor ('speed': true): off")
    if speed is not None and live_cfg.get("save_prayer_pairs"):  # 1 s pairs from prayer windows for offline tests (user 2026-10-31)
        speed.save_dir, speed.save_when = ROOT / "data" / "samples" / "speed_pairs", an.in_prayer
        log.warning("saving the speed pairs of prayer windows to %s", speed.save_dir)
    if live_cfg:
        log.warning("live model %s (no head-size correction, no camera factor; SBA curve %s, approximate bias %s)",
                    LIVE_WEIGHTS, (live_cfg.get("source") or {}).get("sba"), (live_cfg.get("approx") or {}).get("bias"))
    from edge.stream_reader import StreamReader

    reader = StreamReader(args.fps)
    wr = Writer()
    if speed is not None:
        wr.speed_fn = speed.sectors
    pairs = PairCapture()
    from edge.shadow import Shadow, started
    from edge.view_matcher import graphics_mask

    shadow = Shadow.from_config()  # candidate model next to the live one (config/m2.json "shadow"; 2026-10-27)
    if shadow is not None:
        log.warning(started(datetime.now(timezone.utc), shadow.cfg))
    gm_full: dict = {}
    log.info("live shot service started: %.1f fps decoded, bank %s", args.fps,
             f"{len(bank.names)} frames" if bank else "missing")
    from edge.shot import SAME_SHOT
    from edge.shot_bank import thumb

    seq, t_min, n_min, n_skip, busy = 0, time.monotonic(), 0, 0, 0.0
    errors = 0
    prev_th, cut_at, skipped = None, time.monotonic(), False
    while True:
        fr = reader.next(seq, timeout=30)
        if fr is None:
            continue
        if speed is not None:
            speed.offer_next(fr.image, fr.captured_at, fr.seq)  # the second frame of a pending speed pair
        seq = fr.seq
        # half rate on a shot that stays: consecutive frames compared as edge/shot.py does (thumbnail correlation)
        th = thumb(fr.image)
        same = prev_th is not None and float((th * prev_th).mean()) > SAME_SHOT
        prev_th = th
        if not same:
            cut_at = time.monotonic()
        if same and not skipped and time.monotonic() - cut_at > HALF_AFTER_CUT_S:
            skipped = True
            n_skip += 1
            continue
        skipped = False
        t = time.perf_counter()
        try:
            an.source = fr.source
            sr = an.analyse(fr.image, fr.captured_at)
            wr.frame(sr, (fr.image.shape[1], fr.image.shape[0]), fr.source)
            pairs.offer(fr.image, sr)
            if speed is not None:
                size = (fr.image.shape[1], fr.image.shape[0])
                if size not in gm_full:
                    gm_full[size] = graphics_mask(cfg, size)
                speed.offer_analysed(fr.image, sr, fr.seq, gm_full[size])
            if shadow is not None:
                size = (fr.image.shape[1], fr.image.shape[0])
                if size not in gm_full:
                    gm_full[size] = graphics_mask(cfg, size)
                try:
                    shadow.add(fr.image, sr, gm_full[size], fr.source)
                except Exception:
                    log.exception("shadow model failed on this frame")
            errors = 0
        except Exception:
            errors += 1
            log.exception("shot analysis failed")
            if errors >= 20:
                sys.exit(1)  # a fresh process (and CUDA context)
            continue
        busy += time.perf_counter() - t
        n_min += 1
        if time.monotonic() - t_min >= 60:
            span = time.monotonic() - t_min
            msg = wr.flush() if wr.minute and datetime.now(timezone.utc).replace(second=0, microsecond=0) > wr.minute else ""
            if speed is not None:
                st_ = speed.stats
                msg += f" | speed pairs {st_['paired']}/{st_['offered']}" + (f" (camera motion not removed {st_['no_pose']}, errors {st_['error']})" if st_['no_pose'] or st_['error'] else "") + (f", saved {st_['saved']}" if st_['saved'] else "")
                speed.stats.clear()
            if fixes is not None:
                msg += fixes.take_stats()
            if shadow is not None:
                try:
                    msg += f" | shadow {shadow.flush()} frames"
                except Exception:
                    log.exception("shadow rows not written")
            st = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frames_last_min": n_min,
                  "skipped_last_min": n_skip, "source": reader.source,
                  "rate_fps": round(n_min / span, 2),
                  "busy_share": round(busy / span, 2),
                  "methods_total": dict(wr.methods), "frames_total": wr.frames, "gpu": gpu_mb()}
            atomic(STATUS, json.dumps(st))
            log.info("%d frames in %.0f s (%.2f/s, busy %.0f %%; %d skipped, same shot; %s), methods %s%s", n_min, span,
                     n_min / span, 100 * busy / span, n_skip, reader.source, dict(wr.methods), msg)
            t_min, n_min, n_skip, busy = time.monotonic(), 0, 0, 0.0


def offline(args) -> None:
    """Saved frames through the same analysis: per frame the method, camera, people per m^2 and timings."""
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import load_config

    from edge.m2_metric import live_model_config

    cfg = load_config()
    live_cfg = live_model_config()
    model = CrowdModel(live_cfg["weights"]) if live_cfg else CrowdModel()
    an = ShotAnalyzer(cfg, model, Bank(BANK) if BANK.exists() else None, verify=not args.no_anchor,
                      anchor=not args.no_anchor, live_model=live_cfg)
    files = [f for f in sorted(args.frames.glob("*.jpg")) + sorted(args.frames.glob("m2_*.png"))
             if args.since <= f.stem.split("_")[-1] < args.until]
    for f in files:
        img = cv2.imread(str(f))
        stamp = f.stem.split("_")[-1]
        ts = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        sr = an.analyse(img, ts)
        print(json.dumps({"frame": f.name, "method": sr.method, "how": sr.how, "camera": sr.camera,
                          "people_m2": sr.people_m2, "inner": sr.reading.density_corr.get("inner") if sr.reading else None,
                          "rows": sr.rows.density if sr.rows else None, "approx": sr.approx, "ms": sr.ms}, default=float),
              flush=True)
    print(json.dumps({"gpu": gpu_mb()}))


if __name__ == "__main__":
    main()
