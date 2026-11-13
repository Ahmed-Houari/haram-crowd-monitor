"""Step 4: 24/7 crowd collection.

Every 15 s: grab a frame (YouTube, SBA fallback), match the reference views, run DM-Count on
accepted frames, count per zone. Every minute: write the per-zone averages to crowd_minute
(raw table; the forecast reads 5-min averages via db.crowd_5min) and a heartbeat row to
collector_minute (uptime metric). Frames stay in RAM and are dropped right after processing;
only numbers are stored. Missing reference views (cam1) are captured and added automatically.
Opt-in extras (config/views.json "thumbnails"): survey thumbnails + 3 annotated samples, see edge/extras.py.
View-independent metric (2026-10-24, edge/m2_pipeline.py): every frame whose camera is known (a calibrated reference
camera matched by ALIKED + LightGlue, or a camera fitted to the Kaaba) also gives people per m^2 on the Mataf floor in
rings around the Kaaba -> crowd_m2_frame / crowd_m2_minute; the per-camera metric above keeps running as the fallback.
GPU (2026-10-26): with the m2 path off (config/m2.json, the live shot service has it) DM-Count runs on the shot
service's model (edge/model_share.py: one copy on the GPU, none here), on the CPU while that service is down.

Normally started by scripts/supervise.py (Task Scheduler at logon). Manual run:
    .venv/Scripts/python scripts/collect_crowd.py
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge import db  # noqa: E402
from edge.frame_grabber import FrameGrabber, lookups_today  # noqa: E402
from edge.extras import CollectorExtras  # noqa: E402
from edge.heatmap import density_png, heat_scale  # noqa: E402
from edge.model_share import SharedModel  # noqa: E402
from edge.pipeline import CrowdPipeline, warped_zones  # noqa: E402
from edge.runtime import (ALREADY_RUNNING_EXIT, PROJECT_ROOT, private_memory_mb, setup_logging,  # noqa: E402
                          single_instance)
from edge.view_matcher import FULL_VISIBLE, graphics_mask  # noqa: E402

log = logging.getLogger("collect_crowd")
STATUS_PATH = PROJECT_ROOT / "data" / ".crowd_status.json"
LIVE_DIR = PROJECT_ROOT / "data" / "live"  # small density-only heatmaps for the dashboard
MAX_CONSECUTIVE_ERRORS = 20  # then exit(1): the supervisor restarts us with a fresh CUDA context
M2_KAABA_BUDGET_S = 6.0  # the slow Kaaba-fit path only if the frame took less than this so far
RELOAD_EVERY_S = 600
# Memory guard (user request 2026-10-25): private memory swings ~1 GB within seconds (per-frame buffers), so the
# level is the median of the last 10 per-minute minimums. Above MEMORY_LIMIT_MB the collector exits cleanly at a
# quiet moment (no reading for QUIET_S) and the supervisor restarts it; above MEMORY_HARD_MB it exits regardless.
MEMORY_LIMIT_MB = 12000
MEMORY_HARD_MB = 14000
QUIET_S = 120


class MemoryGuard:
    """Per-frame private memory -> per-minute minimum; logs a summary every 10 minutes; says when to restart."""

    def __init__(self, trace: bool = False):
        from collections import deque

        self.minute_min: deque = deque(maxlen=10)
        self.minute_max: deque = deque(maxlen=10)
        self.cur: list[float] = []
        self.last_log = time.monotonic()
        self.trace = trace
        if trace:  # diagnostics: Python-level allocation sites that grow (config/m2.json "trace_memory")
            import tracemalloc

            tracemalloc.start(8)
            self.snap = tracemalloc.take_snapshot()

    def sample(self) -> None:
        mb = private_memory_mb()
        if mb is not None:
            self.cur.append(mb)

    def end_minute(self, totals: dict) -> str | None:
        """Close the minute; returns a reason to restart, or None."""
        if not self.cur:
            return None
        self.minute_min.append(min(self.cur))
        self.minute_max.append(max(self.cur))
        self.cur = []
        level = float(statistics.median(self.minute_min))
        totals["private_mb"] = round(level)
        if time.monotonic() - self.last_log >= 600:
            self.last_log = time.monotonic()
            line = (f"memory: private {level:.0f} MB (10-min minimums {min(self.minute_min):.0f}-"
                    f"{max(self.minute_min):.0f}, peaks up to {max(self.minute_max):.0f})")
            try:
                import torch

                if torch.cuda.is_available():
                    line += (f"; CUDA allocated {torch.cuda.memory_allocated() / 2 ** 20:.0f} MB, reserved "
                             f"{torch.cuda.memory_reserved() / 2 ** 20:.0f} MB")
            except Exception:
                pass
            log.info(line)
            if self.trace:
                import tracemalloc

                snap = tracemalloc.take_snapshot()
                for st in snap.compare_to(self.snap, "traceback")[:5]:
                    if st.size_diff > 256 * 1024:
                        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(st.traceback[-4:]))
                        log.info("memory growth since start: %+.1f MB (%+d blocks) at %s", st.size_diff / 2 ** 20,
                                 st.count_diff, where)
        if len(self.minute_min) < 10:
            return None
        last = max((datetime.fromisoformat(t) for t in (totals.get("last_reading_utc"), totals.get("last_m2_utc")) if t),
                   default=None)
        quiet = last is None or (datetime.now(timezone.utc) - last).total_seconds() >= QUIET_S
        if level > MEMORY_HARD_MB:
            return f"private memory {level:.0f} MB > {MEMORY_HARD_MB} MB"
        if level > MEMORY_LIMIT_MB and quiet:
            return f"private memory {level:.0f} MB > {MEMORY_LIMIT_MB} MB at a quiet moment"
        return None


class MinuteBuffer:
    """Collects one minute of readings, then writes averages to the DB."""

    def __init__(self):
        self.minute: datetime | None = None
        self.zones: dict[tuple, list] = defaultdict(list)  # (view, zone, source) -> [(count, density, zoom, visible)]
        self.frame_rows: list[dict] = []  # crowd_frame rows (numbers only)
        self.frames = self.accepted = self.youtube = self.fallback = 0
        self.infer: list[float] = []

    def add(self, source: str, reading, result) -> None:
        self.frames += 1
        self.youtube += source == "youtube"
        self.fallback += source in ("sba", "fallback")
        if reading:
            self.accepted += reading.accepted
            for zone, v in reading.zones.items():
                if reading.accepted:
                    self.zones[(reading.view_id, zone, reading.source)].append(
                        (v["count"], v["density"], v.get("zoom", reading.zoom), v["visible"]))
                self.frame_rows.append({"ts_utc": reading.ts, "view_id": reading.view_id, "zone": zone,
                                        "source": reading.source, "count": v["count"], "density": v["density"],
                                        "zoom": round(v.get("zoom", reading.zoom), 4), "inliers": reading.inliers,
                                        "accepted": reading.accepted, "ref_id": reading.ref_id,
                                        "density_top": reading.density_top, "density_max": reading.density_max,
                                        "visible": v["visible"],
                                        "h_primary": json.dumps(reading.h_primary) if reading.h_primary else None})
        if result:
            self.infer.append(result.inference_ms)

    def flush(self) -> str:
        rows = [db.CrowdMinute(self.minute, view, zone, statistics.fmean(v[0] for v in vals),
                               statistics.fmean(v[1] for v in vals), statistics.fmean(v[2] for v in vals),
                               len(vals), source, statistics.fmean(v[3] for v in vals),
                               statistics.fmean(full) if (full := [v[0] for v in vals if v[3] >= FULL_VISIBLE]) else None)
                for (view, zone, source), vals in self.zones.items()]
        if rows:
            db.insert_crowd_minute(rows)
        if self.frame_rows:
            db.insert_crowd_frames(self.frame_rows)
        db.insert_collector_minute(self.minute, self.frames, self.accepted, self.youtube, self.fallback,
                                   statistics.fmean(self.infer) if self.infer else None)
        return (f"{self.minute:%H:%M}Z frames {self.frames} (yt {self.youtube}, sba {self.fallback}) accepted "
                f"{self.accepted}" + "".join(f" | {r.view_id}/{r.zone} {r.density_avg:.1f}/10k px, "
                                             f"{100 * r.visible_avg:.0f} % visible" for r in rows))


class M2Buffer:
    """One minute of view-independent readings -> crowd_m2_frame (per frame and ring) + crowd_m2_minute (pooled)."""

    def __init__(self):
        self.minute: datetime | None = None
        self.rows: list[dict] = []

    def add(self, ts: datetime, source: str, fc, rd) -> None:
        c = fc.cam.centre
        depth = fc.info.get("ptz_depth", 0) if fc.path == "ptz" else 0
        path = f"{fc.path}/k{depth}" if depth else fc.path  # ptz/k2: matched through a reference 2 links away
        for ring, v in rd.density_corr.items():
            self.rows.append({"ts_utc": ts, "ring": ring, "source": source, "path": path,
                              "camera": {"ptz": fc.info.get("ptz_view"), "anchor": fc.info.get("anchor_view")}.get(
                                  fc.path, "kaaba"),
                              "people_m2": v, "people_m2_raw": rd.density.get(ring), "count": rd.count.get(ring),
                              "area_m2": rd.area.get(ring), "head_px": rd.head_px.get(ring),
                              "zoom": fc.info.get("ptz_zoom"), "focal_px": round(fc.cam.f, 1),
                              "ptz_err": fc.info.get("ptz_err"), "cam_x": round(float(c[0]), 2),
                              "cam_y": round(float(c[1]), 2), "cam_z": round(float(c[2]), 2),
                              "head_bins": json.dumps(rd.bins) if ring == "all" and rd.bins else None,
                              "sectors": json.dumps(rd.sectors, separators=(",", ":"))
                              if ring == "all" and rd.sectors else None})

    def flush(self) -> str:
        groups: dict[tuple, list] = defaultdict(list)
        for r in self.rows:
            groups[(r["ring"], r["source"])].append(r)
        minutes = []
        for (ring, src), rs in groups.items():
            area = sum(r["area_m2"] for r in rs)
            minutes.append({"minute_utc": self.minute, "ring": ring, "source": src,
                            "people_m2": round(sum(r["people_m2"] * r["area_m2"] for r in rs) / area, 4),
                            "people_m2_raw": round(sum((r["people_m2_raw"] or 0) * r["area_m2"] for r in rs) / area, 4),
                            "area_m2": round(area / len(rs), 1), "n_frames": len(rs),
                            "cameras": ",".join(sorted({r["camera"] or "" for r in rs}))[:200]})
        if self.rows:
            db.insert_m2(self.rows, minutes)
        return "".join(f" | Mataf {m['ring']} {m['people_m2']:.2f}/m2" for m in minutes if m["ring"] == "all")


def write_live_heatmap(reading, result, zones: dict, shape: tuple) -> None:
    """data/live/heatmap_<view>.png (density + zone outlines, no video pixels) + .json with the numbers."""
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        png = LIVE_DIR / f"heatmap_{reading.view_id}.png"
        tmp = png.with_suffix(".tmp")
        tmp.write_bytes(density_png(result.density, (shape[1], shape[0]), zones))
        tmp.replace(png)  # atomic: the dashboard never reads a half-written file
        meta = png.with_suffix(".json")
        meta.with_suffix(".tmp").write_text(json.dumps(reading.to_dict() | {"frame_size": [shape[1], shape[0]]}))
        meta.with_suffix(".tmp").replace(meta)
    except Exception:
        log.exception("could not write live heatmap")


M2_HEAT_DIR = LIVE_DIR / "m2"  # the last 10 minutes of m2 heatmaps, for the dashboard's merged floor map
M2_HEAT_KEEP_S = 600


def write_m2_heatmap(frame, cam_id: str, cam, density, shape: tuple) -> None:
    """Every frame the view-independent metric reads (2026-10-26): the density it used as a small heatmap (no video
    pixels), with the frame's calibrated camera. data/live/heatmap_<cam>.png/.json = the latest per camera (the AI view);
    data/live/m2/<cam>_<UTC>.png/.json = the last M2_HEAT_KEEP_S seconds (the floor map), older ones deleted. CPU only:
    the density is the one the metric already computed."""
    try:
        h, w = shape[:2]
        png = density_png(density, (w, h))
        meta = {"ts": frame.captured_at.isoformat(timespec="seconds"), "view_id": cam_id, "source": frame.source,
                "accepted": True, "density_top": round(heat_scale(density, (w, h)), 6), "frame_size": [w, h],
                "from": "m2", "cam": {"f": round(float(cam.f), 2), "rvec": [round(float(x), 7) for x in cam.rvec],
                                      "tvec": [round(float(x), 5) for x in cam.tvec], "size": list(cam.size)}}
        M2_HEAT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = frame.captured_at.strftime("%Y%m%dT%H%M%SZ")
        for base in (LIVE_DIR / f"heatmap_{cam_id}", M2_HEAT_DIR / f"{cam_id}_{stamp}"):
            tmp = base.with_suffix(".tmp")
            tmp.write_bytes(png)
            tmp.replace(base.with_suffix(".png"))  # atomic: the dashboard never reads a half-written file
            tmp.write_text(json.dumps(meta))
            tmp.replace(base.with_suffix(".json"))
        cutoff = time.time() - M2_HEAT_KEEP_S
        for f in M2_HEAT_DIR.iterdir():
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        log.exception("could not write the m2 heatmap")


def daily_safety(state: dict) -> None:
    """Once per Makkah day: database backup (background thread, SQLite backup API) + disk space in the log."""
    import threading

    from edge.backup import backup_db, disk_status, last_backup

    today = datetime.now(timezone(timedelta(hours=3))).date()
    if state.get("day") == today:
        return
    state["day"] = today
    disk = disk_status()
    state.update(disk)
    (log.warning if disk["disk_low"] else log.info)("disk: %.1f GB free (%.1f %% used)%s", disk["disk_free_gb"],
                                                    disk["disk_used_pct"], " -- LOW" if disk["disk_low"] else "")

    def run() -> None:
        try:
            backup_db()
        except Exception:
            log.exception("database backup failed")
        state["last_backup"] = last_backup()

    threading.Thread(target=run, name="db-backup", daemon=True).start()


def write_status(totals: dict, pipe: CrowdPipeline, safety: dict | None = None) -> None:
    STATUS_PATH.write_text(json.dumps({
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **totals,
        **{k: v for k, v in (safety or {}).items() if k != "day"},
        "views": pipe.view_ids, "waiting_for_reference": sorted(pipe.capturer.todo) if pipe.capturer else [],
        "yt_dlp_lookups_today": lookups_today()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    setup_logging("crowd")
    if not single_instance("crowd"):
        log.warning("another collect_crowd.py is already running; exiting")
        sys.exit(ALREADY_RUNNING_EXIT)

    engine = db.init_db()
    m2cfg: dict = {}
    try:
        m2cfg = json.loads((PROJECT_ROOT / "config" / "m2.json").read_text(encoding="utf-8"))
    except Exception:
        log.exception("config/m2.json unreadable")
    # DM-Count: the live shot service's model (one copy on the GPU; no CUDA in this process, user request 2026-10-26)
    # unless this collector's own m2 path is on, which needs the GPU here
    pipe = CrowdPipeline(None if m2cfg.get("enabled") else SharedModel())
    extras = CollectorExtras(pipe.config)
    m2pipe = None
    try:  # view-independent metric: optional (config/m2.json), the per-camera metric runs without it
        from edge.m2_pipeline import M2Pipeline

        if m2cfg.get("enabled"):
            m2pipe = M2Pipeline(pipe.config, pipe.model, use_kaaba=m2cfg.get("kaaba_fit", True),
                                anchor=m2cfg.get("anchor", True))
            log.info("view-independent metric on (%d calibrated cameras)", len(m2pipe.ptz.cams) if m2pipe.ptz else 0)
    except Exception:
        log.exception("view-independent metric unavailable; per-camera metric only")
    m2buf = M2Buffer()
    log.info("crowd collector started: %s, every %.0f s", db.describe(engine), args.interval)
    buf = MinuteBuffer()
    totals = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frames": 0, "accepted": 0,
              "last_frame_utc": None, "last_reading_utc": None}
    errors = 0
    safety: dict = {}
    guard = MemoryGuard(trace=bool(m2cfg.get("trace_memory")))
    next_reload = time.monotonic() + RELOAD_EVERY_S
    grabber = FrameGrabber.from_env(interval_s=args.interval)
    for frame in grabber.frames():
        minute = frame.captured_at.replace(second=0, microsecond=0)
        if buf.minute is not None and minute != buf.minute:
            try:
                m2buf.minute = buf.minute
                log.info(buf.flush() + m2buf.flush())
                daily_safety(safety)
                restart = guard.end_minute(totals)
                write_status(totals, pipe, safety)
            except Exception:
                restart = None
                log.exception("could not write minute %s to the database", buf.minute)
            buf, m2buf = MinuteBuffer(), M2Buffer()
            if restart:  # the minute is written; the supervisor restarts us in ~10 s
                log.warning("memory guard: %s -> exiting for a clean restart", restart)
                return
        buf.minute = minute
        source = frame.source
        t_frame = time.monotonic()
        try:
            reading, m, result = pipe.process(frame)
            errors = 0
            try:
                extras.handle(frame, m, reading, result, pipe)  # opt-in thumbnails + 3 samples
            except Exception:
                log.exception("extras (thumbnails/samples) failed")
            if reading and reading.accepted:
                write_live_heatmap(reading, result, warped_zones(pipe.matcher, m), frame.image.shape)
            if m2pipe is not None:
                try:
                    h, w = frame.image.shape[:2]
                    gm = graphics_mask(pipe.config, (w, h))
                    fc = m2pipe.camera(frame.image, gm, try_kaaba=time.monotonic() - t_frame < M2_KAABA_BUDGET_S,
                                       t=frame.captured_at.timestamp())
                    if fc.cam is not None:
                        rd = m2pipe.reading(frame.image, fc, gm, result.density if result is not None else None)
                        if rd.density_corr:
                            m2buf.add(frame.captured_at, source, fc, rd)
                            cam_id = {"ptz": fc.info.get("ptz_view"), "anchor": fc.info.get("anchor_view")}.get(fc.path)
                            if cam_id and source in db.MAIN_SOURCES and getattr(m2pipe, "last_density", None) is not None:
                                write_m2_heatmap(frame, cam_id, fc.cam, m2pipe.last_density, frame.image.shape)
                            totals["m2_frames"] = totals.get("m2_frames", 0) + 1
                            totals["last_m2_utc"] = frame.captured_at.isoformat(timespec="seconds")
                            extras.m2samples.maybe_save(frame, fc, rd)  # ground-truth samples (config/samples.json)
                except Exception:
                    log.exception("view-independent metric failed on this frame")
        except Exception:
            errors += 1
            log.exception("pipeline error (%d in a row)", errors)
            if errors >= MAX_CONSECUTIVE_ERRORS:
                log.error("too many errors in a row, exiting for a clean restart")
                sys.exit(1)
            continue
        finally:
            del frame  # frames never outlive this loop iteration
        buf.add(source, reading, result)
        totals["frames"] += 1
        totals["last_frame_utc"] = minute.isoformat()
        if reading and reading.accepted:
            totals["accepted"] += 1
            totals["last_reading_utc"] = reading.ts.isoformat(timespec="seconds")
        guard.sample()
        if time.monotonic() >= next_reload:
            if pipe.reload_if_changed():
                extras = CollectorExtras(pipe.config)
            next_reload = time.monotonic() + RELOAD_EVERY_S


if __name__ == "__main__":
    main()
