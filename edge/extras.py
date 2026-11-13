"""Opt-in extras for the crowd collector (user requests 2026-10-23), all switched by config/views.json.

- Survey thumbnails (config "thumbnails", until `until_utc`): 480x270 JPEGs in data/views/thumbs/
  * unmatched/: one frame every N min that matches no reference view (all hours; tagged day/night),
  * night_matched/: a few night frames that DO match a reference view (cap per view),
  with an index.csv (UTC time, Makkah time, kind, view, zoom, night flag, file).
- Annotated samples: the first 3 accepted readings (one per view first) as test samples in
  data/samples/live_*.png (zones + heatmap), then nothing more.
- Accuracy samples (config/samples.json, user request 2026-10-24): clean full-resolution frames of accepted
  readings in data/samples/acc_<view>_<UTC>.png (+ acc_index.csv) for the manual counting tool, spread over
  cameras and day / evening / night, at most one per camera per hour, up to a total cap. That file is re-read every
  minute, so its limits change without a restart. Counts survive restarts (read from the files on disk).
- Mataf ground-truth samples (config/samples.json "m2", user request 2026-10-25): clean full-resolution frames
  whose camera the view-independent metric found, data/samples/m2_<camera>_<UTC>.png + a sidecar .json with that
  camera's model and the m2 reading (+ m2_index.csv), per camera and period, for manual head counts
  (scripts/count_heads.py) -> per-camera scale factors.
Night = from Maghrib until sunrise (prayer times from the DB).
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import cv2

from edge.pipeline import CrowdPipeline, Reading, annotate_reading
from edge.view_matcher import PROJECT_ROOT, ViewMatch

log = logging.getLogger(__name__)

THUMBS_DIR = PROJECT_ROOT / "data" / "views" / "thumbs"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SAMPLES_CFG = PROJECT_ROOT / "config" / "samples.json"
THUMB_SIZE = (480, 270)
MAX_SAMPLES = 3
MAKKAH = timezone(timedelta(hours=3))


class NightClock:
    """is_night(ts): between Maghrib and sunrise. Prayer events cached for an hour."""

    def __init__(self):
        self._events: list = []
        self._loaded = 0.0

    def is_night(self, ts: datetime) -> bool | None:
        from edge import prayer_times

        if time.monotonic() - self._loaded > 3600 or not self._events:
            try:
                self._events = prayer_times.load_events(ts - timedelta(days=1), ts + timedelta(days=1),
                                                        include_sunrise=True)
            except Exception as e:  # missing table / DB busy: fall back to the clock
                log.warning("prayer times unavailable (%s); night = 18:00-06:00 Makkah", e)
                self._events = []
            self._loaded = time.monotonic()
        ctx = prayer_times.context_from_events(self._events, ts) if self._events else None
        if ctx is None:
            h = ts.astimezone(MAKKAH).hour
            return h >= 18 or h < 6
        return ctx.prev_prayer in ("maghrib", "isha", "fajr")  # fajr -> sunrise is still dark


class AccuracySamples:
    """Clean full-res frames of accepted readings for scripts/count_heads.py, spread over cameras and periods."""

    def __init__(self):
        self.cfg: dict = {}
        self._mtime: float | None = None
        self._checked = 0.0
        self.saved: list[tuple[str, datetime]] = []  # (view, UTC time) of the files already on disk
        for p in SAMPLES_DIR.glob("acc_*.png"):
            try:
                vid, stamp = p.stem[4:].rsplit("_", 1)
                self.saved.append((vid, datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)))
            except ValueError:
                continue

    def _config(self) -> dict:
        if time.monotonic() - self._checked < 60:
            return self.cfg
        self._checked = time.monotonic()
        try:
            mtime = SAMPLES_CFG.stat().st_mtime
            if mtime != self._mtime:
                self.cfg, self._mtime = json.loads(SAMPLES_CFG.read_text(encoding="utf-8")), mtime
                log.info("accuracy samples config: %s", {k: v for k, v in self.cfg.items() if k != "note"})
        except (OSError, ValueError) as e:
            if self._mtime is not None or self.cfg:
                log.warning("accuracy samples config unreadable (%s): sampling paused", e)
            self.cfg, self._mtime = {}, None
        return self.cfg

    def period(self, ts: datetime) -> str:
        h = ts.astimezone(MAKKAH).hour
        for name, (a, b) in (self.cfg.get("periods_makkah") or {}).items():
            if (a <= h < b) if a < b else (h >= a or h < b):
                return name
        return "other"

    def wants(self, view_id: str, ts: datetime, source: str) -> bool:
        cfg = self._config()
        if not cfg.get("enabled"):
            return False
        until = cfg.get("until_utc")
        if until and ts >= datetime.fromisoformat(until):
            return False
        if cfg.get("youtube_only", True) and source != "youtube":
            return False
        mine = [t for v, t in self.saved if v == view_id]
        if len(self.saved) >= cfg.get("max_total", 30) or len(mine) >= cfg.get("max_per_view", 12):
            return False
        per = self.period(ts)
        if sum(self.period(t) == per for t in mine) >= cfg.get("max_per_view_period", 4):
            return False
        hour = ts.astimezone(MAKKAH).strftime("%Y%m%d%H")
        return not (cfg.get("one_per_view_per_hour", True) and
                    any(t.astimezone(MAKKAH).strftime("%Y%m%d%H") == hour for t in mine))

    def maybe_save(self, frame, reading: Reading) -> None:
        ts = frame.captured_at
        if not self.wants(reading.view_id, ts, frame.source):
            return
        name = f"acc_{reading.view_id}_{ts:%Y%m%dT%H%M%SZ}.png"
        cv2.imwrite(str(SAMPLES_DIR / name), frame.image)
        self.saved.append((reading.view_id, ts))
        index = SAMPLES_DIR / "acc_index.csv"
        new = not index.exists()
        zones = {z: {k: v.get(k) for k in ("count", "density", "visible")} for z, v in reading.zones.items()}
        with open(index, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "makkah_time", "period", "view_id", "zoom", "source", "zones", "file"])
            w.writerow([ts.isoformat(timespec="seconds"), ts.astimezone(MAKKAH).strftime("%Y-%m-%d %H:%M"),
                        self.period(ts), reading.view_id, f"{reading.zoom:.3f}", frame.source, json.dumps(zones), name])
        log.warning("saved accuracy sample %s (%d of max %d)", name, len(self.saved), self.cfg.get("max_total", 30))


class M2Samples:
    """Clean full-res frames of m2 readings + the frame's camera, for ground-truth counts per camera (count_heads)."""

    def __init__(self, acc: AccuracySamples):
        self.acc = acc  # shares config/samples.json (section "m2") and its periods
        self.saved: list[tuple[str, datetime]] = []
        for p in SAMPLES_DIR.glob("m2_*.png"):
            try:
                cam, stamp = p.stem[3:].rsplit("_", 1)
                self.saved.append((cam, datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)))
            except ValueError:
                continue

    def maybe_save(self, frame, fc, rd) -> None:
        cfg = (self.acc._config() or {}).get("m2") or {}
        ts = frame.captured_at
        cam = {"ptz": fc.info.get("ptz_view"), "anchor": fc.info.get("anchor_view")}.get(fc.path)
        if not (cfg.get("enabled") and cam and fc.cam is not None and frame.source == "youtube"):
            return
        if cfg.get("until_utc") and ts >= datetime.fromisoformat(cfg["until_utc"]):
            return
        if (rd.area.get("all") or 0) < cfg.get("min_area_m2", 300) or not rd.density_corr:
            return
        mine = [t for c, t in self.saved if c == cam]
        per = self.acc.period(ts)
        if sum(self.acc.period(t) == per for t in mine) >= cfg.get("max_per_camera_period", 6):
            return
        hour = ts.astimezone(MAKKAH).strftime("%Y%m%d%H")
        if cfg.get("one_per_camera_per_hour", True) and any(t.astimezone(MAKKAH).strftime("%Y%m%d%H") == hour
                                                            for t in mine):
            return
        name = f"m2_{cam}_{ts:%Y%m%dT%H%M%SZ}"
        c = fc.cam
        meta = {"ts_utc": ts.isoformat(timespec="seconds"), "makkah_time": ts.astimezone(MAKKAH).strftime("%Y-%m-%d %H:%M"),
                "period": per, "camera": cam, "path": fc.path, "ptz_depth": fc.info.get("ptz_depth"),
                "zoom": fc.info.get("ptz_zoom"), "source": frame.source,
                "cam": {"f": round(float(c.f), 2), "rvec": [round(float(v), 7) for v in c.rvec],
                        "tvec": [round(float(v), 4) for v in c.tvec], "size": list(c.size)},
                "people_m2": rd.density_corr, "people_m2_raw": rd.density, "area_m2": rd.area, "head_px": rd.head_px}
        cv2.imwrite(str(SAMPLES_DIR / f"{name}.png"), frame.image)
        (SAMPLES_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1))
        self.saved.append((cam, ts))
        index = SAMPLES_DIR / "m2_index.csv"
        new = not index.exists()
        with open(index, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "makkah_time", "period", "camera", "path", "zoom", "people_m2_all", "area_all", "file"])
            w.writerow([meta["ts_utc"], meta["makkah_time"], per, cam, fc.path, meta["zoom"] or "",
                        rd.density_corr.get("all", ""), rd.area.get("all", ""), f"{name}.png"])
        log.warning("saved Mataf ground-truth sample %s.png (%s %s: %d of max %d)", name, cam, per,
                    sum(self.acc.period(t) == per for c2, t in self.saved if c2 == cam),
                    cfg.get("max_per_camera_period", 6))


class CollectorExtras:
    def __init__(self, config: dict):
        self.cfg = config.get("thumbnails") or {}
        until = self.cfg.get("until_utc")
        self.until = datetime.fromisoformat(until) if until else None
        self.clock = NightClock()
        self.last_unmatched: datetime | None = None
        self.last_matched: dict[str, datetime] = {}
        (THUMBS_DIR / "unmatched").mkdir(parents=True, exist_ok=True)
        (THUMBS_DIR / "night_matched").mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        # counts survive restarts: read them from the files already on disk
        self.matched_count = {}
        for p in (THUMBS_DIR / "night_matched").glob("*.jpg"):
            vid = p.stem.rsplit("_", 1)[0]
            self.matched_count[vid] = self.matched_count.get(vid, 0) + 1
        self.samples = sorted(SAMPLES_DIR.glob("live_*.png"))
        self.accuracy = AccuracySamples()
        self.m2samples = M2Samples(self.accuracy)

    def handle(self, frame, m: ViewMatch | None, reading: Reading | None, result, pipe: CrowdPipeline) -> None:
        ts = frame.captured_at
        if self.until and ts < self.until:
            self._thumbs(frame, m, ts)
        if reading and reading.accepted:
            self.accuracy.maybe_save(frame, reading)
        if reading and reading.accepted and len(self.samples) < MAX_SAMPLES:
            sampled_views = {p.stem.split("_", 2)[2].rsplit("_", 1)[0] for p in self.samples}
            if reading.view_id not in sampled_views or len(sampled_views) >= len(pipe.view_ids):
                path = SAMPLES_DIR / f"live_{len(self.samples) + 1}_{reading.view_id}_{ts:%Y%m%dT%H%M%SZ}.png"
                cv2.imwrite(str(path), annotate_reading(pipe, frame, m, result, reading))
                self.samples.append(path)
                log.warning("saved annotated sample %s", path.name)

    def _thumbs(self, frame, m: ViewMatch | None, ts: datetime) -> None:
        every = timedelta(minutes=self.cfg.get("unmatched_every_min", 5))
        if m is None:
            if self.last_unmatched is None or ts - self.last_unmatched >= every:
                self._save(frame, "unmatched", None, None, ts)
                self.last_unmatched = ts
            return
        if not self.clock.is_night(ts):
            return
        vid = m.view_id
        every_m = timedelta(minutes=self.cfg.get("night_matched_every_min", 5))
        if self.matched_count.get(vid, 0) < self.cfg.get("night_matched_max_per_view", 12) and \
                (vid not in self.last_matched or ts - self.last_matched[vid] >= every_m):
            self._save(frame, "night_matched", vid, m.zoom, ts)
            self.matched_count[vid] = self.matched_count.get(vid, 0) + 1
            self.last_matched[vid] = ts

    def _save(self, frame, kind: str, vid: str | None, zoom: float | None, ts: datetime) -> None:
        name = f"{vid}_{ts:%Y%m%dT%H%M%SZ}.jpg" if vid else f"{ts:%Y%m%dT%H%M%SZ}.jpg"
        path = THUMBS_DIR / kind / name
        thumb = cv2.resize(frame.image, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        index = THUMBS_DIR / "index.csv"
        new = not index.exists()
        with open(index, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "makkah_time", "kind", "view_id", "zoom", "night", "file"])
            w.writerow([ts.isoformat(timespec="seconds"), ts.astimezone(MAKKAH).strftime("%Y-%m-%d %H:%M"), kind,
                        vid or "", f"{zoom:.3f}" if zoom else "", int(bool(self.clock.is_night(ts))),
                        f"{kind}/{name}"])
