"""Step 6 backend: FastAPI + WebSocket over the collection database (read-only).

REST (JSON, times in UTC ISO 8601):
  GET /api/status                 collector health (running, last frame, accepted today, uptime today)
  GET /api/views                  reference cameras (quality, zoom range, zones, reference captured?)
  GET /api/crowd/latest           latest per-minute reading per camera/zone
  GET /api/crowd/now              per camera/zone: latest reading, trend vs ~30 min earlier, last 2 h,
                                  level (quiet/busy/crowded vs its own history) and "usual for this time"
  GET /api/crowd/series?hours=24  5-minute averages per camera/zone (accepted readings only)
  GET /api/mataf/now              view-independent metric (2026-10-24): people per m^2 on the Mataf floor, all
                                  cameras together (rings 1-10 / 10-25 m from the Kaaba's walls), with level, trend,
                                  last 2 h and usual like the cameras; null until the collector writes crowd_m2_minute
  GET /api/mataf/series?hours=24  its 5-minute averages per ring
  GET /api/mataf/floor            one top view of the Mataf floor (backend/floormap.py): every camera's latest heatmap
                                  projected onto the floor, people per m^2 per 0.75 m cell, newest camera wins, faded
                                  with age, grey where nobody saw the floor in the last 10 minutes; "png" = its texture
  GET /api/mataf/floor.png?v=KEY  that texture (RGBA, the twin's colour ramp)
                                  &style=flat: the four level colours only, flat (main dashboard since 2026-11-01)
                                  /api/mataf/now also carries "sectors" (12 angles x inner / outer ring, people per m^2
                                  pooled over the last SECTOR_WINDOW_MIN of frames, only where enough floor was seen) and
                                  "by_camera" (each camera's own people per m^2 from its latest frames); /api/crowd/now
                                  rows carry their camera's "people_m2" too (2026-10-26)
Crowd numbers (since 2026-10-24): "density" = people per 10,000 px of the VISIBLE part of the camera's zone (px of
its primary reference view) is the main value (levels, trend, last 2 h, usual, forecast); "visible" = the visible
fraction of the zone; "count" = people in the whole zone, only when it was fully visible (else null);
"count_visible" = people in the visible part.
  GET /api/traffic/latest         latest TomTom area aggregates (1 km / 3 km, per road category)
  GET /api/traffic/series?hours=24  mean level + congested % over time (all roads)
  GET /api/traffic/segments       GeoJSON of the latest poll's road segments (for the map)
  GET /api/prayer/today           today's prayer times (Makkah) + next prayer
  GET /api/prayer/window?back=20&ahead=4  prayer times (incl. sunrise) from 20 h back to 4 h ahead
  GET /api/heatmap/{view}.png     small density-only heatmap of the last accepted frame (no video pixels)
  GET /api/aiview/{view}          "What the AI sees": alignment, zone, label point, architecture outline path
  GET /api/aiview/{view}.png      that heatmap recoloured to the dashboard ramp, transparent (backend/aiview.py)
  GET /api/forecast               15/30/60-min forecast per camera (edge/forecast.py; available once a model is
                                  trained on real data into models/forecast)
  GET /api/forecast/next          the Mataf in 30 and 60 min (people per m^2), from the nightly forecast's models
                                  (scripts/forecast_nightly.py predict_live, data/forecast); the line on the dashboard
                                  and /simple (user 2026-11-07); {"available": false, "reason"} without a live reading
  GET /api/alerts                 open + recent alerts and the readiness of each rule (edge/alerts.py)
WebSocket /ws: a snapshot {status, crowd, crowd_now, forecast, alerts, traffic, prayer} on connect and whenever it changes
(checked every 15 s), plus {"type": "reading", "crowd_now": [...]} when a new accepted reading lands.
The built dashboard (dashboard/dist) is served at /; index.html with Cache-Control: no-cache (a normal reload picks
up a new build), the hashed assets as immutable.

Run:  .venv/Scripts/python -m uvicorn backend.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import and_, func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import aiview, floormap  # noqa: E402
from edge import alerts as alert_rules, db, forecast as crowd_forecast, prayer_times  # noqa: E402
from edge.m2_metric import FAR_FROM, MIN_AREA_M2, live_model_config, load_correction  # noqa: E402
from edge.frame_grabber import load_state as ytdlp_state, lookups_today, sba_url, youtube_paused  # noqa: E402
from edge.runtime import running_pid  # noqa: E402
from edge.view_matcher import FULL_VISIBLE, load_config, zoom_range  # noqa: E402

log = logging.getLogger("backend")
MAKKAH = timezone(timedelta(hours=3))
LIVE_DIR = ROOT / "data" / "live"
TWIN_PATH = ROOT / "config" / "twin.json"
OUTLINE_DIR = ROOT / "data" / "views" / "outlines"  # scripts/make_outlines.py (lines only)
SAMPLE_HEATMAP = ROOT / "docs" / "design" / "mockups" / "heatmap-cam0.png"  # for the dashboard's ?demo page
FORECAST_DIR = ROOT / "models" / "forecast"  # trained on the real DB only (scripts/train_forecast.py)
PUSH_EVERY_S = 15

# Crowd level = where the latest 5-min average sits in this camera's own history (same zone + source).
HISTORY_DAYS = 28            # rolling window for levels and "usual for this time"
LEVEL_MIN_DAYS = 3           # days of history (first to latest reading) before a level is shown ("calibrating" until then)
LEVEL_MIN_BUCKETS = 36       # and at least this many 5-min averages (3 h of readings)
LEVEL_PCTS = (40, 80)        # quiet = lowest 40 %, busy = middle, crowded = top 20 %
USUAL_MIN_DAYS = 7           # "X % above/below usual for this time" needs a week of data
USUAL_WINDOW_MIN = 30        # same time of day +- 30 min on earlier days
USUAL_MIN_DAYS_AT_TIME = 4   # earlier days with readings in that window
TREND_MIN, TREND_SLACK_MIN = 30, 10   # trend vs the 5-min average 30 +- 10 min before the latest one
SPARK_HOURS = 2


def iso(dt: datetime | None) -> str | None:
    return db.utc(dt).isoformat() if dt else None


def makkah_midnight_utc() -> datetime:
    return datetime.now(MAKKAH).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


# ---------------------------------------------------------------- queries (sync, read-only)
# /api/status "fresh" (2026-10-30: the shot service hung for hours while the collector's frames kept the status
# green): the collector's frames, the shot service's last analysed frame (data/live/shot.json, any method) and the
# Mataf's last reading must all be recent; stale_reason names what is not
FRESH_FRAME_MIN, FRESH_SHOT_MIN, FRESH_MATAF_MIN = 3, 3, 15


def get_status() -> dict:
    try:
        st = json.loads((ROOT / "data" / ".crowd_status.json").read_text())
    except (OSError, ValueError):
        st = {}
    try:
        shot_last = json.loads((ROOT / "data" / "live" / "shot.json").read_text()).get("ts")
    except (OSError, ValueError):
        shot_last = None
    day = makkah_midnight_utc()
    now = datetime.now(timezone.utc)
    cf, cm, tp = db.crowd_frame, db.collector_minute, db.traffic_poll
    with db.get_engine().connect() as c:
        acc = dict(c.execute(select(cf.c.view_id, func.count(func.distinct(cf.c.ts_utc)))
                             .where(cf.c.accepted, cf.c.ts_utc >= day).group_by(cf.c.view_id)).all())
        frames, alive = c.execute(select(func.coalesce(func.sum(cm.c.frames), 0), func.count())
                                  .where(cm.c.minute_utc >= day, cm.c.frames > 0)).one()
        first = c.execute(select(func.min(cm.c.minute_utc)).where(cm.c.minute_utc >= day)).scalar()
        last_poll = c.execute(select(func.max(tp.c.poll_utc))).scalar()
        m2 = db.crowd_m2_minute
        mataf_last = c.execute(select(func.max(m2.c.minute_utc)).where(m2.c.source.in_(db.MAIN_SOURCES))).scalar()
    minutes = max((now - db.utc(first)).total_seconds() / 60, 1) if first else None
    last_frame = st.get("last_frame_utc")
    running = running_pid("supervisor") is not None
    recent = lambda t, mins: bool(t) and now - (db.utc(t) if isinstance(t, datetime) else datetime.fromisoformat(t)) < timedelta(minutes=mins)  # noqa: E731
    stale = [why for why, ok in (("collector frames", recent(last_frame, FRESH_FRAME_MIN)),
                                 ("shot service", recent(shot_last, FRESH_SHOT_MIN)),
                                 ("Mataf readings", recent(mataf_last, FRESH_MATAF_MIN))) if not ok]
    return {
        "running": running,
        "fresh": not stale,
        "stale_reason": stale or None,
        "last_frame_utc": last_frame, "last_reading_utc": st.get("last_reading_utc"),
        "last_shot_frame_utc": shot_last, "last_mataf_utc": iso(mataf_last),
        "frames_today": int(frames), "accepted_today": acc,
        "uptime_today_pct": round(100 * alive / minutes, 1) if minutes else None,
        "last_traffic_poll_utc": iso(last_poll), "yt_dlp_lookups_today": lookups_today(),
        "waiting_for_reference": st.get("waiting_for_reference", []),
        "youtube_paused_s": round(youtube_paused(ytdlp_state())),
        "server_time_utc": now.isoformat(timespec="seconds"),
    }


def load_twin() -> dict:
    """Schematic camera placement for the 3D Mataf twin (config/twin.json), per view id. Views without an entry
    (new cameras) get a generic placeholder spot on the ring so they still appear (marked "placed": false)."""
    try:
        cams = json.loads(TWIN_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, ValueError, KeyError):
        cams = {}
    extra = [v for v in load_config()["views"] if v["id"] not in cams]
    palette = ["#b48ead", "#88c0d0", "#a3be8c", "#ebcb8b", "#d08770"]
    for i, v in enumerate(extra):
        cams[v["id"]] = {"label": v.get("label", v["id"]), "label_ar": "", "color": palette[i % len(palette)],
                         "ang": 3.9 + 0.75 * i, "r": 11.8, "h": 1.4, "hfov": 0.9, "near": 1.0, "far": 8.0, "aim": 0.0,
                         "placed": False}
    return cams


def get_views() -> list[dict]:
    """The reference cameras of config/views.json, then the calibrated cameras only the m2 metric uses
    (config/cameras.json entries with their own "ref_image", e.g. cam8_roof_b, 2026-10-26: no per-camera zone)."""
    twin = load_twin()
    views = load_config()["views"]
    try:
        cams = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, ValueError, KeyError):
        cams = {}
    known = {v["id"] for v in views}
    extra = [{"id": k, "quality": "ok", "quality_note": None, "zoom_tolerance": 0, "zones": [],
              "has_reference": (ROOT / c["ref_image"]).exists(), "has_heatmap": (LIVE_DIR / f"heatmap_{k}.png").exists(),
              "status": "approved", "label": (twin.get(k) or {}).get("label") or k,
              "references": [{"id": "primary", "status": "approved"}], "twin": twin.get(k), "metric_only": True}
             for k, c in cams.items() if k not in known and c.get("ref_image")]
    return [{"id": v["id"], "quality": v.get("quality"), "quality_note": v.get("quality_note"),
             "zoom_tolerance": v["zoom_tolerance"], "zoom_range": list(zoom_range(v)), "zones": list(v["zones"]),
             "has_reference": (ROOT / v["ref_image"]).exists(),
             "has_heatmap": (LIVE_DIR / f"heatmap_{v['id']}.png").exists(),
             "status": v.get("status", "approved"), "label": v.get("label") or (twin.get(v["id"]) or {}).get("label"),
             "references": [{"id": "primary", "status": "approved"}] + [
                 {k: r.get(k) for k in ("id", "period", "status", "quality", "preview")} for r in v.get("references", [])],
             "twin": twin.get(v["id"])} for v in views] + extra


def get_crowd_latest() -> list[dict]:
    t = db.crowd_minute
    latest = (select(t.c.view_id, t.c.zone, func.max(t.c.minute_utc).label("m"))
              .group_by(t.c.view_id, t.c.zone).subquery())
    q = select(t).join(latest, and_(t.c.view_id == latest.c.view_id, t.c.zone == latest.c.zone,
                                    t.c.minute_utc == latest.c.m))
    with db.get_engine().connect() as c:
        rows = c.execute(q).mappings().all()
    return [{"view_id": r["view_id"], "zone": r["zone"], "minute_utc": iso(r["minute_utc"]), "source": r["source"],
             "density": round(r["density_avg"], 3) if r["density_avg"] is not None else None,
             "visible": _visible(r["visible_avg"]),
             "count": _full(r["count_full_avg"], r["visible_avg"], r["count_avg"]),
             "count_visible": round(r["count_avg"], 1), "zoom": r["zoom_avg"], "n_frames": r["n_frames"]} for r in rows]


def _visible(v: float | None) -> float:
    return round(v, 3) if v is not None else 1.0  # rows before the visible-part metric: zone fully visible


def _full(count_full: float | None, visible: float | None, count: float) -> float | None:
    """People in the whole zone: only from frames where it was fully visible."""
    if count_full is not None:
        return round(count_full, 1)
    return round(count, 1) if visible is None else None


def _pct(values: list[float], p: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def _level(value: float, hist: list[float], days: float) -> dict:
    """quiet / busy / crowded vs this camera's own 5-min averages, or "calibrating" until there is enough history."""
    base = {"days": days, "need_days": LEVEL_MIN_DAYS, "buckets": len(hist), "need_buckets": LEVEL_MIN_BUCKETS,
            "pcts": list(LEVEL_PCTS), "history_days": HISTORY_DAYS}
    if days < LEVEL_MIN_DAYS or len(hist) < LEVEL_MIN_BUCKETS:
        return {"state": "calibrating", **base}
    lo, hi = (_pct(hist, p) for p in LEVEL_PCTS)
    state = "quiet" if value < lo else "crowded" if value >= hi else "busy"
    return {"state": state, "low": round(lo, 2), "high": round(hi, 2),
            "rank": round(sum(v <= value for v in hist) / len(hist), 3), **base}


def _usual(value: float, at_utc: datetime, buckets: list, days: float) -> dict:
    """Latest 5-min average vs the median of earlier days at the same time of day (+- USUAL_WINDOW_MIN)."""
    base = {"days": days, "need_days": USUAL_MIN_DAYS}
    if days < USUAL_MIN_DAYS:
        return {"state": "waiting", **base}
    local = at_utc.astimezone(MAKKAH)
    t0 = local.hour * 60 + local.minute
    per_day: dict = {}
    for b in buckets:
        d = b.bucket_utc.astimezone(MAKKAH)
        shift = (d.hour * 60 + d.minute - t0 + 720) % 1440 - 720   # signed minutes from the same time of day
        day = (d - timedelta(minutes=shift)).date()
        if abs(shift) <= USUAL_WINDOW_MIN and day != local.date():
            per_day.setdefault(day, []).append(b.density_avg)
    if len(per_day) < USUAL_MIN_DAYS_AT_TIME:
        return {"state": "no_history_at_time", "days_at_time": len(per_day), **base}
    usual = statistics.median(statistics.fmean(v) for v in per_day.values())
    return {"state": "ok", "usual": round(usual, 2), "pct": round(100 * (value - usual) / usual, 1) if usual else None,
            "days_at_time": len(per_day), **base}


_now_cache: dict = {"key": None, "t": 0.0, "val": []}


def get_crowd_now() -> list[dict]:
    """Per camera/zone: latest reading, trend vs ~30 min earlier, last 2 h of minutes, level and usual-for-this-time,
    all on the density in the visible part of the zone (people per 10k reference px).

    Cached until a new reading lands (or 5 min, so the time-of-day comparison follows the clock).
    """
    latest = get_crowd_latest()
    key = json.dumps(latest)
    if key == _now_cache["key"] and time.monotonic() - _now_cache["t"] < 300:
        return [_with_m2(r) for r in _now_cache["val"]]
    now = datetime.now(timezone.utc)
    hist = [b for b in db.crowd_5min(now - timedelta(days=HISTORY_DAYS), now + timedelta(minutes=1))
            if b.density_avg is not None]
    t = db.crowd_minute
    with db.get_engine().connect() as c:
        minutes = c.execute(select(t.c.minute_utc, t.c.view_id, t.c.zone, t.c.source, t.c.density_avg)
                            .where(t.c.minute_utc >= now - timedelta(hours=SPARK_HOURS), t.c.density_avg.is_not(None))
                            .order_by(t.c.minute_utc)).all()
    out, seen = [], set()
    for r in sorted(latest, key=lambda r: r["source"] not in db.MAIN_SOURCES):  # one row per camera/zone, main first
        k = (r["view_id"], r["zone"], r["source"])
        if k[:2] in seen:
            continue
        seen.add(k[:2])
        mine = [b for b in hist if (b.view_id, b.zone, b.source) == k]
        days = round((mine[-1].bucket_utc - mine[0].bucket_utc).total_seconds() / 86400, 1) if mine else 0.0
        cur = mine[-1].density_avg if mine else r["density"]
        if cur is None:
            continue
        trend = None
        if mine:
            target = mine[-1].bucket_utc - timedelta(minutes=TREND_MIN)
            near = [b for b in mine if abs(b.bucket_utc - target) <= timedelta(minutes=TREND_SLACK_MIN)]
            ref = min(near, key=lambda b: abs(b.bucket_utc - target), default=None)
            if ref and ref.density_avg > 0:
                trend = {"ref_utc": iso(ref.bucket_utc), "ref_density": round(ref.density_avg, 2),
                         "pct": round(100 * (cur - ref.density_avg) / ref.density_avg, 1)}
        out.append({"view_id": k[0], "zone": k[1], "source": k[2], "minute_utc": r["minute_utc"],
                    "density": r["density"], "visible": r["visible"], "count": r["count"],
                    "count_visible": r["count_visible"], "density_5min": round(cur, 2), "trend": trend,
                    "spark": [[iso(m.minute_utc), round(m.density_avg, 2)] for m in minutes
                              if (m.view_id, m.zone, m.source) == k],
                    "level": _level(cur, [b.density_avg for b in mine], days),
                    "usual": _usual(cur, mine[-1].bucket_utc, mine, days) if mine else None})
    _now_cache.update(key=key, t=time.monotonic(), val=out)
    return [_with_m2(r) for r in out]


_mataf_cache: dict = {"key": None, "t": 0.0, "val": None}
SECTOR_WINDOW_MIN = 10  # a floor sector's value pools every camera's frames of the last 10 minutes
SECTOR_MIN_AREA = {"i": 12.0, "o": 30.0}  # m^2 of a sector seen in that window, else the twin shows the ring average
CAMERA_M2_WINDOW_MIN = 20  # a camera's own people per m^2: its frames of the last 5 minutes it had, within 20 minutes
_recent_cache: dict = {"t": -1e9, "val": {"sectors": {}, "by_camera": {}}}


def get_m2_recent() -> dict:
    """Recent people per m^2 per floor sector (crowd_m2_frame.sectors, edge/m2_metric.SECTORS) and per camera."""
    if time.monotonic() - _recent_cache["t"] < 15:
        return _recent_cache["val"]
    now = datetime.now(timezone.utc)
    t = db.crowd_m2_frame
    try:
        with db.get_engine().connect() as c:
            rows = c.execute(select(t.c.ts_utc, t.c.camera, db.m2_col(t), t.c.area_m2, t.c.sectors)
                             .where(t.c.ring == "all", t.c.ts_utc >= now - timedelta(minutes=CAMERA_M2_WINDOW_MIN))
                             .order_by(t.c.ts_utc)).all()
    except Exception:  # an older database without the sectors column
        rows = []
    acc: dict = {}
    for r in rows:
        ts = r.ts_utc if r.ts_utc.tzinfo else r.ts_utc.replace(tzinfo=timezone.utc)
        if r.sectors and now - ts <= timedelta(minutes=SECTOR_WINDOW_MIN):
            for k, (_raw, corr, area, _hp) in json.loads(r.sectors).items():
                a = acc.setdefault(k, [0.0, 0.0, 0])
                a[0] += corr
                a[1] += area
                a[2] += 1
    sectors = {k: {"people_m2": round(c_ / a, 3), "area_m2": round(a, 1), "frames": n}
               for k, (c_, a, n) in acc.items() if a >= SECTOR_MIN_AREA.get(k[0], 30.0)}
    by_camera: dict = {}
    for cam in {r.camera for r in rows if r.camera}:
        mine = [r for r in rows if r.camera == cam and r.area_m2]
        last = mine[-1].ts_utc
        recent = [r for r in mine if last - r.ts_utc <= timedelta(minutes=5)]
        area = sum(r.area_m2 for r in recent)
        by_camera[cam] = {"people_m2": round(sum(r.people_m2 * r.area_m2 for r in recent) / area, 3),
                          "minute_utc": iso(last), "n_frames": len(recent)}
    val = {"sectors": sectors, "by_camera": by_camera, "sector_window_min": SECTOR_WINDOW_MIN}
    _recent_cache.update(t=time.monotonic(), val=val)
    return val


def _with_m2(row: dict) -> dict:
    """A camera row with its own people per m^2 (None while that camera has no recent view-independent reading)."""
    m = get_m2_recent()["by_camera"].get(row["view_id"])
    return {**row, "people_m2": m["people_m2"] if m else None, "people_m2_utc": m["minute_utc"] if m else None}


def get_mataf_now() -> dict | None:
    """Latest people per m^2 on the Mataf floor (ring "all" + the inner / outer rings), level / trend / last 2 h /
    usual on the 5-min series of ring "all"; None while there is no reading yet."""
    t = db.crowd_m2_minute
    try:
        with db.get_engine().connect() as c:
            main = t.c.source.in_(db.MAIN_SOURCES)
            last = c.execute(select(func.max(t.c.minute_utc)).where(main)).scalar()
            if last is None:
                return None
            rows = [{**r, "people_m2": db.m2_value(r)} for r in
                    c.execute(select(t).where(t.c.minute_utc == last, main).order_by(t.c.n_frames)).mappings().all()]
            now = datetime.now(timezone.utc)
            minutes = c.execute(select(t.c.minute_utc, db.m2_col(t)).where(
                t.c.ring == "all", main, t.c.minute_utc >= now - timedelta(hours=SPARK_HOURS))
                .order_by(t.c.minute_utc)).all()
    except Exception:  # no table yet (collector not restarted with the new metric)
        return None
    key = json.dumps([dict(r) for r in rows], default=str)
    if key == _mataf_cache["key"] and time.monotonic() - _mataf_cache["t"] < 300:
        return {**_mataf_cache["val"], **get_m2_recent(), "speed": get_speed()}
    by = {r["ring"]: r for r in rows}
    if "all" not in by:
        return None
    hist = [b for b in db.m2_5min(now - timedelta(days=HISTORY_DAYS), now + timedelta(minutes=1)) if b.zone == "all"]
    days = round((hist[-1].bucket_utc - hist[0].bucket_utc).total_seconds() / 86400, 1) if hist else 0.0
    cur = hist[-1].density_avg if hist else by["all"]["people_m2"]
    trend = None
    if hist:
        target = hist[-1].bucket_utc - timedelta(minutes=TREND_MIN)
        near = [b for b in hist if abs(b.bucket_utc - target) <= timedelta(minutes=TREND_SLACK_MIN)]
        ref = min(near, key=lambda b: abs(b.bucket_utc - target), default=None)
        if ref and ref.density_avg > 0:
            trend = {"ref_utc": iso(ref.bucket_utc), "ref_density": round(ref.density_avg, 3),
                     "pct": round(100 * (cur - ref.density_avg) / ref.density_avg, 1)}
    val = {"minute_utc": iso(last), "people_m2": round(by["all"]["people_m2"], 3),
           "rings": {k: round(v["people_m2"], 3) for k, v in by.items()},
           "area_m2": by["all"]["area_m2"], "n_frames": by["all"]["n_frames"],
           "cameras": [c_ for c_ in (by["all"]["cameras"] or "").split(",") if c_], "density_5min": round(cur, 3),
           "trend": trend, "spark": [[iso(m.minute_utc), round(m.people_m2, 3)] for m in minutes],
           "level": _level(cur, [b.density_avg for b in hist], days),
           "usual": _usual(cur, hist[-1].bucket_utc, hist, days) if hist else None, "far": get_far(),
           "model": by["all"].get("model")}
    _mataf_cache.update(key=key, t=time.monotonic(), val=val)
    return {**val, **get_m2_recent(), "speed": get_speed()}


SPEED_FRESH_MIN = 3  # the tawaf speed sensor's last minute (data/live/speed.json) counts while this recent


def get_speed() -> dict | None:
    """The tawaf speed sensor's last minute (edge/tawaf_speed.py via scripts/live_shot.py, 2026-10-30): per ring the
    reading with the speed, the count model's value, and the shares of walking / estimated / stopped sectors; per ring
    the sectors' "how" (the twin hatches the estimated ones). None when it is not recent or no sector walked."""
    try:
        sp = json.loads((LIVE_DIR / "speed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not sp.get("used") or datetime.now(timezone.utc) - datetime.fromisoformat(sp["ts"]) > timedelta(minutes=SPEED_FRESH_MIN):
        return None
    return sp


FAR_WINDOW_MIN = 10  # the far Mataf: pooled over this many minutes (fewer frames see it than the rings)
# accuracy of the far Mataf (scripts/far_check.py, 2026-10-27, 96 frames of 26 Oct 21:00-21:20 Makkah): heads 3-6 px out
# there (roof 3.1, low 5.7, wide 5.6) and the cameras that see the same far floor differ 1.5-2x per sector (median
# |log ratio| 0.38-0.73, against 0.11-0.13 on the headline rings) -> low confidence, shown as approximate
FAR_CONFIDENCE = "approximate"


def get_far() -> dict | None:
    """The far Mataf (25 m from the walls to the courtyard's edge, edge/m2_metric.FAR_FROM): people per m^2 pooled over
    the last FAR_WINDOW_MIN minutes (area-weighted). A separate value: not in "people_m2", the rings or the forecast."""
    t = db.crowd_m2_minute
    now = datetime.now(timezone.utc)
    try:
        with db.get_engine().connect() as c:
            rows = c.execute(select(t.c.minute_utc, db.m2_col(t, "people_m2_far"), t.c.area_m2_far, t.c.n_frames_far).where(
                t.c.ring == "all", t.c.source.in_(db.MAIN_SOURCES), t.c.people_m2_far.is_not(None),
                t.c.minute_utc >= now - timedelta(minutes=FAR_WINDOW_MIN))).all()
    except Exception:
        return None
    w = [(r.people_m2_far, (r.area_m2_far or 0) * (r.n_frames_far or 1)) for r in rows]
    a = sum(x for _, x in w)
    if not rows or a <= 0:
        return None
    return {"people_m2": round(sum(v * x for v, x in w) / a, 3), "minute_utc": iso(max(r.minute_utc for r in rows)),
            "minutes": len(rows), "n_frames": int(sum(r.n_frames_far or 0 for r in rows)),
            "area_m2": round(max(r.area_m2_far or 0 for r in rows)), "confidence": FAR_CONFIDENCE,
            "from_m": FAR_FROM}


def get_mataf_series(hours: float) -> list[dict]:
    end = datetime.now(timezone.utc)
    try:
        return [{"bucket_utc": iso(b.bucket_utc), "ring": b.zone, "people_m2": b.density_avg, "area_m2": b.count_avg,
                 "n_frames": b.n_frames} for b in db.m2_5min(end - timedelta(hours=hours), end)]
    except Exception:
        return []


def get_crowd_series(hours: float) -> list[dict]:
    end = datetime.now(timezone.utc)
    return [{"bucket_utc": iso(b.bucket_utc), "view_id": b.view_id, "zone": b.zone, "source": b.source,
             "density": b.density_avg, "visible": _visible(b.visible_avg),
             "count": round(b.count_full_avg, 1) if b.count_full_avg is not None else None, "n_frames": b.n_frames}
            for b in db.crowd_5min(end - timedelta(hours=hours), end)]


def get_traffic_latest() -> dict:
    a = db.traffic_area
    with db.get_engine().connect() as c:
        last = c.execute(select(func.max(a.c.poll_utc))).scalar()
        rows = c.execute(select(a).where(a.c.poll_utc == last)).mappings().all() if last else []
    return {"poll_utc": iso(last), "areas": [{k: r[k] for k in ("area", "road_category", "n_segments", "length_km",
                                                               "mean_level", "congested_pct", "closed_pct")} for r in rows]}


def get_traffic_series(hours: float) -> list[dict]:
    a = db.traffic_area
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    with db.get_engine().connect() as c:
        rows = c.execute(select(a.c.poll_utc, a.c.area, a.c.mean_level, a.c.congested_pct)
                         .where(a.c.poll_utc >= start, a.c.road_category == "all").order_by(a.c.poll_utc)).mappings().all()
    return [{"poll_utc": iso(r["poll_utc"]), "area": r["area"], "mean_level": r["mean_level"],
             "congested_pct": r["congested_pct"]} for r in rows]


def get_traffic_segments(max_km: float = 3.5) -> dict:
    sp, s = db.traffic_segment_poll, db.traffic_segment
    with db.get_engine().connect() as c:
        last = c.execute(select(func.max(sp.c.poll_utc))).scalar()
        rows = c.execute(select(s.c.segment_key, s.c.road_category, s.c.geom, sp.c.traffic_level, sp.c.closed)
                         .join(s, s.c.segment_key == sp.c.segment_key)
                         .where(sp.c.poll_utc == last, s.c.dist_km <= max_km)).mappings().all() if last else []
    features = [{"type": "Feature",
                 "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in json.loads(r["geom"])]},
                 "properties": {"key": r["segment_key"], "category": r["road_category"],
                                "level": r["traffic_level"], "closed": bool(r["closed"])}}
                for r in rows if r["geom"]]
    return {"type": "FeatureCollection", "poll_utc": iso(last), "features": features}


def get_prayer_today() -> dict:
    t = db.prayer_times
    today = datetime.now(MAKKAH).date()
    with db.get_engine().connect() as c:
        r = c.execute(select(t).where(t.c.date == today, t.c.method == prayer_times.METHOD)).mappings().first()
    if not r:
        return {"date": today.isoformat(), "times": None, "next": None}
    names = ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")
    ctx = prayer_times.prayer_context(datetime.now(timezone.utc))
    return {"date": today.isoformat(), "hijri_date": r["hijri_date"],
            "times": {n: {"utc": iso(r[n]), "makkah": db.utc(r[n]).astimezone(MAKKAH).strftime("%H:%M")} for n in names},
            "next": {"name": ctx.next_prayer, "utc": iso(ctx.next_utc), "minutes": round(ctx.minutes_to_next)} if ctx else None,
            "prev": {"name": ctx.prev_prayer, "utc": iso(ctx.prev_utc), "minutes": round(ctx.minutes_since_prev)} if ctx else None}


def get_prayer_window(back_h: float, ahead_h: float) -> list[dict]:
    """Prayer times incl. sunrise between back_h hours ago and ahead_h hours ahead (for the prayer horizon)."""
    now = datetime.now(timezone.utc)
    start, end = now - timedelta(hours=back_h), now + timedelta(hours=ahead_h)
    return [{"name": n, "utc": iso(t), "makkah": t.astimezone(MAKKAH).strftime("%H:%M")}
            for t, n in prayer_times.load_events(start, end, include_sunrise=True) if start <= t <= end]


def _level_k(view_id: str) -> tuple[float, str]:
    """Heat strength from the camera's level (same rule as the 3D twin); neutral while calibrating."""
    now = next((c for c in get_crowd_now() if c["view_id"] == view_id), None)
    level = now["level"] if now else None
    if level and level.get("rank") is not None:
        return 0.25 + 0.75 * level["rank"], level["state"]
    return 0.55, level["state"] if level else "none"


_ai_cache: dict = {}
_ai_bg: dict = {}  # (view, sample) -> background PNG (floor, Kaaba) of the latest AI view


def heat_correction(meta: dict, view_id: str) -> tuple[float | None, float, float]:
    """(h0, k, camera factor) that turn a live heatmap into people per m^2. A heatmap from a fine-tuned live model
    (meta "live_model", config/m2.json "live_model"; v3 since 2026-10-29) has the correction built in: none, except
    that model's own curve for its source (SBA); older heatmaps the original DM-Count's correction."""
    if meta.get("live_model"):
        curve = ((live_model_config() or {}).get("source") or {}).get(meta.get("source", "youtube"))
        return (float(curve["h0"]), float(curve["k"]), 1.0) if curve else (None, 1.0, 1.0)
    h0, k, factors = load_correction()
    return h0, k, factors.get(view_id, 1.0)


def _ring_m2(absolute: dict, ring: str = "ring") -> float | None:
    """People per m^2 over the floor 1-25 m from the Kaaba's walls (the metric's ring; ring="far": the far Mataf),
    weighted by floor area; None when less of it is in view than the metric needs for a reading (2026-10-26: was the
    camera's zone, out to 62 m)."""
    dens, ppm = absolute["dens"], absolute["px_per_m"]
    sel = absolute[ring] & np.isfinite(dens) & np.isfinite(ppm)
    area = 1.0 / np.maximum(ppm[sel], 1e-6) ** 2
    if area.sum() < MIN_AREA_M2:
        return None
    return round(float((dens[sel] * area).sum() / area.sum()), 2)


def get_aiview(view_id: str, sample: bool = False) -> tuple[dict, bytes] | None:
    """"What the AI sees" for one camera (cached per heatmap file + level). KeyError for an unknown view."""
    cfg = load_config()
    if view_id == "shot":
        return _shot_aiview()
    view = next((v for v in cfg["views"] if v["id"] == view_id), None)
    if view is None and view_id in _metric_only_cameras():  # no zone: the floor from its calibration only
        view = {"id": view_id, "zones": {}}
    if view is None:
        raise KeyError(view_id)
    png = SAMPLE_HEATMAP if sample else LIVE_DIR / f"heatmap_{view_id}.png"
    if not png.exists():
        return None
    outline = OUTLINE_DIR / f"{view_id}.json"
    k, state = (0.71, "busy") if sample else _level_k(view_id)
    m2 = None if sample else get_m2_recent()["by_camera"].get(view_id)
    key = (png.stat().st_mtime_ns, outline.stat().st_mtime_ns if outline.exists() else 0, round(k, 2),
           round(m2["people_m2"], 2) if m2 else None)
    hit = _ai_cache.get((view_id, sample))
    if hit and hit[0] == key:
        return hit[1], hit[2]
    try:
        cams = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, ValueError, KeyError):
        cams = {}
    meta = {}
    if not sample:
        try:
            meta = json.loads(png.with_suffix(".json").read_text())
        except (OSError, ValueError):
            pass
    H = None
    if meta.get("frame_size"):  # m2 heatmaps carry the frame's camera instead of the homography
        H = meta.get("h_primary") or (floormap.homography(meta, cams[view_id]) if view_id in cams else None)
    absolute = None  # a calibrated camera: people per m^2 per pixel, the floor and the Kaaba (backend/floormap.py)
    if not sample and view_id in cams and H:
        h0, hk, q = heat_correction(meta, view_id)  # hk: not the level's k, which render() still gets
        absolute = floormap.image_density(png, {**meta, "view_id": view_id}, cams[view_id], cfg, h0, hk, q)
    ring = _ring_m2(absolute) if absolute else None
    far = _ring_m2(absolute, "far") if absolute else None
    info, data, bg = aiview.render(cv2.imread(str(png)), view, tuple(cfg["ref_size"]), outline, k, H, meta.get("frame_size"),
                                   people_m2=m2["people_m2"] if m2 else None, m2=absolute)
    _ai_bg[(view_id, sample)] = bg
    zs = meta.get("zones", {})
    main = max(zs.values(), key=lambda z: z.get("visible", 1.0), default=None)  # one zone per camera so far
    vis = main.get("visible", 1.0) if main else None
    info |= {"view_id": view_id, "ts": meta.get("ts"), "density": main.get("density") if main else None,
             "visible": vis, "count": round(main["count"]) if main and vis >= FULL_VISIBLE else None,
             "level": state, "sample": sample, "has_outline": outline.exists(),
             "method": meta.get("method"), "how": meta.get("how"),
             "people_m2": m2["people_m2"] if m2 else None, "people_m2_utc": m2["minute_utc"] if m2 else None,
             # calibrated camera: people per m^2 of this very heatmap over the metric's ring, 1-25 m from the Kaaba
             **({"people_m2": ring, "people_m2_utc": meta.get("ts"), "people_m2_ring": True} if ring is not None else {}),
             # the far Mataf (25 m to the courtyard's edge) of this heatmap, approximate (FAR_CONFIDENCE)
             "people_m2_far": far, "far_confidence": FAR_CONFIDENCE if far is not None else None,
             "heat_url": f"/api/aiview/{view_id}.png?{'sample=1&' if sample else ''}v={key[0]}",
             "bg_url": f"/api/aiview/{view_id}/bg.png?v={key[0]}" if bg else None}
    _ai_cache[(view_id, sample)] = (key, info, data)
    return info, data


def _metric_only_cameras() -> set[str]:
    try:
        cams = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, ValueError, KeyError):
        return set()
    return {k for k, c in cams.items() if c.get("ref_image")}


def _shot_aiview() -> tuple[dict, bytes] | None:
    """"What the AI sees" for the current shot when no camera is known (method c): the density on a relative scale,
    no floor and no outline, labelled approximate."""
    png = LIVE_DIR / "heatmap_shot.png"
    shot = get_shot()
    if shot is None or not png.exists():
        return None
    key = (png.stat().st_mtime_ns,)
    hit = _ai_cache.get(("shot", False))
    if hit and hit[0] == key:
        return hit[1], hit[2]
    img = cv2.imread(str(png))
    if img is None:
        return None
    h, w = img.shape[:2]
    v, _line = aiview.density_values(img)
    ok, data = cv2.imencode(".png", aiview.colourise(v, 0.71))
    info = {"size": [w, h], "fit": None, "zones": {}, "label": [w / 2, h * 0.5], "outline": None, "k": 0.71,
            "absolute": False, "view_id": "shot", "ts": shot["ts"], "density": None, "visible": None, "count": None,
            "level": "unknown", "sample": False, "has_outline": False, "method": shot["method"], "how": shot["how"],
            "people_m2": shot["people_m2"], "people_m2_utc": shot["ts"], "approximate": True,
            "heat_url": f"/api/aiview/shot.png?v={key[0]}", "bg_url": None}
    _ai_cache[("shot", False)] = (key, info, data.tobytes())
    return info, data.tobytes()


_fc_cache: dict = {"t": 0.0, "val": None}


def get_forecast() -> dict:
    """Live forecast (cached 60 s); {"available": false, ...} until a model is trained on real data."""
    if _fc_cache["val"] is not None and time.monotonic() - _fc_cache["t"] < 60:
        return _fc_cache["val"]
    try:
        val = crowd_forecast.predict_now(db.get_engine(), FORECAST_DIR)
    except Exception as e:  # a broken model must not take the dashboard down
        log.exception("forecast failed")
        val = {"available": False, "reason": f"forecast error: {e}"}
    _fc_cache.update(t=time.monotonic(), val=val)
    return val


_fc_next_cache: dict = {"t": 0.0, "val": None}


def get_forecast_next() -> dict:
    """The forecast line (cached 60 s): the nightly models on the latest 5-minute reading, 30 and 60 min ahead."""
    if _fc_next_cache["val"] is not None and time.monotonic() - _fc_next_cache["t"] < 60:
        return _fc_next_cache["val"]
    try:
        from scripts import forecast_nightly

        val = forecast_nightly.predict_live()
    except Exception as e:  # a broken model must not take the dashboard down
        log.exception("forecast line failed")
        val = {"available": False, "reason": f"forecast error: {e}"}
    _fc_next_cache.update(t=time.monotonic(), val=val)
    return val


ALERTS = alert_rules.AlertEngine()


def get_alerts(crowd_now: list[dict], forecast: dict) -> dict:
    """Evaluate the alert rules now (open / hold / cooldown state lives in ALERTS) + readiness of each rule."""
    now = datetime.now(timezone.utc)
    mataf = get_mataf_now()
    conds = alert_rules.mataf_conditions(mataf, forecast, now)  # crowd-safety levels (2026-10-27)
    conds += alert_rules.road_conditions(get_traffic_series(0.5), now)
    ALERTS.update(now, conds)
    has_fc = any((f.get("view_id"), f.get("zone")) == ("mataf", "all") for f in forecast.get("cameras", [])) \
        if forecast.get("available") else False
    rules = {
        "crowd_now": {"ready": mataf is not None,
                      "note": f"the Mataf is crowded (from {alert_rules.CROWDED_M2:.1f} people per m², watch) or packed "
                              f"(from {alert_rules.PACKED_M2:.1f}, high), {alert_rules.MATAF_AVG_MIN}-min mean"
                      if mataf is not None else "starts with the first Mataf reading"},
        "crowd_forecast": {"ready": has_fc,
                           "note": f"the Mataf forecast reaches crowded (from {alert_rules.CROWDED_M2:.1f} per m²) within "
                                   f"{alert_rules.FORECAST_WITHIN_MIN} min" if has_fc else
                           "starts when the forecast covers the Mataf (trained on its people per m², scripts/train_forecast.py)"},
        "road": {"ready": True, "note": f"over {alert_rules.ROAD_PCT:.0f} % of road length within 1 km congested "
                                        f"on {alert_rules.ROAD_POLLS} polls in a row"},
    }
    return {"items": ALERTS.view(), "rules": rules, "cooldown_min": alert_rules.COOLDOWN_MIN}


_floor_cache: dict = {"key": None, "meta": None, "png": b"", "parts": None, "flat": {}}
CAMERAS_PATH = ROOT / "config" / "cameras.json"


def get_floor() -> tuple[dict, bytes]:
    """The merged floor map (backend/floormap.py), recomputed when a heatmap changes and once a minute (the fade)."""
    now = datetime.now(timezone.utc)
    # every m2 heatmap of the last 10 minutes (the collector keeps them in data/live/m2) and the latest per camera
    files = sorted(LIVE_DIR.glob("heatmap_*.json")) + sorted((LIVE_DIR / "m2").glob("*.json"))
    key = f"{int(now.timestamp() // 60)}-" + "-".join(str(f.stat().st_mtime_ns % 10**9) for f in files)
    if key == _floor_cache["key"]:
        return _floor_cache["meta"], _floor_cache["png"]
    try:
        cams = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, ValueError, KeyError):
        cams = {}
    cfg = load_config()
    maps = []
    for f in files:
        try:
            meta = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        vid = meta.get("view_id")
        if vid in cams and meta.get("source", "youtube") in db.MAIN_SOURCES:
            m = floormap.project(f.with_suffix(".png"), meta, cams[vid], cfg, *heat_correction(meta, vid))
            if m is not None and (now - m.ts).total_seconds() < 6 * 3600:
                maps.append(m)
    latest: dict = {}
    for m in maps:  # per camera, its newest frame is the one listed
        if m.view_id not in latest or m.ts > latest[m.view_id].ts:
            latest[m.view_id] = m
    dens, age = floormap.merge(maps, now)
    hidden = floormap.hidden_by_kaaba(maps, now, dens, age)
    speed = get_speed()
    png = floormap.render(floormap.smooth(dens), age, hidden, speed)
    short = f"{abs(hash(key)) % 10**10:010d}"
    meta = {"key": short, "png": f"/api/mataf/floor.png?v={short}", "extent_m": floormap.FLOOR_R, "cell_m": floormap.CELL,
            "frame": "twin-local: x = -X, z = Y of the m2 world frame; png row 0 = z = -extent",
            "fresh_min": floormap.FRESH_MIN, "max_age_min": floormap.MAX_AGE_MIN,
            "frames": len(maps), "cameras": [m.summary(now) for m in latest.values()],
            "hidden_m2": round(float(hidden.sum()) * floormap.CELL ** 2),
            "hidden_at": floormap.hidden_anchor(hidden)}  # twin-local (x, z) m of the "estimated" label, or None
    _floor_cache.update(key=key, meta=meta, png=png, parts=(dens, age, hidden, speed), flat={})
    return meta, png


def get_floor_flat() -> bytes:
    """The merged floor map in the main dashboard's flat level colours (floormap.render_flat; display only), made once
    per floor map."""
    get_floor()
    if "flat" not in _floor_cache["flat"] and _floor_cache["parts"] is not None:
        dens, age, hidden, speed = _floor_cache["parts"]
        _floor_cache["flat"]["flat"] = floormap.render_flat(dens, age, hidden, speed)
    return _floor_cache["flat"].get("flat", b"")


ON_AIR_S = 40  # the collector grabs a frame every 15 s: a camera identified in the last 40 s is the one on air


SHOT_PATH = LIVE_DIR / "shot.json"
SHOT_FIELDS = ("ts", "method", "how", "camera", "people_m2", "rings", "approx")


def get_shot() -> dict | None:
    """The shot on air as the live shot service last analysed it (scripts/live_shot.py, every 1-2 s): its method
    (calibrated / registered / approximate / none), camera, people per m^2 and age; None without the service."""
    try:
        s = json.loads(SHOT_PATH.read_text())
    except (OSError, ValueError):
        return None
    out = {k: s.get(k) for k in SHOT_FIELDS}
    out["rows"] = (s.get("rows") or {}).get("density")
    ts = datetime.fromisoformat(s["ts"]) if s.get("ts") else None
    out["age_s"] = round((datetime.now(timezone.utc) - ts).total_seconds()) if ts else None
    return out


def get_on_air() -> dict:
    """Which calibrated camera the live broadcast shows now: from the live shot service when it runs (its latest frame,
    under ON_AIR_S old), else the camera of the newest frame the collector identified (per-camera reference match or
    the m2 camera), if that frame is under ON_AIR_S old; else none (another shot)."""
    now = datetime.now(timezone.utc)
    shot = get_shot()
    if shot and shot["age_s"] is not None and shot["age_s"] <= ON_AIR_S:
        cam = shot["camera"] if shot["method"] in ("calibrated", "registered") else None
        return {"view_id": cam, "method": shot["method"], "last_identified": {"view_id": cam, "ts": shot["ts"]} if cam else None}
    cf, mf = db.crowd_frame, db.crowd_m2_frame
    with db.get_engine().connect() as c:
        a = c.execute(select(cf.c.ts_utc, cf.c.view_id).order_by(cf.c.ts_utc.desc()).limit(1)).first()
        try:
            b = c.execute(select(mf.c.ts_utc, mf.c.camera).where(mf.c.ring == "all").order_by(mf.c.ts_utc.desc()).limit(1)).first()
        except Exception:
            b = None
    cand = [(db.utc(r[0]), r[1]) for r in (a, b) if r and r[1] and r[1] != "kaaba"]
    ts, view = max(cand, default=(None, None))
    on = ts is not None and (now - ts).total_seconds() <= ON_AIR_S
    return {"view_id": view if on else None, "last_identified": {"view_id": view, "ts": iso(ts)} if ts else None}


def get_heatmap_times() -> dict:
    """view id -> time (UTC ISO) of its latest live heatmap (the collector writes one per accepted frame)."""
    out = {}
    for p in LIVE_DIR.glob("heatmap_*.json"):
        if p.stem == "heatmap_shot":  # the current shot is served separately (get_shot)
            continue
        try:
            out[p.stem[len("heatmap_"):]] = json.loads(p.read_text()).get("ts")
        except (OSError, ValueError):
            pass
    return out


def snapshot() -> dict:
    crowd_now, fc = get_crowd_now(), get_forecast()
    return {"type": "snapshot", "status": get_status(), "crowd": get_crowd_latest(), "crowd_now": crowd_now,
            "heatmaps": get_heatmap_times(), "floor": get_floor()[0], "on_air": get_on_air(), "shot": get_shot(),
            "mataf_now": get_mataf_now(), "forecast": fc, "alerts": get_alerts(crowd_now, fc),
            "traffic": get_traffic_latest(), "prayer": get_prayer_today()}


# ---------------------------------------------------------------- app
class Hub:
    """Keeps WebSocket clients and pushes a snapshot when the data changed."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.last_key: str | None = None
        self.crowd_key: str | None = None
        self.last: dict | None = None

    async def loop(self) -> None:
        while True:
            try:
                snap = await asyncio.to_thread(snapshot)
                key = json.dumps({k: v for k, v in snap.items() if k != "status"} |
                                 {"f": snap["status"]["last_frame_utc"], "r": snap["status"]["running"]}, default=str)
                crowd_key = json.dumps([snap["crowd"], (snap.get("mataf_now") or {}).get("minute_utc")], default=str)
                if self.crowd_key is not None and crowd_key != self.crowd_key and (snap["crowd"] or snap.get("mataf_now")):
                    await self.broadcast({"type": "reading", "crowd_now": snap["crowd_now"],  # drives the pulse
                                          "mataf_now": snap.get("mataf_now")})
                self.crowd_key = crowd_key
                self.last = snap
                if key != self.last_key:
                    self.last_key = key
                    await self.broadcast(snap)
            except Exception:
                log.exception("snapshot failed")
            await asyncio.sleep(PUSH_EVERY_S)

    async def broadcast(self, msg: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)


hub = Hub()


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(hub.loop())
    yield
    task.cancel()


app = FastAPI(title="Haram Crowd Monitor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


@app.middleware("http")
async def cache_headers(request, call_next):
    """index.html is revalidated on every load (a normal reload shows a new build); hashed assets never change."""
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith(".html"):
        resp.headers["Cache-Control"] = "no-cache"
    elif p.startswith("/assets/"):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


@app.get("/api/meta")
async def api_meta():
    """Public facts for the dashboard: the YouTube video id of the live stream (from STREAM_URL), SBA's HLS stream of
    the same broadcast (the player's fallback) and how long YouTube is still paused (s; the player uses SBA then)."""
    import os
    import re

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    m = re.search(r"(?:v=|youtu\.be/|/live/)([\w-]{11})", os.getenv("STREAM_URL", ""))
    return {"youtube_id": m.group(1) if m else None, "sba_hls": sba_url(),
            "youtube_paused_s": round(youtube_paused(ytdlp_state())), "haram": {"lat": 21.4225, "lon": 39.8262}}


@app.get("/api/status")
async def api_status():
    return await asyncio.to_thread(get_status)


@app.get("/api/views")
async def api_views():
    return await asyncio.to_thread(get_views)


@app.get("/api/crowd/latest")
async def api_crowd_latest():
    return await asyncio.to_thread(get_crowd_latest)


@app.get("/api/crowd/now")
async def api_crowd_now():
    return await asyncio.to_thread(get_crowd_now)


@app.get("/api/crowd/series")
async def api_crowd_series(hours: float = Query(24, gt=0, le=24 * 14)):
    return await asyncio.to_thread(get_crowd_series, hours)


@app.get("/api/mataf/now")
async def api_mataf_now():
    return await asyncio.to_thread(get_mataf_now)


@app.get("/api/mataf/floor")
async def api_mataf_floor():
    return (await asyncio.to_thread(get_floor))[0]


@app.get("/api/mataf/floor.png")
async def api_mataf_floor_png(style: str | None = Query(None)):
    # style=flat: the main dashboard's flat level colours (2026-11-01, display only); default: the twin's ramp (/?ui=old)
    data = (await asyncio.to_thread(get_floor_flat)) if style == "flat" else (await asyncio.to_thread(get_floor))[1]
    return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=600"})  # the URL changes with it


@app.get("/api/mataf/series")
async def api_mataf_series(hours: float = Query(24, gt=0, le=24 * 14)):
    return await asyncio.to_thread(get_mataf_series, hours)


@app.get("/api/traffic/latest")
async def api_traffic_latest():
    return await asyncio.to_thread(get_traffic_latest)


@app.get("/api/traffic/series")
async def api_traffic_series(hours: float = Query(24, gt=0, le=24 * 14)):
    return await asyncio.to_thread(get_traffic_series, hours)


@app.get("/api/traffic/segments")
async def api_traffic_segments():
    return await asyncio.to_thread(get_traffic_segments)


@app.get("/api/prayer/today")
async def api_prayer_today():
    return await asyncio.to_thread(get_prayer_today)


@app.get("/api/prayer/window")
async def api_prayer_window(back: float = Query(20, ge=0, le=48), ahead: float = Query(4, ge=0, le=48)):
    return await asyncio.to_thread(get_prayer_window, back, ahead)


@app.get("/api/heatmap/{view_id}.png")
async def api_heatmap(view_id: str):
    if view_id not in {v["id"] for v in load_config()["views"]}:
        raise HTTPException(404, "unknown view")
    path = LIVE_DIR / f"heatmap_{view_id}.png"
    if not path.exists():
        raise HTTPException(404, "no accepted frame yet")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/forecast")
async def api_forecast():
    return await asyncio.to_thread(get_forecast)


@app.get("/api/forecast/next")
async def api_forecast_next():
    return await asyncio.to_thread(get_forecast_next)


@app.get("/api/alerts")
async def api_alerts():
    snap = hub.last
    if snap and "alerts" in snap:
        return snap["alerts"]
    crowd_now, fc = await asyncio.to_thread(get_crowd_now), await asyncio.to_thread(get_forecast)
    return await asyncio.to_thread(get_alerts, crowd_now, fc)


@app.get("/api/aiview/{view_id}/bg.png")
async def api_aiview_bg(view_id: str):
    try:
        await asyncio.to_thread(get_aiview, view_id, False)
    except KeyError:
        raise HTTPException(404, "unknown view")
    bg = _ai_bg.get((view_id, False))
    if not bg:
        raise HTTPException(404, "no background for this camera")
    return Response(bg, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/aiview/{view_id}.png")  # declared before the JSON route, which would also match "x.png"
async def api_aiview_png(view_id: str, sample: bool = False):
    try:
        r = await asyncio.to_thread(get_aiview, view_id, sample)
    except KeyError:
        raise HTTPException(404, "unknown view")
    if r is None:
        raise HTTPException(404, "no accepted frame yet")
    return Response(r[1], media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/aiview/{view_id}")
async def api_aiview(view_id: str, sample: bool = False):
    try:
        r = await asyncio.to_thread(get_aiview, view_id, sample)
    except KeyError:
        raise HTTPException(404, "unknown view")
    if r is None:
        raise HTTPException(404, "no accepted frame yet")
    return r[0]


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        await websocket.send_json(hub.last or await asyncio.to_thread(snapshot))
        while True:
            await websocket.receive_text()  # keep the connection open; clients do not send anything
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


DIST = ROOT / "dashboard" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")
