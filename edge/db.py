"""Local database: SQLite now, PostgreSQL/TimescaleDB later (only DATABASE_URL changes).

SQLAlchemy Core, no ORM. All timestamps are stored in UTC. On SQLite a DateTime column is
text ("2026-10-23 17:55:00.000000", UTC wall clock, no offset), on PostgreSQL it is
TIMESTAMPTZ. Values read back are always returned as timezone-aware UTC datetimes.

SQLite runs in WAL mode, so a reader (dashboard) and the collectors can work at the same time.

Tables (created by init_db(), idempotent):
- crowd_minute          raw per-minute crowd readings per view/zone/source: density in the visible part of the zone
                        (main series), raw count there, visible fraction, people in the whole zone (fully visible only)
- traffic_poll          one row per TomTom poll (tiles fetched, segments, requests)
- traffic_segment       static info per road segment (category, length, midpoint, geometry)
- traffic_segment_poll  traffic level per segment per poll
- traffic_area          per-poll aggregates per area ring and road category
- prayer_times          daily prayer times (Makkah local date, times in UTC)
- crowd_frame           every frame x zone with a reading (zoom 0.8-2.4, >= 30 % visible), accepted flag, homography
- collector_minute      crowd-collector heartbeat per minute (frames, accepted, source) -> uptime
- crowd_m2_frame        view-independent metric (2026-10-24): people per m^2 on the Mataf floor per frame and ring
                        around the Kaaba (camera from a calibrated reference or fitted to the Kaaba, edge/m2_pipeline.py)
- crowd_m2_minute       its per-minute averages per ring (one Mataf series for all cameras)
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from sqlalchemy import (Boolean, Column, Date, DateTime, Engine, Float, Integer, MetaData, String, Table, Text,
                        create_engine, event, func, inspect, or_, select)
from sqlalchemy.engine import Connection, make_url

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = "sqlite:///data/haram.db"  # relative paths are resolved against the project root

metadata = MetaData()
UTCDateTime = DateTime(timezone=True)

crowd_minute = Table(
    "crowd_minute", metadata,
    Column("minute_utc", UTCDateTime, primary_key=True),
    Column("view_id", String(64), primary_key=True),
    Column("zone", String(64), primary_key=True),
    Column("source", String(16), primary_key=True),  # "youtube" (1080p), "sba" (SBA primary since 2026-10-27, 720p
    # scaled to 1080p) or "fallback" (SBA as YouTube's fallback before 2026-10-27, 720p)
    Column("count_avg", Float, nullable=False),  # raw count in the visible part of the zone (whole zone before 2026-10-24)
    Column("density_avg", Float),  # MAIN SERIES: people per 10,000 px of the visible zone part (primary-reference px)
    Column("zoom_avg", Float),
    Column("n_frames", Integer, nullable=False),
    Column("visible_avg", Float),  # mean visible fraction of the zone (0..1); 1.0 for rows before 2026-10-24
    Column("count_full_avg", Float),  # people in the whole zone: mean over frames with the zone fully visible, else NULL
)

traffic_poll = Table(
    "traffic_poll", metadata,
    Column("poll_utc", UTCDateTime, primary_key=True),
    Column("zoom", Integer, nullable=False),
    Column("tiles_total", Integer, nullable=False),
    Column("tiles_ok", Integer, nullable=False),
    Column("n_segments", Integer, nullable=False),
    Column("n_requests", Integer, nullable=False),  # HTTP requests incl. retries
    Column("error", String(200)),
)

traffic_segment = Table(
    "traffic_segment", metadata,
    Column("segment_key", String(48), primary_key=True),  # "z/x/y/<geometry hash>"
    Column("tile", String(24), nullable=False),
    Column("road_category", String(16), nullable=False),  # motorway trunk primary secondary tertiary street
    Column("road_subcategory", String(16)),  # connecting major_local local minor_local
    Column("road_type", String(32)),  # TomTom FRC name, e.g. "Major road"
    Column("coverage", String(8)),  # "full" (whole road / one-way) or "one_side" (one direction)
    Column("length_m", Float, nullable=False),
    Column("mid_lat", Float, nullable=False),
    Column("mid_lon", Float, nullable=False),
    Column("dist_km", Float, nullable=False),  # midpoint distance to the Haram center
    Column("geom", Text),  # JSON [[lat, lon], ...]
    Column("first_seen_utc", UTCDateTime, nullable=False),
    Column("last_seen_utc", UTCDateTime, nullable=False),
)

traffic_segment_poll = Table(
    "traffic_segment_poll", metadata,
    Column("poll_utc", UTCDateTime, primary_key=True),
    Column("segment_key", String(48), primary_key=True),
    Column("traffic_level", Float),  # relative speed = current / free-flow, 0..1
    Column("closed", Boolean, nullable=False),
    sqlite_with_rowid=False,  # narrow table, many rows; ignored on PostgreSQL
)

traffic_area = Table(
    "traffic_area", metadata,
    Column("poll_utc", UTCDateTime, primary_key=True),
    Column("area", String(16), primary_key=True),  # "r1km", "r3km": road length inside that circle around the Haram
    Column("road_category", String(16), primary_key=True),  # "all" or a road_category
    Column("n_segments", Integer, nullable=False),
    Column("length_km", Float, nullable=False),
    Column("mean_level", Float),  # length-weighted mean traffic_level of open segments
    Column("congested_pct", Float),  # % of open length with traffic_level < threshold (see edge/traffic.py)
    Column("closed_pct", Float),  # % of length closed
)

prayer_times = Table(
    "prayer_times", metadata,
    Column("date", Date, primary_key=True),  # Makkah local date
    Column("method", Integer, primary_key=True),  # Aladhan method id (4 = Umm al-Qura)
    Column("hijri_date", String(10)),  # "YYYY-MM-DD" (Hijri)
    *(Column(p, UTCDateTime, nullable=False) for p in ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")),
    Column("fetched_utc", UTCDateTime, nullable=False),
)

crowd_frame = Table(  # every accepted frame x zone: lets 2d (count vs zoom) and tolerance changes run on real data
    "crowd_frame", metadata,
    Column("ts_utc", UTCDateTime, primary_key=True),
    Column("view_id", String(64), primary_key=True),
    Column("zone", String(64), primary_key=True),
    Column("source", String(16), nullable=False),
    Column("count", Float, nullable=False),
    Column("density", Float),
    Column("zoom", Float, nullable=False),  # zoom factor vs. the reference view
    Column("inliers", Integer),
    Column("accepted", Boolean, nullable=False),  # zoom within the view's tolerance (counted in crowd_minute)
    Column("ref_id", String(96)),  # which reference of the view matched (several per view since 2026-10-24)
    Column("density_top", Float),  # scale of the live heatmap PNG (its colours are density / density_top)
    Column("density_max", Float),  # highest density-map cell of the frame
    Column("visible", Float),  # visible fraction of the zone (count = people in that part, density = per 10k px of it)
    Column("h_primary", Text),  # JSON 3x3 homography primary reference px -> frame px (re-analysis, AI view)
)

# columns added after the table first went live: (table, column, SQL type, SQL that fills old rows or None), added by
# init_db() if missing. Readings before the visible-part metric (2026-10-24) were all of fully visible zones.
LATE_COLUMNS = [("crowd_frame", "ref_id", "VARCHAR(96)", None), ("crowd_frame", "density_top", "FLOAT", None),
                ("crowd_frame", "density_max", "FLOAT", None),
                ("crowd_frame", "visible", "FLOAT", "UPDATE crowd_frame SET visible = 1.0"),
                ("crowd_frame", "h_primary", "TEXT", None),
                ("crowd_minute", "visible_avg", "FLOAT", "UPDATE crowd_minute SET visible_avg = 1.0"),
                ("crowd_minute", "count_full_avg", "FLOAT", "UPDATE crowd_minute SET count_full_avg = count_avg"),
                ("crowd_m2_frame", "head_bins", "TEXT", None), ("crowd_m2_frame", "sectors", "TEXT", None),
                ("crowd_m2_frame", "people_m2_v1", "FLOAT", None), ("crowd_m2_minute", "people_m2_v1", "FLOAT", None),
                ("crowd_m2_minute", "note", "VARCHAR(16)", None),
                # far Mataf (2026-10-27): 25 m to the courtyard's edge, on the ring "all" rows, not part of the rings
                ("crowd_m2_frame", "people_m2_far", "FLOAT", None), ("crowd_m2_frame", "people_m2_far_raw", "FLOAT", None),
                ("crowd_m2_frame", "area_m2_far", "FLOAT", None), ("crowd_m2_frame", "head_px_far", "FLOAT", None),
                ("crowd_m2_frame", "sectors_far", "TEXT", None),
                ("crowd_m2_minute", "people_m2_far", "FLOAT", None), ("crowd_m2_minute", "people_m2_far_raw", "FLOAT", None),
                ("crowd_m2_minute", "area_m2_far", "FLOAT", None), ("crowd_m2_minute", "n_frames_far", "INTEGER", None),
                # v3 deployed (2026-10-29): rows before it on v3's scale (scripts/rescale_m2_to_v3.py); people_m2 keeps
                # the original value, readers take the v3 value where it is set (m2_value / m2_col)
                ("crowd_m2_frame", "people_m2_v3", "FLOAT", None), ("crowd_m2_frame", "people_m2_far_v3", "FLOAT", None),
                ("crowd_m2_minute", "people_m2_v3", "FLOAT", None), ("crowd_m2_minute", "people_m2_far_v3", "FLOAT", None),
                # v4 (2026-10-30): the reading's source; old rows stay NULL (not rescaled)
                ("crowd_m2_frame", "model", "VARCHAR(16)", None), ("crowd_m2_minute", "model", "VARCHAR(16)", None),
                ("crowd_m2_minute", "people_m2_count", "FLOAT", None)]

collector_minute = Table(  # one row per minute the crowd collector was alive -> uptime metric
    "collector_minute", metadata,
    Column("minute_utc", UTCDateTime, primary_key=True),
    Column("frames", Integer, nullable=False),  # frames grabbed
    Column("accepted", Integer, nullable=False),  # frames with a reference view + visible zone
    Column("youtube", Integer, nullable=False),  # frames from YouTube (1080p)
    Column("fallback", Integer, nullable=False),  # frames from SBA (720p; the primary source since 2026-10-27)
    Column("infer_ms", Float),  # mean crowd-model time of accepted frames
)

crowd_m2_frame = Table(  # view-independent metric per frame and ring (research prototype 2026-10-24)
    "crowd_m2_frame", metadata,
    Column("ts_utc", UTCDateTime, primary_key=True),
    Column("ring", String(16), primary_key=True),  # "inner" (1-10 m from the Kaaba's walls), "outer" (10-25), "all"
    Column("source", String(16), nullable=False),
    Column("path", String(8), nullable=False),  # "ptz" (calibrated reference camera; "ptz/k2": matched through a
    #                                             reference 2 links from it, edge/ptz.py) or "kaaba" (fitted to the Kaaba)
    Column("camera", String(64)),  # reference view id for "ptz"
    Column("people_m2", Float, nullable=False),  # corrected for the crowd model's head-size bias
    Column("people_m2_raw", Float),
    Column("count", Float),  # people seen on that ring part (raw model count)
    Column("area_m2", Float),  # visible floor of the ring in this frame
    Column("head_px", Float),  # median head width in px there
    Column("zoom", Float),  # vs the reference camera (ptz)
    Column("focal_px", Float),
    Column("ptz_err", Float),
    Column("cam_x", Float), Column("cam_y", Float), Column("cam_z", Float),  # camera position, m (Kaaba frame)
    Column("head_bins", Text),  # ring "all": JSON [raw count, area m^2] per head-width bin (edge/m2_metric.HEAD_BINS)
    Column("sectors", Text),  # ring "all": JSON {"i3": [raw count, corrected count, m^2, head px], ...} per floor
    #                           sector (edge/m2_metric.SECTORS), for the cross-camera calibration (scripts/zoom_calib.py)
    Column("people_m2_v1", Float),  # rows before 2026-10-26 16:02 UTC: the original value (scripts/rescale_m2_history.py)
    # far Mataf (2026-10-27, ring "all" rows only): 25 m from the walls to the courtyard's edge (edge/m2_metric.FAR_FROM)
    Column("people_m2_far", Float), Column("people_m2_far_raw", Float), Column("area_m2_far", Float),
    Column("head_px_far", Float), Column("sectors_far", Text),
    # rows before v3's deploy (2026-10-29): the value on v3's scale (scripts/rescale_m2_to_v3.py; people_m2 is the original)
    Column("people_m2_v3", Float), Column("people_m2_far_v3", Float),
    Column("model", String(16)),  # the live model that read it (config/m2.json live_model "use", e.g. "v4"); NULL: before v4
)

crowd_m2_minute = Table(  # one Mataf series per ring for all cameras
    "crowd_m2_minute", metadata,
    Column("minute_utc", UTCDateTime, primary_key=True),
    Column("ring", String(16), primary_key=True),
    Column("source", String(16), primary_key=True),
    Column("people_m2", Float, nullable=False),  # pooled over the minute's frames: sum(corrected count) / sum(area)
    Column("people_m2_raw", Float),
    Column("area_m2", Float),  # mean visible floor per frame
    Column("n_frames", Integer, nullable=False),
    Column("cameras", String(200)),  # cameras that minute, e.g. "cam1_mataf_wide,kaaba"
    # minutes before 2026-10-26 16:02 UTC, rescaled to the new correction (scripts/rescale_m2_history.py): the original
    # value and "rescaled_v1", or "prayer_v1" (prayer rows not recoverable: left out of levels, usual and forecast)
    Column("people_m2_v1", Float),
    Column("note", String(16)),
    Column("people_m2_far", Float), Column("people_m2_far_raw", Float),  # far Mataf, ring "all" rows (2026-10-27)
    Column("area_m2_far", Float), Column("n_frames_far", Integer),
    Column("people_m2_v3", Float), Column("people_m2_far_v3", Float),  # before v3: on v3's scale (scripts/rescale_m2_to_v3.py)
    # since v4 (2026-10-30): the reading's source, "v4" (count model) or "v4+speed" (walking ring sectors from the tawaf
    # speed sensor, edge/tawaf_speed.py); people_m2_count: the count model's value where the speed changed it. NULL: before
    Column("model", String(16)), Column("people_m2_count", Float),
)

crowd_m2_shadow = Table(  # shadow run (2026-10-27, edge/shadow.py): every frame the shot service analysed, the live
    # reading next to a candidate model's (the fine-tuned DM-Count: no head-size correction, no camera factor); not read
    # by the API, the levels or the forecast
    "crowd_m2_shadow", metadata,
    Column("ts_utc", UTCDateTime, primary_key=True),
    Column("model", String(16), primary_key=True),
    Column("camera", String(64)), Column("method", String(16)), Column("how", String(40)), Column("source", String(16)),
    Column("rows", Boolean),  # the live reading came from the prayer rows
    Column("live_all", Float), Column("live_inner", Float), Column("live_outer", Float), Column("live_far", Float),
    Column("shadow_all", Float), Column("shadow_inner", Float), Column("shadow_outer", Float), Column("shadow_far", Float),
    Column("live_approx", Float), Column("shadow_approx", Float),  # frames without a camera (method "approximate")
    Column("area_all", Float), Column("head_px_all", Float), Column("count_all", Float),
)

_engines: dict[str, Engine] = {}


def utc(dt: datetime) -> datetime:
    """Aware UTC datetime. Naive input is taken as UTC (that is how SQLite returns it)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def database_url() -> str:
    """DATABASE_URL from .env (default sqlite:///data/haram.db, relative to the project root)."""
    load_dotenv(PROJECT_ROOT / ".env")
    url = make_url(os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)
    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        path = Path(url.database)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = url.set(database=str(path))
    return url.render_as_string(hide_password=False)


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
    cur.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, much faster
    cur.execute("PRAGMA busy_timeout=10000")  # wait up to 10 s for a lock instead of failing
    cur.close()


def describe(engine: Engine) -> str:
    """Readable database location for logs (file path for SQLite, URL without password otherwise)."""
    if engine.dialect.name == "sqlite":
        return f"sqlite file {engine.url.database}"
    return engine.url.render_as_string(hide_password=True)


def get_engine(url: str | None = None) -> Engine:
    """Shared engine per URL. SQLite gets WAL mode + busy timeout on every connection."""
    url = url or database_url()
    if url not in _engines:
        engine = create_engine(url, pool_pre_ping=True)
        if engine.dialect.name == "sqlite":
            event.listen(engine, "connect", _sqlite_pragmas)
        _engines[url] = engine
    return _engines[url]


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables if missing (idempotent). Returns the engine."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    have = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in {t for t, *_ in LATE_COLUMNS}}
    with engine.begin() as conn:
        for table, col, sqltype, fill in LATE_COLUMNS:
            if col not in have[table]:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}")  # nullable: old rows stay valid
                if fill:
                    conn.exec_driver_sql(fill)
                log.warning("added column %s.%s%s", table, col, " (old rows filled)" if fill else "")
    return engine


def upsert(conn: Connection, table: Table, rows: Sequence[dict], update_cols: Iterable[str] | None = None) -> int:
    """Insert rows; on primary-key conflict update `update_cols` (default: all non-key columns).

    Same code path for SQLite and PostgreSQL (both support INSERT ... ON CONFLICT DO UPDATE).
    """
    if not rows:
        return 0
    name = conn.dialect.name
    if name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise NotImplementedError(f"upsert not implemented for {name}")
    keys = [c.name for c in table.primary_key.columns]
    cols = list(update_cols) if update_cols is not None else [c.name for c in table.columns if c.name not in keys]
    stmt = insert(table)
    stmt = stmt.on_conflict_do_update(index_elements=keys, set_={c: stmt.excluded[c] for c in cols}) if cols \
        else stmt.on_conflict_do_nothing(index_elements=keys)
    conn.execute(stmt, list(rows))
    return len(rows)


# ---------------------------------------------------------------- crowd
@dataclass
class CrowdMinute:
    minute_utc: datetime
    view_id: str
    zone: str
    count_avg: float  # raw count in the visible part
    density_avg: float | None  # main series: people per 10k px of the visible part (primary-reference px)
    zoom_avg: float | None
    n_frames: int
    source: str = "youtube"  # or "fallback"
    visible_avg: float | None = None  # visible fraction of the zone
    count_full_avg: float | None = None  # people in the whole zone (frames with the zone fully visible only)


@dataclass
class CrowdBucket:
    bucket_utc: datetime  # start of the 5-minute bucket
    view_id: str
    zone: str
    source: str
    count_avg: float  # frame-weighted mean of the minute averages (raw count in the visible part)
    density_avg: float | None  # main series
    zoom_avg: float | None
    n_frames: int
    n_minutes: int
    visible_avg: float | None = None
    count_full_avg: float | None = None  # people in the whole zone, from minutes where it was fully visible


def insert_crowd_minute(rows: CrowdMinute | Iterable[CrowdMinute], engine: Engine | None = None) -> int:
    """Upsert per-minute crowd readings (key: minute_utc, view_id, zone, source)."""
    rows = [rows] if isinstance(rows, CrowdMinute) else list(rows)
    data = [{**r.__dict__, "minute_utc": utc(r.minute_utc).replace(second=0, microsecond=0)} for r in rows]
    with (engine or get_engine()).begin() as conn:
        return upsert(conn, crowd_minute, data)


def insert_crowd_frames(rows: Sequence[dict], engine: Engine | None = None) -> int:
    """Upsert frame-level zone readings: dicts with the crowd_frame columns (ts_utc aware UTC)."""
    data = [{**r, "ts_utc": utc(r["ts_utc"]).replace(microsecond=0)} for r in rows]
    with (engine or get_engine()).begin() as conn:
        return upsert(conn, crowd_frame, data)


def insert_collector_minute(minute_utc: datetime, frames: int, accepted: int, youtube: int, fallback: int,
                            infer_ms: float | None, engine: Engine | None = None) -> None:
    """Upsert one crowd-collector heartbeat row (uptime % = minutes with frames / minutes since start)."""
    row = {"minute_utc": utc(minute_utc).replace(second=0, microsecond=0), "frames": frames, "accepted": accepted,
           "youtube": youtube, "fallback": fallback, "infer_ms": infer_ms}
    with (engine or get_engine()).begin() as conn:
        upsert(conn, collector_minute, [row])


def _wmean(pairs: list[tuple[float | None, int]]) -> float | None:
    pairs = [(v, w) for v, w in pairs if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


# The main series (2026-10-27): SBA's stream became the primary source after a YouTube bot check. Its frames give the
# same readings as YouTube's (same broadcast, crop and graphics; checked on simultaneous frames, 2026-10-27), so
# both count as the broadcast's main readings. "fallback" (SBA as YouTube's fallback before) stays apart.
MAIN = "main"
MAIN_SOURCES = ("youtube", "sba")


def crowd_5min(start_utc: datetime, end_utc: datetime, source: str | None = None,
               engine: Engine | None = None) -> list[CrowdBucket]:
    """5-minute averages per view/zone/source for start_utc <= minute < end_utc.

    Bucketing is done in Python (portable: no SQLite strftime / PostgreSQL date_bin).
    Minute values are weighted by their number of frames. Sources are kept apart
    (1080p YouTube and 720p fallback counts may differ).
    """
    q = (select(crowd_minute)
         .where(crowd_minute.c.minute_utc >= utc(start_utc), crowd_minute.c.minute_utc < utc(end_utc))
         .order_by(crowd_minute.c.minute_utc))
    if source:
        q = q.where(source_filter(crowd_minute.c.source, source))
    groups: dict[tuple, list] = defaultdict(list)
    with (engine or get_engine()).connect() as conn:
        for r in conn.execute(q).mappings():
            m = utc(r["minute_utc"])
            bucket = m.replace(minute=m.minute - m.minute % 5, second=0, microsecond=0)
            groups[(bucket, r["view_id"], r["zone"], r["source"])].append(r)
    out = []
    for (bucket, view_id, zone, src), rs in sorted(groups.items()):
        w = [r["n_frames"] for r in rs]
        out.append(CrowdBucket(
            bucket, view_id, zone, src,
            count_avg=_wmean([(r["count_avg"], n) for r, n in zip(rs, w)]),
            density_avg=_wmean([(r["density_avg"], n) for r, n in zip(rs, w)]),
            zoom_avg=_wmean([(r["zoom_avg"], n) for r, n in zip(rs, w)]),
            n_frames=sum(w), n_minutes=len(rs),
            visible_avg=_wmean([(r["visible_avg"], n) for r, n in zip(rs, w)]),
            count_full_avg=_wmean([(r["count_full_avg"], n) for r, n in zip(rs, w)])))
    return out


def insert_shadow(rows: Sequence[dict], engine: Engine | None = None) -> None:
    """Upsert shadow-run rows (edge/shadow.py); every row gets every column (a batch needs the same keys)."""
    cols = [c.name for c in crowd_m2_shadow.columns]
    with (engine or get_engine()).begin() as conn:
        upsert(conn, crowd_m2_shadow, [{**{c: r.get(c) for c in cols}, "ts_utc": utc(r["ts_utc"])} for r in rows])


def insert_m2(frames: Sequence[dict], minutes: Sequence[dict], engine: Engine | None = None) -> None:
    """Upsert view-independent readings: crowd_m2_frame rows and crowd_m2_minute rows (dicts with the columns)."""
    with (engine or get_engine()).begin() as conn:
        upsert(conn, crowd_m2_frame, [{**r, "ts_utc": utc(r["ts_utc"]).replace(microsecond=0)} for r in frames])
        upsert(conn, crowd_m2_minute, [{**r, "minute_utc": utc(r["minute_utc"]).replace(second=0, microsecond=0)}
                                       for r in minutes])


def m2_value(r, col: str = "people_m2") -> float | None:
    """A Mataf reading on the live model's scale: the v3-scale value of a row from before v3's deploy (2026-10-29,
    scripts/rescale_m2_to_v3.py) where it is set, else the stored value (rows since the deploy are v3's own)."""
    v = r.get(f"{col}_v3") if hasattr(r, "get") else None
    return v if v is not None else r[col]


def m2_col(t: Table, col: str = "people_m2"):
    """The same as a column expression, labelled as the column (for select())."""
    return func.coalesce(t.c[f"{col}_v3"], t.c[col]).label(col)


def source_filter(col, source: str | None):
    """WHERE clause for a source column: MAIN ("main") = the broadcast as the live services read it (MAIN_SOURCES)."""
    return col.in_(MAIN_SOURCES) if source == MAIN else col == source


def m2_5min(start_utc: datetime, end_utc: datetime, source: str | None = MAIN,
            engine: Engine | None = None) -> list[CrowdBucket]:
    """5-minute averages of the view-independent metric as CrowdBuckets: view_id "mataf", zone = ring,
    density_avg = people per m^2 (frame-weighted), count_avg = mean visible floor m^2. source=MAIN: YouTube and SBA
    rows together (one series, source "main")."""
    t = crowd_m2_minute
    q = select(t).where(t.c.minute_utc >= utc(start_utc), t.c.minute_utc < utc(end_utc),
                        or_(t.c.note.is_(None), t.c.note != "prayer_v1")).order_by(t.c.minute_utc)
    if source:
        q = q.where(source_filter(t.c.source, source))
    groups: dict[tuple, list] = defaultdict(list)
    with (engine or get_engine()).connect() as conn:
        for r in conn.execute(q).mappings():
            m = utc(r["minute_utc"])
            src = MAIN if source == MAIN else r["source"]
            groups[(m.replace(minute=m.minute - m.minute % 5, second=0, microsecond=0), r["ring"], src)].append(r)
    out = []
    for (bucket, ring, src), rs in sorted(groups.items()):
        w = [r["n_frames"] for r in rs]
        out.append(CrowdBucket(bucket, "mataf", ring, src, count_avg=_wmean([(r["area_m2"], n) for r, n in zip(rs, w)]) or 0.0,
                               density_avg=_wmean([(m2_value(r), n) for r, n in zip(rs, w)]), zoom_avg=None,
                               n_frames=sum(w), n_minutes=len(rs)))
    return out


def table_counts(engine: Engine | None = None) -> dict[str, int]:
    """Row count per table (for status output)."""
    engine = engine or get_engine()
    existing = set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        return {t.name: conn.execute(select(func.count()).select_from(t)).scalar_one()
                for t in metadata.sorted_tables if t.name in existing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the local database and show row counts.")
    parser.add_argument("--init", action="store_true", help="create missing tables (idempotent)")
    parser.add_argument("--selftest", action="store_true",
                        help="crowd_minute upsert + crowd_5min round trip on a temporary SQLite file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            engine = init_db(create_engine(f"sqlite:///{Path(d) / 'selftest.db'}"))
            t0 = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
            rows = [CrowdMinute(t0 + timedelta(minutes=i), "cam0", "mataf", 1000 + 10 * i, 5.0, 1.0, 4,
                                visible_avg=1.0 if i % 2 else 0.5, count_full_avg=1000 + 10 * i if i % 2 else None)
                    for i in range(7)]
            insert_crowd_minute(rows, engine)
            insert_crowd_minute(CrowdMinute(t0, "cam0", "mataf", 2000, 6.0, 1.1, 2), engine)  # upsert minute 0
            for b in crowd_5min(t0, t0 + timedelta(minutes=10), engine=engine):
                print(b)
            engine.dispose()
        return
    engine = init_db() if args.init else get_engine()
    print(f"database: {describe(engine)}")
    for name, n in table_counts(engine).items():
        print(f"  {name:22s} {n:>10,d} rows")


if __name__ == "__main__":
    main()
