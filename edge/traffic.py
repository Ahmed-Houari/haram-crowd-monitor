"""Road traffic around Masjid al-Haram from TomTom Traffic Flow vector tiles.

One poll = a few vector tiles (default zoom 13, 2x2 = 4 tiles) that cover the roads within
3 km of the Haram. Each tile is a Mapbox Vector Tile (PBF) with one layer, "Traffic flow".
Every feature carries `traffic_level` (style "relative": current speed / free-flow speed,
0..1), `road_category` (motorway, trunk, primary, secondary, tertiary, street), and more.

Per poll we store:
- segments: each line of each feature that reaches into the 3 km circle around the Haram,
  with a key "z/x/y/<geometry hash>", road category, length, midpoint, distance to the
  Haram and its traffic level;
- area aggregates for circles around the Haram (1 km, 3 km), overall and per road category,
  weighted by the road length INSIDE the circle (stable when TomTom re-splits a road):
  mean traffic level, % of length congested (level < 0.5), % closed.

The API key (TOMTOM_API_KEY in .env) is a URL query parameter, so every message that could
contain a URL goes through redact(). Requests per day (incl. retries) are counted in
data/.tomtom_state.json and a daily cap is enforced. 429/403 stop the poll and back off long.
Endpoint: https://api.tomtom.com/traffic/map/4/tile/flow/relative/{z}/{x}/{y}.pbf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import mapbox_vector_tile
import requests
import shapely
from dotenv import load_dotenv
from shapely.geometry import LineString, Point

from edge import db

log = logging.getLogger(__name__)
# urllib3 logs full request URLs (with key=...) at DEBUG level: never let that through.
logging.getLogger("urllib3").setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "data" / ".tomtom_state.json"

HARAM_LAT, HARAM_LON = 21.4225, 39.8262  # Masjid al-Haram (Kaaba)
DEFAULT_ZOOM = 13  # 4 tiles cover +-3 km; same roads as z14/z15 (measured, see docs/track_b.md)
RADIUS_KM = 3.0  # segments are kept if any part is within this distance of the Haram
AREAS = {"r1km": 1.0, "r3km": 3.0}  # aggregate circles: road length inside each circle
CONGESTED_BELOW = 0.5  # congested = current speed below half the free-flow speed
CATEGORIES = ("motorway", "trunk", "primary", "secondary", "tertiary", "street")

TILE_URL = "https://api.tomtom.com/traffic/map/4/tile/flow/{style}/{z}/{x}/{y}.pbf"
STYLE = "relative"
TAGS = "[road_type,traffic_level,traffic_road_coverage,road_closure,road_category,road_subcategory]"
LAYER = "Traffic flow"
EXTENT = 4096
TIMEOUT_S = 20
ATTEMPTS = 3  # per tile, for timeouts / connection errors / 5xx
RETRY_WAIT_S = 2.0  # doubles per attempt
LONG_BACKOFF_S = {"429": 15 * 60, "403": 60 * 60}  # first wait after a 429 / 401-403, doubles
MAX_LONG_BACKOFF_S = 6 * 60 * 60
DEFAULT_DAILY_CAP = 2000  # requests per UTC day; normal use is 4 tiles x 288 polls = 1,152
EARTH_CIRCUMFERENCE_M = 40_075_016.686


class TrafficError(RuntimeError):
    pass


class QuotaError(TrafficError):
    """TomTom said 429 (too many requests) or 401/403 (key invalid / not allowed / over quota)."""


class BackoffActive(TrafficError):
    """A long backoff after a 429/403 is still running: no request is sent."""


class DailyCapReached(TrafficError):
    """The configured daily request cap would be exceeded."""


@dataclass(frozen=True)
class Tile:
    z: int
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.z}/{self.x}/{self.y}"


@dataclass
class Segment:
    key: str
    tile: str
    road_category: str
    road_subcategory: str | None
    road_type: str | None
    coverage: str | None
    traffic_level: float | None
    closed: bool
    length_m: float
    mid_lat: float
    mid_lon: float
    dist_km: float
    geom: list[tuple[float, float]]  # (lat, lon)


@dataclass
class AreaStat:
    area: str
    road_category: str  # "all" or a category
    n_segments: int
    length_km: float
    mean_level: float | None
    congested_pct: float | None
    closed_pct: float | None


@dataclass
class PollResult:
    poll_utc: datetime
    zoom: int
    tiles_total: int
    tiles_ok: int
    n_requests: int
    segments: list[Segment] = field(default_factory=list)
    areas: list[AreaStat] = field(default_factory=list)
    error: str | None = None

    def stat(self, area: str = "r3km", category: str = "all") -> AreaStat | None:
        return next((a for a in self.areas if a.area == area and a.road_category == category), None)


# ---------------------------------------------------------------- key safety
def _key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("TOMTOM_API_KEY")
    if not key:
        raise TrafficError("TOMTOM_API_KEY is not set in .env")
    return key


def redact(text: str) -> str:
    """Remove the API key from any text (URLs, exception messages)."""
    text = re.sub(r"(key=)[^&\s'\"]+", r"\1***", str(text))
    key = os.getenv("TOMTOM_API_KEY")
    return text.replace(key, "***") if key else text


# ---------------------------------------------------------------- tiles + geometry
def lonlat_to_tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Fractional Web Mercator (slippy map) tile coordinates."""
    n = 2 ** z
    r = math.radians(lat)
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(r)) / math.pi) / 2 * n


def tile_px_to_latlon(tile: Tile, px: float, py: float) -> tuple[float, float]:
    """Tile pixel (0..4096, y down) to (lat, lon)."""
    n = 2 ** tile.z
    lon = (tile.x + px / EXTENT) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + py / EXTENT) / n))))
    return lat, lon


def tiles_for_area(lat: float = HARAM_LAT, lon: float = HARAM_LON, radius_km: float = RADIUS_KM,
                   z: int = DEFAULT_ZOOM) -> list[Tile]:
    """All tiles at zoom z that touch the square of +-radius_km around (lat, lon)."""
    dlat = radius_km / 110.574
    dlon = radius_km / (111.320 * math.cos(math.radians(lat)))
    x0, y0 = lonlat_to_tile(lat + dlat, lon - dlon, z)
    x1, y1 = lonlat_to_tile(lat - dlat, lon + dlon, z)
    return [Tile(z, x, y) for y in range(int(y0), int(y1) + 1) for x in range(int(x0), int(x1) + 1)]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def local_km(lat: float, lon: float) -> tuple[float, float]:
    """(east, north) km from the Haram; equirectangular, accurate to ~0.1 % within a few km."""
    return (lon - HARAM_LON) * 111.320 * math.cos(math.radians(HARAM_LAT)), (lat - HARAM_LAT) * 110.574


@lru_cache(maxsize=8)
def _disc(radius_km: float):
    return Point(0, 0).buffer(radius_km, quad_segs=32)


def length_within_m(geom: list[tuple[float, float]], radius_km: float) -> float:
    """Length (m) of a (lat, lon) polyline inside the circle of radius_km around the Haram."""
    return LineString([local_km(la, lo) for la, lo in geom]).intersection(_disc(radius_km)).length * 1000


def _lines(geom: dict) -> list[list]:
    if geom["type"] == "LineString":
        return [geom["coordinates"]]
    if geom["type"] == "MultiLineString":
        return geom["coordinates"]
    return []


def decode_tile(raw: bytes, tile: Tile) -> list[Segment]:
    """PBF tile -> one Segment per line, clipped to the tile (no overlap with neighbours)."""
    layers = mapbox_vector_tile.decode(raw, default_options={"y_coord_down": True})
    features = layers.get(LAYER, {}).get("features", [])
    out: list[Segment] = []
    seen: dict[str, int] = {}
    for f in features:
        p = f.get("properties", {})
        for coords in _lines(f["geometry"]):
            if len(coords) < 2:
                continue
            clipped = shapely.clip_by_rect(LineString(coords), 0, 0, EXTENT, EXTENT)
            for line in getattr(clipped, "geoms", [clipped]):
                if line.is_empty or line.geom_type != "LineString" or line.length == 0:
                    continue
                pts = [(round(x), round(y)) for x, y in line.coords]
                h = hashlib.sha1(f"{p.get('road_type')}|{pts}".encode()).hexdigest()[:12]
                key = f"{tile}/{h}"
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1:  # identical line twice in one tile (e.g. both directions)
                    key = f"{key}.{seen[key]}"
                mid = line.interpolate(0.5, normalized=True)
                mid_lat, mid_lon = tile_px_to_latlon(tile, mid.x, mid.y)
                geom = [tuple(round(v, 6) for v in tile_px_to_latlon(tile, x, y)) for x, y in line.coords]
                level = p.get("traffic_level")
                out.append(Segment(
                    key=key, tile=str(tile),
                    road_category=p.get("road_category") or "unknown",
                    road_subcategory=p.get("road_subcategory"),
                    road_type=p.get("road_type"),
                    coverage=p.get("traffic_road_coverage"),
                    traffic_level=float(level) if level is not None else None,
                    closed=bool(p.get("road_closure", False)),
                    length_m=LineString([local_km(la, lo) for la, lo in geom]).length * 1000,
                    mid_lat=mid_lat, mid_lon=mid_lon,
                    dist_km=haversine_km(HARAM_LAT, HARAM_LON, mid_lat, mid_lon),
                    geom=geom,
                ))
    return out


def aggregate(segments: list[Segment], areas: dict[str, float] = AREAS,
              threshold: float = CONGESTED_BELOW) -> list[AreaStat]:
    """Stats per circle, overall ("all") and per road category, weighted by length inside the circle.

    mean_level and congested_pct use open roads only; closed_pct is the closed share of all length.
    """
    out = []
    for area, radius in areas.items():
        inside = [(s, w) for s in segments if (w := length_within_m(s.geom, radius)) > 0]
        cats = ["all"] + [c for c in CATEGORIES if any(s.road_category == c for s, _ in inside)]
        cats += sorted({s.road_category for s, _ in inside} - set(cats))  # "unknown" etc.
        for cat in cats:
            ss = inside if cat == "all" else [(s, w) for s, w in inside if s.road_category == cat]
            total = sum(w for _, w in ss)
            open_ = [(s, w) for s, w in ss if not s.closed and s.traffic_level is not None]
            open_len = sum(w for _, w in open_)
            out.append(AreaStat(
                area, cat, len(ss), total / 1000,
                mean_level=sum(s.traffic_level * w for s, w in open_) / open_len if open_len else None,
                congested_pct=100 * sum(w for s, w in open_ if s.traffic_level < threshold) / open_len
                if open_len else None,
                closed_pct=100 * sum(w for s, w in ss if s.closed) / total if total else None,
            ))
    return out


# ---------------------------------------------------------------- request counter + backoff
def load_state() -> dict:
    """Requests per UTC day, long-backoff state (shared by all scripts)."""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["calls"] = dict(sorted(state.get("calls", {}).items())[-40:])  # keep 40 days
    STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def requests_today(state: dict | None = None) -> int:
    """TomTom HTTP requests sent today (UTC day), incl. retries and failures."""
    return (state or load_state()).get("calls", {}).get(_today(), 0)


def _count_request(daily_cap: int) -> None:
    state = load_state()
    if requests_today(state) >= daily_cap:
        raise DailyCapReached(f"daily cap of {daily_cap} TomTom requests reached")
    calls = state.setdefault("calls", {})
    calls[_today()] = calls.get(_today(), 0) + 1
    save_state(state)


def _long_backoff(kind: str, msg: str) -> QuotaError:
    state = load_state()
    state["failures"] = state.get("failures", 0) + 1
    delay = min(LONG_BACKOFF_S[kind] * 2 ** (state["failures"] - 1), MAX_LONG_BACKOFF_S)
    state["next_request_after"] = time.time() + delay
    save_state(state)
    log.error("%s. No TomTom request for %.0f min.", msg, delay / 60)
    return QuotaError(msg)


def _reset_backoff() -> None:
    state = load_state()
    if state.get("failures") or state.get("next_request_after"):
        state.update(failures=0, next_request_after=0)
        save_state(state)


def backoff_remaining_s() -> float:
    return max(0.0, load_state().get("next_request_after", 0) - time.time())


class TileClient:
    """HTTP session + request counter for one poll. Every attempt is counted (also in the daily state)."""

    def __init__(self, daily_cap: int = DEFAULT_DAILY_CAP, style: str = STYLE):
        self.session = requests.Session()
        self.daily_cap = daily_cap
        self.style = style
        self.n_requests = 0

    def close(self) -> None:
        self.session.close()

    def get(self, tile: Tile) -> bytes:
        """GET one tile with retries (timeouts, connection errors, 5xx). Raises QuotaError on 429/401/403."""
        url = TILE_URL.format(style=self.style, z=tile.z, x=tile.x, y=tile.y)
        params = {"key": _key(), "tags": TAGS, "margin": "0"}
        err = ""
        for attempt in range(1, ATTEMPTS + 1):
            _count_request(self.daily_cap)
            self.n_requests += 1
            try:
                r = self.session.get(url, params=params, timeout=TIMEOUT_S)
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {redact(e)[:200]}"
            else:
                if r.status_code == 200:
                    return r.content
                if r.status_code == 429:
                    raise _long_backoff("429", f"TomTom 429 Too Many Requests on tile {tile}")
                if r.status_code in (401, 403):
                    raise _long_backoff("403", f"TomTom {r.status_code} (key invalid or not allowed) on tile {tile}")
                err = f"HTTP {r.status_code}: {redact(r.text)[:200]}"
                if r.status_code < 500:  # 400 / 404 / 596: our request is wrong, retrying will not help
                    raise TrafficError(f"tile {tile}: {err}")
            if attempt < ATTEMPTS:
                wait = RETRY_WAIT_S * 2 ** (attempt - 1)
                log.warning("tile %s attempt %d failed (%s), retry in %.0f s", tile, attempt, err, wait)
                time.sleep(wait)
        raise TrafficError(f"tile {tile} failed after {ATTEMPTS} attempts: {err}")


def poll(zoom: int = DEFAULT_ZOOM, radius_km: float = RADIUS_KM, daily_cap: int = DEFAULT_DAILY_CAP) -> PollResult:
    """Fetch all tiles once and build segments + area aggregates (not saved; see save_poll).

    Raises BackoffActive / DailyCapReached before sending anything if not allowed, QuotaError
    on 429/401/403. If a tile still fails after its retries, the rest of the poll is skipped
    (result.error is set). Area aggregates are only built when every tile arrived (otherwise
    they would be biased).
    """
    wait = backoff_remaining_s()
    if wait > 0:
        raise BackoffActive(f"TomTom backoff active for {wait / 60:.1f} more min")
    tiles = tiles_for_area(radius_km=radius_km, z=zoom)
    if requests_today() + len(tiles) > daily_cap:
        raise DailyCapReached(f"daily cap {daily_cap}: {requests_today()} used, poll needs {len(tiles)}")
    result = PollResult(datetime.now(timezone.utc).replace(microsecond=0), zoom, len(tiles), 0, 0)
    segments: list[Segment] = []
    client = TileClient(daily_cap)
    try:
        for tile in tiles:
            try:
                raw = client.get(tile)
            except (QuotaError, DailyCapReached):
                raise
            except TrafficError as e:  # aggregates need every tile: skip the rest, save requests
                result.error = str(e)[:200]
                log.error("%s; remaining tiles of this poll skipped", e)
                break
            try:
                segments += decode_tile(raw, tile)
            except Exception as e:  # corrupt / unexpected PBF
                result.error = f"tile {tile}: decode failed: {type(e).__name__}: {e}"[:200]
                log.error("%s", result.error)
                break
            result.tiles_ok += 1
    finally:
        result.n_requests = client.n_requests
        client.close()
    if result.tiles_ok:
        _reset_backoff()
    result.segments = [s for s in segments if length_within_m(s.geom, radius_km) > 0]
    if result.tiles_ok == result.tiles_total:
        result.areas = aggregate(result.segments)
    return result


def save_poll(result: PollResult, engine=None) -> None:
    """Write one poll (poll row, segment info, per-segment levels, area aggregates) in one transaction."""
    engine = engine or db.get_engine()
    t = result.poll_utc
    with engine.begin() as conn:
        db.upsert(conn, db.traffic_poll, [{
            "poll_utc": t, "zoom": result.zoom, "tiles_total": result.tiles_total, "tiles_ok": result.tiles_ok,
            "n_segments": len(result.segments), "n_requests": result.n_requests, "error": result.error}])
        db.upsert(conn, db.traffic_segment, [{
            "segment_key": s.key, "tile": s.tile, "road_category": s.road_category,
            "road_subcategory": s.road_subcategory, "road_type": s.road_type, "coverage": s.coverage,
            "length_m": round(s.length_m, 1), "mid_lat": round(s.mid_lat, 6), "mid_lon": round(s.mid_lon, 6),
            "dist_km": round(s.dist_km, 3), "geom": json.dumps(s.geom, separators=(",", ":")),
            "first_seen_utc": t, "last_seen_utc": t} for s in result.segments], update_cols=["last_seen_utc"])
        db.upsert(conn, db.traffic_segment_poll, [{
            "poll_utc": t, "segment_key": s.key,
            "traffic_level": round(s.traffic_level, 4) if s.traffic_level is not None else None,
            "closed": s.closed} for s in result.segments])
        db.upsert(conn, db.traffic_area, [{
            "poll_utc": t, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.__dict__.items()}}
            for a in result.areas])


def poll_line(result: PollResult) -> str:
    """One log line per poll."""
    s = result.stat()
    lvl = f"mean level {s.mean_level:.3f}, congested {s.congested_pct:.1f}%" if s and s.mean_level is not None \
        else "no aggregates"
    return (f"traffic poll {result.poll_utc:%Y-%m-%d %H:%M:%S}Z: tiles {result.tiles_ok}/{result.tiles_total}, "
            f"{len(result.segments)} segments, r3km {lvl}, requests today {requests_today()}")


def print_summary(engine=None, last: int = 5) -> None:
    """Latest polls from the DB: segments per poll, area mean level, per-category stats."""
    from sqlalchemy import select

    engine = engine or db.get_engine()
    with engine.connect() as conn:
        polls = conn.execute(select(db.traffic_poll).order_by(db.traffic_poll.c.poll_utc.desc()).limit(last)) \
            .mappings().all()
        for p in reversed(polls):
            print(f"{db.utc(p['poll_utc']):%Y-%m-%d %H:%M:%S}Z zoom {p['zoom']} tiles {p['tiles_ok']}/"
                  f"{p['tiles_total']} segments {p['n_segments']} requests {p['n_requests']}"
                  + (f" error: {p['error']}" if p["error"] else ""))
            rows = conn.execute(select(db.traffic_area).where(db.traffic_area.c.poll_utc == p["poll_utc"])).mappings()
            for a in rows:
                lvl = "-" if a["mean_level"] is None else f"{a['mean_level']:.3f}"
                cong = "-" if a["congested_pct"] is None else f"{a['congested_pct']:5.1f}%"
                print(f"   {a['area']:5s} {a['road_category']:10s} n={a['n_segments']:4d} "
                      f"len={a['length_km']:6.1f} km  mean={lvl}  congested={cong}  closed={a['closed_pct'] or 0:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="TomTom traffic flow tiles around the Haram.")
    parser.add_argument("--tiles", action="store_true", help="list the tiles of one poll (no request)")
    parser.add_argument("--once", action="store_true", help="one poll, save to the DB, print the summary")
    parser.add_argument("--no-save", action="store_true", help="with --once: do not write to the DB")
    parser.add_argument("--summary", action="store_true", help="show the latest polls from the DB (no request)")
    parser.add_argument("--stats", action="store_true", help="requests per day and backoff state (no request)")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    parser.add_argument("--daily-cap", type=int, default=DEFAULT_DAILY_CAP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.tiles:
        tiles = tiles_for_area(radius_km=args.radius_km, z=args.zoom)
        size_km = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(HARAM_LAT)) / 2 ** args.zoom / 1000
        print(f"zoom {args.zoom}: {len(tiles)} tiles of {size_km:.2f} km: {', '.join(map(str, tiles))}")
        return
    if args.stats:
        st = load_state()
        print(f"TomTom requests per UTC day: {st.get('calls', {})}")
        print(f"backoff: {backoff_remaining_s() / 60:.1f} min left, consecutive quota errors {st.get('failures', 0)}")
        return
    if args.once:
        result = poll(args.zoom, args.radius_km, args.daily_cap)
        log.info(poll_line(result))
        if not args.no_save:
            db.init_db()
            save_poll(result)
            print_summary(last=1)
        return
    if args.summary:
        print_summary()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
