# Track B: database, TomTom traffic, prayer times

Status: built and tested on 2026-10-23 (5 real TomTom polls, 3 Aladhan calls).
Code: `edge/db.py`, `edge/traffic.py`, `edge/prayer_times.py`, `scripts/collect_aux.py`.

## 1. Design in short

```
 TomTom vector flow tiles ──(every 5 min, 4 tiles)──► edge/traffic.py ──┐
 Aladhan calendar API ─────(1 call per month)──────► edge/prayer_times.py ─┤──► edge/db.py ──► data/haram.db
 crowd collector (step 4, later) ─────────────────► insert_crowd_minute() ┘     (SQLite, WAL)
```

- **One database module** (`edge/db.py`): SQLAlchemy Core (no ORM), all table definitions, a
  portable `upsert()` (`INSERT ... ON CONFLICT DO UPDATE`, works on SQLite and PostgreSQL),
  the crowd functions, `init_db()` (idempotent).
- **SQLite now, PostgreSQL/TimescaleDB later.** Only `DATABASE_URL` changes (section 7).
  Default `sqlite:///data/haram.db`; a relative path is resolved against the project root, so
  every script finds the same file no matter where it is started from.
- **WAL mode** on every SQLite connection (`journal_mode=WAL`, `synchronous=NORMAL`,
  `busy_timeout=10000`): the dashboard can read while the collectors write.
- **Time**: every timestamp is stored in UTC. On SQLite a DateTime is text
  (`2026-10-23 18:05:55.000000`, UTC, no offset); on PostgreSQL it is `TIMESTAMPTZ`.
  Everything read back is turned into an aware UTC datetime (`db.utc()`). Makkah time (UTC+3,
  no DST) is used only for display and for the prayer-times date key.
- **5-minute averages** (`crowd_5min`) are computed in Python, not SQL, so there is no
  SQLite-only (`strftime`) or PostgreSQL-only (`date_bin`) code.

## 2. Schema

| Table | Primary key | Columns | Written by |
|---|---|---|---|
| `crowd_minute` | minute_utc, view_id, zone, source | count_avg, density_avg, zoom_avg, n_frames | crowd collector (step 4), `insert_crowd_minute()` |
| `traffic_poll` | poll_utc | zoom, tiles_total, tiles_ok, n_segments, n_requests, error | `traffic.save_poll()` |
| `traffic_segment` | segment_key | tile, road_category, road_subcategory, road_type, coverage, length_m, mid_lat, mid_lon, dist_km, geom (JSON `[[lat,lon],...]`), first_seen_utc, last_seen_utc | `traffic.save_poll()` |
| `traffic_segment_poll` | poll_utc, segment_key | traffic_level, closed | `traffic.save_poll()` |
| `traffic_area` | poll_utc, area, road_category | n_segments, length_km, mean_level, congested_pct, closed_pct | `traffic.save_poll()` |
| `prayer_times` | date (Makkah local), method | hijri_date, fajr, sunrise, dhuhr, asr, maghrib, isha (UTC), fetched_utc | `prayer_times.ensure_prayer_times()` |

Notes:
- `crowd_minute.source` is part of the key: `youtube` (1080p) and `fallback` (720p) readings
  for the same minute are kept apart (compare before mixing). `crowd_5min()` keeps
  them apart too (optional `source=` filter). It weights each minute by `n_frames`.
- Traffic is split into a small static table (`traffic_segment`, one row per segment key,
  with geometry for the dashboard map) and a narrow per-poll table (`traffic_segment_poll`,
  4 columns, `WITHOUT ROWID` on SQLite). `traffic_poll` has one row per poll, also failed
  ones (`tiles_ok < tiles_total`, `error` set), so uptime can be measured.
- `traffic_area.area`: `r1km` and `r3km` = roads inside a circle of 1 km / 3 km around the
  Haram. `road_category` = `all` or one TomTom category (`motorway`, `trunk`, `primary`,
  `secondary`, `tertiary`, `street`).
- `prayer_times.hijri_date` (`YYYY-MM-DD`) is stored too: Ramadan and Hajj season will be
  strong forecast features.

Functions the next steps will use:

```python
from edge import db, prayer_times, traffic
db.insert_crowd_minute([db.CrowdMinute(minute_utc, "cam0_mataf_low", "mataf", 1234.5, 5.6, 1.02, 4, "youtube")])
db.crowd_5min(start_utc, end_utc)                    # -> list[CrowdBucket] per 5 min / view / zone / source
prayer_times.prayer_context(ts_utc)                  # -> prev/next prayer + minutes since/until
prayer_times.load_events(t0, t1); prayer_times.context_from_events(events, ts)   # fast, many timestamps
```

## 3. TomTom Traffic Flow vector tiles

Endpoint (verified 2026-10-23 on docs.tomtom.com, "Vector Flow Tiles", Maps API v4):

```
GET https://api.tomtom.com/traffic/map/4/tile/flow/relative/{z}/{x}/{y}.pbf
    ?key=***&tags=[road_type,traffic_level,traffic_road_coverage,road_closure,road_category,road_subcategory]&margin=0
```

- `relative` style: `traffic_level` = current speed / free-flow speed, 0.00 to 1.00
  (`absolute` would give km/h, `relative-delay` only shows congested roads).
- One layer, `Traffic flow`, extent 4096. Properties: `road_type` (FRC name, e.g. "Major
  road"), `traffic_level`, `traffic_road_coverage` (`full` / `one_side`), `left_hand_traffic`
  and `road_closure` (only present when true). `road_category` and `road_subcategory` are
  on-demand tags: they are only sent when listed in `tags`.
- `margin=0`: no buffer around the tile (default 0.1 = 10 %), so neighbour tiles do not overlap.
  The code also clips every line to the tile box.
- Responses: ~8-18 KB per tile, `Expires` = 60 s after `Date` (TomTom refreshes about every
  minute), ETag per tile. Errors: 400 bad request, 403 key not valid, 429 too many requests,
  500/503 server, 596 unknown version.
- The docs say a newer TomTom Orbis Maps traffic API exists and is recommended for new
  integrations. The v4 endpoint works with our key; see open issues.

Decoding: `mapbox-vector-tile` 2.2.0 (`y_coord_down=True`). Each feature is a LineString or a
MultiLineString (TomTom merges lines with the same properties). Every single line becomes one
segment:

- key = `z/x/y/` + first 12 hex chars of SHA-1 of (road_type, integer tile coordinates); a
  duplicate in the same tile gets `.2`, `.3`;
- length (m) and midpoint from a local metric projection around the Haram;
- `dist_km` = midpoint distance to the Haram (haversine);
- kept only if part of it lies within 3 km of the Haram.

### Zoom and tiles

Haram center 21.4225 N, 39.8262 E. Tiles needed for a square of +-3 km:

| Zoom | Tile size | Tiles | Note |
|---|---|---|---|
| 12 | 9.11 km | 2 (2x1) | |
| **13** | **4.55 km** | **4 (2x2)** | **chosen** |
| 14 | 2.28 km | 9 (3x3) | Haram almost at the tile center |
| 15 | 1.14 km | 36 (6x6) | |

Measured (one tile each at z13, z14, z15, compared inside the z15 tile next to the Haram):
the road content is the same at all three zooms: total length 10.32 / 10.22 / 10.22 km, same
categories (primary, secondary, street incl. minor local streets), length-weighted mean level
0.890 / 0.888 / 0.888. A z13 tile still has 1.1 m resolution (4.55 km / 4096). So z13 gives
the same roads as z14 at less than half the requests. The 4 tiles
(`13/5001/3596, 13/5002/3596, 13/5001/3597, 13/5002/3597`) reach 3.3 km east, 3.4 km north,
5.8 km west and 5.7 km south of the Haram: the whole 3 km circle is inside. Ibrahim Al Khalil,
Ajyad, King Abdulaziz Rd, Al Masjid Al Haram Rd and the tunnels/ring roads within 3 km are
covered. Zoom and radius are flags (`--zoom 14` would use 9 tiles).

### Area aggregates and congestion threshold

Per poll, for each circle (`r1km`, `r3km`) and for `all` + each road category:
- `n_segments`, `length_km` = road length inside the circle (both directions of a two-way road
  count when TomTom reports them separately as `one_side`);
- `mean_level` = mean `traffic_level` weighted by length inside the circle (open roads only);
- `congested_pct` = % of open length with `traffic_level < 0.5`, i.e. traffic moves at less
  than half its free-flow speed (constant `CONGESTED_BELOW` in `edge/traffic.py`);
- `closed_pct` = % of length with `road_closure`.

Weights are the length **inside** the circle, not "midpoint inside": TomTom re-splits a road
whenever the traffic level along it changes, and with the midpoint rule the r1km length jumped
31.2 -> 34.6 km between two polls 2.5 min apart. With the in-circle length it is 32.8 km in
every poll. Aggregates are only written when all tiles of a poll arrived (otherwise they would
be biased); the poll row is written anyway.

### Errors, retries, quota protection

- Timeout 20 s per request. Timeouts, connection errors and 5xx: 3 attempts per tile, waits
  2 s, 4 s. If a tile still fails, the rest of the poll is skipped (no aggregates without all
  tiles), the poll row is stored with `error`.
- 400/404/596: no retry (our request is wrong).
- 429: stop the poll, no TomTom request for 15 min, doubling up to 6 h.
- 401/403 (key invalid, not allowed, or quota used up): stop, 60 min, doubling up to 6 h.
- Backoff and the request counter live in `data/.tomtom_state.json`, so a restart cannot skip
  them. Every HTTP attempt (also retries and failures) is counted per UTC day. A poll is refused
  if it would exceed the daily cap (default 2,000, flag `--daily-cap`).
- The key is a URL query parameter. It is read with `os.getenv` and never logged: every error
  text goes through `traffic.redact()` (`key=***`), exceptions are not chained, and the
  `urllib3` logger is held at INFO (at DEBUG it would log full URLs). Tested with a fake local
  server: 500, 400, 429, 403, connection refused, corrupt PBF; the key never appeared in the
  output.

### Quota math

TomTom free tier (docs.tomtom.com/pricing, checked 2026-10-23): **Traffic Flow & Incidents
API, vector tiles: 200,000 free requests per month** (the pricing changed on 1 July 2026 from
the old daily allowance of 50,000 tile + 2,500 non-tile requests per day to monthly free
tiers). Over the limit TomTom answers 429/403; check the plan of your key in the TomTom
developer dashboard.

| Setup | Requests / day | Requests / 31-day month | Share of 200,000 |
|---|---|---|---|
| z13, 4 tiles every 5 min (default) | 4 x 288 = 1,152 | 35,712 | 18 % |
| z14, 9 tiles every 5 min | 9 x 288 = 2,592 | 80,352 | 40 % |
| daily cap 2,000 (worst case incl. retries) | 2,000 | 62,000 | 31 % |
| what the free tier allows | ~6,450 | 200,000 | 100 % |

Polling every 5 min matches the 5-minute forecast buckets; the tiles refresh every ~60 s, so
faster polling would add requests, not much information.

## 4. Prayer times (Aladhan)

```
GET https://api.aladhan.com/v1/calendar/{year}/{month}
    ?latitude=21.4225&longitude=39.8262&method=4&iso8601=true&timezonestring=Asia/Riyadh
```

- Method 4 = Umm Al-Qura University, Makkah (response meta: Fajr 18.5 deg, Isha "90 min").
- One call = one Gregorian month. `iso8601=true` gives `2026-10-23T05:03:00+03:00`, converted
  to UTC before storing. Hijri date from `date.hijri.date`.
- `ensure_prayer_times(days_ahead=31)` checks the DB for every date from the 1st of the current
  month (Makkah time) to today + 31 days, i.e. current + next month, and calls the API only for
  months with missing dates. Second run: 0 calls. 2 attempts, 30 s timeout.
- `prayer_context(ts_utc)` -> previous / next prayer (Fajr, Dhuhr, Asr, Maghrib, Isha; sunrise
  optional with `include_sunrise=True`), their UTC times, minutes since / until. Returns
  `None` when the needed days are not in the DB. On Fridays Dhuhr is the Jumu'ah prayer
  (use the weekday as a separate feature).

Today (2026-10-23, Hijri 1448-05-12), Makkah local time: Fajr 05:03, Sunrise 06:20,
Dhuhr 12:05, Asr 15:24, Maghrib 17:50, Isha 19:20.

## 5. Test results (2026-10-23, Friday evening, after Isha)

`scripts/collect_aux.py --once`, 5 times, real API:

| Poll (UTC) | Tiles | Segments | r3km mean level | r3km congested | r1km mean level | r1km congested | r1km length |
|---|---|---|---|---|---|---|---|
| 18:02:08 | 4/4 | 498 | 0.824 | 14.8 % | 0.793 | 23.6 % | 32.8 km |
| 18:02:33 | 4/4 | 501 | 0.824 | 14.7 % | 0.789 | 24.4 % | 32.8 km |
| 18:03:02 | 4/4 | 504 | 0.826 | 13.9 % | 0.799 | 23.2 % | 32.8 km |
| 18:04:46 | 4/4 | 516 | 0.824 | 14.3 % | 0.793 | 22.8 % | 32.8 km |
| 18:05:55 | 4/4 | 557 | 0.821 | 14.0 % | 0.795 | 22.0 % | 32.8 km |

Each poll: 4 requests, ~0.9 s for all 4 tiles. Prayer check: 2 Aladhan calls in the first run
(October + November 2026, 61 days), 0 calls in the four runs after it.

(The first four polls were stored with an older "midpoint within 3 km" filter; their
aggregates were recomputed with the in-circle rule. From poll 5 on, segments that cross the
3 km line are stored too, hence 557 segments.)

Per category, poll 18:05:55Z:

| Area | Category | Segments | Length | Mean level | Congested |
|---|---|---|---|---|---|
| r1km | primary | 20 | 7.1 km | 0.507 | 55.4 % |
| r1km | secondary | 21 | 13.6 km | 0.900 | 8.0 % |
| r1km | tertiary | 3 | 1.0 km | 0.846 | 29.4 % |
| r1km | street | 50 | 11.0 km | 0.847 | 17.0 % |
| r3km | primary | 135 | 71.4 km | 0.806 | 16.3 % |
| r3km | secondary | 95 | 40.3 km | 0.804 | 13.6 % |
| r3km | tertiary | 18 | 10.9 km | 0.833 | 21.6 % |
| r3km | street | 309 | 77.4 km | 0.841 | 11.0 % |
| r3km | all | 557 | 200.1 km | 0.821 | 14.0 % |

No motorway/trunk inside 3 km, no closed roads at that time. The primary roads within 1 km of
the Haram are the slowest (about half their free-flow speed).

Segment keys: 94 % of keys were the same in two polls 25 s apart, 83 % after 2.5 min (TomTom
re-splits lines when levels change). Database size after 5 polls + 61 prayer days: 520 KB;
one poll adds ~40-70 KB (mostly `traffic_segment_poll`), so ~12-20 MB per day.

## 6. How to run

```bash
.venv/Scripts/python -m pip install -r requirements.txt   # SQLAlchemy, requests, mapbox-vector-tile, shapely, ...
.venv/Scripts/python -m edge.db --init                     # create tables, show row counts
.venv/Scripts/python -m edge.db --selftest                 # crowd_minute upsert + crowd_5min on a temp DB
.venv/Scripts/python -m edge.traffic --tiles               # which tiles a poll uses (no request)
.venv/Scripts/python -m edge.traffic --stats               # TomTom requests per day, backoff (no request)
.venv/Scripts/python -m edge.traffic --summary             # latest polls from the DB (no request)
.venv/Scripts/python -m edge.prayer_times --ensure --today # fetch missing months, print today
.venv/Scripts/python -m edge.prayer_times --at 2026-10-23T12:30   # previous/next prayer for a UTC time

.venv/Scripts/python scripts/collect_aux.py --once         # 1 traffic poll (4 requests) + prayer check
.venv/Scripts/python scripts/collect_aux.py                # 24/7: traffic every 5 min, prayers daily
```

Collector flags: `--interval 300` (seconds, polls aligned to :00, :05, ...), `--daily-cap 2000`,
`--zoom 13`, `--radius-km 3`. One log line per poll:
`traffic poll 2026-10-23 18:05:55Z: tiles 4/4, 557 segments, r3km mean level 0.821, congested 14.0%, requests today 20 / cap 2000`.
Errors (network, 5xx, 429/403, cap reached, Aladhan down) are logged and the loop goes on.

## 7. Switching to PostgreSQL / TimescaleDB later

1. Install PostgreSQL (+ TimescaleDB extension) and the driver: `pip install "psycopg[binary]"`
   (pin it in requirements.txt).
2. `.env`: `DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/haram`.
3. `python -m edge.db --init` creates the same tables (DateTime -> `TIMESTAMPTZ`,
   `WITHOUT ROWID` is ignored, WAL pragmas only run on SQLite, `upsert()` uses the PostgreSQL
   `ON CONFLICT` of the same SQLAlchemy API). Checked by compiling the DDL with the PostgreSQL
   dialect; not run against a real server.
4. Optional TimescaleDB:
   `SELECT create_hypertable('crowd_minute', 'minute_utc', migrate_data => true);`
   and the same for `traffic_segment_poll` / `traffic_area` / `traffic_poll` on `poll_utc`.
5. Move existing SQLite rows with a small copy script (select from one engine, `upsert()` into
   the other). Not written yet.

## 8. Open issues

- **Segment keys are not stable over time.** They depend on the line geometry, and TomTom
  re-splits a road when the traffic level along it changes (83 % same keys after 2.5 min). The
  area aggregates do not depend on this. For per-road time series (e.g. "Ajyad St"), sample
  fixed probe points along the main roads and take the level of the nearest line per poll.
- **Storage**: ~550 rows per poll in `traffic_segment_poll` (~150,000 rows, ~12-20 MB per day)
  and `traffic_segment` keeps growing with new keys. Fine for weeks on SQLite. If it gets too
  big: keep per-segment rows only for primary/secondary roads, or delete per-segment rows
  older than N days (the aggregates stay).
- **Orbis Maps**: TomTom recommends its newer Orbis traffic API for new integrations. If the
  v4 Maps endpoint is ever retired, switch `TILE_URL` (tile format is also MVT).
- **Free tier**: the 200,000/month figure is from the public pricing page; confirm the plan of
  the key in the TomTom dashboard. The daily cap keeps us below 31 % even with retries.
- **Pedestrian zones**: roads closed to cars around the Haram may be missing from the tiles or
  marked `road_closure`; `closed_pct` was 0 in all test polls. Watch it at prayer times and
  in Ramadan.
- **Ramadan Isha**: Umm al-Qura uses Isha = Maghrib + 120 min in Ramadan (90 min otherwise).
  Aladhan reports "90 min" for method 4; check its times in Ramadan (around Feb 2027) and
  correct with `tune` if needed.
- Aladhan publishes no rate limit; we use 2 calls per month.
- The aux collector is not running yet (no 24/7 process has been started).
