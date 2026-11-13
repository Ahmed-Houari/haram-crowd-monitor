"""OpenStreetMap outlines around the Haram for the dashboard's road map (2026-10-26). Run when OSM changes.

Writes dashboard/public/osm/haram_area.geojson (served at /osm/haram_area.geojson):
  kind "pedestrian": the pedestrian squares around the Haram, where TomTom has no car traffic to report;
  kind "haram":      the Great Mosque of Mecca's building outline (coloured by the crowd level on the map);
  kind "label":      where the label's text box lies most on walking area, clear of the mosque (label_spot).
The features are OSM relations, fetched with their member ways from the OSM API (/relation/<id>/full). They were
found on 2026-10-26 with an Overpass search around the Kaaba (relations tagged highway=pedestrian + place=square,
building=mosque within 1.5 km): the Western and North Haram Squares, and the mosque. Areas tagged area=yes or
area:highway=pedestrian: none there at that time. If OSM adds squares, add their relation ids to PEDESTRIAN.
Data (c) OpenStreetMap contributors, ODbL; the map's attribution already credits OpenStreetMap.

The rest of the walking area (south: Ajyad; east: the Mas'a side; 2026-10-26: OSM maps no pedestrian areas,
area:highway, footway areas or bridge areas there) is DERIVED, approximate: the open ground inside the ring road minus
buildings and street-level car roads (tunnels do not count: the plaza is above them). A 2 m grid, flood-filled from
the mosque outwards, kept within RING_MAX_M, minus the mosque and the mapped squares; small pieces dropped. Those
features carry "approx": true.

  .venv/Scripts/python scripts/fetch_osm_haram.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "public" / "osm" / "haram_area.geojson"
API = "https://api.openstreetmap.org/api/0.6/relation/{}/full.json"
PEDESTRIAN = {9331006: "Western Haram Square", 9331009: "North Haram Square"}  # highway=pedestrian, place=square
HARAM = {1472531: "Great Mosque of Mecca"}  # building=mosque
UA = {"User-Agent": "HaramCrowdMonitor/0.1 (dashboard map outlines; run by hand)"}
LAT, LON = 21.4225, 39.8262  # the Kaaba
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
CELL = 2.0  # m, the derivation grid
HALF = 950.0  # m, grid half-size
RING_MAX_M = 400.0  # the derived area stays this close to the Kaaba: the plazas at the mosque, not the hill (Abu Qubais) or the ring road beyond
# street-level car roads block the walking area (carriageway width, m); pedestrian ways, footways and steps do not
ROAD_W = {"motorway": 16, "trunk": 14, "primary": 12, "motorway_link": 8, "trunk_link": 8, "primary_link": 8,
          "secondary": 10, "tertiary": 9, "unclassified": 7, "residential": 7, "living_street": 6, "service": 5,
          "construction": 8}


def full(rel: int) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(API.format(rel), headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"relation {rel}: {e}; retry", file=sys.stderr)
            time.sleep(5 * 2 ** attempt)
    sys.exit(f"relation {rel}: the OSM API did not answer")


def rings(ways: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join way pieces end to end into closed rings (multipolygon members come in pieces)."""
    pieces = [list(w) for w in ways if len(w) > 1]
    out = []
    while pieces:
        ring = pieces.pop(0)
        while ring[0] != ring[-1]:
            for i, p in enumerate(pieces):
                if p[0] == ring[-1]:
                    ring += p[1:]
                elif p[-1] == ring[-1]:
                    ring += p[-2::-1]
                elif p[-1] == ring[0]:
                    ring = p[:-1] + ring
                elif p[0] == ring[0]:
                    ring = p[:0:-1] + ring
                else:
                    continue
                pieces.pop(i)
                break
            else:
                break  # an open piece (the member list is incomplete): dropped
        if len(ring) > 3 and ring[0] == ring[-1]:
            out.append(ring)
    return out


def area(ring: list[tuple[float, float]]) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:]))) / 2


def inside(pt: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y, c = pt[0], pt[1], False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            c = not c
    return c


def feature(rel: int, kind: str, name: str) -> dict:
    els = full(rel)["elements"]
    nodes = {e["id"]: (round(e["lon"], 6), round(e["lat"], 6)) for e in els if e["type"] == "node"}
    ways = {e["id"]: [nodes[n] for n in e["nodes"] if n in nodes] for e in els if e["type"] == "way"}
    r = next(e for e in els if e["type"] == "relation" and e["id"] == rel)
    members = [(m.get("role") or "outer", ways.get(m["ref"])) for m in r["members"] if m["type"] == "way"]
    outer = rings([w for role, w in members if role == "outer" and w])
    holes = rings([w for role, w in members if role == "inner" and w])
    polys = [[o] for o in sorted(outer, key=area, reverse=True)]
    for h in holes:  # each hole goes to the outer ring that contains it
        host = next((p for p in polys if inside(h[0], p[0])), None)
        if host:
            host.append(h)
    print(f"  {kind} relation/{rel} {name}: {len(polys)} outer, {len(holes)} inner rings")
    return {"type": "Feature", "properties": {"kind": kind, "osm": f"relation/{rel}", "name": name},
            "geometry": {"type": "MultiPolygon", "coordinates": [[list(map(list, ring)) for ring in p] for p in polys]}}


LABEL_BOX_M = (600, 210)  # m: the label's text box at the map's default zoom (2 lines of 10 px at ~8-10 m per px)


def label_spot(feats: list[dict]) -> list[float]:
    """Centre for the label: where a LABEL_BOX_M box lies most fully on open walking area (mapped squares + derived
    areas, holes counted as plaza), never over the mosque; ties go to the spot nearest the mosque."""
    n = int(2 * HALF / CELL)
    plaza = np.zeros((n, n), np.uint8)
    mosque = np.zeros((n, n), np.uint8)
    for f in feats:
        k = f["properties"]["kind"]
        if k == "pedestrian":
            fill(plaza, [[poly[0]] for poly in f["geometry"]["coordinates"]])
        elif k == "haram":
            fill(mosque, [[poly[0]] for poly in f["geometry"]["coordinates"]])
    ok = (plaza > 0) & (mosque == 0)
    bad = mosque > 0
    bw, bh = int(LABEL_BOX_M[0] / CELL), int(LABEL_BOX_M[1] / CELL)
    ii = cv2.integral(ok.astype(np.uint8))
    ib = cv2.integral(bad.astype(np.uint8))
    best, best_s = None, -1.0
    for y in range(0, n - bh, 4):
        for x in range(0, n - bw, 4):
            area = bw * bh
            good = ii[y + bh, x + bw] - ii[y, x + bw] - ii[y + bh, x] + ii[y, x]
            hit = ib[y + bh, x + bw] - ib[y, x + bw] - ib[y + bh, x] + ib[y, x]
            if hit:
                continue
            d = np.hypot(x + bw / 2 - n / 2, y + bh / 2 - n / 2)
            score = good / area - d * 1e-5
            if score > best_s:
                best, best_s = (x + bw / 2, y + bh / 2), score
    print(f"  label box {LABEL_BOX_M[0]} x {LABEL_BOX_M[1]} m: {100 * best_s:.0f} % on walking area, clear of the mosque")
    return to_lonlat(*best)


def main() -> None:
    feats = [feature(i, "pedestrian", n) for i, n in PEDESTRIAN.items()] + [feature(i, "haram", n) for i, n in HARAM.items()]
    feats += derive(feats)
    feats.append({"type": "Feature", "properties": {"kind": "label", "text": "Pedestrian zone\nno car traffic"},
                  "geometry": {"type": "Point", "coordinates": label_spot(feats)}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "attribution": "(c) OpenStreetMap contributors, ODbL",
                               "features": feats}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
