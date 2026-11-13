"""Synthetic density zoom test: is the density in the visible part of the zone the same when the shot zooms?
It sets each camera's accepted zoom range (user decision 2026-10-24: zoom-in up to ~2.0x, zoom-out strict).

The primary reference is digitally zoomed (0.90..2.00x) about several points of its zone. For each synthetic frame
the visible part of the zone, the crowd model and the density per 10k primary px are computed exactly like a live
reading (edge/view_matcher.visible_parts + part_readings).
- drift, same patch = density at zoom s / density of the SAME floor patch in the unzoomed reference - 1: the model's
  zoom bias. Digital zoom-in only magnifies (people bigger but blurrier than with a real optical zoom), so it is a
  pessimistic proxy for zoom-in; zoom-out shrinks people like a real zoom-out.
- drift, whole zone = density at zoom s / density of the whole zone unzoomed - 1: what the series sees when a shot
  shows only part of the zone (model bias + that part being more or less crowded than the whole zone).
Range rule (RANGE_RULE): walking out from zoom 1.0, a tested zoom passes if the mean same-patch drift over the zoom
centres is within DRIFT_LIMIT and no single centre is beyond 2 x DRIFT_LIMIT; the limit is where the mean drift crosses
DRIFT_LIMIT, interpolated between the last passing and the first failing zoom (the drift curves are smooth), else the
last passing zoom. zoom_min is never below the view's old tolerance (zoom-out stays strict: people get too small).
"""
from __future__ import annotations

import cv2
import numpy as np

from edge.view_matcher import _cell_weights, _zone_geometry, graphics_mask, part_readings, visible_parts

SCALES_OUT = (0.975, 0.95, 0.925, 0.90)  # nearest first
SCALES_IN = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00)
CENTRE_OFFSETS = ((0.0, 0.0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))  # x zone bbox, from its centroid
DRIFT_LIMIT = 0.10
RANGE_RULE = (f"zoom where the mean same-patch drift crosses {DRIFT_LIMIT:.0%} (every zoom closer to 1.0 passing, no centre "
              f"beyond {2 * DRIFT_LIMIT:.0%}), interpolated between tested zooms; zoom-out never below the old tolerance")


def zoom_about(img: np.ndarray, scale: float, centre: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(digitally zoomed frame, 3x3 homography reference px -> frame px); `centre` stays where it is."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(centre, 0, scale)
    frame = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return frame, np.vstack([M, [0.0, 0.0, 1.0]])


def run(img: np.ndarray, zones: dict, model, config: dict) -> dict:
    """Density drift per zone and zoom: {"zones": {zone: {"base_density", "points": [...]}}}."""
    h, w = img.shape[:2]
    size = (w, h)
    areas, masks = _zone_geometry(zones, size)
    gm = graphics_mask(config, size)
    d1 = model.predict(img).density
    base = part_readings(visible_parts(np.eye(3), size, size, masks, areas, gm), d1)
    out = {}
    for zone, zm in masks.items():
        if zone not in base:
            continue
        ys, xs = np.nonzero(zm)
        cx, cy, bw, bh = xs.mean(), ys.mean(), xs.max() - xs.min(), ys.max() - ys.min()
        jobs = [(s, (w / 2, h / 2)) for s in SCALES_OUT]
        jobs += [(s, (cx + fx * bw, cy + fy * bh)) for s in SCALES_IN for fx, fy in CENTRE_OFFSETS]
        points = []
        for s, c in jobs:
            frame, H = zoom_about(img, s, c)
            vis = visible_parts(H, size, size, {zone: zm}, {zone: areas[zone]}, gm)
            r = part_readings(vis, model.predict(frame).density).get(zone)
            if r is None:
                continue
            patch = vis[zone]["mask_primary"]  # the same floor patch, unzoomed
            same = float((d1 * _cell_weights(patch, d1.shape[1], d1.shape[0])).sum())
            points.append({"scale": s, "centre": [round(c[0]), round(c[1])], "visible": r["visible"],
                           "density": r["density"], "count": r["count"],
                           "drift_same_patch": round(r["count"] / same - 1, 4) if same > 0 else None,
                           "drift_whole_zone": round(r["density"] / base[zone]["density"] - 1, 4)})
        out[zone] = {"base_density": base[zone]["density"], "base_count": base[zone]["count"], "points": points}
    return {"zones": out}


def summarize(points: list[dict]) -> list[dict]:
    """Per scale: mean / max |drift| over the zoom centres (same patch and whole zone), mean visible fraction."""
    rows = []
    for s in sorted({p["scale"] for p in points}):
        ps = [p for p in points if p["scale"] == s and p["drift_same_patch"] is not None]
        if not ps:
            continue
        same = np.array([p["drift_same_patch"] for p in ps])
        whole = np.array([p["drift_whole_zone"] for p in ps])
        rows.append({"scale": s, "n": len(ps), "visible_mean": round(float(np.mean([p["visible"] for p in ps])), 3),
                     "same_mean": round(float(same.mean()), 4), "same_maxabs": round(float(np.abs(same).max()), 4),
                     "whole_mean": round(float(whole.mean()), 4), "whole_maxabs": round(float(np.abs(whole).max()), 4)})
    return rows


def _limit(rows: list[dict], scales: tuple) -> float:
    """Walking out from 1.0 over `scales`: where the mean same-patch drift crosses DRIFT_LIMIT (see RANGE_RULE)."""
    by = {r["scale"]: r for r in rows}
    s0, d0 = 1.0, 0.0
    for s in scales:
        r = by.get(s)
        if r is None:
            break
        d = abs(r["same_mean"])
        if d <= DRIFT_LIMIT and r["same_maxabs"] <= 2 * DRIFT_LIMIT:
            s0, d0 = s, d
            continue
        if d > DRIFT_LIMIT > d0:  # failed on the mean: interpolate the crossing
            return s0 + (s - s0) * (DRIFT_LIMIT - d0) / (d - d0)
        break
    return s0


def pick_range(rows: list[dict], old_tolerance: float) -> tuple[float, float]:
    """(zoom_min, zoom_max) by RANGE_RULE."""
    lo, hi = _limit(rows, SCALES_OUT), _limit(rows, SCALES_IN)
    return round(max(lo, 1 - old_tolerance), 3), round(max(hi, 1 + old_tolerance), 3)  # never stricter than the old
    #                                                     tolerance on zoom-in, never looser on zoom-out
