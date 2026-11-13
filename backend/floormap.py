"""One top view of the Mataf floor for the 3D twin: every camera's latest live heatmap projected onto the floor.

The collector writes, per accepted frame, data/live/heatmap_<view>.png (the density map, coloured relative to its
99th percentile) and a .json with that percentile ("density_top", people per density-map cell of 8 x 8 frame px) and
the frame's homography from the camera's calibrated reference view ("h_primary"). With the calibrated camera
(config/cameras.json, the m2 metric's camera for that reference) every floor cell projects into the frame:
world (head plane, edge/kaaba_geom.HEAD_Z) -> reference px -> frame px -> heatmap px. There the heatmap gives people
per frame px; the projection's Jacobian (frame px per m^2 of floor) turns that into people per m^2, and the live
correction (edge/m2_metric.load_correction, as in the m2 metric: head-size curve g and the camera's factor) undoes the
counting model's bias for small heads.
Cells behind the Kaaba (the line of sight to the cell at head height passes through it), under broadcast graphics,
outside the frame or outside the Mataf floor are not seen by that camera. Cameras merge per cell: the newest reading wins; a reading fades over 10
minutes; cells nobody saw in the last 10 minutes are grey. Approximate: the heatmap is a 320 px wide image whose
colours top out at the frame's 99th percentile (rare peaks are clipped there).

Grid: twin-local frame (dashboard Twin.tsx: x_local = -X, z_local = Y of the m2 world frame), FLOOR_R metres around the
Kaaba, CELL metres per cell; PNG row 0 = z_local = -FLOOR_R (three.js flips textures: row 0 maps to v = 1, which the
twin's floor disc puts at z_local = -FLOOR_R).
Far Mataf (2026-10-27): the real courtyard reaches 70-95 m from the Kaaba (config/mataf_floor.json), the twin draws its
arcade at TWIN_EDGE. Up to the line 25 m from the walls the twin is to scale; beyond it, every direction is squeezed
evenly so the courtyard's edge lands on the twin's arcade (schematic, like the rest of the twin).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend import aiview
from edge.kaaba_geom import CUT_Z, HEAD_Z, KAABA_L, KAABA_W, Z_TOP as KAABA_H, Camera, box_corners, silhouette
from edge.m2_metric import ALL, FAR_FROM, courtyard_edge, far_mask, wall_distance
from edge.view_matcher import graphics_mask

FLOOR_R = 62.0  # m, the twin's floor disc (dashboard matafArch.ts FLOOR_R)
TWIN_EDGE = 60.0  # m, where the twin draws the arcade's inner face (matafArch.ts ARC_IN): the courtyard's edge goes here
CELL = 0.75  # m
N = int(round(2 * FLOOR_R / CELL))
FRESH_MIN, MAX_AGE_MIN = 2.0, 10.0  # full colour up to 2 min, fading to MAX_AGE_MIN, then "not seen recently"
NO_DATA = (196, 202, 202, 34)  # RGBA: floor nobody saw recently, a light faint veil with soft edges (not an object)
# floor hidden behind the Kaaba from the cameras on air (user request 2026-10-27: "not visible / estimated", never
# drawn as quiet): the people per m^2 of the visible floor beside it at the same distance from the walls (user
# 2026-10-29: the neighbouring sectors, not the whole ring), in its real colour under a light hatch
HIDDEN_ALPHA = 220
HATCH_RGBA = (232, 236, 236, 90)
HATCH_STEP = 6  # cells between the hatch lines (one cell wide)
SECTOR_DEG = 15  # bearing sectors around the Kaaba for the hidden floor's estimate
FEATHER = 1.6  # cells (~1.2 m): the soft edge between seen and not-seen floor
HEAD_M = 0.22


def _g(h: np.ndarray, h0: float | None, k: float) -> np.ndarray:
    if h0 is None:  # a fine-tuned live model: the correction is built in (backend/app.heat_correction)
        return np.ones_like(h)
    return np.maximum(1 - np.exp(-np.power(np.maximum(h, 1e-3) / h0, k)), 0.2)


def r_far_from(theta: np.ndarray) -> np.ndarray:
    """Distance from the Kaaba's centre (m) of the line FAR_FROM m from its walls, at angles theta (rad)."""
    r = np.linspace(0, 60, 2401)
    c, s = np.cos(np.asarray(theta))[..., None], np.sin(np.asarray(theta))[..., None]
    w = wall_distance(r * c, r * s)
    return r[np.argmax(w >= FAR_FROM, axis=-1)]


_GRID: tuple | None = None


def grid() -> tuple[np.ndarray, np.ndarray]:
    """World (X, Y) of every cell centre, (N, N) each; row = z_local, column = x_local. To scale up to the line
    FAR_FROM m from the walls, squeezed beyond it (module docstring); NaN outside the twin's arcade."""
    global _GRID
    if _GRID is None:
        c = -FLOOR_R + (np.arange(N) + 0.5) * CELL
        zl, xl = np.meshgrid(c, c, indexing="ij")
        Xt, Yt = -xl, zl
        rt, th = np.hypot(Xt, Yt), np.arctan2(Yt, Xt)
        r25, edge = r_far_from(th), courtyard_edge(th)
        r = np.where(rt > r25, r25 + (rt - r25) * (edge - r25) / np.maximum(TWIN_EDGE - r25, 1e-3), rt)
        f = np.where(rt <= TWIN_EDGE, r / np.maximum(rt, 1e-6), np.nan)
        _GRID = (Xt * f, Yt * f)
    return _GRID


@dataclass
class CameraMap:
    view_id: str
    ts: datetime
    density: np.ndarray  # (N, N) people per m^2, NaN where this camera did not see the floor
    hidden: np.ndarray | None = None  # (N, N) in this camera's frame but behind / under the Kaaba

    def summary(self, now: datetime) -> dict:
        seen = np.isfinite(self.density)
        return {"view_id": self.view_id, "ts": self.ts.isoformat(timespec="seconds"),
                "age_min": round((now - self.ts).total_seconds() / 60, 1), "cells": int(seen.sum()),
                "mean_people_m2": round(float(np.nanmean(self.density)), 3) if seen.any() else None,
                "p95_people_m2": round(float(np.nanpercentile(self.density, 95)), 3) if seen.any() else None}


def homography(meta: dict, cam: dict) -> list | None:
    """Reference px -> frame px: the per-camera pipeline's "h_primary", or (m2 heatmaps, 2026-10-26) from the frame's
    own calibrated camera ("cam") and the camera's calibrated reference, exact on the head plane."""
    if meta.get("h_primary"):
        return meta["h_primary"]
    c = meta.get("cam")
    if not c:
        return None
    ref = Camera(cam["f"], np.asarray(cam["rvec"], float), np.asarray(cam["tvec"], float), tuple(cam["size"]))
    fr = Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
    H = fr.plane_to_image(HEAD_Z) @ np.linalg.inv(ref.plane_to_image(HEAD_Z))
    return (H / H[2, 2]).tolist()


def behind_kaaba(c: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """For points pts (n, 3): does the straight line from the camera centre c to the point pass through the Kaaba
    (a box KAABA_L x KAABA_W x KAABA_H on the floor)? Slab test."""
    lo = np.array([-KAABA_L / 2, -KAABA_W / 2, 0.0])
    hi = np.array([KAABA_L / 2, KAABA_W / 2, KAABA_H])
    d = pts - c
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - c) / d, (hi - c) / d
    tmin = np.where(d == 0, np.where((c >= lo) & (c <= hi), -np.inf, np.inf), np.minimum(t0, t1)).max(1)
    tmax = np.where(d == 0, np.where((c >= lo) & (c <= hi), np.inf, -np.inf), np.maximum(t0, t1)).min(1)
    return (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)


def project(png: Path, meta: dict, cam: dict, config: dict, h0: float, k: float, factor: float = 1.0) -> CameraMap | None:
    """One camera's latest heatmap on the floor grid, or None when it cannot be placed (h0, k, factor: the correction)."""
    H = homography(meta, cam)
    top = meta.get("density_top")
    if not H or not top or not meta.get("ts"):
        return None
    img = cv2.imread(str(png))
    if img is None:
        return None
    v, _line = aiview.density_values(img)  # 0..255, linear in density / density_top
    hh, hw = v.shape
    fw, fh = meta.get("frame_size") or cam["size"]
    ref = Camera(cam["f"], np.asarray(cam["rvec"], float), np.asarray(cam["tvec"], float), tuple(cam["size"]))
    Hm = np.asarray(H, float)
    to_heat = np.diag([hw / fw, hh / fh, 1.0]) @ Hm  # reference px -> heatmap px
    X, Y = grid()
    pts = np.stack([np.nan_to_num(X.ravel()), np.nan_to_num(Y.ravel()), np.full(X.size, HEAD_Z)], 1)

    def heat_px(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv, depth = ref.project(P)
        q = np.column_stack([uv, np.ones(len(uv))]) @ to_heat.T
        return q[:, :2] / q[:, 2:3], depth

    uv, depth = heat_px(pts)
    d = 0.25  # m: the Jacobian from neighbouring points, in frame px per metre
    ux, _ = heat_px(pts + [d, 0, 0])
    uy, _ = heat_px(pts + [0, d, 0])
    s = fw / hw  # frame px per heatmap px
    J = np.abs((ux[:, 0] - uv[:, 0]) * (uy[:, 1] - uv[:, 1]) - (ux[:, 1] - uv[:, 1]) * (uy[:, 0] - uv[:, 0])) * (s / d) ** 2
    ok = (depth > 1) & (uv[:, 0] >= 0) & (uv[:, 0] <= hw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= hh - 1)
    # not seen: behind the Kaaba (the line from the camera to the floor point at head height passes through it), under
    # broadcast graphics. People in front of the Kaaba's base are seen (2026-10-27: the whole box's silhouette used to
    # drop them, 77-140 people 0-3.5 m from the walls in a low-camera frame; the m2 metric always counted them)
    blocked = graphics_mask(config, (hw, hh)).astype(np.uint8)
    ix = np.clip(np.round(uv[:, 0]).astype(int), 0, hw - 1)
    iy = np.clip(np.round(uv[:, 1]).astype(int), 0, hh - 1)
    wall = np.hypot(np.maximum(np.abs(X.ravel()) - KAABA_L / 2, 0), np.maximum(np.abs(Y.ravel()) - KAABA_W / 2, 0))
    floor = np.isfinite(X.ravel()) & (wall >= 0.5)
    behind = behind_kaaba(ref.centre, pts)
    # cells the shot service's fixes estimated (edge/m2_fill.LiveFixes; only the latest heatmap carries its mask)
    est = np.zeros(len(ix), bool)
    if meta.get("est") and png.name == f"heatmap_{meta.get('view_id')}.png":
        em = cv2.imread(str(png.with_name(meta["est"])), cv2.IMREAD_GRAYSCALE)
        if em is not None:
            em = cv2.resize(em, (hw, hh), interpolation=cv2.INTER_NEAREST) > 0
            est = em[iy, ix]
    # not measured but in view: behind the Kaaba, under the broadcast graphics (user 2026-10-30: estimated and hatched
    # like the floor behind the Kaaba), or estimated by the fixes
    hidden = ok & floor & (behind | (blocked[iy, ix] > 0) | est)
    ok &= (blocked[iy, ix] == 0) & ~behind & floor & ~est
    val = cv2.remap(v.astype(np.float32), uv[:, 0].astype(np.float32).reshape(-1, 1),
                    uv[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    people_px = (val / 255.0) * float(top) / 64.0  # density-map cells are 8 x 8 frame px
    zoom = float(np.sqrt(abs(np.linalg.det(Hm[:2, :2]))))
    head_px = ref.f * zoom * HEAD_M / np.maximum(depth, 1e-3)
    dens = people_px * J / (_g(head_px, h0, k) * factor)
    out = np.where(ok, dens, np.nan).reshape(N, N)
    ts = datetime.fromisoformat(meta["ts"])
    return CameraMap(meta.get("view_id", png.stem), ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc), out,
                     hidden.reshape(N, N))


def image_density(png: Path, meta: dict, cam: dict, config: dict, h0: float, k: float, factor: float = 1.0) -> dict | None:
    """The other direction, for "What the AI sees": people per m^2 at every heatmap pixel that shows the Mataf floor
    (NaN elsewhere), the floor mask, the metric's ring (1-25 m from the Kaaba's walls) and the Kaaba's silhouette, in
    heatmap px."""
    H = homography(meta, cam)
    top = meta.get("density_top")
    img = cv2.imread(str(png)) if H and top else None
    if img is None:
        return None
    v, _line = aiview.density_values(img)
    hh, hw = v.shape
    fw, fh = meta.get("frame_size") or cam["size"]
    ref = Camera(cam["f"], np.asarray(cam["rvec"], float), np.asarray(cam["tvec"], float), tuple(cam["size"]))
    Hm = np.asarray(H, float)
    to_heat = np.diag([hw / fw, hh / fh, 1.0]) @ Hm
    back = np.linalg.inv(to_heat)  # heatmap px -> reference px
    jj, ii = np.meshgrid(np.arange(hw + 1, dtype=float), np.arange(hh + 1, dtype=float))
    q = np.column_stack([jj.ravel(), ii.ravel(), np.ones(jj.size)]) @ back.T
    ref_px = q[:, :2] / q[:, 2:3]
    XY = ref.image_to_plane(ref_px, HEAD_Z)
    _uv, depth = ref.project(np.column_stack([XY, np.full(len(XY), HEAD_Z)]))
    X, Y = XY[:, 0].reshape(hh + 1, hw + 1), XY[:, 1].reshape(hh + 1, hw + 1)
    front = (depth > 1).reshape(hh + 1, hw + 1)
    # floor area under each heatmap pixel (its four corners), the pixel centre's position
    ax, ay = X[:-1, 1:] - X[:-1, :-1], Y[:-1, 1:] - Y[:-1, :-1]
    bx, by = X[1:, :-1] - X[:-1, :-1], Y[1:, :-1] - Y[:-1, :-1]
    area = np.abs(ax * by - ay * bx)
    cx, cy = (X[:-1, :-1] + X[1:, 1:]) / 2, (Y[:-1, :-1] + Y[1:, 1:]) / 2
    ok = front[:-1, :-1] & front[1:, 1:] & (area > 1e-4) & (area < 400)  # was < 25 m^2: the far floor was dropped
    r = np.hypot(cx, cy)
    wall = np.hypot(np.maximum(np.abs(cx) - KAABA_L / 2, 0), np.maximum(np.abs(cy) - KAABA_W / 2, 0))
    floor = ok & (r <= courtyard_edge(np.arctan2(cy, cx))) & (wall >= 0.5)
    in_edge = floor.copy()  # the floor in view inside the courtyard's edge, before the hidden parts come off
    sil = silhouette(ref)
    kaaba = None
    if sil is not None:
        p = np.column_stack([sil[0], np.ones(len(sil[0]))]) @ to_heat.T
        kaaba = p[:, :2] / p[:, 2:3]  # the whole box, for drawing
    cut = silhouette(ref, box_corners(z0=CUT_Z))  # the m2 metric's cut-out: people in front of the base stay floor
    if cut is not None:
        p = np.column_stack([cut[0], np.ones(len(cut[0]))]) @ to_heat.T
        m = np.zeros((hh, hw), np.uint8)
        cv2.fillPoly(m, [np.int32(np.round(p[:, :2] / p[:, 2:3]))], 1)
        floor &= m == 0
        in_edge &= m == 0  # the Kaaba itself is drawn as the Kaaba, not hatched
    floor &= graphics_mask(config, (hw, hh)) == 0
    est = np.zeros((hh, hw), bool)  # cells the shot service's fixes estimated (only the latest heatmap carries its mask)
    if meta.get("est") and png.name == f"heatmap_{meta.get('view_id')}.png":
        em = cv2.imread(str(png.with_name(meta["est"])), cv2.IMREAD_GRAYSCALE)
        if em is not None:
            est = (cv2.resize(em, (hw, hh), interpolation=cv2.INTER_NEAREST) > 0) & floor
    s = fw / hw
    people = (v.astype(float) / 255.0) * float(top) / 64.0 * s * s  # people per heatmap pixel
    zoom = float(np.sqrt(abs(np.linalg.det(Hm[:2, :2]))))
    dist = np.hypot(cx - ref.centre[0], cy - ref.centre[1])
    head_px = ref.f * zoom * HEAD_M / np.maximum(np.hypot(dist, ref.centre[2] - HEAD_Z), 1e-3)
    dens = np.where(floor, people / np.maximum(area, 1e-4) / (_g(head_px, h0, k) * factor), np.nan)
    # floor under the broadcast graphics (user 2026-10-30): estimated from the nearest visible floor at the same
    # distance from the walls (edge/m2_fill.fill_masked_fast), shown hatched as "estimated", not grey
    from edge.m2_fill import fill_masked_fast

    under = in_edge & ~floor & (wall >= 1.0)
    dens, est_g = fill_masked_fast(dens, under, wall, np.arctan2(cy, cx), np.where(ok, area, 0.0))
    est |= est_g
    ring = floor & (wall >= ALL[0]) & (wall < ALL[1])
    far = floor & far_mask(cx, cy) & (head_px >= 2.5)
    # the line FAR_FROM m from the walls, in heatmap px (head plane), for the thin line between "outer" and "far"
    th = np.linspace(-np.pi, np.pi, 361)
    rr = r_far_from(th)
    q, zq = ref.project(np.column_stack([rr * np.cos(th), rr * np.sin(th), np.full_like(th, HEAD_Z)]))
    q = np.column_stack([q, np.ones(len(q))]) @ to_heat.T
    line = (q[:, :2] / q[:, 2:3])[zq > 0]
    return {"dens": dens, "floor": floor, "ring": ring, "far": far, "line25": line, "kaaba": kaaba,
            "hidden": in_edge & ~floor & ~est,  # floor inside the edge not measured and not estimated (grey hatch)
            "est": est,  # estimated: under the graphics from the same ring nearby, or by the fixes (light hatch)
            "px_per_m": np.where(ok, 1 / np.sqrt(np.maximum(area, 1e-4)), np.nan)}


def merge(maps: list[CameraMap], now: datetime) -> tuple[np.ndarray, np.ndarray]:
    """(people per m^2, age in minutes) per cell: the newest camera that saw it; NaN / inf where none did."""
    dens = np.full((N, N), np.nan)
    age = np.full((N, N), np.inf)
    for m in sorted(maps, key=lambda m: m.ts):  # newer frames overwrite older ones
        a = (now - m.ts).total_seconds() / 60
        seen = np.isfinite(m.density)
        dens[seen], age[seen] = m.density[seen], a
    return dens, age


def hidden_by_kaaba(maps: list[CameraMap], now: datetime, dens: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Floor cells that a camera of the last MAX_AGE_MIN had in its frame but behind the Kaaba, and that no camera saw
    in that time (user request 2026-10-27: shown as "not visible / estimated", never as quiet)."""
    hid = np.zeros((N, N), bool)
    for m in maps:
        if m.hidden is not None and (now - m.ts).total_seconds() / 60 <= MAX_AGE_MIN:
            hid |= m.hidden
    return hid & ~(np.isfinite(dens) & (age <= MAX_AGE_MIN))


def estimate_hidden(dens: np.ndarray, age: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """People per m^2 for the hidden cells, from the neighbouring sectors at the same distance (user 2026-10-29): per
    1 m band of distance from the Kaaba's walls and SECTOR_DEG sector of bearing, the mean of the recently seen floor;
    a hidden cell takes the nearest seen sector on each side of it in its band, weighted by how near each is (one side
    only if the band was seen on one side); NaN where its band was not seen at all."""
    X, Y = grid()
    Xn, Yn = np.nan_to_num(X), np.nan_to_num(Y)
    band = np.floor(wall_distance(Xn, Yn)).astype(int)
    ns = int(round(360 / SECTOR_DEG))
    sec = np.floor((np.degrees(np.arctan2(Yn, Xn)) % 360) / SECTOR_DEG).astype(int) % ns
    seen = np.isfinite(dens) & (age <= MAX_AGE_MIN)
    nb = int(band.max()) + 1
    idx = band * ns + sec
    s = np.bincount(idx[seen], dens[seen], nb * ns).reshape(nb, ns)
    n = np.bincount(idx[seen], minlength=nb * ns).reshape(nb, ns)
    mean = np.where(n >= 3, s / np.maximum(n, 1), np.nan)
    est = np.full((nb, ns), np.nan)
    k = np.arange(ns)
    for b in np.unique(band[hidden]):
        ok = np.flatnonzero(np.isfinite(mean[b]))
        if not len(ok):
            continue
        fwd, back = (ok[None, :] - k[:, None]) % ns, (k[:, None] - ok[None, :]) % ns  # sector steps to each seen one
        df, db = fwd.min(1), back.min(1)
        vf, vb = mean[b, ok[fwd.argmin(1)]], mean[b, ok[back.argmin(1)]]
        est[b] = np.where(df + db > 0, (vf * db + vb * df) / np.maximum(df + db, 1), vf)
    return np.where(hidden, est[band, sec], np.nan)


def hidden_anchor(hidden: np.ndarray) -> list[float] | None:
    """Twin-local (x, z) in metres for the hidden floor's "estimated" label: on its largest patch, the mean bearing
    around the Kaaba at the median distance from the centre; None when nothing is hidden."""
    if not hidden.any():
        return None
    _, lab, stats, _ = cv2.connectedComponentsWithStats(hidden.astype(np.uint8), connectivity=8)
    c = -FLOOR_R + (np.arange(N) + 0.5) * CELL
    ii, jj = np.nonzero(lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    x, z = c[jj], c[ii]
    a = np.arctan2(np.sin(np.arctan2(z, x)).mean(), np.cos(np.arctan2(z, x)).mean())
    r = float(np.median(np.hypot(x, z)))
    return [round(r * float(np.cos(a)), 1), round(r * float(np.sin(a)), 1)]


SMOOTH_CELLS = float(np.sqrt(aiview.MIN_AREA_M2 / (2 * np.pi)) / CELL)  # the AI view's MIN_AREA_M2 average (~1.7 cells)


def smooth(dens: np.ndarray, sigma: float = SMOOTH_CELLS) -> np.ndarray:
    """People per m^2 averaged over aiview.MIN_AREA_M2 of floor (a crowd level describes an area, not a few people;
    user 2026-10-27); unseen cells stay unseen (normalised convolution)."""
    w = np.isfinite(dens).astype(np.float32)
    num = cv2.GaussianBlur(np.nan_to_num(dens).astype(np.float32) * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(w > 0, num / np.maximum(den, 1e-6), np.nan)


def render(dens: np.ndarray, age: np.ndarray, hidden: np.ndarray | None = None, speed: dict | None = None) -> bytes:
    """RGBA PNG: people per m^2 on the twin's ramp; faded with age; grey where nothing is recent; hidden (floor behind
    the Kaaba, under the graphics, estimated: hidden_by_kaaba): the estimate (estimate_hidden) in its colour at
    HIDDEN_ALPHA under a light hatch. speed (backend get_speed, 2026-10-30): the walking sectors of each ring in the
    ring's reading (ring averages only), the estimated ones under the same light hatch; stopped sectors keep the
    count model's floor."""
    # the dashboard's crowd-safety levels in people per m^2 (backend/aiview.py LEVELS; rgb below, with the soft edges)
    recent = np.isfinite(dens) & (age <= MAX_AGE_MIN)
    fade = np.clip(1 - (np.where(np.isfinite(age), age, MAX_AGE_MIN) - FRESH_MIN) / (MAX_AGE_MIN - FRESH_MIN), 0, 1)
    # soft edges: blend the data layer into the "not seen" veil over ~1 m; colours extended a little past the edge
    seen = cv2.GaussianBlur(recent.astype(np.float32), (0, 0), FEATHER)[..., None]
    w = recent.astype(np.float32)
    ext = cv2.GaussianBlur(np.where(recent, np.nan_to_num(dens), 0).astype(np.float32), (0, 0), 2 * FEATHER) / \
        np.maximum(cv2.GaussianBlur(w, (0, 0), 2 * FEATHER), 1e-6)
    rgb = aiview.ramp_m2(np.where(recent, dens, ext))
    a_data = 235 * (0.45 + 0.55 * cv2.GaussianBlur(np.where(recent, fade, 0).astype(np.float32), (0, 0), FEATHER)
                    / np.maximum(seen[..., 0], 1e-6))
    out = np.zeros((N, N, 4), np.uint8)
    out[..., :3] = np.clip(rgb * seen + np.array(NO_DATA[:3], np.float32) * (1 - seen), 0, 255)
    out[..., 3] = np.clip(a_data * seen[..., 0] + NO_DATA[3] * (1 - seen[..., 0]), 0, 255)
    if hidden is not None and hidden.any():
        est = estimate_hidden(dens, age, hidden)
        has = hidden & np.isfinite(est)
        out[has, :3] = np.clip(aiview.ramp_m2(est[has]), 0, 255)
        out[has, 3] = HIDDEN_ALPHA
        ii, jj = np.indices((N, N))
        lines = hidden & (((ii + jj) % HATCH_STEP) == 0)
        k = HATCH_RGBA[3] / 255  # the hatch lightens the colour under it, the estimate stays readable
        out[lines, :3] = np.clip(out[lines, :3] * (1 - k) + np.array(HATCH_RGBA[:3]) * k, 0, 255)
        out[lines & ~has, 3] = HATCH_RGBA[3] + 60
    X, Y = grid()
    if speed:
        wall = np.hypot(np.maximum(np.abs(X) - KAABA_L / 2, 0), np.maximum(np.abs(Y) - KAABA_W / 2, 0))
        deg = np.degrees(np.arctan2(Y, X)) % 360
        ii, jj = np.indices((N, N))
        lines = ((ii + jj) % HATCH_STEP) == 0
        k = HATCH_RGBA[3] / 255
        for ring, (d0, d1) in (("inner", (1.0, 10.0)), ("outer", (10.0, 25.0))):
            info, secs = (speed.get("rings") or {}).get(ring), (speed.get("sectors") or {}).get(ring)
            if not info or info.get("speed_ms") is None or not secs:
                continue
            how = np.array([s["how"] for s in secs])
            sec = np.clip((np.nan_to_num(deg) // float(speed.get("sector_deg", 15))).astype(int), 0, len(secs) - 1)
            paint = np.isfinite(X) & (wall >= d0) & (wall < d1) & np.isin(how[sec], ["walking", "slow", "fast", "estimated"])
            out[paint, :3] = np.clip(aiview.ramp_m2(np.array([info["value"]]))[0], 0, 255)
            out[paint, 3] = 235
            h = paint & (how[sec] == "estimated") & lines
            out[h, :3] = np.clip(out[h, :3] * (1 - k) + np.array(HATCH_RGBA[:3]) * k, 0, 255)
    out[~np.isfinite(X)] = 0  # off the floor (beyond the twin's arcade): transparent
    ok, png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
    return png.tobytes()


FLAT_UP = 3  # the flat map is drawn 3x finer than the map (smooth level edges at the twin's size)
FLAT_ALPHA, FLAT_EST_ALPHA = 232, 205
FLAT_HATCH_STEP, FLAT_HATCH_W = 7 * FLAT_UP, 2  # px of the 3x texture: a light diagonal hatch on estimated floor


def _speed_rings(speed: dict | None, X: np.ndarray, Y: np.ndarray, value: np.ndarray, est: np.ndarray) -> None:
    """In place: the walking sectors of each ring take the ring's speed reading (as render() does), the estimated ones
    flagged estimated."""
    if not speed:
        return
    wall = np.hypot(np.maximum(np.abs(X) - KAABA_L / 2, 0), np.maximum(np.abs(Y) - KAABA_W / 2, 0))
    deg = np.degrees(np.arctan2(Y, X)) % 360
    for ring, (d0, d1) in (("inner", (1.0, 10.0)), ("outer", (10.0, 25.0))):
        info, secs = (speed.get("rings") or {}).get(ring), (speed.get("sectors") or {}).get(ring)
        if not info or info.get("speed_ms") is None or not secs:
            continue
        how = np.array([s["how"] for s in secs])
        sec = np.clip((np.nan_to_num(deg) // float(speed.get("sector_deg", 15))).astype(int), 0, len(secs) - 1)
        paint = np.isfinite(X) & (wall >= d0) & (wall < d1) & np.isin(how[sec], ["walking", "slow", "fast", "estimated"])
        value[paint] = info["value"]
        est[paint] = how[sec][paint] == "estimated"


def render_flat(dens: np.ndarray, age: np.ndarray, hidden: np.ndarray | None = None, speed: dict | None = None) -> bytes:
    """RGBA PNG for the main dashboard (user 2026-11-01: "clean flat colours, no glitter or rainbow noise, estimated
    areas lightly hatched"): the floor map smoothed over ~2 m and cut into regions of the four level colours, smooth
    edges, no ramp, no fading. Display only; the readings are unchanged. Estimated (hatched): the floor behind the
    Kaaba (estimate_hidden) and the walking sectors the speed sensor estimated. Floor nobody saw in MAX_AGE_MIN: the
    faint "not seen" veil."""
    X, Y = grid()
    on = np.isfinite(X)
    recent = np.isfinite(dens) & (age <= MAX_AGE_MIN)
    est = np.zeros((N, N), bool)
    value = smooth(np.where(recent, dens, np.nan), 2.6).astype(np.float32)
    if hidden is not None and hidden.any():
        e = estimate_hidden(dens, age, hidden)
        has = hidden & np.isfinite(e) & ~np.isfinite(value)
        value[has] = e[has]
        est |= has
    _speed_rings(speed, X, Y, value, est)
    value[~on] = np.nan
    # 3x finer: the value (filled a little past the seen floor so edges do not darken) and the masks, then the levels
    big = (N * FLAT_UP, N * FLAT_UP)
    have = np.isfinite(value).astype(np.float32)
    filled = cv2.GaussianBlur(np.nan_to_num(value) * have, (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(have, (0, 0), 1.0), 1e-6)
    filled = np.where(have > 0, np.nan_to_num(value), filled).astype(np.float32)
    vbig = cv2.resize(filled, big[::-1], interpolation=cv2.INTER_LINEAR)
    hbig = cv2.resize(have, big[::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
    ebig = cv2.resize(est.astype(np.float32), big[::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
    onbig = cv2.resize(on.astype(np.float32), big[::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
    lvl = np.digitize(vbig, aiview.LEVELS)
    out = np.zeros(big + (4,), np.uint8)
    out[onbig] = NO_DATA
    out[hbig, :3] = aiview.LEVEL_RGB[lvl[hbig]].astype(np.uint8)
    out[hbig, 3] = np.where(ebig[hbig], FLAT_EST_ALPHA, FLAT_ALPHA)
    ii, jj = np.indices(big)
    lines = hbig & ebig & (((ii + jj) % FLAT_HATCH_STEP) < FLAT_HATCH_W)
    kh = HATCH_RGBA[3] / 255
    out[lines, :3] = np.clip(out[lines, :3] * (1 - kh) + np.array(HATCH_RGBA[:3]) * kh, 0, 255).astype(np.uint8)
    out[~onbig] = 0
    ok, png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
    return png.tobytes()
