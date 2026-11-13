"""People per square metre on the Mataf floor: the crowd model's density map + the frame's camera.

Research prototype (2026-10-24, view-independent metric). The camera comes from edge/m2_pipeline.py (a calibrated
broadcast camera through edge/ptz.py, or a fit to the Kaaba's silhouette). Every cell of the density map (8 x 8 px,
people at their heads) is traced to the head plane (HEAD_Z above the floor): its position on the Mataf and its area in
m^2. Cells are grouped into rings by distance from the Kaaba's walls; a ring's reading is people / m^2 over its
visible part. Excluded: the exclude mask (the Kaaba above head level, broadcast graphics) and rays that do not reach
the head plane in front of the camera. People standing in front of the Kaaba's base are kept.
The head size in px (HEAD_M at the cell's distance) is recorded per ring: the crowd model finds fewer people when they
look small (scripts/zoom_bias.py, scripts/zoom_bias_live.py), which the correction argument (g in
edge/m2_pipeline.py) undoes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from edge.kaaba_geom import HEAD_Z, KAABA_L, KAABA_W, Camera

RINGS = {"inner": (1.0, 10.0), "outer": (10.0, 25.0)}  # m from the Kaaba walls
ALL = (1.0, 25.0)
# far Mataf (user request 2026-10-27): 25 m from the walls out to the courtyard floor's edge (config/mataf_floor.json,
# minus its margin). A separate value (M2Reading.far, own DB columns): never part of the rings above, the headline
# ("all") or the forecast. Only floor cells whose heads are at least FAR_HEAD_MIN px count (the far side of the
# courtyard from a camera has heads of 1-2 px, where the model finds almost no one).
FAR_FROM = 25.0
FAR_HEAD_MIN = 2.5
MATAF_FLOOR = Path(__file__).resolve().parents[1] / "config" / "mataf_floor.json"
HEAD_M = 0.22  # m, head width (for the head size in px)
MIN_AREA_M2 = 40.0  # a ring needs this much visible floor for a reading
G_FLOOR = 0.2  # the correction divides by at most 1 / 0.2 (tiny heads: the model finds almost no one there)
# head width bins (px): the raw count and floor area per bin are stored with every reading, so the head-size
# correction can be refitted on live frames (scripts/zoom_bias_live.py)
HEAD_BINS = [0, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 1e9]
# floor sectors (2026-10-25, cross-camera calibration from zoomed-in frames, scripts/zoom_calib.py): each ring is cut
# into SECTORS equal angles around the Kaaba's centre; per sector the raw and corrected count, the area and the median
# head width are stored with every reading, so readings of different cameras can be compared on the same floor
SECTORS = 12
SECTOR_MIN_AREA_M2 = 10.0
# the live correction: config/m2.json "correction" (2026-10-26) = the head-size curve (h0, k) and a factor per
# calibrated camera; without it the curve in models/zoom_bias.json (the shadow-run refit), else LIVE_CURVE
ROOT = Path(__file__).resolve().parents[1]
M2_CONFIG = ROOT / "config" / "m2.json"
ZOOM_BIAS = ROOT / "models" / "zoom_bias.json"
LIVE_CURVE = (6.5, 1.2)


def source_curves() -> dict[str, tuple[float, float]]:
    """Head-size curves (h0, k) of other sources than YouTube 1080p (config/m2.json "correction" "source", e.g. "sba":
    SBA's 720p stream finds fewer of the small heads, 2026-10-27): a frame from such a source divides by its own g, so
    its readings stay on YouTube's level. None configured: every source uses the main curve."""
    try:
        c = json.loads(M2_CONFIG.read_text(encoding="utf-8")).get("correction") or {}
        return {s: (float(v["h0"]), float(v["k"])) for s, v in (c.get("source") or {}).items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def live_model_config() -> dict | None:
    """config/m2.json "live_model" when enabled (user, 2026-10-29: the fine-tuned DM-Count v3 is the live model): its
    weights, its SBA head-size curve ("source") and its approximate-path table and bias ("approx"; scripts/shadow_refit.py
    --section live_model). Such a model has the head-size correction and the camera factors built in. None: the original
    DM-Count with "correction" (load_correction).
    Since v4 (2026-10-30): "live_model" holds "enabled" and "use", the model's own block is "live_model_<use>"
    (live_model_v3, live_model_v4); one setting switches back. The result carries "name" (the use)."""
    try:
        m2 = json.loads(M2_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    c = m2.get("live_model")
    if not c or not c.get("enabled"):
        return None
    if c.get("use"):
        block = m2.get(f"live_model_{c['use']}")
        if not block:
            return None
        return {**block, "enabled": True, "since_utc": c.get("since_utc"), "name": c["use"]}
    return {**c, "name": c.get("weights")}


def load_correction() -> tuple[float, float, dict[str, float]]:
    """(h0, k, camera factors). g(h) = 1 - exp(-(h / h0)^k): the share of people the crowd model finds at head width
    h px. Camera factor: what else a camera's view misses (people hidden behind others at a shallow angle). A reading
    divides the raw count by max(g, G_FLOOR) x factor; cameras without a factor use 1."""
    try:
        c = json.loads(M2_CONFIG.read_text(encoding="utf-8")).get("correction")
    except (OSError, ValueError):
        c = None
    if c:
        return float(c["h0"]), float(c["k"]), {k: float(v) for k, v in (c.get("camera_factor") or {}).items()}
    try:
        d = json.loads(ZOOM_BIAS.read_text())
        return float(d["h0"]), float(d["k"]), {}
    except (OSError, ValueError, KeyError):
        return LIVE_CURVE[0], LIVE_CURVE[1], {}


@dataclass
class M2Reading:
    density: dict = field(default_factory=dict)  # ring -> people per m^2 (raw)
    density_corr: dict = field(default_factory=dict)  # ring -> people per m^2 after the zoom-bias correction
    count: dict = field(default_factory=dict)  # ring -> people in the visible part
    area: dict = field(default_factory=dict)  # ring -> visible m^2
    head_px: dict = field(default_factory=dict)  # ring -> median head width in px
    bins: list | None = None  # ring "all": [raw count per HEAD_BINS bin, area (m^2) per bin]
    sectors: dict | None = None  # "i3" / "o3" (inner / outer ring, sector 3) -> [raw, corrected count, m^2, head px]
    far: dict | None = None  # far Mataf: {"raw", "corr" (people per m^2), "count", "area", "head_px", "sectors"}


def cell_floor(cam: Camera, dshape: tuple[int, int], cell: float = 8.0) -> dict:
    """Per density cell: head-plane position (X, Y), area (m^2), depth (m), valid (in front, below the horizon)."""
    dh, dw = dshape
    jj, ii = np.meshgrid(np.arange(dw), np.arange(dh))
    corners = []
    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
        corners.append(_to_plane(cam, (jj + du) * cell, (ii + dv) * cell))
    centre = _to_plane(cam, (jj + 0.5) * cell, (ii + 0.5) * cell)
    xs = np.stack([c[0] for c in corners])
    ys = np.stack([c[1] for c in corners])
    area = 0.5 * np.abs(xs[0] * ys[1] - xs[1] * ys[0] + xs[1] * ys[2] - xs[2] * ys[1] + xs[2] * ys[3] - xs[3] * ys[2]
                        + xs[3] * ys[0] - xs[0] * ys[3])
    valid = np.all(np.stack([c[2] for c in corners]), axis=0) & centre[2]
    return {"X": centre[0], "Y": centre[1], "area": area, "depth": centre[3], "valid": valid}


def _to_plane(cam: Camera, u: np.ndarray, v: np.ndarray) -> tuple:
    """Rays through pixels (u, v) -> (X, Y, valid, depth) on the head plane."""
    Kinv = np.linalg.inv(cam.K)
    R = cam.R
    C = cam.centre
    d = np.stack([u, v, np.ones_like(u)], -1) @ Kinv.T  # camera-frame directions
    d = d @ R  # world-frame directions (R^T d)
    lam = (HEAD_Z - C[2]) / np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    X = C[0] + lam * d[..., 0]
    Y = C[1] + lam * d[..., 1]
    depth = lam * np.linalg.norm(d, axis=-1)
    return X, Y, (lam > 0) & (depth < 600), depth


_EDGE: tuple | None = None


def courtyard_edge(angle: np.ndarray) -> np.ndarray:
    """Distance (m, from the Kaaba's centre) of the courtyard floor's edge at the given angles (rad), minus the margin."""
    global _EDGE
    if _EDGE is None:
        try:
            d = json.loads(MATAF_FLOOR.read_text(encoding="utf-8"))
            e = np.array(sorted(d["edge"]), float)
            _EDGE = (np.radians(e[:, 0]), e[:, 1] - float(d.get("margin_m", 4.0)))
        except (OSError, ValueError, KeyError):
            _EDGE = (np.array([-np.pi, np.pi]), np.array([0.0, 0.0]))  # no edge known: no far ring
    a, r = _EDGE
    return np.interp(angle, a, r, period=2 * np.pi)


def far_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Head-plane points on the far Mataf: >= FAR_FROM m from the walls and inside the courtyard's edge."""
    return (wall_distance(X, Y) >= FAR_FROM) & (np.hypot(X, Y) <= courtyard_edge(np.arctan2(Y, X)))


def wall_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Distance (m) from the Kaaba's walls (0 inside the footprint)."""
    return np.hypot(np.maximum(np.abs(X) - KAABA_L / 2, 0), np.maximum(np.abs(Y) - KAABA_W / 2, 0))


def measure(cam: Camera, density: np.ndarray, exclude: np.ndarray | None = None, correction=None,
            factor: float = 1.0) -> M2Reading:
    """Rings of people per m^2. exclude: uint8 frame-size mask (Kaaba + graphics) of pixels that are not floor;
    correction: g(head px); factor: the camera's factor (load_correction)."""
    geo = cell_floor(cam, density.shape)
    ok = geo["valid"]
    if exclude is not None:
        dh, dw = density.shape
        ex = exclude[: dh * 8, : dw * 8].astype(np.float32)
        ex = ex.reshape(dh, 8, dw, 8).mean((1, 3)) if ex.shape == (dh * 8, dw * 8) else np.zeros((dh, dw))
        ok &= ex < 0.5
    dist = wall_distance(geo["X"], geo["Y"])
    head_px = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
    w_all = np.maximum(correction(head_px), G_FLOOR) * factor if correction is not None else np.ones_like(head_px)
    out = M2Reading()
    for name, (r0, r1) in {**RINGS, "all": ALL}.items():
        sel = ok & (dist >= r0) & (dist < r1)
        a = float(geo["area"][sel].sum())
        n = float(density[sel].sum())
        out.area[name] = round(a, 1)
        out.count[name] = round(n, 1)
        if a >= MIN_AREA_M2:
            out.density[name] = round(n / a, 4)
            hp = head_px[sel]
            out.head_px[name] = round(float(np.median(hp)), 2)
            if correction is not None:
                w = correction(hp)  # expected count ratio vs large heads, per cell
                out.density_corr[name] = round(float((density[sel] / (np.maximum(w, G_FLOOR) * factor)).sum()) / a, 4)
            if name == "all":
                cnt, _ = np.histogram(hp, HEAD_BINS, weights=density[sel])
                ar, _ = np.histogram(hp, HEAD_BINS, weights=geo["area"][sel])
                out.bins = [[round(float(x), 2) for x in cnt], [round(float(x), 1) for x in ar]]
    sector = (np.floor((np.arctan2(geo["Y"], geo["X"]) + np.pi) / (2 * np.pi / SECTORS)).astype(int)) % SECTORS
    far = ok & far_mask(geo["X"], geo["Y"]) & (head_px >= FAR_HEAD_MIN)
    a = float(geo["area"][far].sum())
    if a >= MIN_AREA_M2:
        n = float(density[far].sum())
        fs = {}
        for s in np.unique(sector[far]):
            sel = far & (sector == s)
            sa = float(geo["area"][sel].sum())
            if sa >= SECTOR_MIN_AREA_M2:
                fs[f"f{s}"] = [round(float(density[sel].sum()), 2), round(float((density[sel] / w_all[sel]).sum()), 2),
                               round(sa, 1), round(float(np.median(head_px[sel])), 2)]
        out.far = {"raw": round(n / a, 4), "corr": round(float((density[far] / w_all[far]).sum()) / a, 4),
                   "count": round(n, 1), "area": round(a, 1), "head_px": round(float(np.median(head_px[far])), 2),
                   "sectors": fs}
    out.sectors = {}
    for name, (r0, r1) in RINGS.items():
        in_ring = ok & (dist >= r0) & (dist < r1)
        for s in np.unique(sector[in_ring]):
            sel = in_ring & (sector == s)
            a = float(geo["area"][sel].sum())
            if a >= SECTOR_MIN_AREA_M2:
                out.sectors[f"{name[0]}{s}"] = [round(float(density[sel].sum()), 2),
                                                round(float((density[sel] / w_all[sel]).sum()), 2), round(a, 1),
                                                round(float(np.median(head_px[sel])), 2)]
    return out
