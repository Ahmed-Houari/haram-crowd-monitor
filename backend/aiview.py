""""What the AI sees": a readable version of the collector's live heatmap PNG (read-only, no video pixels).

The collector (edge/heatmap.py density_png) paints density with OpenCV's JET colours over a dark grey
background, scaled per frame to its 99th percentile, plus the zone outline in pure green, in live-frame
coordinates. This module turns that into:
  - the density value per pixel again (JET inverted with a lookup table),
  - a transparent PNG in the dashboard's ramp (teal -> gold -> coral), whose strength follows the camera's
    level (same rule as the 3D twin), so the page uses one colour scale;
  - the reference -> frame alignment: the reading's homography (primary reference px -> frame px, saved with the
    live heatmap since 2026-10-24; exact also for zoomed / partly visible zones), else fitted from the green zone
    outline; it places the architecture outlines (scripts/make_outlines.py) and the zone polygon on the frame;
  - a label point well inside the visible part of the zone for the number.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BG = 25  # background grey of density_png
RAMP = np.array([[47, 143, 134], [79, 179, 168], [216, 180, 95], [255, 111, 82]], np.float32)  # RGB (relative heat)
# people per m^2 -> colour: crowd-safety levels (user decision 2026-10-27; before: a ramp 0 -> 4+, where full prayer
# rows looked quiet). Fruin's walkway level of service (Fruin 1971, Pedestrian Planning and Design; FHWA-RD-98-107
# Table 3: LOS D 0.9-1.4, E 0.5-0.9, F < 0.5 m^2 per person = 15 / 10 / 5 ft^2):
#   quiet < 0.7 (LOS A-C) · busy 0.7-1.1 (D) · crowded 1.1-2.2 (E) · packed > 2.2 (F) people per m^2.
# Full prayer rows are ~1.3 (1.22 m x 0.64 m per person): crowded. The dashboard's one scale (twin floor, AI view,
# map, legends, overlays; dashboard/src/api.ts CROWD_LEVELS): quiet fades from dim teal (empty floor) to teal, the
# other levels are flat colours. Colours checked for colour-blind separation of neighbouring levels (dataviz
# validator on the dark ground: CVD dE >= 13, normal dE >= 19, contrast >= 3:1).
LEVELS = (0.7, 1.1, 2.2)
LEVEL_NAMES = ("quiet", "busy", "crowded", "packed")
LEVEL_RGB = np.array([[47, 174, 156], [230, 192, 79], [242, 106, 61], [196, 31, 107]], np.float32)
EMPTY_RGB = np.array([36, 90, 90], np.float32)  # quiet at 0 people per m^2
LEVEL_BLEND = 0.04  # people per m^2 either side of a boundary: colours blend there (no hard aliasing)


def ramp_m2(dens: np.ndarray) -> np.ndarray:
    """(..., 3) RGB float for people per m^2 on the crowd-safety levels."""
    t = np.clip(np.nan_to_num(np.asarray(dens, np.float32)), 0, None)
    e = LEVEL_BLEND
    xs = [0.0, LEVELS[0] - e, LEVELS[0] + e, LEVELS[1] - e, LEVELS[1] + e, LEVELS[2] - e, LEVELS[2] + e]
    q, b, c, p = LEVEL_RGB
    cs = [EMPTY_RGB, q, b, b, c, c, p]
    return np.stack([np.interp(t, xs, [col[i] for col in cs]) for i in range(3)], -1).astype(np.float32)


def level_of(v: float) -> str:
    """The crowd-safety level of a people-per-m^2 value."""
    return LEVEL_NAMES[sum(v >= x for x in LEVELS)]


# a crowd level (Fruin) describes an area, not a few people (user 2026-10-27: a small group on empty floor showed as a
# busy spot; "e.g. 10 m^2"): every pixel shows people / floor area over at least this much floor around it, the same
# near and far. 10 m^2 was not enough: groups of 7-8 people on empty floor (27 Oct 15:49, roof) read 0.71-0.75 over
# 10 m^2 (busy) and 0.44-0.53 over 25 m^2 (quiet), while a real tawaf crowd stays at 1.19
MIN_AREA_M2 = 25.0
# measured floor is never drawn empty (user 2026-10-27): an empty part shows the scale's "0" colour at this opacity;
# floor inside the courtyard's edge that is not measured (broadcast graphics) gets grey hatching (hatch)
FLOOR_MIN_ALPHA = 80
HATCH_BGRA = (150, 150, 150, 170)


def heat_m2(dens: np.ndarray, out_size: tuple[int, int], px_per_m: np.ndarray | None = None) -> np.ndarray:
    """BGRA heat for a per-pixel people-per-m^2 image (NaN = not floor): averaged over MIN_AREA_M2 of floor, speckles
    removed, absolute colours.

    The average is area-weighted: people / floor area under a Gaussian whose floor footprint (2 pi sigma^2 in m^2) is
    MIN_AREA_M2, with px_per_m the pixels per metre of floor (the area of a pixel is 1 / px_per_m^2). It is a
    normalised convolution (heat never spreads onto unseen pixels): near the camera it blurs over many pixels, far away
    over few, and a small group on empty floor reads as the density of its area, not as its own. Without px_per_m: a
    1 px blur. Leftover specks smaller than ~3 x 3 heatmap px are dropped. Drawn at out_size."""
    seen = np.isfinite(dens)
    w = seen.astype(np.float32)
    d = np.nan_to_num(dens).astype(np.float32)
    if px_per_m is None:
        num = cv2.GaussianBlur(d * w, (0, 0), 1.0)
        den = cv2.GaussianBlur(w, (0, 0), 1.0)
        sm = np.where(den > 0.2, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    else:
        ppm = np.nan_to_num(px_per_m, nan=1.0).astype(np.float32)
        area = np.where(seen, 1.0 / np.maximum(ppm, 1e-3) ** 2, 0.0).astype(np.float32)
        people = d * area
        sigmas = [2.0 ** (k / 2) for k in range(11)]  # 1 .. 32 px in steps of sqrt(2)
        levels = [np.where(seen, d, 0.0).astype(np.float32)]  # sigma 0: the pixel itself
        for sg in sigmas:
            num = cv2.GaussianBlur(people, (0, 0), sg)
            den = cv2.GaussianBlur(area, (0, 0), sg)
            sup = cv2.GaussianBlur(w, (0, 0), sg)
            levels.append(np.where((sup > 0.2) & (den > 0), num / np.maximum(den, 1e-12), 0.0).astype(np.float32))
        # per pixel: the sigma (px) whose floor footprint is MIN_AREA_M2, between the levels on a log scale
        want = np.sqrt(MIN_AREA_M2 / (2 * np.pi)) * ppm
        pos = np.clip(2 * np.log2(np.maximum(want, 1e-6)) + 1.0, 0.0, len(sigmas))  # sigma <= 0.7 -> 0, 1 -> 1, 32 -> 11
        i = np.clip(pos.astype(int), 0, len(sigmas) - 1)
        f = pos - i
        stack = np.stack(levels)
        rows, cols = np.indices(d.shape)
        sm = stack[i, rows, cols] * (1 - f) + stack[i + 1, rows, cols] * f
    busy = (sm > 0.2).astype(np.uint8)
    keep = cv2.morphologyEx(busy, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # soft edges around kept spots
    sm = np.where(keep > 0, sm, np.minimum(sm, 0.2))
    big = cv2.resize(sm, out_size, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), out_size[0] / 320 * 1.2)
    vis = cv2.resize(seen.astype(np.float32), out_size, interpolation=cv2.INTER_LINEAR)
    rgb = ramp_m2(big)
    # quiet ground (a few people) shows as a faint dim teal: 25 % opacity at ~0.2 people per m^2 (alignment check
    # 2026-10-26: with the old 0.36, sparse zoomed-in crowds were not drawn at all)
    alpha = np.clip((big - 0.06) / 0.5, 0, 1) ** 1.0 * 225 * np.clip(vis * 1.4 - 0.2, 0, 1)
    alpha = np.maximum(alpha, FLOOR_MIN_ALPHA * np.clip(vis * 1.4 - 0.2, 0, 1))  # measured floor: never empty
    return np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], alpha]).astype(np.uint8)


def hatch(img: np.ndarray, mask: np.ndarray, bgra=HATCH_BGRA, step: int | None = None) -> np.ndarray:
    """Grey diagonal hatching on img (BGR or BGRA, drawn in place) where mask (any size, resized to img) is set:
    floor that is not measured."""
    h, w = img.shape[:2]
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    step = step or max(6, w // 110)
    yy, xx = np.indices((h, w))
    lines = ((xx + yy) % step) < max(1, step // 4)
    sel = m & lines
    a = bgra[3] / 255.0
    if img.shape[2] == 4:
        img[sel] = bgra
    else:
        img[sel] = (img[sel] * (1 - a) + np.array(bgra[:3]) * a).astype(np.uint8)
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edge] = bgra if img.shape[2] == 4 else bgra[:3]
    return img


def hatch_light(img: np.ndarray, mask: np.ndarray, step: int | None = None) -> np.ndarray:
    """A light diagonal hatch over the colour (BGRA, in place) where mask (any size) is set: estimated floor, the same
    hatch as the twin's floor behind the Kaaba (backend/floormap.py HATCH_RGBA); the estimate's colour stays readable."""
    h, w = img.shape[:2]
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    step = step or max(6, w // 110)
    yy, xx = np.indices((h, w))
    sel = m & (((xx + yy) % step) < max(1, step // 4))
    k = 90 / 255
    img[sel, :3] = np.clip(img[sel, :3] * (1 - k) + np.array([236, 236, 232]) * k, 0, 255).astype(img.dtype)
    img[sel, 3] = np.maximum(img[sel, 3], 150)
    return img


def backdrop(size: tuple[int, int], floor: np.ndarray | None, kaaba: np.ndarray | None) -> np.ndarray:
    """BGR background: dark ground, the Mataf floor a little lighter, the Kaaba a darker block (no video pixels)."""
    w, h = size
    bg = np.empty((h, w, 3), np.uint8)
    bg[:] = (22, 18, 12)
    if floor is not None:
        f = cv2.resize(floor.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        f = cv2.GaussianBlur(f.astype(np.float32), (0, 0), w / 320 * 1.5)[..., None]
        bg = (bg * (1 - f) + np.array([44, 46, 44], np.float32) * f).astype(np.uint8)
    if kaaba is not None:
        cv2.fillPoly(bg, [np.int32(np.round(kaaba))], (9, 9, 8), cv2.LINE_AA)
    return bg


def _jet_inverse() -> np.ndarray:
    """32x32x32 lookup: quantised BGR colour -> density value 0..255, as density_png paints it."""
    v = np.arange(256)
    jet = cv2.applyColorMap(v.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(256, 3).astype(np.float32)
    a = np.minimum(1.0, 1.5 * v / 255.0)[:, None]
    painted = BG * (1 - a) + jet * a
    centres = (np.indices((32, 32, 32)).reshape(3, -1).T * 8 + 4).astype(np.float32)
    best = np.full(len(centres), np.inf, np.float32)
    out = np.zeros(len(centres), np.uint8)
    for i in range(256):
        d = ((centres - painted[i]) ** 2).sum(1)
        better = d < best
        best[better], out[better] = d[better], i
    return out.reshape(32, 32, 32)


_INV = _jet_inverse()


def density_values(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(value 0..255 per pixel, mask of the green zone-outline pixels)."""
    b, g, r = (img_bgr[..., i].astype(np.int16) for i in range(3))
    line = (g > 170) & (r < 90) & (b < 90)
    q = img_bgr // 8
    v = _INV[q[..., 0], q[..., 1], q[..., 2]]
    v = np.where(line, cv2.medianBlur(np.where(line, 0, v).astype(np.uint8), 5), v).astype(np.uint8)
    return v, line


def fit_ref_to_frame(line: np.ndarray, zones_ref: list[np.ndarray], ref_size: tuple[int, int]) -> tuple[list[float], str]:
    """Axis-aligned scale + shift mapping reference px -> heatmap px, from the zone outline's bounding box.

    The view matcher accepts only small zoom changes (+-5..10 %), so scale + shift is close to the real
    homography. Falls back to plain resizing when the outline is missing or implausible.
    """
    h, w = line.shape
    base = w / ref_size[0]
    ys, xs = np.nonzero(line)
    allp = np.vstack(zones_ref)
    (X0, Y0), (X1, Y1) = allp.min(0), allp.max(0)
    if len(xs) > 30 and X1 > X0 and Y1 > Y0:
        sx, sy = (xs.max() - xs.min()) / (X1 - X0), (ys.max() - ys.min()) / (Y1 - Y0)
        if 0.75 < sx / base < 1.33 and 0.75 < sy / base < 1.33:
            return [float(sx), float(sy), float(xs.min() - sx * X0), float(ys.min() - sy * Y0)], "zone outline"
    return [base, base, 0.0, 0.0], "resize"


DENSITY_MAX = 4.0  # people per m^2 at the top of the ramp, as in the 3D twin's legend (dashboard Twin.tsx)
UPSCALE = 3  # the collector's heatmap is small (320 x 180): smoothed and drawn 3x larger


def colourise(v: np.ndarray, k: float, zone: np.ndarray | None = None, people_m2: float | None = None) -> np.ndarray:
    """BGRA heat in the dashboard ramp, smooth, transparent where there is (almost) nobody.

    With the camera's own people per m^2, the colour follows the twin's scale: the zone's average heat takes the colour
    of that reading and denser spots go warmer (approximate: the heatmap is per image pixel, not per m^2). Without it,
    the strength follows the camera's level (k) as before.
    """
    h, w = v.shape
    vf = cv2.resize(v.astype(np.float32), (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    vf = cv2.GaussianBlur(np.clip(vf, 0, 255), (0, 0), sigmaX=2.2 * UPSCALE)
    if people_m2 is not None and zone is not None and zone.any():
        z = cv2.resize(zone, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_NEAREST) > 0
        mean = float(vf[z & (vf > 4)].mean()) if (z & (vf > 4)).any() else 0.0
        s = np.clip(vf / max(mean, 1.0) * (people_m2 / DENSITY_MAX), 0, 1) if mean else vf / 255.0 * k
    else:
        s = (vf / 255.0) * k
    v = vf
    x = np.clip(s * 3, 0, 2.999)
    i = x.astype(np.int32)
    f = (x - i)[..., None]
    rgb = RAMP[i] + (RAMP[i + 1] - RAMP[i]) * f
    vn = v.astype(np.float32) / 255.0
    alpha = np.minimum(215, 60 + s * 170) * np.clip((vn - 0.05) / 0.2, 0, 1) ** 1.5  # soft fade in, no hard edge
    return np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], alpha]).astype(np.uint8)


def label_point(zones_px: list[np.ndarray], shape: tuple[int, int]) -> list[float]:
    """The point deepest inside the zone(s): room for the count label."""
    m = np.zeros(shape, np.uint8)
    for p in zones_px:
        cv2.fillPoly(m, [np.int32(np.round(p))], 255)
    if not m.any():
        return [shape[1] / 2, shape[0] / 2]
    m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # the frame edge bounds a partly visible zone
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return [float(x - 1), float(y - 1)]


def outline_path(outline_file: Path, to_px) -> str | None:
    """SVG path ('M x y L ...') of the architecture outlines, in heatmap px (to_px maps reference px -> heatmap px)."""
    if not outline_file.exists():
        return None
    lines = json.loads(outline_file.read_text())["lines"]
    return " ".join("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in to_px(np.asarray(line, np.float32)))
                    for line in lines if len(line) > 1)


def render(img_bgr: np.ndarray, view: dict, ref_size: tuple[int, int], outline_file: Path, k: float,
           H_primary: list | None = None, frame_size: list | None = None, people_m2: float | None = None,
           m2: dict | None = None) -> tuple[dict, bytes, bytes | None]:
    """(JSON for the dashboard, transparent heat PNG) for one live heatmap."""
    h, w = img_bgr.shape[:2]
    v, line = density_values(img_bgr)
    if H_primary is not None and frame_size:
        Hh = np.diag([w / frame_size[0], h / frame_size[1], 1.0]) @ np.asarray(H_primary, np.float64)
        to_px = lambda p: cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), Hh).reshape(-1, 2)  # noqa: E731
        fit, how = [round(float(x), 6) for x in Hh.ravel()], "homography"
    else:
        zones_ref = [np.asarray(p, np.float32) for p in view["zones"].values()]
        fit, how = fit_ref_to_frame(line, zones_ref, ref_size)
        sx, sy, tx, ty = fit
        to_px = lambda p: np.asarray(p, np.float32).reshape(-1, 2) * [sx, sy] + [tx, ty]  # noqa: E731
    zones_px = {name: to_px(p) for name, p in view["zones"].items()}
    zone = np.zeros((h, w), np.uint8)
    for p in zones_px.values():
        cv2.fillPoly(zone, [np.int32(np.round(p))], 255)
    out = (w * UPSCALE, h * UPSCALE)
    bg = None
    if m2 is not None:  # a calibrated camera: people per m^2 per pixel on the floor (backend/floormap.py)
        heat = heat_m2(m2["dens"], out, m2.get("px_per_m"))
        if m2.get("est") is not None and m2["est"].any():  # estimated (graphics, fixes): its colour under a light hatch
            hatch_light(heat, m2["est"])
        if m2.get("hidden") is not None and m2["hidden"].any():  # not measured: grey hatching, never empty
            hatch(heat, m2["hidden"])
        if m2.get("line25") is not None and len(m2["line25"]) > 1:  # thin line 25 m from the walls: outer | far Mataf
            pts = np.int32(np.round(m2["line25"] * UPSCALE))
            cv2.polylines(heat, [pts], False, (236, 231, 220, 150), max(1, UPSCALE // 2), cv2.LINE_AA)
        ok, png = cv2.imencode(".png", heat)
        kaaba = m2["kaaba"] * UPSCALE if m2.get("kaaba") is not None else None
        ok, bgpng = cv2.imencode(".png", backdrop(out, m2["floor"], kaaba))
        bg = bgpng.tobytes()
    else:
        ok, png = cv2.imencode(".png", colourise(v, k, zone, people_m2))
    info = {"size": [w, h], "fit": {"scale_shift" if how != "homography" else "h": [round(f, 4) for f in fit],
                                    "from": how},
            "zones": {n: p.round(1).tolist() for n, p in zones_px.items()},
            "label": [round(c, 1) for c in label_point(list(zones_px.values()), (h, w))],
            "outline": outline_path(outline_file, to_px), "k": round(k, 3), "absolute": m2 is not None}
    return info, png.tobytes(), bg
