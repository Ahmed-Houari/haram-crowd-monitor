"""People per m^2 from prayer rows (2026-10-26). During prayer the Mataf fills with rows around the Kaaba; seen with
small heads (the roof camera, 3 px) the counting model reads them as empty floor (Maghrib 2026-10-26 18:06: 0.003
people/m^2 where the rows give 1.4). Rows are easy to measure instead: on a frame with a known camera the image is
sampled along lines leading away from the Kaaba (on a plane at ROW_Z), and the rows show as a regular brightness
pattern: its spacing (the row pitch) tells that the frame shows rows, the rows counted along the lines give rows per
metre of floor. People stand about ALONG_M apart within a row (measured on 2026-10-25/26 frames where heads are 10 px:
0.56-0.68 m), so people per m^2 = rows per metre / ALONG_M; floor without rows counts as empty. Counting only:
brightness profiles, nothing about any person.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from edge.kaaba_geom import Camera
from edge.m2_metric import wall_distance

ROW_Z = 1.2  # m: between standing (1.55) and bowing heads
ALONG_M = 0.6  # m between people within a row
PITCH_RANGE = (0.85, 1.6)  # m between rows (measured 1.1-1.34)
STEP = 0.02  # m along a line
SPAN = (2.0, 24.0)  # m from the Kaaba's walls
BEARINGS = np.arange(0, 360, 7.5)
MIN_PX_PER_ROW = 6.0  # a row pitch must span at least this many image px to be seen
STRENGTH_MIN = 0.35  # autocorrelation at the row pitch: prayer rows 0.4-0.8, crowds without rows < 0.25
STRENGTH_PRAYER = 0.2  # within a prayer window (adhan to +PRAYER_WINDOW_MIN): the low and wide cameras see rows at a
#                        shallow angle, weaker (Isha 2026-10-26: 0.2-0.4); outside prayer frames reach 0.10 at most
PRAYER_WINDOW_MIN = 40


@dataclass
class RowsReading:
    strength: float  # median autocorrelation at the pitch over the usable lines
    pitch: float  # m, median row pitch
    rows_per_m: float  # rows counted per metre of sampled line
    density: float | None  # people per m^2 over the sampled floor (None: no rows)
    lines: int  # usable lines


def _ray_points(d: np.ndarray, bearing: float) -> np.ndarray:
    """Points at wall distances d along the ray from the Kaaba's centre at this bearing (degrees)."""
    a = np.radians(bearing)
    r = np.linspace(0, 90, 9001)
    w = wall_distance(r * np.cos(a), r * np.sin(a))
    rr = np.interp(d, w, r)
    return np.column_stack([rr * np.cos(a), rr * np.sin(a)])


def _autocorr(p: np.ndarray) -> np.ndarray:
    p = (p - p.mean()) / (p.std() + 1e-9)
    return np.correlate(p, p, "full")[len(p) - 1:] / len(p)


def measure(gray: np.ndarray, cam: Camera, blocked: np.ndarray | None = None,
            min_strength: float = STRENGTH_MIN) -> RowsReading:
    """gray: the frame (uint8 or float); blocked: frame-size mask of pixels that are not floor (Kaaba, graphics);
    min_strength: the row pattern's strength needed for a reading (lower within a prayer window: STRENGTH_PRAYER)."""
    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    H = cam.plane_to_image(ROW_Z)
    d = np.arange(SPAN[0], SPAN[1], STEP)
    i0, i1 = int(PITCH_RANGE[0] / STEP), int(PITCH_RANGE[1] / STEP)
    strengths, pitches, rows, length = [], [], 0, 0.0
    for b in BEARINGS:
        xy = _ray_points(d, b)
        t = np.array([-np.sin(np.radians(b)), np.cos(np.radians(b))])
        prof, uv0 = [], None
        for o in (-0.2, 0.0, 0.2):  # a narrow band across the line
            q = np.column_stack([xy + o * t, np.ones(len(xy))]) @ H.T
            u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
            if o == 0.0:
                uv0 = (u, v, q[:, 2])
            prof.append(cv2.remap(g, u.astype(np.float32).reshape(1, -1), v.astype(np.float32).reshape(1, -1),
                                  cv2.INTER_LINEAR).ravel())
        u, v, z = uv0
        ok = (z > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if blocked is not None:
            iu, iv = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
            ok &= blocked[iv, iu] == 0
        # longest run of usable samples
        runs = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
        if len(runs) < 2:
            continue
        k = int(np.argmax(runs[1::2] - runs[::2]))
        a, e = runs[2 * k], runs[2 * k + 1]
        if (e - a) * STEP < 3 * PITCH_RANGE[1]:
            continue
        px_per_m = float(np.hypot(u[e - 1] - u[a], v[e - 1] - v[a]) / ((e - 1 - a) * STEP))
        if px_per_m * PITCH_RANGE[0] < MIN_PX_PER_ROW:
            continue
        p = np.mean(prof, 0)[a:e].astype(np.float64)
        p = p - cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 1.5 / STEP).ravel()  # remove slow changes
        p = cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 0.06 / STEP).ravel()  # and pixel noise
        ac = _autocorr(p)
        if len(ac) <= i1:
            continue
        j = i0 + int(np.argmax(ac[i0:i1]))
        strengths.append(float(ac[j]))
        pitches.append(j * STEP)
        # the rows along this line: bright crests at least 0.7 pitch apart (gaps in the rows count as no rows)
        pk, _ = find_peaks(p, distance=max(int(0.7 * j), 1), prominence=0.5 * p.std())
        rows += len(pk)
        length += len(p) * STEP
    if not strengths:
        return RowsReading(0.0, float("nan"), 0.0, None, 0)
    st, pitch = float(np.median(strengths)), float(np.median(pitches))
    per_m = rows / max(length, 1e-6)
    dens = per_m / ALONG_M if st >= min_strength else None
    return RowsReading(round(st, 3), round(pitch, 3), round(per_m, 3), round(dens, 3) if dens is not None else None,
                       len(strengths))
