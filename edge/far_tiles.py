"""DM-Count on the far band at a larger scale (user request 2026-10-27, a test for the fine-tuned model: prayer rows on
the far side of the Kaaba, heads of ~3 px from the roof cameras, still read low). Not used by the live services.

The part of the frame whose floor has heads smaller than HEAD_MAX px is cut into tiles (TILE px, OVERLAP px), each
tile is upscaled SCALE x (bicubic) and read by the model; its density map is summed over SCALE x SCALE blocks back to
the frame's 8 x 8 px cells (the count is kept), the tiles are blended over their overlap, and the result replaces the
full frame's density there, blended over one px of head size (HEAD_MAX - 1 .. HEAD_MAX)."""
from __future__ import annotations

import numpy as np
import cv2

HEAD_MAX = 5.0  # px: the far band
SCALE = 2
TILE = 768  # frame px per tile side (upscaled: 1536, under the model's 2048 px cap)
OVERLAP = 96  # frame px
CELL = 8
MIN_CELLS = 20


def _ramp(n: int, lo: bool, hi: bool, k: int) -> np.ndarray:
    """Weights along one tile axis: 1 inside, a linear ramp over k cells at a side that overlaps another tile."""
    w = np.ones(n, np.float32)
    k = max(1, min(k, n // 2))
    r = (np.arange(k, dtype=np.float32) + 0.5) / k
    if lo:
        w[:k] = np.minimum(w[:k], r)
    if hi:
        w[-k:] = np.minimum(w[-k:], r[::-1])
    return w


def far_density(model, img: np.ndarray, head: np.ndarray, floor: np.ndarray, base: np.ndarray | None = None,
                scale: int = SCALE, head_max: float = HEAD_MAX) -> tuple[np.ndarray, dict]:
    """(density per 8 x 8 px cell with the far band read at scale x, info). head / floor: per cell (as the density)."""
    base = model.predict(img).density if base is None else base
    dh, dw = base.shape
    far = floor & (head < head_max)
    if far.sum() < MIN_CELLS:
        return base, {"far_cells": int(far.sum()), "tiles": 0}
    ys, xs = np.nonzero(far)
    c0, c1 = max(0, ys.min() - 2), min(dh, ys.max() + 3)  # cell rows / columns of the band (+ a margin)
    d0, d1 = max(0, xs.min() - 2), min(dw, xs.max() + 3)
    tc, oc = TILE // CELL, OVERLAP // CELL  # in cells
    acc = np.zeros((dh, dw), np.float32)
    wsum = np.zeros((dh, dw), np.float32)
    n = 0
    rows = list(range(c0, max(c1 - oc, c0 + 1), tc - oc)) or [c0]
    cols = list(range(d0, max(d1 - oc, d0 + 1), tc - oc)) or [d0]
    for a in rows:
        for b in cols:
            a1, b1 = min(a + tc, c1), min(b + tc, d1)
            if a1 - a < 4 or b1 - b < 4:
                continue
            crop = img[a * CELL:a1 * CELL, b * CELL:b1 * CELL]
            up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
            d = model.predict(up).density
            ch, cw = a1 - a, b1 - b
            d = d[:ch * scale, :cw * scale]
            d = np.pad(d, ((0, ch * scale - d.shape[0]), (0, cw * scale - d.shape[1])))
            d = d.reshape(ch, scale, cw, scale).sum((1, 3))
            w = np.outer(_ramp(ch, a > c0, a1 < c1, oc), _ramp(cw, b > d0, b1 < d1, oc))
            acc[a:a1, b:b1] += d * w
            wsum[a:a1, b:b1] += w
            n += 1
    tiled = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), base)
    t = np.clip(head_max - head, 0.0, 1.0) * (wsum > 0)
    out = base * (1 - t) + tiled * t
    return out.astype(np.float32), {"far_cells": int(far.sum()), "tiles": n, "band_cells": int((t > 0).sum())}
