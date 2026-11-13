"""Per-frame fixes for the people-per-m^2 grid (8 x 8 px cells), without retraining (user, 2026-10-30). Live in the shot
service through LiveFixes (config/m2.json live_model_v4 "fixes"), which counts when each fix fires (stats, the minute log).

  near_wall   v4 under-reads people praying in the first metres beside the Kaaba where cam5's heads are ~3 px (no
              rows labels and no keep anchor there in training; held-out 28 Oct Maghrib 14:59:47); v3 reads them.
              Within NEAR_WALL_M of the walls the cell takes max(v4, v3).
  fill_masked floor under the broadcast graphics (banner, captions): estimated from the nearest visible floor at the
              same distance from the walls (1 m bands, nearest bearing first), flagged "estimated" (hatched).
  fill_far    during prayer the rings fill evenly: a 1 m band of the inner ring directly behind the Kaaba (> FAR_MIN_DEG
              from the camera's bearing) that reads under FAR_RATIO of the near side (people hidden by the Kaaba,
              heads too small) takes the near side's reading of the same band, flagged "estimated".
All take / return people per m^2 per cell (NaN = not floor) plus a bool "estimated" grid.
"""
from __future__ import annotations

import numpy as np

NEAR_WALL_M = 4.0
BAND_M = 1.0
MAX_GAP_DEG = 30.0  # fill_masked: no visible floor in the band within this bearing -> not filled
FAR_RATIO = 0.8  # a far-side ring band reading under this share of the near side is estimated from the near side
NEAR_MIN_M2 = 8.0  # m^2 of near-side floor in a ring band needed before it labels the far side
FAR_MIN_DEG = 120.0  # "directly behind the Kaaba": bearing this far from the camera's
FAR_MAX_M = 10.0  # the inner ring only: further out the rows end and walkways are really empty (28 Oct 14:59 roof)


def near_wall(m2_v4: np.ndarray, m2_v3: np.ndarray, wd: np.ndarray) -> np.ndarray:
    """Cell-wise max of the two models within NEAR_WALL_M of the walls; v4 elsewhere."""
    out = m2_v4.copy()
    s = (wd < NEAR_WALL_M) & np.isfinite(m2_v4) & np.isfinite(m2_v3)
    out[s] = np.maximum(m2_v4[s], m2_v3[s])
    return out


def fill_masked(m2: np.ndarray, hidden: np.ndarray, wd: np.ndarray, th: np.ndarray, area: np.ndarray):
    """hidden: floor cells not measured (under the graphics). Each takes the area-weighted reading of the visible cells
    of its 1 m band nearest in bearing (within +-5 deg, widened to MAX_GAP_DEG). Returns (m2, estimated)."""
    out, est = m2.copy(), np.zeros(m2.shape, bool)
    seen = np.isfinite(m2) & ~hidden
    band = np.floor(wd / BAND_M).astype(int)
    for b in np.unique(band[hidden]):
        vis = seen & (band == b)
        if not vis.any():
            continue
        tv, mv, av = th[vis], m2[vis], area[vis]
        for idx in zip(*np.nonzero(hidden & (band == b))):
            d = np.abs(np.angle(np.exp(1j * (tv - th[idx]))))
            for lim in (5.0, 10.0, 20.0, MAX_GAP_DEG):
                k = d <= np.radians(lim)
                if av[k].sum() >= 2.0:
                    out[idx] = float((mv[k] * av[k]).sum() / av[k].sum())
                    est[idx] = True
                    break
    return out, est


def fill_masked_fast(m2: np.ndarray, hidden: np.ndarray, wd: np.ndarray, th: np.ndarray, area: np.ndarray,
                     bin_deg: float = 5.0):
    """fill_masked on a large grid (the AI view's heatmap pixels): per 1 m band, the visible floor's area-weighted
    reading in bin_deg bearing bins; a hidden cell takes its own bin's, else the nearest bin with floor within
    MAX_GAP_DEG. Returns (m2, estimated)."""
    out, est = m2.copy(), np.zeros(m2.shape, bool)
    seen = np.isfinite(m2) & ~hidden & (area > 0)
    if not hidden.any() or not seen.any():
        return out, est
    nb = int(round(360 / bin_deg))
    band = np.floor(np.nan_to_num(wd, nan=-1.0) / BAND_M).astype(int)
    sec = (np.floor((np.degrees(np.nan_to_num(th)) % 360) / bin_deg).astype(int)) % nb
    reach = int(MAX_GAP_DEG // bin_deg)
    for b in np.unique(band[hidden]):
        if b < 0:
            continue
        vis = seen & (band == b)
        if not vis.any():
            continue
        num = np.bincount(sec[vis], weights=(m2[vis] * area[vis]), minlength=nb)
        den = np.bincount(sec[vis], weights=area[vis], minlength=nb)
        val = np.full(nb, np.nan)
        val[den > 0] = num[den > 0] / den[den > 0]
        filled = val.copy()
        for d in range(1, reach + 1):  # the nearest bins with floor, either side
            for sgn in (1, -1):
                cand = np.roll(val, sgn * d)
                filled = np.where(np.isnan(filled), cand, filled)
        tgt = hidden & (band == b)
        v = filled[sec[tgt]]
        ok = np.isfinite(v)
        idx = np.flatnonzero(tgt)[ok]
        out.flat[idx] = v[ok]
        est.flat[idx] = True
    return out, est


def fill_far(m2: np.ndarray, wd: np.ndarray, th: np.ndarray, area: np.ndarray, head: np.ndarray, cam_az: float,
             bands: list | None = None):
    """Prayer only. Per 1 m band of the inner ring (< FAR_MAX_M): when the floor directly behind the Kaaba (bearing
    > FAR_MIN_DEG from the camera's) reads under FAR_RATIO of the near side (< 90 deg),
    its cells take the near side's area-weighted reading (people hidden by the Kaaba or heads too small
    to see). Bands where the far side reads like the near side keep their own cells (v4 reads the far side's rows right
    from ~6 m out: held-out rows behind the Kaaba 1.01). bands: a list that gets (band m, near, far reading, m^2) per
    band filled. Returns (m2, estimated)."""
    out, est = m2.copy(), np.zeros(m2.shape, bool)
    seen = np.isfinite(m2)
    dth = np.abs(np.angle(np.exp(1j * (th - cam_az))))
    far = (dth > np.radians(FAR_MIN_DEG)) & (wd < FAR_MAX_M)
    near_side = dth < np.pi / 2
    band = np.floor(wd / BAND_M).astype(int)
    for b in np.unique(band[seen & far]):
        near, fsel = seen & near_side & (band == b), seen & far & (band == b)
        if area[near].sum() < NEAR_MIN_M2 or area[fsel].sum() < 1:
            continue
        vn = float((m2[near] * area[near]).sum() / area[near].sum())
        vf = float((m2[fsel] * area[fsel]).sum() / area[fsel].sum())
        if vf < FAR_RATIO * vn:
            out[fsel] = vn
            est[fsel] = True
            if bands is not None:
                bands.append((int(b), vn, vf, float(area[fsel].sum())))
    return out, est


STOPPED_SEE = 0.15  # people per m^2 (smoothed): the model "sees people" in a stopped cell; bare floor reads ~0


class LiveFixes:
    """The fixes on the live shot service's density map (edge/shot.py, config/m2.json live_model_v4 "fixes"; user
    2026-10-30): within near_m of the walls max(live model, near_model on a crop around the Kaaba: only that part of
    the frame goes through the second model); during prayer fill_far, and the stopped rows (user 2026-11-01: the
    camera's "stopped" speed sectors from the tawaf speed sensor, speed_fn = LiveSpeed.sectors, where the model sees
    people take at least stopped_rho people per m^2; 1.2 on 31 Oct's saved 1 s prayer pairs read the rows labels at
    0.87 against v4's 0.73 with the fewest overshoots; stopped_rho 0 = off). Works on people per 8 x 8 px cell (the
    density map) and returns it with the "estimated" cells. last: what the fixes did on the last frame; stats: totals
    since the last take_stats() (the shot service logs them every minute, user 2026-10-31)."""

    def __init__(self, near_model=None, near_m: float = NEAR_WALL_M, far_in_prayer: bool = True,
                 stopped_rho: float = 0.0, speed_fn=None):
        self.near_model, self.near_m, self.far_in_prayer = near_model, near_m, far_in_prayer
        self.stopped_rho, self.speed_fn = stopped_rho, speed_fn
        self.last: dict = {}
        self.stats: dict = self._empty()

    @staticmethod
    def _empty() -> dict:
        return {"frames": 0, "near": 0, "near_people": 0.0, "prayer": 0, "far": 0, "far_bands": set(), "far_m2": 0.0,
                "far_near": 0.0, "far_far": 0.0, "stop": 0, "stop_m2": 0.0, "stop_before": 0.0, "stop_after": 0.0}

    def take_stats(self) -> str:
        """The minute's log note: in how many frames each fix changed the reading, and by how much."""
        s, self.stats = self.stats, self._empty()
        if not s["frames"]:
            return ""
        msg = f" | fixes: near wall {s['near']}/{s['frames']} frames (+{s['near_people'] / s['frames']:.0f} people a frame)"
        if s["prayer"]:
            msg += f", far side {s['far']}/{s['prayer']} prayer frames"
            if s["far"]:
                b = sorted(s["far_bands"])
                msg += (f" (bands {b[0]}-{b[-1] + 1} m, {s['far_far'] / s['far_m2']:.2f} -> "
                        f"{s['far_near'] / s['far_m2']:.2f}/m2)")
            if s["stop"]:
                msg += (f", stopped rows {s['stop']}/{s['prayer']} prayer frames ({s['stop_m2'] / s['stop']:.0f} m2 a frame, "
                        f"{s['stop_before'] / s['stop_m2']:.2f} -> {s['stop_after'] / s['stop_m2']:.2f}/m2)")
        return msg

    @staticmethod
    def _bucket(a: int, b: int, n: int, step: int = 16) -> tuple[int, int]:
        size = min(-(-(b - a) // step) * step, n)
        a = max(0, min(a, n - size))
        return a, a + size

    def apply(self, img: np.ndarray, dens: np.ndarray, cam, blocked: np.ndarray, prayer: bool, camera: str | None = None):
        """dens: people per cell; blocked: per cell, not floor for the metric (the Kaaba's cut-out, graphics); camera:
        the view's name (the speed sectors are used only when they are this camera's)."""
        import math

        import cv2

        from edge.m2_metric import cell_floor, wall_distance

        dh, dw = dens.shape
        geo = cell_floor(cam, (dh, dw))
        wd, area = wall_distance(geo["X"], geo["Y"]), geo["area"]
        floor = geo["valid"] & ~blocked
        out = dens.astype(np.float32).copy()
        near = floor & (wd < self.near_m)
        if self.near_model is not None and near.sum() >= 4:
            rows, cols = np.nonzero(near)
            r0, r1 = max(int(rows.min()) - 4, 0), min(int(rows.max()) + 5, dh)
            c0, c1 = max(int(cols.min()) - 4, 0), min(int(cols.max()) + 5, dw)
            # crop sizes rounded up to 16 cells (128 px): a few input shapes, so cuDNN (benchmark mode) tunes once each
            r0, r1 = self._bucket(r0, r1, dh)
            c0, c1 = self._bucket(c0, c1, dw)
            d3 = self.near_model.predict(np.ascontiguousarray(img[r0 * 8:r1 * 8, c0 * 8:c1 * 8])).density.astype(np.float32)
            if d3.shape != (r1 - r0, c1 - c0):  # the model's map of an odd-sized crop: back to its 8 x 8 px cells
                s = float(d3.sum())
                d3 = cv2.resize(d3, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
                d3 *= s / max(float(d3.sum()), 1e-9)
            sub, m = out[r0:r1, c0:c1], near[r0:r1, c0:c1]
            sub[m] = np.maximum(sub[m], d3[m])
        near_add = float(out.sum() - dens.sum())
        est, bands = np.zeros((dh, dw), bool), []
        if prayer and self.far_in_prayer:
            m2 = np.where(floor & (wd >= 1.0), out / np.maximum(area, 1e-6), np.nan)
            th = np.arctan2(geo["Y"], geo["X"])
            m2f, est = fill_far(m2, wd, th, area, None, math.atan2(cam.centre[1], cam.centre[0]), bands)
            out = np.where(est, m2f * area, out).astype(np.float32)
        stop_m2, stop_before, stop_after = 0.0, 0.0, 0.0
        if prayer and self.stopped_rho and self.speed_fn is not None:
            sp = self.speed_fn()  # (camera, pairs, sectors per ring) of the last 2 min, or None
            if sp is not None and sp[0] == camera:
                from edge.tawaf_speed import RINGS, SECTOR_DEG

                secs = sp[2]
                nsec = len(secs["inner"])
                sec = (np.floor((np.degrees(np.arctan2(geo["Y"], geo["X"])) % 360) / SECTOR_DEG).astype(int)) % nsec
                stop = np.zeros((dh, dw), bool)
                for ring, (a0, a1) in RINGS.items():
                    hows = np.array([s["how"] for s in secs[ring]])
                    stop |= floor & (wd >= a0) & (wd < a1) & (hows[sec] == "stopped")
                sm = (cv2.GaussianBlur(out, (0, 0), 1.5)
                      / np.maximum(cv2.GaussianBlur((area * floor).astype(np.float32), (0, 0), 1.5), 1e-6))
                tgt = stop & (sm >= STOPPED_SEE) & ~est
                if tgt.any():
                    stop_m2, stop_before = float(area[tgt].sum()), float(out[tgt].sum())
                    out[tgt] = np.maximum(out[tgt], self.stopped_rho * area[tgt])
                    stop_after = float(out[tgt].sum())
                    est = est | tgt
        self.last = {"near_people": near_add, "prayer": bool(prayer), "far_bands": bands, "stopped_m2": stop_m2}
        st = self.stats
        st["frames"] += 1
        st["near"] += near_add >= 0.5  # v3 raised the reading by at least half a person
        st["near_people"] += near_add
        st["prayer"] += bool(prayer)
        if stop_m2:
            st["stop"] += 1
            st["stop_m2"] += stop_m2
            st["stop_before"] += stop_before
            st["stop_after"] += stop_after
        if bands:
            st["far"] += 1
            for b, vn, vf, a in bands:
                st["far_bands"].add(b)
                st["far_m2"] += a
                st["far_near"] += vn * a
                st["far_far"] += vf * a
        return out, est
