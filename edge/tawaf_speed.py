"""Tawaf speed sensor (user plan 2026-10-30; offline, not wired into the live services yet).

Tawaf density from walking speed, like traffic speed: crowd motion between two frames 1 s apart -> the counter-clockwise
(tawaf) speed on the floor per ring and 15 deg sector -> people per m^2 from a speed-density curve. Only where the crowd
walks the tawaf: a stopped or milling crowd (prayer rows, queues at the wall) gets no speed reading and keeps the count
model's. Motion only: a dense optical-flow field of the image; no person is detected or followed.

Per frame pair (a, b), same camera, a's pose:
  1. camera motion: SIFT homography on the static part of the frame (not the Mataf floor, not the broadcast graphics),
     b warped onto a; the pair is dropped when it fits badly (< MIN_INLIERS) or the static part still moves > MAX_RESID_PX;
  2. DIS optical flow a -> b, mean per 8 x 8 px cell, both ends on the head plane with a's camera -> m/s;
  3. per cell the tangential (counter-clockwise, the tawaf direction) and radial speed; cells weighted by image texture
     (crowd), so empty marble does not read as stopped people and the far side counts (flow needs texture, not heads).
Per ring (inner 1-10 m, outer 10-25 m from the walls) and SECTOR_DEG sector, over the pairs of a window:
  walking  the sector's mean motion (texture-weighted, median over pairs) >= V_WALK, within DIR_MAX_DEG of
           counter-clockwise, most of the moving texture counter-clockwise -> its speed gives the density from the curve
           (the mean motion's length, not only its tangential part: a pose a little off tilts the circle); slower than V_MIN_CURVE reads as V_MIN_CURVE (flagged "slow": at or past the curve's dense end);
           faster than V_MAX_CURVE -> flagged "fast": a light crowd, the curve's value towards free walking (extrapolated;
           leaving these sectors out would make the ring read only its slow, dense part);
  stopped  -> no speed density (count model);
  hidden   no measured floor (behind the Kaaba, graphics) -> the same ring's nearest walking sectors on both sides,
           weighted by nearness, flagged "estimated".

Curve (Weidmann form): v = V0 (1 - exp(-GAMMA (1/rho - 1/RHO_MAX))), V0 = 1.08 m/s (tawaf free speed, Dridi 2015
Fig. 16), GAMMA and RHO_MAX fitted to the user's two points (2026-10-30): Taraweeh tawaf 0.70 m/s ~ 1.15 people/m^2
(v3) and HAJJv2 Testing_12 / Training_12 0.24 m/s ~ 3.5 people/m^2 (human counts, scale from head size; 3-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import logging

from edge import kaaba_geom as kg
from edge.m2_metric import _to_plane, cell_floor, courtyard_edge, wall_distance

log = logging.getLogger(__name__)

V0, GAMMA, RHO_MAX = 1.08, 1.359, 9.93
V_WALK = 0.10  # m/s: slower = stopped
V_MIN_CURVE, V_MAX_CURVE = 0.20, 0.85  # m/s: the curve's measured range (two points 0.24 and 0.70, a little either side)
CCW_SHARE = 0.6  # share of the moving texture going counter-clockwise for "walking"
DIR_MAX_DEG = 60.0  # the sector's mean motion within this of counter-clockwise for "walking" (recording poses tilt it ~30 deg)
RINGS = {"inner": (1.0, 10.0), "outer": (10.0, 25.0)}
SECTOR_DEG = 15.0
MIN_INLIERS = 15  # static-part matches for the camera motion (live night views give ~100 keypoints there; the
#                  residual check MAX_RESID_PX guards the fit)
MAX_RESID_PX = 1.0
TEXTURE_MIN = 6.0  # mean |gradient| (0-255 grey) per cell: below it the cell is empty floor
MIN_CELLS = 12  # textured floor cells for a sector reading in one pair


def rho_from_speed(v: float) -> float:
    v = float(np.clip(v, 1e-3, V0 - 1e-3))
    return float(1.0 / (1.0 / RHO_MAX - np.log(1.0 - v / V0) / GAMMA))


@dataclass
class PairSpeed:
    vt: np.ndarray  # per cell tangential (counter-clockwise) speed, m/s
    vr: np.ndarray  # per cell radial speed, m/s (+ = away from the Kaaba)
    w: np.ndarray  # per cell weight (texture; 0 = not used)
    wd: np.ndarray  # per cell distance from the walls, m
    th: np.ndarray  # per cell bearing from the Kaaba's centre, rad
    resid_px: float
    inliers: int
    flow: np.ndarray = field(repr=False, default=None)  # per cell image flow (px), for drawing


class TawafSpeed:
    def __init__(self):
        self.sift = cv2.SIFT_create(4000, contrastThreshold=0.02)  # more points on dim architecture
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _static(self, cam: kg.Camera, shape, graphics: np.ndarray) -> np.ndarray:
        h, w = shape
        geo = cell_floor(cam, (h // 8, w // 8))
        floor = geo["valid"] & (np.hypot(geo["X"], geo["Y"]) <= courtyard_edge(np.arctan2(geo["Y"], geo["X"])) + 15)
        m = np.zeros((h, w), np.uint8)
        m[: h // 8 * 8, : w // 8 * 8] = (~floor).astype(np.uint8).repeat(8, 0).repeat(8, 1)
        m[graphics > 0] = 0
        return m

    def _camera_motion(self, ga: np.ndarray, gb: np.ndarray, sm: np.ndarray):
        """(homography b -> a from SIFT on the static part, inliers), or (None, 0) when it cannot be fitted."""
        ka, da = self.sift.detectAndCompute(ga, sm)
        kb, db = self.sift.detectAndCompute(gb, sm)
        if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
            return None, 0
        good = [m for m, n in self.bf.knnMatch(db, da, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        H, inl = cv2.findHomography(np.float32([kb[m.queryIdx].pt for m in good]), np.float32([ka[m.trainIdx].pt for m in good]),
                                    cv2.RANSAC, 2.0)
        if H is None or int(inl.sum()) < MIN_INLIERS:
            return None, 0
        return H, int(inl.sum())

    def pair(self, img_a: np.ndarray, img_b: np.ndarray, cam: kg.Camera, dt: float, graphics: np.ndarray,
             blocked: np.ndarray | None = None) -> PairSpeed | None:
        """Speeds between a and b (dt s apart, a's pose). graphics / blocked: frame-size masks (broadcast graphics; the
        Kaaba's cut-out) whose cells are not measured. None when the camera motion cannot be removed."""
        ga, gb = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        h, w = ga.shape
        sm = self._static(cam, (h, w), graphics)
        H, n_inl = self._camera_motion(ga, gb, sm)
        # no fit (few static keypoints, e.g. dim architecture at night): try "the camera did not move"; the residual
        # check below decides either way
        gbw = cv2.warpPerspective(gb, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if H is not None else gb
        flow = self.dis.calc(ga, gbw, None)
        resid = float(np.median(np.linalg.norm(flow[sm > 0], axis=1))) if (sm > 0).any() else np.inf
        if resid > MAX_RESID_PX:
            return None
        dh, dw = h // 8, w // 8
        fc = flow[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8, 2).mean((1, 3))
        gx, gy = cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1)
        tex = (np.abs(gx) + np.abs(gy))[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) / 8
        jj, ii = np.meshgrid(np.arange(dw), np.arange(dh))
        u0, v0 = (jj + 0.5) * 8, (ii + 0.5) * 8
        X0, Y0, ok0, _ = _to_plane(cam, u0, v0)
        X1, Y1, ok1, _ = _to_plane(cam, u0 + fc[..., 0], v0 + fc[..., 1])
        vx, vy = (X1 - X0) / dt, (Y1 - Y0) / dt
        r = np.maximum(np.hypot(X0, Y0), 1e-6)
        vt, vr = (-Y0 * vx + X0 * vy) / r, (X0 * vx + Y0 * vy) / r
        cell = lambda m: m[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5  # noqa: E731
        bad = cell(graphics > 0) | (cell(blocked > 0) if blocked is not None else False)
        inside = r <= courtyard_edge(np.arctan2(Y0, X0))
        ok = ok0 & ok1 & inside & ~bad & np.isfinite(vt) & (np.abs(vt) < 4) & (tex >= TEXTURE_MIN)
        return PairSpeed(vt=np.where(ok, vt, np.nan), vr=np.where(ok, vr, np.nan), w=np.where(ok, tex, 0.0),
                         wd=wall_distance(X0, Y0), th=np.arctan2(Y0, X0), resid_px=resid, inliers=n_inl, flow=fc)

    @staticmethod
    def sectors(pairs: list[PairSpeed]) -> dict:
        """Per ring: a list of sectors {deg, vt, vr, rho, how}; how = walking | slow | fast | stopped | estimated | none.
        Each pair gives the sector's texture-weighted speeds; the window's value is the median over its pairs."""
        nsec = int(round(360 / SECTOR_DEG))
        out = {}
        for ring, (a0, a1) in RINGS.items():
            per = [[] for _ in range(nsec)]
            for p in pairs:
                inring = (p.wd >= a0) & (p.wd < a1) & (p.w > 0)
                sec = (np.floor((np.degrees(p.th) % 360) / SECTOR_DEG).astype(int)) % nsec
                for s in np.unique(sec[inring]):
                    m = inring & (sec == s)
                    if m.sum() < MIN_CELLS:
                        continue
                    w = p.w[m]
                    vt, vr = p.vt[m], p.vr[m]
                    mov = np.hypot(vt, vr) > V_WALK
                    ccw = float(w[mov & (vt > 0)].sum() / max(w[mov].sum(), 1e-9)) if mov.any() else 0.0
                    per[s].append((float((vt * w).sum() / w.sum()), float((vr * w).sum() / w.sum()), ccw))
            secs = []
            for s in range(nsec):
                d = {"deg": s * SECTOR_DEG + SECTOR_DEG / 2, "vt": None, "vr": None, "rho": None, "how": "none", "pairs": len(per[s])}
                if per[s]:
                    a = np.array(per[s])
                    vt, vr, ccw = float(np.median(a[:, 0])), float(np.median(a[:, 1])), float(np.median(a[:, 2]))
                    sp = float(np.hypot(vt, vr))  # the mean motion's speed: a small pose error tilts it off the tangent
                    d.update(vt=round(vt, 3), vr=round(vr, 3), speed=round(sp, 3))
                    if sp < V_WALK or ccw < CCW_SHARE or abs(np.degrees(np.arctan2(vr, vt))) > DIR_MAX_DEG:
                        d["how"] = "stopped"
                    elif sp > V_MAX_CURVE:
                        d["how"] = "fast"  # light crowd: past the curve's measured end, towards free walking (V0)
                        d["rho"] = round(rho_from_speed(sp), 2)
                    else:
                        d["how"] = "slow" if sp < V_MIN_CURVE else "walking"
                        d["rho"] = round(rho_from_speed(max(sp, V_MIN_CURVE)), 2)
                secs.append(d)
            # floor with no measurement (hidden by the Kaaba, under graphics): the ring's nearest walking sectors
            meas = [i for i, d in enumerate(secs) if d["rho"] is not None]
            if meas and len(meas) < nsec:
                for i, d in enumerate(secs):
                    if d["how"] != "none":
                        continue
                    fw = min(meas, key=lambda j: (j - i) % nsec); bw = min(meas, key=lambda j: (i - j) % nsec)
                    df, db = (fw - i) % nsec, (i - bw) % nsec
                    if min(df, db) > nsec // 4:
                        continue  # more than 90 deg of the ring unmeasured: no estimate
                    wf, wb = 1 / max(df, 1), 1 / max(db, 1)
                    d["rho"] = round((secs[fw]["rho"] * wf + secs[bw]["rho"] * wb) / (wf + wb), 2)
                    d["how"] = "estimated"
            out[ring] = secs
        return out

    @staticmethod
    def ring_reading(secs: list[dict]) -> dict:
        """The ring's speed density (mean of its sectors with one) and how much of it the speed covers."""
        vals = [d["rho"] for d in secs if d["rho"] is not None]
        n = len(secs)
        return {"rho": round(float(np.mean(vals)), 2) if vals else None,
                "walking": sum(d["how"] in ("walking", "slow", "fast") for d in secs) / n,
                "fast": sum(d["how"] == "fast" for d in secs) / n,
                "estimated": sum(d["how"] == "estimated" for d in secs) / n,
                "stopped": sum(d["how"] == "stopped" for d in secs) / n,
                "slow": sum(d["how"] == "slow" for d in secs) / n}


# ring areas (m^2) of the floor between two wall distances: a rounded rectangle around the Kaaba
def ring_area(d0: float, d1: float) -> float:
    a = lambda d: kg.KAABA_L * kg.KAABA_W + 2 * (kg.KAABA_L + kg.KAABA_W) * d + np.pi * d * d  # noqa: E731
    return a(d1) - a(d0)


class LiveSpeed:
    """The sensor in the live shot service (scripts/live_shot.py, 2026-10-30): an analysed frame with a camera pose and
    the stream's next frame (1 s later) make a pair, at most one every EVERY_S s, computed on a worker thread so the
    shot loop never waits; sectors() pools the pairs of the last WINDOW_S s of the camera on air. save_dir + save_when
    (user 2026-10-31: 1 s pairs from real prayers, to test a "stopped -> prayer-row density" rule offline): pairs whose
    first frame save_when(ts) accepts are also written there (both frames, JPEG, and a sidecar with the pose and the
    pair's outcome); local only, never read by the live services."""

    EVERY_S = 6.0
    WINDOW_S = 120.0
    MIN_PAIRS = 3

    def __init__(self, fps: float = 1.0):
        from collections import Counter, deque
        from concurrent.futures import ThreadPoolExecutor

        self.fps = fps  # the stream reader's decode rate: consecutive frames are 1 / fps apart in stream time
        self.stats = Counter()  # offered, paired, no_pose (camera motion not removed), error; read and reset per minute
        self.sensor = TawafSpeed()
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="tawaf-speed")
        self.pending = None
        self.job = None
        self.last = 0.0
        self.pairs = deque()  # (monotonic time, camera, PairSpeed)
        self.save_dir, self.save_when = None, None

    def offer_analysed(self, img: np.ndarray, sr, seq: int, graphics: np.ndarray) -> None:
        import time

        if sr.cam is None or not sr.camera or sr.method not in ("calibrated", "registered") or "+rows" in (sr.how or ""):
            return
        if time.monotonic() - self.last < self.EVERY_S or (self.job is not None and not self.job.done()):
            return
        from edge.m2_pipeline import kaaba_exclusion

        self.pending = (img, sr.cam, sr.camera, sr.ts, seq, graphics, kaaba_exclusion(sr.cam, img.shape[:2]))

    def offer_next(self, img: np.ndarray, ts, seq: int) -> None:
        import time

        p, self.pending = self.pending, None
        if p is None or seq != p[4] + 1:
            return
        # the gap from the frame numbers, not the arrival stamps: the reader caps stream time at "now", so the frames of
        # one 5 s segment can carry nearly the same stamp (2026-10-30: every pair was dropped on that)
        dt = 1.0 / self.fps
        self.last = time.monotonic()
        self.stats["offered"] += 1
        cam_name = p[2]
        save = self.save_dir is not None and self.save_when is not None and self.save_when(p[3])

        def run():
            try:
                ps = self.sensor.pair(p[0], img, p[1], dt, p[5], p[6])
            except Exception:
                self.stats["error"] += 1
                log.exception("tawaf speed pair failed")
                return
            if save:
                self._save(p, img, ts, dt, ps)
            if ps is None:
                self.stats["no_pose"] += 1
                return
            ps.flow = None  # not kept: only the speeds
            self.pairs.append((time.monotonic(), cam_name, ps))
            self.stats["paired"] += 1

        self.job = self.pool.submit(run)

    def _save(self, p, img_b: np.ndarray, ts_b, dt: float, ps) -> None:
        """Both frames of a pair and a sidecar: <save_dir>/<stem>_<camera>_a.jpg / _b.jpg / .json (stem = first frame)."""
        import json

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            cam, stem = p[1], f"{p[3]:%Y%m%dT%H%M%SZ}_{p[2]}"
            for tag, im in (("a", p[0]), ("b", img_b)):
                cv2.imwrite(str(self.save_dir / f"{stem}_{tag}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            side = {"ts_a": p[3].isoformat(), "ts_b": ts_b.isoformat() if ts_b is not None else None, "seq_a": p[4],
                    "dt_s": dt, "camera": p[2],
                    "cam": {"f": float(cam.f), "rvec": [float(x) for x in np.ravel(cam.rvec)],
                            "tvec": [float(x) for x in np.ravel(cam.tvec)], "size": list(cam.size)},
                    "paired": ps is not None, "resid_px": None if ps is None else ps.resid_px,
                    "inliers": None if ps is None else ps.inliers}
            (self.save_dir / f"{stem}.json").write_text(json.dumps(side), encoding="utf-8")
            self.stats["saved"] += 1
        except Exception:
            log.exception("speed pair not saved (the reading is not affected)")

    def sectors(self) -> tuple[str, int, dict] | None:
        """(camera, pairs, sectors per ring) over the last WINDOW_S s of the newest pair's camera, or None."""
        import time

        now = time.monotonic()
        while self.pairs and now - self.pairs[0][0] > self.WINDOW_S:
            self.pairs.popleft()
        if not self.pairs:
            return None
        cam = self.pairs[-1][1]
        ps = [p for _, c, p in self.pairs if c == cam]
        if len(ps) < self.MIN_PAIRS:
            return None
        return cam, len(ps), TawafSpeed.sectors(ps)
