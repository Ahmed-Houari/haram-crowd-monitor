"""Camera geometry from the Kaaba: the view-independent part of the crowd metric (research prototype, 2026-10-24).

The broadcast keeps changing camera, framing and zoom, but most Mataf shots show the Kaaba, an object of known size.
Fitting a box of the Kaaba's size to its silhouette (edge/kaaba_seg.py) gives the camera (focal length, rotation,
position) in metres, so every pixel of the Mataf floor gets a position and an area in square metres.

World frame: origin on the Mataf floor under the centre of the Kaaba, z up, metres; x along the long walls (the door
wall NE and the opposite SW wall), y along the short walls (Hijr Ismail side NW and the opposite SE wall).
Kaaba (Umm al-Qura University Hajj research centre survey, via Saudipedia): walls 11.68 m (Black Stone corner to
Shami corner, with the door), 9.90 m (Shami to Western), 12.04 m (Western to Yamani), 10.18 m (Yamani to Black Stone):
modelled as a box of the mean lengths. Height: the Kiswa is 14 m high (Saudi Press Agency) and hangs from the roof
edge down to the Shadharwan base (~0.25 m); published overall heights range from 13.1 m to 15 m, so Z_TOP is checked
against frames where the base is visible (`fit` with free_height=True).
Camera: pinhole, principal point at the image centre, square pixels, no lens distortion: x ~ K (R X + t).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize

KAABA_L = (11.68 + 12.04) / 2  # m, the two long walls
KAABA_W = (9.90 + 10.18) / 2  # m, the two short walls
Z_BOTTOM = 0.25  # m, bottom of the black Kiswa (top of the Shadharwan)
Z_TOP = 14.25  # m, top of the Kiswa (14 m of cloth above the Shadharwan)
HEAD_Z = 1.55  # m, head-centre height of an adult: crowd density maps put people at their heads
Z_OCC0 = 1.8  # m, prior height up to which people in front hide the Kaaba (head level); fitted per frame
CUT_Z = 2.0  # m: the Kaaba's cut-out in a frame (not floor) is the box from here up; below it people stand in front
# of its base and are counted (edge/m2_pipeline.kaaba_exclusion; the AI view and the twin since 2026-10-27)


@dataclass
class Camera:
    f: float  # focal length, px
    rvec: np.ndarray  # Rodrigues rotation, world -> camera
    tvec: np.ndarray  # translation, world -> camera
    size: tuple[int, int]  # image (w, h)

    @property
    def K(self) -> np.ndarray:
        w, h = self.size
        return np.array([[self.f, 0, w / 2], [0, self.f, h / 2], [0, 0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, np.float64))[0]

    @property
    def centre(self) -> np.ndarray:
        """Camera position in the world frame (m)."""
        return -self.R.T @ np.asarray(self.tvec, np.float64)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(image points (N, 2), depth (N,)) of world points (N, 3)."""
        Xc = (self.R @ np.asarray(X, np.float64).T).T + np.asarray(self.tvec, np.float64)
        z = Xc[:, 2]
        uv = (self.K @ Xc.T).T
        return uv[:, :2] / np.where(np.abs(uv[:, 2:]) < 1e-9, 1e-9, uv[:, 2:]), z

    def plane_to_image(self, z: float) -> np.ndarray:
        """3x3 homography (X, Y, 1) on the horizontal plane at height z -> image px."""
        R, t = self.R, np.asarray(self.tvec, np.float64)
        return self.K @ np.column_stack([R[:, 0], R[:, 1], z * R[:, 2] + t])

    def image_to_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        """(N, 2) image px -> (N, 2) world (X, Y) on the plane at height z (rays that miss it give nonsense)."""
        H = np.linalg.inv(self.plane_to_image(z))
        p = np.column_stack([uv, np.ones(len(uv))]) @ H.T
        return p[:, :2] / p[:, 2:]


def box_corners(l: float = KAABA_L, w: float = KAABA_W, z0: float = Z_BOTTOM, z1: float = Z_TOP) -> np.ndarray:
    """(8, 3): bottom corners 0-3, top corners 4-7, counter-clockwise seen from above."""
    xy = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    return np.vstack([np.column_stack([xy, np.full(4, z0)]), np.column_stack([xy, np.full(4, z1)])])


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def silhouette(cam: Camera, corners: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(hull polygon (M, 2) counter-clockwise in image coords, corner index per hull vertex), None if behind."""
    X = box_corners() if corners is None else corners
    uv, z = cam.project(X)
    if (z <= 0.5).any():
        return None
    idx = cv2.convexHull(uv.astype(np.float32), returnPoints=False).ravel()
    return uv[idx], idx


def look_at_camera(f: float, azim: float, elev: float, dist: float, size: tuple[int, int],
                   target_px: tuple[float, float] | None = None, target: np.ndarray | None = None,
                   roll: float = 0.0) -> Camera:
    """Camera at `dist` m from `target` (default: the Kaaba centre at mid height), seen from azimuth / elevation
    (radians), rotated so that the target lands on `target_px` (default: the image centre)."""
    T = np.array([0, 0, (Z_BOTTOM + Z_TOP) / 2]) if target is None else np.asarray(target, float)
    C = T + dist * np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    fwd = (T - C) / np.linalg.norm(T - C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])  # rows: camera x (right), y (down), z (forward) in world coords
    if roll:
        cr, sr = np.cos(roll), np.sin(roll)
        R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ R
    if target_px is not None:  # pan / tilt so that the target appears at target_px instead of the centre
        w, h = size
        ray = np.array([target_px[0] - w / 2, target_px[1] - h / 2, f])
        ray /= np.linalg.norm(ray)
        # rotation Q taking the optical axis (0, 0, 1) to `ray`, in camera coordinates: R' = Q R
        v = np.cross([0, 0, 1.0], ray)
        s, c = np.linalg.norm(v), ray[2]
        if s > 1e-12:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Q = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
            R = Q @ R
    t = -R @ C
    return Camera(f, cv2.Rodrigues(R)[0].ravel(), t, size)


@dataclass
class Fit:
    cam: Camera
    cost: float  # robust cost at the solution
    rms_px: float  # RMS distance of the silhouette's top and side edges to the observed outline (px)
    iou: float  # silhouette IoU with the observed mask (the base may be hidden by people)
    n_obs: int
    z_top: float = Z_TOP  # Kiswa top used (fitted when free_height)
    z_occ: float = Z_OCC0  # height up to which people hide the Kaaba (fitted)
    f_range: tuple[float, float] = (0.0, 0.0)  # focal lengths whose cost is within 10 % (+ 3) of the best: ambiguity


def _look_cam(q: np.ndarray, size: tuple[int, int]) -> Camera:
    """Camera from the well-conditioned search parameters q = (log f, azimuth, elevation, log distance, u, v, roll,
    z_occ[, height delta]): the Kaaba centre (mid height) seen from that direction and distance, landing at image
    point (u, v); z_occ = how high people in front hide the Kaaba."""
    return look_at_camera(float(np.exp(q[0])), q[1], q[2], float(np.exp(q[3])), size, (q[4], q[5]), roll=q[6])


@dataclass
class _Obs:
    """The observed outline, prepared once per fit."""
    pts: np.ndarray  # (N, 2) outline points, full-image px
    down: np.ndarray  # (N,) outward normal points down: crowd line in front of the base, or the base itself
    border: np.ndarray  # (N,) on the frame border: not an edge of the Kaaba (a zoomed shot cuts it)
    dt: np.ndarray  # distance to the outline (crop px)
    off: np.ndarray  # crop offset (x, y)
    bottom_y: np.ndarray  # lowest observed mask row per crop column (-1 where empty)
    frame: tuple[int, int]  # (w, h) of the frame


def _prepare(mask: np.ndarray, crop: np.ndarray, off: np.ndarray) -> _Obs:
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    n = len(c)
    k = max(3, n // 120)
    tang = np.roll(c, -k, axis=0) - np.roll(c, k, axis=0)
    normal = np.column_stack([tang[:, 1], -tang[:, 0]])  # outward for OpenCV's contour orientation
    if cv2.contourArea(c.astype(np.float32), oriented=True) < 0:
        normal = -normal
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    sel = np.linspace(0, n - 1, min(240, n)).astype(int)
    edge = np.zeros_like(crop)
    cv2.polylines(edge, [(c - off).astype(np.int32).reshape(-1, 1, 2)], True, 1, 1)
    dt = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
    rows = np.arange(crop.shape[0])[:, None]
    bottom_y = np.where(crop.any(0), (rows * (crop > 0)).max(0), -1)
    pts = c[sel]
    border = (pts[:, 0] <= 2) | (pts[:, 1] <= 2) | (pts[:, 0] >= w - 3) | (pts[:, 1] >= h - 3)
    return _Obs(pts, normal[sel, 1] > 0.45, border, dt, off, bottom_y, (w, h))


def _residuals(q: np.ndarray, ob: _Obs, size: tuple[int, int], free_height: bool) -> np.ndarray:
    """Observed outline vs the silhouette of the visible part of the Kaaba: the box from z_occ (people in front hide
    the part below, about head level; the base itself when nobody stands there) up to the top.

    Every observed outline point lies on the model outline (two-sided, robust). Model outline points of the top and
    side edges lie on the observed outline; the bottom edge (the crowd line) is jagged, so it counts only through the
    observed points. A weak prior keeps z_occ near head level.
    """
    z1 = Z_TOP + (q[8] if free_height else 0.0)
    z_occ = float(np.clip(q[7], Z_BOTTOM, 0.5 * z1))
    cam = _look_cam(q, size)
    sil = silhouette(cam, box_corners(z0=z_occ, z1=z1))
    n_q = 120
    if sil is None:
        return np.full(len(ob.pts) + n_q + 1, 1e3)
    poly = sil[0]
    if not np.isfinite(poly).all() or np.abs(poly).max() > 1e6:  # the optimizer tried a camera that throws the box
        return np.full(len(ob.pts) + n_q + 1, 1e3)  # far off-screen: as bad as not seeing it
    d, n_out = _signed_dist(ob.pts, poly)  # + inside
    r_obs = np.where(ob.down, 0.7 * d, d)  # the crowd line is jagged: a little less weight
    r_obs = np.where(ob.border, np.minimum(d, 0.0), r_obs)  # frame border: the Kaaba may continue outside
    pts = []
    for i in range(len(poly)):
        if n_out[i, 1] > 0.45:  # bottom edge: may be hidden by people
            continue
        a, t = poly[i], poly[(i + 1) % len(poly)] - poly[i]
        # one point every 6 px, at most 2000 per edge: only 120 are kept below, and a huge projected box once asked
        # for billions (2026-10-25: 72 GB, the crowd collector hung in the anchor fit)
        k = min(max(2, int(np.hypot(*t) / 6)), 2000)
        pts.append(a + t * np.linspace(0, 1, k)[:, None])
    q_ = np.vstack(pts) if pts else np.zeros((0, 2))
    if len(q_):
        cx = np.clip(np.round(q_[:, 0] - ob.off[0]).astype(int), 0, len(ob.bottom_y) - 1)
        q_ = q_[(q_[:, 1] - ob.off[1]) <= ob.bottom_y[cx] + 2]  # only where the frame shows the Kaaba
        fw, fh = ob.frame
        q_ = q_[(q_[:, 0] > 2) & (q_[:, 1] > 2) & (q_[:, 0] < fw - 3) & (q_[:, 1] < fh - 3)]
    q_ = q_[np.linspace(0, len(q_) - 1, n_q).astype(int)] if len(q_) else np.zeros((n_q, 2)) + 1e4
    r_mod = _bilinear(ob.dt, q_ - ob.off)
    return np.concatenate([r_obs, r_mod, [3.0 * (q[7] - Z_OCC0)]])


def _signed_dist(pts: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of points to a convex polygon (positive inside) and the outward unit normal of each edge."""
    a = poly
    e = np.roll(poly, -1, axis=0) - a
    n = np.column_stack([e[:, 1], -e[:, 0]]) / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    if (n * (a - poly.mean(0))).sum(1).mean() < 0:  # make the normals point outward
        n = -n
    rel = pts[:, None, :] - a[None]
    side = (rel * n[None]).sum(-1)  # > 0 outside that edge's line
    t = np.clip((rel * e[None]).sum(-1) / np.maximum((e ** 2).sum(-1), 1e-9)[None], 0, 1)
    dseg = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(1)
    inside = (side <= 0).all(1)
    return np.where(inside, dseg, -dseg), n


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """img sampled at float (x, y) points (clamped to the image): smooth, so finite differences see a slope."""
    h, w = img.shape
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy
            + img[y0 + 1, x0 + 1] * fx * fy)


def fit(mask: np.ndarray, size: tuple[int, int] | None = None, init: list[Camera] | None = None,
        free_height: bool = False, top_k: int = 3, fixed_f: float | None = None) -> Fit | None:
    """Camera from the Kaaba's silhouette mask (uint8, full-image coordinates).

    A grid of cameras around the Kaaba (focal length, elevation, azimuth; distance and aim set so the model's bounding
    box matches the observed one) is scored by silhouette IoU. From the best few, the focal length is swept in 1-D
    (a bigger focal length with a farther camera gives almost the same silhouette) with the pose solved by least
    squares at each value, refined by golden section, then everything is polished (Powell, then least squares). The
    model is the box from z_occ (people in front hide the part below, fitted) to the top, compared with the outline
    by _residuals. fixed_f: the focal length is known (a calibrated camera, see edge/ptz.py): only the pose is fitted,
    which removes the focal length / distance ambiguity. Synthetic check (2026-10-24): elevated views within 1-3 % of
    the true floor area per pixel; low, close views stay ambiguous (f_range wide), hence the gate in m2_pipeline.
    """
    h, w = mask.shape[:2]
    size = size or (w, h)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.6 * max(x1 - x0, y1 - y0)) + 10
    off = np.array([max(0, x0 - pad), max(0, y0 - pad)], np.float64)
    crop = mask[int(off[1]):y1 + pad, int(off[0]):x1 + pad].astype(np.uint8)
    ob = _prepare(mask, crop, off)
    starts = [_params_of(c) for c in init] if init else _grid(crop, off, (x0, y0, x1, y1), size)
    best = None
    for q0 in starts[:top_k]:
        q0 = np.concatenate([q0[:7], [Z_OCC0], [0.0]] if free_height else [q0[:7], [Z_OCC0]])
        if fixed_f is not None:
            q0[3] += np.log(fixed_f) - q0[0]  # keep the Kaaba's image size
            q0[0] = np.log(fixed_f)
            c, q = _solve_pose(q0, ob, size, free_height)
            curve = [(q[0], c)]
        else:
            c, q, curve = _sweep_f(q0, ob, size, free_height, span=0.25 if init else 0.9)
        if best is None or c < best[0]:
            best = (c, q, curve)
    if best is None:
        return None
    c, qb = _polish(best[1], ob, size, free_height) if fixed_f is None else (best[0], best[1])
    near = [np.exp(lf) for lf, cc in best[2] if cc <= 1.1 * min(c, best[0]) + 3.0]
    f_range = (float(min(near, default=np.exp(qb[0]))), float(max(near, default=np.exp(qb[0]))))
    cam = _look_cam(qb, size)
    z1 = Z_TOP + (qb[8] if free_height else 0.0)
    res = _residuals(qb, ob, size, free_height)
    sil = silhouette(cam, box_corners(z0=float(np.clip(qb[7], Z_BOTTOM, 0.5 * z1)), z1=z1))
    iou = 0.0
    if sil is not None:
        m2 = np.zeros_like(crop)
        cv2.fillPoly(m2, [np.round(sil[0] - off).astype(np.int32)], 1)
        iou = float((m2 & crop).sum() / max((m2 | crop).sum(), 1))
    return Fit(cam, c, float(np.sqrt(np.mean(res[len(ob.pts):-1] ** 2))), iou, len(ob.pts), z1,
               float(np.clip(qb[7], Z_BOTTOM, 0.5 * z1)), f_range)


def fit_from_centre(mask: np.ndarray, centre: np.ndarray, size: tuple[int, int] | None = None,
                    init: Camera | None = None, budget_s: float = 3.0) -> Fit | None:
    """Camera at a known position (a calibrated broadcast camera) from the Kaaba's silhouette mask: only the rotation
    and the focal length are fitted (2026-10-25). With the position known, fit()'s trade-off between focal length and
    distance is gone, so this works at any zoom and framing: zoomed-in shots of a calibrated camera, where feature
    matching to its reference fails or gives a wrong rotation, get their camera this way. init: a rough camera to
    start from (e.g. a ptz match that failed the Kaaba check). budget_s: no further starts after this long (it runs
    in the crowd collector's frame loop)."""
    h, w = mask.shape[:2]
    size = size or (w, h)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.6 * max(x1 - x0, y1 - y0)) + 10
    off = np.array([max(0, x0 - pad), max(0, y0 - pad)], np.float64)
    crop = mask[int(off[1]):y1 + pad, int(off[0]):x1 + pad].astype(np.uint8)
    ob = _prepare(mask, crop, off)
    d = np.asarray(centre, np.float64) - np.array([0, 0, (Z_BOTTOM + Z_TOP) / 2])
    dist = float(np.linalg.norm(d))
    azim, elev = float(np.arctan2(d[1], d[0])), float(np.arcsin(d[2] / dist))
    seen = KAABA_L * abs(np.sin(azim)) + KAABA_W * abs(np.cos(azim))  # rough width of the Kaaba seen from there
    f_box = dist * (x1 - x0) / max(seen, 1.0)
    starts = [np.array([np.log(f_box * s), azim, elev, np.log(dist), (x0 + x1) / 2, (y0 + y1) / 2, 0.0, Z_OCC0])
              for s in (0.8, 1.0, 1.25)]
    if init is not None:
        p = _params_of(init)
        starts.insert(0, np.array([p[0], azim, elev, np.log(dist), p[4], p[5], p[6], Z_OCC0]))
    free = [0, 4, 5, 6, 7]  # log f, the Kaaba's centre in px (u, v), roll, z_occ; the position stays fixed
    best = None
    t0 = time.perf_counter()
    for q0 in starts:
        if best is not None and time.perf_counter() - t0 > budget_s:
            break
        def res(z: np.ndarray, q0: np.ndarray = q0) -> np.ndarray:
            q = q0.copy()
            q[free] = z
            return _residuals(q, ob, size, False)

        try:
            r = least_squares(res, q0[free], loss="soft_l1", f_scale=2.0, x_scale=_SCALE[free], diff_step=1e-3,
                              max_nfev=100)
        except ValueError:
            continue
        q = q0.copy()
        q[free] = r.x
        c = _cost(q, ob, size, False)
        if best is None or c < best[0]:
            best = (c, q)
    if best is None:
        return None
    c, q = best
    cam = _look_cam(q, size)
    res_all = _residuals(q, ob, size, False)
    z_occ = float(np.clip(q[7], Z_BOTTOM, 0.5 * Z_TOP))
    sil = silhouette(cam, box_corners(z0=z_occ))
    iou = 0.0
    if sil is not None:
        m2 = np.zeros_like(crop)
        cv2.fillPoly(m2, [np.round(sil[0] - off).astype(np.int32)], 1)
        iou = float((m2 & crop).sum() / max((m2 | crop).sum(), 1))
    return Fit(cam, c, float(np.sqrt(np.mean(res_all[len(ob.pts):-1] ** 2))), iou, len(ob.pts), Z_TOP, z_occ,
               (cam.f, cam.f))


_SCALE = np.r_[0.05, 0.02, 0.02, 0.05, 5, 5, 0.01, 0.2, 0.1]


def _cost(q: np.ndarray, ob: _Obs, size: tuple[int, int], free_height: bool) -> float:
    r = _residuals(q, ob, size, free_height)
    return float(np.sum(2 * (np.sqrt(1 + (r / 2) ** 2) - 1)))  # soft-l1 with a 2 px scale


def _solve_pose(q0: np.ndarray, ob: _Obs, size: tuple[int, int], free_height: bool) -> tuple[float, np.ndarray]:
    """Best pose (everything but the focal length) from q0, by least squares."""
    free = np.arange(1, len(q0))

    def res(z: np.ndarray) -> np.ndarray:
        q = q0.copy()
        q[free] = z
        return _residuals(q, ob, size, free_height)

    try:
        r = least_squares(res, q0[free], loss="soft_l1", f_scale=2.0, x_scale=_SCALE[free], diff_step=1e-3, max_nfev=200)
        z = r.x
    except ValueError:
        z = q0[free]
    q = q0.copy()
    q[free] = z
    return _cost(q, ob, size, free_height), q


def _sweep_f(q0: np.ndarray, ob: _Obs, size: tuple[int, int], free_height: bool, span: float = 0.9,
             n: int = 7) -> tuple[float, np.ndarray, list]:
    """The focal length is the poorly determined direction (bigger f and a farther camera give almost the same
    silhouette), so it is searched in 1-D, the pose solved for each value, then refined by golden section."""
    logs = q0[0] + np.linspace(-span, span, n)
    sols: dict[float, tuple[float, np.ndarray]] = {}

    def at(lf: float, start: np.ndarray) -> tuple[float, np.ndarray]:
        q = start.copy()
        q[3] += lf - q[0]  # keep the Kaaba's image size: distance scales with f
        q[0] = lf
        sols[lf] = _solve_pose(q, ob, size, free_height)
        return sols[lf]

    mid = n // 2
    at(logs[mid], q0)
    for i in range(mid + 1, n):  # continuation outwards from the start
        at(logs[i], sols[logs[i - 1]][1])
    for i in range(mid - 1, -1, -1):
        at(logs[i], sols[logs[i + 1]][1])
    k = int(np.argmin([sols[lf][0] for lf in logs]))
    lo, hi = logs[max(k - 1, 0)], logs[min(k + 1, n - 1)]
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c1, c2 = b - g * (b - a), a + g * (b - a)
    s1, s2 = at(c1, sols[logs[k]][1]), at(c2, sols[logs[k]][1])
    for _ in range(4):
        if s1[0] < s2[0]:
            b, c2, s2 = c2, c1, s1
            c1 = b - g * (b - a)
            s1 = at(c1, s2[1])
        else:
            a, c1, s1 = c1, c2, s2
            c2 = a + g * (b - a)
            s2 = at(c2, s1[1])
    c, q = min(sols.values(), key=lambda s: s[0])
    return c, q, sorted((lf, cc) for lf, (cc, _) in sols.items())


def _polish(q: np.ndarray, ob: _Obs, size: tuple[int, int], free_height: bool) -> tuple[float, np.ndarray]:
    """All parameters together: derivative-free first (the silhouette's corners make the residuals kinked), then
    least squares."""
    sc = _SCALE[:len(q)]
    rp = minimize(lambda w: _cost(q + w * sc, ob, size, free_height), np.zeros(len(q)), method="Powell",
                  options={"maxiter": 4000, "xtol": 1e-3, "ftol": 1e-7})
    if rp.fun < _cost(q, ob, size, free_height):
        q = q + rp.x * sc
    try:
        r = least_squares(_residuals, q, args=(ob, size, free_height), loss="soft_l1", f_scale=2.0,
                          x_scale=_SCALE[:len(q)], diff_step=1e-3, max_nfev=200)
        if _cost(r.x, ob, size, free_height) < _cost(q, ob, size, free_height):
            q = r.x
    except ValueError:
        pass
    return _cost(q, ob, size, free_height), q


def _params_of(cam: Camera) -> np.ndarray:
    """Search parameters (see _look_cam) of an existing camera."""
    T = np.array([0, 0, (Z_BOTTOM + Z_TOP) / 2])
    d = cam.centre - T
    dist = np.linalg.norm(d)
    azim, elev = np.arctan2(d[1], d[0]), np.arcsin(d[2] / dist)
    uv, _ = cam.project(T[None])
    u, v = uv[0]
    base = look_at_camera(cam.f, azim, elev, dist, cam.size, (u, v))
    dR = cam.R @ base.R.T  # remaining rotation about the target ray ~ roll
    roll = float(np.arctan2(dR[1, 0], dR[0, 0]))
    return np.array([np.log(cam.f), azim, elev, np.log(dist), u, v, roll])


def _grid(crop: np.ndarray, off: np.ndarray, bbox: tuple, size: tuple[int, int], keep: int = 6) -> list[np.ndarray]:
    """Coarse search: for each (focal length, elevation, azimuth) the distance and aim are set so that the model's
    bounding box matches the observed one, then candidates are scored by silhouette IoU. Returns the search
    parameters of the best few, spread over focal lengths (the poorly determined direction)."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    small = cv2.resize(crop, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) > 0
    corners = box_corners(z0=Z_OCC0)
    scored = []
    for f in np.geomspace(max(size) * 0.5, max(size) * 12, 12):
        for elev in np.radians(np.arange(-4, 75, 5)):
            for azim in np.radians(np.arange(0, 180, 7.5)):
                diag = KAABA_L * abs(np.sin(azim)) + KAABA_W * abs(np.cos(azim))
                q = np.array([np.log(f), azim, elev, np.log(max(f * diag / max(bw, 1), 16.0)), (x0 + x1) / 2,
                              (y0 + y1) / 2, 0.0])
                ok = True
                for _ in range(2):  # snap the model's bounding box onto the observed one
                    uv, z = _look_cam(q, size).project(corners)
                    if (z <= 0.5).any():
                        ok = False
                        break
                    mw, mh = np.ptp(uv[:, 0]), np.ptp(uv[:, 1])
                    q[3] += np.log(max(mw / max(bw, 1), mh / max(bh, 1)) if mw > 0 else 1.0)
                    q[4] += (x0 + x1) / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2
                    q[5] += (y0 + y1) / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2
                if not ok or np.exp(q[3]) < 16:
                    continue
                sil = silhouette(_look_cam(q, size), corners)
                if sil is None:
                    continue
                m = np.zeros(small.shape, np.uint8)
                cv2.fillPoly(m, [np.round((sil[0] - off) * 0.25).astype(np.int32)], 1)
                m = m > 0
                scored.append(((m & small).sum() / max((m | small).sum(), 1), q))
    scored.sort(key=lambda s: -s[0])
    out = []
    for iou, q in scored:  # the best per focal-length bin first, then the rest
        if all(abs(q[0] - o[0]) > 0.4 for o in out):
            out.append(q)
        if len(out) >= keep:
            break
    return out
