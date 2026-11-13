"""Visual localisation of any Mataf shot against a 3D landmark map (research prototype, 2026-10-26; NOT in the collector).

Why: the collector measures a frame only when it matches a saved reference of a calibrated camera; shots of other
cameras, or of a known camera in an unfamiliar framing, are skipped. A 3D map of the Mataf's fixed landmarks (arcade
columns and arches, the galleries, the Kaaba's belt and door) lets any new frame be localised with PnP.

Map (the "known poses" variant of Structure-from-Motion, as COLMAP's point_triangulator / hloc's triangulation): the
broadcast cameras stand at only three calibrated places, so a free SfM would be badly conditioned (mostly rotations).
Every frame of those cameras already has a calibrated camera in the Kaaba frame (data/samples/m2_* sidecars, the
calibrated references in config/cameras.json, approved extra references via their homography), so the map is metric
and Kaaba-oriented by construction:
  ALIKED (kornia, 2048 keypoints at 640 x 360, broadcast graphics and the crowded floor within 58 m masked) on every
  posed image; LightGlue on every pair; matches checked with the known geometry (epipolar distance for different camera
  places, homography transfer for the same place); tracks by union-find; a track is triangulated (DLT) and kept only if
  it is seen from >= 2 camera places, with every reprojection error <= 2.5 px (1.5 px and >= 3 deg for a
  two-view track), an angle of >= 1.5 deg between the rays,
  and in frames >= 20 minutes apart (people move; buildings do not).
Localisation of a new frame (focal length unknown: any zoom of any camera): ALIKED + LightGlue against the map images,
2D-3D pairs through the map's tracks, PnP RANSAC at a sweep of focal lengths, then a least-squares refinement of
pose + focal length. Accepted if >= MIN_INLIERS inliers, RMS <= MAX_RMS px and a plausible camera place.

  .venv/Scripts/python -I scripts/vloc.py build              map from the posed frames (data/vloc/map.npz)
  .venv/Scripts/python -I scripts/vloc.py validate           frames with a known camera, not used for the map
  .venv/Scripts/python -I scripts/vloc.py coverage           the survey thumbnails the live collector does not count
  .venv/Scripts/python -I scripts/vloc.py bank-save          the panorama bank for the live shot service (data/vloc/bank.npz)
  .venv/Scripts/python -I scripts/vloc.py bank-add --place ID   add a newly calibrated camera place to it
Outputs stay local (data/vloc/, gitignored). GPU memory capped at 25 %.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge import kaaba_geom as kg  # noqa: E402
from edge.ptz import camera_from_homography  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

OUT = ROOT / "data" / "vloc"
WORK = (960, 540)  # map and query resolution (2026-10-26: 640 x 360 gave too few cross-camera matches)
FLOOR_MASK_R = 58.0  # m: the crowded Mataf floor (moving people), masked when building the map
MIN_INLIERS, MAX_RMS = 25, 2.5
MIN_GAP_S = 20 * 60


def scaled(cam: kg.Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K, R, t) at WORK resolution."""
    s = WORK[0] / cam.size[0]
    K = np.array([[cam.f * s, 0, WORK[0] / 2], [0, cam.f * s, WORK[1] / 2], [0, 0, 1.0]])
    return K, cam.R, np.asarray(cam.tvec, float).reshape(3)


def posed_images(cfg: dict, before: str | None = None, after: str | None = None) -> list[dict]:
    """Every image with a known camera: m2 samples, calibrated references, approved extra references."""
    cams = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    out = []
    for vid, c in cams.items():
        ref = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        out.append({"name": f"ref_{vid}", "path": ROOT / "data" / "views" / "refs" / f"{vid}.png", "cam": ref,
                    "place": vid, "t": None})
        view = next((v for v in cfg["views"] if v["id"] == vid), {})
        for r in view.get("references", []):
            if r.get("status") != "approved":
                continue
            img = ROOT / r["image"]
            fc, err = camera_from_homography(np.asarray(r["H_from_primary"], float), ref, (1920, 1080))
            if fc is not None and err < 0.08:
                out.append({"name": r["id"], "path": img, "cam": fc, "place": vid,
                            "t": datetime.fromisoformat(r["captured_utc"]).timestamp()})
    for side in sorted((ROOT / "data" / "samples").glob("m2_*.json")):
        stamp = side.stem.rsplit("_", 1)[1]
        if (before and stamp >= before) or (after and stamp < after):
            continue
        m = json.loads(side.read_text())
        c = m["cam"]
        out.append({"name": side.stem, "path": side.with_suffix(".png"), "place": m["camera"],
                    "cam": kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"])),
                    "t": datetime.fromisoformat(m["ts_utc"]).timestamp()})
    return out


class Features:
    def __init__(self, n: int = 2048):
        import kornia.feature as KF
        import torch

        self.torch = torch
        self.dev = torch.device("cuda")
        self.det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=n).to(self.dev).eval()
        self.lg = KF.LightGlue("aliked").to(self.dev).eval()

    def detect(self, img: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(img, WORK, interpolation=cv2.INTER_AREA if img.shape[1] >= WORK[0] else cv2.INTER_CUBIC)
        t = self.torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255)
        with self.torch.no_grad():
            f = self.det(t.to(self.dev))[0]
        kp, d = f.keypoints.cpu().numpy(), f.descriptors.cpu().numpy()
        if mask is not None and len(kp):
            gx = np.clip(kp[:, 0].astype(int), 0, WORK[0] - 1)
            gy = np.clip(kp[:, 1].astype(int), 0, WORK[1] - 1)
            keep = cv2.dilate(mask, np.ones((5, 5), np.uint8))[gy, gx] == 0
            kp, d = kp[keep], d[keep]
        return kp.astype(np.float32), d.astype(np.float32)

    def match(self, a, b) -> np.ndarray:
        (ka, da), (kb, db) = a, b
        if len(ka) < 8 or len(kb) < 8:
            return np.zeros((0, 2), int)
        T = self.torch
        size = T.tensor([[WORK[0], WORK[1]]], device=self.dev, dtype=T.float32)
        with T.no_grad():
            out = self.lg({"image0": {"keypoints": T.from_numpy(ka)[None].to(self.dev), "descriptors": T.from_numpy(da)[None].to(self.dev),
                                      "image_size": size},
                           "image1": {"keypoints": T.from_numpy(kb)[None].to(self.dev), "descriptors": T.from_numpy(db)[None].to(self.dev),
                                      "image_size": size}})
        return out["matches"][0].cpu().numpy()


def map_mask(cam: kg.Camera, gm: np.ndarray) -> np.ndarray:
    """Broadcast graphics + the crowded floor (rays reaching the head plane within FLOOR_MASK_R m), minus the Kaaba."""
    K, R, t = scaled(cam)
    jj, ii = np.meshgrid(np.arange(WORK[0]) + 0.5, np.arange(WORK[1]) + 0.5)
    d = np.stack([jj, ii, np.ones_like(jj)], -1) @ np.linalg.inv(K).T @ R  # world directions
    C = cam.centre
    lam = (kg.HEAD_Z - C[2]) / np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    X, Y = C[0] + lam * d[..., 0], C[1] + lam * d[..., 1]
    floor = (lam > 0) & (np.hypot(X, Y) < FLOOR_MASK_R)
    sil = kg.silhouette(kg.Camera(cam.f * WORK[0] / cam.size[0], cam.rvec, cam.tvec, WORK), kg.box_corners(z0=2.0))
    m = (floor | (gm > 0)).astype(np.uint8)
    if sil is not None:
        cv2.fillPoly(m, [np.int32(np.round(sil[0]))], 0)
    return m


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def verify(a: dict, b: dict, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Matches consistent with the two known cameras (bool per match)."""
    Ka, Ra, ta = scaled(a["cam"])
    Kb, Rb, tb = scaled(b["cam"])
    if np.linalg.norm(a["cam"].centre - b["cam"].centre) < 2.0:  # same place: a pure rotation + zoom
        H = Kb @ Rb @ Ra.T @ np.linalg.inv(Ka)
        q = np.column_stack([pa, np.ones(len(pa))]) @ H.T
        return np.linalg.norm(q[:, :2] / q[:, 2:3] - pb, axis=1) < 3.0
    R = Rb @ Ra.T
    t = tb - R @ ta
    F = np.linalg.inv(Kb).T @ skew(t) @ R @ np.linalg.inv(Ka)
    x1 = np.column_stack([pa, np.ones(len(pa))])
    x2 = np.column_stack([pb, np.ones(len(pb))])
    l2 = x1 @ F.T
    l1 = x2 @ F
    d2 = np.abs((x2 * l2).sum(1)) / np.hypot(l2[:, 0], l2[:, 1])
    d1 = np.abs((x1 * l1).sum(1)) / np.hypot(l1[:, 0], l1[:, 1])
    return (d1 < 2.0) & (d2 < 2.0)


def triangulate(obs: list[tuple[int, np.ndarray]], imgs: list[dict]) -> tuple[np.ndarray, float, float] | None:
    """DLT over all observations -> (X, max reprojection error px, max ray angle deg)."""
    A, Ps = [], []
    for i, p in obs:
        K, R, t = scaled(imgs[i]["cam"])
        P = K @ np.column_stack([R, t])
        Ps.append(P)
        A += [p[0] * P[2] - P[0], p[1] * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    err, rays = 0.0, []
    for (i, p), P in zip(obs, Ps):
        x = P @ np.append(X, 1)
        if x[2] <= 0:
            return None
        err = max(err, float(np.linalg.norm(x[:2] / x[2] - p)))
        r = X - imgs[i]["cam"].centre
        rays.append(r / np.linalg.norm(r))
    ang = max(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for k, a in enumerate(rays) for b in rays[k + 1:])
    return X, err, float(ang)


def refine_places(imgs: list[dict], F: list, M: dict) -> None:
    """Rotation + focal bundle adjustment per camera place (a 'panorama' adjustment): each broadcast camera only
    rotates and zooms about its fixed centre, so frames of one place are related by K_b R_b R_a^T K_a^-1. The per-frame
    cameras from the live pipeline disagree by ~30 px at 960 x 540; the images themselves agree to ~1-3 px. The
    calibrated reference of each place stays fixed (it carries the Kaaba fit); every other frame of that place gets
    its rotation and focal length from the point matches. Updates imgs[i]["cam"] in place."""
    from scipy.optimize import least_squares

    for place in sorted({im["place"] for im in imgs}):
        idx = [i for i, im in enumerate(imgs) if im["place"] == place]
        anchor = next((i for i in idx if imgs[i]["name"] == f"ref_{place}"), idx[0])
        free = [i for i in idx if i != anchor]
        pos = {i: k for k, i in enumerate(free)}
        pairs = []
        for (i, j), m in M.items():
            if i in idx and j in idx:
                pa, pb = F[i][0][m[:, 0]], F[j][0][m[:, 1]]
                H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
                if H is not None and inl.sum() >= 15:
                    k = inl.ravel() > 0
                    pairs.append((i, j, pa[k].astype(float), pb[k].astype(float)))
        if not pairs or not free:
            continue
        C = imgs[anchor]["cam"].centre
        x0 = np.concatenate([np.concatenate([np.asarray(imgs[i]["cam"].rvec, float).ravel(), [np.log(scaled(imgs[i]["cam"])[0][0, 0])]])
                             for i in free])
        Ka0, Ra0, _ = scaled(imgs[anchor]["cam"])

        def KR(i, x):
            if i == anchor:
                return Ka0, Ra0
            v = x[4 * pos[i]: 4 * pos[i] + 4]
            f = np.exp(v[3])
            return np.array([[f, 0, WORK[0] / 2], [0, f, WORK[1] / 2], [0, 0, 1.0]]), cv2.Rodrigues(v[:3])[0]

        def res(x):
            out = []
            for i, j, pa, pb in pairs:
                Ki, Ri = KR(i, x)
                Kj, Rj = KR(j, x)
                H = Kj @ Rj @ Ri.T @ np.linalg.inv(Ki)
                q = np.column_stack([pa, np.ones(len(pa))]) @ H.T
                out.append((q[:, :2] / q[:, 2:3] - pb).ravel())
                Hb = np.linalg.inv(H)
                q = np.column_stack([pb, np.ones(len(pb))]) @ Hb.T
                out.append((q[:, :2] / q[:, 2:3] - pa).ravel())
            return np.concatenate(out)

        r0 = res(x0)
        sol = least_squares(res, x0, loss="huber", f_scale=2.0, max_nfev=200)
        r1 = res(sol.x)
        print(f"  {place}: {len(free) + 1} frames, {len(pairs)} pairs: transfer error median {np.median(np.abs(r0)):.1f} -> "
              f"{np.median(np.abs(r1)):.1f} px")
        for i in free:
            v = sol.x[4 * pos[i]: 4 * pos[i] + 4]
            R = cv2.Rodrigues(v[:3])[0]
            f_full = float(np.exp(v[3])) * imgs[i]["cam"].size[0] / WORK[0]
            imgs[i]["cam"] = kg.Camera(f_full, v[:3].copy(), -R @ C, imgs[i]["cam"].size)


def build(args) -> None:
    cfg = load_config()
    gm = graphics_mask(cfg, WORK)
    imgs = [im for im in posed_images(cfg, before=args.before) if im["path"].exists()]
    print(f"{len(imgs)} posed images from {len({im['place'] for im in imgs})} camera places")
    feat = Features(n=4096)
    t0 = time.time()
    F = []
    for im in imgs:
        img = cv2.imread(str(im["path"]))
        F.append(feat.detect(img, map_mask(im["cam"], gm)))
    print(f"features: {np.mean([len(k) for k, _ in F]):.0f} per image ({time.time() - t0:.0f} s)")
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    t0 = time.time()
    M = {}
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            m = feat.match(F[i], F[j])
            if len(m) >= 8:
                M[(i, j)] = m
    print(f"matched {len(M)} pairs ({time.time() - t0:.0f} s)")
    refine_places(imgs, F, M)
    n_pairs, n_ok = 0, 0
    for (i, j), m in M.items():
        if True:
            ok = verify(imgs[i], imgs[j], F[i][0][m[:, 0]], F[j][0][m[:, 1]])
            n_pairs += 1
            n_ok += int(ok.sum())
            for a, b in m[ok]:
                ra, rb = find((i, int(a))), find((j, int(b)))
                if ra != rb:
                    parent[ra] = rb
    print(f"verified {n_ok} matches in {n_pairs} pairs")
    tracks: dict = {}
    for node in list(parent):
        tracks.setdefault(find(node), []).append(node)
    pts, obs_all, stats = [], [], {"tracks": len(tracks), "too_short": 0, "conflict": 0, "one_place": 0, "moving": 0, "bad": 0}
    for nodes in tracks.values():
        if len(nodes) < 2:
            stats["too_short"] += 1
            continue
        if len({i for i, _ in nodes}) < len(nodes):
            stats["conflict"] += 1
            continue
        places = {imgs[i]["place"] for i, _ in nodes}
        if len(places) < 2:
            stats["one_place"] += 1
            continue
        ts = [imgs[i]["t"] for i, _ in nodes if imgs[i]["t"] is not None]
        if len(ts) == len(nodes) and max(ts) - min(ts) < MIN_GAP_S:
            stats["moving"] += 1
            continue
        obs = [(i, F[i][0][k]) for i, k in nodes]
        r = triangulate(obs, imgs)
        if r is None or r[1] > (2.5 if len(nodes) >= 3 else 1.5) or r[2] < (1.5 if len(nodes) >= 3 else 3.0):
            stats["bad"] += 1
            continue
        pts.append(r[0])
        obs_all.append([(i, k) for i, k in nodes])
    pts = np.asarray(pts)
    print(f"map: {len(pts)} 3D points; rejected {stats}")
    OUT.mkdir(parents=True, exist_ok=True)
    flat = np.array([(p, i, k) for p, ob in enumerate(obs_all) for i, k in ob], int)
    np.savez_compressed(OUT / "map.npz", points=pts, obs=flat,
                        kps=np.array([f[0] for f in F], dtype=object), descs=np.array([f[1] for f in F], dtype=object),
                        names=np.array([im["name"] for im in imgs]), places=np.array([im["place"] for im in imgs]),
                        times=np.array([im["t"] or 0 for im in imgs], float))
    r = np.hypot(pts[:, 0], pts[:, 1]) if len(pts) else np.zeros(0)
    print(f"points by distance from the Kaaba: <15 m {int((r < 15).sum())}, 15-55 m {int(((r >= 15) & (r < 55)).sum())}, "
          f"55-80 m {int(((r >= 55) & (r < 80)).sum())}, >= 80 m {int((r >= 80).sum())}; height {np.percentile(pts[:, 2], [5, 50, 95]).round(1) if len(pts) else ''}")


class Localiser:
    def __init__(self, feat: Features | None = None):
        z = np.load(OUT / "map.npz", allow_pickle=True)
        self.points, self.kps, self.descs = z["points"], list(z["kps"]), list(z["descs"])
        self.names, self.places = list(z["names"]), list(z["places"])
        self.lut = [dict() for _ in self.kps]  # per map image: keypoint index -> 3D point
        for p, i, k in z["obs"]:
            self.lut[i][int(k)] = int(p)
        self.useful = [i for i, l in enumerate(self.lut) if len(l) >= 15]
        self.feat = feat or Features(n=4096)

    def localise(self, img: np.ndarray, gm: np.ndarray, exclude: set[str] = frozenset()) -> dict | None:
        q = self.feat.detect(img, gm)
        if len(q[0]) < 30:
            return None
        votes: dict = {}
        for i in self.useful:
            if self.names[i] in exclude:
                continue
            m = self.feat.match(q, (self.kps[i], self.descs[i]))
            for a, b in m:
                p = self.lut[i].get(int(b))
                if p is not None:
                    votes.setdefault(int(a), []).append(p)
        if len(votes) < MIN_INLIERS:
            return None
        qa = np.array(list(votes))
        pidx = np.array([max(set(v), key=v.count) for v in votes.values()])
        obj = self.points[pidx].astype(np.float64)
        p2 = q[0][qa].astype(np.float64)
        best = None
        for f in np.geomspace(300, 2600, 22):
            K = np.array([[f, 0, WORK[0] / 2], [0, f, WORK[1] / 2], [0, 0, 1.0]])
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, p2, K, None, iterationsCount=400, reprojectionError=4.0,
                                                     confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if ok and inl is not None and (best is None or len(inl) > best[0]):
                best = (len(inl), f, rvec, tvec, inl.ravel())
        if best is None or best[0] < MIN_INLIERS:
            return None
        from scipy.optimize import least_squares

        n, f, rvec, tvec, inl = best

        def res(x):
            K = np.array([[np.exp(x[6]), 0, WORK[0] / 2], [0, np.exp(x[6]), WORK[1] / 2], [0, 0, 1.0]])
            pr, _ = cv2.projectPoints(obj[inl], x[:3], x[3:6], K, None)
            return (pr[:, 0] - p2[inl]).ravel()

        x = least_squares(res, np.concatenate([rvec.ravel(), tvec.ravel(), [np.log(f)]]), loss="huber", f_scale=2.0).x
        K = np.array([[np.exp(x[6]), 0, WORK[0] / 2], [0, np.exp(x[6]), WORK[1] / 2], [0, 0, 1.0]])
        pr, _ = cv2.projectPoints(obj, x[:3], x[3:6], K, None)
        err = np.linalg.norm(pr[:, 0] - p2, axis=1)
        inl2 = err < 4.0
        rms = float(np.sqrt(np.mean(err[inl2] ** 2))) if inl2.any() else 99.0
        fh = img.shape[1]
        cam = kg.Camera(float(np.exp(x[6])) * fh / WORK[0], x[:3], x[3:6], (img.shape[1], img.shape[0]))
        C = cam.centre
        plausible = np.hypot(C[0], C[1]) < 400 and 2 < C[2] < 250
        return {"cam": cam, "inliers": int(inl2.sum()), "rms": rms, "pairs": len(qa), "ok": bool(inl2.sum() >= MIN_INLIERS
                and rms <= MAX_RMS and plausible), "centre": C}


def validate(args) -> None:
    """Frames with a known camera that the map did not use (built with --before): pose and people per m^2 both ways."""
    from edge.crowd_model import CrowdModel
    from edge.m2_metric import measure
    from edge.m2_pipeline import g, kaaba_exclusion

    cfg = load_config()
    gm_full = graphics_mask(cfg, (1920, 1080))
    gm = graphics_mask(cfg, WORK)
    loc = Localiser()
    model = CrowdModel()
    test = [im for im in posed_images(cfg, after=args.after) if im["name"].startswith("m2_") and im["path"].exists()]
    rows = []
    for im in test:
        img = cv2.imread(str(im["path"]))
        t0 = time.time()
        r = loc.localise(img, gm)
        dt = time.time() - t0
        row = {"frame": im["name"], "camera": im["place"], "ok": bool(r and r["ok"]), "inliers": r["inliers"] if r else 0,
               "rms": round(r["rms"], 2) if r else "", "seconds": round(dt, 2)}
        if r:
            ct, cl = im["cam"], r["cam"]
            row["centre_err_m"] = round(float(np.linalg.norm(ct.centre - cl.centre)), 1)
            Rrel = cl.R @ ct.R.T
            row["rot_err_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))), 2)
            row["f_ratio"] = round(cl.f / ct.f, 3)
            dens = model.predict(img).density
            ex_t = np.maximum(kaaba_exclusion(ct, img.shape[:2]), gm_full)
            ex_l = np.maximum(kaaba_exclusion(cl, img.shape[:2]), gm_full)
            mt, ml = measure(ct, dens, ex_t, g), measure(cl, dens, ex_l, g)
            row["m2_known"] = mt.density_corr.get("all")
            row["m2_localised"] = ml.density_corr.get("all")
        rows.append(row)
        print(row, flush=True)
    write(rows, OUT / "validate.csv")
    ok = [r for r in rows if r["ok"]]
    print(f"\nlocalised {len(ok)} of {len(rows)} frames with a known camera")
    if ok:
        ce = np.array([r["centre_err_m"] for r in ok]); re = np.array([r["rot_err_deg"] for r in ok])
        pr = np.array([(r["m2_localised"] or np.nan) / (r["m2_known"] or np.nan) for r in ok], float)
        print(f"camera place error median {np.median(ce):.1f} m (p90 {np.percentile(ce, 90):.1f}); rotation error median "
              f"{np.median(re):.2f} deg (p90 {np.percentile(re, 90):.2f}); people/m2 localised / known: median "
              f"{np.nanmedian(pr):.3f}, p10-p90 {np.nanpercentile(pr, 10):.2f}-{np.nanpercentile(pr, 90):.2f}")


def coverage(args) -> None:
    """Survey thumbnails (one per 5 min, labelled) that show the Mataf but the live configuration did not count."""
    cfg = load_config()
    gm = graphics_mask(cfg, WORK)
    loc = Localiser()
    rep = list(csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "replay_m2_anchor.csv")))
    rows = []
    for r in rep:
        mataf = r["label"] in ("kaaba", "kaaba_partial", "mataf")
        if not mataf:
            continue
        img = cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"]))
        if img is None:
            continue
        res = loc.localise(img, gm)
        rows.append({"file": r["file"], "label": r["label"], "period": r["period"], "counted_now": r["new"] == "True",
                     "localised": bool(res and res["ok"]), "inliers": res["inliers"] if res else 0,
                     "rms": round(res["rms"], 2) if res else "",
                     "centre": ",".join(f"{v:.0f}" for v in res["centre"]) if res else ""})
        if res and res["ok"] and args.overlays and r["new"] != "True":
            overlay(img, res["cam"], OUT / f"cov_{Path(r['file']).stem}.jpg")
    write(rows, OUT / "coverage.csv")
    n_all = len(rep)
    now = sum(r["counted_now"] for r in rows)
    new = sum(r["localised"] and not r["counted_now"] for r in rows)
    both = sum(r["localised"] and r["counted_now"] for r in rows)
    print(f"Mataf shots {len(rows)} of {n_all}; counted now {now}; localised among those {both}; newly localised "
          f"{new} of {len(rows) - now} not counted now")
    print(f"frames counted: {now} -> {now + new} of {n_all} (~{60 * now / n_all:.1f} -> ~{60 * (now + new) / n_all:.1f} min/h "
          f"by frame share; ceiling {60 * len(rows) / n_all:.1f})")
    for per in ("day", "night"):
        sub = [r for r in rows if r["period"] == per]
        print(f"  {per}: Mataf shots {len(sub)}, counted now {sum(r['counted_now'] for r in sub)}, newly localised "
              f"{sum(r['localised'] and not r['counted_now'] for r in sub)}")


def overlay(img: np.ndarray, cam: kg.Camera, out: Path) -> None:
    o = img.copy()
    sil = kg.silhouette(cam)
    if sil is not None:
        cv2.polylines(o, [np.int32(sil[0])], True, (255, 255, 255), 1, cv2.LINE_AA)
    for r in (10, 25):
        a = np.linspace(0, 2 * np.pi, 200)
        P = np.column_stack([(kg.KAABA_L / 2 + r) * np.cos(a), (kg.KAABA_W / 2 + r) * np.sin(a), np.full(200, kg.HEAD_Z)])
        uv, z = cam.project(P)
        uv = uv[z > 1]
        if len(uv) > 1:
            cv2.polylines(o, [np.int32(uv)], False, (80, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), o, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bank_coverage(args) -> None:
    """The narrower route that the data supports: every new framing of a KNOWN camera place matched against all of
    that place's frames, made consistent by refine_places (a 'panorama bank'); the frame's camera then follows from the
    homography (rotation + zoom about the known centre), as in edge/ptz.py, with the same rotation check."""
    cfg = load_config()
    gm = graphics_mask(cfg, WORK)
    imgs = [im for im in posed_images(cfg, before=args.before) if im["path"].exists()]
    feat = Features(n=4096)
    F = [feat.detect(cv2.imread(str(im["path"])), map_mask(im["cam"], gm)) for im in imgs]
    M = {}
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            m = feat.match(F[i], F[j])
            if len(m) >= 8:
                M[(i, j)] = m
    refine_places(imgs, F, M)
    full = [feat.detect(cv2.imread(str(im["path"])), gm) for im in imgs]  # bank features without the floor mask
    rep = list(csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "replay_m2_anchor.csv")))
    rows = []
    for r in rep:
        if r["label"] not in ("kaaba", "kaaba_partial", "mataf"):
            continue
        img = cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"]))
        q = feat.detect(img, gm)
        best = None
        for i, im in enumerate(imgs):
            m = feat.match(q, full[i])
            if len(m) < 30:
                continue
            pq, pb = q[0][m[:, 0]].astype(float), full[i][0][m[:, 1]].astype(float)
            H, inl = cv2.findHomography(pb, pq, cv2.RANSAC, 3.0)  # bank image px -> query px (WORK)
            if H is None or inl.sum() < 30:
                continue
            S = np.diag([WORK[0] / im["cam"].size[0], WORK[1] / im["cam"].size[1], 1.0])
            Hf = np.linalg.inv(S) @ H @ S  # in full-resolution pixels of both
            cam, err = camera_from_homography(Hf, im["cam"], im["cam"].size)
            if cam is None:
                continue
            ok = (err <= 0.08 and inl.sum() >= 30) or (err <= 0.15 and inl.sum() >= 50)
            if ok and (best is None or inl.sum() > best[0]):
                best = (int(inl.sum()), err, im["place"], cam)
        rows.append({"file": r["file"], "label": r["label"], "period": r["period"], "counted_now": r["new"] == "True",
                     "bank_match": best is not None, "place": best[2] if best else "", "inliers": best[0] if best else 0,
                     "ptz_err": round(best[1], 3) if best else ""})
        if best and args.overlays and r["new"] != "True":
            overlay(cv2.resize(img, (1920, 1080)), best[3], OUT / f"bank_{Path(r['file']).stem}.jpg")
    OUT.mkdir(parents=True, exist_ok=True)
    write(rows, OUT / "bank_coverage.csv")
    n_all = len(rep)
    now = sum(r["counted_now"] for r in rows)
    new = [r for r in rows if r["bank_match"] and not r["counted_now"]]
    print(f"Mataf shots {len(rows)} of {n_all}; counted now {now}; of the {len(rows) - now} not counted, matched to a "
          f"known camera through the bank: {len(new)} ({dict((p, sum(r['place'] == p for r in new)) for p in sorted({r['place'] for r in new}))})")
    print(f"frames counted: {now} -> {now + len(new)} of {n_all} (~{60 * now / n_all:.1f} -> ~{60 * (now + len(new)) / n_all:.1f} min/h by "
          f"frame share; ceiling {60 * len(rows) / n_all:.1f})")
    for per in ("day", "night"):
        sub = [r for r in rows if r["period"] == per]
        print(f"  {per}: Mataf {len(sub)}, counted now {sum(r['counted_now'] for r in sub)}, newly matched "
              f"{sum(r['bank_match'] and not r['counted_now'] for r in sub)}")


BANK = OUT / "bank.npz"
THUMB = (96, 54)


def bank_save(args) -> None:
    """The panorama bank for the live shot service (edge/shot_bank.py): every posed image of a known camera place, its
    camera made consistent within the place (refine_places), its ALIKED features without the floor mask (people move,
    but whole-frame matching needs them for close framings) and a small grey thumbnail for the shortlist."""
    cfg = load_config()
    gm = graphics_mask(cfg, WORK)
    imgs = [im for im in posed_images(cfg) if im["path"].exists()]
    feat = Features(n=4096)
    raw = [cv2.imread(str(im["path"])) for im in imgs]
    F = [feat.detect(img, map_mask(im["cam"], gm)) for img, im in zip(raw, imgs)]
    M = {}
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            if imgs[i]["place"] == imgs[j]["place"]:
                m = feat.match(F[i], F[j])
                if len(m) >= 8:
                    M[(i, j)] = m
    refine_places(imgs, F, M)
    full = [feat.detect(img, gm) for img in raw]
    arrays = {"names": np.array([im["name"] for im in imgs]), "places": np.array([im["place"] for im in imgs]),
              "cams": np.array([[im["cam"].f, *np.asarray(im["cam"].rvec, float).ravel(), *np.asarray(im["cam"].tvec, float).ravel(),
                                 *im["cam"].size] for im in imgs]),
              "thumbs": np.stack([cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)
                                  for img in raw])}
    for i, (kp, d) in enumerate(full):
        arrays[f"kp{i}"], arrays[f"d{i}"] = kp, d.astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BANK, **arrays)
    print(f"bank: {len(imgs)} images ({dict((p, sum(im['place'] == p for im in imgs)) for p in sorted({im['place'] for im in imgs}))}) "
          f"-> {BANK.relative_to(ROOT)} ({BANK.stat().st_size // 1024} KB)")


def bank_add(args) -> None:
    """Add one camera place to the existing bank (a newly calibrated camera, 2026-10-26): its reference and the saved
    frames posed by the live pipeline (data/samples/dvr/<frame>.json from scripts/dvr_cameras.py, one per --every s),
    made consistent within the place; the other places stay as they are."""
    cfg = load_config()
    gm = graphics_mask(cfg, WORK)
    c = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))["cameras"][args.place]
    ref = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
    imgs = [{"name": f"ref_{args.place}", "path": ROOT / c["ref_image"], "cam": ref, "place": args.place, "t": None}]
    last = -1e9
    for side in sorted((ROOT / "data" / "samples" / "dvr").glob("*.json")):
        try:
            m = json.loads(side.read_text())
        except ValueError:
            continue
        if m.get("view") != args.place or not m.get("cam"):
            continue
        t = datetime.strptime(side.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        if t - last < args.every:
            continue
        last = t
        k = m["cam"]
        imgs.append({"name": f"dvr_{side.stem}", "path": side.with_suffix(".jpg"), "place": args.place, "t": t,
                     "cam": kg.Camera(k["f"], np.asarray(k["rvec"], float), np.asarray(k["tvec"], float), tuple(k["size"]))})
    feat = Features(n=4096)
    raw = [cv2.imread(str(im["path"])) for im in imgs]
    F = [feat.detect(img, map_mask(im["cam"], gm)) for img, im in zip(raw, imgs)]
    M = {(i, j): m for i in range(len(imgs)) for j in range(i + 1, len(imgs))
         if len(m := feat.match(F[i], F[j])) >= 8}
    refine_places(imgs, F, M)
    full = [feat.detect(img, gm) for img in raw]
    z = dict(np.load(BANK))
    n0 = len(z["names"])
    z["names"] = np.concatenate([z["names"], [im["name"] for im in imgs]])
    z["places"] = np.concatenate([z["places"], [im["place"] for im in imgs]])
    z["cams"] = np.concatenate([z["cams"], [[im["cam"].f, *np.asarray(im["cam"].rvec, float).ravel(),
                                             *np.asarray(im["cam"].tvec, float).ravel(), *im["cam"].size] for im in imgs]])
    z["thumbs"] = np.concatenate([z["thumbs"], [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB,
                                                           interpolation=cv2.INTER_AREA) for img in raw]])
    for i, (kp, d) in enumerate(full):
        z[f"kp{n0 + i}"], z[f"d{n0 + i}"] = kp, d.astype(np.float16)
    np.savez_compressed(BANK, **z)
    print(f"bank: +{len(imgs)} images of {args.place} -> {len(z['names'])} ({BANK.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "validate", "coverage", "bank", "bank-save", "bank-add"])
    ap.add_argument("--place", help="bank-add: the camera place (config/cameras.json id)")
    ap.add_argument("--every", type=float, default=10.0, help="bank-add: seconds between saved frames taken")
    ap.add_argument("--before", default=None, help="build: only m2 samples with a UTC stamp before this (e.g. 20261026)")
    ap.add_argument("--after", default="20261026", help="validate: m2 samples from this stamp on")
    ap.add_argument("--overlays", action="store_true")
    args = ap.parse_args()
    {"build": build, "validate": validate, "coverage": coverage, "bank": bank_coverage, "bank-save": bank_save,
     "bank-add": bank_add}[args.cmd](args)


if __name__ == "__main__":
    main()

