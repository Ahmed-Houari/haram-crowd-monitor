"""Known broadcast cameras as pan / tilt / zoom cameras (research prototype, 2026-10-24).

Each broadcast camera sits at a fixed place and only pans, tilts and zooms. Calibrate it once (its reference frame,
camera fitted to the Kaaba by edge/kaaba_geom.py) and every later frame of that camera follows from the homography
between the reference and the frame: for a camera that only rotates and zooms about a fixed centre,
H ~ K_t R K_ref^-1, so the frame's focal length and rotation come out directly, with or without the Kaaba in view.
The same relation is the check that a match is real: K_t^-1 H K_ref must be a (scaled) rotation.

Matching: ALIKED (BSD-3) + LightGlue (Apache-2.0) via kornia, which match day frames to night references where SIFT
fails (scripts/match_eval.py, 2026-10-24: 46 of 95 daytime Mataf thumbnails vs 0 with SIFT).

Besides the calibrated reference frame, a camera can be matched through its approved extra references (another light,
config/views.json, with their homography to the reference) and through keyframes: the caller keeps a very good match
whose camera it has verified (edge/m2_pipeline.py: the projected Kaaba lies on the Kaaba) as one more reference of
that camera (its features, in memory only, never the frame), so the matches follow the light from night into day.
Every homography is chained back to the calibrated reference before the rotation check.

The rotation check alone is not enough (replay of the saved thumbnails, 2026-10-24): matches with an error above about
0.06 can be rotations of the right camera that are still several degrees off (the projected Kaaba a Kaaba-width away),
e.g. matches dominated by distant, repetitive arcades. Hence the caller's check on the Kaaba.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from edge.kaaba_geom import Camera

WORK = (640, 360)  # matching resolution
MIN_INLIERS = 30
# A match counts when K^-1 H K_ref is close to a scaled rotation (relative deviation): <= 0.08 with >= 30 inliers, or
# <= 0.15 with >= 50. Live 1080p frames (2026-10-24): real matches 0.006-0.14 (most < 0.06), false ones >= 0.25.
PTZ_GATES = ((0.08, 30), (0.15, 50))
KEY_GATE = (0.04, 100)  # a match at least this good (error, inliers) can become a keyframe
KEY_NOVEL = 400  # ... unless it matched with more inliers than this (the bank already covers that look)
KEY_EVERY_S = 120  # at most one new keyframe per camera this often
MAX_KEYS = 12  # keyframes kept per camera, oldest dropped (the light changes through the day)
MAX_DEPTH = 3  # links from the calibrated reference: reference -> keyframe -> keyframe -> keyframe
STRONG = 200  # a valid match with this many inliers ends the search (references tried most recently useful first)


def ptz_ok(inliers: int, err: float) -> bool:
    return any(err <= e and inliers >= n for e, n in PTZ_GATES)


@dataclass(eq=False)  # compared by identity (the features are tensors)
class _Ref:
    view_id: str
    feats: tuple  # ALIKED keypoints, descriptors (WORK px)
    size: tuple[int, int]  # full-res px
    H0: np.ndarray  # calibrated reference px -> this reference's px (identity for the calibrated reference)
    depth: int
    t: float = 0.0  # keyframes: frame time (s); 0 for references from disk (kept)
    used: float = 0.0  # last time this reference gave the accepted match (tried first)


@dataclass
class PtzMatch:
    view_id: str
    inliers: int
    H: np.ndarray  # calibrated reference full-res px -> frame full-res px
    cam: Camera | None  # the frame's camera (None if the homography is not a rotation + zoom of that camera)
    ptz_err: float
    zoom: float  # f_frame / f_reference (frame-size normalised)
    depth: int = 0  # 0 matched the calibrated reference itself, n through a reference n links from it
    ref: _Ref | None = field(default=None, repr=False)  # the reference that matched
    feats: tuple | None = field(default=None, repr=False)  # the frame's features (for keep())
    size: tuple[int, int] = (0, 0)  # the frame's size


def camera_from_homography(H: np.ndarray, ref: Camera, frame_size: tuple[int, int]) -> tuple[Camera | None, float]:
    """(frame camera, PTZ error) from a reference->frame homography of the same pan/tilt/zoom camera."""
    w, h = frame_size
    rw, rh = ref.size
    Tf = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])  # centred pixel coordinates
    Tr = np.array([[1, 0, -rw / 2], [0, 1, -rh / 2], [0, 0, 1.0]])
    A = Tf @ H @ np.linalg.inv(Tr) @ np.diag([ref.f, ref.f, 1.0])  # = K_t' R (up to scale), K_t' = diag(f_t, f_t, 1)
    AAt = A @ A.T
    if AAt[2, 2] <= 0:
        return None, 1.0
    f2a, f2b = AAt[0, 0] / AAt[2, 2], AAt[1, 1] / AAt[2, 2]
    if f2a <= 0 or f2b <= 0:
        return None, 1.0
    f_t = float(np.sqrt((f2a + f2b) / 2))
    M = np.diag([1 / f_t, 1 / f_t, 1.0]) @ A
    MMt = M @ M.T
    s2 = np.trace(MMt) / 3
    err = float(np.abs(MMt / s2 - np.eye(3)).max())  # 0 for an exact rotation
    U, _, Vt = np.linalg.svd(M)
    Rrel = U @ Vt
    if np.linalg.det(Rrel) < 0:  # the homography's scale was negative
        Rrel = -Rrel
    R_t = Rrel @ ref.R
    C = ref.centre
    t = -R_t @ C
    return Camera(f_t, cv2.Rodrigues(R_t)[0].ravel(), t, (w, h)), err


class PtzMatcher:
    """ALIKED + LightGlue matching of frames against the calibrated reference frames (features cached), their
    approved extra references and, with keyframes=True, keyframes collected on the way."""

    def __init__(self, refs: dict[str, tuple[np.ndarray, Camera]], graphics: np.ndarray | None = None,
                 device: str = "cuda", extra: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
                 keyframes: bool = False, shared=None):
        """refs: {view id: (reference frame, camera)}; extra: [(view id, image, H calibrated reference px -> image
        px)]; keyframes: keep() adds keyframes (else it does nothing); shared: an object with the same models as .det
        (ALIKED, 2048 keypoints) and .lg (LightGlue), e.g. edge.shot_bank.Features, used instead of a second copy."""
        import kornia.feature as KF
        import torch

        self.torch = torch
        self.dev = torch.device(device)
        if shared is not None:  # one copy on the GPU (user request 2026-10-26)
            self.det, self.lg = shared.det, shared.lg
        else:
            self.det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=2048).to(self.dev).eval()
            self.lg = KF.LightGlue("aliked", mp=True).to(self.dev).eval()  # mixed precision (fp16 attention)
        self.gmask = graphics  # (h, w) at WORK size, 1 = graphics
        self.keyframes = keyframes
        self.cams = {vid: cam for vid, (_, cam) in refs.items()}
        self.bank = [_Ref(vid, self._feats(img), (img.shape[1], img.shape[0]), np.eye(3), 0)
                     for vid, (img, _) in refs.items()]
        for vid, img, H in extra or []:
            if vid in self.cams:
                self.bank.append(_Ref(vid, self._feats(img), (img.shape[1], img.shape[0]), np.asarray(H, float), 1))

    def n_keyframes(self) -> dict[str, int]:
        return {vid: sum(r.t > 0 for r in self.bank if r.view_id == vid) for vid in self.cams}

    def _feats(self, img: np.ndarray):
        small = cv2.resize(img, WORK, interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255)
        with self.torch.no_grad():
            f = self.det(t.to(self.dev))[0]
        kp, d = f.keypoints, f.descriptors
        if self.gmask is not None:
            gx = kp[:, 0].long().clamp(0, WORK[0] - 1).cpu().numpy()
            gy = kp[:, 1].long().clamp(0, WORK[1] - 1).cpu().numpy()
            keep = self.torch.from_numpy(self.gmask[gy, gx] == 0).to(self.dev)
            kp, d = kp[keep], d[keep]
        return kp, d

    def _match(self, a, b) -> tuple[np.ndarray, np.ndarray]:
        (ka, da), (kb, db) = a, b
        if len(ka) < 8 or len(kb) < 8:
            return np.zeros((0, 2)), np.zeros((0, 2))
        size = self.torch.tensor([[WORK[0], WORK[1]]], device=self.dev, dtype=self.torch.float32)
        with self.torch.no_grad():
            out = self.lg({"image0": {"keypoints": ka[None], "descriptors": da[None], "image_size": size},
                           "image1": {"keypoints": kb[None], "descriptors": db[None], "image_size": size}})
        m = out["matches"][0].cpu().numpy()
        return ka[m[:, 0]].cpu().numpy(), kb[m[:, 1]].cpu().numpy()

    def match(self, frame_bgr: np.ndarray, t: float | None = None) -> PtzMatch | None:
        """Best reference for this frame (most RANSAC inliers, a rotation + zoom of that camera), or None.
        t: the frame's time (s, default now): references that matched recently are tried first."""
        t = time.time() if t is None else t
        fh, fw = frame_bgr.shape[:2]
        f = self._feats(frame_bgr)
        S_frm_inv = np.diag([fw / WORK[0], fh / WORK[1], 1.0])
        best = None
        for ref in sorted(self.bank, key=lambda r: -r.used):
            pa, pb = self._match(ref.feats, f)
            if len(pa) < MIN_INLIERS:
                continue
            Hw, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            if Hw is None or int(inl.sum()) < MIN_INLIERS:
                continue
            n = int(inl.sum())
            rw, rh = ref.size
            H = S_frm_inv @ Hw @ np.diag([WORK[0] / rw, WORK[1] / rh, 1.0]) @ ref.H0  # calibrated reference -> frame
            cam = self.cams[ref.view_id]
            fcam, err = camera_from_homography(H, cam, (fw, fh))
            ok = fcam is not None and ptz_ok(n, err)
            zoom = (fcam.f / fw) / (cam.f / cam.size[0]) if fcam is not None else 0.0
            cand = PtzMatch(ref.view_id, n, H, fcam if ok else None, err, zoom, ref.depth, ref, f, (fw, fh))
            if best is None or (cand.cam is not None, n) > (best.cam is not None, best.inliers):
                best = cand
            if best.cam is not None and best.inliers >= STRONG:
                break
        if best is not None and best.cam is not None:
            best.ref.used = t
        return best

    def keep(self, m: PtzMatch, t: float) -> bool:
        """Keep a very good, verified match's features as one more reference of its camera (memory only)."""
        keys = [r for r in self.bank if r.view_id == m.view_id and r.t > 0]
        if (not self.keyframes or m.cam is None or m.ptz_err > KEY_GATE[0] or m.inliers < KEY_GATE[1]
                or m.inliers > KEY_NOVEL or m.depth >= MAX_DEPTH or any(abs(t - r.t) < KEY_EVERY_S for r in keys)):
            return False
        if len(keys) >= MAX_KEYS:
            self.bank.remove(min(keys, key=lambda r: r.t))
        self.bank.append(_Ref(m.view_id, m.feats, m.size, m.H, m.depth + 1, t, t))
        return True

    def forget(self, m: PtzMatch) -> None:
        """Drop the keyframe behind a match whose camera failed verification (references from disk stay)."""
        if m.ref is not None and m.ref.t > 0 and m.ref in self.bank:
            self.bank.remove(m.ref)
