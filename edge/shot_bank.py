"""Method b of the live shot service (2026-10-26): register a shot that matches no calibrated reference against the
"panorama bank" (scripts/vloc.py bank-save -> data/vloc/bank.npz): every saved frame of the three calibrated camera
places, with its camera made consistent within its place. A shot of a known place in any framing (zoomed in, panned,
a close-up with arcade, Kaaba or floor landmarks) matches some bank frame; the homography gives the shot's camera as a
rotation + zoom of that frame's camera about the place's fixed centre (edge/ptz.camera_from_homography). Accepted
under the gate validated on 2026-10-26 (scripts/vloc.py bank: 26 of 27 known frames at the right place, people/m^2
ratio median 0.99); about 1 in 10 lock onto the wrong arch bay, so the caller also checks the Kaaba when it is in view.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edge.kaaba_geom import Camera
from edge.ptz import camera_from_homography

WORK = (960, 540)  # the bank's feature resolution (scripts/vloc.py)
THUMB = (96, 54)
SHORTLIST = 10  # bank frames matched per shot, by thumbnail similarity
GATE = ((0.08, 30), (0.15, 50))  # (max rotation error, min inliers): either passes


@dataclass
class BankMatch:
    cam: Camera
    place: str
    image: str
    inliers: int
    err: float


class Features:
    """ALIKED (4096 keypoints at WORK) + LightGlue, as the bank was built (kornia)."""

    def __init__(self, n: int = 4096):
        import kornia.feature as KF
        import torch

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=n).to(self.dev).eval()
        self.lg = KF.LightGlue("aliked", mp=True).to(self.dev).eval()  # mixed precision (fp16 attention, 2026-10-26)

    def detect(self, img: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(img, WORK, interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255)
        with self.torch.no_grad():
            f = self.det(t.to(self.dev))[0]
        kp, d = f.keypoints.cpu().numpy(), f.descriptors.cpu().numpy()
        if mask is not None and len(kp):
            gx = np.clip(kp[:, 0].astype(int), 0, WORK[0] - 1)
            gy = np.clip(kp[:, 1].astype(int), 0, WORK[1] - 1)
            keep = mask[gy, gx] == 0
            kp, d = kp[keep], d[keep]
        return kp.astype(np.float32), d.astype(np.float32)

    def match(self, a, b) -> np.ndarray:
        (ka, da), (kb, db) = a, b
        if len(ka) < 8 or len(kb) < 8:
            return np.zeros((0, 2), int)
        T = self.torch
        size = T.tensor([[WORK[0], WORK[1]]], device=self.dev, dtype=T.float32)
        with T.no_grad():
            out = self.lg({"image0": {"keypoints": T.from_numpy(ka)[None].to(self.dev),
                                      "descriptors": T.from_numpy(da)[None].to(self.dev), "image_size": size},
                           "image1": {"keypoints": T.from_numpy(kb)[None].to(self.dev),
                                      "descriptors": T.from_numpy(np.asarray(db, np.float32))[None].to(self.dev),
                                      "image_size": size}})
        return out["matches"][0].cpu().numpy()


def thumb(img: np.ndarray) -> np.ndarray:
    t = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, THUMB, interpolation=cv2.INTER_AREA)
    t = t.astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


class Bank:
    def __init__(self, path: Path, feat: Features | None = None):
        z = np.load(path)
        self.names, self.places = list(z["names"]), list(z["places"])
        self.cams = [Camera(float(c[0]), c[1:4].copy(), c[4:7].copy(), (int(c[7]), int(c[8]))) for c in z["cams"]]
        self.thumbs = np.stack([thumb(t) for t in z["thumbs"]])
        self.feats = [(z[f"kp{i}"], z[f"d{i}"].astype(np.float32)) for i in range(len(self.names))]
        self.feat = feat or Features(n=2048)  # queries: 2048 keypoints against the bank's 4096 (GPU memory)

    def localise(self, img: np.ndarray, graphics_work: np.ndarray | None = None,
                 shortlist: int = SHORTLIST) -> BankMatch | None:
        """The best gated match of this frame (any framing of a known camera place), or None. shortlist: bank frames
        tried, most similar thumbnail first."""
        q = self.feat.detect(img, graphics_work)
        if len(q[0]) < 30:
            return None
        sim = (self.thumbs * thumb(img)[None]).mean((1, 2))
        h, w = img.shape[:2]
        best = None
        for i in np.argsort(-sim)[:shortlist]:
            m = self.feat.match(q, self.feats[i])
            if len(m) < GATE[0][1]:
                continue
            pq, pb = q[0][m[:, 0]].astype(float), self.feats[i][0][m[:, 1]].astype(float)
            H, inl = cv2.findHomography(pb, pq, cv2.RANSAC, 3.0)  # bank px -> query px (WORK)
            if H is None:
                continue
            n = int(inl.sum())
            ref = self.cams[i]
            S_b = np.diag([WORK[0] / ref.size[0], WORK[1] / ref.size[1], 1.0])
            S_q = np.diag([WORK[0] / w, WORK[1] / h, 1.0])
            cam, err = camera_from_homography(np.linalg.inv(S_q) @ H @ S_b, ref, (w, h))
            if cam is None or not any(err <= e and n >= k for e, k in GATE):
                continue
            if best is None or n > best.inliers:
                best = BankMatch(cam, self.places[i], self.names[i], n, round(float(err), 4))
        return best
