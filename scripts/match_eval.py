"""Day / night matching test (research, 2026-10-24): SIFT (the current matcher) vs learned matchers on the thumbnails.

Every labelled Mataf thumbnail (data/views/thumbs/labels.csv: kaaba, kaaba_partial, mataf) is matched against the three
NIGHT reference frames (cam0, cam1, cam5), all at 640 x 360:
  sift          OpenCV SIFT 4000 + ratio test 0.75 (as edge/view_matcher.py)
  disk+lg       DISK (2048 keypoints) + LightGlue      (kornia; Apache-2.0 code and weights)
  aliked+lg     ALIKED-n16 (2048) + LightGlue          (kornia; ALIKED BSD-3, LightGlue Apache-2.0)
then RANSAC homography (3 px). A match = the best reference with >= T inliers, a sane homography (convex warped
corners) and zoom 0.33-3. Night SIFT matches are the reference identities; day matches of the learned matchers are
checked by eye on data/views/discover/match_<method>.jpg (thumbnail | reference, with the warped outline).
Prints matches per period, method and threshold, and the time per pair.

  .venv/Scripts/python scripts/match_eval.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from edge.view_matcher import _sane, graphics_mask, load_config, zoom_factor  # noqa: E402

SIZE = (640, 360)
REFS = ["cam0_mataf_low", "cam1_mataf_wide", "cam5_roof"]


class Sift:
    name = "sift"

    def __init__(self, mask):
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
        self.mask = mask

    def feats(self, img):
        kp, d = self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.mask)
        return np.float32([k.pt for k in kp]), d

    def match(self, a, b):
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(da) < 4 or len(db) < 4:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pairs = self.flann.knnMatch(da, db, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        return ka[[m.queryIdx for m in good]], kb[[m.trainIdx for m in good]]


class Learned:
    def __init__(self, kind, mask):
        import kornia.feature as KF
        import torch

        self.torch, self.kind, self.name = torch, kind, f"{kind}+lg"
        self.dev = torch.device("cuda")
        if kind == "disk":
            self.det = KF.DISK.from_pretrained("depth").to(self.dev).eval()
        else:
            self.det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=2048).to(self.dev).eval()
        self.lg = KF.LightGlue(kind).to(self.dev).eval()
        self.mask = mask

    def feats(self, img):
        t = self.torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        with self.torch.no_grad():
            if self.kind == "disk":
                f = self.det(t, n=2048, pad_if_not_divisible=True)[0]
                kp, d = f.keypoints, f.descriptors
            else:
                f = self.det(t)[0]
                kp, d = f.keypoints, f.descriptors
        keep = self.mask[kp[:, 1].long().clamp(0, self.mask.shape[0] - 1).cpu().numpy(),
                         kp[:, 0].long().clamp(0, self.mask.shape[1] - 1).cpu().numpy()] > 0
        keep = self.torch.from_numpy(keep).to(self.dev)
        return kp[keep], d[keep]

    def match(self, a, b):
        (ka, da), (kb, db) = a, b
        if len(ka) < 4 or len(kb) < 4:
            return np.zeros((0, 2)), np.zeros((0, 2))
        size = self.torch.tensor([[SIZE[0], SIZE[1]]], device=self.dev, dtype=self.torch.float32)
        with self.torch.no_grad():
            out = self.lg({"image0": {"keypoints": ka[None], "descriptors": da[None], "image_size": size},
                           "image1": {"keypoints": kb[None], "descriptors": db[None], "image_size": size}})
        m = out["matches"][0].cpu().numpy()
        return ka[m[:, 0]].cpu().numpy(), kb[m[:, 1]].cpu().numpy()


def verify(pa, pb, ref_size):
    if len(pa) < 8:
        return 0, None, None
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return 0, None, None
    n = int(inl.sum())
    if not _sane(H, ref_size):
        return 0, None, None
    z = zoom_factor(H, ref_size, ref_size)
    if not 0.33 < z < 3.0:
        return 0, None, None
    return n, z, H


def main() -> None:
    cfg = load_config()
    gm = ((1 - graphics_mask(cfg, SIZE)) * 255).astype(np.uint8)
    refs = {r: cv2.resize(cv2.imread(str(ROOT / "data" / "views" / "refs" / f"{r}.png")), SIZE, interpolation=cv2.INTER_AREA)
            for r in REFS}
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "labels.csv"))
            if r["label"] in ("kaaba", "kaaba_partial", "mataf")]
    methods = [Sift(gm), Learned("disk", gm), Learned("aliked", gm)]
    results = defaultdict(dict)
    times = {}
    for meth in methods:
        rf = {r: meth.feats(im) for r, im in refs.items()}
        t0 = time.perf_counter()
        for row in rows:
            img = cv2.resize(cv2.imread(str(ROOT / "data" / "views" / "thumbs" / row["file"])), SIZE, interpolation=cv2.INTER_CUBIC)
            f = meth.feats(img)
            best = (0, None, None, None)
            for r in REFS:
                n, z, H = verify(*meth.match(rf[r], f), SIZE)
                if n > best[0]:
                    best = (n, r, z, H)
            results[row["file"]][meth.name] = best
        times[meth.name] = (time.perf_counter() - t0) / (len(rows) * len(REFS)) * 1e3
    for T in (15, 25, 50):
        print(f"\n== matches with >= {T} inliers (best of the 3 night references)")
        for period in ("night", "day"):
            sel = [r for r in rows if (r["night"] == "1") == (period == "night")]
            line = f"{period:5s} n={len(sel):3d} "
            for meth in methods:
                k = sum(results[r["file"]][meth.name][0] >= T for r in sel)
                line += f"| {meth.name} {k:3d} "
            sift_ids = {r["file"]: results[r["file"]]["sift"][1] for r in sel if results[r["file"]]["sift"][0] >= 25}
            agree = {m.name: sum(results[f][m.name][1] == ref for f, ref in sift_ids.items() if results[f][m.name][0] >= T)
                     for m in methods[1:]}
            print(line + f"| agree with SIFT>=25 identity: {agree} of {len(sift_ids)}")
    print("\nms per pair (features of the query + match + RANSAC):", {k: round(v, 1) for k, v in times.items()})
    # by-eye check sheets: day matches of the learned matchers (thumbnail | reference, warped outline in yellow)
    for meth in methods[1:]:
        tiles = []
        for r in rows:
            if r["night"] == "1":
                continue
            n, ref, z, H = results[r["file"]][meth.name]
            if n < 25:
                continue
            th = cv2.resize(cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"])), (320, 180), interpolation=cv2.INTER_AREA)
            outline = cv2.perspectiveTransform(np.float32([[0, 0], [SIZE[0], 0], [SIZE[0], SIZE[1]], [0, SIZE[1]]]).reshape(-1, 1, 2), H)
            cv2.polylines(th, [np.int32(outline.reshape(-1, 2) * 0.5)], True, (0, 255, 255), 1, cv2.LINE_AA)
            rf = cv2.resize(refs[ref], (320, 180), interpolation=cv2.INTER_AREA)
            tile = np.hstack([th, rf])
            cv2.putText(tile, f"{r['makkah_time'][11:]} {ref[:4]} n={n} z={z:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
            cv2.imwrite(str(ROOT / "data" / "views" / "discover" / f"match_{meth.name.replace('+', '_')}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{meth.name}: {len(tiles)} day matches drawn")


if __name__ == "__main__":
    main()
