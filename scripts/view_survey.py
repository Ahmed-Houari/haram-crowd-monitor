"""Step 2a view survey: which camera views does the broadcast show, and how often?

Grabs one frame every 15 s for ~90 min, embeds each frame with ResNet18 (ImageNet weights,
feature map pooled to a 3x4 grid so layout/zoom matters), clusters the embeddings
(agglomerative, cosine distance) and reports each cluster's share of frames.

Privacy: frames stay in RAM. Only 3 small thumbnails per cluster are written, to
data/views/survey/ (allowed test samples). Embeddings (numbers, no images) are checkpointed
to embeddings.npz so a crash does not lose the run.

Run from the project root:  .venv/Scripts/python scripts/view_survey.py --minutes 90
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from torchvision.models import ResNet18_Weights, resnet18  # noqa: E402

from edge.frame_grabber import PROJECT_ROOT, FrameGrabber  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "views" / "survey"

# Rough boxes (x0, y0, x1, y1, normalized) over the broadcast graphics: top-left text,
# top-right channel logo, bottom banner + date + clock. Step 2c replaces this with a drawn mask.
GRAPHICS_BOXES = [(0.05, 0.06, 0.24, 0.16), (0.86, 0.04, 0.96, 0.23), (0.02, 0.87, 0.97, 1.0)]
EMBED_SIZE = (448, 256)  # (w, h), multiples of 32
GRID = (3, 4)  # pooled feature grid (rows, cols)
THUMB_SIZE = (480, 270)
SHEET_THUMB = (320, 180)
# Cosine distance for merging clusters. Step 1 samples: same fixed view 0.03-0.09 apart,
# zoom steps 0.08-0.12 per 15 s, different camera 0.37-0.51. 0.15 splits zoom states.
DEFAULT_THRESHOLD = 0.15
IMAGENET_MEAN_BGR = (104, 116, 124)


def mask_graphics(frame_bgr: np.ndarray) -> np.ndarray:
    """Copy of the frame with the broadcast graphics painted flat gray."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in GRAPHICS_BOXES:
        out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = IMAGENET_MEAN_BGR
    return out


class Embedder:
    """ResNet18 trunk -> feature map pooled to GRID -> L2-normalized vector (512*12 = 6144)."""

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(*list(net.children())[:-2]).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(mask_graphics(frame_bgr), EMBED_SIZE, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).to(self.device)
        x = (x.permute(2, 0, 1).unsqueeze(0).float() / 255.0 - self.mean) / self.std
        f = F.adaptive_avg_pool2d(self.trunk(x), GRID).flatten()
        return F.normalize(f, dim=0).cpu().numpy().astype(np.float32)


@dataclass
class Sample:
    ts: datetime
    emb: np.ndarray
    thumb_jpg: bytes  # in RAM only; written to disk only for the 3 picks per cluster


def cluster(embs: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster labels sorted so cluster 0 is the most frequent."""
    if len(embs) < 2:
        return np.zeros(len(embs), int)
    raw = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    ).fit_predict(embs)
    order = np.argsort(-np.bincount(raw), kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[raw]


def runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """(label, length) of consecutive runs = how long the broadcast stays on a view."""
    out: list[tuple[int, int]] = []
    for lab in labels:
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1] + 1)
        else:
            out.append((int(lab), 1))
    return out


def pick_thumbs(idx: np.ndarray, embs: np.ndarray) -> list[int]:
    """Medoid (most typical frame) + two frames spread in time."""
    centroid = embs[idx].mean(0)
    medoid = int(idx[np.argmax(embs[idx] @ centroid)])
    picks = [medoid]
    for k in (len(idx) // 3, 2 * len(idx) // 3):
        if int(idx[k]) not in picks:
            picks.append(int(idx[k]))
    return picks[:3]


def report(samples: list[Sample], threshold: float, interval_s: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    embs = np.stack([s.emb for s in samples])
    labels = cluster(embs, threshold)
    n = len(samples)
    all_runs = runs(labels)

    clusters = []
    sheet_rows = []
    for c in range(labels.max() + 1):
        idx = np.flatnonzero(labels == c)
        c_runs = [length for lab, length in all_runs if lab == c]
        thumbs = []
        for j, i in enumerate(pick_thumbs(idx, embs)):
            path = OUT_DIR / f"cluster_{c:02d}_{'abc'[j]}.jpg"
            path.write_bytes(samples[i].thumb_jpg)
            thumbs.append(path.name)
        info = {
            "cluster": c,
            "frames": int(len(idx)),
            "pct": round(100 * len(idx) / n, 1),
            "visits": len(c_runs),
            "median_stay_s": float(np.median(c_runs) * interval_s),
            "longest_stay_s": float(max(c_runs) * interval_s),
            "first_seen": samples[idx[0]].ts.isoformat(timespec="seconds"),
            "last_seen": samples[idx[-1]].ts.isoformat(timespec="seconds"),
            "thumbs": thumbs,
        }
        clusters.append(info)
        if c < 24:
            sheet_rows.append(_sheet_row(info, [OUT_DIR / t for t in thumbs]))

    (OUT_DIR / "clusters.json").write_text(json.dumps(
        {"frames": n, "threshold": threshold, "interval_s": interval_s, "clusters": clusters}, indent=2))
    with open(OUT_DIR / "timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cluster"])
        w.writerows((s.ts.isoformat(timespec="seconds"), int(lab)) for s, lab in zip(samples, labels))
    if sheet_rows:
        cv2.imwrite(str(OUT_DIR / "clusters_sheet.jpg"), np.vstack(sheet_rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"\n=== {n} frames, {len(clusters)} clusters (threshold {threshold}) ===")
    for c in clusters:
        print(f"C{c['cluster']:02d}  {c['pct']:5.1f}%  frames={c['frames']:4d}  visits={c['visits']:3d}  "
              f"median stay={c['median_stay_s'] / 60:4.1f} min  longest={c['longest_stay_s'] / 60:4.1f} min")
    print(f"Output: {OUT_DIR}  (clusters_sheet.jpg, clusters.json, timeline.csv)")


def _sheet_row(info: dict, thumb_paths: list[Path]) -> np.ndarray:
    tw, th = SHEET_THUMB
    label = np.full((th, 300, 3), 30, np.uint8)
    lines = [f"C{info['cluster']:02d}  {info['pct']}%", f"{info['frames']} frames",
             f"{info['visits']} visits", f"median stay {info['median_stay_s'] / 60:.1f} min"]
    for k, text in enumerate(lines):
        cv2.putText(label, text, (12, 40 + 38 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if k == 0 else 0.65,
                    (255, 255, 255), 2 if k == 0 else 1, cv2.LINE_AA)
    tiles = [label]
    for p in thumb_paths:
        tiles.append(cv2.resize(cv2.imread(str(p)), SHEET_THUMB, interpolation=cv2.INTER_AREA))
    while len(tiles) < 4:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    row = np.hstack(tiles)
    return np.vstack([row, np.full((6, row.shape[1], 3), 255, np.uint8)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2a: survey camera views of the live stream.")
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="cosine distance to merge")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    grabber = FrameGrabber.from_env(interval_s=args.interval)
    expected = int(args.minutes * 60 / args.interval)
    deadline = time.monotonic() + args.minutes * 60
    samples: list[Sample] = []
    print(f"Survey: {args.minutes} min, every {args.interval} s (~{expected} frames), device={embedder.device}")

    try:
        for frame in grabber.frames():
            emb = embedder(frame.image)
            thumb = cv2.resize(frame.image, THUMB_SIZE, interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            samples.append(Sample(frame.captured_at, emb, jpg.tobytes()))
            del frame, thumb  # full frame dropped here
            dist = 1.0 - float(emb @ samples[-2].emb) if len(samples) > 1 else 0.0
            print(f"[{len(samples):3d}/{expected}] {samples[-1].ts:%H:%M:%S} UTC  dist to previous {dist:.3f}"
                  f"{'  <- view change?' if dist > args.threshold else ''}", flush=True)
            if len(samples) % 20 == 0:
                np.savez(OUT_DIR / "embeddings.npz", emb=np.stack([s.emb for s in samples]),
                         ts=np.array([s.ts.isoformat() for s in samples]))
            if time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        print("Stopped early, clustering what we have.")

    if samples:
        np.savez(OUT_DIR / "embeddings.npz", emb=np.stack([s.emb for s in samples]),
                 ts=np.array([s.ts.isoformat() for s in samples]))
        report(samples, args.threshold, args.interval)


if __name__ == "__main__":
    main()
