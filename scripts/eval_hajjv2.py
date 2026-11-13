"""Evaluate crowd counters on HAJJv2-CrowdCount (human whole-frame counts at 1 fps on Hajj footage, 2026-10-25).

Benchmark: arXiv 2607.07322; annotations from its release (data/benchmarks/hajjv2/annotations, licence "TBC" there),
videos from the HAJJv2 release (data/benchmarks/hajjv2/videos, no licence file, citation requested). Both are
gitignored and stay on this PC; only numbers are written (data/benchmarks/hajjv2/results.csv). Frames stay in memory.

Counters: dm_qnrf = DM-Count QNRF, the live model with the live settings (fp16, max side 2048); dm_sha = DM-Count
ShanghaiTech-A; apgcc = APGCC SHHA (edge/apgcc_model.py). The live Mataf metric adds a head-size correction on top of
DM-Count that needs the camera geometry, which these videos lack, so the raw counts are compared here.
Density bands as in the paper: sparse < 100, medium 100-300, dense 300-1000; Testing_12 (~1700) reported apart.

Runs beside the live collector: GPU memory capped (--gpu-fraction), one frame at a time.

  .venv/Scripts/python -I scripts/eval_hajjv2.py [--split test train] [--models dm_qnrf dm_sha apgcc]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "data" / "benchmarks" / "hajjv2"
VIDEOS = BENCH / "videos" / "HAJJv2.Dataset"
BANDS = (("sparse <100", 0, 100), ("medium 100-300", 100, 300), ("dense 300-1000", 300, 1000))
EXTREME = "Testing_12"


def load_counters(names: list[str]) -> dict:
    out = {}
    for n in names:
        if n.startswith("dm_"):
            from edge.crowd_model import CrowdModel

            m = CrowdModel(weights=n[3:])
            out[n] = (lambda m: lambda f: _dm(m, f))(m)
        elif n.startswith("apgcc"):
            from edge.apgcc_model import PointCounter

            m = PointCounter(half=n != "apgcc_fp32")
            out[n] = (lambda m: lambda f: _ap(m, f))(m)
        else:
            raise SystemExit(f"unknown model {n}")
    return out


def _dm(m, f):
    r = m.predict(f)
    return r.count, r.inference_ms


def _ap(m, f):
    r = m.predict(f)
    return float(r.count), r.inference_ms


def frames(split: str):
    ann = BENCH / "annotations" / f"{split}_counts.csv"
    rows = list(csv.DictReader(open(ann, newline="")))
    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append((int(r["time_s"]), float(r["GT_count"])))
    for video, items in by_video.items():
        part, num = video.split("_")
        cap = cv2.VideoCapture(str(VIDEOS / ("Testing" if part == "Testing" else "Training") / "Videos" / f"{num}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for t, gt in items:
            k = min(round(t * fps), n - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            if ok:
                yield video, t, gt, f
            else:
                print(f"  {video} t={t}: could not read frame {k}/{n}")
        cap.release()


def metrics(gt: np.ndarray, pr: np.ndarray) -> str:
    if len(gt) == 0:
        return "-"
    e = pr - gt
    return (f"n {len(gt):3d}  MAE {np.abs(e).mean():7.1f}  RMSE {np.sqrt((e ** 2).mean()):7.1f}  "
            f"MAPE {100 * np.mean(np.abs(e) / gt):6.1f} %  bias {100 * (pr.sum() / gt.sum() - 1):+6.1f} %")


def report(rows: list[dict], models: list[str]) -> None:
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        groups = [("all (excl. Testing_12)", [r for r in rs if r["video"] != EXTREME])]
        groups += [(b, [r for r in rs if lo <= r["gt"] < hi and r["video"] != EXTREME]) for b, lo, hi in BANDS]
        groups += [(EXTREME, [r for r in rs if r["video"] == EXTREME])]
        print(f"\n== {split} ({len(rs)} frames)")
        for g, sel in groups:
            if not sel:
                continue
            gt = np.array([r["gt"] for r in sel])
            print(f"  {g}: GT mean {gt.mean():.0f}")
            for m in models:
                print(f"    {m:10s} {metrics(gt, np.array([r[m] for r in sel]))}")
        print("  per video (GT mean -> prediction means):")
        for v in sorted({r["video"] for r in rs}, key=lambda s: int(s.split("_")[1])):
            sel = [r for r in rs if r["video"] == v]
            gt = np.mean([r["gt"] for r in sel])
            print(f"    {v:12s} GT {gt:7.1f}  " + "  ".join(f"{m} {np.mean([r[m] for r in sel]):7.1f}" for m in models)
                  + f"  ({sel[0]['w']}x{sel[0]['h']})")
        for m in models:
            ms = [r[f"{m}_ms"] for r in rs]
            print(f"  {m}: {np.median(ms):.0f} ms median per frame")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", nargs="+", default=["test", "train"])
    ap.add_argument("--models", nargs="+", default=["dm_qnrf", "dm_sha", "apgcc"])
    ap.add_argument("--out", default=str(BENCH / "results.csv"))
    ap.add_argument("--report-only", action="store_true", help="re-print the report from --out")
    args = ap.parse_args()

    if args.report_only:
        rows = []
        for r in csv.DictReader(open(args.out, newline="")):
            rows.append({k: (v if k in ("split", "video") else float(v)) for k, v in r.items()})
        models = [k for k in rows[0] if k not in ("split", "video", "t", "gt", "w", "h") and not k.endswith("_ms")]
        report(rows, models)
        return

    import torch

    counters = load_counters(args.models)
    rows = []
    for split in args.split:
        for video, t, gt, f in frames(split):
            r = {"split": split, "video": video, "t": t, "gt": gt, "w": f.shape[1], "h": f.shape[0]}
            for m, fn in counters.items():
                c, ms = fn(f)
                r[m], r[f"{m}_ms"] = round(c, 1), round(ms, 1)
            rows.append(r)
            if len(rows) % 50 == 0:
                print(f"  {len(rows)} frames", flush=True)
    if torch.cuda.is_available():
        print(f"peak CUDA allocated {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, args.models)


if __name__ == "__main__":
    main()
