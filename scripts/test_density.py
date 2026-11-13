"""Step 1 test: grab N frames from the live stream, run DM-Count, save side-by-side PNGs.

Saves data/samples/density_XX_<UTC time>.png = [frame | heatmap overlay] (explicit test
samples, allowed by the privacy rules). Prints count + inference time per frame, then
average inference time and peak VRAM.

Run from the project root:  .venv/Scripts/python scripts/test_density.py
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from edge.crowd_model import WEIGHTS_FILES, CrowdModel  # noqa: E402
from edge.frame_grabber import SAMPLES_DIR, FrameGrabber  # noqa: E402
from edge.heatmap import side_by_side  # noqa: E402

MB = 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between frames")
    parser.add_argument("--weights", default="qnrf", choices=list(WEIGHTS_FILES))
    parser.add_argument("--fp32", action="store_true", help="disable FP16 autocast")
    parser.add_argument("--allow-cpu", action="store_true", help="run even without CUDA (slow)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    elif not args.allow_cpu:
        raise SystemExit("CUDA not available. Install the cu130 PyTorch build or pass --allow-cpu.")

    model = CrowdModel(args.weights, half=not args.fp32)
    cuda = model.device.type == "cuda"
    if cuda:
        weights_mb = torch.cuda.memory_allocated() / MB
        torch.cuda.reset_peak_memory_stats()
    print(f"Model: DM-Count ({args.weights}), device={model.device}, fp16={model.half}")

    grabber = FrameGrabber.from_env(interval_s=args.interval)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    times: list[float] = []
    counts: list[float] = []
    for i, frame in enumerate(grabber.frames(args.count), 1):
        if i == 1:
            warm = model.predict(frame.image)  # first call: cuDNN picks kernels for this size
            print(f"warm-up: {warm.inference_ms:.0f} ms (not counted)")
        r = model.predict(frame.image)
        h, w = frame.image.shape[:2]
        label = f"{frame.captured_at:%Y-%m-%d %H:%M:%S} UTC   count={r.count:,.0f}   {r.inference_ms:.1f} ms   {w}x{h}"
        out = SAMPLES_DIR / f"density_{i:02d}_{frame.captured_at:%Y%m%dT%H%M%SZ}.png"
        cv2.imwrite(str(out), side_by_side(frame.image, r.density, label))
        print(f"[{i:2d}/{args.count}] {frame.captured_at:%H:%M:%S} UTC  count={r.count:9.1f}  "
              f"infer={r.inference_ms:6.1f} ms  frame={w}x{h}  -> {out.name}", flush=True)
        times.append(r.inference_ms)
        counts.append(r.count)
        del frame  # frame is never kept beyond this point

    print("\n=== summary ===")
    print(f"frames: {len(times)}   samples: {SAMPLES_DIR}")
    if times:
        print(f"inference ms: mean {statistics.mean(times):.1f}, median {statistics.median(times):.1f}, "
              f"min {min(times):.1f}, max {max(times):.1f}")
        print(f"count: mean {np.mean(counts):,.0f}, min {min(counts):,.0f}, max {max(counts):,.0f}")
    if cuda:
        print(f"VRAM: weights {weights_mb:.0f} MB, peak allocated {torch.cuda.max_memory_allocated() / MB:.0f} MB, "
              f"peak reserved {torch.cuda.max_memory_reserved() / MB:.0f} MB (+ CUDA context, see nvidia-smi)")


if __name__ == "__main__":
    main()
