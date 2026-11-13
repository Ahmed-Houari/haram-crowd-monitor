"""Step 2b: capture one clean full-res reference frame per chosen view.

For each view in config/views.json without a reference image, watch the live stream and
match frames against the view's survey thumbnail (source_thumb). Save the first frame at the
same framing (zoom within ±3 %, centered within 3 % of the width) to ref_image.
Allowed by the privacy rules: one reference frame per chosen view, gitignored.
The crowd collector does the same automatically (edge/ref_capture.py), so this script is
only needed for a standalone capture.

Run from the project root:  .venv/Scripts/python scripts/capture_refs.py --hours 12
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.frame_grabber import FrameGrabber  # noqa: E402
from edge.ref_capture import RefCapturer  # noqa: E402
from edge.view_matcher import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    capturer = RefCapturer(load_config())
    if not capturer.todo:
        print("All reference frames exist.")
        return
    print(f"Waiting for: {', '.join(capturer.todo)}  (up to {args.hours} h)")

    deadline = time.monotonic() + args.hours * 3600
    for frame in FrameGrabber.from_env(interval_s=args.interval).frames():
        saved, m, shift = capturer.offer(frame)
        line = f"{frame.captured_at:%H:%M:%S} UTC  "
        line += "no match" if m is None else f"{m.view_id:16s} inliers={m.inliers:4d} zoom={m.zoom:.3f} shift={shift:.3f}"
        print(line + (f"  -> SAVED {saved}" if saved else ""), flush=True)
        del frame
        if not capturer.todo:
            print("All reference frames captured.")
            break
        if time.monotonic() >= deadline:
            print(f"Timeout. Still missing: {', '.join(capturer.todo)}")
            break


if __name__ == "__main__":
    main()
