"""Find the Kaaba in a frame: open-vocabulary box (OWLv2) -> precise mask (SAM 2.1 on a zoomed crop).

Research prototype (2026-10-24) for the view-independent metric (edge/kaaba_geom.py). Models (both Apache-2.0,
weights from the Hugging Face hub, downloaded once to the local cache):
  google/owlv2-base-patch16-ensemble  text-prompted boxes ("the Kaaba", "a black cube building")
  facebook/sam2.1-hiera-large         mask from the box + a point on the dark Kiswa, run on a crop around the box
                                      (the mask decoder works at 256x256, so a crop gives a much finer outline)
The geometry fit is the real check: boxes are tried best-first and a mask only counts if a Kaaba-sized box fits its
outline (edge.kaaba_geom.fit), so the TV logo or a dark crowd patch cannot pass. Frames stay in memory.
"""
from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

OWL_ID = "google/owlv2-base-patch16-ensemble"
SAM_ID = "facebook/sam2.1-hiera-large"
TEXTS = [["the Kaaba", "a black cube building"]]
MIN_SCORE = 0.12
CROP_MARGIN = 0.3  # crop = box enlarged by this on each side (square)
BOX_GROW = 0.15  # OWLv2 boxes are often tight (they cut the Kaaba's sides): SAM gets the box grown by this
OWL_SIDE = 960  # OWLv2 sees a 960 x 960 input: the frame is shrunk to this first (preprocessing a 1080p frame
#                 cost ~750 MB of process memory that stayed allocated; same model input, 2026-10-25)


@dataclass
class KaabaMask:
    mask: np.ndarray  # uint8 (h, w), 1 = Kaaba (the part people do not hide)
    box: tuple[float, float, float, float]  # detector box, frame px
    score: float  # detector score
    sam_iou: float  # SAM's predicted IoU of the chosen mask
    dark: float  # share of dark pixels in the mask (the Kiswa is black)
    ms: dict  # timings


def _on_device(fn):
    @functools.wraps(fn)
    def run(self, *args, **kwargs):
        with self.resident():
            return fn(self, *args, **kwargs)
    return run


class KaabaSegmenter:
    def __init__(self, device: str = "cuda", sam_id: str = SAM_ID, on_demand: bool = False):
        """on_demand: the models live in RAM and are on the GPU only while a call runs (the live shot service from
        2026-10-26; nobody since 2026-10-28, user: everything on the GPU by default, ~1.2 GB of VRAM held)."""
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection, Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device
        self.on_demand = on_demand and device != "cpu"
        home = "cpu" if self.on_demand else device
        self._depth = 0
        self.oproc = AutoProcessor.from_pretrained(OWL_ID)
        self.owl = Owlv2ForObjectDetection.from_pretrained(OWL_ID, torch_dtype=torch.float16).to(home).eval()
        self.sproc = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id).to(home).eval()

    @contextmanager
    def resident(self):
        """With on_demand: the models on the GPU for the block, then back in RAM and the GPU cache freed."""
        if not self.on_demand:
            yield
            return
        if self._depth == 0:
            self.torch.cuda.empty_cache()  # the caller's freed buffers first: the peak is the larger job, not the sum
            self.owl.to(self.device)
            self.sam.to(self.device)
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self.owl.to("cpu")
                self.sam.to("cpu")
                self.torch.cuda.empty_cache()

    @_on_device
    def boxes(self, frame_bgr: np.ndarray, graphics: np.ndarray | None = None, k: int = 3) -> list[tuple]:
        """Up to k candidate boxes (x0, y0, x1, y1, score), best first; boxes mostly on broadcast graphics dropped."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        f = min(1.0, OWL_SIDE / max(h, w))
        small = cv2.resize(frame_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else frame_bgr
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        inp = self.oproc(text=TEXTS, images=pil, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].half()
        with self.torch.no_grad():
            out = self.owl(**inp)
        side = max(small.shape[:2])  # OWLv2 pads to a square
        res = self.oproc.post_process_grounded_object_detection(out, threshold=MIN_SCORE, target_sizes=[(side, side)],
                                                                text_labels=TEXTS)[0]
        cands = []
        for s, b in sorted(zip(res["scores"].tolist(), res["boxes"].tolist()), reverse=True):
            b = [v / f for v in b]  # back to frame px
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
            if x1 - x0 < 8 or y1 - y0 < 8 or (x1 - x0) * (y1 - y0) > 0.6 * w * h:
                continue
            if graphics is not None and graphics[int(y0):int(y1), int(x0):int(x1)].mean() > 0.5:
                continue
            if any(_iou((x0, y0, x1, y1), c[:4]) > 0.5 for c in cands):
                continue
            cands.append((x0, y0, x1, y1, s))
            if len(cands) >= k:
                break
        return cands

    @_on_device
    def mask(self, frame_bgr: np.ndarray, box: tuple) -> tuple[np.ndarray, float, float]:
        """(full-frame uint8 mask, SAM predicted IoU, dark share) for one box, SAM run on a zoomed crop."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = box[:4]
        gx, gy = BOX_GROW * (x1 - x0) / 2, BOX_GROW * (y1 - y0) / 2
        x0, y0, x1, y1 = max(0.0, x0 - gx), max(0.0, y0 - gy), min(float(w), x1 + gx), min(float(h), y1 + gy)
        s = max(x1 - x0, y1 - y0) * (1 + 2 * CROP_MARGIN)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        X0, Y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        X1, Y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
        crop = frame_bgr[Y0:Y1, X0:X1]
        b = [x0 - X0, y0 - Y0, x1 - X0, y1 - Y0]
        v = cv2.blur(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2], (9, 9))
        yb0, yb1 = int(b[1] + 0.3 * (b[3] - b[1])), int(b[1] + 0.6 * (b[3] - b[1]))
        xb0, xb1 = int(b[0] + 0.3 * (b[2] - b[0])), int(b[0] + 0.7 * (b[2] - b[0]))
        sub = v[yb0:max(yb1, yb0 + 1), xb0:max(xb1, xb0 + 1)]
        py, px = np.unravel_index(int(sub.argmin()), sub.shape)
        pt = [float(xb0 + px), float(yb0 + py)]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        sin = self.sproc(images=pil, input_boxes=[[b]], input_points=[[[pt]]], input_labels=[[[1]]],
                         return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            so = self.sam(**sin, multimask_output=True)
        masks = self.sproc.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0][0].numpy()
        ious = so.iou_scores[0, 0].cpu().numpy()
        vv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
        best = None
        box_area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8)
            if not m.any():
                continue
            dark = float((vv[m > 0] < 90).mean())
            score = ious[i] + 0.3 * dark + 0.2 * min(m.sum() / box_area, 1.0)
            if best is None or score > best[0]:
                best = (score, m, float(ious[i]), dark)
        full = np.zeros((h, w), np.uint8)
        if best is None:
            return full, 0.0, 0.0
        full[Y0:Y1, X0:X1] = best[1]
        n, lab, st, _ = cv2.connectedComponentsWithStats(full, 8)  # keep the largest piece
        if n > 2:
            full = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        return full, best[2], best[3]

    @_on_device
    def find(self, frame_bgr: np.ndarray, graphics: np.ndarray | None = None, k: int = 3,
             enough=None) -> list[KaabaMask]:
        """Masks for the candidate boxes, best box first (the caller fits the geometry to pick the real one).
        enough(mask) -> True stops after that mask (saves the remaining SAM runs)."""
        t0 = time.perf_counter()
        cands = self.boxes(frame_bgr, graphics, k)
        t1 = time.perf_counter()
        out = []
        for c in cands:
            m, iou, dark = self.mask(frame_bgr, c)
            if m.sum() < 60:
                continue
            out.append(KaabaMask(m, tuple(c[:4]), float(c[4]), iou, dark, {}))
            if enough is not None and enough(m):
                break
        t2 = time.perf_counter()
        for o in out:
            o.ms = {"detect": round((t1 - t0) * 1e3), "sam": round((t2 - t1) * 1e3 / max(len(out), 1))}
        return out


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)
