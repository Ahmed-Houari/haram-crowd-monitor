"""APGCC point-based crowd counter (Chen et al., ECCV 2024; MIT licence) - evaluation and calibration only.

Not part of the live pipeline. Used by scripts/eval_hajjv2.py (HAJJv2-CrowdCount benchmark) and scripts/zoom_calib.py
(self-calibration from zoomed-in frames). Counting only: the output is head points, no identity of any kind.

Code: the authors' decoder (models/apgcc_src/apgcc/models/Decoder.py + modules.py, github.com/AaronCIH/APGCC main,
read and reviewed 2026-10-25: plain PyTorch, no file or network access) is loaded under a private package name, so
its generic module names (models, util, ...) never shadow ours. The VGG16-BN encoder is torchvision's, built without
weights, because the authors' backbone loader calls torch.load without weights_only. Weights: the authors' SHHA
checkpoint (Google Drive 1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu, sha256 cd9aa0f6...cf90) in models/apgcc/SHHA_best.pth,
a plain state dict loaded with weights_only=True.
"""
from __future__ import annotations

import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "apgcc_src" / "apgcc" / "models"
WEIGHTS = ROOT / "models" / "apgcc" / "SHHA_best.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# configs/SHHA_test.yml (the checkpoint's training config)
DECODER_KWARGS = dict(num_classes=2, inner_planes=64, feat_layers=[3, 4], pos_dim=32, ultra_pe=True, learn_pe=False,
                      unfold=False, local=True, no_aspp=False, require_grad=True, out_type="Normal",
                      head_layers=[1024, 512, 256, 256], line=2, row=2, num_anchor_points=4, sync_bn=False,
                      AUX_EN=True, AUX_NUMBER=[2, 2], AUX_RANGE=[2, 8],
                      AUX_kwargs={"pos_coef": 1., "neg_coef": 1., "pos_loc": 0.0002, "neg_loc": 0.0002})
PAD = 16  # the anchor grid is ceil(H/8) x ceil(W/8) but the features are floor(H/8): pad to a multiple


def _decoder_module():
    name = "_apgcc_models"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(SRC)]  # package without running its __init__ (which pulls in training code)
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.Decoder")


class _Encoder(nn.Module):
    """APGCC's Base_VGG('vgg16_bn', last_pool=False): the same slices of torchvision's VGG16-BN features."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16_bn

        f = list(vgg16_bn(weights=None).features.children())
        self.body1, self.body2 = nn.Sequential(*f[:13]), nn.Sequential(*f[13:23])
        self.body3, self.body4 = nn.Sequential(*f[23:33]), nn.Sequential(*f[33:43])

    def forward(self, x):
        out = []
        for body in (self.body1, self.body2, self.body3, self.body4):
            x = body(x)
            out.append(x)
        return out


class _APGCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _decoder_module().IFI_Decoder_Model(in_planes=[128, 256, 512, 512], **DECODER_KWARGS)

    def forward(self, x):
        return self.decoder(x, self.encoder(x))


@dataclass
class PointResult:
    points: np.ndarray  # (N, 2) head points (x, y) in the input frame's pixels
    scores: np.ndarray  # (N,)
    count: int
    inference_ms: float


class PointCounter:
    """Loads APGCC once; .predict(frame_bgr) -> head points. CUDA only (the decoder hard-codes .cuda())."""

    def __init__(self, weights: Path = WEIGHTS, half: bool = True, max_side: int = 2560, threshold: float = 0.5):
        if not torch.cuda.is_available():
            raise RuntimeError("APGCC needs CUDA")
        self.device = torch.device("cuda")
        self.half, self.max_side, self.threshold = half, max_side, threshold  # 2560: the authors' test-time cap
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):  # the decoder prints its settings
            self.model = _APGCC()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray, scale: float = 1.0) -> PointResult:
        """scale > 1 upsamples the frame first (small heads); points come back in the input frame's pixels."""
        h, w = frame_bgr.shape[:2]
        s = min(scale, self.max_side / max(h, w))
        img = frame_bgr if s == 1.0 else cv2.resize(frame_bgr, (round(w * s), round(h * s)),
                                                   interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).to(self.device)
        x = (x.permute(2, 0, 1).unsqueeze(0).float().div_(255.0) - self._mean) / self._std
        hh, ww = x.shape[-2:]
        x = nn.functional.pad(x, (0, -ww % PAD, 0, -hh % PAD))  # zeros = the mean colour after normalising
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
            out = self.model(x)
        score = torch.softmax(out["pred_logits"].float(), -1)[0, :, 1]
        pts = out["pred_points"][0].float()
        keep = (score > self.threshold) & (pts[:, 0] < ww) & (pts[:, 1] < hh)  # drop points in the padding
        pts, score = pts[keep].cpu().numpy() / s, score[keep].cpu().numpy()
        ms = (time.perf_counter() - t0) * 1000.0
        return PointResult(points=pts, scores=score, count=len(pts), inference_ms=ms)
