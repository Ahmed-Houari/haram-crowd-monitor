"""Crowd density model: DM-Count (VGG19 backbone).

DM-Count: Wang et al., "Distribution Matching for Crowd Counting", NeurIPS 2020, MIT license
(github.com/cvlab-stonybrook/DM-Count). Official checkpoints mirrored as GitHub release
assets at github.com/tersekmatija/lwcc_weights (v0.1). Default: trained on UCF-QNRF
(very dense, high-resolution crowds), closest to the Mataf. Alternative: ShanghaiTech A.

Input: one BGR frame. Output: density map at 1/8 resolution, count = sum of the map.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
WEIGHTS_BASE_URL = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1"
WEIGHTS_FILES = {"qnrf": "DM-Count_QNRF.pth", "sha": "DM-Count_SHA.pth",
                 # fine-tuned on our own frames (scripts/finetune_dmcount.py, 2026-10-27): in shadow only (edge/shadow.py)
                 "qnrf_ft": "DM-Count_QNRF_ft.pth",
                 # v2: + daytime prayers (Dhuhr, Asr 27 Oct) in training; offline tests only, not in the shadow run
                 "qnrf_ft2": "DM-Count_QNRF_ft2.pth",
                 "qnrf_ft2b": "DM-Count_QNRF_ft2b.pth",  # v2b: v1 fine-tuned further on night + day (2 epochs)
                 "qnrf_ft3": "DM-Count_QNRF_ft3.pth",  # v3: v2 + rows labels carried behind the Kaaba, refitted poses
                 # v4 (2026-10-30): prayer rows only, live DVR + recordings, the two v4 test prayers held out
                 "qnrf_v4": "DM-Count_QNRF_v4.pth"}
# where a missing checkpoint comes from: the official ones from WEIGHTS_BASE_URL, ours from the project's GitHub
# release (DMCOUNT_FT_URL in .env, the asset's download URL; models/ is not in git)
WEIGHTS_URLS = {"qnrf_ft": "env:DMCOUNT_FT_URL"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# VGG19 conv layers without the last max-pool (output stride 16, upsampled to 8).
VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]


class DMCount(nn.Module):
    """Same layer names as the official DM-Count model, so checkpoints load as-is."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for v in VGG19_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.reg_layer(x)
        return self.density_layer(x)


@dataclass
class CrowdResult:
    density: np.ndarray  # float32, ~(H/8, W/8); each value = people in that cell
    count: float
    inference_ms: float  # upload + forward + download, GPU-synchronized


def weights_path(name: str) -> Path:
    """Local path of the checkpoint; downloads it to models/ on first use."""
    if name not in WEIGHTS_FILES:
        raise ValueError(f"unknown weights '{name}', choose from {list(WEIGHTS_FILES)}")
    path = MODELS_DIR / WEIGHTS_FILES[name]
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"{WEIGHTS_BASE_URL}/{WEIGHTS_FILES[name]}"
        if WEIGHTS_URLS.get(name, "").startswith("env:"):
            import os

            from dotenv import load_dotenv

            load_dotenv(MODELS_DIR.parent / ".env")
            url = os.getenv(WEIGHTS_URLS[name][4:]) or ""
            if not url:
                raise FileNotFoundError(f"{path} is missing and {WEIGHTS_URLS[name][4:]} is not set in .env")
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


class CrowdModel:
    """Loads DM-Count once and runs it on single frames."""

    def __init__(self, weights: str = "qnrf", device: str | None = None, half: bool = True, max_side: int = 2048):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = half and self.device.type == "cuda"
        self.max_side = max_side  # QNRF training images were capped at 2048 px
        self.weights = weights

        self.model = DMCount()
        state = torch.load(weights_path(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("model", state))  # mirror wraps weights in {"model": ...}
        self.model.eval().to(self.device)

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # frame size is constant -> pick fastest kernels

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> CrowdResult:
        """Run the model on one BGR uint8 frame."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        x = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = (x - self._mean) / self._std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
            mu = self.model(x)
        density = mu[0, 0].float().cpu().numpy()  # .cpu() waits for the GPU

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return CrowdResult(density=density, count=float(density.sum(dtype=np.float64)), inference_ms=inference_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DM-Count on an image file (quick check).")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", default="qnrf", choices=list(WEIGHTS_FILES))
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")
    model = CrowdModel(args.weights, half=not args.fp32)
    model.predict(frame)  # warm-up
    r = model.predict(frame)
    print(f"count={r.count:.1f}  inference={r.inference_ms:.1f} ms  density={r.density.shape}  device={model.device}")


if __name__ == "__main__":
    main()
