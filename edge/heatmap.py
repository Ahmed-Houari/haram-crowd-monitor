"""Turn a density map into a colored heatmap / overlay for display."""
from __future__ import annotations

import cv2
import numpy as np


def colorize(density: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize density to `size` (w, h) and color it.

    Returns (BGR heatmap, alpha in [0, 1]). Scaled per frame to the 99th percentile so
    the densest areas are red; empty areas get alpha 0.
    """
    d = cv2.resize(density, size, interpolation=cv2.INTER_CUBIC)
    d = np.clip(d, 0, None)
    top = float(np.percentile(d, 99)) or 1e-6
    norm = np.clip(d / top, 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return color, norm


def heat_scale(density: np.ndarray, frame_size: tuple[int, int], out_width: int = 320) -> float:
    """The `top` that density_png() divides by (its colours are relative): stored per reading so they can be
    turned back into absolute densities later."""
    w, h = frame_size
    d = np.clip(cv2.resize(density, (out_width, round(h * out_width / w)), interpolation=cv2.INTER_CUBIC), 0, None)
    return float(np.percentile(d, 99))


def overlay(frame_bgr: np.ndarray, density: np.ndarray, max_alpha: float = 0.6) -> np.ndarray:
    """Blend the heatmap over the frame; denser = more opaque."""
    h, w = frame_bgr.shape[:2]
    color, norm = colorize(density, (w, h))
    a = (norm * max_alpha)[:, :, None]
    return (frame_bgr * (1 - a) + color * a).astype(np.uint8)


def density_png(density: np.ndarray, frame_size: tuple[int, int], zones: dict | None = None,
                out_width: int = 320) -> bytes:
    """Small heatmap PNG for the dashboard: density colors + zone outlines on a dark background.

    Contains NO video pixels (privacy: only numbers and this image leave the PC).
    frame_size = (w, h) of the frame the density came from; zones = {name: Nx2 points in frame px}.
    """
    w, h = frame_size
    size = (out_width, round(h * out_width / w))
    color, norm = colorize(density, size)
    a = (np.clip(norm * 1.5, 0, 1))[:, :, None]
    img = (np.full_like(color, 25) * (1 - a) + color * a).astype(np.uint8)
    for pts in (zones or {}).values():
        cv2.polylines(img, [np.int32(np.asarray(pts) * size[0] / w)], True, (0, 255, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def side_by_side(frame_bgr: np.ndarray, density: np.ndarray, label: str, panel_width: int = 1280) -> np.ndarray:
    """[frame | heatmap overlay] with a text bar on top."""
    h, w = frame_bgr.shape[:2]
    ph = round(h * panel_width / w)
    left = cv2.resize(frame_bgr, (panel_width, ph), interpolation=cv2.INTER_AREA)
    right = overlay(left, density)
    img = np.hstack([left, right])
    bar = np.zeros((40, img.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])
