"""Capture one clean full-res reference frame per view that does not have one yet.

Frames are matched against each missing view's survey thumbnail (source_thumb). The first
YouTube (1080p) frame at the same framing (zoom within ±3 %, centered within 3 % of the width)
is saved to the view's ref_image, plus a sidecar <ref>.json with the thumbnail->reference
homography (zones drafted on the thumbnail move with it, see view_matcher.finalize_zones).
Allowed by the privacy rules: one reference frame per chosen view, gitignored.
"""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np

from edge.frame_grabber import Frame
from edge.view_matcher import PROJECT_ROOT, ViewMatch, ViewMatcher

log = logging.getLogger(__name__)

MAX_ZOOM_DEV = 0.03
MAX_SHIFT = 0.03  # fraction of frame width


def center_shift(H: np.ndarray, ref_size: tuple[int, int], frame_size: tuple[int, int]) -> float:
    """Distance between the warped reference center and the frame center, as a fraction of frame width."""
    c = np.float32([[[ref_size[0] / 2, ref_size[1] / 2]]])
    x, y = cv2.perspectiveTransform(c, H)[0, 0]
    return float(np.hypot(x - frame_size[0] / 2, y - frame_size[1] / 2) / frame_size[0])


class RefCapturer:
    """Watches frames for views without a reference image and saves one when it shows up."""

    def __init__(self, config: dict):
        self.todo = {v["id"]: v for v in config["views"] if not (PROJECT_ROOT / v["ref_image"]).exists()}
        thumbs = {vid: cv2.imread(str(PROJECT_ROOT / v["source_thumb"])) for vid, v in self.todo.items()}
        self.thumb_size = {vid: (t.shape[1], t.shape[0]) for vid, t in thumbs.items()}
        # thumbnails are 480 px wide: match at that width, accept weaker matches (geometry check is strict)
        self.matcher = ViewMatcher(thumbs, config, work_width=480, min_inliers=30) if thumbs else None

    def offer(self, frame: Frame) -> tuple[str | None, ViewMatch | None, float | None]:
        """Try one frame. Returns (saved view id or None, match, center shift)."""
        if not self.todo or frame.source != "youtube":
            return None, None, None
        h, w = frame.image.shape[:2]
        m = self.matcher.match(frame.image)
        if m is None:
            return None, None, None
        shift = center_shift(m.H, self.thumb_size[m.view_id], (w, h))
        if abs(m.zoom - 1) > MAX_ZOOM_DEV or shift > MAX_SHIFT:
            return None, m, shift
        path = PROJECT_ROOT / self.todo[m.view_id]["ref_image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame.image)
        path.with_suffix(".json").write_text(json.dumps({
            "H_thumb_to_ref": m.H.tolist(), "thumb_size": list(self.thumb_size[m.view_id]),
            "captured_at": frame.captured_at.isoformat(timespec="seconds"),
            "zoom": round(m.zoom, 4), "shift": round(shift, 4), "inliers": m.inliers}, indent=2))
        log.warning("Captured reference for %s (zoom %.3f, shift %.3f, %d inliers)", m.view_id, m.zoom, shift, m.inliers)
        del self.todo[m.view_id]
        self.matcher.refs = [r for r in self.matcher.refs if r.id in self.todo]
        return m.view_id, m, shift
