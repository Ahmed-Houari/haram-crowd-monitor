"""Per-frame crowd pipeline: reference view match -> visible part of the zone -> crowd model -> per-zone density.

process(frame) returns a Reading only when the frame shows one of the reference views (zoom within STORE_ZOOM)
and at least MIN_VISIBLE (30 %) of a zone is visible (inside the frame, not under graphics); the reading is the
density in the visible part (people per 10k primary-reference px) + the raw count there + the visible fraction
(see edge/view_matcher.py). reading.accepted = strict match (>= 50 inliers) and zoom within the view's own range
(zoom_min..zoom_max). The crowd model runs only for those frames, so the GPU idles on close-ups, other cameras and
graphics.

Views without a reference image are captured automatically (edge/ref_capture.py); the config
is then finalized (zones moved onto the reference and clipped to the safe area) and reloaded,
so a new view (e.g. cam1) joins without a restart. reload_if_changed() also picks up config
edits made by other tools.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from edge.crowd_model import CrowdModel, CrowdResult
from edge.frame_grabber import Frame
from edge.heatmap import heat_scale
from edge.ref_capture import RefCapturer
from edge.references import VARIANT_MIN_INLIERS, ReferenceManager
from edge.view_matcher import (CONFIG_PATH, MIN_VISIBLE, PROJECT_ROOT, ViewMatch, ViewMatcher, finalize_zones,
                               load_config, save_config)

log = logging.getLogger(__name__)

# Readings are computed for zoom within STORE_ZOOM of the primary reference and stored per frame (crowd_frame),
# so the zoom check keeps running on live data. Only frames within the view's own range (config/views.json
# zoom_min..zoom_max, from the synthetic density zoom test) are "accepted" into the per-minute table.
STORE_ZOOM = (0.80, 2.40)


@dataclass
class Reading:
    ts: datetime  # UTC
    view_id: str
    source: str  # "youtube" (1080p) or "fallback" (720p)
    zoom: float
    inliers: int
    zones: dict = field(default_factory=dict)  # {zone: {"count", "density", "visible", "zoom"}} (visible part)
    accepted: bool = True  # strict match and zoom within the view's range
    ref_id: str = "primary"  # which reference of the view matched
    density_top: float | None = None  # live heatmap scale (edge.heatmap.heat_scale)
    density_max: float | None = None
    h_primary: list | None = None  # 3x3 homography primary reference px -> frame px (rounded)

    def to_dict(self) -> dict:
        """The Step 2 output format (+ zoom, source, visible part and homography)."""
        return {"ts": self.ts.isoformat(timespec="seconds"), "view_id": self.view_id,
                "zones": self.zones, "zoom": round(self.zoom, 4), "source": self.source, "accepted": self.accepted,
                "ref_id": self.ref_id, "density_top": self.density_top, "density_max": self.density_max,
                "h_primary": self.h_primary}


class CrowdPipeline:
    def __init__(self, model: CrowdModel | None = None, auto_capture: bool = True):
        self.model = model or CrowdModel()
        self.auto_capture = auto_capture
        self.refman = ReferenceManager(self.model) if auto_capture else None
        self._load()

    def _state_key(self) -> tuple:
        paths = [CONFIG_PATH] + [PROJECT_ROOT / v["ref_image"] for v in self.config["views"]]
        return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)

    def _load(self) -> None:
        self.config = load_config()
        changed = [v["id"] for v in self.config["views"]
                   if v.get("zones_base") == "thumb" and (PROJECT_ROOT / v["ref_image"]).exists()
                   and finalize_zones(v, self.config)]
        if changed:
            save_config(self.config)
            log.warning("Zones finalized on new reference(s): %s", ", ".join(changed))
        if self.refman:
            try:  # newly captured cameras: draft zone + checks + preview (stays pending unless all checks pass)
                if self.refman.review_new_views(self.config):
                    save_config(self.config)
            except Exception:
                log.exception("review of new cameras failed")
        self.matcher = ViewMatcher.from_config(self.config)
        self.capturer = RefCapturer(self.config) if self.auto_capture else None
        self._key = self._state_key()
        log.warning("Pipeline views: %s%s", ", ".join(f"{r.id}/{r.ref_id}" for r in self.matcher.refs) or "-",
                    f"; waiting for reference: {', '.join(self.capturer.todo)}" if self.capturer and self.capturer.todo else "")

    def reload_if_changed(self) -> bool:
        if self._state_key() != self._key:
            self._load()
            return True
        return False

    @property
    def view_ids(self) -> list[str]:
        return list(dict.fromkeys(r.id for r in self.matcher.refs))  # one view may have several references

    def process(self, frame: Frame) -> tuple[Reading | None, ViewMatch | None, CrowdResult | None]:
        """(reading or None, view match or None, crowd model result or None) for one frame.

        A reading is returned for zoom within STORE_ZOOM and >= MIN_VISIBLE of a zone in view; reading.accepted says
        if the zoom is within the view's range (only accepted readings count for the crowd time series)."""
        if self.capturer and self.capturer.todo:
            saved, _, _ = self.capturer.offer(frame)
            if saved:
                self._load()
        m = self.matcher.match(frame.image, min_inliers=VARIANT_MIN_INLIERS) if self.matcher.refs else None
        if m is not None and m.inliers < self.matcher.min_inliers:
            # weak match: the light changed (day / evening / night). Not counted; maybe a new reference
            if self.refman:
                try:
                    if self.refman.consider_variant(frame, m, self.matcher):
                        self._load()
                except Exception:
                    log.exception("new reference capture failed")
            m = None
        if m is None:
            return None, None, None
        vis = self.matcher.visibility(m, frame.image.shape)
        main = max(vis.values(), key=lambda v: v["visible"], default=None)  # zoom where people are counted
        if main is None or main["visible"] < MIN_VISIBLE or not STORE_ZOOM[0] <= main["zoom"] <= STORE_ZOOM[1]:
            return None, m, None
        result = self.model.predict(frame.image)
        zones = self.matcher.zone_readings(m, result.density, frame.image.shape, vis)
        if not zones:
            return None, m, result
        lo, hi = self.matcher.ref_of(m).zoom_range
        h, w = frame.image.shape[:2]
        hp = [[float(f"{x:.6g}") for x in row] for row in m.H_primary]
        return Reading(frame.captured_at, m.view_id, frame.source, main["zoom"], m.inliers, zones,
                       lo <= main["zoom"] <= hi, m.ref_id, round(heat_scale(result.density, (w, h)), 6),
                       round(float(result.density.max()), 6), hp), m, result


def warped_zones(matcher: ViewMatcher, m: ViewMatch) -> dict[str, np.ndarray]:
    """Zone polygons of the matched view (primary px), warped into frame px (for drawing; may reach outside the frame)."""
    import cv2

    ref = matcher.ref_of(m)
    return {name: cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), m.H_primary).reshape(-1, 2)
            for name, p in ref.zones.items()}


def annotate_reading(pipe: CrowdPipeline, frame: Frame, m: ViewMatch, result: CrowdResult, reading: Reading) -> np.ndarray:
    """Test-sample image: [frame with zones (green) + graphics mask (red) | heatmap overlay], counts in the bar."""
    import cv2

    from edge.heatmap import side_by_side
    from edge.view_matcher import graphics_mask

    img = frame.image.copy()
    h, w = img.shape[:2]
    cnts, _ = cv2.findContours(graphics_mask(pipe.config, (w, h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    for name, pts in warped_zones(pipe.matcher, m).items():
        cv2.polylines(img, [np.int32(pts)], True, (0, 255, 0), 4)
        z = reading.zones.get(name)
        cx, cy = np.clip(np.int32(pts.mean(0)), [160, 40], [w - 160, h - 40])
        cv2.putText(img, f"{name}: {z['density']:.1f} /10k px, {100 * z['visible']:.0f} % visible" if z else
                    f"{name}: < {100 * MIN_VISIBLE:.0f} % visible", (cx - 150, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4, cv2.LINE_AA)
    label = (f"{frame.captured_at:%Y-%m-%d %H:%M:%S} UTC  {m.view_id}  zoom {reading.zoom:.3f}  inliers {m.inliers}  "
             f"source {frame.source}  " + "  ".join(f"{z}: {v['density']:.1f}/10k px, {v['count']:,.0f} people in "
                                                    f"{100 * v['visible']:.0f} %" for z, v in reading.zones.items()))
    return side_by_side(img, result.density, label)
