"""Step 2b: does a frame show one of the reference views, and which part of its zone is visible?

SIFT keypoints + ratio test + RANSAC homography between each reference frame and the new
frame. The best view (most inliers) is a match if the homography is strong and sane; it is
*accepted* if its zoom factor is within the view's zoom range (config/views.json zoom_min..zoom_max).

Metric (user decision 2026-10-24): density in the VISIBLE part of the zone. Every frame is mapped to its view's
PRIMARY reference (homographies chained through the matched reference, so all references of a view share one
coordinate system and one zone). The zone is intersected with what the frame shows (inside the frame, not under
broadcast graphics); a reading is people per 10,000 px of that visible part's area, measured in primary-reference
px (the same floor patch has the same area whatever the zoom or framing), plus the visible fraction of the zone.
A zone counts when >= MIN_VISIBLE of it is visible; "people in the zone" only means something when it is fully
visible (>= FULL_VISIBLE). Each view's zoom range (zoom_min..zoom_max) comes from the synthetic density zoom test
(edge/zoom_range.py, scripts/test_pipeline.py --synthetic): zoom-in was meant to go to ~2x, but the model finds more
people when they look bigger, so the ranges end where that drift reaches 10 % (1.16-1.69x); zoom-out stays strict.

Broadcast graphics (logos, banner, clock) are masked out of the keypoints and the counts.

A view may have several references of the same framing (day / evening / night, config "references", see
edge/references.py): a frame matches the view if it matches any approved one; each extra reference stores its
homography from the primary ("H_from_primary").
Views with "status": "pending" (new cameras waiting for review) are not counted.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "views.json"

WORK_WIDTH = 960  # match at half resolution: fast, still thousands of keypoints
# Second pass when the first finds nothing: in dense crowds most of the 4000 SIFT keypoints at 960 px sit on people
# (which never match another frame); at 640 px the building dominates. Real sample (2026-10-24, roof camera):
# 44 inliers at 960 px (missed) vs 103 at 640 px, same zoom.
FALLBACK_WIDTH = 640
SIFT_FEATURES = 4000
RATIO_TEST = 0.75
RANSAC_PX = 4.0  # at WORK_WIDTH
MIN_INLIERS = 50
ZOOM_SANE = (0.33, 3.0)  # anything outside is a bad homography, not a zoom
MIN_VISIBLE = 0.30  # a zone reading needs >= 30 % of the zone area (primary px) in view (user decision 2026-10-24)
FULL_VISIBLE = 0.99  # zone "fully visible" (<= 1 % outside the frame or under graphics): people-in-zone is meaningful


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(config: dict, path: Path = CONFIG_PATH) -> None:
    text = json.dumps(config, indent=2)
    text = re.sub(r"\[\s+(-?\d+),\s+(-?\d+)\s+\]", r"[\1, \2]", text)  # one [x, y] per point
    path.write_text(text + "\n", encoding="utf-8")


def zoom_range(view: dict) -> tuple[float, float]:
    """Accepted zoom vs the primary reference (zoom_min, zoom_max): set per view by the synthetic density zoom test,
    else 1 -+ zoom_tolerance."""
    tol = view.get("zoom_tolerance", 0.05)
    return float(view.get("zoom_min", 1 - tol)), float(view.get("zoom_max", 1 + tol))


def graphics_mask(config: dict, size: tuple[int, int]) -> np.ndarray:
    """uint8 mask (h, w), 1 = broadcast graphics. Polygons are in ref_size px, scaled to `size` (w, h)."""
    w, h = size
    rw, rh = config["ref_size"]
    mask = np.zeros((h, w), np.uint8)
    for poly in config.get("graphics_mask", []):
        pts = np.round(np.float32(poly) * [w / rw, h / rh]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def zones_in_ref_px(view: dict, config: dict) -> dict[str, list]:
    """Zones in full-res reference px.

    Zones drafted on the upscaled survey thumbnail (zones_base == "thumb") are moved onto the
    captured reference with the thumbnail->reference homography that capture_refs.py saves
    next to the reference image (<ref>.json). Returns {} while there is no reference yet.
    """
    if view.get("zones_base", "ref") == "ref" or not view["zones"]:
        return view["zones"]
    sidecar = (PROJECT_ROOT / view["ref_image"]).with_suffix(".json")
    if not sidecar.exists():
        return {}
    d = json.loads(sidecar.read_text(encoding="utf-8"))
    (tw, th), (rw, rh) = d["thumb_size"], config["ref_size"]
    H = np.array(d["H_thumb_to_ref"]) @ np.diag([tw / rw, th / rh, 1])
    return {name: np.round(cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), H).reshape(-1, 2)).astype(int).tolist()
            for name, p in view["zones"].items()}


# Zones must stay inside this part of the reference frame (x0, y0, x1, y1 as fractions) so they
# stay fully visible when the camera zooms in up to ~20 %; bottom stops above the banner (87 %).
SAFE_AREA = (0.10, 0.10, 0.90, 0.86)


def clip_polygon(poly: list, x0: float, y0: float, x1: float, y1: float) -> list[list[int]]:
    """Sutherland-Hodgman clip of any polygon against an axis-aligned rectangle."""
    pts = [tuple(map(float, p)) for p in poly]
    edges = [(0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)]  # (axis, limit, keep >=)
    for axis, lim, keep_ge in edges:
        if not pts:
            break
        inside = (lambda p: p[axis] >= lim) if keep_ge else (lambda p: p[axis] <= lim)
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur) != inside(prev):
                t = (lim - prev[axis]) / (cur[axis] - prev[axis])
                out.append(tuple(prev[k] + t * (cur[k] - prev[k]) for k in (0, 1)))
            if inside(cur):
                out.append(cur)
        pts = out
    clipped = [[round(x), round(y)] for x, y in pts]
    return [p for i, p in enumerate(clipped) if p != clipped[i - 1]]  # drop repeated points


def finalize_zones(view: dict, config: dict) -> bool:
    """Move thumbnail-drafted zones onto the captured reference, clip them to SAFE_AREA, and
    store them in reference px (zones_base = "ref"). Returns False if there is no reference yet."""
    if view.get("zones_base", "ref") == "ref":
        return True
    zones = zones_in_ref_px(view, config)
    if not zones and view["zones"]:
        return False
    img = cv2.imread(str(PROJECT_ROOT / view["ref_image"]))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = SAFE_AREA
    view["zones"] = {n: clip_polygon(p, x0 * w, y0 * h, x1 * w, y1 * h) for n, p in zones.items()}
    view["zones_base"] = "ref"
    return True


@dataclass
class ViewMatch:
    view_id: str
    H: np.ndarray  # 3x3, matched reference full-res px -> frame full-res px
    zoom: float  # vs the view's PRIMARY reference, at the frame centre; >1 = frame is zoomed in
    inliers: int
    accepted: bool  # zoom within the view's range (zoom_min..zoom_max)
    ref_id: str = "primary"  # which reference of the view matched
    H_primary: np.ndarray | None = None  # primary reference px -> frame px (H chained through the matched reference)


@dataclass
class _Ref:
    id: str
    size: tuple[int, int]  # full-res (w, h) of this reference image
    kp: np.ndarray  # (N, 2) keypoints at work res
    desc: np.ndarray
    zoom_range: tuple[float, float]  # accepted zoom vs the primary reference
    zones: dict = field(default_factory=dict)  # the view's zones in PRIMARY reference px (same for all its references)
    ref_id: str = "primary"
    H_p2r: np.ndarray = field(default_factory=lambda: np.eye(3))  # primary reference px -> this reference's px
    primary_size: tuple[int, int] | None = None
    zone_area: dict = field(default_factory=dict)  # zone polygon area (primary px): density normalisation
    zone_masks: dict = field(default_factory=dict)  # zone -> uint8 mask in primary px
    kp_lo: np.ndarray | None = None  # features at FALLBACK_WIDTH
    desc_lo: np.ndarray | None = None


def _zone_geometry(zones: dict, size: tuple[int, int]) -> tuple[dict, dict]:
    """(area per zone, uint8 mask per zone) in reference px."""
    areas, masks = {}, {}
    for name, pts in zones.items():
        poly = np.float32(pts).reshape(-1, 1, 2)
        areas[name] = float(cv2.contourArea(poly))
        masks[name] = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(masks[name], [np.round(poly).astype(np.int32)], 1)
    return areas, masks


class ViewMatcher:
    """Matches frames against reference images. Build with from_config() or from_images()."""

    def __init__(self, refs: dict[str, np.ndarray], config: dict, ranges: dict[str, tuple[float, float]] | None = None,
                 zones: dict[str, dict] | None = None, work_width: int = WORK_WIDTH, min_inliers: int = MIN_INLIERS,
                 fallback_width: int | None = FALLBACK_WIDTH):
        self.config = config
        self.work_width = work_width
        self.fallback_width = fallback_width if fallback_width and fallback_width < work_width else None
        self.min_inliers = min_inliers
        self.sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        self.flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
        self.refs: list[_Ref] = []
        for vid, img in refs.items():
            kp, desc = self._features(img)
            z = (zones or {}).get(vid, {})
            size = (img.shape[1], img.shape[0])
            area, masks = _zone_geometry(z, size)
            self.refs.append(_Ref(vid, size, kp, desc, (ranges or {}).get(vid, (0.8, 1.2)), z, "primary", np.eye(3),
                                  size, area, masks, *self._lo(img)))
            log.info("Reference %s: %d keypoints", vid, len(kp))

    @classmethod
    def from_config(cls, config: dict | None = None) -> "ViewMatcher":
        config = config or load_config()
        refs, rng, zones, extra = {}, {}, {}, []
        for v in config["views"]:
            if v.get("status") == "pending":
                continue  # new camera waiting for review: not counted
            path = PROJECT_ROOT / v["ref_image"]
            img = cv2.imread(str(path)) if path.exists() else None
            if img is None:
                log.warning("No reference image for %s (%s), skipping", v["id"], path)
                continue
            refs[v["id"]], rng[v["id"]], zones[v["id"]] = img, zoom_range(v), zones_in_ref_px(v, config)
            for r in v.get("references", []):
                rp = PROJECT_ROOT / r["image"]
                if r.get("status") == "approved" and rp.exists():
                    extra.append((v["id"], r, cv2.imread(str(rp))))
        m = cls(refs, config, rng, zones)
        for vid, r, img in extra:
            if img is None:
                continue
            H = np.array(r["H_from_primary"], np.float64) if r.get("H_from_primary") else m._chain_to_primary(vid, img)
            if H is None:
                log.warning("Reference %s / %s: no homography to the primary reference, skipped", vid, r["id"])
                continue
            m.add_ref(vid, r["id"], img, H)
        return m

    def add_ref(self, view_id: str, ref_id: str, img: np.ndarray, H_from_primary: np.ndarray) -> None:
        """Another reference (another lighting) of an existing view. Zones, areas and the zoom range stay the
        primary's: readings of all references are in primary px, so one view stays one series."""
        primary = next(r for r in self.refs if r.id == view_id and r.ref_id == "primary")
        kp, desc = self._features(img)
        self.refs.append(_Ref(view_id, (img.shape[1], img.shape[0]), kp, desc, primary.zoom_range, primary.zones,
                              ref_id, np.asarray(H_from_primary, np.float64), primary.primary_size, primary.zone_area,
                              primary.zone_masks, *self._lo(img)))
        log.info("Reference %s / %s: %d keypoints", view_id, ref_id, len(kp))

    def _chain_to_primary(self, view_id: str, img: np.ndarray) -> np.ndarray | None:
        """Primary px -> img px for an extra reference saved without its homography (captured by older code)."""
        primary = [r for r in self.refs if r.id == view_id and r.ref_id == "primary"]
        saved, self.refs = self.refs, primary
        try:
            m = self.match(img, min_inliers=30)
        finally:
            self.refs = saved
        return m.H if m is not None else None

    def _lo(self, img: np.ndarray) -> tuple:
        return self._features(img, self.fallback_width) if self.fallback_width else (None, None)

    def _features(self, img: np.ndarray, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        width = width or self.work_width
        size = (width, round(h * width / w))
        gray = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = (1 - graphics_mask(self.config, size)) * 255
        kp, desc = self.sift.detectAndCompute(gray, mask)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2)
        return pts, desc if desc is not None else np.zeros((0, 128), np.float32)

    def match(self, frame_bgr: np.ndarray, min_inliers: int | None = None) -> ViewMatch | None:
        """Best matching reference (any reference of any view), or None if nothing matches with >= min_inliers."""
        need = min_inliers or self.min_inliers
        best = self._match_at(frame_bgr, need, self.work_width, False)
        if self.fallback_width and (best is None or best.inliers < self.min_inliers):
            lo = self._match_at(frame_bgr, need, self.fallback_width, True)
            if lo is not None and (best is None or lo.inliers > best.inliers):
                best = lo
        return best

    def _match_at(self, frame_bgr: np.ndarray, need: int, width: int, low: bool) -> ViewMatch | None:
        fh, fw = frame_bgr.shape[:2]
        kp, desc = self._features(frame_bgr, width)
        if len(desc) < need:
            return None
        best: ViewMatch | None = None
        for ref in self.refs:
            rkp, rdesc = (ref.kp_lo, ref.desc_lo) if low else (ref.kp, ref.desc)
            if rdesc is None or len(rdesc) < need:
                continue
            pairs = self.flann.knnMatch(rdesc, desc, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
            if len(good) < need:
                continue
            src = rkp[[m.queryIdx for m in good]]
            dst = kp[[m.trainIdx for m in good]]
            Hw, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX * width / WORK_WIDTH)
            if Hw is None:
                continue
            n_in = int(inl.sum())
            if n_in < need or (best and n_in <= best.inliers):
                continue
            # work-res homography -> full-res (ref px -> frame px)
            sr = width / ref.size[0]
            sf = width / fw
            H = np.diag([1 / sf, 1 / sf, 1]) @ Hw @ np.diag([sr, sr, 1])
            if not _sane(H, ref.size):
                continue
            Hp = H @ ref.H_p2r
            zoom = zoom_at_frame_centre(Hp, ref.primary_size or ref.size, (fw, fh))
            if not (ZOOM_SANE[0] < zoom < ZOOM_SANE[1]):
                continue
            best = ViewMatch(ref.id, H, zoom, n_in, ref.zoom_range[0] <= zoom <= ref.zoom_range[1], ref.ref_id, Hp)
        return best

    def ref_of(self, match: ViewMatch) -> _Ref:
        return next(r for r in self.refs if r.id == match.view_id and r.ref_id == match.ref_id)

    def visibility(self, match: ViewMatch, frame_shape: tuple[int, ...]) -> dict:
        """Which part of each zone the frame shows: {zone: {"visible", "area", "zoom", "mask"}}, see visible_parts."""
        ref = self.ref_of(match)
        h, w = frame_shape[:2]
        return visible_parts(match.H_primary, (w, h), ref.primary_size or ref.size, ref.zone_masks, ref.zone_area,
                             graphics_mask(self.config, (w, h)))

    def zone_readings(self, match: ViewMatch, density: np.ndarray, frame_shape: tuple[int, ...],
                      vis: dict | None = None) -> dict:
        """{zone: {"count", "density", "visible", "zoom"}} for zones with >= MIN_VISIBLE of their area in view."""
        return part_readings(vis if vis is not None else self.visibility(match, frame_shape), density)


def visible_parts(H_primary: np.ndarray, frame_size: tuple[int, int], primary_size: tuple[int, int],
                  zone_masks: dict, zone_area: dict, gmask: np.ndarray) -> dict:
    """Per zone, the part a frame shows: inside the frame and not under broadcast graphics.

    {zone: {"visible": that part's share of the zone area (primary px), "area": its area in primary px
    (density normalisation: the same floor patch has the same area at any zoom), "zoom": zoom at its centroid
    (people size where they are counted), "mask": uint8 frame-px mask of the part, "mask_primary": the same part in
    primary px}}.
    """
    w, h = frame_size
    seen_f = (1 - gmask).astype(np.uint8)  # frame px that show the scene
    seen_p = cv2.warpPerspective(seen_f, np.linalg.inv(H_primary), primary_size, flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = {}
    for name, zm in zone_masks.items():
        part = zm & seen_p
        n_part, n_zone = int(part.sum()), int(zm.sum())
        frac = n_part / n_zone if n_zone else 0.0
        mo = cv2.moments(part, binaryImage=True)
        zoom = (local_zoom(H_primary, (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]), primary_size, frame_size)
                if mo["m00"] else zoom_at_frame_centre(H_primary, primary_size, frame_size))
        mask = cv2.warpPerspective(zm, H_primary, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0) & seen_f
        out[name] = {"visible": frac, "area": frac * zone_area.get(name, n_zone), "zoom": zoom, "mask": mask,
                     "mask_primary": part}
    return out


def part_readings(vis: dict, density: np.ndarray) -> dict:
    """{zone: {"count", "density", "visible", "zoom"}} for zones with >= MIN_VISIBLE in view.

    count = people in the visible part (the density map summed over it); density = count per 10,000 px of the visible
    part, in primary-reference px; visible = the part's share of the zone. count is the whole zone's count only
    when visible >= FULL_VISIBLE.
    """
    dh, dw = density.shape
    out = {}
    for name, v in vis.items():
        if v["visible"] < MIN_VISIBLE or v["area"] <= 0:
            continue
        count = float((density * _cell_weights(v["mask"], dw, dh)).sum())
        out[name] = {"count": round(count, 1), "density": round(count / v["area"] * 1e4, 3),
                     "visible": round(v["visible"], 3), "zoom": round(v["zoom"], 4)}
    return out


def _cell_weights(mask: np.ndarray, dw: int, dh: int) -> np.ndarray:
    """Fraction of each density cell covered by the mask (density cells are 8x8 frame px)."""
    h, w = mask.shape
    if dh * 8 <= h and dw * 8 <= w:
        mask = mask[: dh * 8, : dw * 8]
    return cv2.resize(mask.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA)


def zoom_factor(H: np.ndarray, ref_size: tuple[int, int], frame_size: tuple[int, int]) -> float:
    """Linear scale of a small square at the reference center, after warping, normalized so
    1.0 = same field of view as the reference (frame and reference may differ in resolution)."""
    return local_zoom(H, (ref_size[0] / 2, ref_size[1] / 2), ref_size, frame_size)


def zoom_at_frame_centre(H: np.ndarray, ref_size: tuple[int, int], frame_size: tuple[int, int]) -> float:
    """Zoom where the frame looks: local scale at the reference point shown in the frame centre (a zoomed-in shot
    may not show the reference centre at all)."""
    c = cv2.perspectiveTransform(np.float32([[[frame_size[0] / 2, frame_size[1] / 2]]]), np.linalg.inv(H))[0, 0]
    return local_zoom(H, (float(c[0]), float(c[1])), ref_size, frame_size)


def local_zoom(H: np.ndarray, at: tuple[float, float], ref_size: tuple[int, int], frame_size: tuple[int, int]) -> float:
    """Linear scale of a small square at `at` (reference px) after warping, normalized so 1.0 = same field of view
    as the reference (frame and reference may differ in resolution)."""
    cx, cy = at
    d = ref_size[0] * 0.05
    sq = np.float32([[cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(sq, H)
    area_ratio = abs(cv2.contourArea(warped)) / (2 * d) ** 2
    return float(np.sqrt(area_ratio) * ref_size[0] / frame_size[0])


def _sane(H: np.ndarray, ref_size: tuple[int, int]) -> bool:
    """Reject folded / wildly skewed homographies: warped reference corners must stay convex."""
    w, h = ref_size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)
    return bool(cv2.isContourConvex(warped.astype(np.float32)))
