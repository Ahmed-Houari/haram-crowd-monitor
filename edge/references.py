"""More references and more cameras: zone drafting, approval checks and review previews (user request 2026-10-24).

A camera ("view") may have several reference frames (day / evening / night) of the SAME framing. Each extra reference
stores its homography from the view's primary reference ("H_from_primary", chained through the reference it was
matched to), so every reading is measured in primary px on the primary zone and a view stays one series. A new
camera (a recurring wide framing found by scripts/discover_views.py) gets its own reference and a drafted zone.

Decisions (user, 2026-10-24): an extra reference whose only failed check is the zoom drift (cam1, cam5: known) is
approved with "quality": "low" instead of staying pending; a new camera is approved automatically when all checks
pass (the user reviews the previews later), and its zoom range then comes from the density zoom test
(edge/zoom_range.py).

Approval (all checks must pass, else the reference / camera stays "pending" for the user):
  1. strong homography match (inliers >= MIN_INLIERS_APPROVE, sane geometry)
  2. zone fully inside the safe margin (view_matcher.SAFE_AREA) and still large after clipping
  3. 0 % overlap with the broadcast graphics masks
  4. the Kaaba and walls outside the zone: no large people-free block inside it (the Kaaba and walls carry nobody),
     few long people-free architecture edges, a compact shape (solidity); thresholds calibrated on the three zones
     the user approved by hand (--calibrate). Inherited when the zone is mapped from an approved zone.
  5. a plausible count (people per 10,000 zone px within the range seen on the approved views, x0.25..x4)
  6. zoom stability: digital zoom 0.90..1.10 changes the zone count by <= 10 %
A preview (reference + zone + check results) is written to data/views/previews/<id>.jpg for every candidate.

  .venv/Scripts/python -m edge.references --calibrate      metrics of the hand-approved zones (sanity check)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from edge.view_matcher import (PROJECT_ROOT, SAFE_AREA, _cell_weights, clip_polygon, graphics_mask, load_config,
                               zones_in_ref_px)

log = logging.getLogger(__name__)

PREVIEW_DIR = PROJECT_ROOT / "data" / "views" / "previews"
MIN_INLIERS_APPROVE = 30
MIN_ZONE_FRAC = 0.04  # zone must cover >= 4 % of the frame after clipping
KAABA_MAX_OVERLAP = 0.002  # fraction of zone pixels that are Kaaba (Kiswa) pixels
WALL_MAX_FRAC = 0.004  # long people-free edge pixels per zone pixel; approved zones: 0-0.0012 (--calibrate)
DENSITY_RANGE = (2.0, 160.0)  # people per 10,000 zone px (approved views: ~10-60), wide on purpose
EMPTY_MAX = 0.05  # largest people-free block inside the zone (Kaaba, walls, empty floor); approved zones: 0-3.2 %,
#                  zones that include the Kaaba / walls: 10-32 %
MIN_SOLIDITY = 0.85  # zone area / convex hull area; approved zones 0.886-0.917, a spiky draft 0.816
ZOOM_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10)
MAX_ZOOM_DRIFT = 0.10


@dataclass
class Checks:
    inliers: int | None = None
    zone_frac: float | None = None
    in_safe_area: bool | None = None
    graphics_overlap: float | None = None
    kaaba_overlap: float | None = None
    empty_block: float | None = None
    solidity: float | None = None
    wall_frac: float | None = None
    count: float | None = None
    density: float | None = None
    zoom_drift: float | None = None
    failed: list = field(default_factory=list)
    soft: list = field(default_factory=list)  # failures that only lower an extra reference's quality (zoom drift)

    @property
    def passed(self) -> bool:
        return not self.failed


def kaaba_mask(img: np.ndarray, density_up: np.ndarray, config: dict | None = None) -> np.ndarray:
    """uint8 1 = the Kaaba: the largest compact block that is dark AND has nobody on it (a dark crowd has density;
    the black Kiswa has none). The gold belt is bridged by a closing."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = ((hsv[..., 2] < 80) & (hsv[..., 1] < 120) & (density_up < 0.003)).astype(np.uint8)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = SAFE_AREA  # the Kaaba is never hidden in a corner or under the broadcast graphics
    keep = np.zeros_like(dark)
    keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1
    dark &= keep & (1 - graphics_mask(config or load_config(), (w, h)))
    k = max(5, img.shape[1] // 128)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))  # bridge the gold belt
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))  # drop dark clothing specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = np.zeros_like(dark)
    min_area = 0.004 * img.shape[0] * img.shape[1]
    best, best_a = 0, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and 0.3 < w / max(h, 1) < 3.0 and a / (w * h) > 0.45 and a > best_a:
            best, best_a = i, a
    if best:
        out[lab == best] = 1
    return cv2.dilate(out, np.ones((15, 15), np.uint8))


def people_density(img: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """(density at 1/8 res, density upsampled + blurred to frame size)."""
    d = model.predict(img).density.astype(np.float32)
    h, w = img.shape[:2]
    up = cv2.GaussianBlur(cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
    return d, up


def wall_edges(img: np.ndarray, density_up: np.ndarray) -> np.ndarray:
    """uint8 1 = long edges where there are no people (arcades, walls, columns, building lines)."""
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 40, 40)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(10, 0.6 * med)), int(min(255, 1.3 * med)))
    edges[density_up > 0.004] = 0
    out = np.zeros_like(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) >= 0.04 * img.shape[1]:
            cv2.drawContours(out, [c], -1, 1, 1)
    return out


def zone_mask(shape: tuple, polys: list) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, np.float32)).astype(np.int32)], 1)
    return m


def zone_count(density: np.ndarray, mask: np.ndarray) -> float:
    return float((density * _cell_weights(mask, density.shape[1], density.shape[0])).sum())


def zoom_drift(img: np.ndarray, polys: list, model) -> float:
    """Max relative count change for digital zoom 0.90..1.10 about the centre (same floor patch)."""
    h, w = img.shape[:2]
    mg = 0.5 - 0.5 / max(ZOOM_SCALES)
    common = [np.float32(clip_polygon(p, mg * w, mg * h, (1 - mg) * w, (1 - mg) * h)) for p in polys]
    counts = []
    for s in ZOOM_SCALES:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        frame = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        pts = [cv2.transform(c.reshape(-1, 1, 2), M).reshape(-1, 2) for c in common if len(c) >= 3]
        counts.append(zone_count(model.predict(frame).density, zone_mask(frame.shape, pts)))
    c = np.array(counts)
    base = c[list(ZOOM_SCALES).index(1.0)]
    return float((c.max() - c.min()) / base) if base > 0 else 1.0


def draft_zone(img: np.ndarray, density_up: np.ndarray, config: dict) -> list[list[int]]:
    """A floor zone drafted like the hand-drawn ones: the main crowded floor area, Kaaba cut out, safe area only."""
    h, w = img.shape[:2]
    people = (density_up > 0.01).astype(np.uint8)
    people &= 1 - graphics_mask(config, (w, h))
    km = kaaba_mask(img, density_up)
    people &= 1 - km
    x0, y0, x1, y1 = SAFE_AREA
    safe = np.zeros_like(people)
    safe[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1
    people &= safe
    people = cv2.morphologyEx(people, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    people = cv2.morphologyEx(people, cv2.MORPH_OPEN, np.ones((61, 61), np.uint8))  # cut thin spikes into arcades
    n, lab, stats, _ = cv2.connectedComponentsWithStats(people, 8)
    if n < 2:
        return []
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8)
    comp &= 1 - km  # keep the Kaaba cut-out after closing
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.006 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return clip_polygon(poly.tolist(), x0 * w, y0 * h, x1 * w, y1 * h)


def check(img: np.ndarray, polys: list, model, config: dict, inliers: int | None, inherited: bool = False,
          full_visible: bool = True) -> Checks:
    """Run all approval checks on a reference image + its zone polygons (reference px).

    inherited=True: the zone was mapped by homography from a zone the user approved, so the Kaaba and walls are
    outside it by construction (those two checks are inherited, not re-detected).
    """
    ck = Checks(inliers=inliers)
    h, w = img.shape[:2]
    if not polys or any(len(p) < 3 for p in polys):
        ck.failed.append("no zone")
        return ck
    zm = zone_mask(img.shape, polys)
    area = int(zm.sum())
    ck.zone_frac = round(area / (h * w), 4)
    x0, y0, x1, y1 = SAFE_AREA
    pts = np.vstack([np.asarray(p, float) for p in polys])
    ck.in_safe_area = bool(pts[:, 0].min() >= x0 * w - 1 and pts[:, 0].max() <= x1 * w + 1 and
                           pts[:, 1].min() >= y0 * h - 1 and pts[:, 1].max() <= y1 * h + 1)
    ck.graphics_overlap = round(float((zm & graphics_mask(config, (w, h))).sum()) / max(area, 1), 4)
    d, up = people_density(img, model)
    if not inherited:
        km = kaaba_mask(img, up, config)
        ck.kaaba_overlap = round(float((zm & km).sum()) / max(area, 1), 4) if km.any() else None
        free = cv2.morphologyEx(((up < 0.002) & (zm > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(free, 8)
        ck.empty_block = round((st[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0) / max(area, 1), 4)
        hull = sum(cv2.contourArea(cv2.convexHull(np.float32(p).reshape(-1, 1, 2))) for p in polys)
        ck.solidity = round(sum(cv2.contourArea(np.float32(p).reshape(-1, 1, 2)) for p in polys) / max(hull, 1), 3)
        ck.wall_frac = round(float((zm & wall_edges(img, up)).sum()) / max(area, 1), 4)
    ck.count = round(zone_count(d, zm), 1)
    rw, rh = config["ref_size"]
    ck.density = round(ck.count / (area * (rw * rh) / (w * h)) * 1e4, 2)  # per 10k px in reference-size px
    ck.zoom_drift = round(zoom_drift(img, polys, model), 3)
    if inliers is not None and inliers < MIN_INLIERS_APPROVE:
        ck.failed.append(f"weak match ({inliers} inliers < {MIN_INLIERS_APPROVE})")
    if ck.zone_frac < MIN_ZONE_FRAC:
        ck.failed.append(f"zone too small ({100 * ck.zone_frac:.1f} % of the frame)")
    if not ck.in_safe_area or not full_visible:
        ck.failed.append("zone not fully inside the safe margin")
    if ck.graphics_overlap > 0:
        ck.failed.append(f"overlaps broadcast graphics ({100 * ck.graphics_overlap:.2f} %)")
    if not inherited:
        # the Kaaba (and walls) carry nobody: a large people-free block inside the zone means they are in it
        if ck.empty_block > EMPTY_MAX:
            ck.failed.append(f"people-free block inside the zone ({100 * ck.empty_block:.0f} %: Kaaba, wall or empty floor)")
        if ck.kaaba_overlap is not None and ck.kaaba_overlap > KAABA_MAX_OVERLAP:
            ck.failed.append(f"Kaaba inside the zone ({100 * ck.kaaba_overlap:.1f} %)")
        if ck.solidity < MIN_SOLIDITY:
            ck.failed.append(f"zone shape too ragged (solidity {ck.solidity} < {MIN_SOLIDITY}: reaches into arcades?)")
        if ck.wall_frac > WALL_MAX_FRAC:
            ck.failed.append(f"walls / building edges inside the zone ({ck.wall_frac:.4f} > {WALL_MAX_FRAC})")
    if not DENSITY_RANGE[0] <= ck.density <= DENSITY_RANGE[1]:
        ck.failed.append(f"implausible density {ck.density} per 10k px")
    if ck.zoom_drift > MAX_ZOOM_DRIFT:
        ck.failed.append(f"count drifts {100 * ck.zoom_drift:.0f} % over zoom 0.9-1.1 (> 10 %; people too small?)")
        ck.soft.append(ck.failed[-1])
    return ck


def preview(img: np.ndarray, polys: list, ck: Checks, title: str, out: Path, density_up: np.ndarray | None = None,
            soft_ok: bool = False) -> Path:
    """Reference with the zone (green = approved, amber = pending) and the check results, for the user's review."""
    vis = img.copy()
    if density_up is not None:  # Kaaba mask in magenta, so the reviewer sees what the check saw
        k = kaaba_mask(img, density_up, load_config()).astype(bool)
        vis[k] = (0.6 * vis[k] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    col = (80, 200, 80) if ck.passed else (40, 170, 245)
    for p in polys:
        cv2.polylines(vis, [np.round(np.asarray(p, np.float32)).astype(np.int32)], True, col, 3, cv2.LINE_AA)
    h, w = vis.shape[:2]
    x0, y0, x1, y1 = SAFE_AREA
    cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (200, 200, 200), 1)
    hard = [f for f in ck.failed if f not in ck.soft]
    state = "APPROVED" if ck.passed else "APPROVED, lower quality" if soft_ok and not hard else "PENDING"
    lines = [f"{title}: {state}",
             f"inliers {ck.inliers}  zone {100 * (ck.zone_frac or 0):.1f}%  graphics {ck.graphics_overlap}  kaaba {ck.kaaba_overlap}  "
             f"empty {ck.empty_block}  solidity {ck.solidity}  walls {ck.wall_frac}  count {ck.count}  density {ck.density}  "
             f"zoom drift {ck.zoom_drift}"]
    lines += [f"- {f}" for f in ck.failed]
    bar = np.zeros((28 * len(lines) + 12, w, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    small = cv2.resize(np.vstack([bar, vis]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


EXTRA_DIR = PROJECT_ROOT / "data" / "views" / "refs" / "extra"
VARIANT_MIN_INLIERS = 25  # weak match (lighting changed): candidate for a new reference of that view
VARIANT_MAX_PER_VIEW = 6
VARIANT_MAX_PER_PERIOD = 2
VARIANT_MIN_GAP_MIN = 60
VARIANT_MAX_ZOOM_DEV = 0.03  # same framing as the reference it matched
VARIANT_MAX_SHIFT = 0.03


def _rel(p: Path) -> str:
    """Path as stored in config/views.json: relative to the project, forward slashes."""
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def period_of(ts) -> str:
    """day 06-16, evening 16-22, night 22-06 (Makkah)."""
    from datetime import timedelta, timezone

    h = ts.astimezone(timezone(timedelta(hours=3))).hour
    return "day" if 6 <= h < 16 else "evening" if 16 <= h < 22 else "night"


class ReferenceManager:
    """Runs inside the crowd collector: adds references of new lightings, reviews newly captured cameras."""

    def __init__(self, model):
        self.model = model
        self.last_try: dict[str, object] = {}

    def consider_variant(self, frame, m, matcher) -> bool:
        """A weak match of an approved view at the same framing -> save as a candidate reference, check, approve.

        Returns True if config/views.json changed (the pipeline reloads)."""
        from datetime import timedelta

        from edge.ref_capture import center_shift
        from edge.view_matcher import save_config, zoom_factor

        ts, vid = frame.captured_at, m.view_id
        ref = matcher.ref_of(m)
        h, w = frame.image.shape[:2]
        if frame.source != "youtube" or abs(zoom_factor(m.H, ref.size, (w, h)) - 1) > VARIANT_MAX_ZOOM_DEV:
            return False  # same framing as the reference it matched
        last = self.last_try.get(vid)
        if last is not None and ts - last < timedelta(minutes=VARIANT_MIN_GAP_MIN):
            return False
        cfg = load_config()
        view = next((v for v in cfg["views"] if v["id"] == vid), None)
        if view is None or view.get("status") == "pending":
            return False
        refs = view.setdefault("references", [])
        per = period_of(ts)
        if len(refs) >= VARIANT_MAX_PER_VIEW or sum(r.get("period") == per for r in refs) >= VARIANT_MAX_PER_PERIOD:
            return False
        if center_shift(m.H, ref.size, (w, h)) > VARIANT_MAX_SHIFT:
            return False
        self.last_try[vid] = ts
        H_new = m.H @ ref.H_p2r  # primary px -> this frame (the new reference): chained through the matched one
        x0, y0, x1, y1 = SAFE_AREA
        polys = []
        for poly in ref.zones.values():  # the primary zone, in the new reference's px (for the checks + preview)
            mapped = cv2.perspectiveTransform(np.float32(poly).reshape(-1, 1, 2), H_new).reshape(-1, 2)
            polys.append(clip_polygon(mapped.tolist(), x0 * w, y0 * h, x1 * w, y1 * h))
        rid = f"{vid}@{per}-{ts:%Y%m%dT%H%MZ}"
        path = EXTRA_DIR / f"{vid}__{per}_{ts:%Y%m%dT%H%M%SZ}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame.image)  # allowed: reference frames of chosen views (user request 2026-10-24)
        # the zone stays the primary's (readings use visible parts), so the reference need not show all of it
        ck = check(frame.image, polys, self.model, cfg, m.inliers, inherited=True)
        low = not ck.passed and all(f in ck.soft for f in ck.failed)  # only the zoom drift failed: lower quality
        _, up = people_density(frame.image, self.model)
        pv = preview(frame.image, polys, ck, f"{rid} (from {ref.ref_id})", PREVIEW_DIR / f"{rid.replace('@', '__')}.jpg", up,
                     soft_ok=True)
        refs.append({"id": rid, "image": _rel(path), "period": per,
                     "status": "approved" if ck.passed or low else "pending", "from_ref": ref.ref_id,
                     **({"quality": "low", "quality_note": "approved despite: " + "; ".join(ck.soft)} if low else {}),
                     "H_from_primary": np.round(H_new, 8).tolist(),
                     "inliers": m.inliers, "zoom": round(m.zoom, 4), "captured_utc": ts.isoformat(timespec="seconds"),
                     "zones": {n: p for n, p in zip(ref.zones, polys)}, "checks": asdict(ck),
                     "preview": _rel(pv)})
        save_config(cfg)
        log.warning("New reference %s for %s: %s %s", rid, vid,
                    "APPROVED" if ck.passed else "APPROVED (lower quality)" if low else "pending", ck.failed or "")
        return True

    def review_new_views(self, cfg: dict) -> bool:
        """Pending cameras whose reference was just captured: draft the zone, run the checks, write a preview."""
        changed = False
        for v in cfg["views"]:
            if v.get("status") != "pending" or "auto_checks" in v:
                continue
            path = PROJECT_ROOT / v["ref_image"]
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            _, up = people_density(img, self.model)
            polys = [p for p in [draft_zone(img, up, cfg)] if len(p) >= 3]
            ck = check(img, polys, self.model, cfg, json.loads(path.with_suffix(".json").read_text()).get("inliers")
                       if path.with_suffix(".json").exists() else None)
            pv = preview(img, polys, ck, f"{v['id']} (new camera)", PREVIEW_DIR / f"{v['id']}.jpg", up)
            v["zones"] = {"mataf_ground": polys[0]} if polys else {}
            v["zones_base"] = "ref"
            v["auto_checks"] = asdict(ck)
            v["preview"] = _rel(pv)
            if ck.passed:
                v["status"] = "approved"  # user decision 2026-10-24: automatic, previews reviewed later
                try:
                    self._set_zoom_range(v, img, cfg)
                except Exception:
                    log.exception("zoom test of %s failed; it keeps 1 -+ zoom_tolerance", v["id"])
            log.warning("New camera %s reviewed: %s %s", v["id"], "APPROVED" if ck.passed else "pending", ck.failed or "")
            changed = True
        return changed

    def _set_zoom_range(self, v: dict, img: np.ndarray, cfg: dict) -> None:
        """Density zoom test on a newly approved camera's reference -> its zoom_min / zoom_max (~15 s of GPU)."""
        from edge import zoom_range

        res = zoom_range.run(img, v["zones"], self.model, cfg)["zones"].get("mataf_ground")
        if res is None:
            return
        rows = zoom_range.summarize(res["points"])
        v["zoom_min"], v["zoom_max"] = zoom_range.pick_range(rows, v["zoom_tolerance"])
        out = PROJECT_ROOT / "data" / "views" / "2d" / f"zoom_{v['id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rule": zoom_range.RANGE_RULE, "by_scale": rows, "points": res["points"]}, indent=1))
        v["zoom_test"] = _rel(out)
        log.warning("New camera %s: zoom range %.3f..%.3f", v["id"], v["zoom_min"], v["zoom_max"])


def calibrate() -> None:
    """Check metrics of the hand-approved primary zones (they define what 'normal' looks like)."""
    from edge.crowd_model import CrowdModel

    cfg = load_config()
    model = CrowdModel()
    for v in cfg["views"]:
        p = PROJECT_ROOT / v["ref_image"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        polys = list(zones_in_ref_px(v, cfg).values())
        ck = check(img, polys, model, cfg, None)
        out = preview(img, polys, ck, f"{v['id']} (hand-approved)", PREVIEW_DIR / f"calibrate_{v['id']}.jpg")
        print(v["id"], json.dumps(asdict(ck)), "->", out.relative_to(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true")
    if ap.parse_args().calibrate:
        calibrate()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
