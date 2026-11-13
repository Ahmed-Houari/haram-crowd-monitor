"""View-independent crowd reading for one frame: camera -> people per m^2 on the Mataf (research, 2026-10-24).

Two ways to the frame's camera:
  ptz    the frame matches one of the calibrated broadcast cameras (config/cameras.json, scripts/calibrate_cameras.py)
         and is a rotation + zoom of it (edge/ptz.py, ALIKED + LightGlue; directly, through an approved extra
         reference or through a keyframe from an earlier verified match): works by day and night, with or without
         the Kaaba in view. Verified when the Kaaba is in view: its projected box must lie on the Kaaba's mask
         (OWLv2 + SAM 2.1); without the Kaaba only a small rotation error passes. Only verified matches become
         keyframes;
  anchor otherwise, when the Kaaba is in view: a calibrated camera's position with the rotation and focal length
         fitted to the Kaaba's mask (kaaba_geom.fit_from_centre; 2026-10-25). Zoomed-in shots, where feature matching
         fails or gives a wrong rotation, get their camera this way; the position comes from the calibration, so the
         focal length / distance trade-off of a free fit does not arise;
  kaaba  otherwise a camera fitted to the Kaaba's silhouette (edge/kaaba_seg.py + edge/kaaba_geom.py), accepted only
         under a strict gate (the silhouette alone leaves the focal length / distance ambiguous for some views).
Then the crowd model's density map is traced to the floor (edge/m2_metric.py): people per m^2 in rings around the
Kaaba, raw and corrected for the model's head-size bias (g: scripts/zoom_bias.py on digital zoom, refitted on live
shadow runs by scripts/zoom_bias_live.py). Frames stay in memory.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from edge import kaaba_geom as kg
from edge.m2_metric import M2Reading, live_model_config, load_correction, measure, source_curves

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT / "config" / "cameras.json"
ZOOM_BIAS = ROOT / "models" / "zoom_bias.json"
FULL_FIT_EVERY_S = 90  # a full Kaaba-fit search (~3 s CPU, much more when the CPU is busy) at most this often;
#                        otherwise only warm starts from the cameras of recent good fits (~0.7 s)
IN_VIEW_MIN = 0.6  # a ptz camera is checked on the Kaaba when this share of its projected Kaaba is in view
ALIGN_MIN = (0.5, 0.7)  # ... and passes when the Kaaba mask covers this share of the projected Kaaba (2 m up) and
#                         this share of the mask lies inside the projected box (replay of the saved thumbnails)
NO_CHECK_MAX_ERR = 0.05  # without the Kaaba in view, a ptz camera needs a rotation error this small
ANCHOR_GATE = (1.5, 0.93)  # anchor camera: rms of the Kaaba's outline (px) at most, silhouette IoU at least (1080p;
#                            low cameras see the base hidden by people: IoU 0.94-0.95 on correct fits, 2026-10-25)
ANCHOR_SEPARATION = 0.5  # ... and at most this times the rms at the next-best calibrated position
ANCHOR_MIN_PX = 40  # the Kaaba's mask at least this wide for an anchor fit
ANCHOR_BUDGET_S = 6.0  # anchor fits for further camera positions only while the frame took less than this


def load_zoom_bias() -> tuple[float, float]:
    """The live head-size curve (edge/m2_metric.load_correction: config/m2.json "correction", else models/zoom_bias.json:
    6.5, 1.2 refit on live shadow runs, scripts/zoom_bias_live.py, 2026-10-24; digital-zoom fit 3.51, 1.79)."""
    return load_correction()[:2]


H0, KB = load_zoom_bias()
CAMERA_FACTOR = load_correction()[2]  # per calibrated camera, read at start


def g(h: np.ndarray) -> np.ndarray:
    """Share of people the crowd model finds at head width h px (-> 1 for big heads)."""
    return 1.0 - np.exp(-np.power(np.maximum(h, 0.1) / H0, KB))


SOURCE_CURVES = source_curves()  # read at start


def g_for(source: str | None):
    """g for frames from this source ("youtube", "sba", ...): its own curve when config/m2.json has one, else g."""
    if source not in SOURCE_CURVES:
        return g
    h0, k = SOURCE_CURVES[source]
    return lambda h: 1.0 - np.exp(-np.power(np.maximum(h, 0.1) / h0, k))


class Correction:
    """How a model's density becomes the metric: the head-size curve per source and the camera factor. Without a
    live_model config (edge/m2_metric.live_model_config) the original DM-Count's: g_for(source), CAMERA_FACTOR. With one
    (a fine-tuned model, correction built in): no curve and factor 1, except the sources it has its own curve for (SBA)."""

    def __init__(self, live_model: dict | None = None):
        self.live = live_model
        self.curves = {s: (float(v["h0"]), float(v["k"])) for s, v in ((live_model or {}).get("source") or {}).items()}

    def g(self, source: str | None):
        if self.live is None:
            return g_for(source)
        if source not in self.curves:
            return lambda h: np.ones_like(h)
        h0, k = self.curves[source]
        return lambda h: 1.0 - np.exp(-np.power(np.maximum(h, 0.1) / h0, k))

    def factor(self, view: str | None) -> float:
        return 1.0 if self.live is not None else CAMERA_FACTOR.get(view or "", 1.0)

    def has_curve(self, source: str) -> bool:
        return source in (self.curves if self.live is not None else SOURCE_CURVES)


def good_fit(r: kg.Fit | None) -> tuple[bool, str]:
    """Strict gate for a camera fitted to the Kaaba alone."""
    if r is None:
        return False, "no fit"
    d = float(np.linalg.norm(r.cam.centre[:2]))
    if r.rms_px > 1.5:
        return False, f"rms {r.rms_px:.1f} px"
    if r.iou < 0.96:
        return False, f"iou {r.iou:.2f}"
    if r.f_range[1] > 1.3 * max(r.f_range[0], 1):
        return False, "focal length ambiguous"
    if not (20 <= d <= 450 and -5 <= r.cam.centre[2] <= 200) or not 500 <= r.cam.f <= 20000:
        return False, "implausible camera"
    return True, "ok"


def kaaba_exclusion(cam: kg.Camera, shape: tuple[int, int]) -> np.ndarray:
    """Pixels covered by the Kaaba above head level (people in front of its base stay counted)."""
    m = np.zeros(shape, np.uint8)
    sil = kg.silhouette(cam, kg.box_corners(z0=kg.CUT_Z))
    if sil is not None:
        cv2.fillPoly(m, [np.round(sil[0]).astype(np.int32)], 1)
    return m


def kaaba_in_view(cam: kg.Camera, shape: tuple[int, int], gm: np.ndarray) -> float:
    """Share of the projected Kaaba (above head level) inside the frame and off the broadcast graphics."""
    sil = kg.silhouette(cam, kg.box_corners(z0=2.0))
    if sil is None:
        return 0.0
    total = cv2.contourArea(sil[0].astype(np.float32))
    if total < 1:
        return 0.0
    m = kaaba_exclusion(cam, shape)
    return float(((m > 0) & (gm == 0)).sum() / total)


def kaaba_alignment(cam: kg.Camera, mask: np.ndarray) -> tuple[float, float]:
    """(share of the projected Kaaba above head level covered by the Kaaba's mask, share of the mask inside the
    projected box): both near 1 when the camera is right, both low when it is a Kaaba-width off."""
    up = kaaba_exclusion(cam, mask.shape) > 0
    full = np.zeros(mask.shape, np.uint8)
    sil = kg.silhouette(cam)
    if sil is not None:
        cv2.fillPoly(full, [np.round(sil[0]).astype(np.int32)], 1)
    full = cv2.dilate(full, np.ones((9, 9), np.uint8)) > 0  # a few px of tolerance
    mk = mask > 0
    return float((up & mk).sum() / max(up.sum(), 1)), float((mk & full).sum() / max(mk.sum(), 1))


def kaaba_overlap(cam: kg.Camera, mask: np.ndarray) -> float:
    """IoU of the Kaaba's mask with the camera's projected Kaaba, the best of three bottoms (people hide the base up
    to ~2 m in low views; the roof sees it): 1 = the drawn Kaaba sits exactly on the real one (2026-10-27)."""
    mk = mask > 0
    best = 0.0
    for z0 in (kg.Z_BOTTOM, 1.0, 2.0):
        sil = kg.silhouette(cam, kg.box_corners(z0=z0))
        if sil is None:
            continue
        m = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(m, [np.round(sil[0]).astype(np.int32)], 1)
        m = m > 0
        best = max(best, float((m & mk).sum() / max((m | mk).sum(), 1)))
    return best


def whole(mask: np.ndarray, margin: int = 4) -> bool:
    """The Kaaba's mask does not touch the frame border. An anchor fit needs the whole Kaaba: on a mask cut by the
    border a camera at the wrong position can fit better than the right one (synthetic test 2026-10-25: cam1
    zoomed 4x, rms 0.29 px at cam0's position vs 2.61 px at its own)."""
    x, y, w, h = cv2.boundingRect(mask)
    return x >= margin and y >= margin and x + w <= mask.shape[1] - margin and y + h <= mask.shape[0] - margin


def load_cameras(config: dict) -> dict[str, tuple[np.ndarray, kg.Camera]]:
    """Calibrated reference cameras: {view id: (reference frame, camera)}. The reference frame is the view's
    (config/views.json) or, for a camera only the view-independent metric uses, the registry's own "ref_image"."""
    if not CAMERAS.exists():
        return {}
    reg = json.loads(CAMERAS.read_text())["cameras"]
    out = {}
    for vid, c in reg.items():
        v = next((v for v in config["views"] if v["id"] == vid), None)
        ref = c.get("ref_image") or (v["ref_image"] if v else None)
        img = cv2.imread(str(ROOT / ref)) if ref else None
        if img is not None:
            out[vid] = (img, kg.Camera(c["f"], np.array(c["rvec"]), np.array(c["tvec"]), tuple(c["size"])))
    return out


def load_extra_refs(config: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Approved extra references (another light) with their homography: [(view id, image, H reference -> image)]."""
    out = []
    for v in config["views"]:
        for r in v.get("references", []):
            p = ROOT / r["image"]
            img = cv2.imread(str(p)) if r.get("status") == "approved" and r.get("H_from_primary") and p.exists() else None
            if img is not None:
                out.append((v["id"], img, np.array(r["H_from_primary"], np.float64)))
    return out


@dataclass
class FrameCamera:
    path: str = ""  # "ptz" | "kaaba" | ""
    cam: kg.Camera | None = None
    kaaba_mask: np.ndarray | None = None
    info: dict = field(default_factory=dict)  # ptz view / inliers / error / zoom, Kaaba fit quality, timings


class M2Pipeline:
    def __init__(self, config: dict, model=None, use_kaaba: bool = True, keyframes: bool = True,
                 verify: bool = True, anchor: bool = True, anchor_gate: tuple[float, float] = ANCHOR_GATE,
                 feat=None, seg_on_demand: bool = False):
        """use_kaaba: fit a free camera to the Kaaba when nothing else gives one; verify: check ptz cameras on the
        Kaaba; anchor: fit rotation + focal length at the calibrated positions when the Kaaba is in view (all three
        need the OWLv2 + SAM 2.1 segmenter, ~1.2 GB VRAM; seg_on_demand: on the GPU only while it runs). feat: the
        caller's ALIKED + LightGlue for the ptz matcher (edge.ptz.PtzMatcher shared=)."""
        from edge.ptz import WORK, PtzMatcher
        from edge.view_matcher import graphics_mask

        self.config = config
        self.model = model
        self.use_kaaba, self.verify, self.anchor, self.anchor_gate = use_kaaba, verify, anchor, anchor_gate
        refs = load_cameras(config)
        self.centres = {vid: cam.centre for vid, (_, cam) in refs.items()}
        self.ptz = PtzMatcher(refs, graphics_mask(config, WORK), extra=load_extra_refs(config),
                              keyframes=keyframes, shared=feat) if refs else None
        self.seg = None
        if use_kaaba or verify or anchor:
            from edge.kaaba_seg import KaabaSegmenter

            self.seg = KaabaSegmenter(on_demand=seg_on_demand)
        self.recent: list[kg.Camera] = []
        self.last_full = 0.0

    def camera(self, img: np.ndarray, gm: np.ndarray, try_kaaba: bool = True, t: float | None = None) -> FrameCamera:
        """t: the frame's time (s), for the PTZ keyframes (default now)."""
        import time

        out = FrameCamera()
        t = time.time() if t is None else t
        t0 = time.perf_counter()
        pm = self.ptz.match(img, t) if self.ptz else None
        out.info["ms_ptz"] = round((time.perf_counter() - t0) * 1e3)
        cands = None
        if pm is not None:
            out.info.update(ptz_view=pm.view_id, ptz_inliers=pm.inliers, ptz_err=round(pm.ptz_err, 4),
                            ptz_zoom=round(pm.zoom, 3), ptz_depth=pm.depth, ptz_H=pm.H)
            if pm.cam is not None:
                check, cands, mask = self.check_on_kaaba(img, gm, pm.cam, out.info)
                out.info["ptz_check"] = check
                if check == "kaaba ok" or (check == "no check" and pm.ptz_err <= NO_CHECK_MAX_ERR):
                    out.path, out.cam, out.kaaba_mask = "ptz", pm.cam, mask
                    if check == "kaaba ok":
                        self.ptz.keep(pm, t)
                    return out
                if check == "kaaba off":
                    self.ptz.forget(pm)
        if self.anchor and self.seg and self.centres:
            ta = time.perf_counter()
            if cands is None:
                cands = self.seg.find(img, gm, k=2)
                if cands:
                    out.info.update({f"ms_{k}": v for k, v in cands[0].ms.items()})
            r, c, vid = self.fit_anchor(cands, pm, time.perf_counter() - t0, out.info)
            out.info["ms_anchor"] = round((time.perf_counter() - ta) * 1e3)
            if r is not None:
                out.path, out.cam, out.kaaba_mask = "anchor", r.cam, c.mask
                out.info["anchor_view"] = vid
                return out
        if not (try_kaaba and self.use_kaaba and self.seg):
            return out
        r, c = self.fit_kaaba(img, gm, out.info, cands)
        if r is not None:
            out.path, out.cam, out.kaaba_mask = "kaaba", r.cam, c.mask
        return out

    def fit_anchor(self, cands: list, pm, spent_s: float, info: dict) -> tuple[kg.Fit | None, object, str]:
        """(fit, Kaaba candidate, camera id) of the calibrated position whose rotation + focal length fit the Kaaba,
        or (None, None, ""). The ptz match's camera is tried first (starting from its rotation), the other positions
        while the frame took less than ANCHOR_BUDGET_S. Accepted: the best fit passes the gate (rms, IoU) and, when
        other positions were tried, fits at least 1 / ANCHOR_SEPARATION times better than the next one (a camera at
        another position gives rms >= 4 px on the replay; only tiny Kaabas fit everywhere)."""
        import time

        c = next((c for c in cands if c.dark >= 0.3 and np.ptp(np.nonzero(c.mask)[1]) >= ANCHOR_MIN_PX
                  and whole(c.mask)), None)
        if c is None:
            return None, None, ""
        order = sorted(self.centres, key=lambda v: v != (pm.view_id if pm is not None else None))
        t0 = time.perf_counter()
        fits = []
        for i, vid in enumerate(order):
            if i > 0 and spent_s + time.perf_counter() - t0 > ANCHOR_BUDGET_S:
                break
            init = pm.cam if pm is not None and pm.view_id == vid else None
            r = kg.fit_from_centre(c.mask, self.centres[vid], init=init)
            if r is not None:
                fits.append((r.rms_px, r, vid))
        if not fits:
            return None, None, ""
        fits.sort(key=lambda x: x[0])
        rms, r, vid = fits[0]
        info.update(anchor_try=vid, anchor_rms=round(rms, 2), anchor_iou=round(r.iou, 3), anchor_f=round(r.cam.f),
                    anchor_next=round(fits[1][0], 2) if len(fits) > 1 else "")
        ok = rms <= self.anchor_gate[0] and r.iou >= self.anchor_gate[1]
        if len(fits) > 1:
            ok = ok and rms <= ANCHOR_SEPARATION * fits[1][0]
        return (r, c, vid) if ok else (None, None, "")

    def check_on_kaaba(self, img: np.ndarray, gm: np.ndarray, cam: kg.Camera, info: dict) -> tuple[str, list | None,
                                                                                                   np.ndarray | None]:
        """("kaaba ok" | "kaaba off" | "no check", Kaaba candidates, the matching Kaaba mask): a ptz camera is checked
        on the Kaaba when its projected Kaaba is in view (projected box on the OWLv2 + SAM mask)."""
        if not (self.verify and self.seg) or kaaba_in_view(cam, img.shape[:2], gm) < IN_VIEW_MIN:
            return "no check", None, None
        def aligned(mask: np.ndarray) -> bool:
            rec, prec = kaaba_alignment(cam, mask)
            return rec >= ALIGN_MIN[0] and prec >= ALIGN_MIN[1]

        cands = self.seg.find(img, gm, k=2, enough=aligned)  # the 2nd box only if the 1st does not fit
        if cands:
            info.update({f"ms_{k}": v for k, v in cands[0].ms.items()})
        best = None
        for c in cands:
            rec, prec = kaaba_alignment(cam, c.mask)
            if best is None or min(rec / ALIGN_MIN[0], prec / ALIGN_MIN[1]) > best[0]:
                best = (min(rec / ALIGN_MIN[0], prec / ALIGN_MIN[1]), rec, prec, c)
        if best is None:
            return "no check", cands, None
        info.update(align_rec=round(best[1], 3), align_prec=round(best[2], 3))
        return ("kaaba ok", cands, best[3].mask) if best[0] >= 1 else ("kaaba off", cands, None)

    def fit_kaaba(self, img: np.ndarray, gm: np.ndarray, info: dict, cands: list | None = None
                  ) -> tuple[kg.Fit | None, object]:
        """Best strict-gated Kaaba fit (or None); fills info with the fit's quality. cands: Kaaba masks already found
        for this frame."""
        import time

        if cands is None:
            cands = self.seg.find(img, gm, k=2)
            if cands:
                info.update({f"ms_{k}": v for k, v in cands[0].ms.items()})
        t = time.perf_counter()
        best = None
        for c in cands:
            if c.dark < 0.3:  # the Kiswa is black
                continue
            r = kg.fit(c.mask, init=self.recent[-3:]) if self.recent else None
            ok, why = good_fit(r)
            if not ok and time.monotonic() - self.last_full > FULL_FIT_EVERY_S:
                self.last_full = time.monotonic()
                r2 = kg.fit(c.mask)
                ok2, why2 = good_fit(r2)
                if r is None or ok2 or (r2 is not None and r2.cost < r.cost):
                    r, ok, why = r2, ok2, why2
            key = (ok, -(r.rms_px if r else 99))
            if best is None or key > best[0]:
                best = (key, c, r, ok, why)
            if ok:
                break
        info["ms_fit"] = round((time.perf_counter() - t) * 1e3)
        if best is None:
            return None, None
        _, c, r, ok, why = best
        info.update(kaaba_score=round(c.score, 3), sam_iou=round(c.sam_iou, 3), dark=round(c.dark, 3), fit_ok=int(ok),
                    fit_why=why)
        if r is not None:
            info.update(f=round(r.cam.f), f_lo=round(r.f_range[0]), f_hi=round(r.f_range[1]),
                        cam=[round(float(x), 2) for x in r.cam.centre], rms_px=round(r.rms_px, 2), iou=round(r.iou, 3),
                        z_occ=round(r.z_occ, 2))
        if not ok:
            return None, None
        self.recent = [x for x in self.recent if np.linalg.norm(x.centre - r.cam.centre) > 5][-5:] + [r.cam]
        return r, c

    def reading(self, img: np.ndarray, fc: FrameCamera, gm: np.ndarray, density: np.ndarray | None = None,
                source: str | None = None, corr: Correction | None = None) -> M2Reading:
        dens = density if density is not None else self.model.predict(img).density
        self.last_density = dens  # the frame's density map, for the dashboard's heatmap (no extra model run)
        h, w = img.shape[:2]
        ex = np.maximum(fc.kaaba_mask if fc.kaaba_mask is not None else kaaba_exclusion(fc.cam, (h, w)), gm)
        view = fc.info.get("view") or {"ptz": fc.info.get("ptz_view"), "anchor": fc.info.get("anchor_view")}.get(fc.path)
        corr = corr or Correction()
        return measure(fc.cam, dens, ex, corr.g(source), corr.factor(view))
