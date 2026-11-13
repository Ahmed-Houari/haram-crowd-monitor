"""Live analysis of whatever shot the broadcast shows (2026-10-26, user request): every frame, as often as the GPU
allows, people per m^2 by the first method that works, and the method is kept with the reading.

  a "calibrated"   the frame is a known calibrated camera (edge/m2_pipeline.M2Pipeline: a reference match checked on
                   the Kaaba, or the Kaaba fit at a calibrated camera position);
  b "registered"   otherwise the frame is registered against the panorama bank (edge/shot_bank.py): any framing of a
                   known camera place; the Kaaba, when in view, must sit where the camera puts it (it is the only black
                   object of its size: a cheap check instead of the segmenter);
  - "tracked"      a frame of the same shot as the previous one: its camera follows from the previous camera through
                   the frame-to-frame homography; a full check again every REVERIFY_S;
  c "approximate"  no camera: DM-Count's count over the crowd's part of the frame, with the floor area from the head
                   size the model's own scale response gives (count at half size vs full size, inverted through the
                   digital-zoom curve) and an assumed viewing angle; not placed on the Mataf floor.
With a camera (a, b, tracked) the reading is the m2 metric's (edge/m2_metric.measure, with the live correction), and
when the frame shows prayer rows (edge/prayer_rows.py: the counting model reads them as empty floor at small head
sizes) the rows give the reading instead. Frames stay in memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from edge import kaaba_geom as kg
from edge import prayer_rows
from edge.m2_metric import ALL, HEAD_M, M2Reading, cell_floor, wall_distance
from edge.m2_pipeline import (IN_VIEW_MIN, Correction, FrameCamera, M2Pipeline, kaaba_exclusion, kaaba_in_view,
                              kaaba_overlap, whole)
from edge.ptz import camera_from_homography
from edge.shot_bank import WORK, Bank, Features, thumb

# Kaaba gate (user decision 2026-10-27): in a full search (a cut, or every REVERIFY_S) a camera whose Kaaba is in view
# is fitted to the Kaaba's mask at its own position (rotation + focal length) and kept only if the drawn Kaaba then
# overlaps the real one at least GATE_OVERLAP (IoU); frames in between are tracked from it.
# Live only after scripts/gate_check.py passes (coverage drops < ~20 %, overlap >= 0.8): KAABA_GATE.
KAABA_GATE = True  # scripts/gate_check.py 2026-10-27: coverage +0 %, overlap median 0.83 -> 0.89
REVERIFY_S = 20.0 if KAABA_GATE else 10.0  # a tracked shot gets a full check this often (10 s without the gate)
GATE_OVERLAP = 0.8
KAABA_FIT_BUDGET_S = 1.5
TRACK_MAX_S = 120.0  # a shot is tracked at most this long without a full check that finds its camera
SAME_SHOT = 0.75  # thumbnail correlation above which two frames are the same shot
ROWS_HOLD_S, ROWS_HOLD_MIN = 30.0, 0.1  # rows seen by this camera this recently: threshold 0.1 (frames without rows
#                                         in prayer windows: <= 0.17 at the roof before the rows were complete)
TRACK_GATE = (0.03, 80)  # frame-to-frame: max rotation error, min inliers
KAABA_IN_VIEW = 0.4  # share of the projected Kaaba (2 m up) inside the frame for the darkness check
KAABA_DARK = 0.55  # the projected Kaaba must be at most this bright relative to the floor around it
# method c
APPROX_MIN_PEOPLE = 30.0  # a crowd in view, else no reading
APPROX_BANDS = 4  # horizontal bands, each with its own head width
APPROX_BIAS = 1.7  # measured on 12 frames with a known camera (2026-10-26): median c / calibrated 1.7; the spread
#                    after this stays about x2 either way (approximate, never placed on the floor)
SIN_DOWN = 0.35  # assumed sine of the viewing angle below horizontal (calibrated cameras: 0.21-0.50)
DIGITAL_CURVE = (3.51, 1.79)  # models/zoom_bias.json "synthetic": the model's count vs head size under digital zoom


@dataclass
class ShotReading:
    ts: datetime
    method: str  # calibrated | registered | approximate | none
    how: str = ""  # ptz, ptz/k1, anchor, bank, tracked, ... ("+rows" when the prayer rows gave the reading)
    camera: str | None = None
    cam: kg.Camera | None = None
    reading: M2Reading | None = None
    rows: prayer_rows.RowsReading | None = None
    people_m2: float | None = None
    approx: dict | None = None
    density: np.ndarray | None = None  # DM-Count's map (people per 8 x 8 px cell)
    heat: np.ndarray | None = None  # the map for the dashboard (synthetic when the rows gave the reading)
    est: np.ndarray | None = None  # per density cell: estimated by the fixes (edge/m2_fill.LiveFixes), not measured
    ms: dict = field(default_factory=dict)


def free_gpu_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def kaaba_dark(gray: np.ndarray, cam: kg.Camera) -> bool | None:
    """True / False when the projected Kaaba is in view and is / is not the dark block it should be; None when not in
    view enough to tell."""
    h, w = gray.shape[:2]
    sil = kg.silhouette(cam, kg.box_corners(z0=2.0))
    if sil is None:
        return None
    poly = np.round(sil[0]).astype(np.int32)
    full = cv2.contourArea(poly.astype(np.float32))
    if full < 400:
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    if m.sum() < KAABA_IN_VIEW * full:
        return None
    inner = cv2.erode(m, np.ones((7, 7), np.uint8)) > 0
    ring = (cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0) & ~(cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0)
    if inner.sum() < 100 or ring.sum() < 100:
        return None
    return float(np.median(gray[inner])) <= KAABA_DARK * float(np.median(gray[ring]))


def head_from_scale(ratio: float) -> float:
    """Head width (px) from count(half size) / count(full size), through the digital-zoom curve, then the linear
    correction measured on calibrated frames (2026-10-26: the raw inversion compresses the range)."""
    hs = np.geomspace(1.5, 40, 400)
    G = lambda h: 1 - np.exp(-(np.maximum(h, 0.05) / DIGITAL_CURVE[0]) ** DIGITAL_CURVE[1])  # noqa: E731
    est = float(hs[int(np.argmin(np.abs(G(0.5 * hs) / G(hs) - ratio)))])
    return float(np.clip(2.2 * (est - 2.9), 2.0, 40.0))


class ShotAnalyzer:
    def __init__(self, config: dict, model, bank: Bank | None, verify: bool = True, anchor: bool = True,
                 kaaba_gate: bool | None = None, live_model: dict | None = None, fixes=None):
        from edge.view_matcher import graphics_mask

        self.config, self.model, self.bank = config, model, bank
        # one ALIKED + LightGlue for the bank, the tracking and the calibrated references; the Kaaba segmenter stays on
        # the GPU (user, 2026-10-28: everything on the GPU by default)
        self.m2 = M2Pipeline(config, model, use_kaaba=False, verify=verify, anchor=anchor,
                             feat=bank.feat if bank else None)
        self.feat = bank.feat if bank else Features()
        self.gm_work = graphics_mask(config, WORK)
        self._gm: dict = {}
        self.track: dict | None = None
        self.last_full = 0.0
        self.last_found = False
        self.last_thumb: np.ndarray | None = None
        self._events: tuple | None = None
        self._rows_seen: dict[str, float] = {}
        self.kaaba_gate = KAABA_GATE if kaaba_gate is None else kaaba_gate
        self.source = "youtube"  # the frame's source (set by the caller per frame): its head-size curve (g_for)
        # live_model (config/m2.json "live_model", edge/m2_metric.live_model_config): model is that fine-tuned DM-Count,
        # read without a head-size correction or camera factor, the approximate path with its own table and bias
        self.corr = Correction(live_model)
        a = (live_model or {}).get("approx") or {}
        self.approx_table = (np.asarray(a["table_ratio"]), np.asarray(a["table_head"])) if a.get("table_ratio") else None
        self.approx_bias = float(a.get("bias") or APPROX_BIAS)
        self.fixes = fixes  # edge/m2_fill.LiveFixes (v4, 2026-10-30) or None

    def graphics(self, w: int, h: int) -> np.ndarray:
        from edge.view_matcher import graphics_mask

        if (w, h) not in self._gm:
            self._gm[(w, h)] = graphics_mask(self.config, (w, h))
        return self._gm[(w, h)]

    def analyse(self, img: np.ndarray, ts: datetime) -> ShotReading:
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        gm = self.graphics(w, h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ms = {}
        dens = self.model.predict(img).density
        ms["model"] = round((time.perf_counter() - t0) * 1e3)
        t = time.perf_counter()
        feats = self.feat.detect(img, self.gm_work)
        th = thumb(img)
        out = self._tracked(feats, (w, h), ts)
        if out is not None and kaaba_dark(gray, out.cam) is False:  # drifted: the Kaaba is not where it should be
            out = None
        ms["track"] = round((time.perf_counter() - t) * 1e3)
        now = time.monotonic()
        same = self.last_thumb is not None and float((th * self.last_thumb).mean()) > SAME_SHOT
        self.last_thumb = th
        # the same shot that the last full search could not place: no new search until a cut or REVERIFY_S
        quiet = out is None and same and not self.last_found and now - self.last_full <= REVERIFY_S
        if not quiet and (out is None or now - self.last_full > REVERIFY_S):
            # a full search: when nothing is tracked, and every REVERIFY_S for a tracked shot (when that search finds
            # nothing, e.g. prayer rows hide the floor's features, a good track goes on for up to TRACK_MAX_S)
            t = time.perf_counter()
            self.last_full = now
            full = self._full(img, gm, gray, ts)
            ms.update(full.ms)  # the Kaaba gate's overlap
            free_gpu_cache()  # the search's matching buffers; DM-Count's own come back on the next frame
            ms["find"] = round((time.perf_counter() - t) * 1e3)
            self.last_found = full.cam is not None
            if full.cam is not None or out is None or now - self.track["verified"] > TRACK_MAX_S:
                out = full
        if out is None:
            out = ShotReading(ts, "none")
        out.density, out.heat = dens, dens
        if out.cam is not None and self.fixes is not None:
            t = time.perf_counter()
            blocked = np.maximum(kaaba_exclusion(out.cam, (h, w)), gm)
            dh, dw = dens.shape
            cells = blocked[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5
            dens, out.est = self.fixes.apply(img, dens, out.cam, cells, self.in_prayer(ts), out.camera)
            out.density, out.heat = dens, dens
            ms["fixes"] = round((time.perf_counter() - t) * 1e3)
        if out.cam is not None:
            t = time.perf_counter()
            self.track = {"feats": feats, "cam": out.cam, "camera": out.camera, "method": out.method,
                          "how": out.how.split("+")[0], "verified": self.track["verified"] if out.how == "tracked" and self.track
                          else now, "ts": ts}
            fc = FrameCamera(path=out.how, cam=out.cam, kaaba_mask=None, info={"view": out.camera})
            out.reading = self.m2.reading(img, fc, gm, dens, self.source, self.corr)
            out.people_m2 = out.reading.density_corr.get("all")
            blocked = np.maximum(kaaba_exclusion(out.cam, (h, w)), gm)
            # within a prayer window the threshold is lower, and lower still for a camera that showed rows in the
            # last ROWS_HOLD_S (a row pattern near the threshold would otherwise flip the reading frame to frame)
            prayer = self.in_prayer(ts)
            held = prayer and now - self._rows_seen.get(out.camera, -1e9) <= ROWS_HOLD_S
            out.rows = prayer_rows.measure(gray, out.cam, blocked, ROWS_HOLD_MIN if held else
                                           prayer_rows.STRENGTH_PRAYER if prayer else prayer_rows.STRENGTH_MIN)
            if out.rows.density is not None:
                self._rows_seen[out.camera] = now
            if out.rows.density is not None:
                self._rows_reading(out, dens)
                out.est = None  # the rows' synthetic map replaces the cells
            ms["reading"] = round((time.perf_counter() - t) * 1e3)
        else:
            self.track = None
            t = time.perf_counter()
            out.approx = self._approximate(img, dens, gm)
            out.people_m2 = out.approx.get("people_m2") if out.approx else None
            out.method = "approximate" if out.people_m2 is not None else "none"
            ms["approx"] = round((time.perf_counter() - t) * 1e3)
        ms["total"] = round((time.perf_counter() - t0) * 1e3)
        out.ms = ms
        return out

    def in_prayer(self, ts: datetime) -> bool:
        """Within PRAYER_WINDOW_MIN after an adhan (prayer_times table; reloaded every 6 h)."""
        from edge import prayer_times

        if self._events is None or time.monotonic() - self._events[0] > 6 * 3600:
            try:
                from datetime import timedelta

                self._events = (time.monotonic(), prayer_times.load_events(ts - timedelta(days=1), ts + timedelta(days=2)))
            except Exception:  # no DB: rows only at the normal threshold
                self._events = (time.monotonic(), [])
        ctx = prayer_times.context_from_events(self._events[1], ts) if self._events[1] else None
        return ctx is not None and ctx.minutes_since_prev <= prayer_rows.PRAYER_WINDOW_MIN

    def _tracked(self, feats, size, ts) -> ShotReading | None:
        tr = self.track
        if tr is None:
            return None
        m = self.feat.match(tr["feats"], feats)
        if len(m) < TRACK_GATE[1]:
            return None
        pa, pb = tr["feats"][0][m[:, 0]].astype(float), feats[0][m[:, 1]].astype(float)
        H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
        if H is None or inl.sum() < TRACK_GATE[1]:
            return None
        S = np.diag([WORK[0] / size[0], WORK[1] / size[1], 1.0])
        cam, err = camera_from_homography(np.linalg.inv(S) @ H @ S, tr["cam"], size)
        if cam is None or err > TRACK_GATE[0]:
            return None
        return ShotReading(ts, tr["method"], "tracked", tr["camera"], cam)

    def _full(self, img, gm, gray, ts) -> ShotReading:
        fc = self.m2.camera(img, gm, try_kaaba=False, t=ts.timestamp())
        if fc.cam is not None:
            view = {"ptz": fc.info.get("ptz_view"), "anchor": fc.info.get("anchor_view")}.get(fc.path)
            depth = fc.info.get("ptz_depth", 0) if fc.path == "ptz" else 0
            out = ShotReading(ts, "calibrated", f"{fc.path}/k{depth}" if depth else fc.path, view, fc.cam)
            return self._gate(out, img, gm, fc.kaaba_mask)
        if self.bank is not None:
            bm = self.bank.localise(img, self.gm_work)
            if bm is not None and kaaba_dark(gray, bm.cam) is not False:
                return self._gate(ShotReading(ts, "registered", "bank", bm.place, bm.cam), img, gm, None)
        return ShotReading(ts, "none")

    def _gate(self, out: ShotReading, img, gm, mask) -> ShotReading:
        """The Kaaba gate (see GATE_OVERLAP): fitted to the Kaaba when it is in view; rejected if it still does not
        sit on it. Without the Kaaba in view (or no mask found) the camera stays as found."""
        if not self.kaaba_gate or self.m2.seg is None or kaaba_in_view(out.cam, img.shape[:2], gm) < IN_VIEW_MIN:
            return out
        masks = [mask] if mask is not None else [c.mask for c in self.m2.seg.find(img, gm, k=2) if c.dark >= 0.3]
        best = None
        for mk in masks:
            cands = [(kaaba_overlap(out.cam, mk), out.cam, "")]
            if whole(mk):
                r = kg.fit_from_centre(mk, out.cam.centre, init=out.cam, budget_s=KAABA_FIT_BUDGET_S)
                if r is not None:
                    cands.append((kaaba_overlap(r.cam, mk), r.cam, "+kaaba"))
            c = max(cands, key=lambda x: x[0])
            if best is None or c[0] > best[0]:
                best = c
        if best is None:
            return out
        ov, cam, tag = best
        out.ms["overlap"] = round(ov, 3)
        if ov < GATE_OVERLAP:
            return ShotReading(out.ts, "none", f"kaaba gate {ov:.2f}")
        out.cam, out.how = cam, out.how + tag
        return out

    def _rows_reading(self, out: ShotReading, dens: np.ndarray) -> None:
        """Prayer rows: their people per m^2 for every ring and sector, and a matching synthetic map for the dashboard."""
        rho = out.rows.density
        rd = out.reading
        rd.far = None  # prayer rows: the model reads bent-over rows as empty floor; the far Mataf is not measured then
        rd.density_corr = {k: round(rho, 4) for k in rd.density_corr}
        if rd.sectors:
            rd.sectors = {k: [v[0], round(rho * v[2], 2), v[2], v[3]] for k, v in rd.sectors.items()}
        out.people_m2 = rho
        out.how += "+rows"
        geo = cell_floor(out.cam, dens.shape)
        dist = wall_distance(geo["X"], geo["Y"])
        head = out.cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
        floor = geo["valid"] & (dist >= ALL[0]) & (dist < ALL[1])
        out.heat = np.where(floor, rho * geo["area"] * np.maximum(self.corr.g(self.source)(head), 0.2) * self.corr.factor(out.camera),
                            0).astype(np.float32)

    def _approximate(self, img, dens, gm) -> dict | None:
        """Method c: per horizontal band of the frame (perspective: heads shrink towards the top) the head width from the
        model's scale response, the band's floor area from it, people over area for the whole crowd."""
        dh, dw = dens.shape
        crowd = cv2.GaussianBlur(dens.astype(np.float32), (0, 0), 2.0) > 0.01
        gmc = gm[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5
        crowd &= ~gmc
        n = float(dens[crowd].sum())
        if n < APPROX_MIN_PEOPLE:
            return {"people_m2": None, "count": round(n, 1), "why": "no crowd in view"}
        h, w = img.shape[:2]
        half = self.model.predict(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)).density
        hh = half.shape[0]
        area, people, heads = 0.0, 0.0, []
        for k in range(APPROX_BANDS):
            r0, r1 = k * dh // APPROX_BANDS, (k + 1) * dh // APPROX_BANDS
            c_full = float(dens[r0:r1][crowd[r0:r1]].sum())
            if c_full < APPROX_MIN_PEOPLE / APPROX_BANDS:
                continue
            c_half = float(half[r0 * hh // dh:r1 * hh // dh].sum())
            ratio = c_half / max(float(dens[r0:r1].sum()), 1e-6)
            head = float(np.interp(ratio, *self.approx_table)) if self.approx_table else head_from_scale(ratio)
            if head <= 2.0 or head >= 40.0:  # outside what the scale response can tell
                continue
            area += float(crowd[r0:r1].sum()) * 64 * (HEAD_M / head) ** 2 / SIN_DOWN
            people += c_full
            heads.append(round(head, 1))
        if area <= 0 or people < APPROX_MIN_PEOPLE / 2:
            return {"people_m2": None, "count": round(n, 1), "why": "head size not measurable"}
        return {"people_m2": round(people / area / self.approx_bias, 3), "count": round(people, 1), "head_px": heads,
                "area_m2": round(area * self.approx_bias), "assumed_sin_down": SIN_DOWN}
