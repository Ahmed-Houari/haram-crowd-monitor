"""Shadow run of the fine-tuned DM-Count (user request 2026-10-27: 24 h next to the live model, both logged, no dashboard
change). The shot service (scripts/live_shot.py) hands every frame it analysed to Shadow.add after the live analysis;
the fine-tuned model (models/DM-Count_QNRF_ft.pth, scripts/finetune_dmcount.py) reads the same frame and the pair goes
to the table crowd_m2_shadow (edge/db.py). Nothing here feeds a reading the API serves.

The new model has the head-size correction and the camera factors built in (it was trained to read the live metric's
corrected values on walking crowds and the true counts on prayer rows), so its reading is
  - with a camera (calibrated / registered / tracked): edge/m2_metric.measure without a head-size correction and
    without a camera factor; SBA frames with their own head-size curve refitted for this model;
  - without a camera: the approximate path (edge/shot.py method c) with the table count ratio -> head width and the
    bias refitted for this model.
Configuration: config/m2.json "shadow" (scripts/shadow_refit.py). Counting only; frames stay in memory.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import cv2
import numpy as np

from edge import db
from edge.m2_metric import HEAD_M, M2_CONFIG, measure
from edge.m2_pipeline import kaaba_exclusion

log = logging.getLogger(__name__)


def config() -> dict | None:
    try:
        c = json.loads(M2_CONFIG.read_text(encoding="utf-8")).get("shadow")
    except (OSError, ValueError):
        return None
    return c if c and c.get("enabled") else None


class Shadow:
    def __init__(self, cfg: dict):
        from edge.crowd_model import CrowdModel

        self.cfg = cfg
        self.model = CrowdModel(cfg.get("weights", "qnrf_ft"))
        self.name = cfg.get("weights", "qnrf_ft")
        self.curves = {s: (float(v["h0"]), float(v["k"])) for s, v in (cfg.get("source") or {}).items()}
        a = cfg.get("approx") or {}
        self.tab_r, self.tab_h = np.asarray(a.get("table_ratio") or []), np.asarray(a.get("table_head") or [])
        self.bias = float(a.get("bias") or 1.0)
        self.rows: list[dict] = []

    @classmethod
    def from_config(cls) -> "Shadow | None":
        c = config()
        if not c:
            return None
        try:
            return cls(c)
        except Exception:
            log.exception("shadow model not loaded: no shadow run")
            return None

    def _g(self, source: str):
        if source not in self.curves:
            return lambda h: np.ones_like(h)  # no head-size correction (built into the model)
        h0, k = self.curves[source]
        return lambda h: 1.0 - np.exp(-np.power(np.maximum(h, 0.1) / h0, k))

    def _approximate(self, img: np.ndarray, dens: np.ndarray, gm: np.ndarray) -> dict | None:
        """edge/shot.py method c with this model's table and bias."""
        from edge.shot import APPROX_BANDS, APPROX_MIN_PEOPLE, SIN_DOWN

        if not len(self.tab_r):
            return None
        dh, dw = dens.shape
        crowd = cv2.GaussianBlur(dens.astype(np.float32), (0, 0), 2.0) > 0.01
        crowd &= ~(gm[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5)
        if float(dens[crowd].sum()) < APPROX_MIN_PEOPLE:
            return None
        h, w = img.shape[:2]
        half = self.model.predict(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)).density
        hh = half.shape[0]
        area = people = 0.0
        for k in range(APPROX_BANDS):
            r0, r1 = k * dh // APPROX_BANDS, (k + 1) * dh // APPROX_BANDS
            c_full = float(dens[r0:r1][crowd[r0:r1]].sum())
            if c_full < APPROX_MIN_PEOPLE / APPROX_BANDS:
                continue
            ratio = float(half[r0 * hh // dh:r1 * hh // dh].sum()) / max(float(dens[r0:r1].sum()), 1e-6)
            head = float(np.interp(ratio, self.tab_r, self.tab_h))
            if not 2.0 < head < 40.0:
                continue
            area += float(crowd[r0:r1].sum()) * 64 * (HEAD_M / head) ** 2 / SIN_DOWN
            people += c_full
        if area <= 0 or people < APPROX_MIN_PEOPLE / 2:
            return None
        return {"people_m2": round(people / area / self.bias, 4), "count": round(people, 1)}

    def add(self, img: np.ndarray, sr, gm: np.ndarray, source: str) -> None:
        """The shadow reading of one analysed frame, next to the live one (sr: edge.shot.ShotReading)."""
        if sr.method == "none" and sr.approx is None:
            return
        dens = self.model.predict(img).density
        row = {"ts_utc": sr.ts, "model": self.name, "camera": sr.camera, "method": sr.method, "how": (sr.how or "")[:40],
               "source": source, "rows": "+rows" in (sr.how or "")}
        live = sr.reading.density_corr if sr.reading else {}
        row.update(live_all=live.get("all"), live_inner=live.get("inner"), live_outer=live.get("outer"),
                   live_far=(sr.reading.far or {}).get("corr") if sr.reading else None)
        if sr.cam is not None:
            h, w = img.shape[:2]
            ex = np.maximum(kaaba_exclusion(sr.cam, (h, w)), gm)
            rd = measure(sr.cam, dens, ex, self._g(source), 1.0)
            row.update(shadow_all=rd.density_corr.get("all"), shadow_inner=rd.density_corr.get("inner"),
                       shadow_outer=rd.density_corr.get("outer"), shadow_far=(rd.far or {}).get("corr"),
                       area_all=rd.area.get("all"), head_px_all=rd.head_px.get("all"),
                       count_all=rd.count.get("all"))
        elif sr.approx is not None:
            row.update(live_approx=sr.approx.get("people_m2"))
            a = self._approximate(img, dens, gm)
            if a:
                row.update(shadow_approx=a["people_m2"], count_all=a["count"])
        if all(row.get(k) is None for k in ("live_all", "shadow_all", "live_approx", "shadow_approx")):
            return  # neither model has a reading for this frame
        self.rows.append(row)

    def flush(self) -> int:
        """Write the collected pairs (once a minute)."""
        rows, self.rows = self.rows, []
        if rows:
            db.insert_shadow(rows)
        return len(rows)


def started(when: datetime, cfg: dict) -> str:
    return (f"shadow run of {cfg.get('weights')} since {when:%Y-%m-%d %H:%M} UTC: no head-size correction, no camera factor, "
            f"SBA curve {cfg.get('source', {}).get('sba')}, approximate bias {(cfg.get('approx') or {}).get('bias')}")
