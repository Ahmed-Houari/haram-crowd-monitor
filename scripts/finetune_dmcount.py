"""Fine-tune DM-Count on our own frames, then test it on held-out prayers (user request 2026-10-27: far crowds with
heads under 4 px read low; deploy only after asking). Offline only: nothing here touches the live services.

Data: scripts/finetune_data.py (data/finetune/<window>.npz/.json: placed DVR frames, the live model's density maps,
rows labels). Labels (people in a floor region of a frame):
  rows  the prayer-rows estimator run locally (finetune_data.py): a 6 m strip along a line from the Kaaba, people per
        m^2 = rows per metre / 0.6 m; only on frames where the live estimator found prayer rows ("+rows") and on close
        cam0 zooms from the iqama to the end of the prayer (rows_spans), there not where the 1 s pairs' speed sensor
        says the crowd walks (user 2026-11-02: the estimator rarely tags close zooms; rows_why), strips with a strong
        row pattern (ROWS_MIN_STRENGTH, pitch in ROWS_PITCH) where a metre spans >= ROWS_PX_PER_M_MIN px (far strips
        with blurred rows undercount), on the Mataf floor even with a FLOOR_POSE_DEG pose error (on_floor);
  zoom  a zoomed-in frame of a camera without a camera factor (heads >= ZOOM_REF_HEAD px: the model counts them about
        right) and another frame within ZOOM_WINDOW_S that sees the same floor with heads under ZOOM_TARGET_HEAD px:
        the reference's people per m^2 on the shared floor (polar bins of BIN_DEG x BIN_M), times the target's area.
Training: VGG block 5 + the density head (features[TRAIN_FROM:], reg_layer, density_layer); the rest stays as it
is. Loss per labelled region |log(1 + predicted) - log(1 + label)|, plus, elsewhere, the live METRIC's map (4 x 4
cell patches): on the floor the live model's count divided by the live head-size correction and camera factor (zoom
pairs: right for walking crowds, 1.07 at heads under 4 px), off the floor the live model's own map. On prayer-row
frames that anchor holds both ways only off the floor and where heads are >= KEEP_HEAD_ROWS px; on the rest of the
floor it only stops the model reading LESS than the live metric (the live metric under-reads small rows, but seated
groups that are not rows must not drop). All placed frames of the training windows are used (frames with rows labels drawn ROWS_OVERSAMPLE x
as often), so the model learns the rows without changing how it reads walking crowds: deployed, it needs no
head-size correction and no camera factor. Augmentation: mirror, brightness / contrast,
JPEG quality. Full 1920 x 1080 frames, fp16, frozen part without gradients.
Test: the held-out windows (default the Maghrib of 26 Oct): predicted / label per head size, for the live model with
the live correction (what the metric does now), the new model raw, and the new model with a head-size curve refitted
on the training labels; the tawaf window: the headline ring's change. Writes --out (default
models/DM-Count_QNRF_ft.pth, NOT used by the live services; the shadow run reads it) and data/finetune/<--report>, and
a before / after picture of a held-out prayer frame. Every placed window not in --test trains.

  .venv/Scripts/python scripts/finetune_dmcount.py [--test 2026-10-26-maghrib] [--epochs 8] [--eval-only]
  .venv/Scripts/python scripts/finetune_dmcount.py --out models/DM-Count_QNRF_ft2.pth --report report_v2.json  # v2
  .venv/Scripts/python scripts/finetune_dmcount.py --data data/finetune data/finetune_rec ...  # live DVR + recordings
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.kept_frames import frame_file, sba_stems  # noqa: E402  (data/kept_frames first, then data/samples/dvr)

DATA = ROOT / "data" / "finetune"
WEIGHTS_OUT = ROOT / "models" / "DM-Count_QNRF_ft.pth"
ROWS_MIN_STRENGTH = 0.4  # per strip (tawaf crowds reach 0.35 median, 0.46 p90: strips only from rows frames)
ROWS_PITCH = (1.05, 1.4)  # m (measured 1.1-1.34)
ROWS_PX_PER_M_MIN = 12.0
STRIP_HALF_M = 1.0
ZOOM_WINDOW_S = 120
ZOOM_REF_HEAD = 10.0
ZOOM_TARGET_HEAD = 7.0
ZOOM_REF_CAMERAS = ("cam0_mataf_low", "cam1_mataf_wide")  # no camera factor (config/m2.json)
BIN_DEG, BIN_M = 2.0, 1.0
ZOOM_MIN_AREA = 15.0  # m^2 of shared floor per band, in the reference and in the target
TRAIN_FROM = 28  # DMCount.features index: VGG block 5 (conv5_1 .. conv5_4)
HEAD_BINS = [0, 3, 4, 5, 7, 10, 1e9]
KEEP_HEAD_ROWS = 7.0
ROWS_OVERSAMPLE = 3
CLOSE_CAM0_F = 2000.0  # px: a close cam0 zoom; its rows strips are labels inside a prayer's rows span (rows_why)
PRAYER_WORDS = ("fajr", "dhuhr", "asr", "maghrib", "isha", "taraweeh", "jumuah", "friday")  # recordings' prayer tags
MOVING = ("walking", "slow", "fast")  # the speed sensor's sector states of a walking crowd (edge/tawaf_speed.py)
ROWS_GAP_S = 240.0  # s: "+rows" frames further apart than this are in different rows spans (the rows broke up; rows_spans)
ROWS_RUN_MIN = 3  # "+rows" frames a rows span needs (fewer: a stray tag, not a prayer)
ROWS_RUN_MIN_S = 60.0  # s a recording's rows span lasts at least (user 2026-11-02: a 13 s burst before a Fajr iqama)
FLOOR_POSE_DEG = 1.0  # deg: a strip that this camera tilt error would push past the courtyard edge may be on the gallery
NO_CLOSE_RULE = ("screen_20261029T145041Z_DISPLAY1",)  # user 2026-11-02: a live-stream grab whose "+rows" runs fall
#                                                          between the prayers (recitation, before the adhan): no close labels
TEST_ZOOM_MIN_HEAD = 4.0  # zoom labels for heads below this (px) stay out of every test: too low (user, 2026-10-29)
HELD_OUT_FILE = ROOT / "config" / "v4_test.json"  # test-only prayers, never trained on (user, 2026-10-29)


def held_out() -> set[str]:
    try:
        return set(json.loads(HELD_OUT_FILE.read_text(encoding="utf-8"))["held_out"])
    except (OSError, ValueError, KeyError):
        return set()
XFER_S = 300  # s: rows stay put during a prayer; a strip one camera measured labels the others' frames this close in time
CELL = 8


@dataclass
class Region:
    idx: np.ndarray  # flat cell indices in the density map (int32: hundreds of thousands of regions)
    label: float  # people
    kind: str  # rows | zoom
    head: float  # median head width (px) of its cells
    wd: float  # median distance from the Kaaba's walls (m)
    seg: dict | None = None  # rows labels: the strip (finetune_data.py rows_segments)

    def __post_init__(self):
        self.idx = np.asarray(self.idx, np.int32)


def g_curve(h, h0, k, floor=0.2):
    return np.maximum(1 - np.exp(-(np.maximum(h, 0.1) / h0) ** k), floor)


WINDOW_DIR: dict[str, Path] = {}  # window name -> the --data folder it was found in


def frame_path(fr: dict) -> Path:
    """The frame's image: recordings (data/finetune_rec) carry their own "path"; DVR frames are found by stem."""
    return ROOT / fr["path"] if fr.get("path") else frame_file(fr["stem"])


def load(names: list[str], first: set[str] = frozenset()) -> list[dict]:
    """The placed windows' frames; DVR frames recorded from SBA are left out (user, 2026-10-29: only YouTube frames,
    DVR or the user's screen grabs, train or test). The stem check is for DVR frames only: a screen grab of the YouTube
    live stream can share a stem with an SBA frame of the same second. A frame (stem + camera) found in two windows is
    kept once; the `first` windows (test, held out) load first, so a copy of a test frame never trains."""
    sba = sba_stems()
    frames, seen, dup, gone = [], set(), 0, defaultdict(int)
    for n in sorted(names, key=lambda n: n not in first):
        d = WINDOW_DIR.get(n, DATA)
        meta = json.loads((d / f"{n}.json").read_text())
        dens = np.load(d / f"{n}.npz")["density"]
        for m in meta["frames"]:
            if not m.get("path") and m["stem"] in sba:
                continue
            key = (m["stem"], m.get("camera"))
            if key in seen:
                dup += 1
                continue
            if not frame_path(m).exists():  # DVR frames that were never kept are deleted with the DVR's old files
                gone[n] += 1
                continue
            seen.add(key)
            m["window"] = n
            m["old"] = dens[m["i"]].astype(np.float32)
            frames.append(m)
    if dup:
        print(f"{dup} frames found in two windows, kept once", flush=True)
    if gone:
        print(f"{sum(gone.values())} frames without their image left out: {dict(gone)}", flush=True)
    return frames


def geometry(fr: dict, gm: np.ndarray, factors: dict) -> None:
    """Per cell of the frame's density map: floor position, area, head size, the floor mask (in place)."""
    from edge import kaaba_geom as kg
    from edge.m2_metric import HEAD_M, cell_floor, courtyard_edge, wall_distance

    c = fr["cam"]
    cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
    dh, dw = fr["old"].shape
    geo = cell_floor(cam, (dh, dw), CELL)
    X, Y, area = geo["X"], geo["Y"], geo["area"]
    head = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
    wd = wall_distance(X, Y)
    r, th = np.hypot(X, Y), np.arctan2(Y, X)
    k = np.zeros((1080, 1920), np.uint8)
    sil = kg.silhouette(cam, kg.box_corners(z0=kg.CUT_Z))  # the m2 metric's cut-out: people in front of the base count
    if sil is not None:
        cv2.fillPoly(k, [np.round(sil[0]).astype(np.int32)], 1)
    cellm = lambda m: m[: dh * CELL, : dw * CELL].reshape(dh, CELL, dw, CELL).mean((1, 3)) >= 0.5  # noqa: E731
    floor = geo["valid"] & (r <= courtyard_edge(th)) & (wd >= 1.0) & ~cellm(k) & ~cellm(gm)
    f32 = lambda a: a.astype(np.float32)  # noqa: E731  (8000+ frames: float64 here filled the PC's memory, 2026-10-30)
    fr.update(X=f32(X), Y=f32(Y), area=f32(area), head=f32(head), wd=f32(wd), r=f32(r), th=f32(th), floor=floor, cam_obj=cam,
              factor=factors.get(fr["camera"], 1.0), regions=[])


def prayer_events(frames: list[dict]) -> list:
    """The adhan times around the live frames (prayer_times table), for in_prayer; [] without the DB."""
    from edge import prayer_times

    ts = [datetime.strptime(f["stem"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) for f in frames
          if not f.get("path") or f["window"].startswith("speedpairs_")]
    if not ts:
        return []
    try:
        return prayer_times.load_events(min(ts), max(ts))
    except Exception as e:  # noqa: BLE001
        print(f"no prayer times ({e}): live frames count as outside prayer", flush=True)
        return []


def in_prayer(fr: dict, events: list) -> bool:
    """Inside a prayer window: the window builder's "in_prayer" when it has one; else a recording tagged with a prayer
    (never a tawaf video), or a live frame within PRAYER_WINDOW_MIN after an adhan (the shot service's window)."""
    if "in_prayer" in fr:
        return bool(fr["in_prayer"])
    if fr.get("path"):
        p = str(fr.get("prayer") or "").lower()
        return any(w in p for w in PRAYER_WORDS) and "tawaf" not in p
    from edge import prayer_times
    from edge.prayer_rows import PRAYER_WINDOW_MIN

    ctx = prayer_times.context_from_events(events, datetime.strptime(fr["stem"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
    return ctx is not None and ctx.minutes_since_prev <= PRAYER_WINDOW_MIN


def stamp(fr: dict) -> float:
    return datetime.strptime(fr["stem"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def span_key(fr: dict, events: list):
    """Which rows spans a frame belongs to: live frames (DVR, and the live 1 s pairs) by their prayer's adhan, so all
    live windows of a prayer share its spans; a recording by its window. None: a live frame outside a prayer window."""
    if fr.get("path") and not fr["window"].startswith("speedpairs_"):
        return fr["window"]
    from edge import prayer_times
    from edge.prayer_rows import PRAYER_WINDOW_MIN

    ctx = prayer_times.context_from_events(events, datetime.fromtimestamp(stamp(fr), timezone.utc))
    return ("adhan", ctx.prev_utc) if ctx is not None and ctx.minutes_since_prev <= PRAYER_WINDOW_MIN else None


def rows_spans(frames: list[dict], events: list) -> dict:
    """From the iqama to the end of the prayer (user 2026-11-02), from the "+rows" frames: runs of them split where
    two are more than ROWS_GAP_S apart (the rows broke up), a run of fewer than ROWS_RUN_MIN is a stray tag. Live: the
    first run after the adhan (rows first appear: the iqama); a recording: every run of at least ROWS_RUN_MIN_S (a
    video can hold several prayers and the time between). span_key -> [(t0, t1)] (unix s)."""
    ts = defaultdict(list)
    for f in frames:
        if "+rows" in f["how"] and (k := span_key(f, events)) is not None:
            ts[k].append(stamp(f))
    spans = {}
    for k, v in ts.items():
        runs = []
        for t in sorted(v):
            if runs and t - runs[-1][1] <= ROWS_GAP_S:
                runs[-1][1] = t; runs[-1][2] += 1
            else:
                runs.append([t, t, 1])
        runs = [(a, b) for a, b, n in runs if n >= ROWS_RUN_MIN]
        spans[k] = runs[:1] if isinstance(k, tuple) else [(a, b) for a, b in runs if b - a >= ROWS_RUN_MIN_S]
    return spans


def rows_why(fr: dict, events: list, spans: dict | None = None) -> str | None:
    """Why the frame's rows strips are labels: "tag" (the live rows estimator found prayer rows, the shot service's
    "+rows"), "close" (user 2026-11-02: a close cam0 zoom from the iqama to the end of the prayer, rows_spans; the
    estimator rarely tags those), or None (no rows labels)."""
    if "+rows" in fr["how"]:
        return "tag"
    if (spans and fr["window"] not in NO_CLOSE_RULE and fr["camera"] == "cam0_mataf_low" and fr["cam"]["f"] >= CLOSE_CAM0_F
            and in_prayer(fr, events)
            and any(a <= stamp(fr) <= b for a, b in spans.get(span_key(fr, events), ()))):
        return "close"
    return None


def on_floor(fr: dict, idx: np.ndarray) -> bool:
    """The strip's cells stay inside the courtyard edge when the camera tilts FLOOR_POSE_DEG further up (user
    2026-11-02: a strip on the gallery wall; a tracked pose of a far zoom can be off by degrees). Tilting up moves a
    floor point away from the camera along the line of sight: D' = h / tan(atan(h / D) - error)."""
    from edge.m2_metric import courtyard_edge

    cx, cy, h = (float(v) for v in fr["cam_obj"].centre)
    X, Y = fr["X"].ravel()[idx].astype(np.float64), fr["Y"].ravel()[idx].astype(np.float64)
    dx, dy = X - cx, Y - cy
    D = np.maximum(np.hypot(dx, dy), 1e-3)
    dep = np.arctan2(h, D) - math.radians(FLOOR_POSE_DEG)
    if (dep <= 0).any():
        return False
    s = h / np.tan(dep) / D
    X2, Y2 = cx + dx * s, cy + dy * s
    return bool((np.hypot(X2, Y2) <= courtyard_edge(np.arctan2(Y2, X2))).all())


def moving(fr: dict, s: dict) -> bool:
    """The speed sensor says the strip's crowd walks (its ring and sector walking / slow / fast in the frame's 1 s
    pairs, "speed" from the window builder). No reading there (no pairs, hidden floor, past the outer ring): False."""
    from edge.tawaf_speed import RINGS, SECTOR_DEG

    sp = fr.get("speed")
    if not sp:
        return False
    mid = (s["d0"] + s["d1"]) / 2
    for ring, (a0, a1) in RINGS.items():
        if a0 <= mid < a1:
            hows = sp[ring]
            return hows[int((s["b"] % 360) // SECTOR_DEG) % len(hows)] in MOVING
    return False


def rows_regions(fr: dict, events: list = (), spans: dict | None = None) -> None:
    """Rows labels on frames where the live rows estimator found prayer rows (the shot service's "+rows"), and on close
    cam0 zooms from the iqama to the end of the prayer, there only where the speed sensor does not say the crowd walks
    (rows_why); strips that may lie off the Mataf floor are left out (on_floor)."""
    fr["rows_why"] = why = rows_why(fr, events, spans)
    if why is None:
        return
    for s in fr["segs"]:
        if (s["strength"] < ROWS_MIN_STRENGTH or s["px_per_m"] < ROWS_PX_PER_M_MIN
                or not ROWS_PITCH[0] <= s["pitch"] <= ROWS_PITCH[1]):
            continue
        if why == "close" and moving(fr, s):
            continue
        b = math.radians(s["b"])
        dth = np.angle(np.exp(1j * (fr["th"] - b)))
        lat = fr["r"] * np.sin(dth)
        m = fr["floor"] & (np.cos(dth) > 0) & (fr["wd"] >= s["d0"]) & (fr["wd"] < s["d1"]) & (np.abs(lat) <= STRIP_HALF_M)
        idx = np.flatnonzero(m)
        a = float(fr["area"].ravel()[idx].sum())
        if len(idx) < 2 or a < 4:
            continue
        if not on_floor(fr, idx):
            fr["off_floor"] = fr.get("off_floor", 0) + 1
            continue
        fr["regions"].append(Region(idx, s["dens"] * a, "rows", float(np.median(fr["head"].ravel()[idx])),
                                    float(np.median(fr["wd"].ravel()[idx])), s))


def strip_cells(fr: dict, s: dict) -> np.ndarray:
    """Flat cell indices of a rows strip (bearing s["b"], wall distance d0-d1, +-STRIP_HALF_M across) on the floor."""
    b = math.radians(s["b"])
    dth = np.angle(np.exp(1j * (fr["th"] - b)))
    lat = fr["r"] * np.sin(dth)
    m = fr["floor"] & (np.cos(dth) > 0) & (fr["wd"] >= s["d0"]) & (fr["wd"] < s["d1"]) & (np.abs(lat) <= STRIP_HALF_M)
    return np.flatnonzero(m)


def good_strip(s: dict) -> bool:
    return (s["strength"] >= ROWS_MIN_STRENGTH and s["px_per_m"] >= ROWS_PX_PER_M_MIN
            and ROWS_PITCH[0] <= s["pitch"] <= ROWS_PITCH[1])


def xrows_regions(frames: list[dict]) -> int:
    """Rows labels carried over between cameras (user request 2026-10-27: the floor behind the Kaaba): a strip that a
    camera measured close up (a rows label on its own near side of the Kaaba) labels the same floor strip in another
    camera's frame within XFER_S during the same prayer, where that strip lies behind the Kaaba for that camera and it
    has no label of its own there. Only frames whose pose was refitted on the Kaaba take part
    (scripts/refit_frames.py, data/finetune_r; never cam0_mataf_low: its saved frames come from two sides)."""
    def ok(f):
        return (f.get("refit") or {}).get("status") == "refit"

    def t(f):
        return datetime.strptime(f["stem"], "%Y%m%dT%H%M%SZ").timestamp()

    def az(f):
        return math.atan2(f["cam_obj"].centre[1], f["cam_obj"].centre[0])

    n = 0
    for win in sorted({f["window"] for f in frames}):
        fs = [f for f in frames if f["window"] == win]
        rows_fs = [f for f in fs if "+rows" in f["how"]]
        if not rows_fs:
            continue
        t0, t1 = min(map(t, rows_fs)) - 60, max(map(t, rows_fs)) + 60  # the prayer: rows in place
        src = [(t(f), f["camera"], s) for f in fs if ok(f) and t0 <= t(f) <= t1
               for s in f["segs"] if good_strip(s) and math.cos(math.radians(s["b"]) - az(f)) > 0]
        for f in fs:
            if not ok(f) or not t0 <= t(f) <= t1:
                continue
            own = {(r.seg["b"], round(r.seg["d0"] / 6)) for r in f["regions"] if r.kind == "rows"}
            used = set()
            for ts, cam, s in sorted(src, key=lambda x: abs(x[0] - t(f))):
                key = (s["b"], round(s["d0"] / 6))
                if (cam == f["camera"] or abs(ts - t(f)) > XFER_S or key in used or key in own
                        or math.cos(math.radians(s["b"]) - az(f)) >= 0):
                    continue
                idx = strip_cells(f, s)
                a = float(f["area"].ravel()[idx].sum())
                if len(idx) < 2 or a < 4:
                    continue
                f["regions"].append(Region(idx, s["dens"] * a, "xrows", float(np.median(f["head"].ravel()[idx])),
                                           float(np.median(f["wd"].ravel()[idx])), s))
                used.add(key)
                n += 1
    return n


def bins(fr: dict, sel: np.ndarray) -> np.ndarray:
    """Polar floor bin of every cell (angle BIN_DEG x wall distance BIN_M); -1 where not sel."""
    tb = np.floor((np.degrees(fr["th"]) + 180) / BIN_DEG).astype(np.int64)
    db = np.floor(fr["wd"] / BIN_M).astype(np.int64)
    key = tb * 1000 + db
    return np.where(sel, key, -1)


def zoom_regions(frames: list[dict], h0: float, k: float) -> int:
    """Zoom labels between frames of the same window (see the module docstring); returns how many."""
    n = 0
    by_win = defaultdict(list)
    for fr in frames:
        by_win[fr["window"]].append(fr)
    for fs in by_win.values():
        fs.sort(key=lambda f: f["stem"])
        t = {f["stem"]: time.mktime(time.strptime(f["stem"], "%Y%m%dT%H%M%SZ")) for f in fs}
        refs = []
        for f in fs:
            if f["camera"] not in ZOOM_REF_CAMERAS:
                continue
            sel = f["floor"] & (f["head"] >= ZOOM_REF_HEAD)
            if f["area"][sel].sum() < 30:
                continue
            key = bins(f, sel).ravel()
            ok = key >= 0
            corr = (f["old"] / g_curve(f["head"], h0, k)).ravel()
            refs.append((f, key[ok], corr[ok], f["area"].ravel()[ok]))
        for f in fs:
            sel = f["floor"] & (f["head"] < ZOOM_TARGET_HEAD)
            if f["area"][sel].sum() < 30:
                continue
            keyB = bins(f, sel).ravel()
            for A, keyA, corrA, areaA in refs:
                if A is f or abs(t[A["stem"]] - t[f["stem"]]) > ZOOM_WINDOW_S:
                    continue
                shared = np.intersect1d(np.unique(keyA), np.unique(keyB[keyB >= 0]))
                if len(shared) < 5:
                    continue
                wdb = shared % 1000 * BIN_M
                for lo, hi in ((1, 10), (10, 25), (25, 45), (45, 200)):
                    sb = shared[(wdb >= lo) & (wdb < hi)]
                    if len(sb) < 3:
                        continue
                    ia = np.isin(keyA, sb)
                    ib = np.flatnonzero(np.isin(keyB, sb))
                    aA, aB = float(areaA[ia].sum()), float(f["area"].ravel()[ib].sum())
                    if aA < ZOOM_MIN_AREA or aB < ZOOM_MIN_AREA:
                        continue
                    rho = float(corrA[ia].sum()) / aA
                    f["regions"].append(Region(ib, rho * aB, "zoom", float(np.median(f["head"].ravel()[ib])),
                                               float(np.median(f["wd"].ravel()[ib]))))
                    n += 1
    return n


def build_model(weights: Path):
    import torch
    from edge.crowd_model import DMCount, weights_path

    m = DMCount()
    state = torch.load(weights if weights.exists() else weights_path("qnrf"), map_location="cpu", weights_only=True)
    m.load_state_dict(state.get("model", state))
    return m


def to_input(img_bgr: np.ndarray, device):
    import torch
    from edge.crowd_model import IMAGENET_MEAN, IMAGENET_STD

    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).to(device).permute(2, 0, 1)[None].float() / 255
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def forward(model, x, train: bool):
    """The density map; the frozen part (features[:TRAIN_FROM]) without gradients."""
    import torch
    import torch.nn.functional as F

    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            f = model.features[:TRAIN_FROM](x)
        with torch.set_grad_enabled(train):
            f = model.features[TRAIN_FROM:](f)
            f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=True)
            return model.density_layer(model.reg_layer(f))[0, 0].float()


def augment(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, bool]:
    flip = rng.random() < 0.5
    if flip:
        img = img[:, ::-1]
    c, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 95)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, flip


def keep_target(fr: dict, h0: float, k: float) -> np.ndarray:
    """The live metric's map: on the floor the live count / (max(g, 0.2) x camera factor), elsewhere the live count."""
    corr = fr["old"] / (g_curve(fr["head"], h0, k) * fr["factor"])
    return np.where(fr["floor"], corr, fr["old"]).astype(np.float32)


def train(frames: list[dict], args, h0: float, k: float) -> None:
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    model = build_model(Path(args.init) if args.init else Path("none")).to(dev)  # default: the original QNRF weights
    for p in model.features[:TRAIN_FROM].parameters():
        p.requires_grad_(False)
    model.eval()  # no batch norm / dropout in DM-Count: eval mode only fixes nothing; gradients still flow
    opt = torch.optim.AdamW([{"params": model.features[TRAIN_FROM:].parameters(), "lr": args.lr},
                             {"params": list(model.reg_layer.parameters()) + list(model.density_layer.parameters()),
                              "lr": args.lr * 5}], weight_decay=0.0)
    rng = random.Random(0)
    train_fs = frames
    wts = [ROWS_OVERSAMPLE if any(r.kind in ("rows", "xrows") for r in f["regions"]) else 1 for f in train_fs]
    dh, dw = train_fs[0]["old"].shape
    flipmap = (np.arange(dh)[:, None] * dw + (dw - 1 - np.arange(dw))[None, :]).ravel()
    steps = args.epochs * len(train_fs)
    print(f"training on {len(train_fs)} frames, {sum(len(f['regions']) for f in train_fs)} regions, {steps} steps", flush=True)
    t0, log = time.time(), []
    for step in range(steps):
        fr = rng.choices(train_fs, weights=wts)[0]
        img = cv2.imread(str(frame_path(fr)))
        img, flip = augment(img, rng)
        pred = forward(model, to_input(img, dev), True)
        flat = pred.reshape(-1)
        if fr["regions"]:
            idx = np.concatenate([r.idx for r in fr["regions"]]).astype(np.int64)
            rid = np.concatenate([np.full(len(r.idx), i) for i, r in enumerate(fr["regions"])])
            if flip:
                idx = flipmap[idx]
            lab = torch.tensor([r.label for r in fr["regions"]], device=dev)
            ps = torch.zeros(len(fr["regions"]), device=dev).index_add_(0, torch.from_numpy(rid).to(dev),
                                                                        flat[torch.from_numpy(idx).to(dev)])
            l_reg = (torch.log1p(ps) - torch.log1p(lab)).abs().mean()
        else:
            idx, l_reg = np.zeros(0, np.int64), torch.zeros((), device=dev)
        tk = fr.setdefault("keep", keep_target(fr, h0, k))
        old = torch.from_numpy(tk[:, ::-1].copy() if flip else tk).to(dev)
        free = torch.ones(dh * dw, device=dev)
        free[torch.from_numpy(idx).to(dev)] = 0
        floor_min = torch.zeros(dh * dw, device=dev)
        if fr.get("rows_why"):  # prayer rows: the live metric under-reads small rows: both ways off the floor and on
            fl = np.flatnonzero((fr["floor"] & (fr["head"] < KEEP_HEAD_ROWS)).ravel())  # big heads, else "not less"
            fl = torch.from_numpy(flipmap[fl] if flip else fl).to(dev)
            floor_min[fl] = free[fl]
            free[fl] = 0
        free, floor_min = free.view(1, 1, dh, dw), floor_min.view(1, 1, dh, dw)
        pp = F.avg_pool2d((pred[None, None] * free), 4) * 16
        po = F.avg_pool2d((old[None, None] * free), 4) * 16
        l_keep = (torch.log1p(pp) - torch.log1p(po)).abs().mean()
        if floor_min.any():
            pp = F.avg_pool2d((pred[None, None] * floor_min), 4) * 16
            po = F.avg_pool2d((old[None, None] * floor_min), 4) * 16
            l_keep = l_keep + torch.relu(torch.log1p(po) - torch.log1p(pp)).mean()
        loss = (l_reg + args.keep * l_keep) / args.accum
        loss.backward()
        if (step + 1) % args.accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        log.append((float(l_reg.detach()), float(l_keep.detach())))
        if (step + 1) % 100 == 0:
            a = np.mean(log[-100:], 0)
            print(f"  step {step + 1}/{steps}: region loss {a[0]:.3f}, keep loss {a[1]:.3f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "note": "DM-Count QNRF fine-tuned on our frames (scripts/finetune_dmcount.py, "
                f"{time.strftime('%Y-%m-%d')}, windows {sorted({f['window'] for f in frames})}); NOT used live until the "
                "user approves"}, out)
    print(f"saved {out}", flush=True)


def predict_all(frames: list[dict], weights: Path | None) -> None:
    """Every frame's density map from the fine-tuned model (key "new")."""
    import torch

    dev = torch.device("cuda")
    model = build_model(weights).to(dev).eval()
    for i, fr in enumerate(frames):
        img = cv2.imread(str(frame_path(fr)))
        with torch.no_grad():
            fr["new"] = forward(model, to_input(img, dev), False).cpu().numpy()


def table(frames: list[dict], h0: float, k: float, gnew: tuple | None) -> dict:
    """predicted / label per kind and head size: live model + live correction, new model raw, new + refit curve."""
    out = {}
    for kind in ("rows", "xrows", "zoom"):
        rows = []
        for fr in frames:
            for r in fr["regions"]:
                if r.kind != kind:
                    continue
                h = fr["head"].ravel()[r.idx]
                old_c = float((fr["old"].ravel()[r.idx] / (g_curve(h, h0, k) * fr["factor"])).sum())
                new_raw = float(fr["new"].ravel()[r.idx].sum()) if "new" in fr else np.nan
                new_c = float((fr["new"].ravel()[r.idx] / g_curve(h, *gnew)).sum()) if gnew and "new" in fr else np.nan
                rows.append((r.head, r.label, old_c, new_raw, new_c, r.wd))
        if not rows:
            continue
        a = np.array(rows)
        res = []
        for lo, hi in zip(HEAD_BINS[:-1], HEAD_BINS[1:]):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if m.sum() >= 3:
                res.append({"heads": f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+", "n": int(m.sum()),
                            "live": round(a[m, 2].sum() / a[m, 1].sum(), 3), "new_raw": round(a[m, 3].sum() / a[m, 1].sum(), 3),
                            "new_refit": round(a[m, 4].sum() / a[m, 1].sum(), 3) if gnew else None})
        out[kind] = res
    return out


def headline(frames: list[dict], h0: float, k: float, gnew: tuple) -> dict:
    """The headline ring (1-25 m from the walls) per frame: live model + correction vs new model + refit curve."""
    r = []
    for fr in frames:
        sel = fr["floor"] & (fr["wd"] >= 1) & (fr["wd"] < 25)
        a = fr["area"][sel].sum()
        if a < 40 or "new" not in fr:
            continue
        h = fr["head"][sel]
        old = (fr["old"][sel] / (g_curve(h, h0, k) * fr["factor"])).sum() / a
        new = (fr["new"][sel] / g_curve(h, *gnew)).sum() / a
        r.append((fr["camera"], old, new))
    per = defaultdict(list)
    for c, o, n in r:
        per[c].append(n / o if o > 0.05 else np.nan)
    return {c: {"n": len(v), "new_over_live": round(float(np.nanmedian(v)), 3)} for c, v in per.items()} | \
        {"all": {"live": round(float(np.median([o for _, o, _ in r])), 3) if r else None,
                 "new": round(float(np.median([n for _, _, n in r])), 3) if r else None}}


def refit(frames: list[dict]) -> tuple[float, float, float]:
    """A head-size curve for the new model from the training labels (share found = new raw / label per head bin)."""
    from scipy.optimize import least_squares

    hs, sh, w = [], [], []
    for lo, hi in zip([0, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10], [2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 13]):
        lab = pred = 0.0
        for fr in frames:
            for r in fr["regions"]:
                if lo <= r.head < hi:
                    lab += r.label
                    pred += float(fr["new"].ravel()[r.idx].sum())
        if lab > 50:
            hs.append((lo + hi) / 2); sh.append(pred / lab); w.append(math.sqrt(lab))
    hs, sh, w = map(np.array, (hs, sh, w))
    if len(hs) < 3 or sh.min() > 0.9:
        return (0.5, 2.0, 0.2)  # finds (nearly) everyone: no correction
    res = least_squares(lambda p: (1 - np.exp(-(hs / p[0]) ** p[1]) - np.minimum(sh, 1)) * w / w.max(), [3.0, 2.0],
                        bounds=([0.5, 0.5], [20, 10]))
    return (float(res.x[0]), float(res.x[1]), 0.2)


def picture(fr: dict, h0: float, k: float, gnew: tuple, path: Path) -> None:
    """The frame with the live and the new reading side by side (people per m^2 on the crowd-safety levels)."""
    from backend import aiview

    img = cv2.imread(str(frame_path(fr)))
    tiles = []
    for name, dens, gg, fac in (("live model + live correction", fr["old"], (h0, k, 0.2), fr["factor"]),
                                ("fine-tuned model", fr["new"], gnew, 1.0)):
        pm2 = np.where(fr["floor"], dens / np.maximum(fr["area"], 1e-4) / (g_curve(fr["head"], *gg) * fac), np.nan)
        heat = aiview.heat_m2(pm2, (1920, 1080), np.where(fr["floor"], 1 / np.sqrt(np.maximum(fr["area"], 1e-4)), np.nan))
        a = heat[..., 3:4].astype(np.float32) / 255
        out = (img * (1 - a) + heat[..., :3] * a).astype(np.uint8)
        cv2.rectangle(out, (0, 0), (1920, 56), (0, 0, 0), -1)
        cv2.putText(out, f"{name} - {fr['camera']} {fr['stem']} (held out)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2)
        tiles.append(cv2.resize(out, (960, 540)))
    cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", nargs="+", default=["2026-10-26-maghrib"], help="held-out windows")
    ap.add_argument("--tawaf", default="2026-10-26-tawaf-2110", help="window without prayer: its first --tawaf-train share "
                    "of the time trains (walking crowds), the rest tests the headline's change")
    ap.add_argument("--tawaf-train", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--keep", type=float, default=1.5, help="weight of staying with the live metric where no label")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=str(WEIGHTS_OUT), help="the weights file to write (--eval-only: to read); keep "
                    "models/DM-Count_QNRF_ft.pth while the shadow run uses it")
    ap.add_argument("--report", default="report.json", help="report name in the first --data folder")
    ap.add_argument("--init", help="start from these weights (e.g. v1) instead of the original QNRF")
    ap.add_argument("--data", nargs="+", default=[str(DATA)], help="folders of placed windows (data/finetune_r: poses "
                    "refitted on the Kaaba; data/finetune_rec: the recordings); the report goes to the first")
    ap.add_argument("--transfer", action="store_true", help="carry rows labels to the floor behind the Kaaba (xrows)")
    ap.add_argument("--mem", type=float, default=0.0, help="memory guard (user 2026-11-01, v5 trained next to the live "
                    "services): this process may use at most this share of the GPU's memory; 0 = no cap")
    args = ap.parse_args()
    if args.mem > 0:
        import torch

        torch.cuda.set_per_process_memory_fraction(args.mem)
        print(f"memory guard: at most {args.mem:.0%} of the GPU's memory for this process", flush=True)
    from edge.m2_metric import load_correction
    from edge.view_matcher import graphics_mask, load_config

    h0, k, factors = load_correction()
    gm = graphics_mask(load_config(), (1920, 1080))
    DATA = Path(args.data[0])
    for d in map(Path, args.data):
        for p in d.glob("*.json"):
            if p.stem.startswith("report") or p.stem.endswith("-test") or p.stem == "refit_summary":
                continue
            if p.stem in WINDOW_DIR:
                raise SystemExit(f"window {p.stem} in both {WINDOW_DIR[p.stem]} and {d}")
            WINDOW_DIR[p.stem] = d
    names = sorted(WINDOW_DIR)
    test_names = [n for n in names if n in args.test]
    hold = held_out()
    train_names = [n for n in names if n not in args.test and n != args.tawaf and n not in hold]
    if hold & set(names):
        print(f"held out, never trained on (config/v4_test.json): {sorted(hold & set(names))}", flush=True)
    frames = load(names, set(test_names) | hold)
    events = prayer_events(frames)
    spans = rows_spans(frames, events)
    for fr in frames:
        geometry(fr, gm, factors)
        rows_regions(fr, events, spans)
    print(f"rows labels: {sum(f['rows_why'] == 'tag' for f in frames)} frames by +rows, "
          f"{sum(f['rows_why'] == 'close' for f in frames)} close cam0 frames in rows spans; "
          f"{sum(f.get('off_floor', 0) for f in frames)} strips off the Mataf floor left out", flush=True)
    nx = xrows_regions(frames) if args.transfer else 0
    nz = zoom_regions(frames, h0, k)
    tw_all = sorted((f for f in frames if f["window"] == args.tawaf), key=lambda f: f["stem"])
    cut = int(len(tw_all) * args.tawaf_train)
    tr = [f for f in frames if f["window"] in train_names] + tw_all[:cut]
    te = [f for f in frames if f["window"] in test_names]
    tw = tw_all[cut:]
    for f in te + tw:  # the zoom calibration's small-head labels read too low (user, 2026-10-29): training only
        f["regions"] = [r for r in f["regions"] if r.kind != "zoom" or r.head >= TEST_ZOOM_MIN_HEAD]
    cnt = lambda fs, kd: sum(r.kind == kd for f in fs for r in f["regions"])  # noqa: E731
    print(f"windows: train {train_names} + the first {args.tawaf_train:.0%} of {args.tawaf}, test {test_names} + the "
          f"rest of {args.tawaf}; zoom labels {nz}, carried-over rows labels {nx}")
    print(f"train: {len(tr)} frames, rows {cnt(tr, 'rows')}, xrows {cnt(tr, 'xrows')}, zoom {cnt(tr, 'zoom')}; "
          f"test: {len(te)} frames, rows {cnt(te, 'rows')}, xrows {cnt(te, 'xrows')}, zoom {cnt(te, 'zoom')}; tawaf {len(tw)} frames, zoom {cnt(tw, 'zoom')}",
          flush=True)
    if not args.eval_only:
        train(tr, args, h0, k)
    predict_all(tr + te + tw, Path(args.out))
    gnew = refit(tr)
    print(f"refit head-size curve for the new model (training labels): h0 {gnew[0]:.2f}, k {gnew[1]:.2f}")
    rep = {"weights": Path(args.out).name, "train_windows": train_names, "test_windows": test_names, "curve_live": [h0, k], "curve_new": gnew,
           "train": table(tr, h0, k, gnew), "test": table(te, h0, k, gnew), "tawaf_zoom": table(tw, h0, k, gnew),
           "tawaf_headline": headline(tw, h0, k, gnew), "test_headline": headline(te, h0, k, gnew)}
    (DATA / args.report).write_text(json.dumps(rep, indent=1))
    for part in ("test", "tawaf_zoom", "train"):
        print(f"\n{part}: predicted / label (1.00 = right), by head size")
        for kind, rows in rep[part].items():
            for r in rows:
                print(f"  {kind:4s} heads {r['heads']:>5s} px  n {r['n']:4d}   live {r['live']:.2f}   new raw {r['new_raw']:.2f}"
                      f"   new + refit {r['new_refit']:.2f}")
    print("\nheadline ring 1-25 m, new / live (median per frame):", json.dumps(rep["tawaf_headline"]), "(tawaf),",
          json.dumps(rep["test_headline"]), "(held-out prayer)")
    rows_te = [f for f in te if any(r.kind == "rows" for r in f["regions"])]
    if rows_te:
        fr = max(rows_te, key=lambda f: sum(r.kind == "rows" for r in f["regions"]))
        path = ROOT / "data" / "samples" / "debug_align" / f"ft_before_after_{fr['stem']}.jpg"
        picture(fr, h0, k, gnew, path)
        print("picture:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
