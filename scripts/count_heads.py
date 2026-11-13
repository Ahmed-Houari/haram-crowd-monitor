"""Manual head counting for the accuracy metric (success metric 1): you click heads, the model's count is compared.

Privacy: reads frames ONLY from data/samples/ (the test samples the project rules allow) and never writes images.
Your clicks are saved as point coordinates in data/annotations/<frame>.json; reports go to data/reports/accuracy/.

Frames: the collector's accuracy samples data/samples/acc_<view>_<UTC>.png (clean, full resolution; see
config/samples.json), raw frames (frame_grabber --save-samples) or the side-by-side test images (density_*.png,
live_*.png: the left half is the frame). The frame is matched to a reference camera view (edge/view_matcher.py) to
find its counting zone. Dense crowds hold thousands of people, so you count TILES: N squares inside the zone,
spread from near to far, shown enlarged one at a time. Count a head when its centre is inside the bright square.

  .venv/Scripts/python scripts/count_heads.py --list                       samples, matched view, tiles
  .venv/Scripts/python scripts/count_heads.py data/samples/<file>.png      click tool (OpenCV window)
        left click = head   right click = remove nearest   u = undo   n / space = next tile   p = previous
        +/- = zoom   s = save   q / Esc = save and quit
  .venv/Scripts/python scripts/count_heads.py --report                     your counts vs the model, error %
  .venv/Scripts/python scripts/count_heads.py --selftest                   checks the geometry and maths (no window)
Options: --tiles 8 (tiles per frame), --tile 128 (tile size in 720p px, scaled for full-res frames),
--device cpu|cuda (report).

Mataf metric (people per m², 2026-10-25): samples data/samples/m2_<camera>_<UTC>.png carry their camera in a
sidecar .json (edge/extras.py M2Samples). Their tiles lie on the Mataf floor 1-25 m from the Kaaba's walls (not on
the Kaaba, not under graphics), 6 per frame, each about 3 m of floor wide (tens of heads), spread from near (big
heads) to far (small heads). --report then gives, per camera, your heads / the model's corrected count =
the scale factor for that camera (with a bootstrap interval over frames), and your heads / the raw model count per
head size (checks the head-size correction).
  .venv/Scripts/python scripts/count_heads.py --plan                       which m2 files to count, how long
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_metric import ALL, G_FLOOR, HEAD_M, cell_floor, wall_distance  # noqa: E402
from edge.pipeline import warped_zones  # noqa: E402
from edge.view_matcher import ViewMatcher, _cell_weights, graphics_mask, load_config  # noqa: E402

SAMPLES = (ROOT / "data" / "samples").resolve()
ANN_DIR = ROOT / "data" / "annotations"
REPORT_DIR = ROOT / "data" / "reports" / "accuracy"
MAKKAH = timezone(timedelta(hours=3))
WIN = "count heads (q = save and quit)"
MIN_COVER = 0.7  # a tile must lie at least 70 % inside the zone
MIN_INLIERS = 25  # samples are 720p (and may carry drawn zone lines), so accept weaker matches than the collector's 50
M2_TILES, M2_TILE_M = 6, 3.0  # Mataf samples: 6 tiles of ~3 m of floor (~200 heads, ~6 min per frame); fixed so the
#                               tiles stay the same between counting sessions and the report


def sample_path(p: str | Path) -> Path:
    """Resolve a sample path and refuse anything outside data/samples/ (privacy rule)."""
    path = Path(p).resolve()
    if SAMPLES not in path.parents:
        sys.exit(f"only frames in {SAMPLES} may be used (privacy rule): {path}")
    return path


def load_frame(path: Path) -> np.ndarray:
    """The raw frame: the left panel of a side-by-side test image (40 px text bar on top), else the image itself."""
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"cannot read {path}")
    h, w = img.shape[:2]
    if w > 2 * h and h > 40:  # [bar] over [frame | overlay]
        img = img[40:, : w // 2]
    return np.ascontiguousarray(img)


@dataclass
class Prepared:
    frame: np.ndarray
    view_id: str
    zoom: float
    mask: np.ndarray  # uint8 1 = inside the counting zone and not under broadcast graphics
    tiles: list[list[int]]  # [x, y, w, h]


def m2_meta(path: Path) -> dict | None:
    side = path.with_suffix(".json")
    return json.loads(side.read_text()) if path.name.startswith("m2_") and side.exists() else None


def m2_geometry(frame: np.ndarray, meta: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 px mask of the Mataf floor 1-25 m from the walls, off the Kaaba and graphics; head width px per px)."""
    from edge.m2_pipeline import kaaba_exclusion

    c = meta["cam"]
    cam = kg.Camera(c["f"], np.array(c["rvec"]), np.array(c["tvec"]), tuple(c["size"]))
    h, w = frame.shape[:2]
    geo = cell_floor(cam, (h // 8, w // 8))
    d = wall_distance(geo["X"], geo["Y"])
    cells = geo["valid"] & (d >= ALL[0]) & (d < ALL[1])
    up = lambda a: cv2.resize(a.astype(np.float32), (w // 8 * 8, h // 8 * 8), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    mask = np.zeros((h, w), np.uint8)
    mask[: h // 8 * 8, : w // 8 * 8] = up(cells) > 0.5
    mask &= 1 - np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(config, (w, h)).astype(np.uint8))
    head = np.zeros((h, w), np.float32)
    head[: h // 8 * 8, : w // 8 * 8] = up(cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3))
    return mask, head


def prepare_m2(path: Path, meta: dict, config: dict, n_tiles: int, tile_m: float) -> Prepared | None:
    frame = load_frame(path)
    mask, head = m2_geometry(frame, meta, config)
    h, w = frame.shape[:2]
    cands = []
    for y in range(0, h, 24):
        for x in range(0, w, 24):
            if not mask[y, x]:
                continue
            side = int(np.clip(round(tile_m * head[y, x] / HEAD_M), 40, 320))  # tile_m metres of floor there
            x0, y0 = x - side // 2, y - side // 2
            if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
                continue
            if mask[y0:y0 + side, x0:x0 + side].mean() >= MIN_COVER:
                cands.append(([x0, y0, side, side], float(head[y, x])))
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))
    tiles: list[list[int]] = []
    if cands:  # spread over head size (near -> far), no overlaps
        cands.sort(key=lambda c: c[1])
        for band in np.array_split(np.arange(len(cands)), min(n_tiles, len(cands))):
            for i in rng.permutation(band):
                r = cands[int(i)][0]
                if all(r[0] + r[2] <= t[0] or t[0] + t[2] <= r[0] or r[1] + r[3] <= t[1] or t[1] + t[3] <= r[1]
                       for t in tiles):
                    tiles.append(r)
                    break
    return Prepared(frame, meta["camera"], float(meta.get("zoom") or 1.0), mask, tiles)


def prepare(path: Path, matcher: ViewMatcher, n_tiles: int, tile: int) -> Prepared | None:
    meta = m2_meta(path)
    if meta is not None:
        return prepare_m2(path, meta, matcher.config, M2_TILES, M2_TILE_M)
    frame = load_frame(path)
    m = matcher.match(frame)
    if m is None:
        return None
    h, w = frame.shape[:2]
    tile = round(tile * w / 1280)  # --tile is in 720p px: the same floor area on full-res frames (acc_*.png)
    mask = np.zeros((h, w), np.uint8)
    for poly in warped_zones(matcher, m).values():
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    mask &= 1 - graphics_mask(matcher.config, (w, h)).astype(np.uint8)
    cands = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            if mask[y:y + tile, x:x + tile].mean() >= MIN_COVER:
                cands.append([x, y, tile, tile])
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))  # stable: same tiles every session
    tiles = []
    if cands:  # spread over depth: sort by y, one random tile per band
        cands.sort(key=lambda t: (t[1], t[0]))
        for band in np.array_split(np.arange(len(cands)), min(n_tiles, len(cands))):
            tiles.append(cands[int(rng.choice(band))])
    return Prepared(frame, m.view_id, m.zoom, mask, tiles)


def ann_path(path: Path) -> Path:
    return ANN_DIR / f"{path.stem}.json"


def load_ann(path: Path, prep: Prepared) -> dict:
    p = ann_path(path)
    if p.exists():
        ann = json.loads(p.read_text())
        if ann.get("tiles") and [t["rect"] for t in ann["tiles"]] == prep.tiles:
            return ann
    return {"frame": str(path.relative_to(ROOT)), "view_id": prep.view_id, "zoom": round(prep.zoom, 3),
            "frame_size": [prep.frame.shape[1], prep.frame.shape[0]],
            "tiles": [{"rect": t, "points": [], "done": False} for t in prep.tiles]}


def save_ann(path: Path, ann: dict) -> None:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    ann["saved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ann_path(path).write_text(json.dumps(ann, indent=1))


def annotate(path: Path, prep: Prepared, ann: dict, scale: float) -> None:
    """OpenCV click tool, one enlarged tile at a time (with a dimmed margin for context)."""
    state = {"i": next((i for i, t in enumerate(ann["tiles"]) if not t["done"]), 0), "scale": scale, "dirty": False}
    margin = 24
    h, w = prep.frame.shape[:2]

    def view_box():
        x, y, tw, th = ann["tiles"][state["i"]]["rect"]
        return max(0, x - margin), max(0, y - margin), min(w, x + tw + margin), min(h, y + th + margin)

    def render():
        t = ann["tiles"][state["i"]]
        x, y, tw, th = t["rect"]
        x0, y0, x1, y1 = view_box()
        crop = prep.frame[y0:y1, x0:x1].astype(np.float32)
        inside = np.zeros(crop.shape[:2], bool)
        inside[y - y0:y - y0 + th, x - x0:x - x0 + tw] = True
        inside &= prep.mask[y0:y1, x0:x1].astype(bool)
        crop[~inside] *= 0.35  # outside the tile or the zone: dimmed, not counted
        s = state["scale"]
        img = cv2.resize(crop.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(img, (int((x - x0) * s), int((y - y0) * s)), (int((x - x0 + tw) * s), int((y - y0 + th) * s)),
                      (92, 164, 200), 1)
        for px, py in t["points"]:
            c = (int((px - x0) * s), int((py - y0) * s))
            cv2.circle(img, c, max(3, int(s * 1.6)), (82, 111, 255), -1)
            cv2.circle(img, c, max(4, int(s * 1.6)) + 1, (255, 255, 255), 1)
        bar = np.full((52, img.shape[1], 3), 21, np.uint8)
        done = sum(tt["done"] for tt in ann["tiles"])
        cv2.putText(bar, f"tile {state['i'] + 1}/{len(ann['tiles'])} ({done} done)   heads: {len(t['points'])}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (236, 231, 220), 1, cv2.LINE_AA)
        cv2.putText(bar, "L-click head  R-click remove  u undo  n/space next  p prev  +/- zoom  s save  q quit",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (155, 165, 165), 1, cv2.LINE_AA)
        cv2.imshow(WIN, np.vstack([bar, img]))

    def on_mouse(ev, mx, my, _flags, _param):
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        t = ann["tiles"][state["i"]]
        x0, y0, _, _ = view_box()
        s = state["scale"]
        fx, fy = x0 + mx / s, y0 + (my - 52) / s
        x, y, tw, th = t["rect"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            if x <= fx < x + tw and y <= fy < y + th and prep.mask[int(fy), int(fx)]:
                t["points"].append([round(fx, 1), round(fy, 1)])
        elif t["points"]:
            d = [np.hypot(px - fx, py - fy) for px, py in t["points"]]
            if min(d) <= 12 / s + 3:
                t["points"].pop(int(np.argmin(d)))
        state["dirty"] = True
        render()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    render()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            k = ord("q")
        if k in (ord("q"), 27):
            break
        if k in (ord("n"), ord(" ")):
            ann["tiles"][state["i"]]["done"] = True
            save_ann(path, ann)
            state["i"] = min(state["i"] + 1, len(ann["tiles"]) - 1)
        elif k == ord("p"):
            state["i"] = max(0, state["i"] - 1)
        elif k == ord("u") and ann["tiles"][state["i"]]["points"]:
            ann["tiles"][state["i"]]["points"].pop()
        elif k in (ord("+"), ord("=")):
            state["scale"] = min(10.0, state["scale"] * 1.25)
        elif k == ord("-"):
            state["scale"] = max(1.5, state["scale"] / 1.25)
        elif k == ord("s"):
            save_ann(path, ann)
        else:
            continue
        render()
    save_ann(path, ann)
    cv2.destroyAllWindows()
    print(f"saved {ann_path(path)} ({sum(len(t['points']) for t in ann['tiles'])} heads in "
          f"{sum(t['done'] for t in ann['tiles'])} finished tiles)")


def model_tile_counts(frame: np.ndarray, mask: np.ndarray, rects: list, model) -> list[float]:
    """Model count inside each tile x zone (same cell weighting as the collector's zone counts)."""
    r = model.predict(frame)
    dh, dw = r.density.shape
    out = []
    for x, y, tw, th in rects:
        m = np.zeros_like(mask)
        m[y:y + th, x:x + tw] = mask[y:y + th, x:x + tw]
        out.append(float((r.density * _cell_weights(m, dw, dh)).sum()))
    return out


def m2_tile_counts(prep: Prepared, meta: dict, config: dict, rects: list, model) -> list[tuple[float, float, float]]:
    """(raw model count, corrected count, median head px) per tile: DM-Count's density x the tile's floor cells,
    corrected per cell by the live head-size correction g (edge/m2_pipeline.py), as the Mataf metric does."""
    from edge.m2_pipeline import g

    r = model.predict(prep.frame)
    dh, dw = r.density.shape
    _, head = m2_geometry(prep.frame, meta, config)
    head_c = head[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3))
    corr = r.density / np.maximum(g(head_c), G_FLOOR)
    out = []
    for x, y, tw, th in rects:
        m = np.zeros_like(prep.mask)
        m[y:y + th, x:x + tw] = prep.mask[y:y + th, x:x + tw]
        wgt = _cell_weights(m, dw, dh)
        out.append((float((r.density * wgt).sum()), float((corr * wgt).sum()), float(np.median(head[y:y + th, x:x + tw]))))
    return out


def report_m2(model, config: dict, n_tiles: int = M2_TILES, tile_m: float = M2_TILE_M) -> dict:
    """Per camera: your heads vs the model's corrected count on the Mataf floor -> scale factor (bootstrap 90 %)."""
    rows = []
    for p in sorted(ANN_DIR.glob("m2_*.json")):
        ann = json.loads(p.read_text())
        path = sample_path(ROOT / ann["frame"])
        meta = m2_meta(path)
        tiles = [t for t in ann["tiles"] if t["done"]]
        if meta is None or not tiles:
            continue
        prep = prepare_m2(path, meta, config, n_tiles, tile_m)
        for t, (raw, cor, hp) in zip(tiles, m2_tile_counts(prep, meta, config, [t["rect"] for t in tiles], model)):
            rows.append({"frame": path.name, "camera": meta["camera"], "period": meta.get("period"),
                         "manual": len(t["points"]), "raw": round(raw, 1), "corrected": round(cor, 1),
                         "head_px": round(hp, 1)})
    res: dict = {"tiles": len(rows), "per_camera": {}, "rows": rows}
    rng = np.random.default_rng(0)
    for cam in sorted({r["camera"] for r in rows}):
        rs = [r for r in rows if r["camera"] == cam]
        frames = sorted({r["frame"] for r in rs})
        by = {f: (sum(r["manual"] for r in rs if r["frame"] == f), sum(r["corrected"] for r in rs if r["frame"] == f))
              for f in frames}
        man, cor = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
        boot = []
        for _ in range(2000):
            pick = rng.choice(frames, len(frames))
            m_, c_ = sum(by[f][0] for f in pick), sum(by[f][1] for f in pick)
            if c_ > 0:
                boot.append(m_ / c_)
        res["per_camera"][cam] = {
            "frames": len(frames), "tiles": len(rs), "manual": man, "corrected": round(cor, 1),
            "raw": round(sum(r["raw"] for r in rs), 1),
            "scale": round(man / cor, 3) if cor else None,
            "scale_90": [round(float(np.percentile(boot, 5)), 3), round(float(np.percentile(boot, 95)), 3)] if boot else None,
            "per_period": {per: round(sum(r["manual"] for r in rs if r["period"] == per)
                                      / max(sum(r["corrected"] for r in rs if r["period"] == per), 1e-9), 3)
                           for per in sorted({r["period"] for r in rs})}}
    return res


def plan_m2(model, config: dict, n_tiles: int = M2_TILES, tile_m: float = M2_TILE_M, per_cam_period: int = 3) -> str:
    """Which m2 samples to count (per camera, day and dark, spread in time) and how long it should take."""
    files = sorted(SAMPLES.glob("m2_*.png"))
    metas = [(f, m2_meta(f)) for f in files]
    chosen = []
    for cam in sorted({m["camera"] for _, m in metas if m}):
        for dark in (False, True):
            fs = [(f, m) for f, m in metas if m and m["camera"] == cam and (m.get("period") != "day") == dark]
            fs.sort(key=lambda fm: fm[1]["ts_utc"])
            idx = np.linspace(0, len(fs) - 1, min(per_cam_period, len(fs))).round().astype(int) if fs else []
            chosen += [fs[i] for i in dict.fromkeys(idx)]
    lines, total_s = [], 0.0
    for f, m in chosen:
        prep = prepare_m2(f, m, config, n_tiles, tile_m)
        heads = sum(c for _, c, _ in m2_tile_counts(prep, m, config, prep.tiles, model)) if prep.tiles else 0
        sec = 60 + 20 * len(prep.tiles) + 1.3 * heads  # open + per tile + ~1.3 s per click
        total_s += sec
        done = " (done)" if ann_path(f).exists() else ""
        lines.append(f"  {f.name}  {m['camera']:<16} {m.get('period', ''):<8} {len(prep.tiles)} tiles, ~{heads:.0f} heads, "
                     f"~{sec / 60:.0f} min{done}")
    have = {cam: sum(1 for _, m in metas if m and m["camera"] == cam) for cam in sorted({m["camera"] for _, m in metas if m})}
    head = (f"Mataf samples on disk: {len(files)} ({', '.join(f'{c} {n}' for c, n in have.items())}). "
            f"Count these {len(chosen)} ({per_cam_period} per camera by day and {per_cam_period} in the dark), "
            f"~{total_s / 3600:.1f} h in total:")
    return "\n".join([head] + lines + ["Each: .venv/Scripts/python scripts/count_heads.py data/samples/<file>.png "
                                        "(then --report)."])


def report(model, matcher: ViewMatcher, n_tiles: int, tile: int, only: list[Path] | None = None) -> dict:
    rows = []
    for p in sorted(ANN_DIR.glob("*.json")) if only is None else [ann_path(x) for x in only]:
        if p.name.startswith("m2_"):
            continue  # Mataf samples: report_m2
        ann = json.loads(p.read_text())
        path = sample_path(ROOT / ann["frame"])
        prep = prepare(path, matcher, n_tiles, tile)
        tiles = [t for t in ann["tiles"] if t["done"]]
        if prep is None or not tiles:
            continue
        model_counts = model_tile_counts(prep.frame, prep.mask, [t["rect"] for t in tiles], model)
        for t, mc in zip(tiles, model_counts):
            rows.append({"frame": ann["frame"], "view_id": ann["view_id"], "rect": t["rect"], "manual": len(t["points"]),
                         "model": round(mc, 1)})
    if not rows:
        return {"tiles": 0}
    man, mod = np.array([r["manual"] for r in rows], float), np.array([r["model"] for r in rows])
    big = man >= 5
    res = {"tiles": len(rows), "frames": len({r["frame"] for r in rows}), "manual_total": int(man.sum()),
           "model_total": round(float(mod.sum()), 1),
           "total_error_pct": round(100 * (mod.sum() - man.sum()) / man.sum(), 1) if man.sum() else None,
           "mae_per_tile": round(float(np.abs(mod - man).mean()), 2),
           "mape_per_tile_pct": round(float(100 * np.mean(np.abs(mod[big] - man[big]) / man[big])), 1) if big.any() else None,
           "per_view": {}, "rows": rows}
    for v in sorted({r["view_id"] for r in rows}):
        sel = np.array([r["view_id"] == v for r in rows])
        res["per_view"][v] = {"tiles": int(sel.sum()), "manual": int(man[sel].sum()), "model": round(float(mod[sel].sum()), 1),
                              "error_pct": round(100 * (mod[sel].sum() - man[sel].sum()) / man[sel].sum(), 1) if man[sel].sum() else None}
    return res


def print_report(res: dict) -> str:
    if not res.get("tiles"):
        return "No finished tiles yet: count some first (count_heads.py data/samples/<file>.png)."
    lines = [f"Counting accuracy: {res['tiles']} tiles in {res['frames']} frames",
             f"  you counted {res['manual_total']} heads, the model {res['model_total']:.0f}: "
             f"total error {res['total_error_pct']:+.1f} %  (negative = the model undercounts)",
             f"  per tile: MAE {res['mae_per_tile']} heads, mean abs. error {res['mape_per_tile_pct']} % (tiles with 5+ heads)"]
    for v, r in res["per_view"].items():
        lines.append(f"  {v}: {r['tiles']} tiles, you {r['manual']} vs model {r['model']:.0f} ({r['error_pct']:+.1f} %)")
    lines.append(f"  {'manual':>7} {'model':>7}  frame / tile")
    for r in res["rows"]:
        lines.append(f"  {r['manual']:>7} {r['model']:>7.1f}  {Path(r['frame']).name} {r['rect']}")
    return "\n".join(lines)


def selftest(matcher: ViewMatcher, model, n_tiles: int, tile: int) -> None:
    """Fake "manual" points from the model's own density: the report must show ~0 % error."""
    cands = sorted(SAMPLES.glob("*.png"))
    for path in cands:
        prep = prepare(path, matcher, n_tiles, tile)
        if prep is None or not prep.tiles:
            continue
        expected = model_tile_counts(prep.frame, prep.mask, prep.tiles, model)
        rng = np.random.default_rng(0)
        ann = load_ann(path, prep)
        for t, n in zip(ann["tiles"], expected):
            x, y, tw, th = t["rect"]
            ys, xs = np.nonzero(prep.mask[y:y + th, x:x + tw])
            pick = rng.choice(len(xs), size=min(len(xs), int(round(n))), replace=False)
            t["points"], t["done"] = [[float(x + xs[i]), float(y + ys[i])] for i in pick], True
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        ann_saved = ann_path(path)
        backup = ann_saved.read_text() if ann_saved.exists() else None  # never lose real counts
        try:
            ann_saved.write_text(json.dumps(ann))
            res = report(model, matcher, n_tiles, tile, only=[path])
        finally:
            if backup is None:
                ann_saved.unlink(missing_ok=True)
            else:
                ann_saved.write_text(backup)
        assert res["tiles"] == len(prep.tiles), res
        assert abs(res["total_error_pct"]) < 3, res["total_error_pct"]
        print(f"selftest OK on {path.name} ({prep.view_id}, zoom {prep.zoom:.2f}): {len(prep.tiles)} tiles of "
              f"{tile} px, model {res['model_total']:.0f} vs fake manual {res['manual_total']} -> "
              f"{res['total_error_pct']:+.1f} % (rounding only)")
        return
    sys.exit("selftest: no sample in data/samples matches a reference view")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("frame", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--device", default="cuda", help="crowd model device for --plan / --report / --selftest")
    ap.add_argument("--plan", action="store_true", help="Mataf samples to count and the time it takes")
    args = ap.parse_args()

    matcher = ViewMatcher.from_config(load_config())
    matcher.min_inliers = MIN_INLIERS  # only here: the collector keeps its own stricter setting
    if args.list:
        for p in sorted(SAMPLES.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            prep = prepare(p, matcher, args.tiles, args.tile)
            done = ann_path(p).exists()
            print(f"{p.name}: " + (f"{prep.view_id} (zoom {prep.zoom:.2f}), {len(prep.tiles)} tiles"
                                   + (" · annotated" if done else "") if prep else "no reference view (skip)"))
        return
    if args.plan:
        from edge.crowd_model import CrowdModel

        print(plan_m2(CrowdModel(device=args.device, half=args.device != "cpu"), matcher.config))
        return
    if args.report or args.selftest:
        from edge.crowd_model import CrowdModel

        model = CrowdModel(device=args.device, half=args.device != "cpu")
        if args.selftest:
            return selftest(matcher, model, args.tiles, args.tile)
        res = report(model, matcher, args.tiles, args.tile)
        text = print_report(res)
        m2 = report_m2(model, matcher.config)
        if m2["tiles"]:
            text += "\n\nMataf metric, per camera: your heads / the model's corrected count = scale factor (90 % interval)"
            for cam, r in m2["per_camera"].items():
                text += (f"\n  {cam}: {r['frames']} frames, {r['tiles']} tiles, you {r['manual']} vs corrected "
                         f"{r['corrected']:.0f} (raw {r['raw']:.0f}): scale {r['scale']} {r['scale_90']}, by period "
                         f"{r['per_period']}")
            res["m2"] = m2
        print(text)
        if res.get("tiles") or res.get("m2"):
            out = REPORT_DIR / datetime.now(MAKKAH).strftime("%Y%m%d_%H%M")
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.txt").write_text(text + "\n", encoding="utf-8")
            (out / "accuracy.json").write_text(json.dumps(res, indent=1))
            print(f"saved {out}")
        return
    if not args.frame:
        ap.error("give a frame from data/samples/, or --list / --report / --selftest")
    path = sample_path(args.frame)
    prep = prepare(path, matcher, args.tiles, args.tile)
    if prep is None:
        sys.exit(f"{path.name}: no reference camera view matches this frame, so there is no counting zone")
    if not prep.tiles:
        sys.exit(f"{path.name}: the zone is too small for {args.tile} px tiles (try --tile 96)")
    annotate(path, prep, load_ann(path, prep), args.scale)


if __name__ == "__main__":
    main()
