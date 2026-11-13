"""Prayer-rows labels for the kept recording frames (user request 2026-10-29), the same way scripts/finetune_data.py
labels the live DVR prayers: each folder in data/kept_frames/recordings/ (yt-dlp downloads and screen grabs, with
index.csv) runs in time order through the live shot analysis (edge/shot.ShotAnalyzer, original DM-Count, Kaaba gate
on) to place each frame on its camera, then finetune_data.rows_segments gives the rows labels along the lines from
the Kaaba (broadcast graphics, the folder's extra mask from run.json, e.g. a sign-language box, and the Kaaba blocked).

Written to data/finetune_rec/<folder>.json + .npz in finetune_data's format (camera, density map, segs), each frame
with "path" (its JPEG; recording stems are not unique across folders, so edge.kept_frames.frame_file does not find
them). A separate folder: data/finetune is read by the live rescale and window lists. No training here (user: v4
later). Frames stay on this PC.

  .venv/Scripts/python scripts/label_recordings.py [folder ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge.kept_frames import KEPT  # noqa: E402

REC = KEPT / "recordings"
OUT = ROOT / "data" / "finetune_rec"
BEHIND_DEG = 135.0  # a rows strip is "behind the Kaaba" when its bearing is this far from the camera's


def stamp(stem: str) -> datetime:
    return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def behind(cam: dict, segs: list[dict], deg: float = BEHIND_DEG) -> list[dict]:
    """Rows strips on the far side of the Kaaba from the camera (bearing around the Kaaba's centre)."""
    r, _ = cv2.Rodrigues(np.asarray(cam["rvec"], float))
    c = -r.T @ np.asarray(cam["tvec"], float).reshape(3)
    cb = np.degrees(np.arctan2(c[1], c[0]))
    return [s for s in segs if abs((s["b"] - cb + 180) % 360 - 180) > deg]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*", help="folders in data/kept_frames/recordings (default: all with index.csv)")
    args = ap.parse_args()
    import torch

    from edge import kaaba_geom as kg
    from edge import shot as shot_mod
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import graphics_mask, load_config
    from finetune_data import Clock, kg_mask, rows_segments

    shot_mod.time = Clock
    cfg = load_config()
    model = CrowdModel()
    bank = Bank(ROOT / "data" / "vloc" / "bank.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    folders = [REC / f for f in args.folders] if args.folders else sorted(
        p for p in REC.iterdir() if (p / "index.csv").exists())
    total = {"frames": 0, "placed": 0, "rows": 0, "behind": 0}
    for d in folders:
        with (d / "index.csv").open(encoding="utf-8") as f:
            idx = list(csv.DictReader(f))
        if not idx:
            continue
        run = {}
        for name in ("run.json", "video.json"):
            if (d / name).exists():
                run = json.loads((d / name).read_text(encoding="utf-8"))
        extra = run.get("extra_graphics_mask") or []
        scfg = cfg | {"graphics_mask": list(cfg.get("graphics_mask", [])) + extra}
        gm = graphics_mask(scfg, (1920, 1080))
        an = ShotAnalyzer(scfg, model, bank)
        meta, maps, last = [], [], None
        for row in sorted(idx, key=lambda r: r["stem"]):
            ts = stamp(row["stem"])
            if last is not None and ts.timestamp() - last > 120:
                an.track, an.last_thumb = None, None
            last = ts.timestamp()
            Clock.now = ts.timestamp()
            path = d / f"{row['stem']}.jpg"
            img = cv2.imread(str(path))
            if img is None or img.shape[1] != 1920:
                continue
            sr = an.analyse(img, ts)
            if sr.cam is None or sr.density is None:
                continue
            cam = sr.cam
            blocked = np.maximum(gm, kg_mask(kg, cam, img.shape[:2]))
            segs = rows_segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam, blocked)
            meta.append({"stem": row["stem"], "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "camera": sr.camera, "how": sr.how, "method": sr.method,
                         "cam": {"f": float(cam.f), "rvec": [float(x) for x in cam.rvec],
                                 "tvec": [float(x) for x in cam.tvec], "size": list(cam.size)},
                         "rows": sr.rows.density if sr.rows else None, "segs": segs, "i": len(maps),
                         "prayer": row.get("prayer") or run.get("tag", ""), "time_of_day": row.get("time_of_day", "")})
            maps.append(sr.density.astype(np.float16))
        np.savez_compressed(OUT / f"{d.name}.npz",
                            density=np.stack(maps) if maps else np.zeros((0, 135, 240), np.float16))
        (OUT / f"{d.name}.json").write_text(json.dumps({"window": d.name, "frames": meta}, ensure_ascii=False),
                                            encoding="utf-8")
        nrows = sum(bool(m["segs"]) for m in meta)
        nbehind = sum(bool(behind(m["cam"], m["segs"])) for m in meta)
        for k, v in (("frames", len(idx)), ("placed", len(meta)), ("rows", nrows), ("behind", nbehind)):
            total[k] += v
        print(f"{d.name}: {len(idx)} frames, {len(meta)} placed, {nrows} with rows labels "
              f"({sum(len(m['segs']) for m in meta)} strips), {nbehind} with rows behind the Kaaba", flush=True)
        torch.cuda.empty_cache()
    print(f"total: {total}", flush=True)


if __name__ == "__main__":
    main()
