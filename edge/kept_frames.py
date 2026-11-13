"""Training and test frames are kept (user, 2026-10-28, after housekeeping's 2-day limit on data/samples/dvr deleted
the 26 Oct windows): every DVR frame a fine-tune window uses (data/finetune*/<window>.json) is copied to
data/kept_frames, which housekeeping never deletes; the scripts read a frame from there first.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from edge.runtime import PROJECT_ROOT

DVR = PROJECT_ROOT / "data" / "samples" / "dvr"
KEPT = PROJECT_ROOT / "data" / "kept_frames"
WINDOW_DIRS = [PROJECT_ROOT / "data" / "finetune", PROJECT_ROOT / "data" / "finetune_r"]


def frame_file(stem: str) -> Path:
    """The saved frame <stem>.jpg: the kept copy if there is one, else the DVR folder's."""
    kept = KEPT / f"{stem}.jpg"
    return kept if kept.exists() else DVR / f"{stem}.jpg"


def sba_stems() -> set[str]:
    """DVR frames recorded from SBA (index tag / result "sba"; while YouTube is paused). User, 2026-10-29: never used
    for training or testing, only YouTube frames are."""
    import csv

    try:
        with (DVR / "index.csv").open(newline="") as f:
            return {Path(r["file"]).stem for r in csv.DictReader(f)
                    if "sba" in (r.get("tag") or "") or "sba" in (r.get("result") or "")}
    except OSError:
        return set()


def window_stems() -> set[str]:
    """Every frame stem a fine-tune window uses (training or test)."""
    stems: set[str] = set()
    for d in WINDOW_DIRS:
        for p in d.glob("*.json"):
            if p.stem.startswith("report") or p.stem == "refit_summary":
                continue
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            frames = meta.get("frames", []) if isinstance(meta, dict) else meta
            stems |= {f["stem"] for f in frames if isinstance(f, dict) and "stem" in f}
    return stems


def keep(stems) -> int:
    """Copy these frames (and their .json sidecars) from the DVR folder to data/kept_frames; returns how many."""
    KEPT.mkdir(parents=True, exist_ok=True)
    n = 0
    sba = sba_stems()
    for s in stems:
        if s in sba:  # SBA frames are never training or test frames
            continue
        for ext in (".jpg", ".json"):
            src, dst = DVR / f"{s}{ext}", KEPT / f"{s}{ext}"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                n += ext == ".jpg"
    return n
