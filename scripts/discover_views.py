"""Find camera framings that recur in the broadcast (candidates for new counting views), read-only.

Thumbnails in data/views/thumbs/ (480x270, privacy-approved survey images) are matched pairwise with SIFT +
RANSAC at thumbnail resolution. Two thumbnails are "the same framing" when the homography is strong (>= --min-inliers)
and the zoom between them is within +-20 %. Connected groups are candidate views. Per group: size, Makkah hours and
periods it appeared in, whether it is one of the existing reference views (matched against them), and a contact
sheet row for review (data/views/discover/sheet.jpg, thumbnails only). Writes data/views/discover/groups.json.

Run:  .venv/Scripts/python scripts/discover_views.py [--min-inliers 30] [--min-size 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.view_matcher import RATIO_TEST, ViewMatcher, load_config, zoom_factor, _sane  # noqa: E402

THUMBS = ROOT / "data" / "views" / "thumbs"
OUT = ROOT / "data" / "views" / "discover"


def period(makkah: str) -> str:
    h = int(makkah[11:13])
    return "day" if 6 <= h < 16 else "evening" if 16 <= h < 22 else "night"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    rows = list(csv.DictReader(open(THUMBS / "index.csv", newline="")))
    imgs = [cv2.imread(str(THUMBS / r["file"])) for r in rows]
    keep = [i for i, im in enumerate(imgs) if im is not None]
    rows, imgs = [rows[i] for i in keep], [imgs[i] for i in keep]
    m = ViewMatcher({}, cfg, work_width=480, min_inliers=args.min_inliers)
    feats = [m._features(im) for im in imgs]
    n = len(imgs)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    size = (480, 270)
    for i in range(n):
        ki, di = feats[i]
        if len(di) < args.min_inliers:
            continue
        for j in range(i + 1, n):
            kj, dj = feats[j]
            if len(dj) < args.min_inliers:
                continue
            pairs = m.flann.knnMatch(di, dj, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
            if len(good) < args.min_inliers:
                continue
            H, inl = cv2.findHomography(ki[[g.queryIdx for g in good]], kj[[g.trainIdx for g in good]], cv2.RANSAC, 2.0)
            if H is None or int(inl.sum()) < args.min_inliers or not _sane(H, size):
                continue
            if abs(zoom_factor(H, size, size) - 1) <= 0.2:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    refs = {v["id"]: cv2.imread(str(ROOT / v["ref_image"])) for v in cfg["views"] if (ROOT / v["ref_image"]).exists()}
    known = ViewMatcher(refs, cfg, work_width=480, min_inliers=args.min_inliers)
    out = []
    for gid, (root, idx) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if len(idx) < args.min_size:
            continue
        hours = sorted({rows[i]["makkah_time"][:13] for i in idx})
        pers = defaultdict(int)
        for i in idx:
            pers[period(rows[i]["makkah_time"])] += 1
        hits = defaultdict(list)
        for i in idx:
            km = known.match(imgs[i])
            if km is not None:
                hits[km.view_id].append(round(km.zoom, 2))
        # representative = the member with most matches inside the group (most central framing)
        rep = idx[len(idx) // 2]
        out.append({"group": gid, "size": len(idx), "periods": dict(pers), "hours": len(hours),
                    "first": rows[idx[0]]["makkah_time"], "last": rows[idx[-1]]["makkah_time"],
                    "known_view": max(hits, key=lambda k: len(hits[k])) if hits else None,
                    "known_zooms": {k: v for k, v in hits.items()}, "rep": rows[rep]["file"],
                    "members": [rows[i]["file"] for i in idx]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "groups.json").write_text(json.dumps(out, indent=1))
    sheet = []
    for g in out[:16]:
        picks = [g["members"][k] for k in np.linspace(0, len(g["members"]) - 1, min(4, len(g["members"]))).astype(int)]
        tiles = [cv2.resize(cv2.imread(str(THUMBS / p)), (240, 135)) for p in picks]
        tiles += [np.zeros((135, 240, 3), np.uint8)] * (4 - len(tiles))
        label = np.zeros((135, 260, 3), np.uint8)
        for k, txt in enumerate([f"group {g['group']}  n={g['size']}", f"hours {g['hours']}",
                                 ", ".join(f"{p} {c}" for p, c in g["periods"].items()),
                                 f"known: {g['known_view'] or '-'}"]):
            cv2.putText(label, txt, (8, 26 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        sheet.append(np.hstack([label] + tiles))
    if sheet:
        cv2.imwrite(str(OUT / "sheet.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{n} thumbnails -> {len(out)} groups with >= {args.min_size} frames")
    for g in out:
        print(f"  group {g['group']:>3}: n={g['size']:>3} hours={g['hours']:>2} {dict(g['periods'])} known={g['known_view']} "
              f"{g['known_zooms'] if g['known_zooms'] else ''} rep={g['rep']}")


if __name__ == "__main__":
    main()
