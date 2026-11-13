"""View classifier (research, 2026-10-24): DINOv2-small embedding + logistic regression on the labelled thumbnails.

Classes (data/views/thumbs/labels.csv, labelled by eye from contact sheets): kaaba (Kaaba fully in view: the camera can
be fitted to it), mataf (Mataf floor in view, Kaaba absent or cut by the frame), other (tower / aerial views, close-ups,
interiors, the imam, the city). Evaluated with stratified 5-fold cross-validation and with lighting transfer (train on
day, test on night and the reverse). Writes models/view_classifier.npz (scaler + weights) and prints the report.

  .venv/Scripts/python scripts/view_classifier.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAP3 = {"kaaba": "kaaba", "kaaba_partial": "mataf", "mataf": "mataf", "tower": "other", "other": "other"}
CLASSES = ["kaaba", "mataf", "other"]


def embed(images: list[np.ndarray]) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()
    out = []
    for im in images:
        small = cv2.resize(im, (448, 252), interpolation=cv2.INTER_AREA)
        inp = proc(images=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), return_tensors="pt").to("cuda")
        with torch.no_grad():
            out.append(model(**inp).last_hidden_state[0, 0].float().cpu().numpy())
    return np.vstack(out)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = list(csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "labels.csv")))
    X = embed([cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"])) for r in rows])
    y = np.array([CLASSES.index(MAP3[r["label"]]) for r in rows])
    night = np.array([r["night"] == "1" for r in rows])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"{len(rows)} thumbnails, 5-fold CV accuracy {np.mean(pred == y):.3f}")
    print(classification_report(y, pred, target_names=CLASSES, digits=3))
    print("confusion (rows true, cols predicted):", CLASSES)
    print(confusion_matrix(y, pred))
    for tr, te, name in ((~night, night, "train day -> test night"), (night, ~night, "train night -> test day")):
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        rec = {c: round(float(np.mean(p[y[te] == i] == i)), 3) for i, c in enumerate(CLASSES) if (y[te] == i).any()}
        print(f"{name}: accuracy {np.mean(p == y[te]):.3f}, recall per class {rec}")
    clf.fit(X, y)
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    out = ROOT / "models" / "view_classifier.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_, intercept=lr.intercept_, classes=np.array(CLASSES))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
