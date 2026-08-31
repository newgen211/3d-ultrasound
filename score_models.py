#!/usr/bin/env python3
"""Money table: old student vs Model A, same 160-frame truth.
Truth circles are auto-classified open/crescent by interior darkness (same
spirit as harvest_exemplars); Model A is an open-lumen detector, so its
primary row is scored against open truth only. Both models get both rows."""
import json, sys
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

AUD = Path("audit")
truth = json.loads((AUD / "audit_truth.json").read_text())

def classify(gray, x, y, r):
    m = np.zeros(gray.shape, np.uint8)
    cv2.circle(m, (int(x), int(y)), max(int(r), 2), 1, -1)
    interior = float(gray[m > 0].mean())
    return "open" if interior < 0.75 * np.median(gray) else "crescent"

def score(model_path, morph_filter, per_sec=None):
    model = YOLO(model_path)
    tp = fp = fn = 0
    dices = []
    for k, ent in truth.items():
        if ent.get("unusable"):
            continue
        img_p = AUD / "frames" / k
        gray = cv2.imread(str(img_p), 0)
        if gray is None:
            continue
        gt_all = ent["vessels"]
        gt = [g for g in gt_all
              if morph_filter is None or classify(gray, *g) == morph_filter]
        res = model.predict(str(img_p), verbose=False, conf=CONF)[0]
        preds = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append(((x1 + x2) / 2, (y1 + y2) / 2, (y2 - y1) / 2))
        gm = np.zeros(gray.shape, np.uint8); pm = np.zeros(gray.shape, np.uint8)
        for g in gt: cv2.circle(gm, (int(g[0]), int(g[1])), int(g[2]), 1, -1)
        for p in preds: cv2.circle(pm, (int(p[0]), int(p[1])), max(int(p[2]), 2), 1, -1)
        if gm.sum() or pm.sum():
            dices.append(2 * (gm & pm).sum() / max(gm.sum() + pm.sum(), 1))
        used = set()
        for (px, py, pr) in preds:
            hit = None
            for j, (gx, gy, gr) in enumerate(gt):
                if j in used: continue
                if (px - gx) ** 2 + (py - gy) ** 2 <= gr ** 2:
                    hit = j; break
            if hit is None: fp += 1
            else: used.add(hit); tp += 1
        fn += len(gt) - len(used)
        if per_sec is not None:
            sec = k.split("__")[0]
            d = per_sec.setdefault(sec, [0, 0, 0])
            d[0] += len([1 for _ in used]); d[1] += len(preds) - len(used); d[2] += len(gt) - len(used)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    return P, R, F, np.mean(dices), tp, fp, fn

for name, path in [("old student", "models/best_regated.pt"),
                   ("Model A    ", "models/modelA_best.pt")]:
    per = {}
    P, R, F, D, tp, fp, fn = score(path, "open", per)
    print(f"\n{name} vs open truth:  P {P:.3f}  R {R:.3f}  F1 {F:.3f}")
    for sec in sorted(per):
        t, f_, n_ = per[sec]
        p_ = t / max(t + f_, 1); r_ = t / max(t + n_, 1)
        print(f"  {sec:14s} P {p_:.2f}  R {r_:.2f}   (tp {t} fp {f_} fn {n_})")
