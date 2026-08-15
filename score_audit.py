#!/usr/bin/env python3
"""
Score teacher (SAM) and/or student vs audited truth.
Circle-mask Dice + detection P/R/F1 (match = center inside truth circle).
    python3 score_audit.py                       # teacher vs truth
    python3 score_audit.py --labels runs/detect/runs/studentXX/labels --tag student
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

AUD = Path("audit")
man = json.loads((AUD / "audit_manifest.json").read_text())
truth = json.loads((AUD / "audit_truth.json").read_text())


def mask_of(circles, shape):
    m = np.zeros(shape, np.uint8)
    for cx, cy, r in circles:
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    return m


def teacher_circles(k):
    out = []
    for d in man[k]["teacher"]:
        cx = d.get("cx_px", d.get("cx")); cy = d.get("cy_px", d.get("cy"))
        r = d.get("r_px", d.get("r"))
        if r is None and d.get("r_mm") is not None:
            r = d["r_mm"] / 0.051333          # axial um/sample -> px
        if r is None: r = 8
        if cx is not None and cy is not None:
            out.append((float(cx), float(cy), float(r)))
    return out


def student_circles(k, labels_dir, shape):
    # yolo txt: cls cx cy w h [conf], normalized
    stem = k.replace(".jpg", "")
    f = labels_dir / f"{stem}.txt"
    out = []
    if f.exists():
        H, W = shape
        for line in f.read_text().splitlines():
            v = line.split()
            if len(v) >= 5:
                cx, cy, w, h = (float(v[1]) * W, float(v[2]) * H,
                                float(v[3]) * W, float(v[4]) * H)
                out.append((cx, cy, (w + h) / 4))
    return out


def score(pred_fn, tag):
    dices, tp, fp, fn = [], 0, 0, 0
    per_sec = defaultdict(lambda: [[], 0, 0, 0])
    for k, ent in truth.items():
        if ent.get("unusable") or k not in man:
            continue
        img = cv2.imread(str(AUD / "frames" / k), 0)
        gt = ent["vessels"]
        pr = pred_fn(k, img.shape)
        gm, pm = mask_of(gt, img.shape), mask_of(pr, img.shape)
        inter = (gm & pm).sum()
        d = 2 * inter / max(gm.sum() + pm.sum(), 1) if (gm.sum() or pm.sum()) else 1.0
        dices.append(d)
        used = set()
        m_tp = 0
        for (px, py, pr_) in pr:
            hit = None
            for j, (gx, gy, gr) in enumerate(gt):
                if j in used: continue
                if (px - gx) ** 2 + (py - gy) ** 2 <= gr ** 2:
                    hit = j; break
            if hit is None:
                fp += 1; per_sec[man[k]["section"]][2] += 1
            else:
                used.add(hit); m_tp += 1
        tp += m_tp; fn += len(gt) - len(used)
        s = per_sec[man[k]["section"]]
        s[0].append(d); s[1] += m_tp; s[3] += len(gt) - len(used)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    print(f"\n== {tag} vs audited truth ==")
    print(f"Dice (mask): mean {np.mean(dices):.3f} median {np.median(dices):.3f}")
    print(f"Detection:   P {P:.3f}  R {R:.3f}  F1 {F:.3f}   (tp {tp} fp {fp} fn {fn})")
    for s, (dd, t, f_, n_) in sorted(per_sec.items()):
        p_ = t / max(t + f_, 1); r_ = t / max(t + n_, 1)
        print(f"  {s:12s} dice {np.mean(dd):.3f}  P {p_:.2f} R {r_:.2f}")


ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=None)
ap.add_argument("--tag", default="student")
a = ap.parse_args()

score(lambda k, sh: teacher_circles(k), "SAM teacher")
if a.labels:
    L = Path(a.labels)
    score(lambda k, sh: student_circles(k, L, sh), a.tag)
