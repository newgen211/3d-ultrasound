#!/usr/bin/env python3
"""Score v2 SAM labels against the audited 160-frame truth. Same rules as
score_audit.py (circle-mask Dice; detection match = center in truth circle)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

AUD = Path("audit")
man = json.loads((AUD / "audit_manifest.json").read_text())
truth = json.loads((AUD / "audit_truth.json").read_text())

# v2 detections indexed [section][frame] -> circles
v2 = {}
for sec_dir in Path("data/clarius_sessions").glob("section_*"):
    p = sec_dir / "sam_detections_v2.json"
    if not p.exists():
        continue
    by_frame = defaultdict(list)
    for d in json.loads(p.read_text())["detections"]:
        r_px = (d["r_mm"] / 0.051333) if d.get("r_mm") and np.isfinite(d["r_mm"]) else 8.0
        by_frame[int(d["frame_index"])].append(
            (float(d["cx"]), float(d["cy"]), max(float(r_px), 2.0), d.get("morph", "?")))
    v2[sec_dir.name] = by_frame

def mask_of(circles, shape):
    m = np.zeros(shape, np.uint8)
    for c in circles:
        cv2.circle(m, (int(round(c[0])), int(round(c[1]))), int(round(c[2])), 1, -1)
    return m

dices, tp, fp, fn = [], 0, 0, 0
per = defaultdict(lambda: [[], 0, 0, 0])
morph_hits = defaultdict(int)
for k, ent in truth.items():
    if ent.get("unusable") or k not in man:
        continue
    sec, fname = k.split("__")
    if sec not in v2:
        continue
    img = cv2.imread(str(AUD / "frames" / k), 0)
    gt = ent["vessels"]
    pr = v2[sec].get(int(fname.replace(".jpg", "")), [])
    gm, pm = mask_of(gt, img.shape), mask_of(pr, img.shape)
    inter = (gm & pm).sum()
    d = 2 * inter / max(gm.sum() + pm.sum(), 1) if (gm.sum() or pm.sum()) else 1.0
    dices.append(d)
    used = set(); m_tp = 0
    for (px, py, pr_, morph) in pr:
        hit = None
        for j, (gx, gy, gr) in enumerate(gt):
            if j in used: continue
            if (px - gx) ** 2 + (py - gy) ** 2 <= gr ** 2:
                hit = j; break
        if hit is None:
            fp += 1; per[sec][2] += 1
        else:
            used.add(hit); m_tp += 1; morph_hits[morph] += 1
    tp += m_tp; fn += len(gt) - len(used)
    per[sec][0].append(d); per[sec][1] += m_tp; per[sec][3] += len(gt) - len(used)

P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
F = 2 * P * R / max(P + R, 1e-9)
print("== SAM v2 teacher vs audited truth ==")
print(f"Dice: mean {np.mean(dices):.3f}   Detection: P {P:.3f}  R {R:.3f}  F1 {F:.3f}  (tp {tp} fp {fp} fn {fn})")
print(f"true-positive morphs: {dict(morph_hits)}")
for s, (dd, t, f_, n_) in sorted(per.items()):
    p_ = t / max(t + f_, 1); r_ = t / max(t + n_, 1)
    print(f"  {s:12s} dice {np.mean(dd):.3f}  P {p_:.2f}  R {r_:.2f}")
