#!/usr/bin/env python3
"""Turn audited truth circles into per-section SAM exemplar boxes (normalized xywh)."""
import json
from pathlib import Path
import numpy as np
import cv2

AUD = Path("audit")
truth = json.loads((AUD / "audit_truth.json").read_text())
out = {}
for k, ent in truth.items():
    if ent.get("unusable") or not ent["vessels"]:
        continue
    sec, fname = k.split("__")
    img = cv2.imread(str(AUD / "frames" / k), 0)
    H, W = img.shape
    med = max(float(np.median(img)), 1.0)
    for cx, cy, r in ent["vessels"]:
        m = np.zeros_like(img)
        cv2.circle(m, (int(cx), int(cy)), max(int(r), 2), 1, -1)
        inner = float(img[m > 0].mean())
        cls = "open" if inner < 0.75 * med else "crescent"
        side = 2.2 * r
        x, y = max(cx - side / 2, 0), max(cy - side / 2, 0)
        w, h = min(side, W - x), min(side, H - y)
        out.setdefault(sec, {}).setdefault(cls, []).append(
            dict(frame=int(fname.replace(".jpg", "")),
                 box=[x / W, y / H, w / W, h / H],
                 quality=abs(inner - med) / med))
# keep top-3 per class per section, best contrast first
for sec in out:
    for cls in out[sec]:
        out[sec][cls] = sorted(out[sec][cls], key=lambda d: -d["quality"])[:3]
    counts = {c: len(v) for c, v in out[sec].items()}
    print(f"{sec}: {counts}")
Path("exemplars.json").write_text(json.dumps(out, indent=1))
print("wrote exemplars.json")
