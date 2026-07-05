#!/usr/bin/env python3
"""
sam_overlay.py — draw SAM's REAL detection boxes from sam_detections.json

Now draws SAM's actual out_boxes_xywh (green rectangle) instead of a synthesized
circle, so you see the true vessel extent — including the lateral (horizontal)
width that beam-spread blooms. Red dot = centroid. Title shows instance count.

    python3 sam_overlay.py section_59

Needs frames_jpg/ (run sam3_track.py with --keep-jpg).
Output: sam_overlay.png in the section folder.
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_tube import find_section

section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
det = json.loads((section / "sam_detections.json").read_text())["detections"]
jpg_dir = section / "frames_jpg"
if not jpg_dir.exists():
    sys.exit("no frames_jpg/ — re-run sam3_track.py with --keep-jpg")

by_frame = {}
for d in det:
    by_frame.setdefault(d["frame_index"], []).append(d)

# pick: 4 with most instances + 4 biggest-radius + 4 median, to see the full range
inst_count = {fi: len(ds) for fi, ds in by_frame.items()}
frame_maxr = {fi: max(x["r_mm"] for x in ds) for fi, ds in by_frame.items()}
med = np.median([d["r_mm"] for d in det if np.isfinite(d["r_mm"])])
most = sorted(inst_count, key=lambda f: -inst_count[f])[:4]
worst = sorted(frame_maxr, key=lambda f: -frame_maxr[f])[:4]
mid = sorted(by_frame, key=lambda f: abs(frame_maxr[f] - med))[:4]
picks, seen = [], set()
for f in most + worst + mid:
    if f not in seen:
        picks.append(f); seen.add(f)

cols, rows = 4, int(np.ceil(len(picks) / 4))
fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
for ax, fi in zip(np.ravel(axs), picks):
    img = cv2.cvtColor(cv2.imread(str(jpg_dir / f"{fi:05d}.jpg")), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for d in by_frame[fi]:
        b = d.get("sam_box")
        if b:
            x, y, w, h = b
            if max(b) <= 1.5:                  # normalized -> pixels
                x, y, w, h = x * W, y * H, w * W, h * H
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.circle(img, (int(d["cx"]), int(d["cy"])), 3, (255, 0, 0), -1)
    ax.imshow(img)
    ax.set_title(f"frame {fi}  {len(by_frame[fi])} inst  "
                 f"r={max(x['r_mm'] for x in by_frame[fi]):.1f}mm", fontsize=8)
    ax.axis("off")
for ax in np.ravel(axs)[len(picks):]:
    ax.axis("off")
out = section / "sam_overlay.png"
fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out}  (rows: most-instances / biggest / median)")