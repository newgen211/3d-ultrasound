#!/usr/bin/env python3
"""
smooth_cam_log.py — smooth the camera pose stream before merging.

Servo sweeps are stop-and-go: frames clump at settle points, so per-frame
camera jitter shows up as offset slabs in the reconstruction instead of
averaging out along a continuous hand path. Smoothing the pose stream
(~0.4 s window, all 6 DOF, per marker id) restores the continuous-path
behavior. Same uniform_filter1d fix as the pose stair-stepping.

    python smooth_cam_log.py in_cam.jsonl out_cam_smooth.jsonl
    python src/pose/merge_poses_cam.py section_67 out_cam_smooth.jsonl
"""
import json, sys
import numpy as np
from scipy.ndimage import uniform_filter1d

if len(sys.argv) != 3:
    sys.exit("usage: smooth_cam_log.py in.jsonl out.jsonl")

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by_id = {}
for i, r in enumerate(rows):
    if r.get("coords"):
        by_id.setdefault(r.get("id"), []).append(i)

for mid, idx in by_id.items():
    if len(idx) < 5:
        print(f'id{mid}: {len(idx)} poses — too few, left raw')
        continue
    C = np.array([rows[i]["coords"] for i in idx], dtype=np.float64)
    T = np.array([rows[i]["t_ns"] for i in idx], dtype=np.float64)
    import numpy as _np
    rate = 1.0 / max(float(_np.median(_np.diff(T)) / 1e9), 1e-3)
    w = max(3, int(0.4 * rate))
    Cs = uniform_filter1d(C, size=w, axis=0, mode="nearest")
    for k, i in enumerate(idx):
        rows[i]["coords"] = [round(float(v), 3) for v in Cs[k]]
    print(f"id{mid}: {len(idx)} poses smoothed (window {w} samples ~0.4 s)")

with open(sys.argv[2], "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {sys.argv[2]}")