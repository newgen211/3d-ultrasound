#!/usr/bin/env python3
"""
vet_frames.py — pre-click quality check for hand-eye capture frames.

For each captured frame, examines id0's rotation stream in the cam log within
a +-0.4 s window: a settled, flip-free pose reads STEADY; a PnP branch flip or
unsettled motion reads as a rotation cliff. Click only the GOOD ones.

Usage:
    python3 vet_frames.py section_78 data/pose_logs/probe_pose_log.jsonl
"""
import glob, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) != 3:
    sys.exit("usage: vet_frames.py section_<N> cam_log.jsonl")

secdir = sys.argv[1] if os.path.isdir(sys.argv[1]) \
         else os.path.join("data", "clarius_sessions", sys.argv[1])
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
T, RR = [], []
for r in rows:
    if r.get("id") == 0 and r.get("coords"):
        T.append(int(r["t_ns"]))
        RR.append(Rotation.from_euler("xyz", r["coords"][3:6], degrees=True))
T = np.array(T)
order = np.argsort(T); T = T[order]; RR = [RR[i] for i in order]

W = int(0.4e9)
print(f"{'frame':28s} {'n':>3s} {'max step':>9s} {'spread':>8s}  verdict")
good = bad = 0
for sc in sorted(glob.glob(os.path.join(secdir, "raw_*.json"))):
    h = json.load(open(sc)).get("host_timestamp_ns")
    stem = os.path.basename(sc)[:-5]
    if h is None:
        print(f"{stem:28s}   -         -        -  no host timestamp"); continue
    i0, i1 = np.searchsorted(T, h - W), np.searchsorted(T, h + W)
    if i1 - i0 < 4:
        print(f"{stem:28s} {i1-i0:3d}         -        -  TOO FEW POSES — id0 dropped?")
        bad += 1; continue
    Rw = RR[i0:i1]
    steps = [np.degrees((Rw[k].inv() * Rw[k+1]).magnitude()) for k in range(len(Rw)-1)]
    mean = Rw[0]
    spread = max(np.degrees((mean.inv() * r).magnitude()) for r in Rw)
    max_step = max(steps)
    if max_step > 6.0:
        verdict = "FLIP/JUMP — skip"; bad += 1
    elif spread > 3.0:
        verdict = "not settled — skip"; bad += 1
    else:
        verdict = "GOOD"; good += 1
    print(f"{stem:28s} {len(Rw):3d} {max_step:8.2f}° {spread:7.2f}°  {verdict}")
print(f"\n{good} good / {bad} to skip. Click only the GOOD ones in digitize_beads.")