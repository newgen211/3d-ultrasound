#!/usr/bin/env python3
"""
make_hold_log.py — turn a drag log's held poses into a replayable dwell script.

Finds stillness segments in a pose_logger jsonl (joint speed low for >=1.2 s),
takes each segment's median joint angles, dedupes near-identical poses, and
emits a joint log where the arm visits each pose and DWELLS ~3 s (servos
holding — backlash taken up, zero shake). Play with execute_sweep; its
exec_*.jsonl (fresh timestamps, held poses) is the new cobot log for
solve_bridge.

    python make_hold_log.py data/pose_logs/pi_pose_log.jsonl > hold_replay.jsonl
"""
import json, sys
import numpy as np

if len(sys.argv) != 2:
    sys.exit("usage: make_hold_log.py pi_pose_log.jsonl > hold_replay.jsonl")

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("angles") and r.get("coords")]
T = np.array([r["t_ns"] for r in rows], np.int64)
A = np.array([r["angles"] for r in rows], float)
C = np.array([r["coords"] for r in rows], float)
o = np.argsort(T); T, A, C = T[o], A[o], C[o]

from scipy.ndimage import uniform_filter1d
dt = np.diff(T) / 1e9
rate = 1.0 / max(float(np.median(dt)), 1e-3)
w = max(3, int(0.35 * rate))
As = uniform_filter1d(A, size=w, axis=0, mode="nearest")   # tremor averages out
jspeed = np.zeros(len(T))
jspeed[1:] = np.abs(np.diff(As, axis=0)).max(1) / np.maximum(dt, 1e-3)  # deg/s

# stillness segments >= 1.0 s (limp-arm hand-holds: tremor-tolerant gate)
GATE = 4.0
holds = []
i = 0
while i < len(T):
    if jspeed[i] < GATE:
        j = i
        while j + 1 < len(T) and jspeed[j + 1] < GATE:
            j += 1
        if (T[j] - T[i]) / 1e9 >= 1.0:
            holds.append((i, j))
        i = j + 1
    else:
        i += 1

# median angles per hold, dedupe (<5 deg from a kept pose)
poses = []
for i, j in holds:
    ang = np.median(A[i:j + 1], axis=0)
    coo = np.median(C[i:j + 1], axis=0)
    if all(np.abs(ang - p[0]).max() >= 5.0 for p in poses):
        poses.append((ang, coo))
print(f"# {len(holds)} stillness segments -> {len(poses)} distinct held poses",
      file=sys.stderr)
if len(poses) < 8:
    print("# !! few poses — bridge wants 12+; consider a longer source drag",
          file=sys.stderr)

# emit: travel 4 s between poses, dwell 3 s at each (duplicate sample holds it)
t = 0
for ang, coo in poses:
    for dwell in (0, 3_000_000_000):
        print(json.dumps({"t_ns": int(t + dwell),
                          "coords": [round(float(v), 1) for v in coo],
                          "angles": [round(float(v), 2) for v in ang]}))
    t += 7_000_000_000
print(f"# total runtime ~{len(poses)*7} s", file=sys.stderr)