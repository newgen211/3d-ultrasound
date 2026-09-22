#!/usr/bin/env python3
# Shape repeatability v2: station-init Kabsch + ICP refinement, NN residuals.
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import POSE_LOGS, SESSIONS

ROOT = SESSIONS
SECS = ["section_106", "section_108", "section_109", "section_111"]
N = 200

def load(sec):
    return np.asarray(json.load(open(ROOT / sec / "centerline.json"))["centerline_mm"], float)

def resample(P, n=N):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, P[:, k]) for k in range(3)])

def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    ds = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, ds]) @ U.T
    return R, ca - R @ cb

def nn_stats(P, Q):
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    both = np.concatenate([d.min(1), d.min(0)])
    return both.mean(), np.percentile(both, 95), both.max()

def icp(P, Q, iters=15):
    """Align Q onto P. Station init, then NN-correspondence Kabsch."""
    best = None
    for Q0 in (Q, Q[::-1]):
        A, B = resample(P), resample(Q0)
        R, t = kabsch(A, B)                       # station init
        Bc = (R @ B.T).T + t
        for _ in range(iters):                    # ICP refine
            idx = np.linalg.norm(Bc[:, None, :] - A[None, :, :], axis=2).argmin(1)
            R2, t2 = kabsch(A[idx], Bc)
            Bc = (R2 @ Bc.T).T + t2
        cand = nn_stats(A, Bc)
        if best is None or cand[0] < best[0]: best = cand
    return best

C = {s: load(s) for s in SECS}
print(f"{'pair':<14}{'mean':>7}{'p95':>8}{'max':>8}")
for i in range(len(SECS)):
    for j in range(i + 1, len(SECS)):
        m, p, mx = icp(C[SECS[i]], C[SECS[j]])
        print(f"{SECS[i][-3:]} vs {SECS[j][-3:]}  {m:7.2f}{p:8.2f}{mx:8.2f}")
