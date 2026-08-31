#!/usr/bin/env python3
"""E1: pairwise centerline distance across identical-command flights."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SECS = ["section_106", "section_108", "section_109", "section_111"]
lines = {}
for s in SECS:
    d = json.loads((Path("data/clarius_sessions") / s / "centerline.json").read_text())
    assert not d["straightened"], f"{s} centerline was straightened — rerun vessel_tube without --straighten"
    lines[s] = np.array(d["centerline_mm"])

def dist_a_to_b(A, B):
    """for each point of A, distance to nearest point of B (B densified)"""
    t = np.linspace(0, 1, len(B))
    tt = np.linspace(0, 1, 400)
    Bd = np.column_stack([np.interp(tt, t, B[:, k]) for k in range(3)])
    return np.array([np.min(np.linalg.norm(Bd - a, axis=1)) for a in A])

# common overlap: crop each pair to the along-axis interval both cover
print("pairwise mean / p95 centerline distance (mm):")
allpair = []
for i, a in enumerate(SECS):
    for b in SECS[i+1:]:
        A, B = lines[a], lines[b]
        axis = (np.linalg.svd(np.vstack([A, B]) - np.vstack([A, B]).mean(0))[2])[0]
        pa, pb = A @ axis, B @ axis
        lo, hi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
        Ac = A[(pa >= lo) & (pa <= hi)]
        Bc = B[(pb >= lo) & (pb <= hi)]
        d = np.concatenate([dist_a_to_b(Ac, Bc), dist_a_to_b(Bc, Ac)])
        allpair.append(d)
        print(f"  {a.split('_')[1]} vs {b.split('_')[1]}:  mean {d.mean():.2f}  p95 {np.percentile(d,95):.2f}")
d = np.concatenate(allpair)
print(f"\nE1 REPEATABILITY: mean {d.mean():.2f} mm, median {np.median(d):.2f} mm, p95 {np.percentile(d,95):.2f} mm across {len(SECS)} flights")

fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111, projection="3d")
for s, C in lines.items():
    ax.plot(C[:, 0], C[:, 1], C[:, 2], label=s.replace("section_", "flight "), lw=2)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
ax.legend(); ax.set_title("E1: four identical-command flights — recovered centerlines")
plt.tight_layout(); plt.savefig("e1_centerlines.png", dpi=140)
print("saved e1_centerlines.png")
