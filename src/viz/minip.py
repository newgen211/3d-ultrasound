#!/usr/bin/env python3
"""
minip.py — minimum-intensity projection (shows DARK/anechoic structures)

MIP keeps the brightest voxel per ray, so an anechoic tube is invisible in it.
MinIP keeps the darkest *real* voxel per ray, so the tube shows up. Background
(no-data, =0) is ignored so it doesn't swamp the projection.

    python3 minip.py                 # latest section's volume_handeye.npy
    python3 minip.py section_22
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2] / "data" / "clarius_sessions"
arg = sys.argv[1] if len(sys.argv) > 1 else None
section = (Path(arg) if arg and Path(arg).exists()
           else (root / arg) if arg
           else sorted([d for d in root.iterdir() if d.name.startswith("section_")],
                       key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)[-1])

npy = Path(section) / "volume_handeye.npy"
if not npy.exists():
    sys.exit(f"❌ {npy} not found (run reconstruct_handeye.py first)")
v = np.load(npy)

# ignore empty voxels so background 0 doesn't dominate the min
vv = np.where(v > 0, v, np.nan)

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")           # all-NaN rays -> NaN, fine
    for col, axis_, name in ((0, 2, "X–Y top-down"), (1, 1, "X–Z"), (2, 0, "Y–Z")):
        mn = np.nanmin(vv, axis=axis_).T
        mx = np.nanmax(vv, axis=axis_).T
        a0 = ax[0, col]; a1 = ax[1, col]
        a0.imshow(np.nan_to_num(mn, nan=np.nanmax(mn)), cmap="gray", origin="lower", aspect="equal")
        a0.set_title(f"MinIP {name}  (tube = dark)"); a0.axis("off")
        a1.imshow(np.nan_to_num(mx, nan=0), cmap="gray", origin="lower", aspect="equal")
        a1.set_title(f"MIP {name}  (bright wins)"); a1.axis("off")

fig.suptitle(f"{Path(section).name} — MinIP (top) vs MIP (bottom)")
out = Path(section) / "volume_minip.png"
fig.tight_layout()
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"💾 {out}")
plt.show()