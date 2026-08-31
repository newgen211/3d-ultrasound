import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8, candidates

sec = Path("data/clarius_sessions/section_106")
raws = sorted(glob.glob(str(sec / "raw_*.json")))
idxs = list(range(60, 880, 70))
fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
for ax, i in zip(axes.flat, idxs):
    meta_full = json.load(open(raws[i]))
    meta = meta_full["frame"]
    img = load_frame(Path(raws[i].replace(".json", ".bin")), meta_full)
    cands = candidates(img, meta["axial_um_per_sample"]/1000.0,
                       meta["lateral_um_per_line"]/1000.0)
    ax.imshow(to_u8(img), cmap="gray", aspect="auto")
    for c in cands:
        (cx, cy), (w, h), ang = c["ellipse"]
        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fill=False, color="red", lw=1.5))
    ax.set_title(f"f{i}  v={len(cands)}", fontsize=8); ax.axis("off")
fig.tight_layout(); fig.savefig("viz_106_sampled.png", dpi=110)
print("wrote viz_106_sampled.png")
