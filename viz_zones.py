import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8, candidates

def sheet(sec_name, idxs, out_png):
    sec = Path("data/clarius_sessions") / sec_name
    raws = sorted(glob.glob(str(sec / "raw_*.json")))
    rep_p = sec / "compression_report.json"
    rep = json.loads(rep_p.read_text())["frames"] if rep_p.exists() else None
    cols = 6
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3.2*rows))
    for ax, i in zip(axes.flat, idxs):
        meta_full = json.load(open(raws[i]))
        meta = meta_full["frame"]
        img = load_frame(Path(raws[i].replace(".json", ".bin")), meta_full)
        cands = candidates(img, meta["axial_um_per_sample"]/1000.0,
                           meta["lateral_um_per_line"]/1000.0)
        ax.imshow(to_u8(img), cmap="gray", aspect="auto")
        for c in cands:
            (cx, cy), (w, h), ang = c["ellipse"]
            ax.add_patch(Ellipse((cx, cy), w, h, angle=ang,
                                 fill=False, color="red", lw=1.5))
        t = rep[i]["state"] if rep and i < len(rep) else "?"
        ax.set_title(f"f{i}  v={len(cands)}  truth={t}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(idxs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)

sheet("section_118",
      list(range(505, 541, 3)) + list(range(694, 736, 3)),
      "viz_118_zones.png")
sheet("section_115", list(range(40, 940, 75)), "viz_115_sampled.png")
