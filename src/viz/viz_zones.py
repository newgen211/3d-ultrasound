import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame, to_u8
from us3d.paths import out_dir
from us3d.sections import find_section
from us3d.tube import candidates

def sheet(sec_name, idxs, out_png):
    sec = find_section(sec_name)
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
    out = out_dir("figures") / out_png
    fig.savefig(out, dpi=110)
    print("wrote", out)

# The sheets this was built for. The section_106 one was viz_ctrl.py, which
# was this same function inlined for a single section.
PRESETS = {
    "section_118": (list(range(505, 541, 3)) + list(range(694, 736, 3)), "zones_118.png"),
    "section_115": (list(range(40, 940, 75)), "sampled_115.png"),
    "section_106": (list(range(60, 880, 70)), "sampled_106.png"),
}

wanted = sys.argv[1:] or ["section_118", "section_115"]
for name in wanted:
    if name not in PRESETS:
        sys.exit("no preset for %s (have: %s)" % (name, ", ".join(sorted(PRESETS))))
    idxs, out_png = PRESETS[name]
    sheet(name, idxs, out_png)
