import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8

sec = sys.argv[1] if len(sys.argv) > 1 else "section_106"
rows = [json.loads(l) for l in open(f"data/clarius_sessions/{sec}/gauge_replay.jsonl")]
locked = [r for r in rows if r["cx"] is not None]
first = locked[0]["frame"]
idxs = [first + k for k in (0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 550, 600)]
byf = {r["frame"]: r for r in rows}
raws = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
fig, axes = plt.subplots(2, 6, figsize=(19, 6.6))
for ax, i in zip(axes.flat, idxs):
    if i >= len(raws): ax.axis("off"); continue
    mf = json.load(open(raws[i]))
    u8 = to_u8(load_frame(Path(raws[i].replace(".json", ".bin")), mf))
    ax.imshow(u8, cmap="gray", aspect="auto")
    r = byf.get(i)
    if r and r["cx"] is not None:
        ax.add_patch(Rectangle((r["cx"]-r["ww"], r["cy"]-r["hw"]),
                               2*r["ww"], 2*r["hw"],
                               fill=False, color="lime", lw=1.6))
    t = f"f{i} {r['state']}" + (f" r={r['ratio']}" if r and r["ratio"] is not None else "")
    ax.set_title(t, fontsize=8); ax.axis("off")
fig.tight_layout(); fig.savefig(f"viz_roi_{sec}.png", dpi=110)
print(f"wrote viz_roi_{sec}.png")
