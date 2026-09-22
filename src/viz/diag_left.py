import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8
from ultralytics import YOLO
model = YOLO("models/gold_n.pt")
sec = sys.argv[1]; idxs = [int(x) for x in sys.argv[2:]]
raws = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
fig, axes = plt.subplots(2, (len(idxs)+1)//2, figsize=(4*((len(idxs)+1)//2), 8))
for ax, i in zip(np.array(axes).flat, idxs):
    mf = json.load(open(raws[i]))
    u8 = to_u8(load_frame(Path(raws[i].replace(".json",".bin")), mf))
    ok, buf = cv2.imencode(".jpg", u8)
    res = model.predict(cv2.imdecode(buf, cv2.IMREAD_COLOR), conf=0.01, verbose=False)[0]
    ax.imshow(u8, cmap="gray", aspect="auto")
    for b, c in zip(res.boxes.xywh.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        col = "lime" if c >= 0.25 else ("yellow" if c >= 0.1 else "red")
        ax.add_patch(Rectangle((b[0]-b[2]/2, b[1]-b[3]/2), b[2], b[3],
                     fill=False, color=col, lw=1.4))
        ax.text(b[0]-b[2]/2, b[1]-b[3]/2-3, f"{c:.2f}", color=col, fontsize=7)
    ax.set_title(f"f{i}", fontsize=9); ax.axis("off")
fig.tight_layout(); fig.savefig(f"diag_left_{sec}.png", dpi=110)
print(f"wrote diag_left_{sec}.png")
