import json, glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame, to_u8
from us3d.paths import out_dir
from us3d.paths import model as _model
from us3d.sections import find_section
from ultralytics import YOLO
argv = sys.argv[1:]
MODEL_NAME = "gold_n.pt"
if "--model" in argv:
    i = argv.index("--model"); MODEL_NAME = argv[i + 1]; del argv[i:i + 2]
model = YOLO(str(_model(MODEL_NAME)))
sec = find_section(argv[0]); idxs = [int(x) for x in argv[1:]]
raws = sorted(glob.glob(str(sec / "raw_*.json")))
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
out = out_dir("figures") / ("diag_left_%s_%s.png" % (sec.name, Path(MODEL_NAME).stem))
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote %s" % out)
