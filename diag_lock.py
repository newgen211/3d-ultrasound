import json, glob, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8
from ultralytics import YOLO

sec = Path("data/clarius_sessions/section_118")
raws = sorted(glob.glob(str(sec / "raw_*.json")))
model = YOLO("models/best_regated.pt")
det = conf_max = 0
confs = []
for i in range(0, len(raws), 5):
    meta_full = json.load(open(raws[i]))
    img = load_frame(Path(raws[i].replace(".json",".bin")), meta_full)
    u8 = to_u8(img)
    ok, buf = cv2.imencode(".jpg", u8)
    rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    res = model.predict(rgb, conf=0.05, verbose=False)[0]
    if len(res.boxes):
        det += 1
        confs.append(float(res.boxes.conf.max()))
n = len(range(0, len(raws), 5))
confs = np.array(confs)
print(f"frames sampled {n}, any detection @0.05: {det} ({100*det/n:.0f}%)")
if len(confs):
    print(f"top-conf per frame: p10 {np.percentile(confs,10):.2f} "
          f"med {np.median(confs):.2f} p90 {np.percentile(confs,90):.2f}  "
          f">=0.35: {(confs>=0.35).mean()*100:.0f}%  >=0.25: {(confs>=0.25).mean()*100:.0f}%")
