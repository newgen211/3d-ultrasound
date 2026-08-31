import json, glob, sys
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8
from ultralytics import YOLO

TRUTH = json.load(open("audit/audit_truth.json"))
MATCH_MM = 3.0

def entries():
    for key, val in TRUTH.items():
        if val.get("unusable"): continue
        sec, fs = key.split("__")
        yield sec, int(fs.split(".")[0]), val.get("vessels", [])

def is_open(u8, x, y, r):
    x, y, r = int(x), int(y), max(int(r*0.6), 2)
    patch = u8[max(y-r,0):y+r, max(x-r,0):x+r]
    return patch.size > 0 and patch.mean() < 0.75*np.median(u8)

for mp in ["models/best_regated.pt", "models/bench_s.pt", "models/bench_m.pt", "models/gold_n.pt"]:
    model = YOLO(mp)
    allt = [0,0,0]
    for sec, fi, vessels in entries():
        g = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
        if fi >= len(g): continue
        mf = json.load(open(g[fi]))
        u8 = to_u8(load_frame(Path(g[fi].replace(".json",".bin")), mf))
        ax_, lat = mf["frame"]["axial_um_per_sample"]/1000.0, mf["frame"]["lateral_um_per_line"]/1000.0
        ok, buf = cv2.imencode(".jpg", u8)
        res = model.predict(cv2.imdecode(buf, cv2.IMREAD_COLOR), conf=0.25, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy() if len(res.boxes) else []
        dpos = [(b[0]*lat, b[1]*ax_) for b, c in
                zip(res.boxes.xywh.cpu().numpy(), cls) if int(c) == 0]
        tpos = [(v[0]*lat, v[1]*ax_) for v in vessels if is_open(u8, v[0], v[1], v[2])]
        used = set(); tp = 0
        for tx, ty in tpos:
            best, bj = 1e9, -1
            for j,(dx,dy) in enumerate(dpos):
                if j in used: continue
                d = ((dx-tx)**2+(dy-ty)**2)**0.5
                if d < best: best, bj = d, j
            if bj >= 0 and best <= MATCH_MM: used.add(bj); tp += 1
        allt[0]+=tp; allt[1]+=len(dpos)-tp; allt[2]+=len(tpos)-tp
    tp, fp, fn = allt
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    print(f"{Path(mp).stem:14s} P {P:.3f}  R {R:.3f}  F1 {F1:.3f}  (tp {tp} fp {fp} fn {fn})")
