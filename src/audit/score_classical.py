import json, sys, glob
import numpy as np
from pathlib import Path
sys.path.insert(0, "src/segment")
import segment_tube as st
from segment_tube import load_frame, to_u8, candidates

TRUTH = json.load(open("audit/audit_truth.json"))
MATCH_MM = 3.0

def entries():
    for key, val in TRUTH.items():
        if val.get("unusable"): continue
        sec, fs = key.split("__")
        fi = int(fs.split(".")[0])
        yield sec, fi, val.get("vessels", [])

def is_open(u8, x, y, r):
    x, y, r = int(x), int(y), max(int(r*0.6), 2)
    patch = u8[max(y-r,0):y+r, max(x-r,0):x+r]
    return patch.size > 0 and patch.mean() < 0.75*np.median(u8)

def run(tag, tweak):
    saved = (st.TOP_CROP_MM, st.SMOOTH_PCT, st.DARK_PCT)
    tweak()
    rows, allt = {}, [0,0,0]
    for sec, fi, vessels in entries():
        g = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
        if fi >= len(g): continue
        jp = Path(g[fi])
        meta_full = json.load(open(jp))
        meta = meta_full["frame"]
        img = load_frame(Path(str(jp).replace(".json",".bin")), meta_full)
        u8 = to_u8(img)
        ax, lat = meta["axial_um_per_sample"]/1000.0, meta["lateral_um_per_line"]/1000.0
        tpos = [(v[0]*lat, v[1]*ax) for v in vessels if is_open(u8, v[0], v[1], v[2])]
        dets = candidates(img, ax, lat)
        dpos = [(d["cx"]*lat, d["cy"]*ax) for d in dets]
        used = set(); tp = 0
        for tx, ty in tpos:
            best, bj = 1e9, -1
            for j,(dx,dy) in enumerate(dpos):
                if j in used: continue
                dist = ((dx-tx)**2+(dy-ty)**2)**0.5
                if dist < best: best, bj = dist, j
            if bj >= 0 and best <= MATCH_MM:
                used.add(bj); tp += 1
        fp, fn = len(dpos)-tp, len(tpos)-tp
        r = rows.setdefault(sec,[0,0,0])
        r[0]+=tp; r[1]+=fp; r[2]+=fn
        for i,v in enumerate((tp,fp,fn)): allt[i]+=v
    st.TOP_CROP_MM, st.SMOOTH_PCT, st.DARK_PCT = saved
    tp,fp,fn = allt
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    print(f"\n== {tag} ==  P {P:.3f}  R {R:.3f}  F1 {F1:.3f}  (tp {tp} fp {fp} fn {fn})")
    for sec in sorted(rows):
        tp,fp,fn = rows[sec]
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        print(f"  {sec:12s} P {p:.2f}  R {r:.2f}   (tp {tp} fp {fp} fn {fn})")

run("classical AS-IS", lambda: None)
run("classical NO-CROP", lambda: setattr(st, "TOP_CROP_MM", 0.5))

ds, above = [], 0
for sec, fi, vessels in entries():
    g = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
    if fi >= len(g): continue
    ax = json.load(open(g[fi]))["frame"]["axial_um_per_sample"]/1000.0
    for v in vessels: ds.append(v[1]*ax)
ds = np.array(ds)
print(f"\ntruth vessel depths (ALL morphs): n={len(ds)}  above 3.0mm: "
      f"{(ds<3.0).sum()} ({100*(ds<3.0).mean():.0f}%)  "
      f"p10 {np.percentile(ds,10):.1f}  med {np.median(ds):.1f}  "
      f"p90 {np.percentile(ds,90):.1f} mm")

# ---- student arm: best_regated @ 0.25, jpg domain, same matching ----
import cv2
from ultralytics import YOLO
model = YOLO("models/best_regated.pt")

rows, allt = {}, [0,0,0]
for sec, fi, vessels in entries():
    g = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
    if fi >= len(g): continue
    meta_full = json.load(open(g[fi]))
    meta = meta_full["frame"]
    img = load_frame(Path(g[fi].replace(".json",".bin")), meta_full)
    u8 = to_u8(img)
    ax, lat = meta["axial_um_per_sample"]/1000.0, meta["lateral_um_per_line"]/1000.0
    jp = Path(f"data/clarius_sessions/{sec}/frames_jpg/{fi:05d}.jpg")
    if jp.exists():
        rgb = cv2.imread(str(jp))
    else:
        ok, buf = cv2.imencode(".jpg", u8)
        rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    sy, sx = u8.shape[0]/rgb.shape[0], u8.shape[1]/rgb.shape[1]
    res = model.predict(rgb, conf=0.25, verbose=False)[0]
    dpos = []
    for b in res.boxes.xywh.cpu().numpy():
        dpos.append((b[0]*sx*lat, b[1]*sy*ax))
    tpos = [(v[0]*lat, v[1]*ax) for v in vessels if is_open(u8, v[0], v[1], v[2])]
    used = set(); tp = 0
    for tx, ty in tpos:
        best, bj = 1e9, -1
        for j,(dx,dy) in enumerate(dpos):
            if j in used: continue
            dist = ((dx-tx)**2+(dy-ty)**2)**0.5
            if dist < best: best, bj = dist, j
        if bj >= 0 and best <= MATCH_MM:
            used.add(bj); tp += 1
    fp, fn = len(dpos)-tp, len(tpos)-tp
    r = rows.setdefault(sec,[0,0,0])
    r[0]+=tp; r[1]+=fp; r[2]+=fn
    for i,v in enumerate((tp,fp,fn)): allt[i]+=v
tp,fp,fn = allt
P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
F1 = 2*P*R/(P+R) if P+R else 0
print(f"\n== student best_regated @0.25 (jpg) ==  P {P:.3f}  R {R:.3f}  F1 {F1:.3f}  (tp {tp} fp {fp} fn {fn})")
for sec in sorted(rows):
    tp,fp,fn = rows[sec]
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    print(f"  {sec:12s} P {p:.2f}  R {r:.2f}   (tp {tp} fp {fp} fn {fn})")
