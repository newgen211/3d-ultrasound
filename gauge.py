#!/usr/bin/env python3
"""gauge.py v5.1 — detector-anchored, geometry-gauged, CLASS-AWARE.
class 0 (open): anchors ROI, blocks pinch, migrate target.
class 1 (crushed): anchors ROI + POSITIVE pinch evidence (fast PRESSED).
Usage: python3 gauge.py section_106 [models/gold_n.pt]
"""
import json, glob, sys
from pathlib import Path
from collections import Counter, deque
import numpy as np
import cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8
from ultralytics import YOLO

MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "models/gold_n.pt"
DET_CONF   = 0.25
GATE_MM    = 5.0
ANCHOR_EMA = 0.4
VOTE_WIN, VOTE_K, CLUSTER_MM = 40, 8, 4.0
BUFFER     = 1.10
SEARCH_PX  = 12
DEAD_FRAC  = 0.4
PINCH_CONFIRM = 5
CRUSH_CONFIRM = 2      # near class-1 evidence: faster PRESSED
LOST_TIMEOUT  = 400

def profile_h(u8, cy, cx, half_h, half_w, ax):
    r0, r1 = max(int(cy-half_h),0), min(int(cy+half_h), u8.shape[0])
    c0, c1 = max(int(cx-half_w),0), min(int(cx+half_w), u8.shape[1])
    if r1-r0 < 6 or c1-c0 < 3: return None
    prof = u8[r0:r1, c0:c1].mean(axis=1)
    n = len(prof); third = max(n//3, 1)
    wall = min(prof[:third].max(), prof[-third:].max())
    lum = prof[third:-third].min() if n > 2*third else prof.min()
    if wall <= lum: return None
    th = lum + 0.35*(wall - lum)
    below = prof < th
    best = run = s = 0; bs = be = 0
    for i, b in enumerate(below):
        run = run+1 if b else 0
        if b and run == 1: s = i
        if run > best: best, bs, be = run, s, i
    return dict(h_mm=best*ax, contrast=float(wall-lum),
                row=r0+(bs+be)/2.0, lum=float(lum))

def detections(model, u8, conf):
    ok, buf = cv2.imencode(".jpg", u8)
    res = model.predict(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                        conf=conf, verbose=False)[0]
    if not len(res.boxes): return []
    xywh = res.boxes.xywh.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy()
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(c))
            for b, c in zip(xywh, cls)]

def best_cluster(votes, lat, W):
    if not votes: return None
    arr = list(votes); used = [False]*len(arr); best = None
    for i in range(len(arr)):
        if used[i]: continue
        grp = [arr[i]]
        for j in range(i+1, len(arr)):
            if not used[j] and abs(arr[j][1]-arr[i][1])*lat <= CLUSTER_MM:
                grp.append(arr[j]); used[j] = True
        if len(grp) >= VOTE_K:
            med = np.median(np.array([g[1:5] for g in grp]), axis=0)
            centr = abs(med[0] - W/2)
            if best is None or len(grp) > best[0] or \
               (len(grp) == best[0] and centr < best[1]):
                best = (len(grp), centr, med)
    return None if best is None else best[2]

def try_lock(u8, box, deep, ax):
    cx, cy, bw, bh = box[:4]
    g = profile_h(u8, cy, cx, BUFFER*0.9*bh, BUFFER*0.35*bw, ax)
    if g and g["h_mm"] >= 0.8:
        return dict(cx=float(cx), cy=float(cy), hw=BUFFER*0.9*float(bh),
                    ww=BUFFER*0.35*float(bw), h0=g["h_mm"],
                    c0=g["contrast"], d0=deep)
    return None

def main():
    sec = Path("data/clarius_sessions") / sys.argv[1]
    raws = sorted(glob.glob(str(sec / "raw_*.json")))
    rep_p = sec / "compression_report.json"
    rep = json.loads(rep_p.read_text())["frames"] if rep_p.exists() else None
    model = YOLO(MODEL_PATH)
    lock, votes, dead, blind, crushed_seen, episode, log = \
        None, deque(), 0, 0, 0, 0, []
    for i, jp in enumerate(raws):
        meta_full = json.load(open(jp)); meta = meta_full["frame"]
        img = load_frame(Path(jp.replace(".json", ".bin")), meta_full)
        u8 = to_u8(img)
        ax = meta["axial_um_per_sample"]/1000.0
        lat = meta["lateral_um_per_line"]/1000.0
        deep = float(u8[int(0.6*u8.shape[0]):int(0.9*u8.shape[0]), :].mean())
        dets = detections(model, u8, DET_CONF)
        open_d  = [d for d in dets if d[4] == 0]
        crush_d = [d for d in dets if d[4] == 1]
        state, ratio = "ACQ", None
        if lock is None:
            for b in open_d: votes.append((i,)+b)   # lock on OPEN only
            while votes and votes[0][0] < i - VOTE_WIN: votes.popleft()
            box = best_cluster(votes, lat, u8.shape[1])
            if box is not None:
                lk = try_lock(u8, box, deep, ax)
                if lk:
                    lock, dead, blind, crushed_seen = lk, 0, 0, 0
                    episode += 1; votes.clear()
                    state, ratio = "GOOD", 1.0
        else:
            near_any = [d for d in dets
                        if abs(d[0]-lock["cx"])*lat <= GATE_MM
                        and abs(d[1]-lock["cy"])*ax <= GATE_MM]
            near_open  = [d for d in near_any if d[4] == 0]
            near_crush = [d for d in near_any if d[4] == 1]
            if near_any:
                d = min(near_any, key=lambda d: abs(d[0]-lock["cx"]))
                lock["cx"] += ANCHOR_EMA*(d[0]-lock["cx"])
                lock["cy"] += ANCHOR_EMA*(d[1]-lock["cy"])
            if deep < 0.55*lock["d0"]:
                state, ratio = "WASHED", None
                dead += 1
            else:
                best = None
                for dx in range(-SEARCH_PX, SEARCH_PX+1, 3):
                    g = profile_h(u8, lock["cy"], lock["cx"]+dx,
                                  lock["hw"], lock["ww"], ax)
                    if g and (best is None or g["lum"] < best["lum"]):
                        best = g
                alive = best is not None and \
                        best["contrast"] >= DEAD_FRAC*lock["c0"]
                if near_crush and not near_open:
                    crushed_seen += 1
                else:
                    crushed_seen = 0
                if crushed_seen >= CRUSH_CONFIRM:
                    # positive evidence: the vessel is HERE and it's crushed
                    state, ratio = "PRESSED", 0.0
                    dead += 1; blind = 0
                elif alive:
                    dead = blind = 0
                    ratio = best["h_mm"]/lock["h0"]
                    if   ratio > 0.8: state = "GOOD"
                    elif ratio > 0.5: state = "SQUEEZING"
                    else:             state = "PRESSED"
                elif near_open:
                    state, ratio = "SQUEEZING", None
                    dead = blind = 0
                else:
                    far_open = [d for d in open_d if d not in near_open]
                    lk = None
                    if far_open:
                        j = int(np.argmin([abs(d[0]-u8.shape[1]/2)
                                           for d in far_open]))
                        lk = try_lock(u8, far_open[j], deep, ax)
                    if lk:
                        lock, dead, blind, crushed_seen = lk, 0, 0, 0
                        episode += 1
                        state, ratio = "MIGRATE", 1.0
                    else:
                        blind += 1; dead += 1
                        if blind >= PINCH_CONFIRM:
                            state, ratio = "PRESSED", 0.0
                        else:
                            state, ratio = "SQUEEZING", None
            if dead > LOST_TIMEOUT:
                state, lock = "LOST", None
                votes.clear()
        truth = rep[i]["state"] if (rep and i < len(rep)) else None
        log.append(dict(frame=i, state=state, ep=episode,
                        ratio=None if ratio is None else round(float(ratio),2),
                        cx=None if lock is None else round(float(lock["cx"]),1),
                        cy=None if lock is None else round(float(lock["cy"]),1),
                        hw=None if lock is None else round(float(lock["hw"]),1),
                        ww=None if lock is None else round(float(lock["ww"]),1),
                        truth=truth))
    out = sec / "gauge_replay.jsonl"
    with open(out, "w") as f:
        for l in log: f.write(json.dumps(l)+"\n")
    print(f"{sec.name} [{Path(MODEL_PATH).stem}]: {len(log)} frames  "
          f"episodes={episode}  "
          f"states={dict(Counter(l['state'] for l in log))}")
    if rep:
        col = [l for l in log if l["truth"] == "COLLAPSED"]
        act = sum(1 for l in col if l["state"] in ("SQUEEZING", "PRESSED"))
        opn = [l for l in log if l["truth"] == "OPEN" and l["state"] != "ACQ"]
        ok = sum(1 for l in opn if l["state"] in ("GOOD","SQUEEZING","MIGRATE"))
        print(f"sensitivity: {act}/{len(col)} ({100*act/max(len(col),1):.0f}%)"
              f"   specificity: {ok}/{len(opn)} "
              f"({100*ok/max(len(opn),1):.0f}%)")
    m = {"GOOD":".", "SQUEEZING":"s", "PRESSED":"P", "WASHED":"W",
         "LOST":"L", "ACQ":"a", "MIGRATE":"M"}
    print("".join(m[l["state"]] for l in log[::10]))
    print("wrote", out)

if __name__ == "__main__":
    main()
