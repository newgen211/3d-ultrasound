#!/usr/bin/env python3
"""supervisor_v3.py — live gauge-v5 supervisor. Tails the newest (or named)
section; per fresh frame runs the detector-anchored gauge; at 2 Hz maps
state -> cumulative dz and sends {"dz": x} to the Pi (absolute, idempotent).
--shadow: full pipeline, log only, nothing sent.
Frame safety: only consumes raw N once raw N+1's json exists (mid-write guard).
Usage: python3 supervisor_v3.py --section section_124 --model models/best_regated.pt \
         --pi 192.168.196.134 [--shadow]
"""
import argparse, json, glob, socket, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--section", required=True)
ap.add_argument("--model", default="models/best_regated.pt")
ap.add_argument("--pi", default="192.168.196.134")
ap.add_argument("--port", type=int, default=5006)
ap.add_argument("--shadow", action="store_true")
args = ap.parse_args()

DET_CONF, GATE_MM, ANCHOR_EMA = 0.25, 5.0, 0.4
VOTE_WIN, VOTE_K, CLUSTER_MM = 40, 8, 4.0
BUFFER, SEARCH_PX, DEAD_FRAC = 1.10, 12, 0.4
PINCH_CONFIRM, LOST_TIMEOUT = 5, 400
DECIDE_S = 0.5
LIFT_STEP, PRESS_STEP = 0.3, 0.15
CAP_UP, CAP_DN = 3.0, -1.5

def profile_h(u8, cy, cx, hh, hw, ax):
    r0, r1 = max(int(cy-hh),0), min(int(cy+hh), u8.shape[0])
    c0, c1 = max(int(cx-hw),0), min(int(cx+hw), u8.shape[1])
    if r1-r0 < 6 or c1-c0 < 3: return None
    prof = u8[r0:r1, c0:c1].mean(axis=1)
    n = len(prof); third = max(n//3, 1)
    wall = min(prof[:third].max(), prof[-third:].max())
    lum = prof[third:-third].min() if n > 2*third else prof.min()
    if wall <= lum: return None
    th = lum + 0.35*(wall-lum); below = prof < th
    best = run = s = 0; bs = be = 0
    for i, b in enumerate(below):
        run = run+1 if b else 0
        if b and run == 1: s = i
        if run > best: best, bs, be = run, s, i
    return dict(h_mm=best*ax, contrast=float(wall-lum), row=r0+(bs+be)/2.0,
                lum=float(lum))

model = YOLO(args.model)
def detect(u8):
    ok, buf = cv2.imencode(".jpg", u8)
    res = model.predict(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                        conf=DET_CONF, verbose=False)[0]
    if not len(res.boxes): return []
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b in res.boxes.xywh.cpu().numpy()]

sec = Path("data/clarius_sessions") / args.section
print(f"supervising {sec}  model={args.model}  shadow={args.shadow}")
sock = None
def send_dz(dz):
    global sock
    if args.shadow: return
    try:
        if sock is None:
            sock = socket.create_connection((args.pi, args.port), timeout=2)
        sock.sendall((json.dumps({"dz": round(dz, 2)}) + "\n").encode())
    except Exception as e:
        sock = None
        print(f"  [send failed: {e}]", flush=True)

from collections import deque
lock_st, votes = None, deque()
dead = blind = episode = 0
proc_n = 0
cum_dz, last_decide = 0.0, 0.0
acq_since, ACQ_GRACE, ACQ_CAP = None, 6.0, 2.0
seen = 0
import time as _t
while not sec.exists():
    print(f"waiting for {sec} to appear...", flush=True)
    _t.sleep(1.0)
logf = open(sec / "supervisor_v3_live.jsonl", "a")
try:
    while True:
        raws = sorted(glob.glob(str(sec / "raw_*.json")))
        # mid-write guard: process only frames with a successor
        avail = len(raws) - 1
        if avail <= seen:
            time.sleep(0.1); continue
        i = avail - 1                      # newest safe frame; skip backlog
        seen = avail
        jp = raws[i]
        try:
            meta_full = json.load(open(jp)); meta = meta_full["frame"]
            img = load_frame(Path(jp.replace(".json", ".bin")), meta_full)
        except Exception:
            continue
        u8 = to_u8(img)
        proc_n += 1
        ax = meta["axial_um_per_sample"]/1000.0
        lat = meta["lateral_um_per_line"]/1000.0
        deep = float(u8[int(0.6*u8.shape[0]):int(0.9*u8.shape[0]), :].mean())
        dets = detect(u8)
        state, ratio = "ACQ", None
        if lock_st is None:
            for b in dets: votes.append((proc_n,)+b)
            while votes and votes[0][0] < proc_n - VOTE_WIN: votes.popleft()
            arr = list(votes); best = None
            used = [False]*len(arr)
            for a_ in range(len(arr)):
                if used[a_]: continue
                grp = [arr[a_]]
                for b_ in range(a_+1, len(arr)):
                    if not used[b_] and \
                       abs(arr[b_][1]-arr[a_][1])*lat <= CLUSTER_MM:
                        grp.append(arr[b_]); used[b_] = True
                if len(grp) >= VOTE_K:
                    med = np.median(np.array([g[1:5] for g in grp]), axis=0)
                    c_ = abs(med[0]-u8.shape[1]/2)
                    if best is None or len(grp) > best[0] or \
                       (len(grp) == best[0] and c_ < best[1]):
                        best = (len(grp), c_, med)
            if best is not None:
                cx, cy, bw, bh = best[2]
                g = profile_h(u8, cy, cx, BUFFER*0.9*bh, BUFFER*0.35*bw, ax)
                if g and g["h_mm"] >= 0.8:
                    lock_st = dict(cx=float(cx), cy=float(cy),
                                   hw=BUFFER*0.9*float(bh),
                                   ww=BUFFER*0.35*float(bw),
                                   h0=g["h_mm"], c0=g["contrast"], d0=deep)
                    dead = blind = 0; episode += 1; votes.clear()
                    state, ratio = "GOOD", 1.0
        else:
            near = [d for d in dets
                    if abs(d[0]-lock_st["cx"])*lat <= GATE_MM
                    and abs(d[1]-lock_st["cy"])*ax <= GATE_MM]
            if near:
                d = min(near, key=lambda d: abs(d[0]-lock_st["cx"]))
                lock_st["cx"] += ANCHOR_EMA*(d[0]-lock_st["cx"])
                lock_st["cy"] += ANCHOR_EMA*(d[1]-lock_st["cy"])
            if deep < 0.55*lock_st["d0"]:
                state, ratio = "WASHED", None; dead += 1
            else:
                best = None
                for dx in range(-SEARCH_PX, SEARCH_PX+1, 3):
                    g = profile_h(u8, lock_st["cy"], lock_st["cx"]+dx,
                                  lock_st["hw"], lock_st["ww"], ax)
                    if g and (best is None or g["lum"] < best["lum"]):
                        best = g
                alive = best is not None and \
                        best["contrast"] >= DEAD_FRAC*lock_st["c0"]
                if alive:
                    dead = blind = 0
                    ratio = best["h_mm"]/lock_st["h0"]
                    state = ("GOOD" if ratio > 0.8 else
                             "SQUEEZING" if ratio > 0.5 else "PRESSED")
                elif near:
                    state, ratio = "SQUEEZING", None; dead = blind = 0
                else:
                    far = [d for d in dets if d not in near]
                    relocked = False
                    if far:
                        j = int(np.argmin([abs(d[0]-u8.shape[1]/2)
                                           for d in far]))
                        cx, cy, bw, bh = far[j]
                        g = profile_h(u8, cy, cx, BUFFER*0.9*bh,
                                      BUFFER*0.35*bw, ax)
                        if g and g["h_mm"] >= 0.8:
                            lock_st = dict(cx=float(cx), cy=float(cy),
                                           hw=BUFFER*0.9*float(bh),
                                           ww=BUFFER*0.35*float(bw),
                                           h0=g["h_mm"], c0=g["contrast"],
                                           d0=deep)
                            dead = blind = 0; episode += 1
                            state, ratio = "MIGRATE", 1.0
                            relocked = True
                    if not relocked:
                        blind += 1; dead += 1
                        state, ratio = (("PRESSED", 0.0)
                                        if blind >= PINCH_CONFIRM
                                        else ("SQUEEZING", None))
            if dead > LOST_TIMEOUT:
                state, lock_st = "LOST", None; votes.clear()
        now = time.time()
        if now - last_decide >= DECIDE_S:
            last_decide = now
            if state == "PRESSED":
                cum_dz = min(cum_dz + LIFT_STEP, CAP_UP)
                acq_since = None
            elif state == "WASHED":
                cum_dz = max(cum_dz - PRESS_STEP, CAP_DN)
                acq_since = None
            elif state == "ACQ":
                if acq_since is None: acq_since = now
                if now - acq_since > ACQ_GRACE:
                    cum_dz = min(cum_dz + PRESS_STEP, ACQ_CAP)
            else:
                acq_since = None
            send_dz(cum_dz)
        logf.write(json.dumps(dict(frame=i, state=state,
                   ratio=None if ratio is None else round(float(ratio), 2),
                   dz=round(cum_dz, 2), t=round(now, 2))) + "\n")
        logf.flush()
        print(f"f{i:5d} {state:9s} "
              f"r={'' if ratio is None else round(ratio,2)!s:5s} "
              f"dz={cum_dz:+.2f}", flush=True)
except KeyboardInterrupt:
    print("\nstopped.")
