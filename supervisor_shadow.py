#!/usr/bin/env python3
"""supervisor_shadow.py v2 — calibrate-then-track shadow supervisor.
CALIBRATE: cluster candidates over the first N decisions -> 2 vessel tracks
           with baseline position + area.
TRACK:     match candidates to tracks (gated, quality-picked); smoothed
           area-ratio per track; misses freeze the ratio, they don't vote.
JUDGE:     both tracks healthy -> GOOD | sustained sag/absence w/ deep ok
           -> PRESSED | deep crater -> WASHED. Commands NOTHING.
Usage: python3 supervisor_shadow.py section_N
"""
import glob, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src/segment")
from segment_tube import candidates, load_frame

GATE_MM      = 5.0      # match radius around a track
CAL_DECISIONS= 20       # ~10 s calibration
MIN_SIGHT    = 3        # sightings to accept a track at calibration end
AREA_EMA     = 0.3
SAG_RATIO    = 0.6      # smoothed area below this = squeezed
STARVE_S     = 12.0     # no sighting on a track this long = lost
WASHED_RATIO = 0.60
DECIDE_S     = 0.5

def cand_list(img, m):
    try:
        return candidates(img, m["axial_um_per_sample"]/1000.0,
                          m["lateral_um_per_line"]/1000.0)
    except Exception:
        return []

def area_of(c, m):
    (ew, eh) = c["ellipse"][1]
    w = ew * m["lateral_um_per_line"]/1000.0
    h = eh * m["axial_um_per_sample"]/1000.0
    return math.pi * (w/2) * (h/2)

class Track:
    def __init__(self, x, y, area):
        self.x, self.y = x, y
        self.base_area = area
        self.ratio = 1.0
        self.last_seen = time.time()
    def feed(self, c, m):
        self.x = 0.7*self.x + 0.3*c["cx_mm"]
        self.y = 0.7*self.y + 0.3*c["depth_mm"]
        self.ratio = (1-AREA_EMA)*self.ratio + AREA_EMA*(area_of(c, m)/self.base_area)
        self.last_seen = time.time()
    def starved(self): return time.time() - self.last_seen > STARVE_S

sec = Path("data/clarius_sessions")/sys.argv[1]
print(f"[shadow v2] {sec} — calibrating over first {CAL_DECISIONS} decisions")
seen, log = set(), open(sec/"supervisor_shadow.jsonl", "a")
cal_pts, tracks, deep_base = [], None, None
n_dec, last = 0, 0.0
try:
    while True:
        for jp in sorted(glob.glob(str(sec/"raw_*.json"))):
            if jp in seen: continue
            seen.add(jp)
            if time.time() - last < DECIDE_S: continue
            try:
                meta_full = json.loads(Path(jp).read_text())
                img = load_frame(Path(jp.replace(".json",".bin")), meta_full)
            except Exception:
                continue
            m = meta_full["frame"]
            deep = float(img[int(0.60*img.shape[0]):int(0.90*img.shape[0]), :].mean())
            cs = cand_list(img, m)
            last = time.time(); n_dec += 1
            # ---------- calibration ----------
            if tracks is None:
                cal_pts += [(c["cx_mm"], c["depth_mm"], area_of(c, m)) for c in cs]
                deep_base = deep if deep_base is None else 0.9*deep_base + 0.1*deep
                if n_dec >= CAL_DECISIONS:
                    used, cl = [False]*len(cal_pts), []
                    for i,(x,y,a) in enumerate(cal_pts):
                        if used[i]: continue
                        grp = [(x,y,a)]
                        for j in range(i+1, len(cal_pts)):
                            if not used[j] and abs(cal_pts[j][0]-x)<GATE_MM and abs(cal_pts[j][1]-y)<GATE_MM:
                                used[j] = True; grp.append(cal_pts[j])
                        cl.append(grp)
                    cl = sorted([g for g in cl if len(g) >= MIN_SIGHT], key=len, reverse=True)[:2]
                    if len(cl) < 2:
                        print(f"[cal] only {len(cl)} persistent vessels — extending calibration"); n_dec = CAL_DECISIONS//2
                        continue
                    tracks = [Track(np.median([p[0] for p in g]), np.median([p[1] for p in g]),
                                    np.median([p[2] for p in g])) for g in cl]
                    for t in tracks:
                        print(f"[cal] vessel @ ({t.x:+.1f}, {t.y:.1f}) mm, base area {t.base_area:.1f} mm2")
                print(f"CAL {n_dec}/{CAL_DECISIONS}  ncand={len(cs)}")
                continue
            # ---------- tracking ----------
            for t in tracks:
                near = [c for c in cs if abs(c["cx_mm"]-t.x)<GATE_MM and abs(c["depth_mm"]-t.y)<GATE_MM]
                if near: t.feed(max(near, key=lambda c: c.get("quality",0)), m)
            r = deep / deep_base if deep_base else 1.0
            # ---------- judgement ----------
            if r < WASHED_RATIO: state = "WASHED"
            elif all(t.starved() or t.ratio < SAG_RATIO for t in tracks): state = "PRESSED"
            elif any(t.starved() or t.ratio < SAG_RATIO for t in tracks): state = "SQUEEZING"
            else: state = "GOOD"
            rec = dict(t=time.time(), state=state, ratio=round(r,2), ncand=len(cs),
                       tracks=[dict(x=round(t.x,1), y=round(t.y,1),
                                    ar=round(t.ratio,2),
                                    starve=round(time.time()-t.last_seen,1)) for t in tracks])
            log.write(json.dumps(rec)+"\n"); log.flush()
            tr = "  ".join(f"({d['x']:+.0f},{d['y']:.0f}) ar={d['ar']:.2f} dry={d['starve']:.0f}s"
                           for d in rec["tracks"])
            print(f"{state:<10} deep={r:.2f}  {tr}")
        time.sleep(0.1)
except KeyboardInterrupt:
    print(f"\n[shadow v2] log -> {sec}/supervisor_shadow.jsonl")
