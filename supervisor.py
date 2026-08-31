#!/usr/bin/env python3
"""supervisor.py — closed-loop contact supervisor (replay-first build).

States from raw-numpy signals only:
  GOOD    vessel visible, deep coupling ok        -> hold
  PRESSED vessel lost, deep structure intact       -> lift (fixed step)
  WASHED  deep structure lost (coupling)           -> press (smaller step)
  AMBIG   vessel lost, deep ratio in gray zone     -> escalating micro-lift

Replay:  python3 supervisor.py --replay section_85
         (compares decisions against compression_report.json if present)
Live:    stub main_live() — bench day. --supervise on|off for E5 A/B.
"""
import argparse, glob, json, sys
import numpy as np
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src/segment")   # adjust if find said elsewhere
from segment_tube import candidates, load_frame

# ---- tunables (provisional; replay calibrates, E5 confirms) -----------------
DECIDE_EVERY_S = 0.5          # 2 Hz decision rate
LIFT_STEP      = 0.3          # mm, PRESSED response
PRESS_STEP     = 0.15         # mm, WASHED response (down = danger = smaller)
AMBIG_STEPS    = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5]   # escalating micro-lift
LIFT_CAP       =  3.0         # mm total upward authority
PRESS_CAP      = -1.5         # mm total downward authority
WASHED_RATIO   = 0.60         # deep ratio below -> coupling loss (documented)
PRESSED_RATIO  = 0.80         # deep ratio above (w/ vessel lost) -> collapse
GOOD_EXIT      = 3            # consecutive bad decisions before leaving GOOD
BAD_EXIT       = 2            # consecutive good before returning to GOOD

# ---- raw-numpy signals ------------------------------------------------------
def frame_signals(img):
    """img: (samples, lines) uint8. Returns deep-band mean + vessel visibility."""
    H, W = img.shape
    deep = img[int(0.60*H):int(0.90*H), :].mean()
    band = img[int(0.15*H):int(0.60*H), :]            # vessel depth band
    # classical rule, patchwise: vessel = dark AND smooth
    ph, pw = max(8, band.shape[0]//12), max(8, band.shape[1]//12)
    g = band[:(band.shape[0]//ph)*ph, :(band.shape[1]//pw)*pw]
    g = g.reshape(g.shape[0]//ph, ph, g.shape[1]//pw, pw)
    means, vars_ = g.mean(axis=(1,3)), g.var(axis=(1,3))
    dark = means < 0.75 * np.median(img)
    smooth = vars_ < np.percentile(vars_, 25)
    return deep, bool((dark & smooth).sum() >= 2)     # >=2 patches = a lumen

class Supervisor:
    def __init__(self):
        self.baseline = None       # EMA of deep mean over GOOD frames
        self.z = 0.0               # cumulative correction, mm (+ = lifted)
        self.state = "GOOD"
        self.ambig_i = 0
        self.bad_run = self.good_run = 0
        self.capped = False

    def classify(self, deep, visible):
        r = deep / self.baseline if self.baseline else 1.0
        if visible and r >= WASHED_RATIO: return "GOOD", r
        if r < WASHED_RATIO:              return "WASHED", r
        if r >= PRESSED_RATIO:            return "PRESSED", r
        return "AMBIG", r

    def decide(self, deep, visible):
        """Returns (state, dz, ratio). dz in mm, + = lift."""
        if self.baseline is None: self.baseline = deep
        state, r = self.classify(deep, visible)
        # hysteresis: don't leave GOOD on a single blink
        if self.state == "GOOD" and state != "GOOD":
            self.bad_run += 1
            if self.bad_run < GOOD_EXIT: return "GOOD", 0.0, r
        else:
            self.bad_run = 0
        if state == "GOOD":
            self.good_run += 1
            if self.state != "GOOD" and self.good_run < BAD_EXIT:
                return self.state, 0.0, r
            self.state, self.ambig_i, self.capped = "GOOD", 0, False
            self.baseline = 0.98*self.baseline + 0.02*deep     # slow EMA
            return "GOOD", 0.0, r
        self.good_run = 0
        self.state = state
        if self.capped: return state, 0.0, r
        if state == "PRESSED": dz = LIFT_STEP
        elif state == "WASHED": dz = -PRESS_STEP
        else:
            dz = AMBIG_STEPS[min(self.ambig_i, len(AMBIG_STEPS)-1)]
            self.ambig_i += 1
        nz = self.z + dz
        if nz > LIFT_CAP or nz < PRESS_CAP:
            self.capped = True
            return state + "_CAPPED", 0.0, r
        self.z = nz
        return state, dz, r

# ---- replay driver ----------------------------------------------------------
def replay(sec_name):
    sec = Path("data/clarius_sessions") / sec_name
    raws = sorted(glob.glob(str(sec / "raw_*.json")))
    if not raws: sys.exit(f"{sec_name}: no raw frames")
    rep_p = sec / "compression_report.json"
    report = json.loads(rep_p.read_text())["frames"] if rep_p.exists() else None
    sup, log = Supervisor(), []
    fps_guess, decide_every = 20.0, max(1, int(DECIDE_EVERY_S * 20))
    for i, jp in enumerate(raws):
        meta_full = json.load(open(jp))
        meta = meta_full["frame"]
        img = load_frame(Path(jp.replace(".json", ".bin")), meta_full)
        axial = meta["axial_um_per_sample"] / 1000.0
        lateral = meta["lateral_um_per_line"] / 1000.0
        deep = img[int(0.60*img.shape[0]):int(0.90*img.shape[0]), :].mean()
        try:
            vis = len(candidates(img, axial, lateral)) > 0
        except Exception:
            vis = False
        if i % decide_every: continue
        state, dz, r = sup.decide(deep, vis)
        truth = report[i]["state"] if (report and i < len(report)) else None
        log.append(dict(frame=i, state=state, dz=round(float(dz),2),
                        z=round(float(sup.z),2), ratio=round(float(r),2), truth=truth))
    out = sec / "supervisor_replay.jsonl"
    out.write_text("\n".join(json.dumps(l) for l in log) + "\n")
    # summary
    n = len(log)
    from collections import Counter
    print(f"{sec_name}: {n} decisions  states={dict(Counter(l['state'] for l in log))}")
    print(f"  final z={sup.z:+.2f} mm  capped={sup.capped}")
    if report:
        col = [l for l in log if l["truth"] == "COLLAPSED"]
        if col:
            lifted = sum(1 for l in col if l["state"].startswith(("PRESSED","AMBIG")))
            print(f"  truth-COLLAPSED decisions: {len(col)}, supervisor lifting on {lifted} "
                  f"({lifted/len(col):.0%})  <-- the replay score")
    print(f"  log -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="SECTION")
    ap.add_argument("--supervise", choices=["on","off"], default="on")
    a = ap.parse_args()
    if a.replay: replay(a.replay)
    else: sys.exit("live mode lands on bench day; use --replay SECTION")