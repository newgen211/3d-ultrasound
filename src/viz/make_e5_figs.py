import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Fig 1: matched-region pressed% bar chart ----
def rows_for(s):
    sec = Path(f"data/clarius_sessions/section_{s}")
    lp = sec / "supervisor_v3_live.jsonl"
    if lp.exists():
        r = [json.loads(l) for l in open(lp)]
        if any(x["state"] not in ("ACQ",) for x in r):
            return r
    return [json.loads(l) for l in open(sec / "gauge_replay.jsonl")]

CUT = 0.40
flights = [(136,"OFF"),(137,"OFF"),(133,"OFF"),(138,"OFF"),(139,"OFF"),
           (134,"ON"),(143,"ON"),(141,"ON")]
labels, vals, cols = [], [], []
for s, arm in flights:
    rows = rows_for(s)
    N = max(r["frame"] for r in rows) + 1
    win = [r for r in rows if r["frame"] >= CUT*N and r["state"] != "ACQ"]
    pr = 100*sum(1 for r in win if r["state"]=="PRESSED")/len(win)
    labels.append(f"{s}\n{arm}")
    vals.append(pr)
    cols.append("#b0b0b0" if arm=="OFF" else "#2e7d32")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(labels, vals, color=cols)
off = [v for v,(_,a) in zip(vals,flights) if a=="OFF"]
on  = [v for v,(_,a) in zip(vals,flights) if a=="ON"]
ax.axhline(np.mean(off), color="#707070", ls="--", lw=1,
           label=f"OFF mean {np.mean(off):.0f}%")
ax.axhline(np.mean(on), color="#2e7d32", ls="--", lw=1,
           label=f"ON mean {np.mean(on):.0f}%")
ax.set_ylabel("PRESSED fraction, matched region (%)")
ax.legend(fontsize=8); ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig("paper/figs/e5_matched_region.png", dpi=200)
print("wrote paper/figs/e5_matched_region.png")

# ---- Fig 2: flight-143 timeline (ratio trace colored by state + dz) ----
rows = [json.loads(l) for l in
        open("data/clarius_sessions/section_143/supervisor_v3_live.jsonl")]
COL = {"GOOD":"#2e7d32","SQUEEZING":"#f9a825","PRESSED":"#c62828",
       "WASHED":"#0277bd","MIGRATE":"#8e24aa","ACQ":"#9e9e9e","LOST":"#000"}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                             gridspec_kw={"height_ratios":[2,1]})
f = [r["frame"] for r in rows]
for r in rows:
    if r["ratio"] is not None:
        a1.plot(r["frame"], r["ratio"], ".", ms=2, color=COL[r["state"]])
a1.set_ylabel("lumen height ratio h/h0")
a1.set_ylim(-0.05, 1.4)
for st, c in COL.items():
    a1.plot([], [], ".", color=c, label=st, ms=6)
a1.legend(fontsize=6, ncol=4, loc="upper right")
a2.plot(f, [r["dz"] for r in rows], color="#37474f", lw=1)
a2.set_ylabel("commanded dz (mm)"); a2.set_xlabel("frame")
a2.set_ylim(-0.2, 3.4)
fig.tight_layout(); fig.savefig("paper/figs/e5_flight143_timeline.png", dpi=200)
print("wrote paper/figs/e5_flight143_timeline.png")
