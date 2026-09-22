#!/usr/bin/env python3
"""
Audit tool: step through audit frames, correct vessel circles.
  left-click        add vessel (or select nearby existing)
  up/down           grow/shrink selected circle
  d                 delete selected
  n / p             next / previous frame (autosaves)
  w                 toggle frame "unusable" flag
  q                 save + quit
Truth = your circles. Teacher's shown dashed yellow for reference only.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import AUDIT

AUD = AUDIT
man = json.loads((AUD / "audit_manifest.json").read_text())
keys = sorted(man.keys())
outp = AUD / "audit_truth.json"
truth = json.loads(outp.read_text()) if outp.exists() else {}

state = {"i": 0, "sel": None}
fig, ax = plt.subplots(figsize=(6, 10))


def teacher_circles(k):
    out = []
    for d in man[k]["teacher"]:
        cx = d.get("cx_px", d.get("cx")); cy = d.get("cy_px", d.get("cy"))
        r = d.get("r_px", d.get("r"))
        if r is None and d.get("r_mm") is not None:
            r = d["r_mm"] / 0.051333          # axial um/sample -> px
        if r is None: r = 8
        if cx is not None and cy is not None:
            out.append((float(cx), float(cy), float(r)))
    return out


def draw():
    ax.clear()
    k = keys[state["i"]]
    img = plt.imread(AUD / "frames" / k)
    ax.imshow(img, cmap="gray")
    ent = truth.setdefault(k, {"vessels": [], "unusable": False})
    for cx, cy, r in teacher_circles(k):
        ax.add_patch(Circle((cx, cy), r, fill=False, ls="--", color="yellow", lw=1))
    for j, (cx, cy, r) in enumerate(ent["vessels"]):
        col = "lime" if j != state["sel"] else "red"
        ax.add_patch(Circle((cx, cy), r, fill=False, color=col, lw=2))
    flag = "  [UNUSABLE]" if ent["unusable"] else ""
    done = sum(1 for kk in keys if kk in truth and (truth[kk]["vessels"] or truth[kk]["unusable"]))
    ax.set_title(f"{state['i']+1}/{len(keys)}  {k}{flag}   done:{done}\n"
                 "click add/select | ↑↓ size | d del | n/p nav | w flag | q quit")
    fig.canvas.draw_idle()


def save():
    outp.write_text(json.dumps(truth, indent=1))


def on_click(ev):
    if ev.inaxes != ax or ev.xdata is None:
        return
    k = keys[state["i"]]
    vs = truth[k]["vessels"]
    for j, (cx, cy, r) in enumerate(vs):
        if (ev.xdata - cx) ** 2 + (ev.ydata - cy) ** 2 < max(r, 6) ** 2:
            state["sel"] = j; draw(); return
    vs.append([float(ev.xdata), float(ev.ydata), 8.0])
    state["sel"] = len(vs) - 1
    draw()


def on_key(ev):
    k = keys[state["i"]]
    vs = truth[k]["vessels"]
    if ev.key == "n":
        save(); state["i"] = min(state["i"] + 1, len(keys) - 1); state["sel"] = None
    elif ev.key == "p":
        save(); state["i"] = max(state["i"] - 1, 0); state["sel"] = None
    elif ev.key == "d" and state["sel"] is not None and vs:
        vs.pop(state["sel"]); state["sel"] = None
    elif ev.key == "up" and state["sel"] is not None:
        vs[state["sel"]][2] += 1
    elif ev.key == "down" and state["sel"] is not None:
        vs[state["sel"]][2] = max(2, vs[state["sel"]][2] - 1)
    elif ev.key == "w":
        truth[k]["unusable"] = not truth[k]["unusable"]
    elif ev.key == "q":
        save(); plt.close(fig); return
    draw()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
draw()
plt.show()
save()
print(f"saved {outp}")
