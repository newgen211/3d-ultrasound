#!/usr/bin/env python3
"""Box annotator v3 — auto-saves on navigate. TWO CLASSES:
  class 0 = OPEN lumen (clearly patent dark interior)
  class 1 = CRUSHED/crescent (vessel visibly compressed: sliver, crescent,
            kissing walls — the pressed morphologies)
All controls on screen. Resume-safe. lime = open, red = crushed.
"""
import json, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame, to_u8
from us3d.paths import SESSIONS, YOLO_HUMAN

PLAN = [("section_106", 12), ("section_109", 14), ("section_123", 12),
        ("section_123", 5)]   # extra 123 pass: squish-heavy = crushed supply
OUT = YOLO_HUMAN
(OUT/"images/train").mkdir(parents=True, exist_ok=True)
(OUT/"labels/train").mkdir(parents=True, exist_ok=True)

tasks, seen = [], set()
for sec, stride in PLAN:
    raws = sorted(glob.glob(str(SESSIONS / sec / "raw_*.json")))
    for i in range(0, len(raws), stride):
        if (sec, i) not in seen:
            seen.add((sec, i)); tasks.append((sec, i, raws[i]))

state = {"k": 0, "boxes": [], "u8": None, "stem": None,
         "cls": 0, "dirty": False}
COLOR = {0: "lime", 1: "red"}

def label_path(stem): return OUT/f"labels/train/{stem}.txt"

def load_existing(stem, W, H):
    p = label_path(stem)
    if not p.exists(): return []
    out = []
    for line in p.read_text().splitlines():
        c, xc, yc, w, h = line.split()
        c = int(c); xc, yc, w, h = map(float, (xc, yc, w, h))
        out.append([xc*W - w*W/2, yc*H - h*H/2, w*W, h*H, c])
    return out

def first_unlabeled():
    for idx, (sec, fi, _) in enumerate(tasks):
        if not label_path(f"{sec}__h{fi:05d}").exists(): return idx
    return len(tasks)-1

def save():
    import cv2
    u8 = state["u8"]; H, W = u8.shape
    cv2.imwrite(str(OUT/f"images/train/{state['stem']}.jpg"), u8)
    with open(label_path(state["stem"]), "w") as f:
        for (bx, by, bw, bh, c) in state["boxes"]:
            f.write(f"{c} {(bx+bw/2)/W:.6f} {(by+bh/2)/H:.6f} "
                    f"{bw/W:.6f} {bh/H:.6f}\n")
    state["dirty"] = False

def nav(k):
    if state["dirty"]: save()          # <-- auto-save on ANY navigation
    show(k)

def show(k):
    state["k"] = max(0, min(k, len(tasks)-1))
    sec, fi, jp = tasks[state["k"]]
    state["stem"] = f"{sec}__h{fi:05d}"
    mf = json.load(open(jp))
    state["u8"] = to_u8(load_frame(Path(jp.replace(".json", ".bin")), mf))
    H, W = state["u8"].shape
    state["boxes"] = load_existing(state["stem"], W, H)
    state["dirty"] = False
    draw()

def draw():
    ax.clear()
    u8 = state["u8"]
    ax.imshow(u8, cmap="gray", aspect="auto")
    for b in state["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3],
                               fill=False, color=COLOR[b[4]], lw=1.8))
    done = label_path(state["stem"]).exists()
    n0 = sum(1 for b in state["boxes"] if b[4] == 0)
    n1 = sum(1 for b in state["boxes"] if b[4] == 1)
    mode = "OPEN(lime)" if state["cls"] == 0 else "CRUSHED(red)"
    ax.set_title(
        f"[{state['k']+1}/{len(tasks)}] {state['stem']}"
        f"{' SAVED' if done else ''}{' *unsaved*' if state['dirty'] else ''}"
        f"   open={n0} crushed={n1}   DRAW MODE: {mode}\n"
        "drag=box | c=toggle open/crushed | d=next(saves) | a=prev(saves) | "
        "u=undo | 0=save EMPTY | g=first unlabeled | q=quit(saves)",
        fontsize=10)
    ax.axis("off")
    fig.canvas.draw_idle()

def on_select(e0, e1):
    x0, y0, x1, y1 = e0.xdata, e0.ydata, e1.xdata, e1.ydata
    if None in (x0, y0, x1, y1): return
    if abs(x1-x0) > 2 and abs(y1-y0) > 2:
        state["boxes"].append([min(x0,x1), min(y0,y1),
                               abs(x1-x0), abs(y1-y0), state["cls"]])
        state["dirty"] = True
        draw()

def on_key(ev):
    k = ev.key
    if k == "d": nav(state["k"]+1)
    elif k == "a": nav(state["k"]-1)
    elif k == "s": save(); show(state["k"]+1)
    elif k == "0": state["boxes"] = []; save(); show(state["k"]+1)
    elif k == "c": state["cls"] = 1-state["cls"]; draw()
    elif k == "u":
        if state["boxes"]:
            state["boxes"].pop(); state["dirty"] = True; draw()
    elif k == "g": nav(first_unlabeled())
    elif k == "q":
        if state["dirty"]: save()
        n = len(list((OUT/"labels/train").glob("*.txt")))
        print(f"labeled so far: {n}/{len(tasks)}")
        plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 11))
sel = RectangleSelector(ax, on_select, useblit=True, button=[1],
                        minspanx=3, minspany=3, interactive=False)
fig.canvas.mpl_connect("key_press_event", on_key)
show(first_unlabeled())
plt.show()
n = len(list((OUT/"labels/train").glob("*.txt")))
print(f"labeled: {n}/{len(tasks)}")
