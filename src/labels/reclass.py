#!/usr/bin/env python3
"""Reclassify pass: click INSIDE a box to toggle open(lime) <-> crushed(red).
No drawing. Auto-saves on navigate. Only visits frames that have boxes.
Keys: d=next(saves) | a=prev(saves) | g=next frame with all-open boxes | q=quit(saves)
"""
import glob, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import YOLO_HUMAN
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

OUT = YOLO_HUMAN
stems = []
for lp in sorted(glob.glob(str(OUT/"labels/train/*.txt"))):
    if open(lp).read().strip():
        stems.append(Path(lp).stem)
print(f"{len(stems)} frames with boxes")

state = {"k": 0, "boxes": [], "img": None, "dirty": False}
COLOR = {0: "lime", 1: "red"}

def paths(stem):
    return OUT/f"labels/train/{stem}.txt", OUT/f"images/train/{stem}.jpg"

def load(stem):
    lp, ip = paths(stem)
    img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    boxes = []
    for line in open(lp).read().splitlines():
        if not line.strip(): continue
        c, xc, yc, w, h = line.split()
        c = int(c); xc, yc, w, h = map(float, (xc, yc, w, h))
        boxes.append([xc*W - w*W/2, yc*H - h*H/2, w*W, h*H, c])
    return img, boxes

def save():
    stem = stems[state["k"]]
    lp, _ = paths(stem)
    H, W = state["img"].shape
    with open(lp, "w") as f:
        for (bx, by, bw, bh, c) in state["boxes"]:
            f.write(f"{c} {(bx+bw/2)/W:.6f} {(by+bh/2)/H:.6f} "
                    f"{bw/W:.6f} {bh/H:.6f}\n")
    state["dirty"] = False

def show(k):
    if state["dirty"]: save()
    state["k"] = max(0, min(k, len(stems)-1))
    state["img"], state["boxes"] = load(stems[state["k"]])
    draw()

def draw():
    ax.clear()
    ax.imshow(state["img"], cmap="gray", aspect="auto")
    for b in state["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3],
                               fill=False, color=COLOR[b[4]], lw=2.0))
    n1 = sum(1 for b in state["boxes"] if b[4] == 1)
    ax.set_title(f"[{state['k']+1}/{len(stems)}] {stems[state['k']]}   "
                 f"crushed={n1}/{len(state['boxes'])}\n"
                 "click inside box = toggle | d=next | a=prev | "
                 "g=next all-open frame | q=quit",
                 fontsize=10)
    ax.axis("off")
    fig.canvas.draw_idle()

def on_click(ev):
    if ev.xdata is None or ev.ydata is None: return
    for b in reversed(state["boxes"]):          # topmost box wins
        if b[0] <= ev.xdata <= b[0]+b[2] and b[1] <= ev.ydata <= b[1]+b[3]:
            b[4] = 1 - b[4]
            state["dirty"] = True
            draw()
            return

def on_key(ev):
    if ev.key == "d": show(state["k"]+1)
    elif ev.key == "a": show(state["k"]-1)
    elif ev.key == "g":
        for j in range(state["k"]+1, len(stems)):
            _, bx = load(stems[j])
            if bx and all(b[4] == 0 for b in bx):
                show(j); return
        print("no later all-open frames")
    elif ev.key == "q":
        if state["dirty"]: save()
        tot = n1 = 0
        for s in stems:
            _, bx = load(s)
            tot += len(bx); n1 += sum(1 for b in bx if b[4] == 1)
        print(f"final: {n1} crushed / {tot} boxes")
        plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 11))
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
show(0)
plt.show()
