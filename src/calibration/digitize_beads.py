#!/usr/bin/env python3
"""
digitize_beads.py — mark the calibration point in each captured frame

Step 3 of hand-eye calibration. Shows every raw frame in a capture section and
lets you click the bead / wire-cross. Saves the pixel locations for
calibrate_handeye.py. Runs on the Mac (needs a display).

    Left-click  = mark the point (advances to the next frame)
    Right-click = skip this frame (point not visible, two dots, bad image)

Resumable: re-running picks up the existing handeye_clicks.json and only asks
about frames you haven't marked yet (delete the file to start over).

Usage:
    python3 digitize_beads.py                 # latest section
    python3 digitize_beads.py section_25
    python3 digitize_beads.py section_25 --redo   # re-click everything

Output:
    <section>/handeye_clicks.json     { "raw_<ts>": [u, v], ... }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Anchor to the repo's data/ folder, regardless of where this is launched.
DATA = Path(__file__).resolve().parents[2] / "data"


def find_section(arg):
    root = DATA / "clarius_sessions"
    if not root.exists():
        sys.exit(f"❌ No clarius_sessions/ folder at {root}")
    if arg is None:
        sections = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("section_")],
            key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
        )
        if not sections:
            sys.exit(f"❌ No section_N folders in {root}")
        return sections[-1]
    for cand in (Path(arg), root / arg):
        if cand.exists():
            return cand
    sys.exit(f"❌ Section folder not found: {arg}")


def load_frame(bin_path, meta):
    f = meta["frame"]
    lines, samples, bps = f["lines"], f["samples"], f["bps"]
    jpg = f.get("jpg_size", 0)
    raw = bin_path.read_bytes()
    if jpg > 0:
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(raw)).convert("L")).astype(np.float32)
    dtype = np.uint8 if bps == 8 else np.uint16
    arr = np.frombuffer(raw, dtype=dtype)
    expected = lines * samples
    if arr.size != expected:
        usable = (arr.size // lines) * lines
        arr = arr[:usable]
        samples = usable // lines
    return arr.reshape(lines, samples).T.astype(np.float32)  # (depth rows, width cols)


def snap_to_blob(img, u, v, win=20):
    """Refine a click to the intensity-weighted centroid of the bright blob
    around it, so the mark lands on the same spot every frame regardless of how
    precise the click was. Returns (u, v) unchanged if no clear blob is found."""
    H, W = img.shape
    u0, v0 = int(round(u)), int(round(v))
    x0, x1 = max(0, u0 - win), min(W, u0 + win + 1)
    y0, y1 = max(0, v0 - win), min(H, v0 + win + 1)
    patch = img[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0 or patch.max() <= patch.min():
        return u, v
    thr = patch.min() + 0.6 * (patch.max() - patch.min())   # keep the brightest part
    ys, xs = np.nonzero(patch >= thr)
    if len(xs) == 0:
        return u, v
    w = patch[ys, xs] - thr                                 # weight by brightness
    if w.sum() <= 0:
        return u, v
    cu = x0 + float((xs * w).sum() / w.sum())
    cv = y0 + float((ys * w).sum() / w.sum())
    return cu, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--redo", action="store_true", help="ignore existing clicks and start over")
    ap.add_argument("--no-snap", action="store_true",
                    help="don't snap to the bright centroid; record the raw click")
    ap.add_argument("--win", type=int, default=20,
                    help="snap search half-window in pixels (default 20)")
    args = ap.parse_args()

    section = find_section(args.section)
    out = section / "handeye_clicks.json"
    clicks = {}
    if out.exists() and not args.redo:
        clicks = json.loads(out.read_text())
        print(f"↻ resuming — {len(clicks)} already marked")

    jsons = sorted(section.glob("raw_*.json"))
    if not jsons:
        sys.exit("❌ No raw_*.json frames in this section")

    n_pose = sum("cobot_pose" in json.loads(jp.read_text()) for jp in jsons)
    print(f"📂 {section}: {len(jsons)} frames, {n_pose} with a cobot_pose")
    if n_pose == 0:
        print("⚠️  No frames have a cobot_pose yet — run merge_poses.py first, "
              "or the calibration won't be able to use them.")

    todo = [jp for jp in jsons if jp.stem not in clicks]
    print(f"   {len(todo)} to mark. Left-click the point, right-click to skip.\n")

    for i, jp in enumerate(todo):
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            continue
        img = load_frame(bp, meta)
        has_pose = "cobot_pose" in meta

        fig, ax = plt.subplots(figsize=(7, 9))
        vmin, vmax = np.percentile(img, [1, 99.5])          # contrast stretch
        ax.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=max(vmax, vmin + 1))
        snap_note = "raw click" if args.no_snap else "snaps to bright center"
        ax.set_title(f"{jp.stem}   ({i+1}/{len(todo)})   "
                     f"pose: {'yes' if has_pose else 'MISSING'}\n"
                     f"left-click = mark ({snap_note})   •   right-click = skip",
                     fontsize=10)
        pts = plt.ginput(1, timeout=0,
                         mouse_add=MouseButton.LEFT, mouse_stop=MouseButton.RIGHT)

        if pts:
            u, v = pts[0]
            if not args.no_snap:
                u, v = snap_to_blob(img, u, v, args.win)
            ax.plot(u, v, "+", color="lime", markersize=16, markeredgewidth=1.5)
            fig.canvas.draw()
            plt.pause(0.4)                                  # show where it landed
            clicks[jp.stem] = [round(float(u), 2), round(float(v), 2)]
            print(f"   ✓ {jp.stem}: ({u:.1f}, {v:.1f})")
        else:
            print(f"   – {jp.stem}: skipped")
        plt.close(fig)
        out.write_text(json.dumps(clicks, indent=2))  # save after each one

    print(f"\n💾 {out}  ({len(clicks)} points marked)")
    if len(clicks) < 8:
        print("⚠️  Fewer than 8 points — calibration will be shaky. Capture more poses.")


if __name__ == "__main__":
    main()