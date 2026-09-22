#!/usr/bin/env python3
"""
demo_replay.py — stream a stored section through RewardSignals, frame by
frame, exactly as a live controller would see it. Lets the RL side inspect
the reward traces with zero hardware.

    python3 demo_replay.py section_85
    python3 demo_replay.py section_81 --model models/best_regated.pt --csv

Writes <section>/reward_trace.png (+ optional reward_trace.csv).
Sanity expectations: section_85 (autonomous, over-pressed) should show
patency dropping and going None through the collapse zones; section_81
(freehand, gentle) should sit high and mostly defined.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.sections import find_section
from reward_signals import RewardSignals, combine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--model", default="best_regated.pt",
                    help="checkpoint: a path, or a name under models/")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    sec = find_section(args.section)
    jpgs = sorted((sec / "frames_jpg").glob("*.jpg"))
    if not jpgs:
        sys.exit("no frames_jpg/ — decode the section first")
    meta = json.loads(next(sec.glob("raw_*.json")).read_text())["frame"]
    axial = meta["axial_um_per_sample"] / 1000.0
    lateral = meta["lateral_um_per_line"] / 1000.0

    rs = RewardSignals(model=args.model, conf=args.conf)
    rows = []
    for p in jpgs:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        s = rs.update(gray, axial, lateral)
        s["frame"] = int(p.stem)
        s["reward"] = combine(s)
        rows.append(s)

    f = np.array([r["frame"] for r in rows])
    def col(k):
        return np.array([np.nan if r[k] is None else r[k] for r in rows], float)

    fig, ax = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    for a, (k, c) in zip(ax, [("visible", "tab:blue"), ("centered", "tab:orange"),
                              ("patency", "tab:green"), ("coupling", "tab:purple"),
                              ("reward", "k")]):
        v = col(k)
        a.plot(f, v, lw=1.0, color=c)
        a.set_ylabel(k); a.set_ylim(-0.05, 1.05)
        miss = np.isnan(v)
        if miss.any():
            a.plot(f[miss], np.full(miss.sum(), -0.02), "|", color="crimson",
                   ms=6, label="None (unknown)")
            a.legend(fontsize=7, loc="lower right")
    ax[-1].set_xlabel("frame")
    fig.suptitle(f"{sec.name} — reward signals (causal, live-equivalent)")
    fig.tight_layout()
    fig.savefig(sec / "reward_trace.png", dpi=130)

    d = {k: col(k) for k in ("visible", "centered", "patency", "coupling", "reward")}
    print(f"{len(rows)} frames")
    for k, v in d.items():
        ok = ~np.isnan(v)
        print(f"  {k:9s} defined {ok.sum():4d}/{len(v)}  "
              f"mean {np.nanmean(v):.2f}  p10 {np.nanpercentile(v,10):.2f}  "
              f"p90 {np.nanpercentile(v,90):.2f}" if ok.any() else
              f"  {k:9s} never defined")
    print(f"wrote {sec/'reward_trace.png'}")

    if args.csv:
        import csv
        with open(sec / "reward_trace.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "visible", "centered",
                                               "patency", "coupling",
                                               "n_vessels", "tracked", "reward"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {sec/'reward_trace.csv'}")


if __name__ == "__main__":
    main()