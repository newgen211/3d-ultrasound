#!/usr/bin/env python3
"""
controller_preview.py — what the closed loop WILL do, computed from real data
before the loop exists.

Reads a section's reward_trace.csv (from demo_replay.py) and derives the
contact-control action the depth axis should have taken at every frame:

    LIFT  : vessels vanishing while the image stays well-coupled
            -> pressed too hard (the sec85 failure, measured)
    PRESS : image washing out (coupling low)
            -> contact too light / poor gel
    HOLD  : vessels visible and open enough

Pure signal -> action mapping with hysteresis; no learning, no hardware.
This is the hand-crafted baseline the RL policy has to beat — and the figure
that shows the reward signals already contain a usable control law.

    python3 controller_preview.py section_85
Writes <section>/controller_preview.png
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.sections import find_section

VIS_LOW, VIS_OK = 0.15, 0.35     # hysteresis on visibility
COUP_LOW = 0.85                  # below this = washed -> PRESS
SMOOTH = 9                       # frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    args = ap.parse_args()
    sec = find_section(args.section)
    path = sec / "reward_trace.csv"
    if not path.exists():
        sys.exit("no reward_trace.csv — run demo_replay.py --csv first")

    rows = list(csv.DictReader(open(path)))
    f = np.array([int(r["frame"]) for r in rows])
    def col(k):
        return np.array([float(r[k]) if r[k] not in ("", "None") else np.nan
                         for r in rows])
    vis, pat, coup = col("visible"), col("patency"), col("coupling")

    k = np.ones(SMOOTH) / SMOOTH
    vis_s = np.convolve(np.nan_to_num(vis), k, mode="same")
    coup_s = np.where(np.isnan(coup), 1.0, coup)
    coup_s = np.convolve(coup_s, k, mode="same")

    # action state machine with hysteresis
    action = np.zeros(len(f), int)          # 0 HOLD, +1 LIFT, -1 PRESS
    state = 0
    for i in range(len(f)):
        if coup_s[i] < COUP_LOW:
            state = -1                       # washed -> press
        elif vis_s[i] < VIS_LOW:
            state = +1                       # vessels gone, coupled -> lift
        elif vis_s[i] > VIS_OK and state == +1:
            state = 0                        # recovered -> hold
        elif state == -1 and coup_s[i] > COUP_LOW + 0.05:
            state = 0
        action[i] = state

    n_lift = int((action == 1).sum()); n_press = int((action == -1).sum())
    print(f"{sec.name}: LIFT {n_lift} frames ({100*n_lift/len(f):.0f}%), "
          f"PRESS {n_press}, HOLD {len(f)-n_lift-n_press}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 0.55]))
    ax[0].plot(f, vis, lw=0.8, color="tab:blue")
    ax[0].plot(f, vis_s, lw=1.4, color="navy")
    ax[0].axhline(VIS_LOW, color="crimson", ls="--", lw=0.7)
    ax[0].axhline(VIS_OK, color="gray", ls="--", lw=0.7)
    ax[0].set_ylabel("visible")
    ax[1].plot(f, pat, lw=0.8, color="tab:green")
    ax[1].plot(f, coup_s, lw=1.0, color="tab:purple", label="coupling (smoothed)")
    ax[1].legend(fontsize=8); ax[1].set_ylabel("patency / coupling")
    for a, b, lab, c in [(1, 1, "LIFT (pressed too hard)", "crimson"),
                         (-1, -1, "PRESS (washed)", "royalblue")]:
        m = action == a
        if m.any():
            ax[2].fill_between(f, 0, 1, where=m, color=c, alpha=0.6, label=lab)
    ax[2].fill_between(f, 0, 1, where=action == 0, color="mediumseagreen",
                       alpha=0.45, label="HOLD")
    ax[2].set_yticks([]); ax[2].legend(fontsize=8, ncol=3, loc="upper right")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("action")
    fig.suptitle(f"{sec.name} — contact-control preview: the action the depth "
                 f"axis SHOULD have taken (from image signals alone)")
    fig.tight_layout()
    fig.savefig(sec / "controller_preview.png", dpi=130)
    print(f"wrote {sec/'controller_preview.png'}")


if __name__ == "__main__":
    main()