#!/usr/bin/env python3
"""
vessel_centerline.py — build the vessel as a 3D tube from the tracker

Instead of hunting for the anechoic vessel in the dense intensity volume (where
gel and vessel are both dark, so MIP/MinIP can't isolate it), build the vessel
directly: the tracker already knows the vessel's centroid + radius in every
frame. Project those into 3D with the hand-eye transform + pose and you get the
vessel's centerline — a real tube.

Also the definitive geometry check:
  - points form a clean thin curve  -> poses are good, the volume tilt is harmless
  - points scatter into a cloud      -> the pose/rotation is wrong

    python3 vessel_centerline.py section_50
    python3 vessel_centerline.py section_50 --straighten

--straighten : spin the centerline's principal axis onto X so it reads horizontal
    instead of running diagonally through the camera frame. COSMETIC ONLY — the
    spread/length are measured before the rotation, so the numbers are identical.

Interaction: scroll wheel zooms whichever panel the cursor is over; drag rotates
the 3D panel.

Reuses the detector/tracker from segment_tube.py (keep both in the same folder).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from segment_tube import candidates, load_frame, find_section, track

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    straighten = "--straighten" in sys.argv
    section = find_section(pos_args[0] if pos_args else None)

    # hand-eye
    he = None
    for c in [section / "handeye.json", _REPO_ROOT / "handeye.json"]:
        if c.exists():
            he = json.loads(c.read_text()); break
    if he is None:
        sys.exit("no handeye.json (run calibrate_handeye.py / copy to project root)")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

    # per-frame candidates + pose
    jsons = sorted(Path(section).glob("raw_*.json"))
    frames_cands, poses = [], []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        pose = meta.get("cobot_pose")
        if not bp.exists() or not pose or len(pose.get("coords", [])) < 6:
            frames_cands.append([]); poses.append(None); continue
        f = meta["frame"]
        axial = f["axial_um_per_sample"] / 1000.0
        lateral = f["lateral_um_per_line"] / 1000.0
        try:
            frames_cands.append(candidates(load_frame(bp, meta), axial, lateral))
        except Exception:
            frames_cands.append([])
        poses.append(pose["coords"])

    picks = track(frames_cands)

    # project each vessel centroid into 3D world (same chain as reconstruct_handeye)
    pts, radii = [], []
    for pick, coords in zip(picks, poses):
        if pick is None or coords is None:
            continue
        p_img = np.array([pick["cx_mm"], 0.0, pick["depth_mm"]])   # image-plane point (mm)
        T = np.array(coords[:3], float)
        Rf = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
        pts.append(T + Rf @ (R_X @ p_img + t_X))
        radii.append(pick["r_mm"])
    pts = np.array(pts)
    if len(pts) < 5:
        sys.exit("not enough tracked points")

    # straightness / tube test via PCA: spread perpendicular to the main axis
    c = pts.mean(0)
    u, s_, vt = np.linalg.svd(pts - c)
    axis = vt[0]
    along = (pts - c) @ axis
    perp = (pts - c) - np.outer(along, axis)
    perp_rms = float(np.sqrt((perp ** 2).sum(1).mean()))
    length = float(along.max() - along.min())
    print(f"{Path(section).name}: {len(pts)} vessel points")
    print(f"  centerline length: {length:.1f} mm")
    print(f"  perpendicular spread (tube radius): {perp_rms:.2f} mm   "
          f"(vessel radius ~{np.median(radii):.2f} mm)")
    print("  -> clean tube if spread ~ vessel radius; cloud if much larger")

    # straighten for the figure (cosmetic): principal axis -> X
    if straighten:
        Rrot = vt.copy()
        if np.linalg.det(Rrot) < 0:          # keep it a proper rotation, no mirror
            Rrot[2] *= -1
        pts = (pts - c) @ Rrot.T
        print("  straightened: principal axis aligned to X (cosmetic, numbers unchanged)")

    # plot: 3D centerline + the 3 projections
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(141, projection="3d")
    col = np.arange(len(pts))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, cmap="viridis", s=8)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.6", lw=0.5)
    ax.set_title("3D centerline"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    try:
        ax.set_box_aspect(np.ptp(pts, 0))
    except Exception:
        pass
    for k, (i, j, name) in enumerate([(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]):
        a = fig.add_subplot(1, 4, k + 2)
        a.scatter(pts[:, i], pts[:, j], c=col, cmap="viridis", s=8)
        a.set_title(name); a.set_aspect("equal", "datalim")
        a.set_xlabel(name[0]); a.set_ylabel(name[2])
    fig.suptitle(f"{Path(section).name} — vessel centerline from tracker")
    out = Path(section) / "vessel_centerline.png"
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    # --- scroll wheel = zoom the panel under the cursor (drag rotates the 3D one) ---
    def on_scroll(event):
        a = event.inaxes
        if a is None:
            return
        f = 0.85 if event.button == "up" else 1.0 / 0.85
        if getattr(a, "name", "") == "3d":
            for get, set_ in ((a.get_xlim3d, a.set_xlim3d),
                              (a.get_ylim3d, a.set_ylim3d),
                              (a.get_zlim3d, a.set_zlim3d)):
                lo, hi = get(); mid = (lo + hi) / 2; half = (hi - lo) / 2 * f
                set_(mid - half, mid + half)
        else:
            for get, set_, cur in ((a.get_xlim, a.set_xlim, event.xdata),
                                   (a.get_ylim, a.set_ylim, event.ydata)):
                lo, hi = get()
                cur = cur if cur is not None else (lo + hi) / 2
                set_(cur - (cur - lo) * f, cur + (hi - cur) * f)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()


if __name__ == "__main__":
    main()