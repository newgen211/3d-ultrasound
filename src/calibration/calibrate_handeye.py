#!/usr/bin/env python3
"""
calibrate_handeye.py — solve the flange -> image-plane transform (hand-eye)

Step 4 of hand-eye calibration. Uses a single fixed point (bead / wire-cross)
imaged from many arm poses: for the correct transform X, the clicked point must
back-project to the SAME location in the robot base frame from every pose. We
solve for the X that minimises how much those back-projected points scatter.

Model (X = image-frame -> flange-frame, the thing we want):
    p_base = flange_pos + R_flange @ (R_X @ p_image + t_X)
where p_image = [(u - W/2)*lateral_mm, 0, v*axial_mm] comes from the click.

The cobot reports orientation as Euler angles with an unknown convention, so
rather than guess, we try every common convention and keep whichever makes the
points cluster tightest. That winner is almost certainly the real convention.

Inputs come from the section's merged sidecars (cobot_pose) + handeye_clicks.json.

Usage:
    python3 calibrate_handeye.py                # latest section
    python3 calibrate_handeye.py section_25
    python3 calibrate_handeye.py section_25 --restarts 20

Output:
    <section>/handeye.json   { convention, R_flange_to_image, t_flange_to_image,
                               rms_mm, n_points }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# extrinsic (lowercase) and intrinsic (uppercase) Tait-Bryan orders
CONVENTIONS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
               "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

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


def gather(section, drop=()):
    """Return arrays: p_image (N,3 mm), flange_pos (N,3 mm), euler (N,3 deg)."""
    clicks_path = section / "handeye_clicks.json"
    if not clicks_path.exists():
        sys.exit(f"❌ No handeye_clicks.json in {section} — run digitize_beads.py first")
    clicks = json.loads(clicks_path.read_text())

    p_img, pos, eul, used = [], [], [], []
    for stem, (u, v) in clicks.items():
        if stem in drop:
            continue
        meta_path = section / f"{stem}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        pose = meta.get("cobot_pose")
        if not pose or "coords" not in pose or len(pose["coords"]) < 6:
            continue
        f = meta["frame"]
        lateral_mm = f["lateral_um_per_line"] / 1000.0
        axial_mm = f["axial_um_per_sample"] / 1000.0
        W = f["lines"]
        p_img.append([(u - W / 2.0) * lateral_mm, 0.0, v * axial_mm])
        pos.append(pose["coords"][:3])
        eul.append(pose["coords"][3:6])
        used.append(stem)

    if len(used) < 6:
        sys.exit(f"❌ Only {len(used)} usable points (clicked AND merged). "
                 f"Need ≥6 (≥10 recommended).")
    return (np.array(p_img, float), np.array(pos, float),
            np.array(eul, float), used)


def solve_for_convention(R_flange, pos, p_img, restarts, rng):
    """Solve X for one Euler convention. Returns (rotvec, t, rms_mm)."""
    def residual(params):
        R_X = Rotation.from_rotvec(params[:3]).as_matrix()
        t_X = params[3:6]
        # p_base_i = pos_i + R_flange_i @ (R_X @ p_img_i + t_X)
        flange_pts = p_img @ R_X.T + t_X                       # (N,3)
        base_pts = pos + np.einsum("nij,nj->ni", R_flange, flange_pts)
        return (base_pts - base_pts.mean(0)).ravel()

    best = None
    for k in range(restarts):
        if k == 0:
            x0 = np.array([0, 0, 0, 0, 0, 50.0])   # probe roughly below the flange
        else:
            x0 = np.concatenate([rng.uniform(-np.pi, np.pi, 3),
                                 rng.uniform(-120, 120, 3)])
        try:
            sol = least_squares(residual, x0, method="lm", max_nfev=4000)
        except Exception:
            continue
        r = sol.fun.reshape(-1, 3)
        rms = float(np.sqrt((r ** 2).sum(axis=1).mean()))      # RMS dist to centroid
        if best is None or rms < best[2]:
            best = (sol.x[:3].copy(), sol.x[3:6].copy(), rms)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--restarts", type=int, default=15,
                    help="random restarts per convention (default 15)")
    ap.add_argument("--drop", default="",
                    help="comma-separated frame stems to exclude, e.g. raw_123,raw_456")
    args = ap.parse_args()

    drop = {s.strip() for s in args.drop.split(",") if s.strip()}
    section = find_section(args.section)
    p_img, pos, eul, used = gather(section, drop)
    if drop:
        print(f"   (dropped {len(drop)} frame(s) by request)")
    print(f"📂 {section}: {len(used)} usable points")

    lo, hi = eul.min(0), eul.max(0)
    print(f"   euler spread (deg): rx [{lo[0]:.0f},{hi[0]:.0f}]  "
          f"ry [{lo[1]:.0f},{hi[1]:.0f}]  rz [{lo[2]:.0f},{hi[2]:.0f}]")
    n_gimbal = int(np.sum(np.abs(np.abs(eul[:, 1]) - 90) < 15))
    if n_gimbal:
        print(f"   ⚠️  {n_gimbal}/{len(used)} poses have ry within 15° of ±90° — "
              f"near gimbal lock, where the cobot's orientation reads get noisy")

    rng = np.random.default_rng(0)

    results = []
    for conv in CONVENTIONS:
        R_flange = Rotation.from_euler(conv, eul, degrees=True).as_matrix()  # (N,3,3)
        best = solve_for_convention(R_flange, pos, p_img, args.restarts, rng)
        if best:
            results.append((conv, *best))

    results.sort(key=lambda r: r[3])
    print("\n   convention   RMS scatter (mm)")
    print("   ----------   ----------------")
    for conv, _, _, rms in results[:6]:
        print(f"   {conv:<10}   {rms:6.2f}")

    conv, rotvec, t_X, rms = results[0]
    runner_up = results[1][3] if len(results) > 1 else float("inf")
    R_X = Rotation.from_rotvec(rotvec).as_matrix()

    print(f"\n✅ best convention: '{conv}'   RMS = {rms:.2f} mm "
          f"(next best {runner_up:.2f} mm)")
    if runner_up - rms < 0.5:
        print("   ⚠️  top conventions are close — add more / more varied poses to "
              "separate them confidently.")
    print(f"   flange→image translation (mm): "
          f"[{t_X[0]:+.1f}, {t_X[1]:+.1f}, {t_X[2]:+.1f}]")
    print(f"   flange→image rotation (deg, '{conv}'): "
          f"{np.round(Rotation.from_matrix(R_X).as_euler(conv, degrees=True), 1)}")

    if rms < 2:
        print("   → looks good for a myCobot (a few mm is the realistic floor).")
    elif rms < 5:
        print("   → usable but loose; more poses / wider tilt spread should tighten it.")
    else:
        print("   → too high to trust yet — check the clicks and pose spread.")

    # per-frame error for the winning convention — exposes outliers
    R_flange = Rotation.from_euler(conv, eul, degrees=True).as_matrix()

    spread = 0.0
    for a in range(len(R_flange)):
        for b in range(a + 1, len(R_flange)):
            c = (np.trace(R_flange[a].T @ R_flange[b]) - 1) / 2
            spread = max(spread, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    print(f"\n   pose orientation spread: {spread:.0f}°  "
          f"(want wide; under ~30° is too narrow to pin the transform)")

    base_pts = pos + np.einsum("nij,nj->ni", R_flange, p_img @ R_X.T + t_X)
    dist = np.linalg.norm(base_pts - base_pts.mean(0), axis=1)
    med = float(np.median(dist))
    order = np.argsort(dist)[::-1]
    print(f"\n   per-frame error (median {med:.2f} mm) — worst first:")
    for i in order:
        flag = "  ← outlier" if dist[i] > 2 * med else ""
        print(f"     {dist[i]:6.2f} mm   {used[i]}{flag}")
    print("   If a few frames dominate, re-click them or re-run with")
    print("   --drop <stem1>,<stem2> (only drop frames you genuinely distrust).")
    print("   If every frame is uniformly high, it's systematic — pose spread,")
    print("   near-gimbal-lock poses, or click consistency, not one bad frame.")

    out = section / "handeye.json"
    out.write_text(json.dumps({
        "convention": conv,
        "R_flange_to_image": R_X.tolist(),
        "t_flange_to_image_mm": t_X.tolist(),
        "rms_mm": rms,
        "n_points": len(used),
        "note": "p_base = flange_pos + R_flange @ (R_X @ p_image + t_X); "
                "p_image = [(u-W/2)*lateral_mm, 0, v*axial_mm]",
    }, indent=2))
    print(f"\n💾 {out}")
    print("   Next: wire this into reconstruct_volume.py and re-reconstruct the "
          "phantom to validate.")


if __name__ == "__main__":
    main()