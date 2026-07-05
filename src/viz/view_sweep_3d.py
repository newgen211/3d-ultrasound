#!/usr/bin/env python
"""
view_sweep_3d.py — IMU-only 3D preview of a sweep

Places each 2D B-mode frame in 3D space using:
  - rotation from the IMU quaternion
  - translation assumed linear at constant velocity along one axis
    (since we don't have cobot pose yet — Phase 3 fixes this)

This is a "does the data look like a real object?" test, not a real
reconstruction. A real reconstruction needs cobot end-effector pose.

Usage:
    python view_sweep_3d.py                         # latest section
    python view_sweep_3d.py section_3               # specific
    python view_sweep_3d.py section_3 --axis y      # translate along Y
    python view_sweep_3d.py section_3 --span 50     # 50 mm total sweep
    python view_sweep_3d.py section_3 --stride 4    # render every 4th frame (faster)
    python view_sweep_3d.py section_3 --no-imu      # ignore IMU, just stack flat
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401


def find_section(arg: str | None) -> Path:
    root = Path(__file__).resolve().parents[2] / "data" / "clarius_sessions"
    if not root.exists():
        print(f"❌ No clarius_sessions/ folder at {root}")
        sys.exit(1)
    if arg is None:
        sections = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("section_")],
            key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
        )
        if not sections:
            print(f"❌ No section_N folders in {root}")
            sys.exit(1)
        return sections[-1]
    p = Path(arg)
    if p.exists():
        return p
    if (root / arg).exists():
        return root / arg
    print(f"❌ Section folder not found: {arg}")
    sys.exit(1)


def load_frame(bin_path: Path, meta: dict) -> np.ndarray:
    """Decode a raw .bin frame into a 2D numpy array (samples × lines)."""
    frame = meta["frame"]
    lines = frame["lines"]
    samples = frame["samples"]
    bps = frame["bps"]
    jpg_size = frame.get("jpg_size", 0)

    raw_bytes = bin_path.read_bytes()
    if jpg_size > 0:
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(raw_bytes)))

    dtype = np.uint8 if bps == 8 else np.uint16
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    expected = lines * samples
    if arr.size != expected:
        usable = (arr.size // lines) * lines
        arr = arr[:usable]
        samples = usable // lines
    return arr.reshape(lines, samples).T  # depth × width


def quat_to_matrix(qw, qx, qy, qz):
    """Convert scalar-first quaternion to 3x3 rotation matrix."""
    # normalize first to be safe
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def frame_corners(width_mm: float, depth_mm: float):
    """Return the 4 corners of a frame plane in its own coords (mm)."""
    # frame plane = X (width, lateral) × Z (depth), Y = 0 (elevational/out-of-plane)
    w = width_mm / 2
    d = depth_mm
    return np.array([
        [-w, 0, 0],
        [+w, 0, 0],
        [+w, 0, d],
        [-w, 0, d],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"],
                    help="World axis to translate along (default: y)")
    ap.add_argument("--span", type=float, default=50.0,
                    help="Assumed total sweep distance in mm (default: 50)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Render every Nth frame (default: 1 = all)")
    ap.add_argument("--no-imu", action="store_true",
                    help="Ignore IMU rotation, just stack frames flat")
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()

    section = find_section(args.section)
    print(f"📂 {section}")

    jsons = sorted(section.glob("raw_*.json"))[::args.stride]
    if not jsons:
        print("❌ No frames found")
        sys.exit(1)
    print(f"   {len(jsons)} frames (stride={args.stride})")

    # load frames + IMU
    frames = []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            continue
        try:
            img = load_frame(bp, meta)
        except Exception:
            continue
        # take the first IMU sample of the frame for orientation
        q = (1.0, 0.0, 0.0, 0.0)  # identity if no IMU
        if meta.get("imu_samples"):
            s = meta["imu_samples"][0]
            qw, qx, qy, qz = s.get("qw"), s.get("qx"), s.get("qy"), s.get("qz")
            if None not in (qw, qx, qy, qz):
                q = (qw, qx, qy, qz)
        frames.append((meta, img, q))

    if not frames:
        print("❌ No usable frames")
        sys.exit(1)

    # compute physical dimensions per frame
    f0 = frames[0][0]["frame"]
    depth_mm = f0["samples"] * f0["axial_um_per_sample"] / 1000.0
    width_mm = f0["lines"] * f0["lateral_um_per_line"] / 1000.0
    print(f"   frame: {width_mm:.1f} mm wide × {depth_mm:.1f} mm deep")
    print(f"   sweep: {args.span} mm along {args.axis.upper()} axis (assumed)")

    # axis index for translation
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]

    # set up figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        f"Sweep 3D preview — {section.name}\n"
        f"({len(frames)} frames, IMU-{'off' if args.no_imu else 'on'}, "
        f"linear motion assumed along {args.axis.upper()})",
        fontsize=11
    )

    # use the first frame's IMU as the reference orientation
    # so the volume "starts upright" in the world frame
    q_ref = frames[0][2]
    R_ref_inv = quat_to_matrix(*q_ref).T

    for i, (meta, img, q) in enumerate(frames):
        # frame position along sweep axis
        s = (i / max(1, len(frames) - 1)) * args.span
        translation = np.zeros(3)
        translation[axis_idx] = s

        # rotation: IMU rotation relative to first frame
        if args.no_imu:
            R = np.eye(3)
        else:
            R_cur = quat_to_matrix(*q)
            R = R_ref_inv @ R_cur

        # transform the 4 corners
        corners = frame_corners(width_mm, depth_mm)  # (4, 3)
        world_corners = (R @ corners.T).T + translation  # (4, 3)

        # show frame as a textured-ish quad
        # we approximate by sampling a low-res grid from the image and plotting
        # as a 3D wireframe + mean color
        img_norm = img.astype(float)
        img_norm = (img_norm - img_norm.min()) / max(1.0, img_norm.ptp())

        # draw the quad outline
        outline = np.vstack([world_corners, world_corners[:1]])
        ax.plot(outline[:, 0], outline[:, 1], outline[:, 2],
                color="0.5", alpha=0.3, linewidth=0.5)

        # render as a colored surface via a coarse grid
        # interpolate corners into a small grid
        gh, gw = 24, 24  # render resolution per frame
        u = np.linspace(0, 1, gw)[None, :]
        v = np.linspace(0, 1, gh)[:, None]
        # bilinear interp of 3D corner positions
        top = world_corners[0] * (1 - u[..., None]) + world_corners[1] * u[..., None]
        bot = world_corners[3] * (1 - u[..., None]) + world_corners[2] * u[..., None]
        grid_xyz = top * (1 - v[..., None]) + bot * v[..., None]  # (gh, gw, 3)

        # sample image at corresponding grid points
        ih, iw = img_norm.shape
        ys = (v[:, 0] * (ih - 1)).astype(int)
        xs = (u[0, :] * (iw - 1)).astype(int)
        sample = img_norm[np.ix_(ys, xs)]  # (gh, gw)

        ax.plot_surface(
            grid_xyz[..., 0], grid_xyz[..., 1], grid_xyz[..., 2],
            facecolors=plt.get_cmap(args.cmap)(sample),
            shade=False, alpha=args.alpha, antialiased=False, linewidth=0,
        )

    # axes labels with units
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z / depth (mm)")
    ax.invert_zaxis()  # depth increases downward like in B-mode
    # equal-ish aspect ratio
    ax.set_box_aspect([1.0, max(1.0, args.span / max(width_mm, depth_mm)), 1.0])

    out = section / "preview_3d.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"💾 Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()