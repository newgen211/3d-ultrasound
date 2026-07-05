#!/usr/bin/env python
"""
reconstruct_volume.py — freehand 3D ultrasound reconstruction with NIfTI export

Takes a sweep folder (clarius_sessions/section_N/) and produces:
  - volume.nii.gz   3D voxel volume for 3D Slicer / ITK-SNAP (mm voxel spacing)
  - volume.npy      raw float32 voxel array
  - volume_mips.png top / front / side maximum-intensity projections (quick look)
  - volume_slices.png three orthogonal mid-slices

How it works:
  1. Place each frame in 3D space (IMU rotation + sweep translation)
  2. Trilinear-splat each pixel into its 8 surrounding voxels
  3. Anisotropic Gaussian smoothing (more along the sparse sweep axis, little
     in-plane) to fill gaps without blurring real structure
  4. Save as NIfTI with correct voxel spacing in mm

Note on what you'll see: a fluid-filled tube is ANECHOIC, so its lumen is dark
and the bright parts are the gel and the tube walls. That's correct — scroll the
slices in Slicer to see/measure the tube directly. (To measure a diameter, use
Slicer's Segment Editor: Threshold the dark lumen on one slice, or drop a Line
markup across it.)

Usage:
    python reconstruct_volume.py                       # latest section
    python reconstruct_volume.py section_16
    python reconstruct_volume.py section_16 --span 50 --near-crop-mm 3
    python reconstruct_volume.py section_16 --no-imu   # stack planes straight
    python reconstruct_volume.py section_16 --sigma 1.2 --sigma-inplane 0.4
    python reconstruct_volume.py section_16 --no-png   # skip PNG previews

    --span is the assumed physical sweep length in mm (no cobot pose yet). Set
    it to your taped start->end distance so measurements are metrically correct.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


def find_section(arg):
    root = Path("clarius_sessions")
    if not root.exists():
        sys.exit(f"❌ No clarius_sessions/ folder in {Path.cwd()}")
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
    return arr.reshape(lines, samples).T.astype(np.float32)


def quat_to_matrix(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    if n == 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def splat_trilinear(accum, weight, coords_vox, vals):
    """Distribute each (voxel-space coord, value) into its 8 neighbour voxels."""
    nx, ny, nz = accum.shape
    fa, fw = accum.reshape(-1), weight.reshape(-1)
    base = np.floor(coords_vox).astype(np.int64)
    frac = coords_vox - base
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        ix = base[:, 0] + dx
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            iy = base[:, 1] + dy
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                iz = base[:, 2] + dz
                w = (wx * wy * wz).astype(np.float32)
                m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
                lin = (ix[m] * ny + iy[m]) * nz + iz[m]
                np.add.at(fa, lin, vals[m] * w[m])
                np.add.at(fw, lin, w[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--voxel", type=float, default=0.5, help="voxel size mm (default 0.5)")
    ap.add_argument("--span", type=float, default=50.0,
                    help="assumed sweep length mm (set to taped distance)")
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"],
                    help="world axis to sweep along (default y)")
    ap.add_argument("--sigma", type=float, default=1.2,
                    help="gap-fill blur ALONG sweep axis, voxels (default 1.2)")
    ap.add_argument("--sigma-inplane", type=float, default=0.4,
                    help="gap-fill blur in-plane, voxels (default 0.4)")
    ap.add_argument("--near-crop-mm", type=float, default=0.0,
                    help="drop this many mm off the top of each frame (gel standoff)")
    ap.add_argument("--no-imu", action="store_true",
                    help="ignore IMU rotation; stack planes straight")
    ap.add_argument("--no-png", action="store_true", help="skip PNG previews")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    section = find_section(args.section)
    print(f"📂 {section}")

    jsons = sorted(section.glob("raw_*.json"))
    if args.max_frames:
        jsons = jsons[:args.max_frames]
    if not jsons:
        sys.exit("❌ No frames")
    print(f"   {len(jsons)} frames")

    # ---- load frames ----
    frames = []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            continue
        try:
            img = load_frame(bp, meta)
        except Exception as e:
            print(f"  ⚠️  skip {bp.name}: {e}")
            continue
        q = (1.0, 0.0, 0.0, 0.0)
        if meta.get("imu_samples"):
            s = meta["imu_samples"][0]
            vals = (s.get("qw"), s.get("qx"), s.get("qy"), s.get("qz"))
            if None not in vals:
                q = vals
        frames.append((meta, img, q, meta.get("probe_timestamp_ns")))
    if not frames:
        sys.exit("❌ No usable frames")

    # ---- physical scale + near-field crop ----
    f0 = frames[0][0]["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    near_rows = int(round(args.near_crop_mm / axial_mm)) if args.near_crop_mm > 0 else 0
    if near_rows > 0:
        frames = [(m, img[near_rows:, :], q, ts) for (m, img, q, ts) in frames]

    H, W = frames[0][1].shape
    depth_mm = (H + near_rows) * axial_mm
    width_mm = W * lateral_mm
    print(f"   frame: {width_mm:.1f}×{depth_mm:.1f} mm  "
          f"(pixel {lateral_mm:.3f}×{axial_mm:.3f} mm, near-crop {near_rows}px)")

    # ---- timestamp-proportional sweep positions ----
    t0 = frames[0][3]
    t_span = (frames[-1][3] - t0) if (t0 is not None and frames[-1][3] is not None) else None
    if not t_span or t_span <= 0:
        t_span = None

    # ---- volume bounds ----
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    margin = 2.0
    bounds_min = np.array([-width_mm / 2 - margin, -margin, -margin], dtype=np.float32)
    bounds_max = np.array([+width_mm / 2 + margin, +margin, depth_mm + margin], dtype=np.float32)
    bounds_min[axis_idx] = -margin
    bounds_max[axis_idx] = args.span + margin

    voxel = args.voxel
    grid_dims = np.ceil((bounds_max - bounds_min) / voxel).astype(int)
    nx, ny, nz = grid_dims
    print(f"   volume: {nx}×{ny}×{nz} voxels @ {voxel} mm ({nx*ny*nz/1e6:.2f} M)")

    accum = np.zeros((nx, ny, nz), dtype=np.float32)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)
    R_ref_inv = quat_to_matrix(*frames[0][2]).T

    j_idx, i_idx = np.meshgrid(np.arange(W), np.arange(H))
    px = (j_idx - W / 2.0).astype(np.float32) * lateral_mm
    pz = (i_idx + near_rows).astype(np.float32) * axial_mm
    py = np.zeros_like(px)
    pts_frame = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=0)

    # ---- accumulation: trilinear splat ----
    print("   accumulating frames (trilinear)…")
    t_start = time.time()
    for fi, (meta, img, q, ts) in enumerate(frames):
        if t_span is not None and ts is not None:
            s_pos = ((ts - t0) / t_span) * args.span
        else:
            s_pos = (fi / max(1, len(frames) - 1)) * args.span
        translation = np.zeros(3, dtype=np.float32)
        translation[axis_idx] = s_pos

        R = (np.eye(3, dtype=np.float32) if args.no_imu
             else (R_ref_inv @ quat_to_matrix(*q)).astype(np.float32))

        pts_world = R @ pts_frame + translation[:, None]
        coords_vox = (pts_world.T - bounds_min) / voxel

        I = img.ravel().astype(np.float32)
        if I.max() > 0:
            I = I / I.max()  # per-frame normalise (handles gain drift across sweep)
        splat_trilinear(accum, weight, coords_vox, I)

    print(f"   accumulate: {time.time()-t_start:.1f}s")

    # ---- anisotropic blur, then divide ----
    sigma_vec = [args.sigma_inplane, args.sigma_inplane, args.sigma_inplane]
    sigma_vec[axis_idx] = args.sigma
    print(f"   smoothing (σ sweep={args.sigma}, in-plane={args.sigma_inplane})…")
    accum_blur = gaussian_filter(accum, sigma=sigma_vec, mode="constant", cval=0)
    weight_blur = gaussian_filter(weight, sigma=sigma_vec, mode="constant", cval=0)

    valid = weight_blur > 1e-3
    volume = np.zeros_like(accum_blur)
    volume[valid] = accum_blur[valid] / weight_blur[valid]
    print(f"   occupancy: {100*valid.mean():.1f}%")

    # ---- export NIfTI (rescaled 0..1000 for nice Slicer windowing) ----
    affine = np.array([
        [voxel, 0,     0,     bounds_min[0]],
        [0,     voxel, 0,     bounds_min[1]],
        [0,     0,     voxel, bounds_min[2]],
        [0,     0,     0,     1.0],
    ], dtype=np.float32)
    vmin, vmax = float(volume.min()), float(volume.max())
    scaled = (((volume - vmin) / (vmax - vmin)) * 1000.0).astype(np.float32) if vmax > vmin else volume
    nii = nib.Nifti1Image(scaled, affine)
    nii.header.set_xyzt_units(xyz="mm")
    nii_path = section / "volume.nii.gz"
    nib.save(nii, str(nii_path))
    print(f"💾 {nii_path}  ({nii_path.stat().st_size/1024:.0f} KB)")
    np.save(section / "volume.npy", volume)

    # ---- PNG previews ----
    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(volume.max(axis=2).T, cmap="gray", aspect="auto", origin="lower",
                       extent=[bounds_min[0], bounds_max[0],
                               bounds_min[axis_idx], bounds_max[axis_idx]])
        axes[0].set_title(f"Top-down (X-{args.axis.upper()})")
        axes[0].set_xlabel("X (mm)"); axes[0].set_ylabel(f"{args.axis.upper()} / sweep (mm)")
        axes[1].imshow(volume.max(axis=1).T, cmap="gray", aspect="auto", origin="upper",
                       extent=[bounds_min[0], bounds_max[0], bounds_max[2], bounds_min[2]])
        axes[1].set_title("Front (X-Z) — looking along sweep")
        axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Z / depth (mm)")
        axes[2].imshow(volume.max(axis=0).T, cmap="gray", aspect="auto", origin="upper",
                       extent=[bounds_min[axis_idx], bounds_max[axis_idx],
                               bounds_max[2], bounds_min[2]])
        axes[2].set_title(f"Side ({args.axis.upper()}-Z)")
        axes[2].set_xlabel(f"{args.axis.upper()} / sweep (mm)"); axes[2].set_ylabel("Z / depth (mm)")
        fig.suptitle(f"{section.name} — MIPs (voxel={voxel}mm, {len(frames)} frames, "
                     f"imu={'off' if args.no_imu else 'on'})")
        fig.tight_layout()
        fig.savefig(section / "volume_mips.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("💾 volume_mips.png")

        sx, sy, sz = nx // 2, ny // 2, nz // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(volume[sx, :, :].T, cmap="gray", origin="upper", aspect="auto")
        axes[0].set_title(f"Slice X={sx}")
        axes[1].imshow(volume[:, sy, :].T, cmap="gray", origin="upper", aspect="auto")
        axes[1].set_title(f"Slice {args.axis.upper()}={sy}")
        axes[2].imshow(volume[:, :, sz].T, cmap="gray", origin="lower", aspect="auto")
        axes[2].set_title(f"Slice Z={sz}")
        fig.suptitle(f"{section.name} — orthogonal mid-slices")
        fig.tight_layout()
        fig.savefig(section / "volume_slices.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("💾 volume_slices.png")

    print(f"\n✅ Open {nii_path} in 3D Slicer (File → Add Data). Scroll the slice "
          f"views to see the tube; it reads dark because fluid is anechoic.")


if __name__ == "__main__":
    main()