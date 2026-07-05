#!/usr/bin/env python3
"""
reconstruct_handeye.py — metric 3D reconstruction using the hand-eye transform

Places every pixel at its TRUE world location:

    p_base   = flange_pos + R_orient @ (R_X @ p_image + t_X)
    p_image  = [(col - W/2)*lateral_mm, 0, row*axial_mm]

Orientation source (R_orient):
  default : cobot euler  coords[3:6]
  --imu   : Clarius IMU quaternion, auto-aligned to the cobot frame (translation
            still comes from the cobot). Use when the cobot orientation is noisy.
  --smooth N : moving-average the cobot poses over N frames before placement
            (kills pose-noise corrugation on slow dense sweeps; try 25-50).

Viewing:
  --derotate : rotate the whole volume into the slab's mean orientation before
            binning, so the MIPs come out face-on instead of as a sheared,
            tilted parallelogram. Cosmetic only — same data, distances preserved.
            SINGLE-SWEEP ONLY: it puts the volume in the slab frame, not the
            cobot base frame, so do NOT use it for multi-sweep merging.

Needs: a section whose frames have 'cobot_pose' (run merge_poses.py first) and
handeye.json. With --imu the frames must also carry 'imu_samples'.

Usage:
    python3 reconstruct_handeye.py section_22
    python3 reconstruct_handeye.py section_48 --derotate
    python3 reconstruct_handeye.py section_46 --imu --smooth 35

Output (in the section): volume_handeye.nii.gz, .npy, volume_handeye_mips.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.spatial.transform import Rotation

# Anchor to the repo root, regardless of where this is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = _REPO_ROOT / "data"


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


def quat_to_matrix(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], float)


def imu_quat(meta):
    """Return (qw,qx,qy,qz) from the frame's first IMU sample, or None."""
    samples = meta.get("imu_samples") or []
    if not samples:
        return None
    s = samples[0]
    vals = (s.get("qw"), s.get("qx"), s.get("qy"), s.get("qz"))
    return vals if None not in vals else None


def mean_rotation(mats):
    """Average a list of 3x3 rotation matrices (SVD orthogonalization)."""
    A = np.sum(mats, axis=0)
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


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
    ap.add_argument("--handeye", default=None, help="path to handeye.json")
    ap.add_argument("--voxel", type=float, default=0.5, help="voxel size mm (default 0.5)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="gap-fill blur in voxels, isotropic (default 1.0)")
    ap.add_argument("--imu", action="store_true",
                    help="use Clarius IMU orientation (auto-aligned to cobot frame) "
                         "instead of the cobot's noisy euler angles")
    ap.add_argument("--smooth", type=int, default=0,
                    help="moving-avg window (frames) over cobot poses; try 25-50 for "
                         "slow dense sweeps (kills pose-noise corrugation)")
    ap.add_argument("--derotate", action="store_true",
                    help="rotate volume into the slab's mean orientation so the MIPs "
                         "are face-on (cosmetic, single-sweep only — not for merging)")
    ap.add_argument("--near-crop-mm", type=float, default=0.0,
                    help="drop this many mm off the top of each frame")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    section = find_section(args.section)

    # ---- locate handeye.json ----
    cands = [Path(args.handeye)] if args.handeye else [section / "handeye.json", _REPO_ROOT / "handeye.json"]
    he, he_path = None, None
    for c in cands:
        if c and c.exists():
            he = json.loads(c.read_text())
            he_path = c
            break
    if he is None:
        sys.exit("❌ No handeye.json found. Run calibrate_handeye.py, copy it to the "
                 "project root, or pass --handeye PATH.")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]
    print(f"📐 hand-eye: {he_path}  (convention '{conv}', rms {he.get('rms_mm', '?')} mm)")

    # ---- load posed frames ----
    jsons = sorted(section.glob("raw_*.json"))
    if args.max_frames:
        jsons = jsons[:args.max_frames]
    frames = []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            continue
        pose = meta.get("cobot_pose")
        if not pose or "coords" not in pose or len(pose["coords"]) < 6:
            continue
        try:
            img = load_frame(bp, meta)
        except Exception as e:
            print(f"  ⚠️  skip {bp.name}: {e}")
            continue
        frames.append((meta, img, pose["coords"]))
    if len(frames) < 2:
        sys.exit(f"❌ Need ≥2 frames with cobot_pose in {section} — run merge_poses.py")
    print(f"📂 {section}: {len(frames)} posed frames")

    # ---- optional pose smoothing (remove pose-noise corrugation) ----
    if args.smooth > 1:
        cz = np.array([c for (_, _, c) in frames], float)            # (N, 6)
        cz = uniform_filter1d(cz, size=args.smooth, axis=0, mode="nearest")
        frames = [(m, im, cz[i].tolist()) for i, (m, im, _) in enumerate(frames)]
        print(f"   smoothed poses (moving-avg window {args.smooth} frames)")

    # ---- per-frame translation + orientation ----
    Ts = [np.array(c[:3], float) for (_, _, c) in frames]
    Rc = [Rotation.from_euler(conv, c[3:6], degrees=True).as_matrix() for (_, _, c) in frames]

    if args.imu:
        Ri, has = [], []
        for (meta, _, _) in frames:
            q = imu_quat(meta)
            Ri.append(quat_to_matrix(*q) if q else None)
            has.append(q is not None)
        if not any(has):
            sys.exit("❌ --imu requested but no frames carry imu_samples.")
        N = mean_rotation([Ri[k].T @ Rc[k] for k in range(len(frames)) if has[k]])
        Rs = [Ri[k] @ N if has[k] else Rc[k] for k in range(len(frames))]
        errs = np.array([rot_angle_deg(Rs[k].T @ Rc[k]) for k in range(len(frames)) if has[k]])
        half = max(1, len(errs) // 2)
        print(f"   IMU orientation: {int(np.sum(has))} frames "
              f"({len(frames)-int(np.sum(has))} fell back to cobot)")
        print(f"   IMU↔cobot residual: mean {errs.mean():.2f}°, max {errs.max():.2f}°  |  "
              f"first-half {errs[:half].mean():.2f}° vs last-half {errs[half:].mean():.2f}° "
              f"(growing = IMU yaw drift)")
    else:
        Rs = Rc

    # ---- optional derotation into the slab frame (cosmetic, face-on MIPs) ----
    G = np.eye(3)
    if args.derotate:
        G = mean_rotation(Rs).T
        print("   derotated into slab frame for viewing "
              "(single-sweep only — not in cobot base frame, don't merge with this)")

    # ---- scale + image-plane pixel grid (mm) ----
    f0 = frames[0][0]["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    near = int(round(args.near_crop_mm / axial_mm)) if args.near_crop_mm > 0 else 0
    H_full, W = frames[0][1].shape
    H = H_full - near

    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    px = (jj - W / 2.0).astype(np.float64) * lateral_mm
    pz = (ii + near).astype(np.float64) * axial_mm
    pts_img = np.stack([px.ravel(), np.zeros(px.size), pz.ravel()], axis=0)  # (3, Npix)
    q_flange = R_X @ pts_img + t_X[:, None]                                  # image pts in flange frame

    # frame corners for world bounds
    cj = np.array([0, W - 1, W - 1, 0]); ci = np.array([0, 0, H - 1, H - 1])
    corners_img = np.stack([(cj - W / 2.0) * lateral_mm, np.zeros(4),
                            (ci + near) * axial_mm], axis=0)
    corners_flange = R_X @ corners_img + t_X[:, None]                        # (3,4)

    wc = [(G @ (Ts[k][:, None] + Rs[k] @ corners_flange)).T for k in range(len(frames))]
    wc = np.concatenate(wc, axis=0)
    margin = 3.0
    bmin = wc.min(0) - margin
    bmax = wc.max(0) + margin
    voxel = args.voxel
    nx, ny, nz = np.ceil((bmax - bmin) / voxel).astype(int)
    ext = bmax - bmin
    print(f"   volume {nx}x{ny}x{nz} @ {voxel}mm ({nx*ny*nz/1e6:.1f} M voxels), "
          f"extent {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} mm")

    # ---- accumulate (full-pose placement) ----
    accum = np.zeros((nx, ny, nz), np.float32)
    weight = np.zeros_like(accum)
    t0 = time.time()
    for k, (meta, img, coords) in enumerate(frames):
        p_world = G @ (Ts[k][:, None] + Rs[k] @ q_flange)    # (3, Npix)
        coords_vox = (p_world.T - bmin) / voxel
        I = img[near:, :] if near > 0 else img
        I = I.ravel().astype(np.float32)
        m = I.max()
        if m > 0:
            I = I / m
        splat_trilinear(accum, weight, coords_vox, I)
    print(f"   splat: {time.time()-t0:.1f}s")

    # ---- blur + divide ----
    a = gaussian_filter(accum, args.sigma, mode="constant", cval=0)
    w = gaussian_filter(weight, args.sigma, mode="constant", cval=0)
    vol = np.zeros_like(a)
    valid = w > 1e-3
    vol[valid] = a[valid] / w[valid]
    print(f"   occupancy: {100*valid.mean():.1f}%")

    # ---- export NIfTI ----
    affine = np.array([[voxel, 0, 0, bmin[0]],
                       [0, voxel, 0, bmin[1]],
                       [0, 0, voxel, bmin[2]],
                       [0, 0, 0, 1.0]], float)
    vmin, vmax = float(vol.min()), float(vol.max())
    scaled = (((vol - vmin) / (vmax - vmin)) * 1000.0).astype(np.float32) if vmax > vmin else vol
    nii = nib.Nifti1Image(scaled, affine)
    nii.header.set_xyzt_units(xyz="mm")
    out = section / "volume_handeye.nii.gz"
    nib.save(nii, str(out))
    np.save(section / "volume_handeye.npy", vol)
    print(f"💾 {out}")

    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for a_, axis_, name in ((ax[0], 2, "X–Y (top-down)"),
                                (ax[1], 1, "X–Z"),
                                (ax[2], 0, "Y–Z")):
            a_.imshow(vol.max(axis=axis_).T, cmap="gray", origin="lower", aspect="equal")
            a_.set_title(f"MIP {name}")
            a_.set_xticks([]); a_.set_yticks([])
        tags = []
        if args.imu:
            tags.append("IMU orient")
        if args.smooth > 1:
            tags.append(f"smooth {args.smooth}")
        if args.derotate:
            tags.append("derotated")
        tag = f", {', '.join(tags)}" if tags else ""
        fig.suptitle(f"{section.name} — hand-eye reconstruction ({len(frames)} frames{tag})")
        fig.tight_layout()
        fig.savefig(section / "volume_handeye_mips.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("💾 volume_handeye_mips.png")

    print("\n✅ Open volume_handeye.nii.gz in 3D Slicer and measure.")


if __name__ == "__main__":
    main()