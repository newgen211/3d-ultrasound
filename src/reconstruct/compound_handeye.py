#!/usr/bin/env python3
"""
compound_handeye.py — multi-view compounding into one shared grid, with optional
image-based registration to beat the arm's pose error.

    within a view  -> average   (overlapping frames denoise each other)
    across views   -> MAX       (different angles fill each other's gaps/shadows)
    --register     -> slide each later view onto the first by maximizing image
                      overlap (FFT cross-corr) BEFORE the max-merge. Translation
                      only, so it corrects the few-mm pose slip without undoing
                      the intended angle difference. Prints the shift it applied
                      — that number IS the pose error between views.

Views can be separate sections, or one continuous recording auto-split by beam
direction (--split-by-angle).

Usage:
    python3 compound_handeye.py section_A section_B --register
    python3 compound_handeye.py section_46 --split-by-angle --register
    python3 compound_handeye.py section_A section_B --register --max-shift-mm 8

Output: data/clarius_sessions/compound_<ids>/volume_compound.nii.gz, .npy, _mips.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, shift as nd_shift
from scipy.spatial.transform import Rotation

# Anchor to the repo root / data/ folder, regardless of where this is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = _REPO_ROOT / "data"


def resolve_section(arg):
    root = DATA / "clarius_sessions"
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


def splat_trilinear(accum, weight, coords_vox, vals):
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


def posed_frames(section, max_frames):
    out = []
    jsons = sorted(section.glob("raw_*.json"))
    if max_frames:
        jsons = jsons[:max_frames]
    for jp in jsons:
        meta = json.loads(jp.read_text())
        if not jp.with_suffix(".bin").exists():
            continue
        pose = meta.get("cobot_pose")
        if not pose or "coords" not in pose or len(pose["coords"]) < 6:
            continue
        out.append((jp, meta, pose["coords"]))
    return out


def beam_dir(coords, R_X, conv):
    Rf = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
    v = Rf @ (R_X @ np.array([0.0, 0.0, 1.0]))
    return v / (np.linalg.norm(v) + 1e-12)


def split_by_angle(frames, R_X, conv, tol_deg, min_frames):
    clusters = []
    for fr in frames:
        b = beam_dir(fr[2], R_X, conv)
        best, best_ang = None, 1e9
        for c in clusters:
            ang = np.degrees(np.arccos(np.clip(float(np.dot(b, c["mean"])), -1, 1)))
            if ang < best_ang:
                best_ang, best = ang, c
        if best is not None and best_ang <= tol_deg:
            best["items"].append(fr); best["sum"] += b
            best["mean"] = best["sum"] / np.linalg.norm(best["sum"])
        else:
            clusters.append({"mean": b.copy(), "sum": b.copy(), "items": [fr]})
    kept = [c for c in clusters if len(c["items"]) >= min_frames]
    dropped = sum(len(c["items"]) for c in clusters) - sum(len(c["items"]) for c in kept)
    return kept, dropped


def register_shift(ref, mov, max_shift_vox):
    """Integer-voxel translation that aligns `mov` onto `ref` (FFT cross-corr,
    restricted to a small window). Returns (shift_to_apply, overlap_voxels)."""
    a, b = ref.copy(), mov.copy()
    ma, mb = a > 0, b > 0
    if ma.any(): a[ma] -= a[ma].mean()
    if mb.any(): b[mb] -= b[mb].mean()
    a[~ma] = 0; b[~mb] = 0
    C = np.fft.irfftn(np.fft.rfftn(a) * np.conj(np.fft.rfftn(b)), s=a.shape)
    C = np.fft.fftshift(C)
    center = np.array(a.shape) // 2
    sl = tuple(slice(c - max_shift_vox, c + max_shift_vox + 1) for c in center)
    win = C[sl]
    pk = np.array(np.unravel_index(np.argmax(win), win.shape)) - max_shift_vox
    overlap = int(np.count_nonzero(ma & mb))
    return -pk, overlap   # apply nd_shift(mov, -pk) to land mov on ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="+", help="section folder(s)")
    ap.add_argument("--split-by-angle", action="store_true",
                    help="pool all frames and auto-split into views by beam direction")
    ap.add_argument("--angle-tol", type=float, default=8.0)
    ap.add_argument("--min-seg-frames", type=int, default=15)
    ap.add_argument("--register", action="store_true",
                    help="image-register each view onto the first before merging "
                         "(beats arm pose error; translation only)")
    ap.add_argument("--max-shift-mm", type=float, default=6.0,
                    help="max correction the registration may apply (default 6 mm)")
    ap.add_argument("--handeye", default=None)
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--near-crop-mm", type=float, default=0.0)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    sections = [resolve_section(s) for s in args.sections]

    cands = [Path(args.handeye)] if args.handeye else [sections[0] / "handeye.json", _REPO_ROOT / "handeye.json"]
    he, he_path = None, None
    for c in cands:
        if c and c.exists():
            he = json.loads(c.read_text()); he_path = c; break
    if he is None:
        sys.exit("❌ No handeye.json found. Pass --handeye PATH or copy it to the project root.")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]
    print(f"📐 hand-eye: {he_path}  (convention '{conv}', rms {he.get('rms_mm', '?')} mm)")

    by_sec = {sec.name: posed_frames(sec, args.max_frames) for sec in sections}
    all_frames = [fr for sec in sections for fr in by_sec[sec.name]]
    if len(all_frames) < 2:
        sys.exit("❌ Need ≥2 posed frames total — run merge_poses.py.")

    f0 = all_frames[0][1]["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    near = int(round(args.near_crop_mm / axial_mm)) if args.near_crop_mm > 0 else 0
    H_full, W = f0["samples"], f0["lines"]
    H = H_full - near

    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    px = (jj - W / 2.0).astype(np.float64) * lateral_mm
    pz = (ii + near).astype(np.float64) * axial_mm
    pts_img = np.stack([px.ravel(), np.zeros(px.size), pz.ravel()], axis=0)
    q_flange = R_X @ pts_img + t_X[:, None]

    cj = np.array([0, W - 1, W - 1, 0]); ci = np.array([0, 0, H - 1, H - 1])
    corners_img = np.stack([(cj - W / 2.0) * lateral_mm, np.zeros(4),
                            (ci + near) * axial_mm], axis=0)
    corners_flange = R_X @ corners_img + t_X[:, None]

    if args.split_by_angle:
        segs, dropped = split_by_angle(all_frames, R_X, conv, args.angle_tol, args.min_seg_frames)
        ref = segs[0]["mean"] if segs else np.array([0, 0, 1.0])
        segments = []
        for i, c in enumerate(segs):
            tilt = np.degrees(np.arccos(np.clip(float(np.dot(c["mean"], ref)), -1, 1)))
            segments.append((f"view{i}(+{tilt:.0f}°)", c["items"]))
        print(f"🔪 auto-split: {len(segments)} views, dropped {dropped} transition frames")
        if len(segments) < 2:
            print("⚠️  <2 views — angles too similar (or a roll, not a tilt).")
    else:
        segments = [(sec.name, by_sec[sec.name]) for sec in sections]
    for label, items in segments:
        print(f"     {label}: {len(items)} frames")

    used = [fr for _, items in segments for fr in items]
    wc = []
    for (_, _, coords) in used:
        T = np.array(coords[:3], float)
        Rf = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
        wc.append((T[:, None] + Rf @ corners_flange).T)
    wc = np.concatenate(wc, axis=0)
    margin = 3.0
    bmin, bmax = wc.min(0) - margin, wc.max(0) + margin
    voxel = args.voxel
    nx, ny, nz = np.ceil((bmax - bmin) / voxel).astype(int)
    ext = bmax - bmin
    print(f"🗺  shared grid {nx}x{ny}x{nz} @ {voxel}mm ({nx*ny*nz/1e6:.1f} M voxels), "
          f"extent {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} mm")

    def build_view(items):
        accum = np.zeros((nx, ny, nz), np.float32)
        weight = np.zeros_like(accum)
        for jp, meta, coords in items:
            img = load_frame(jp.with_suffix(".bin"), meta)
            T = np.array(coords[:3], float)
            Rf = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
            p_world = T[:, None] + Rf @ q_flange
            I = (img[near:, :] if near > 0 else img).ravel().astype(np.float32)
            m = I.max()
            if m > 0:
                I = I / m
            splat_trilinear(accum, weight, (p_world.T - bmin) / voxel, I)
        a = gaussian_filter(accum, args.sigma, mode="constant", cval=0)
        w = gaussian_filter(weight, args.sigma, mode="constant", cval=0)
        v = np.zeros_like(a)
        cov = w > 1e-3
        v[cov] = a[cov] / w[cov]
        return v, cov

    max_shift_vox = int(round(args.max_shift_mm / voxel))
    vol_max = np.zeros((nx, ny, nz), np.float32)
    n_cov = np.zeros((nx, ny, nz), np.int16)
    ref_vol = None
    for i, (label, items) in enumerate(segments):
        t0 = time.time()
        vol_k, cov_k = build_view(items)

        if args.register and ref_vol is not None:
            s, overlap = register_shift(ref_vol, vol_k, max_shift_vox)
            vol_k = nd_shift(vol_k, s, order=1, mode="constant", cval=0)
            cov_k = nd_shift(cov_k.astype(np.float32), s, order=0, mode="constant", cval=0) > 0.5
            mm = np.array(s) * voxel
            print(f"   {label}: registered  shift (x,y,z) = "
                  f"({mm[0]:+.1f},{mm[1]:+.1f},{mm[2]:+.1f}) mm, |{np.linalg.norm(mm):.1f}| mm, "
                  f"overlap {overlap/1e3:.0f}k voxels")
        elif args.register:
            ref_vol = vol_k.copy()
            print(f"   {label}: reference view (no shift)")

        np.maximum(vol_max, vol_k, out=vol_max)
        n_cov += cov_k.astype(np.int16)
        print(f"   {label}: built {time.time()-t0:.1f}s, coverage {100*cov_k.mean():.1f}%")

    vol = vol_max
    overlap = 100 * (n_cov >= 2).sum() / max(1, (n_cov >= 1).sum())
    print(f"🔗 voxels seen by ≥2 views: {overlap:.1f}% of covered volume")

    tag = ("compound_split_" if args.split_by_angle else "compound_") + \
        "_".join(s.name.split("_")[-1] for s in sections) + ("_reg" if args.register else "")
    out_dir = DATA / "clarius_sessions" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = np.array([[voxel, 0, 0, bmin[0]], [0, voxel, 0, bmin[1]],
                       [0, 0, voxel, bmin[2]], [0, 0, 0, 1.0]], float)
    vmin, vmax = float(vol.min()), float(vol.max())
    scaled = (((vol - vmin) / (vmax - vmin)) * 1000.0).astype(np.float32) if vmax > vmin else vol
    nii = nib.Nifti1Image(scaled, affine)
    nii.header.set_xyzt_units(xyz="mm")
    nib.save(nii, str(out_dir / "volume_compound.nii.gz"))
    np.save(out_dir / "volume_compound.npy", vol)
    np.save(out_dir / "coverage.npy", n_cov)
    print(f"💾 {out_dir / 'volume_compound.nii.gz'}")

    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for a_, axis_, name in ((ax[0], 2, "X–Y (top-down)"),
                                (ax[1], 1, "X–Z"), (ax[2], 0, "Y–Z")):
            a_.imshow(vol.max(axis=axis_).T, cmap="gray", origin="lower", aspect="equal")
            a_.set_title(f"MIP {name}"); a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"{out_dir.name} — {len(segments)} views, max-merge"
                     f"{' + register' if args.register else ''}")
        fig.tight_layout()
        fig.savefig(out_dir / "volume_compound_mips.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("💾 volume_compound_mips.png")

    print("\n✅ Open volume_compound.nii.gz in 3D Slicer.")


if __name__ == "__main__":
    main()