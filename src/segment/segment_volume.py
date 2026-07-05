#!/usr/bin/env python3
"""
segment_volume.py — confirm the 2D vessel inside the dense volume (seeded)

The blind Frangi version drowned in clutter: every frame-stripe and wall echo in
the volume is just as tubular as the lumen, so a generic filter can't pick the
vessel out. So we flip it around — SEED with the 2D vessel centerline (the
reliable, tracked detector) and ask the volume ONE question:

    does the dense data actually support a vessel where the 2D path put it?

For each point on the 2D centerline we sample the volume at the center (should be
DARK = anechoic lumen) and on a ring at the vessel radius (should be BRIGHTER =
wall). Positive contrast along the centerline = the volume confirms the vessel.
We also snap each point to the local dark centroid and report how far it moved:
a small shift means the volume agrees with where the 2D path placed the vessel.

    python3 segment_volume.py section_59_cam

Outputs (in the section):
  volume_vessel_confirm.png       volume MIPs with 2D (cyan) + refined (yellow) centerline + contrast profile
  volume_vessel_confirm.mrk.json  refined centerline (world mm) for Slicer
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates

from segment_tube import candidates, load_frame, find_section, track

_REPO_ROOT = Path(__file__).resolve().parents[2]

RING_N = 16            # samples around the wall ring
REFINE_MM = 3.0        # half-width of the perpendicular search for the dark centroid


def smooth_seq(P, w=5):
    out = P.copy().astype(float)
    for i in range(len(P)):
        out[i] = np.median(P[max(0, i - w): i + w + 1], axis=0)
    return out


def perp_frame(T):
    """two unit vectors perpendicular to unit tangent T."""
    a = np.array([0, 0, 1.0]) if abs(T[2]) < 0.9 else np.array([0, 1.0, 0])
    N = np.cross(T, a); N /= (np.linalg.norm(N) + 1e-9)
    B = np.cross(T, N)
    return N, B


def main():
    section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)

    # ---- hand-eye ----
    he = None
    for c in [section / "handeye.json", _REPO_ROOT / "handeye.json"]:
        if c.exists():
            he = json.loads(c.read_text()); break
    if he is None:
        sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

    # ---- 2D vessel centerline (same tracked detector as vessel_tube) ----
    jsons = sorted(Path(section).glob("raw_*.json"))
    frames_cands, poses = [], []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        pose = meta.get("cobot_pose")
        if not bp.exists() or not pose or len(pose.get("coords", [])) < 6:
            frames_cands.append([]); poses.append(None); continue
        f = meta["frame"]
        try:
            frames_cands.append(candidates(load_frame(bp, meta),
                                           f["axial_um_per_sample"] / 1000.0,
                                           f["lateral_um_per_line"] / 1000.0))
        except Exception:
            frames_cands.append([])
        poses.append(pose["coords"])

    picks = track(frames_cands)
    pts, radii = [], []
    for pick, coords in zip(picks, poses):
        if pick is None or coords is None:
            continue
        p_img = np.array([pick["cx_mm"], 0.0, pick["depth_mm"]])
        T = np.array(coords[:3], float)
        Rf = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
        pts.append(T + Rf @ (R_X @ p_img + t_X))
        radii.append(pick["r_mm"])
    pts, radii = np.array(pts), np.array(radii)
    if len(pts) < 8:
        sys.exit("not enough 2D vessel points")

    # outlier reject + smooth (mirror vessel_tube)
    sm = smooth_seq(pts)
    d = np.linalg.norm(pts - sm, axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    keep = d < max(4.0, np.median(d) + 3 * 1.4826 * mad)
    cl_world = smooth_seq(pts[keep])
    r_mm = float(np.median(radii[keep]))
    print(f"2D centerline: {len(cl_world)} pts, vessel radius ~{r_mm:.2f} mm")

    # ---- volume ----
    nii_path = section / "volume_handeye.nii.gz"
    if not nii_path.exists():
        sys.exit(f"no volume at {nii_path} — run reconstruct_handeye.py first")
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    inv = np.linalg.inv(affine)
    vox = float(abs(affine[0, 0]))
    occ = (vol > vol.max() * 0.02).astype(np.float32)
    print(f"volume {vol.shape} @ {vox:.2f} mm/vox")

    cl_vox = nib.affines.apply_affine(inv, cl_world)
    r_vox = r_mm / vox
    Tg = np.gradient(cl_vox, axis=0)
    Tg /= (np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9)

    def sample(p):
        return float(map_coordinates(vol, np.array(p).reshape(3, 1), order=1, mode="constant")[0])
    def sample_occ(p):
        return float(map_coordinates(occ, np.array(p).reshape(3, 1), order=1, mode="constant")[0])

    th = np.linspace(0, 2 * np.pi, RING_N, endpoint=False)
    lumen, wall, refined = [], [], []
    for i, c in enumerate(cl_vox):
        N, B = perp_frame(Tg[i])
        lumen.append(sample(c))
        ring = [sample(c + r_vox * (np.cos(a) * N + np.sin(a) * B)) for a in th]
        ringo = [sample_occ(c + r_vox * (np.cos(a) * N + np.sin(a) * B)) for a in th]
        valid = [v for v, o in zip(ring, ringo) if o > 0.5]
        wall.append(np.mean(valid) if valid else np.nan)

        # refine: weighted dark centroid in the perpendicular plane
        gs = np.linspace(-REFINE_MM / vox, REFINE_MM / vox, 9)
        du = dv = wsum = 0.0
        vals = []
        for u in gs:
            for v in gs:
                pp = c + u * N + v * B
                if sample_occ(pp) > 0.5:
                    vals.append((u, v, sample(pp)))
        if vals:
            vmax = max(x[2] for x in vals)
            for u, v, val in vals:
                wgt = max(0.0, vmax - val)            # darker = heavier
                du += wgt * u; dv += wgt * v; wsum += wgt
            if wsum > 0:
                du /= wsum; dv /= wsum
        shift = min(np.hypot(du, dv), REFINE_MM / vox)
        ang = np.arctan2(dv, du)
        refined.append(c + shift * np.cos(ang) * N + shift * np.sin(ang) * B)

    lumen = np.array(lumen); wall = np.array(wall); refined = np.array(refined)
    contrast = wall - lumen
    ok = np.isfinite(contrast)
    supported = float(np.mean(contrast[ok] > 0)) * 100.0
    shift_mm = float(np.median(np.linalg.norm(refined - cl_vox, axis=1)) * vox)

    print(f"\n=== does the volume support the 2D vessel? ===")
    print(f"  wall>lumen contrast positive on {supported:.0f}% of the centerline "
          f"(mean contrast {np.nanmean(contrast):.0f})")
    print(f"  refine shift to local dark centroid: median {shift_mm:.2f} mm "
          f"({'volume agrees with the 2D location' if shift_mm < r_mm else 'volume pulls it elsewhere'})")
    verdict = ("CONFIRMED — dark lumen + bright wall where the 2D path put it"
               if supported > 60 and shift_mm < r_mm else
               "WEAK — volume only partly supports it (sparse/cluttered volume)")
    print(f"  verdict: {verdict}")

    # ---- Slicer markup (refined, world mm) ----
    refined_world = nib.affines.apply_affine(affine, refined)
    markup = {"@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
              "markups": [{"type": "Curve", "coordinateSystem": "RAS", "label": "vessel_confirmed",
                           "controlPoints": [{"id": str(i+1), "position": [float(p[0]), float(p[1]), float(p[2])]}
                                             for i, p in enumerate(refined_world)]}]}
    (section / "volume_vessel_confirm.mrk.json").write_text(json.dumps(markup, indent=2))

    # ---- figure: volume MIPs + overlays + contrast profile ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a_, (axis_, name, h, w) in zip(ax[:3], [(2, "X-Y", 0, 1), (1, "X-Z", 0, 2), (0, "Y-Z", 1, 2)]):
        a_.imshow(vol.max(axis=axis_).T, cmap="gray", origin="lower", aspect="equal")
        a_.plot(cl_vox[:, h], cl_vox[:, w], color="#16BFA6", lw=2, label="2D centerline")
        a_.plot(refined[:, h], refined[:, w], color="#F2C200", lw=1.5, ls="--", label="volume-refined")
        a_.set_title(f"volume MIP {name}"); a_.set_xticks([]); a_.set_yticks([])
    ax[0].legend(loc="upper right", fontsize=8)
    s = np.arange(len(lumen))
    ax[3].plot(s, lumen, color="#C2410C", label="lumen (center)")
    ax[3].plot(s, wall, color="#0F6E56", label="wall (ring)")
    ax[3].set_title(f"intensity along centerline\nwall>lumen on {supported:.0f}%")
    ax[3].set_xlabel("centerline point"); ax[3].legend(fontsize=8)
    fig.suptitle(f"{section.name} — does the volume confirm the 2D vessel?  ({verdict.split(' — ')[0]})")
    fig.tight_layout()
    fig.savefig(section / "volume_vessel_confirm.png", dpi=120, bbox_inches="tight")
    print("\nsaved volume_vessel_confirm.png and volume_vessel_confirm.mrk.json")


if __name__ == "__main__":
    main()
