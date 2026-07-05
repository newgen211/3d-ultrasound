#!/usr/bin/env python3
"""
vessel_to_slicer.py — export the vessel tube into 3D Slicer

Writes two files into the section folder, both in the SAME base-frame mm as
volume_handeye.nii.gz, so they overlay on the volume in Slicer:

  vessel_centerline.mrk.json   markups curve (measure length; render as a tube)
  vessel_tube.vtk              model mesh (solid tube surface)

    python3 vessel_to_slicer.py section_50

In Slicer:
  1. Load volume_handeye.nii.gz  (Add Data -> Volume)
  2. Drag in vessel_tube.vtk      -> shows as a Model
     and/or vessel_centerline.mrk.json -> a Markups curve
  3. For the curve: Markups module -> Display -> set "Tube" radius to the vessel
     radius for a tube view; the curve also reports its length.
  4. If it appears mirrored vs the volume, change "RAS" to "LPS" in the .mrk.json
     and re-load (NIfTI/Slicer coordinate-system mismatch escape hatch).

Keep segment_tube.py, vessel_centerline.py, vessel_tube.py in the same folder.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from segment_tube import candidates, load_frame, find_section
from vessel_centerline import track
from vessel_tube import smooth_seq, tube_mesh, SEQ_W, OUTLIER_MM

_REPO_ROOT = Path(__file__).resolve().parents[2]

N_CTRL = 40          # control points in the Slicer curve
COORD_SYS = "RAS"    # switch to "LPS" if it loads mirrored vs the volume


def write_vtk_tube(path, X, Y, Z):
    M, n = X.shape
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    faces = [(i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j)
             for i in range(M - 1) for j in range(n - 1)]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nvessel tube\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write(f"POLYGONS {len(faces)} {len(faces)*5}\n")
        for a, b, c, d in faces:
            f.write(f"4 {a} {b} {c} {d}\n")


def write_mrk_curve(path, C, radius):
    mk = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/"
                   "Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Curve",
            "coordinateSystem": COORD_SYS,
            "controlPoints": [{"id": str(i + 1),
                               "position": [float(p[0]), float(p[1]), float(p[2])]}
                              for i, p in enumerate(C)],
            "display": {"visibility": True, "color": [0.086, 0.749, 0.651],
                        "selectedColor": [0.086, 0.749, 0.651],
                        "lineThickness": 0.4}
        }]
    }
    Path(path).write_text(json.dumps(mk, indent=1))


def main():
    section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    he = None
    for c in [section / "handeye.json", _REPO_ROOT / "handeye.json"]:
        if c.exists():
            he = json.loads(c.read_text()); break
    if he is None:
        sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

    frames_cands, poses = [], []
    for jp in sorted(Path(section).glob("raw_*.json")):
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
        sys.exit("not enough points")

    # outlier reject + clean centerline (same as vessel_tube)
    d = np.linalg.norm(pts - smooth_seq(pts, SEQ_W), axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    keep = d < max(OUTLIER_MM, np.median(d) + 3 * 1.4826 * mad)
    C = smooth_seq(pts[keep], SEQ_W)
    crad = radii[keep]

    idx = np.linspace(0, len(C) - 1, min(N_CTRL, len(C))).astype(int)
    Cs = smooth_seq(C[idx], 2)
    Rs = np.maximum(0.4, smooth_seq(crad[idx].reshape(-1, 1), 2).ravel())
    r_med = float(np.median(crad))

    curve_path = section / "vessel_centerline.mrk.json"
    tube_path = section / "vessel_tube.vtk"
    write_mrk_curve(curve_path, Cs, r_med)
    X, Y, Z = tube_mesh(Cs, Rs)
    write_vtk_tube(tube_path, X, Y, Z)

    length = float(np.linalg.norm(np.diff(Cs, axis=0), axis=1).sum())
    print(f"{section.name}: kept {keep.sum()}/{len(pts)} points, "
          f"length ~{length:.1f} mm, radius ~{r_med:.2f} mm")
    print(f"saved {curve_path.name}  (markups curve, {COORD_SYS})")
    print(f"saved {tube_path.name}  (model mesh)")
    print("load these + volume_handeye.nii.gz together in Slicer")


if __name__ == "__main__":
    main()