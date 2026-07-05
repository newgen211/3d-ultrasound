#!/usr/bin/env python3
"""
vessel_tube.py — clean vessel model: reject outliers, render a real 3D tube

Builds on vessel_centerline: tracks the vessel, projects to 3D, then
  1. rejects tracker outliers (points far from the smoothed path),
  2. smooths the centerline,
  3. sweeps a circle of the vessel's radius along it -> an actual 3D tube.

Reports the CLEAN perpendicular spread (the true tube tightness) so you have a
baseline to compare cobot-pose vs camera-pose reconstructions.

    python3 vessel_tube.py section_50
    python3 vessel_tube.py section_50 --straighten

--straighten : spin the centerline's principal axis onto X so the tube renders
    horizontal instead of running diagonally across the camera frame. COSMETIC
    ONLY — the spread is measured before the rotation, so the number is identical.

Keep segment_tube.py and vessel_centerline.py in the same folder.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from segment_tube import candidates, load_frame, find_section
from vessel_centerline import track

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = _REPO_ROOT / "data"

SEQ_W = 5            # window for smoothing the ordered path (frames each side)
OUTLIER_MM = 4.0     # min reject threshold (also uses robust MAD)
TUBE_PTS = 60        # centerline points in the tube mesh
RING_N = 18          # facets around the tube


def smooth_seq(P, w):
    out = P.copy().astype(float)
    for i in range(len(P)):
        out[i] = np.median(P[max(0, i - w): i + w + 1], axis=0)
    return out


def tube_mesh(C, R, n=RING_N):
    M = len(C)
    T = np.gradient(C, axis=0)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    ref = np.array([0, 0, 1.0])
    if abs(T[0] @ ref) > 0.9:
        ref = np.array([0, 1.0, 0])
    Ns = [np.cross(T[0], ref) / (np.linalg.norm(np.cross(T[0], ref)) + 1e-9)]
    for i in range(1, M):                      # rotation-minimizing-ish frame
        n_ = Ns[-1] - (Ns[-1] @ T[i]) * T[i]
        nn = np.linalg.norm(n_)
        Ns.append(n_ / nn if nn > 1e-6 else
                  np.cross(T[i], ref) / (np.linalg.norm(np.cross(T[i], ref)) + 1e-9))
    Ns = np.array(Ns)
    Bs = np.cross(T, Ns)
    th = np.linspace(0, 2 * np.pi, n)
    X, Y, Z = (np.zeros((M, n)) for _ in range(3))
    for i in range(M):
        ring = C[i] + R[i] * (np.cos(th)[:, None] * Ns[i] + np.sin(th)[:, None] * Bs[i])
        X[i], Y[i], Z[i] = ring[:, 0], ring[:, 1], ring[:, 2]
    return X, Y, Z


def save_html(section, X, Y, Z, Cs, out_pts, length, spread):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (--html needs plotly:  pip install plotly --break-system-packages)")
        return
    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z, surfacecolor=np.zeros_like(X),
                    colorscale=[[0, "#16BFA6"], [1, "#0F6E56"]], showscale=False,
                    lighting=dict(ambient=0.55, diffuse=0.8, specular=0.3, roughness=0.6),
                    lightposition=dict(x=100, y=200, z=300))
    fig.add_scatter3d(x=Cs[:, 0], y=Cs[:, 1], z=Cs[:, 2], mode="lines",
                      line=dict(color="#0B2B36", width=4), name="centerline")
    if len(out_pts):
        fig.add_scatter3d(x=out_pts[:, 0], y=out_pts[:, 1], z=out_pts[:, 2], mode="markers",
                          marker=dict(size=3, color="#C2410C", opacity=0.6), name="rejected outliers")
    fig.update_layout(
        title=f"{Path(section).name} — vessel tube (len {length:.0f} mm, spread {spread:.1f} mm)",
        template="plotly_white", width=1050, height=720,
        scene=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"))
    out = Path(section) / "vessel_tube.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"saved {out}  — open in a browser, spin it, screen-grab or record a GIF")


def save_vtk(section, X, Y, Z, lps=False):
    """Write the tube surface as a VTK legacy PolyData model for Slicer overlay.
    tube_mesh uses th = linspace(0, 2pi, n), so ring point j=0 and j=n-1 coincide
    (the ring already closes); connect j in 0..n-2 with no wrap-around.
    lps=True pre-negates X/Y so the model lands correctly under Slicer's default
    LPS model import (use this if the Add Data 'Coordinate system: RAS' option
    isn't available for .vtk in your Slicer build)."""
    sx, sy = (-1.0, -1.0) if lps else (1.0, 1.0)
    M, n = X.shape
    pts = [(sx * X[i, j], sy * Y[i, j], Z[i, j]) for i in range(M) for j in range(n)]
    quads = []
    for i in range(M - 1):
        for j in range(n - 1):
            a = i * n + j
            quads.append((a, a + 1, a + 1 + n, a + n))
    out = Path(section) / "vessel_tube.vtk"
    with open(out, "w") as f:
        f.write("# vtk DataFile Version 3.0\nvessel tube\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write(f"POLYGONS {len(quads)} {len(quads) * 5}\n")
        for q in quads:
            f.write(f"4 {q[0]} {q[1]} {q[2]} {q[3]}\n")
    print(f"saved {out}  ({'LPS' if lps else 'RAS'} coords)  — Add Data, load as a Model, overlay on the volume")
    if not lps:
        print("  (RAS coords matching the volume affine; pick RAS on import.")
        print("   if it loads mirrored/apart, re-export with --lps and import as default LPS)")


def main():
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    straighten = "--straighten" in sys.argv
    html = "--html" in sys.argv
    slicer = "--slicer" in sys.argv
    lps = "--lps" in sys.argv
    section = find_section(pos_args[0] if pos_args else None)

    he = None
    for c in [section / "handeye.json", _REPO_ROOT / "handeye.json"]:
        if c.exists():
            he = json.loads(c.read_text()); break
    if he is None:
        sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

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
        sys.exit("not enough points")

    # --- outlier rejection: distance from the smoothed ordered path ---
    sm = smooth_seq(pts, SEQ_W)
    d = np.linalg.norm(pts - sm, axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    thr = max(OUTLIER_MM, np.median(d) + 3 * 1.4826 * mad)
    keep = d < thr
    cpts, crad = pts[keep], radii[keep]
    print(f"{Path(section).name}: {len(pts)} points, kept {keep.sum()} "
          f"({100*keep.mean():.0f}%), dropped {len(pts)-keep.sum()} outliers")

    # --- principal axes of the centerline ---
    ctr = cpts.mean(0)
    _, _, vt = np.linalg.svd(cpts - ctr)
    axis = vt[0]

    # --- clean spread + length (rotation-invariant; measured BEFORE straighten) ---
    along = (cpts - ctr) @ axis
    perp = (cpts - ctr) - np.outer(along, axis)
    spread = float(np.sqrt((perp ** 2).sum(1).mean()))
    length = float(along.max() - along.min())
    print(f"  centerline length: {length:.1f} mm")
    print(f"  CLEAN perpendicular spread: {spread:.2f} mm   "
          f"(vessel radius ~{np.median(crad):.2f} mm)")

    # --- straighten for the figure (cosmetic): principal axis -> X ---
    if straighten:
        Rrot = vt.copy()
        if np.linalg.det(Rrot) < 0:          # keep it a proper rotation, no mirror
            Rrot[2] *= -1
        rot = lambda P: (P - ctr) @ Rrot.T
        pts = rot(pts)                        # keeps out_pts (pts[~keep]) consistent
        cpts = rot(cpts)
        print("  straightened: principal axis aligned to X (cosmetic, spread unchanged)")

    # --- clean centerline + tube mesh ---
    C = smooth_seq(cpts, SEQ_W)
    idx = np.linspace(0, len(C) - 1, min(TUBE_PTS, len(C))).astype(int)
    Cs = smooth_seq(C[idx], 2)
    Rs = np.maximum(0.4, smooth_seq(crad[idx].reshape(-1, 1), 2).ravel())
    X, Y, Z = tube_mesh(Cs, Rs)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="#16BFA6", alpha=0.9, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot(Cs[:, 0], Cs[:, 1], Cs[:, 2], color="#0B2B36", lw=1.2)
    out_pts = pts[~keep]
    if len(out_pts):
        ax.scatter(out_pts[:, 0], out_pts[:, 1], out_pts[:, 2],
                   c="#C2410C", s=10, alpha=0.5, label="rejected outliers")
        ax.legend(loc="upper right")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title(f"{Path(section).name} — vessel tube  "
                 f"(len {length:.0f} mm, spread {spread:.1f} mm)")
    try:
        ax.set_box_aspect(np.ptp(Cs, 0) + 1e-3)
    except Exception:
        pass
    ax.view_init(elev=22, azim=-60)
    o = Path(section) / "vessel_tube.png"
    fig.tight_layout(); fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"saved {o}")

    if html:
        save_html(section, X, Y, Z, Cs, out_pts, length, spread)

    if slicer:
        if straighten:
            print("  warning: --slicer with --straighten -> the model is rotated and")
            print("  will NOT align with the volume. Re-run without --straighten to overlay.")
        save_vtk(section, X, Y, Z, lps)

    # --- scroll wheel = zoom in/out (drag still rotates) ---
    def on_scroll(event):
        f = 0.85 if event.button == "up" else 1.0 / 0.85
        for get, set_ in ((ax.get_xlim3d, ax.set_xlim3d),
                          (ax.get_ylim3d, ax.set_ylim3d),
                          (ax.get_zlim3d, ax.set_zlim3d)):
            lo, hi = get(); mid = (lo + hi) / 2; half = (hi - lo) / 2 * f
            set_(mid - half, mid + half)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    if not html:
        plt.show()


if __name__ == "__main__":
    main()
