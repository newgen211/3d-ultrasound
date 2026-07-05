#!/usr/bin/env python3
"""
render_tubes_3d.py — presentation render: the 2 recovered tubes as solid 3D cylinders

Projects section_62 SAM detections, splits into the 2 parallel target tubes (across-
tube 2nd-PC split, same as view_planning_3d --split-lateral), fits a smooth outlier-
rejected centerline through each, and sweeps the vessel radius into a solid tube mesh.
Clean presentation figure (PNG, hi-DPI) + optional interactive HTML.

    python3 render_tubes_3d.py section_62 --handeye data/clarius_sessions/section_60/handeye.json
    python3 render_tubes_3d.py section_62 --html
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def find_section(arg):
    root = Path("data/clarius_sessions")
    for c in (Path(arg), root / arg):
        if c.exists():
            return c
    sys.exit(f"section not found: {arg}")


def smooth_seq(P, w=4):
    out = P.astype(float).copy()
    for i in range(len(P)):
        out[i] = np.median(P[max(0, i-w): i+w+1], axis=0)
    return out


def tube_mesh(C, R, n=20):
    M = len(C)
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True)+1e-9)
    ref = np.array([0,0,1.0])
    if abs(T[0]@ref) > 0.9: ref = np.array([0,1.0,0])
    Ns = [np.cross(T[0], ref)/(np.linalg.norm(np.cross(T[0], ref))+1e-9)]
    for i in range(1, M):
        nv = Ns[-1] - (Ns[-1]@T[i])*T[i]; nn = np.linalg.norm(nv)
        Ns.append(nv/nn if nn>1e-6 else np.cross(T[i], ref)/(np.linalg.norm(np.cross(T[i], ref))+1e-9))
    Ns = np.array(Ns); Bs = np.cross(T, Ns)
    th = np.linspace(0, 2*np.pi, n)
    X, Y, Z = (np.zeros((M, n)) for _ in range(3))
    for i in range(M):
        ring = C[i] + R[i]*(np.cos(th)[:,None]*Ns[i] + np.sin(th)[:,None]*Bs[i])
        X[i], Y[i], Z[i] = ring[:,0], ring[:,1], ring[:,2]
    return X, Y, Z


def fit_tube(P, rad):
    """order along principal axis, reject outliers, smooth -> centerline + radius + length."""
    ctr = P.mean(0); _, _, Vt = np.linalg.svd(P-ctr, full_matrices=False)
    axis = Vt[0]
    t = (P-ctr)@axis
    order = np.argsort(t)
    Po, ro = P[order], rad[order]
    sm = smooth_seq(Po, 5)
    d = np.linalg.norm(Po-sm, axis=1)
    mad = np.median(np.abs(d-np.median(d)))+1e-6
    keep = d < max(4.0, np.median(d)+3*1.4826*mad)
    Pk, rk = Po[keep], ro[keep]
    C = smooth_seq(Pk, 5)
    idx = np.linspace(0, len(C)-1, min(60, len(C))).astype(int)
    Cs = smooth_seq(C[idx], 2)
    Rs = np.maximum(0.6, smooth_seq(rk[idx].reshape(-1,1), 2).ravel())
    length = float(((Cs-Cs.mean(0))@axis).ptp())
    return Cs, Rs, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--handeye", default=None)
    ap.add_argument("--eps", type=float, default=3.5)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()

    section = find_section(args.section)
    det_path = section / "sam_detections.json"
    if not det_path.exists():
        alt = Path("data/clarius_sessions")/section.name.replace("_cam","")/"sam_detections.json"
        det_path = alt if alt.exists() else det_path
    detections = json.loads(det_path.read_text())["detections"]

    cands = [Path(args.handeye)] if args.handeye else [section/"handeye.json", Path("handeye.json")]
    he = next((json.loads(c.read_text()) for c in cands if c and c.exists()), None)
    if he is None: sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float); t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

    jsons = sorted(section.glob("raw_*.json"))
    f0 = json.loads(jsons[0].read_text())["frame"]
    axial_mm = f0["axial_um_per_sample"]/1000.0; lateral_mm = f0["lateral_um_per_line"]/1000.0
    W = f0["lines"]
    pose_cache = {}
    def pose_for(fi):
        if fi not in pose_cache:
            m = json.loads(jsons[fi].read_text()); pp = m.get("cobot_pose")
            pose_cache[fi] = pp["coords"] if pp and "coords" in pp and len(pp["coords"])>=6 else None
        return pose_cache[fi]

    pts, rad = [], []
    for d in detections:
        c = pose_for(d["frame_index"])
        if c is None: continue
        T = np.array(c[:3], float); R = Rotation.from_euler(conv, c[3:6], degrees=True).as_matrix()
        p_img = np.array([(d["cx"]-W/2.0)*lateral_mm, 0.0, d["cy"]*axial_mm])
        pts.append(T + R@(R_X@p_img + t_X)); rad.append(d.get("r_mm", 1.4))
    pts = np.array(pts); rad = np.array(rad)

    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(pts).labels_
    big = np.zeros(len(pts), bool)
    for l in [x for x in set(labels) if x!=-1]:
        m = labels==l
        if m.sum()>=20: big |= m
    P = pts[big]; Pc = P-P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    proj = (Pc@Vt[1]).reshape(-1,1)                  # 2nd PC = across-tube
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(proj)
    idx = np.where(big)[0]

    COLORS = ["#D6336C", "#1C7293"]
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    allC = []
    for c in (0, 1):
        sel = idx[km.labels_==c]
        Cs, Rs, length = fit_tube(pts[sel], rad[sel])
        X, Y, Z = tube_mesh(Cs, Rs)
        ax.plot_surface(X, Y, Z, color=COLORS[c], alpha=0.95, linewidth=0, antialiased=True, shade=True)
        allC.append(Cs)
        print(f"tube {c+1}: {len(sel)} pts, recovered span {length:.1f} mm, r_med {np.median(rad[sel]):.2f} mm")

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("section_62 — two recovered target vessels (3D)", fontsize=14, color="#21295C", weight="bold")
    allP = np.vstack(allC)
    try: ax.set_box_aspect(np.ptp(allP, 0)+1e-3)
    except Exception: pass
    ax.view_init(elev=18, azim=-72)
    ax.grid(True, alpha=0.3)
    out = section / "recovered_tubes_3d.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

    if args.html:
        try:
            import plotly.graph_objects as go
            figh = go.Figure()
            for c in (0,1):
                sel = idx[km.labels_==c]; Cs, Rs, _ = fit_tube(pts[sel], rad[sel])
                X, Y, Z = tube_mesh(Cs, Rs)
                figh.add_surface(x=X, y=Y, z=Z, showscale=False,
                                 colorscale=[[0, COLORS[c]],[1, COLORS[c]]], opacity=0.97)
            figh.update_layout(title="section_62 — recovered vessels", template="plotly_white",
                               scene=dict(aspectmode="data", xaxis_title="X (mm)",
                                          yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
                               width=1000, height=720)
            oh = section/"recovered_tubes_3d.html"; figh.write_html(str(oh), include_plotlyjs="cdn")
            print(f"saved {oh}")
        except ImportError:
            print("(--html needs plotly)")
    plt.show()


if __name__ == "__main__":
    main()