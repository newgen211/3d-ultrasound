#!/usr/bin/env python3
"""
view_clusters_3d.py — render the STRONG vessel clusters in 3D

Re-projects sam_detections.json into 3D (same hand-eye math as cluster_vessels.py /
reconstruct_handeye.py), re-runs DBSCAN, keeps only STRONG tubes (enough points AND
enough frame span — strips the bumpy-sweep fragments), and shows them as colored 3D
point clouds with a fitted centerline each. Interactive (matplotlib) + saves a PNG.

    python3 view_clusters_3d.py section_59_cam --eps 1.8 --min-span 50 --min-pts 20

--min-pts is the strength filter: a "strong" tube needs at least this many member
detections (not just frame span). Bumpy/fast cobot sweep => some real vessels are
sparse; raising --min-pts keeps only the confidently-supported ones.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def find_section(arg):
    root = Path("data/clarius_sessions")
    for c in (Path(arg), root / arg):
        if c.exists():
            return c
    sys.exit(f"section not found: {arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--handeye", default=None)
    ap.add_argument("--eps", type=float, default=1.8)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-span", type=int, default=50)
    ap.add_argument("--min-pts", type=int, default=20, help="strength filter: min member detections")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    section = find_section(args.section)
    det_path = section / "sam_detections.json"
    if not det_path.exists():
        alt = Path("data/clarius_sessions") / section.name.replace("_cam", "") / "sam_detections.json"
        det_path = alt if alt.exists() else det_path
    detections = json.loads(det_path.read_text())["detections"]

    cands = [Path(args.handeye)] if args.handeye else [section / "handeye.json", Path("handeye.json")]
    he = next((json.loads(c.read_text()) for c in cands if c and c.exists()), None)
    if he is None:
        sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]

    jsons = sorted(section.glob("raw_*.json"))
    f0 = json.loads(jsons[0].read_text())["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    W = f0["lines"]

    pose_cache = {}
    def pose_for(fi):
        if fi not in pose_cache:
            m = json.loads(jsons[fi].read_text())
            p = m.get("cobot_pose")
            pose_cache[fi] = p["coords"] if p and "coords" in p and len(p["coords"]) >= 6 else None
        return pose_cache[fi]

    pts, fr = [], []
    for d in detections:
        c = pose_for(d["frame_index"])
        if c is None:
            continue
        T = np.array(c[:3], float)
        R = Rotation.from_euler(conv, c[3:6], degrees=True).as_matrix()
        p_img = np.array([(d["cx"] - W / 2.0) * lateral_mm, 0.0, d["cy"] * axial_mm])
        pts.append(T + R @ (R_X @ p_img + t_X))
        fr.append(d["frame_index"])
    pts = np.array(pts); fr = np.array(fr)

    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(pts).labels_

    strong = []
    for l in [x for x in set(labels) if x != -1]:
        m = labels == l
        span = int(fr[m].max() - fr[m].min() + 1)
        if m.sum() >= args.min_pts and span >= args.min_span:
            strong.append((l, m))
    strong.sort(key=lambda t: -t[1].sum())
    print(f"kept {len(strong)} strong tubes (min_pts {args.min_pts}, min_span {args.min_span})")

    fig = plt.figure(figsize=(13, 6))
    axA = fig.add_subplot(121, projection="3d")
    axB = fig.add_subplot(122, projection="3d")
    cmap = plt.cm.tab10
    for i, (l, m) in enumerate(strong):
        P = pts[m]; col = cmap(i % 10)
        for ax in (axA, axB):
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, color=col, alpha=0.5)
            # centerline = PCA 1st axis, drawn min->max
            c = P - P.mean(0)
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            t = c @ Vt[0]
            line = P.mean(0) + np.outer(np.array([t.min(), t.max()]), Vt[0])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=col, lw=2.5,
                    label=f"tube {i+1} ({m.sum()} pts)")
    axA.view_init(elev=20, azim=-60); axA.set_title("perspective")
    axB.view_init(elev=90, azim=-90); axB.set_title("top-down (X–Y)")
    for ax in (axA, axB):
        ax.set_xlabel("X mm"); ax.set_ylabel("Y mm"); ax.set_zlabel("Z mm")
    axA.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{section.name} — {len(strong)} strong vessels "
                 f"(eps {args.eps}, min_pts {args.min_pts}, hand-eye {he.get('rms_mm',0):.2f} mm)")
    fig.tight_layout()
    out = section / "vessel_clusters_3d.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()