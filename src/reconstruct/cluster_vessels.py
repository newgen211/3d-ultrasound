#!/usr/bin/env python3
"""
cluster_vessels.py — project SAM per-frame detections into 3D and cluster into N tubes

Step 1 of the de-risked plan. Takes sam_detections.json (per-frame vessel
centroids) + each frame's camera pose (cobot_pose.coords in the _cam folder) +
handeye.json, projects every detection to a single 3D point in the base frame
using the SAME math as reconstruct_handeye.py:

    p_image = [(cx - W/2)*lateral_mm, 0, cy*axial_mm]
    p_base  = T + R_pose @ (R_X @ p_image + t_X)

then DBSCAN-clusters the labeled point cloud. Each cluster = one vessel. Reports
N tubes with length / mean radius / position, and drops clusters that span too few
frames (noise). No SAM, no GPU — pure geometry on existing data.

    python3 cluster_vessels.py section_59_cam
    python3 cluster_vessels.py section_59_cam --eps 4 --min-samples 6 --min-span 8

EXPECTATION: cluster separation is pose-limited (~2.85 mm camera spread). Vessels
closer than a few mm may MERGE into one cluster — so recovering, say, 6-9 clean
tubes of 12 is the expected honest result at current calibration, and it directly
motivates the bead phantom (tighter hand-eye -> more tubes resolve). A vessel that
dips out of plane and back may SPLIT into two clusters; --eps / --min-span tune that.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("need scikit-learn:  pip install scikit-learn")


def find_section(arg):
    root = Path("data/clarius_sessions")
    for cand in (Path(arg), root / arg):
        if cand.exists():
            return cand
    sys.exit(f"section not found: {arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--detections", default=None, help="sam_detections.json (default: in section)")
    ap.add_argument("--handeye", default=None)
    ap.add_argument("--eps", type=float, default=4.0, help="DBSCAN neighbourhood radius mm")
    ap.add_argument("--min-samples", type=int, default=5, help="DBSCAN core min points")
    ap.add_argument("--min-span", type=int, default=8,
                    help="drop clusters spanning fewer than this many frames (noise)")
    args = ap.parse_args()

    section = find_section(args.section)

    det_path = Path(args.detections) if args.detections else section / "sam_detections.json"
    if not det_path.exists():
        # detections may live in the base section, poses in the _cam copy
        alt = Path("data/clarius_sessions") / section.name.replace("_cam", "") / "sam_detections.json"
        det_path = alt if alt.exists() else det_path
    if not det_path.exists():
        sys.exit(f"no sam_detections.json (looked at {det_path})")
    detections = json.loads(det_path.read_text())["detections"]
    print(f"📂 {len(detections)} detections from {det_path}")

    # hand-eye
    cands = [Path(args.handeye)] if args.handeye else [section / "handeye.json", Path("handeye.json")]
    he = None
    for c in cands:
        if c and c.exists():
            he = json.loads(c.read_text()); break
    if he is None:
        sys.exit("no handeye.json")
    R_X = np.array(he["R_flange_to_image"], float)
    t_X = np.array(he["t_flange_to_image_mm"], float)
    conv = he["convention"]
    print(f"📐 hand-eye rms {he.get('rms_mm','?')} mm, convention '{conv}'")

    # frame scale + per-frame pose (read from the section sidecars by frame_index)
    jsons = sorted(section.glob("raw_*.json"))
    if not jsons:
        sys.exit(f"no raw_*.json in {section}")
    f0 = json.loads(jsons[0].read_text())["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    W = f0["lines"]

    # cache pose per frame_index
    pose_cache = {}
    def pose_for(fi):
        if fi in pose_cache:
            return pose_cache[fi]
        meta = json.loads(jsons[fi].read_text())
        p = meta.get("cobot_pose")
        coords = p["coords"] if p and "coords" in p and len(p["coords"]) >= 6 else None
        pose_cache[fi] = coords
        return coords

    # project each detection centroid -> one 3D base-frame point
    pts, meta_pts = [], []
    no_pose = 0
    for d in detections:
        fi = d["frame_index"]
        coords = pose_for(fi)
        if coords is None:
            no_pose += 1
            continue
        T = np.array(coords[:3], float)
        R = Rotation.from_euler(conv, coords[3:6], degrees=True).as_matrix()
        p_img = np.array([(d["cx"] - W / 2.0) * lateral_mm, 0.0, d["cy"] * axial_mm])
        p_base = T + R @ (R_X @ p_img + t_X)
        pts.append(p_base)
        meta_pts.append((fi, d.get("r_mm", np.nan)))
    if no_pose:
        print(f"   {no_pose} detections had no pose (skipped)")
    if len(pts) < args.min_samples:
        sys.exit("too few projected points to cluster")
    pts = np.array(pts)
    print(f"🌐 projected {len(pts)} detections into 3D")

    # cluster
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(pts)
    labels = db.labels_
    uniq = [l for l in set(labels) if l != -1]
    noise = int(np.sum(labels == -1))

    # build clusters, filter by frame span
    tubes = []
    for l in uniq:
        m = labels == l
        cl_pts = pts[m]
        frames = np.array([meta_pts[i][0] for i in range(len(meta_pts)) if m[i]])
        radii = np.array([meta_pts[i][1] for i in range(len(meta_pts)) if m[i]])
        span = int(frames.max() - frames.min() + 1)
        if span < args.min_span:
            continue
        # length along the cluster's principal axis
        c = cl_pts - cl_pts.mean(0)
        if len(c) >= 2:
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            proj = c @ Vt[0]
            length = float(proj.max() - proj.min())
        else:
            length = 0.0
        tubes.append(dict(
            n=int(m.sum()), span=span, length_mm=length,
            r_med=float(np.nanmedian(radii)),
            centroid=cl_pts.mean(0).round(1).tolist(),
        ))

    tubes.sort(key=lambda t: -t["length_mm"])
    print(f"\n=== {len(tubes)} vessel(s) resolved "
          f"({len(uniq)} raw clusters, {len(uniq)-len(tubes)} dropped <{args.min_span}-frame span, "
          f"{noise} noise pts) ===")
    for i, t in enumerate(tubes, 1):
        print(f"  tube {i}: {t['n']} pts, spans {t['span']} frames, "
              f"len {t['length_mm']:.1f} mm, r_med {t['r_med']:.2f} mm, "
              f"centroid {t['centroid']} mm")

    out = section / "vessel_clusters.json"
    out.write_text(json.dumps(dict(
        section=section.name, eps=args.eps, min_samples=args.min_samples,
        min_span=args.min_span, n_tubes=len(tubes), tubes=tubes), indent=2))
    print(f"\n💾 {out}")
    print("\nNote: cluster count is pose-limited (~2.85 mm). Merged tubes = vessels closer "
          "than ~eps; tighter hand-eye (bead phantom) resolves more. Tune --eps / --min-span.")


if __name__ == "__main__":
    main()