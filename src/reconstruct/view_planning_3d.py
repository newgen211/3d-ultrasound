#!/usr/bin/env python3
"""
view_planning_3d.py — planning view, depth-weighted (inverse-variance) clustering

Honest analysis of a ROBUSTNESS sweep — NO spline, nothing invented in gaps.

Two ideas baked in, both from physics not aesthetics:

1. ANISOTROPIC, DATA-DRIVEN distance. Axial (depth) resolution is finer than
   lateral, so a point's depth is more trustworthy than its lateral position.
   Rather than assert the ~2.5x resolution ratio, we MEASURE the per-axis
   detection scatter and weight by inverse variance (Mahalanobis-style). The 2.5x
   floor then emerges from the measurement instead of being hardcoded — and the
   computed factor is printed so you can sanity-check it lands near ~2.5x.
   CRUCIAL: variance is measured PERPENDICULAR to the vessel (cross-sectional
   measurement noise to down-weight), NOT along it (real structure to keep). Done
   by detrending each tentative vessel's image-space track over frame index and
   taking the residual scatter per axis. Weighting is applied in IMAGE SPACE
   before projection (where depth/lateral are well-defined; probe rotation
   scrambles world axes).

2. KEEP the unassigned points. DBSCAN "noise" is shown in a distinct color, NOT
   discarded — because points between two vessels may be real CONNECTING STRUCTURE
   (anastomosis), which is clinically critical for needle planning. You judge by eye.

    python3 view_planning_3d.py section_59_cam --eps 2.0 --n-targets 2
    python3 view_planning_3d.py section_59_cam --aniso 2.5   # force factor, skip auto-measure
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
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
    ap.add_argument("--eps", type=float, default=2.0,
                    help="DBSCAN radius, in DEPTH-equivalent mm (lateral is down-weighted)")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--n-targets", type=int, default=2)
    ap.add_argument("--aniso", type=float, default=None,
                    help="force lateral down-weight factor (skip auto inverse-variance measure)")
    ap.add_argument("--present", action="store_true",
                    help="presentation styling: distinct per-target colors, clean export")
    ap.add_argument("--merge-tail", type=int, default=0,
                    help="absorb clusters smaller than this many pts into the nearest TARGET "
                         "(rejoins a small gap-split tail; 0 = off)")
    ap.add_argument("--split-lateral", action="store_true",
                    help="two parallel side-by-side tubes: pool the big clusters and split "
                         "into 2 targets across the SEPARATION axis (left vs right tube), "
                         "instead of letting DBSCAN carve them the wrong way")
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
    res_ratio = lateral_mm / axial_mm
    print(f"resolution ratio (lateral/axial px size) = {res_ratio:.2f}x  (physics floor for anisotropy)")

    pose_cache = {}
    def pose_for(fi):
        if fi not in pose_cache:
            m = json.loads(jsons[fi].read_text())
            p = m.get("cobot_pose")
            pose_cache[fi] = p["coords"] if p and "coords" in p and len(p["coords"]) >= 6 else None
        return pose_cache[fi]

    # image-space (lateral_mm, depth_mm) + frame, plus pose, per detection
    rows = []
    for d in detections:
        c = pose_for(d["frame_index"])
        if c is None:
            continue
        lat = (d["cx"] - W / 2.0) * lateral_mm
        dep = d["cy"] * axial_mm
        rows.append((d["frame_index"], lat, dep, np.array(c[:3], float),
                     Rotation.from_euler(conv, c[3:6], degrees=True).as_matrix()))
    fr = np.array([r[0] for r in rows])
    lat = np.array([r[1] for r in rows])
    dep = np.array([r[2] for r in rows])

    def project(lateral_vals, depth_vals):
        out = np.empty((len(rows), 3))
        for i, (_, _, _, T, R) in enumerate(rows):
            p_img = np.array([lateral_vals[i], 0.0, depth_vals[i]])
            out[i] = T + R @ (R_X @ p_img + t_X)
        return out

    # ---- measure anisotropy (inverse variance, perpendicular to vessel) ----
    if args.aniso is not None:
        factor = args.aniso
        print(f"anisotropy factor (forced): lateral down-weighted {factor:.2f}x")
    else:
        pts0 = project(lat, dep)                       # isotropic, real mm
        lbl0 = DBSCAN(eps=2.0, min_samples=args.min_samples).fit(pts0).labels_
        res_lat, res_dep = [], []
        used = 0
        for l in [x for x in set(lbl0) if x != -1]:
            m = lbl0 == l
            if m.sum() < 12:
                continue
            f = fr[m]; order = np.argsort(f)
            f = f[order]; L = lat[m][order]; D = dep[m][order]
            deg = min(2, len(f) - 1)
            # detrend over frame index => remove slow ALONG-vessel drift (real structure),
            # leaving residual = CROSS-SECTIONAL measurement noise
            L_res = L - np.polyval(np.polyfit(f, L, deg), f)
            D_res = D - np.polyval(np.polyfit(f, D, deg), f)
            res_lat.append(L_res); res_dep.append(D_res); used += 1
        if used >= 2:
            s_lat = np.std(np.concatenate(res_lat))
            s_dep = np.std(np.concatenate(res_dep))
            factor = s_lat / s_dep if s_dep > 1e-6 else res_ratio
            print(f"measured cross-sectional noise: lateral {s_lat:.3f} mm vs depth {s_dep:.3f} mm "
                  f"(from {used} tentative vessels)")
            print(f"anisotropy factor (measured): lateral down-weighted {factor:.2f}x  "
                  f"[physics floor ~{res_ratio:.2f}x — {'sane' if factor > res_ratio*0.6 else 'BELOW FLOOR, check'}]")
        else:
            factor = res_ratio
            print(f"too few vessels to measure; falling back to resolution ratio {factor:.2f}x")

    # ---- anisotropic clustering: down-weight lateral in image space, then project ----
    lat_w = lat / factor                              # compress lateral => counts less
    pts_w = project(lat_w, dep)                        # distorted space, for cluster LABELS
    pts_real = project(lat, dep)                       # true mm, for positions/display
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(pts_w).labels_

    clusters = []
    for l in [x for x in set(labels) if x != -1]:
        m = labels == l
        clusters.append((m.sum(), m))
    clusters.sort(key=lambda t: -t[0])
    noise = labels == -1

    # two parallel tubes that DBSCAN carves wrong: pool the substantial clusters and
    # split across the axis of maximum BETWEEN-tube separation (KMeans on that axis).
    if args.split_lateral:
        big = np.zeros(len(pts_real), bool)
        for n, m in clusters:
            if n >= 20:
                big |= m
        P = pts_real[big]
        Pc = P - P.mean(0)
        # Two tubes run PARALLEL. The tube-length direction = 1st principal axis (shared
        # by both tubes). The SEPARATION between tubes is the perpendicular direction with
        # the most variance => the 2nd principal axis. Split along THAT, not along length.
        _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
        sep_axis = Vt[1]                      # 2nd PC = across-tube (length is Vt[0])
        proj = Pc @ sep_axis
        # 2-means on the 1-D separation coordinate => clean left/right cut
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(proj.reshape(-1, 1))
        idx = np.where(big)[0]
        new_clusters = []
        for c in (0, 1):
            mm = np.zeros(len(pts_real), bool)
            mm[idx[km.labels_ == c]] = True
            new_clusters.append((int(mm.sum()), mm))
        # keep any remaining big-but-excluded? (none — all >=20 pooled). small clusters -> obstacles
        small = [(n, m) for n, m in clusters if n < 20]
        new_clusters.sort(key=lambda t: -t[0])
        clusters = new_clusters + small
        args.n_targets = 2
        print("   (split-lateral: divided pooled tubes into 2 by separation axis)")

    # optionally absorb tiny clusters into the nearest TARGET (gap-split tail rejoin)
    if args.merge_tail > 0 and len(clusters) > args.n_targets:
        tgt = clusters[:args.n_targets]
        tgt_ctr = [pts_real[m].mean(0) for _, m in tgt]
        kept = list(tgt)
        for n, m in clusters[args.n_targets:]:
            if n < args.merge_tail:
                c = pts_real[m].mean(0)
                j = int(np.argmin([np.linalg.norm(c - tc) for tc in tgt_ctr]))
                # OR the masks together
                merged = kept[j][1] | m
                kept[j] = (int(merged.sum()), merged)
            else:
                kept.append((n, m))
        kept.sort(key=lambda t: -t[0])
        clusters = kept
        print(f"   (merge-tail: absorbed small clusters into nearest target)")

    print(f"\n{len(clusters)} clusters, {noise.sum()} unassigned points "
          f"(shown as possible connecting structure, NOT discarded)")
    for k, (n, m) in enumerate(clusters, 1):
        tag = "TARGET" if k <= args.n_targets else "obstacle"
        P = pts_real[m]
        c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False)
        length = float((c @ Vt[0]).ptp())
        print(f"  cluster {k} [{tag}]: {n} pts, ref-axis len {length:.1f} mm")

    # ---- plot: raw points only, faint REFERENCE axis (labeled, not reconstruction) ----
    fig = plt.figure(figsize=(13, 6))
    for sp, (elev, azim, ttl) in zip((121, 122), [(20, -60, "perspective"), (90, -90, "top-down X–Y")]):
        ax = fig.add_subplot(sp, projection="3d")
        # unassigned first (behind), distinct color
        if noise.any():
            N = pts_real[noise]
            ax.scatter(N[:,0], N[:,1], N[:,2], s=9, color="tab:blue", alpha=0.5,
                       label="unassigned (possible connecting structure)")
        TGT_COLORS = ["#D6336C", "#1C7293", "#BA7517", "#1D9E75"]  # distinct per target
        for k, (n, m) in enumerate(clusters):
            is_t = k < args.n_targets
            P = pts_real[m]
            if is_t and args.present:
                col = TGT_COLORS[k % len(TGT_COLORS)]
                lbl = f"target {k+1}  ({len(P)} pts, {float(((P-P.mean(0))@np.linalg.svd(P-P.mean(0),full_matrices=False)[2][0]).ptp()):.0f} mm)"
            elif is_t:
                col = "crimson"; lbl = ("target" if k == 0 else None)
            else:
                col = "0.6"; lbl = ("obstacle" if k == args.n_targets else None)
            ax.scatter(P[:,0], P[:,1], P[:,2], s=(18 if is_t else 7), color=col,
                       alpha=(0.9 if is_t else 0.3), label=lbl)
            # faint REFERENCE axis only (dashed) — explicitly NOT a reconstruction
            c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False)
            t = c @ Vt[0]
            a = P.mean(0) + t.min()*Vt[0]; b = P.mean(0) + t.max()*Vt[0]
            ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],
                    color=col, lw=(1.5 if is_t else 1.0), ls="--", alpha=0.6)
        ax.view_init(elev=elev, azim=azim); ax.set_title(ttl)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    h, lg = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lg, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(f"{section.name} — depth-weighted (×{factor:.1f}) | dashed = reference axis, NOT reconstruction "
                 f"| robustness sweep", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = section / "vessel_planning_3d.png"
    fig.savefig(out, dpi=(200 if args.present else 120), bbox_inches="tight")
    print(f"\nsaved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()