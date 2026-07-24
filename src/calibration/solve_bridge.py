#!/usr/bin/env python3
"""
solve_bridge.py — camera<->cobot bridge: solve flange->marker (AX=XB).

Feeds on a limp-arm drag recorded by BOTH track_probe (Mac) and pose_logger
(Pi). Solves the rigid flange->marker transform X and the fixed camera->base
transform, so cobot poses can predict marker poses (dropout fill / fusion)
and vice versa.

Self-defending: measures the Pi<->Mac clock offset by speed-profile
cross-correlation, uses only rotation-STABLE camera samples (flip guard,
vet_frames logic), and requires orientation diversity before trusting itself.

Usage:
    python solve_bridge.py data/pose_logs/probe_pose_log.jsonl data/pose_logs/pi_pose_log.jsonl
Output: bridge.json + accuracy report (expect few-mm class — arm-limited).
"""
import json, os, sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d

def load(path, want_id=None):
    T, C, A = [], [], []
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if want_id is not None and r.get("id") != want_id: continue
        if r.get("coords"):
            T.append(int(r["t_ns"])); C.append(r["coords"])
            a = r.get("angles")
            A.append(a if a and len(a) == 6 else [np.nan]*6)
    T = np.array(T, np.int64); C = np.array(C, float); A = np.array(A, float)
    o = np.argsort(T)
    return T[o], C[o], A[o]

def pose_mat(c, conv="xyz"):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(conv, c[3:6], degrees=True).as_matrix()
    T[:3, 3] = c[:3]
    return T

def main():
    cam_path, pi_path = sys.argv[1], sys.argv[2]
    Tc, Cc, _ = load(cam_path, want_id=0)
    Tp, Cp, Ap = load(pi_path)
    print(f"cam id0: {len(Tc)}   pi: {len(Tp)}")

    # JOINT-FK mode: if the pi log carries varying joint angles and a URDF is
    # findable, build flange poses by forward kinematics — rotation MATRICES
    # from joints, no euler decomposition, gimbal lock has nothing to break.
    fk_rots = None
    if np.isfinite(Ap).all() and np.nanstd(Ap) > 1.0:
        for u in ("mycobot_320_pi.urdf",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "mycobot_320_pi.urdf"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "mycobot_320_pi.urdf")):
            if os.path.exists(u):
                import warnings; warnings.filterwarnings("ignore")
                from ikpy.chain import Chain
                chain = Chain.from_urdf_file(u, base_elements=['base'],
                                             active_links_mask=[False]+[True]*6)
                NQ = len(chain.links)
                fk_rots = []
                for a in Ap:
                    q = np.zeros(NQ); q[1:7] = np.radians(a)
                    fk_rots.append(chain.forward_kinematics(q)[:3, :3])
                print(f"flange rotations: URDF joint-FK ({u}) — euler bypassed")
                break
    if fk_rots is None:
        print("flange rotations: firmware euler (no angles/URDF) — gimbal-prone")

    # window cam to pi's span +- 60 s (survives appended logs)
    m = (Tc >= Tp[0] - np.int64(60e9)) & (Tc <= Tp[-1] + np.int64(60e9))
    if m.sum() < 200: sys.exit("cam log barely overlaps pi log — wrong files?")
    Tc, Cc = Tc[m], Cc[m]

    # ---- clock offset via speed-profile cross-correlation ----
    def speed(T, P):
        t = (T - T[0]) / 1e9
        w = max(3, int(0.4 / max(np.median(np.diff(t)), 1e-3)))
        Ps = uniform_filter1d(P[:, :3], size=w, axis=0, mode="nearest")
        v = np.linalg.norm(np.diff(Ps, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
        return t[1:], v
    tc, vc = speed(Tc, Cc); tp, vp = speed(Tp, Cp)
    dt = 0.05
    gc = np.interp(np.arange(0, tc[-1], dt), tc, vc)
    gp = np.interp(np.arange(0, tp[-1], dt), tp, vp)
    gc = (gc - gc.mean()) / (gc.std() + 1e-9)
    gp = (gp - gp.mean()) / (gp.std() + 1e-9)
    corr = np.correlate(gc, gp, mode="full")
    lag = (np.argmax(corr) - (len(gp) - 1)) * dt
    peak = corr.max() / min(len(gc), len(gp))
    off = int(Tc[0]) - int(Tp[0]) + int(lag * 1e9)     # Mac = Pi + off
    print(f"clock: lag {lag:+.2f}s  peak {peak:.2f}  Pi->Mac offset {off/1e9:+.3f}s")
    if peak < 0.15: sys.exit("motion correlation too weak — not the same session?")
    if peak < 0.3: print("!! weak correlation — verify the offset looks sane before trusting")

    # ---- stable-sample selection (flip guard on the cam stream) ----
    Rc = Rotation.from_euler("xyz", Cc[:, 3:6], degrees=True)
    stable = np.zeros(len(Tc), bool)
    W = np.int64(0.25e9)
    for k in range(len(Tc)):
        i0, i1 = np.searchsorted(Tc, Tc[k]-W), np.searchsorted(Tc, Tc[k]+W)
        if i1 - i0 < 4: continue
        rw = Rc[i0:i1]
        dmax = max(np.degrees((rw[0].inv()*rw[j]).magnitude()) for j in range(i1-i0))
        steps = max(np.degrees((rw[j].inv()*rw[j+1]).magnitude()) for j in range(i1-i0-1))
        stable[k] = steps < 4.0 and dmax < 6.0
    print(f"stable cam samples: {stable.sum()}/{len(Tc)}")

    # per-sample speed for both streams (temporal mismatch is harmless when slow)
    def speeds_at(T, C):
        t = (T - T[0]) / 1e9
        w = max(3, int(0.3 / max(float(np.median(np.diff(t))), 1e-3)))
        Ps = uniform_filter1d(C[:, :3], size=w, axis=0, mode="nearest")
        v = np.zeros(len(T))
        v[1:] = np.linalg.norm(np.diff(Ps, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
        v[0] = v[1]
        return v
    vC, vP = speeds_at(Tc, Cc), speeds_at(Tp, Cp)

    # ---- pair + subsample for diversity (slow samples only) ----
    pairs = []
    for k in np.where(stable & (vC < 12.0))[0]:
        t_mac = Tc[k]; t_pi = t_mac - off
        j = np.searchsorted(Tp, t_pi)
        for jj in (j-1, j):
            if 0 <= jj < len(Tp) and abs(int(Tp[jj]) - t_pi) < 50e6 and vP[jj] < 12.0:
                pairs.append((k, jj)); break
    print(f"time-paired samples: {len(pairs)}")
    if len(pairs) < 10: sys.exit("too few slow-moment pairs — the drag needs pauses")
    # greedy diversity subsample: keep pose if rotation differs >=8 deg from all kept
    kept = []
    for k, j in pairs:
        r = Rc[k]
        if all(np.degrees((r.inv()*Rc[kk]).magnitude()) >= 8.0 for kk, _ in kept):
            kept.append((k, j))
    print(f"diverse pairs kept: {len(kept)}")
    if len(kept) < 8: sys.exit("not enough orientation diversity — use the WILD drag log")
    rots = [Rc[k] for k, _ in kept]
    div = max(np.degrees((rots[0].inv()*r).magnitude()) for r in rots)
    print(f"orientation spread among kept: {div:.0f} deg")

    # ---- AX = XB via cv2.calibrateHandEye ----
    Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
    for k, j in kept:
        Tf = pose_mat(Cp[j])                    # flange pose in base
        Tm = np.linalg.inv(pose_mat(Cc[k]))     # camera pose in marker frame
        Rg2b.append(Tf[:3, :3]); tg2b.append(Tf[:3, 3])   # (convention verified
        Rt2c.append(Tm[:3, :3]); tt2c.append(Tm[:3, 3])   #  against synthetic truth)
    Rx, tx = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c,
                                  method=cv2.CALIB_HAND_EYE_TSAI)
    X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel()   # flange->marker
    print(f"\nflange->marker translation: {np.round(X[:3,3],1)} mm  "
          f"(|t| = {np.linalg.norm(X[:3,3]):.1f} mm)")

    # ---- recover camera->base and validate ----
    Ts = []
    for k, j in kept:
        Tcb = pose_mat(Cc[k]) @ np.linalg.inv(X) @ np.linalg.inv(pose_mat(Cp[j]))
        Ts.append(Tcb)
    t_cb = np.mean([T[:3, 3] for T in Ts], axis=0)
    R_cb = Rotation.from_matrix([T[:3, :3] for T in Ts]).mean().as_matrix()
    Tcb = np.eye(4); Tcb[:3, :3] = R_cb; Tcb[:3, 3] = t_cb

    def flange_mat(j, conv="xyz"):
        if fk_rots is not None:
            T = np.eye(4); T[:3, :3] = fk_rots[j]; T[:3, 3] = Cp[j][:3]
            return T          # position from firmware (mm, validated), rotation from FK
        return pose_mat(Cp[j], conv)

    def solve_from(kept_pairs, conv="xyz"):
        Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
        for k, j in kept_pairs:
            Tf = flange_mat(j, conv)
            Tm = np.linalg.inv(pose_mat(Cc[k]))
            Rg2b.append(Tf[:3, :3]); tg2b.append(Tf[:3, 3])
            Rt2c.append(Tm[:3, :3]); tt2c.append(Tm[:3, 3])
        Rx, tx = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c,
                                      method=cv2.CALIB_HAND_EYE_TSAI)
        Xs = np.eye(4); Xs[:3, :3] = Rx; Xs[:3, 3] = tx.ravel()
        Ts_ = [pose_mat(Cc[k]) @ np.linalg.inv(Xs) @ np.linalg.inv(flange_mat(j, conv))
               for k, j in kept_pairs]
        tcb = np.mean([t[:3, 3] for t in Ts_], axis=0)
        rcb = Rotation.from_matrix([t[:3, :3] for t in Ts_]).mean().as_matrix()
        Tcbs = np.eye(4); Tcbs[:3, :3] = rcb; Tcbs[:3, 3] = tcb
        e = np.array([np.linalg.norm((Tcbs @ flange_mat(j, conv) @ Xs)[:3, 3] - Cc[k, :3])
                      for k, j in kept_pairs])
        return Xs, Tcbs, e

    def pair_with(off_ns):
        prs = []
        for k in np.where(stable & (vC < 12.0))[0]:
            t_pi = Tc[k] - off_ns
            j = np.searchsorted(Tp, t_pi)
            for jj in (j-1, j):
                if 0 <= jj < len(Tp) and abs(int(Tp[jj]) - t_pi) < 50e6 and vP[jj] < 12.0:
                    prs.append((k, jj)); break
        kp = []
        for k, j in prs:
            r = Rc[k]
            if all(np.degrees((r.inv()*Rc[kk]).magnitude()) >= 6.0 for kk, _ in kp):
                kp.append((k, j))
        return kp

    # refine the clock offset by geometry (coarse correlation is only ~±100 ms)
    best = (np.inf, off, kept)
    for d_ms in range(-250, 251, 25):
        kp = pair_with(off + int(d_ms * 1e6))
        if len(kp) < 8: continue
        _, _, e = solve_from(kp)
        if np.median(e) < best[0]:
            best = (np.median(e), off + int(d_ms * 1e6), kp)
    off, kept = best[1], best[2]
    print(f"offset refined by geometry: {off/1e9:+.3f}s  ({len(kept)} pairs)")

    # gimbal report: cobot euler goes unstable near ry = +-90 (probe-down!)
    ry = np.array([Cp[j][4] for _, j in kept])
    n_gimbal = int((np.abs(np.abs(ry) - 90) < 15).sum())
    if n_gimbal:
        print(f"!! {n_gimbal}/{len(kept)} kept poses within 15 deg of cobot "
              f"gimbal lock (ry~+-90, probe-down) — their euler may be unreliable")

    # ---- mutual-consistency filter: reject STABLY-FLIPPED camera holds ----
    # Relative rotation ANGLE and relative translation MAGNITUDE are invariant
    # to X and to either base frame: flange and marker must agree on both for
    # every pair of poses. A stably-flipped hold (PnP wrong branch, steady for
    # the whole hold — invisible to continuity vetting) breaks the equality.
    n = len(kept)
    RA = [flange_mat(j)[:3, :3] for _, j in kept]
    RB = [pose_mat(Cc[k])[:3, :3] for k, _ in kept]
    ok = np.zeros((n, n), bool)
    for a in range(n):
        for b_ in range(a + 1, n):
            ra = np.degrees(np.arccos(np.clip((np.trace(RA[a].T @ RA[b_]) - 1) / 2, -1, 1)))
            rb = np.degrees(np.arccos(np.clip((np.trace(RB[a].T @ RB[b_]) - 1) / 2, -1, 1)))
            # only the relative-rotation ANGLE is X-invariant (translation
            # differs by lever-arm motion whenever rotation changes)
            ok[a, b_] = ok[b_, a] = abs(ra - rb) < 6.0
    # greedy max-consistent subset
    order = np.argsort(-ok.sum(1))
    subset = []
    for i in order:
        if all(ok[i, j] for j in subset):
            subset.append(int(i))
    print(f"mutual-consistency filter: {len(subset)}/{n} poses agree "
          f"(rejected {n - len(subset)} — stable flips / bad samples)")
    if len(subset) >= 6:
        kept = [kept[i] for i in subset]
    else:
        print("!! consistent subset too small — capture is broadly corrupted")

    # the cobot's euler convention is folklore — search it, like calibrate_handeye does
    convs = ["xyz"] if fk_rots is not None else \
            ["xyz","XYZ","zyx","ZYX","xzy","XZY","yxz","YXZ","yzx","YZX","zxy","ZXY"]
    results = []
    print("\n   cobot-euler   median err (mm)")
    for cv in convs:
        try:
            _, _, e = solve_from(kept, cv)
            results.append((float(np.median(e)), cv))
        except Exception:
            continue
    results.sort()
    for m, cv in results[:5]:
        print(f"   {cv:10s}   {m:8.2f}")
    best_conv = results[0][1]
    if best_conv != "xyz":
        print(f"!! best convention is '{best_conv}', not 'xyz' — cobot euler "
              f"assumption was wrong everywhere it was assumed")
    X, Tcb, errs = solve_from(kept, best_conv)

    # ---- residual trim: drop pairs > 2.5x median, one pass ----
    med = float(np.median(errs))
    good = [p for p, e in zip(kept, errs) if e <= 2.5 * med]
    if len(good) >= 6 and len(good) < len(kept):
        print(f"residual trim: dropping {len(kept)-len(good)} pair(s) > "
              f"{2.5*med:.1f} mm, re-solving")
        kept = good
        X, Tcb, errs = solve_from(kept, best_conv)

    # ---- observability audit (rotation-axis diversity) ----
    # AX=XB pins X's translation only along directions the relative-rotation
    # axes span. Weak third axis => that component of X is half-guessed.
    axes = []
    for a in range(len(kept)):
        for b_ in range(a + 1, len(kept)):
            Rrel = flange_mat(kept[a][1], best_conv)[:3, :3].T @ \
                   flange_mat(kept[b_][1], best_conv)[:3, :3]
            rv = Rotation.from_matrix(Rrel).as_rotvec()
            ang = np.linalg.norm(rv)
            if np.degrees(ang) > 10:
                axes.append(rv / ang)
    if axes:
        Sv = np.linalg.svd(np.array(axes), compute_uv=False)
        Sv = Sv / Sv[0]
        U = np.linalg.svd(np.array(axes).T @ np.array(axes))[0]
        print(f"observability: rotation-axis singular values "
              f"{np.round(Sv, 2)} (want all >~0.3)")
        if Sv[2] < 0.3:
            weak = U[:, 2]
            print(f"!! X translation along base-frame direction "
                  f"{np.round(weak, 2)} is UNDER-CONSTRAINED — add held poses "
                  f"whose rotation axes point that way (fanning, not rolling)")

    # ---- joint-space audit: what did the servos actually explore? ----
    if fk_rots is not None:
        Ak = np.array([Ap[j] for _, j in kept])
        rng = Ak.max(0) - Ak.min(0)
        print("joint ranges among kept pairs (deg): "
              + "  ".join(f"J{i+1}:{rng[i]:.0f}" for i in range(6)))
        lowJ = [f"J{i+1}" for i in range(3, 6) if rng[i] < 15]
        if lowJ:
            print(f"!! wrist joints {','.join(lowJ)} barely explored — vary "
                  f"them in any top-up capture")
    print(f"flange->marker translation: {np.round(X[:3,3],1)} mm  "
          f"(|t| = {np.linalg.norm(X[:3,3]):.1f} mm)")
    print(f"bridge accuracy (predict marker from cobot): "
          f"median {np.median(errs):.2f} mm  RMS {np.sqrt((errs**2).mean()):.2f} mm  "
          f"worst {errs.max():.2f} mm")
    print("(few-mm class expected — inherits the arm's error; jobs: dropout "
          "fill, pose benchmark, servoing map. NOT a replacement for the "
          "marker hand-eye.)")

    json.dump({"X_flange_to_marker": X.tolist(),
               "T_cam_to_base": Tcb.tolist(),
               "pairs_used": len(kept), "orientation_spread_deg": float(div),
               "median_err_mm": float(np.median(errs)),
               "rms_err_mm": float(np.sqrt((errs**2).mean()))},
              open("bridge.json", "w"), indent=2)
    print("wrote bridge.json")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: solve_bridge.py cam_log.jsonl pi_log.jsonl")
    main()