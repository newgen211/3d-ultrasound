#!/usr/bin/env python3
"""
pose_benchmark.py — compare two pose sources directly from their logs

Camera log  : track_probe.py probe_pose_log.jsonl  (filtered to id 0)
Cobot log   : pose_logger.py  cobot_log.jsonl

Both lines carry {"t_ns": ..., "coords": [x,y,z, rx,ry,rz]} (mm, degrees).
We window the camera log to the cobot sweep's time range (the camera log is an
append pile), then report three things that ONE sweep can honestly give:

  1. Static jitter   — noise floor over the longest still segment (mm, deg).
  2. Smoothness      — frame-to-frame step + how quantized the source is
                       (the cobot stair-steps; the camera is continuous).
  3. Agreement       — after a rigid Kabsch alignment of the camera path onto
                       the cobot path (different frames), the residual RMS:
                       how much the two disagree about the SAME motion.

Known-move accuracy and repeatability are NOT here — they need a dedicated
capture (command known moves, repeat them). This is the single-sweep subset.

    python3 pose_benchmark.py cobot_log.jsonl probe_pose_log.jsonl
"""

import sys, json
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

CAM_ID = 0


def load(path, only_id=None):
    t, c = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if only_id is not None and r.get("id") != only_id:
            continue
        co = r.get("coords")
        if "t_ns" in r and co and len(co) >= 6:
            t.append(int(r["t_ns"])); c.append([float(x) for x in co[:6]])
    t = np.array(t, np.int64); c = np.array(c, float)
    o = np.argsort(t)
    return t[o], c[o]


def speed_mm_s(t, pos):
    dt = np.diff(t) / 1e9
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return d / np.maximum(dt, 1e-6)


def longest_still(t, pos, thr):
    """Return (i0, i1) of the longest contiguous run with speed < thr mm/s."""
    if len(t) < 4:
        return None
    sp = speed_mm_s(t, pos)            # length N-1
    still = sp < thr
    best = (0, 0); run = None
    for i, s in enumerate(still):
        if s:
            run = (run[0], i + 1) if run else (i, i + 1)
            if run[1] - run[0] > best[1] - best[0]:
                best = run
        else:
            run = None
    return best if best[1] > best[0] else None


def find_still_window(t, pos, min_dur=0.4):
    for thr in (1.0, 1.5, 2.5, 4.0, 6.0):
        w = longest_still(t, pos, thr)
        if w:
            i0, i1 = w
            if (t[i1] - t[i0]) / 1e9 >= min_dur:
                return i0, i1, thr
    return None


def jitter(t, pos, t0, t1):
    """std of each coord over [t0,t1] (native samples)."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 4:
        return None
    return pos[m].std(axis=0), int(m.sum())


def kabsch(P, Q):
    """Rigid-align P onto Q (both N x 3). Returns aligned P and RMS residual."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pa = (R @ Pc.T).T + Q.mean(0)
    rms = float(np.sqrt(((Pa - Q) ** 2).sum(1).mean()))
    return Pa, rms


def interp_to(t_src, pos_src, t_ref):
    out = np.zeros((len(t_ref), 3))
    for k in range(3):
        out[:, k] = np.interp(t_ref.astype(float), t_src.astype(float), pos_src[:, k])
    return out


def summary(name, t, pos):
    dur = (t[-1] - t[0]) / 1e9
    rate = len(t) / dur if dur > 0 else 0
    print(f"  {name:7s}: {len(t)} samples, {dur:.1f}s, ~{rate:.0f} Hz")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python pose_benchmark.py cobot_log.jsonl probe_pose_log.jsonl")
    cob_path, cam_path = sys.argv[1], sys.argv[2]

    tc, cc = load(cob_path)                      # cobot (clean, defines the window)
    if len(tc) < 4:
        sys.exit("not enough cobot poses")
    t0, t1 = tc[0] - int(1e9), tc[-1] + int(1e9)

    ta, ca = load(cam_path, only_id=CAM_ID)      # camera (append pile -> window it)
    m = (ta >= t0) & (ta <= t1)
    ta, ca = ta[m], ca[m]
    if len(ta) < 4:
        sys.exit("no camera poses overlap the cobot sweep window")

    cob_xyz, cob_rpy = cc[:, :3], cc[:, 3:6]
    cam_xyz, cam_rpy = ca[:, :3], ca[:, 3:6]

    print("=== samples ===")
    summary("camera", ta, cam_xyz)
    summary("cobot",  tc, cob_xyz)

    # ---- 1. static jitter (noise floor over the longest still segment) ----
    print("\n=== static jitter (noise floor) ===")
    win = find_still_window(ta, cam_xyz)         # detect on the high-rate camera
    if win is None:
        print("  no clear still segment in this sweep — for a clean jitter number,")
        print("  capture a 2-3 s hold (probe stationary) and re-run.")
    else:
        i0, i1, thr = win
        wt0, wt1 = ta[i0], ta[i1]
        print(f"  still window {(wt1-wt0)/1e9:.1f}s (speed < {thr:.0f} mm/s)")
        for nm, t, xyz, rpy in (("camera", ta, cam_xyz, cam_rpy), ("cobot", tc, cob_xyz, cob_rpy)):
            jp = jitter(t, xyz, wt0, wt1); jr = jitter(t, rpy, wt0, wt1)
            if jp and jr:
                sx, sy, sz = jp[0]; rx, ry, rz = jr[0]
                mag = float(np.sqrt(sx*sx + sy*sy + sz*sz))
                print(f"  {nm:7s}: pos x/y/z {sx:.3f}/{sy:.3f}/{sz:.3f} mm "
                      f"(|{mag:.3f}|)   rot {rx:.2f}/{ry:.2f}/{rz:.2f} deg  [{jp[1]} samples]")
            else:
                print(f"  {nm:7s}: too few samples in window")

    # ---- 2. smoothness / quantization ----
    print("\n=== smoothness / quantization ===")
    for nm, t, xyz in (("camera", ta, cam_xyz), ("cobot", tc, cob_xyz)):
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        repeated = float(np.mean(step < 1e-4)) * 100.0          # identical-to-previous %
        uniq = len(np.unique(np.round(xyz, 3), axis=0))
        jerk = float(np.std(np.diff(xyz, n=2, axis=0)))         # 2nd-difference spread
        print(f"  {nm:7s}: median step {np.median(step):.3f} mm   "
              f"repeated samples {repeated:.0f}%   unique poses {uniq}/{len(xyz)}   "
              f"jerk {jerk:.3f}")

    # ---- 3. trajectory agreement (Kabsch align camera -> cobot) ----
    print("\n=== trajectory agreement (rigid-aligned) ===")
    cam_on_cob = interp_to(ta, cam_xyz, tc)      # camera resampled to cobot timestamps
    _, rms = kabsch(cam_on_cob, cob_xyz)
    path_cob = float(np.linalg.norm(np.diff(cob_xyz, axis=0), axis=1).sum())
    print(f"  Kabsch RMS residual: {rms:.2f} mm over a {path_cob:.0f} mm path")
    print("  (combined disagreement about the same motion — both sources contribute;")
    print("   the camera is the tighter source, so most of this is cobot error.)")

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cam_a, _ = kabsch(cam_on_cob, cob_xyz)
        tt = (tc - tc[0]) / 1e9
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for k, lab in enumerate("XYZ"):
            ax[k].plot(tt, cob_xyz[:, k], color="#C2410C", lw=1.4, label="cobot")
            ax[k].plot(tt, cam_a[:, k],  color="#0F6E56", lw=1.4, label="camera (aligned)")
            ax[k].set_ylabel(f"{lab} (mm)")
            if win:
                ax[k].axvspan((ta[win[0]]-tc[0])/1e9, (ta[win[1]]-tc[0])/1e9, color="0.85", zorder=0)
        ax[0].legend(loc="upper right", fontsize=9)
        ax[0].set_title(f"pose sources over the sweep — Kabsch RMS {rms:.2f} mm "
                        f"(shaded = still window)")
        ax[-1].set_xlabel("time (s)")
        out_png = _REPO_ROOT / "pose_benchmark.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=130)
        print(f"\nsaved {out_png}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()