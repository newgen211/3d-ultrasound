#!/usr/bin/env python3
"""
trim_section.py — exclude hover/retract frames using the exec run manifest.

Problem: pysidecaster captures from before the descent to after the retract;
those air frames pollute downstream segmentation. The exec meta knows the
playback window — but in the PI's clock, and frames are stamped in the MAC's
clock (unsynced).

Solution: the exec log (Pi clock) and the camera probe log (Mac clock) both
recorded the SAME physical motion. Cross-correlating their speed profiles
yields the Pi->Mac clock offset directly from data — no sync assumptions.
Frames outside [playback_start, playback_end] (+offset, +margin) are MOVED to
section_<N>/excluded/ (never deleted).

Usage:
    python trim_section.py section_67 pose_logs/exec_<stamp>.jsonl \
        pose_logs/exec_<stamp>_meta.json pose_logs/auto_sweep1_cam.jsonl
    # dry-run by default; add --apply to actually move files
"""

import argparse, glob, json, os, shutil, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("section")
ap.add_argument("exec_log")
ap.add_argument("exec_meta")
ap.add_argument("cam_log", help="track_probe log covering this capture")
ap.add_argument("--margin", type=float, default=0.5,
                help="seconds kept on each side of the playback window")
ap.add_argument("--target-id", type=int, default=0)
ap.add_argument("--apply", action="store_true", help="actually move files")
args = ap.parse_args()

secdir = args.section if os.path.isdir(args.section) \
         else os.path.join("data", "clarius_sessions", args.section)
if not os.path.isdir(secdir):
    sys.exit(f"section dir not found: {args.section}")

def load_track(path, want_id=None):
    T, P = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if want_id is not None and r.get("id") != want_id:
            continue
        if r.get("coords"):
            T.append(int(r["t_ns"])); P.append(r["coords"][:3])
    return np.array(T, dtype=np.int64), np.array(P, dtype=np.float64)

Te, Pe = load_track(args.exec_log)                     # Pi clock
Tc, Pc = load_track(args.cam_log, args.target_id)      # Mac clock
meta = json.load(open(args.exec_meta))
t_start_pi = meta["events"]["playback_start"]
t_end_pi   = meta["events"]["playback_end"]
if len(Te) < 20:
    sys.exit("exec log too short to align")

# the frames and the cam log share the MAC clock — so the frames' own time
# span tells us exactly which slice of the cam log is THIS capture. Windowing
# there prevents the correlator locking onto some other arm motion (staging,
# air tests) elsewhere in an appended log.
sidecars = sorted(glob.glob(os.path.join(secdir, "raw_*.json")))
if not sidecars:
    sys.exit(f"no raw_*.json in {secdir}")
frame_t = [json.load(open(sc)).get("host_timestamp_ns") for sc in sidecars]
frame_t = np.array([t for t in frame_t if t is not None], dtype=np.int64)
pad = np.int64(60e9)
m = (Tc >= frame_t.min() - pad) & (Tc <= frame_t.max() + pad)
if m.sum() < 50:
    sys.exit(f"only {m.sum()} cam poses overlap the frames' time span — wrong "
             f"cam log for this capture?")
Tc, Pc = Tc[m], Pc[m]
print(f"# cam log windowed to the frames' span: {m.sum()} poses, "
      f"{(Tc[-1]-Tc[0])/1e9:.0f} s")

def speed_series(T, P):
    t = (T - T[0]) / 1e9
    # smooth positions (~0.4 s window) BEFORE differentiating: camera pose
    # jitter (1-2 mm/frame) otherwise swamps the ~6 mm/s sweep motion
    from scipy.ndimage import uniform_filter1d
    rate = max(1.0, len(t) / max(t[-1], 1e-3))
    w = max(3, int(0.4 * rate))
    Ps = uniform_filter1d(P, size=w, axis=0, mode="nearest")
    v = np.linalg.norm(np.diff(Ps, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
    return t[1:], v

te, ve = speed_series(Te, Pe)
tc, vc = speed_series(Tc, Pc)
dt = 0.05
ge = np.interp(np.arange(0, te[-1], dt), te, ve)
gc = np.interp(np.arange(0, tc[-1], dt), tc, vc)
ge = (ge - ge.mean()) / (ge.std() + 1e-9)
gc = (gc - gc.mean()) / (gc.std() + 1e-9)
corr = np.correlate(gc, ge, mode="full")
lags = (np.arange(len(corr)) - (len(ge) - 1)) * dt

# constrain to physically possible lags: the playback window must OVERLAP the
# frames' span (offset = Tc[0]-Te[0]+lag; window = playback_[start,end]+offset)
base = int(Tc[0]) - int(Te[0])
lo = (frame_t.min() - t_end_pi - base) / 1e9      # window end   >= first frame
hi = (frame_t.max() - t_start_pi - base) / 1e9    # window start <= last frame
feasible = (lags >= lo) & (lags <= hi)
if not feasible.any():
    sys.exit("no feasible lag — exec meta and frames can't correspond; wrong files?")
corr_f = np.where(feasible, corr, -np.inf)
best = int(np.argmax(corr_f))
lag_s = float(lags[best])
peak = float(corr[best]) / min(len(ge), len(gc))

# exec sample at Pi time Te[0]+x occurs at Mac time Tc[0]+x+lag
#  => Mac = Pi + (Tc[0]-Te[0]) + lag
offset_ns = int(Tc[0]) - int(Te[0]) + int(lag_s * 1e9)

print(f"alignment: lag {lag_s:+.2f} s, correlation peak {peak:.2f} "
      f"({'GOOD' if peak > 0.3 else 'WEAK'})")
print(f"Pi->Mac clock offset: {offset_ns/1e9:+.3f} s")

w0 = t_start_pi + offset_ns - int(args.margin * 1e9)
w1 = t_end_pi   + offset_ns + int(args.margin * 1e9)
print(f"keep window: {(w1-w0)/1e9:.1f} s of frames")

keep_n, drop = 0, []
for sc, h in zip(sidecars, [json.load(open(s)).get("host_timestamp_ns") for s in sidecars]):
    if h is not None and w0 <= int(h) <= w1:
        keep_n += 1
    else:
        drop.append(sc)
print(f"frames: keep {keep_n}, exclude {len(drop)} of {len(sidecars)}")

if not np.isfinite(peak):
    sys.exit("correlation peak is NaN: a null coordinate in the exec or cam log. "
             "Not moving anything.")
if peak <= 0.3:
    sys.exit("correlation too weak to trust the window — not moving anything. "
             "Check that the cam log covers this capture.")
if keep_n == 0:
    sys.exit("window keeps ZERO frames — offset must be wrong; not moving anything.")
if keep_n < 0.5 * len(sidecars):
    sys.exit(f"window keeps only {keep_n} of {len(sidecars)} frames (under half): "
             f"offset or exec pair is wrong. Not moving anything.")
if abs(offset_ns) > 5e9:
    sys.exit(f"Pi->Mac clock offset {offset_ns/1e9:+.1f} s is outside the sane window "
             f"(5 s): wrong exec pair or cam log. Not moving anything.")
if not args.apply:
    print("(dry run — add --apply to move excluded frames to excluded/)")
    sys.exit(0)

outdir = os.path.join(secdir, "excluded")
os.makedirs(outdir, exist_ok=True)
moved = 0
for sc in drop:
    stem = os.path.splitext(sc)[0]
    for f in glob.glob(stem + ".*"):
        shutil.move(f, os.path.join(outdir, os.path.basename(f)))
        moved += 1
print(f"moved {moved} files -> {outdir}/")