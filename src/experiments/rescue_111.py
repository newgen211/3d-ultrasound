#!/usr/bin/env python3
# Undo the camera bump on flight 111: Kabsch on static markers 1/2/3.
import json, glob
import numpy as np
from pathlib import Path

ROOT  = Path("data/clarius_sessions")
CAM   = Path("data/pose_logs/probe_pose_log_smooth.jsonl")
ERA_A = ["section_106", "section_108", "section_109"]
ERA_B = "section_111"
IDS   = [1, 2, 3]

def frame_span(sec):
    ts = []
    for f in glob.glob(str(ROOT / sec / "*.json")):
        try:
            t = json.load(open(f)).get("host_timestamp_ns")
        except Exception:
            continue
        if t: ts.append(int(t))
    return (min(ts), max(ts), len(ts)) if ts else None

def load_centerline(sec):
    return np.asarray(json.load(open(ROOT / sec / "centerline.json"))["centerline_mm"], float)

def nn_dist(P, Q):
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    both = np.concatenate([d.min(1), d.min(0)])
    return both.mean(), np.percentile(both, 95)

# --- era windows from the frames themselves -------------------------------
spans = {s: frame_span(s) for s in ERA_A + [ERA_B]}
for s, v in spans.items():
    if v is None: raise SystemExit(f"no frame timestamps found in {s}")
    print(f"{s}: {v[2]} frames")
A_lo = min(spans[s][0] for s in ERA_A); A_hi = max(spans[s][1] for s in ERA_A)
B_lo, B_hi, _ = spans[ERA_B]
print(f"\nera A window {(A_hi-A_lo)/1e9/60:.1f} min | era B {(B_hi-B_lo)/1e9/60:.1f} min "
      f"| gap {(B_lo-A_hi)/1e9/60:.1f} min")
if B_lo <= A_hi: raise SystemExit("ERROR: eras overlap — check section list")

# --- marker medians per era -----------------------------------------------
buckets = {("A", i): [] for i in IDS} | {("B", i): [] for i in IDS}
for line in open(CAM):
    try: r = json.loads(line)
    except Exception: continue
    i, t = r.get("id"), r.get("t_ns")
    if i not in IDS or t is None: continue
    era = "A" if A_lo <= t <= A_hi else ("B" if B_lo <= t <= B_hi else None)
    if era: buckets[(era, i)].append(r["coords"][:3])

print("\nmarker    era A median            era B median            |delta| mm")
A_pts, B_pts = [], []
for i in IDS:
    a, b = np.asarray(buckets[("A", i)], float), np.asarray(buckets[("B", i)], float)
    if len(a) < 20 or len(b) < 20:
        raise SystemExit(f"id {i}: too few samples (A={len(a)} B={len(b)})")
    ma, mb = np.median(a, 0), np.median(b, 0)
    A_pts.append(ma); B_pts.append(mb)
    print(f"id {i}  [{ma[0]:7.1f}{ma[1]:7.1f}{ma[2]:7.1f}]  "
          f"[{mb[0]:7.1f}{mb[1]:7.1f}{mb[2]:7.1f}]  {np.linalg.norm(ma-mb):8.1f}"
          f"   (n={len(a)}/{len(b)})")
A_pts, B_pts = np.array(A_pts), np.array(B_pts)

# --- Kabsch: B -> A --------------------------------------------------------
ca, cb = A_pts.mean(0), B_pts.mean(0)
U, S, Vt = np.linalg.svd((B_pts - cb).T @ (A_pts - ca))
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, d]) @ U.T
t = ca - R @ cb
resid = np.linalg.norm((R @ B_pts.T).T + t - A_pts, axis=1)
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print(f"\nfit: rotation {ang:.2f}deg, translation {np.linalg.norm(t):.1f} mm")
print(f"per-marker residual {np.round(resid,2)} mm  (RMS {np.sqrt((resid**2).mean()):.2f})")

# --- apply + score ---------------------------------------------------------
C111 = load_centerline(ERA_B)
C111c = (R @ C111.T).T + t
print("\npair            before (mean/p95)     after (mean/p95)")
for s in ERA_A:
    Q = load_centerline(s)
    m0, p0 = nn_dist(C111, Q); m1, p1 = nn_dist(C111c, Q)
    print(f"111 vs {s[-3:]}    {m0:7.2f} /{p0:7.2f}      {m1:7.2f} /{p1:7.2f}")

out = ROOT / ERA_B / "centerline_corrected.json"
json.dump({"centerline_mm": C111c.tolist(),
           "correction": {"R": R.tolist(), "t": t.tolist(),
                          "rot_deg": ang, "resid_rms_mm": float(np.sqrt((resid**2).mean())),
                          "source": "kabsch on marker ids 1/2/3, era A=106/108/109"}},
          open(out, "w"), indent=1)
print(f"\nwrote {out}")
