#!/usr/bin/env python
"""
merge_poses_cam.py — attach the nearest RealSense (camera/marker) pose to each
frame sidecar, by timestamp. Camera counterpart of merge_poses.py.

Reads the depth-fused pose from track_probe.py's log, filters to the probe
marker (TARGET_ID), and writes it into each frame's sidecar under the SAME field
the rest of the pipeline reads -- so digitize_beads.py, calibrate_handeye.py and
reconstruct_handeye.py work unchanged, now solving/using marker->image instead of
flange->image.

Usage:
    python merge_poses_cam.py section_<N> probe_pose_log.jsonl
"""
import sys, json, glob, os
from pathlib import Path
import numpy as np

# Anchor to the repo's data/ folder, regardless of where this is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA      = _REPO_ROOT / "data"
FIELD     = "cobot_pose"   # <-- field calibrate_handeye / reconstruct_handeye read.
                           #     If yours uses a different name, change this one string.
TARGET_ID = 0
TOL_MS    = 100            # max frame<->pose time gap to accept


def main():
    if len(sys.argv) != 3:
        print("usage: python merge_poses_cam.py section_<N> probe_pose_log.jsonl")
        sys.exit(1)
    sec, pose_log = sys.argv[1], sys.argv[2]
    secdir = sec if os.path.isdir(sec) else os.path.join(DATA, "clarius_sessions", sec)

    # resolve the pose log: explicit path, else fall back to data/pose_logs/<name>
    if not os.path.exists(pose_log):
        alt = os.path.join(DATA, "pose_logs", os.path.basename(pose_log))
        if os.path.exists(alt):
            pose_log = alt
    if not os.path.exists(pose_log):
        print(f"❌ Pose log not found: {pose_log}  (looked in cwd and {os.path.join(DATA, 'pose_logs')})")
        sys.exit(1)

    # load camera poses for the probe marker (fused coords)
    poses = []
    for line in open(pose_log):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("id") == TARGET_ID and r.get("coords") is not None:
            poses.append((int(r["t_ns"]), r["coords"]))
    poses.sort()
    if not poses:
        print(f"No id{TARGET_ID} poses found in {pose_log}")
        sys.exit(1)
    pt = np.array([p[0] for p in poses], dtype=np.int64)
    pc = [p[1] for p in poses]
    print(f"{len(poses)} id{TARGET_ID} poses, {(pt[-1]-pt[0])/1e9:.1f}s span")

    sidecars = sorted(glob.glob(os.path.join(secdir, "raw_*.json")))
    if not sidecars:
        print(f"No raw_*.json sidecars in {secdir}")
        sys.exit(1)

    tol = TOL_MS * 1_000_000
    matched = 0
    for sc in sidecars:
        meta = json.load(open(sc))
        h = meta.get("host_timestamp_ns")
        if h is None:
            continue
        i = int(np.argmin(np.abs(pt - int(h))))
        gap = abs(int(pt[i]) - int(h))
        if gap <= tol:
            meta[FIELD] = {"coords": pc[i]}   # calibrate_handeye reads pose["coords"]
            with open(sc, "w") as f:
                json.dump(meta, f, indent=2)
            matched += 1
        else:
            print(f"  no pose within {TOL_MS}ms for {os.path.basename(sc)} "
                  f"(nearest {gap/1e6:.0f}ms)")

    print(f"matched {matched}/{len(sidecars)} frames -> sidecar '{FIELD}' "
          f"(tol {TOL_MS}ms)")
    if matched < len(sidecars):
        print("  unmatched frames have no pose -> they'll be skipped downstream")


if __name__ == "__main__":
    main()