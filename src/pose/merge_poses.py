#!/usr/bin/env python
"""
merge_poses.py — attach cobot poses to Clarius frames by timestamp

Each ultrasound frame in a section folder has a host-clock timestamp
(host_timestamp_ns). Each line in pose_log.jsonl has the same kind of
timestamp (t_ns). Because the Pi and the Mac are on the same (NTP) clock,
we can match every frame to the arm pose recorded closest to it in time.

For each frame this adds a "cobot_pose" field to its sidecar .json:
    "cobot_pose": {
        "coords": [x, y, z, rx, ry, rz],   # mm, degrees, arm base frame
        "angles": [...],                    # 6 joint angles, if logged
        "pose_t_ns": <pose timestamp>,
        "dt_ms": <frame-to-pose time gap>
    }
Frames with no pose within the time window get "cobot_pose": null
(they're not deleted — the reconstruction just skips them).

Run on the Mac, after copying pose_log.jsonl over from the Pi.

Usage:
    python merge_poses.py                              # latest section, ./pose_log.jsonl
    python merge_poses.py section_19 pose_log.jsonl
    python merge_poses.py section_19 pose_log.jsonl --max-dt-ms 100
"""

import argparse
import bisect
import json
import sys
from pathlib import Path

# Anchor to the repo's data/ folder, regardless of where this is launched.
DATA = Path(__file__).resolve().parents[2] / "data"


def find_section(arg):
    root = DATA / "clarius_sessions"
    if not root.exists():
        sys.exit(f"❌ No clarius_sessions/ folder at {root}")
    if arg is None:
        sections = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("section_")],
            key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
        )
        if not sections:
            sys.exit("❌ No section_N folders")
        return sections[-1]
    for cand in (Path(arg), root / arg):
        if cand.exists():
            return cand
    sys.exit(f"❌ Section not found: {arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("pose_log", nargs="?", default=str(DATA / "pose_logs" / "pose_log.jsonl"))
    ap.add_argument("--max-dt-ms", type=float, default=100.0,
                    help="Max frame-to-pose time gap to count as a match (default 100 ms)")
    args = ap.parse_args()

    section = find_section(args.section)
    pose_path = Path(args.pose_log)
    if not pose_path.exists():                       # fall back to data/pose_logs/<name>
        alt = DATA / "pose_logs" / pose_path.name
        if alt.exists():
            pose_path = alt
    if not pose_path.exists():
        sys.exit(f"❌ Pose log not found: {pose_path}  (copy it over from the Pi first)")

    # ---- load + sort poses ----
    poses = []
    for line in pose_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "t_ns" in p and "coords" in p:
            poses.append(p)
    if not poses:
        sys.exit("❌ No usable poses in the log")
    poses.sort(key=lambda p: p["t_ns"])
    pose_ts = [p["t_ns"] for p in poses]
    print(f"📋 {len(poses)} poses spanning {(pose_ts[-1]-pose_ts[0])/1e9:.2f} s")

    # ---- match each frame to nearest pose ----
    sidecars = sorted(section.glob("raw_*.json"))
    if not sidecars:
        sys.exit(f"❌ No raw_*.json frames in {section}")
    print(f"📂 {section.name}: {len(sidecars)} frames")

    max_dt_ns = args.max_dt_ms * 1e6
    matched, dropped, dts = 0, 0, []

    for sc in sidecars:
        meta = json.loads(sc.read_text())
        ft = meta.get("host_timestamp_ns")
        pose = None
        if ft is not None:
            i = bisect.bisect_left(pose_ts, ft)
            cands = [j for j in (i, i - 1) if 0 <= j < len(pose_ts)]
            best = min(cands, key=lambda j: abs(pose_ts[j] - ft))
            dt = abs(pose_ts[best] - ft)
            if dt <= max_dt_ns:
                p = poses[best]
                pose = {
                    "coords": p["coords"],
                    "angles": p.get("angles"),
                    "pose_t_ns": p["t_ns"],
                    "dt_ms": round(dt / 1e6, 2),
                }
                dts.append(dt / 1e6)

        meta["cobot_pose"] = pose
        sc.write_text(json.dumps(meta, indent=2))
        if pose:
            matched += 1
        else:
            dropped += 1

    print(f"\n✅ matched {matched} frames, dropped {dropped} "
          f"(no pose within {args.max_dt_ms:.0f} ms)")
    if dts:
        dts.sort()
        print(f"   match gap  min {dts[0]:.1f} / median {dts[len(dts)//2]:.1f} "
              f"/ max {dts[-1]:.1f} ms")
    print(f"   added 'cobot_pose' to each sidecar in {section.name}/")


if __name__ == "__main__":
    main()