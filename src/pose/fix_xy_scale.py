#!/usr/bin/env python3
"""
fix_xy_scale.py — retroactively apply full 3-axis depth fusion to a cam log.

track_probe fused depth into z only; ax,ay stayed raw PnP and carry PnP's
6-8% viewpoint scale bias. PnP bearing is reliable, its range isn't, and
depth has the true range: ax' = ax*(dz/az), ay' = ay*(dz/az), z = dz.
Works on any existing log because aruco_z and depth_z are stored per line.

    python fix_xy_scale.py in.jsonl out.jsonl
"""
import json, sys
if len(sys.argv) != 3:
    sys.exit("usage: fix_xy_scale.py in.jsonl out.jsonl")
n = fixed = 0
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line); n += 1
        az, dz = r.get("aruco_z"), r.get("depth_z")
        if az and dz and az > 1.0:
            s = dz / az
            if 0.7 < s < 1.3:                      # sane fusion only
                c = r["coords"]
                r["coords"] = [round(c[0]*s, 3), round(c[1]*s, 3), dz] + c[3:]
                fixed += 1
        out.write(json.dumps(r) + "\n")
print(f"{fixed}/{n} poses rescaled (bearing x true depth range)")