#!/usr/bin/env python3
"""Pair-consistency gate: keep frames with >=2 dets forming a plausible parallel
pair (depth agreement, similar radii, sane lateral separation). Kept frames keep
ALL their dets. --census prints survival only; without it writes v5fp files."""
import json, sys, itertools
from pathlib import Path
DEPTH_TOL = 4.0     # mm, |depth difference| for the pair
R_RATIO   = 1.8     # max/min radius
SEP_MIN, SEP_MAX = 2.0, 30.0   # mm lateral separation
census = "--census" in sys.argv
secs = [a for a in sys.argv[1:] if not a.startswith("--")]
for sec in secs:
    p = Path(f"data/clarius_sessions/{sec}/sam_detections_v5f.json")
    j = json.loads(p.read_text())
    byf = {}
    for d in j["detections"]:
        byf.setdefault(d["frame_index"], []).append(d)
    kept, kept_frames = [], 0
    for fi, ds in byf.items():
        ok = False
        for a, b in itertools.combinations(ds, 2):
            dz  = abs(a["depth_mm"] - b["depth_mm"])
            sep = abs(a["cx_mm"] - b["cx_mm"])
            rr  = max(a["r_mm"], b["r_mm"]) / max(min(a["r_mm"], b["r_mm"]), 1e-6)
            if dz <= DEPTH_TOL and SEP_MIN <= sep <= SEP_MAX and rr <= R_RATIO:
                ok = True; break
        if ok:
            kept.extend(ds); kept_frames += 1
    print(f"{sec}: frames {len(byf)} -> {kept_frames}, dets {len(j['detections'])} -> {len(kept)}")
    if not census:
        out = dict(j); out["detections"] = kept
        out["gate"] = dict(rule="pair-consistency", depth_tol_mm=DEPTH_TOL,
                           r_ratio=R_RATIO, sep_mm=[SEP_MIN, SEP_MAX])
        Path(f"data/clarius_sessions/{sec}/sam_detections_v5fp.json").write_text(json.dumps(out))
