#!/usr/bin/env python3
"""Stratified audit sample: N frames/section + teacher detections -> audit/."""
import json, random, shutil, sys
from pathlib import Path

SECTIONS = ["section_62", "section_81", "section_85", "section_90",
            "section_92", "section_94", "section_103", "section_104"]
N_PER = 20
random.seed(7)

root = Path("data/clarius_sessions")
out = Path("audit"); (out / "frames").mkdir(parents=True, exist_ok=True)
manifest = {}
for s in SECTIONS:
    jpgs = sorted((root / s / "frames_jpg").glob("*.jpg"))
    if not jpgs:
        print(f"!! {s}: no frames_jpg — decode first, skipping"); continue
    step = max(len(jpgs) // N_PER, 1)
    picks = jpgs[::step][:N_PER]
    # teacher labels, if real SAM ones exist
    det_by_frame = {}
    dp = root / s / "sam_detections.json"
    if dp.exists():
        for d in json.loads(dp.read_text()).get("detections", []):
            det_by_frame.setdefault(int(d.get("frame_index", d.get("frame", -1))), []).append(d)
    for p in picks:
        idx = int(p.stem)
        key = f"{s}__{p.name}"
        shutil.copy(p, out / "frames" / key)
        manifest[key] = {"section": s, "frame": idx,
                         "teacher": det_by_frame.get(idx, [])}
print(f"{len(manifest)} frames from {len(set(v['section'] for v in manifest.values()))} sections")
(out / "audit_manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote audit/audit_manifest.json")
