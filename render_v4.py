#!/usr/bin/env python3
"""Draw v2/v3 labels on sample frames -> overlay jpgs for eyeballing."""
import json, sys
from pathlib import Path
import cv2
import numpy as np

for sec_name in sys.argv[1:]:
    sec = Path("data/clarius_sessions") / sec_name
    dets = json.loads((sec / "sam_detections_v4.json").read_text())["detections"]
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d["frame_index"], []).append(d)
    rep_p = sec / "compression_report.json"
    rep = json.loads(rep_p.read_text())["frames"] if rep_p.exists() else None
    out = Path("overlays") / sec_name
    out.mkdir(parents=True, exist_ok=True)
    # sample: every Nth labeled frame, max 25
    frames = sorted(by_frame.keys())
    step = max(1, len(frames) // 25)
    for fi in frames[::step]:
        jp = sec / "frames_jpg" / f"{fi:05d}.jpg"
        if not jp.exists():
            jp = sec / "frames_jpg" / f"{fi}.jpg"
            if not jp.exists():
                continue
        img = cv2.imread(str(jp))
        state = rep[fi]["state"] if (rep and fi < len(rep)) else "no-report"
        for d in by_frame[fi]:
            color = (0, 255, 0) if d.get("morph") == "open" else (0, 165, 255)
            cv2.circle(img, (int(d["cx"]), int(d["cy"])), 12, color, 2)
            cv2.putText(img, f"{d.get('morph','?')[:4]} p{d.get('prob',0):.2f}",
                        (int(d["cx"]) - 30, int(d["cy"]) - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(img, f"f{fi} {state}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"{fi:05d}.jpg"), img)
    print(f"{sec_name}: {len(list(out.glob('*.jpg')))} overlays -> {out}/")
