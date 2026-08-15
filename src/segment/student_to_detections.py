#!/usr/bin/env python3
"""
student_to_detections.py — convert YOLO student predictions into
sam_detections.json format, so view_planning_3d (and every downstream tool)
can consume the student's output unchanged. The interface IS the design:
student mimics teacher's format, pipeline never notices the swap.

    python3 student_to_detections.py section_85 runs/student85/labels \
        --out data/clarius_sessions/section_85/student_detections.json

Then the tube-map exam:
    python src/reconstruct/view_planning_3d.py section_85 \
        --detections data/clarius_sessions/section_85/student_detections.json ...
(or temporarily swap filenames if view_planning_3d hardcodes sam_detections.json)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from segment_tube import find_section

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("labels", help="YOLO predict labels dir (save_txt output)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    sec = find_section(args.section)
    jsons = sorted(sec.glob("raw_*.json"))
    jpg = sec / "frames_jpg"
    sample = cv2.imread(str(next(jpg.glob("*.jpg"))), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape

    dets = []
    n_frames_with = 0
    for i, jp in enumerate(jsons):
        meta = json.loads(jp.read_text())["frame"]
        axial = meta["axial_um_per_sample"] / 1000.0
        lateral = meta["lateral_um_per_line"] / 1000.0
        t = Path(args.labels) / f"{i:05d}.txt"
        if not t.exists():
            continue
        any_ = False
        for k, line in enumerate(t.read_text().strip().split("\n")):
            v = line.split()
            if len(v) < 5:
                continue
            conf = float(v[5]) if len(v) > 5 else 1.0
            if conf < args.conf:
                continue
            _, cx, cy, w, h = map(float, v[:5])
            cx_px, cy_px = cx * W, cy * H
            r_mm = (h * H * axial) / 2.0          # axial semi-axis, like the teacher
            dets.append(dict(frame_index=i, stem=jp.stem, inst=k,
                             cx=cx_px, cy=cy_px,
                             cx_mm=(cx_px - W / 2) * lateral,
                             depth_mm=cy_px * axial,
                             r_mm=round(r_mm, 3),
                             sam_box=[cx - w / 2, cy - h / 2, w, h],
                             prob=conf, carried=False, source="student"))
            any_ = True
        if any_:
            n_frames_with += 1

    out = Path(args.out) if args.out else sec / "student_detections.json"
    out.write_text(json.dumps(dict(section=sec.name, n_frames=len(jsons),
                                   detections=dets), indent=2))
    rr = np.array([d["r_mm"] for d in dets])
    print(f"{len(dets)} detections across {n_frames_with} frames -> {out}")
    if rr.size:
        print(f"radius: median {np.median(rr):.2f} mm "
              f"(p10 {np.percentile(rr,10):.2f}, p90 {np.percentile(rr,90):.2f})")

if __name__ == "__main__":
    main()