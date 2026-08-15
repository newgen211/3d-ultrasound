#!/usr/bin/env python3
"""
make_sidebyside_video.py — SAM (teacher) vs YOLO (student) side-by-side clip.

    python3 make_sidebyside_video.py section_85 runs/student85_regated/labels
    python3 make_sidebyside_video.py section_62 "runs/detect/runs/student62/labels"

Left = teacher (green, from sam_detections.json; pass --teacher to override,
e.g. sam_detections_TEACHER.json if the swap dance renamed it).
Right = student (cyan, YOLO predict save_txt labels dir).
Writes <section>/sidebyside.mp4 at 15 fps.
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
    ap.add_argument("labels", help="YOLO save_txt labels dir")
    ap.add_argument("--teacher", default=None, help="teacher json (default sam_detections.json)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    sec = find_section(args.section)
    jpg = sec / "frames_jpg"
    tpath = Path(args.teacher) if args.teacher else sec / "sam_detections.json"
    sam = {}
    for d in json.loads(tpath.read_text())["detections"]:
        sam.setdefault(d["frame_index"], []).append(d)
    lbl = Path(args.labels)

    frames = sorted(jpg.glob("*.jpg"))
    sample = cv2.imread(str(frames[0])); H, W = sample.shape[:2]
    GAP = 8
    tmp = sec / "sbs_frames"
    tmp.mkdir(exist_ok=True)
    for p in frames:
        i = int(p.stem)
        base = cv2.imread(str(p))
        L, R = base.copy(), base.copy()
        for d in sam.get(i, []):
            if d.get("sam_box"):
                x, y, w, h = d["sam_box"]
                cv2.rectangle(L, (int(x*W), int(y*H)),
                              (int((x+w)*W), int((y+h)*H)), (0, 220, 0), 2)
        t = lbl / f"{i:05d}.txt"
        if t.exists():
            for line in t.read_text().strip().split("\n"):
                v = line.split()
                if len(v) >= 5 and (len(v) < 6 or float(v[5]) >= args.conf):
                    _, cx, cy, w, h = map(float, v[:5])
                    cv2.rectangle(R, (int((cx-w/2)*W), int((cy-h/2)*H)),
                                  (int((cx+w/2)*W), int((cy+h/2)*H)), (255, 220, 0), 2)
        canvas = np.full((H + 36, W * 2 + GAP, 3), 15, np.uint8)
        canvas[36:, :W] = L
        canvas[36:, W + GAP:] = R
        cv2.putText(canvas, "SAM teacher (25 min / cluster)", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        cv2.putText(canvas, "YOLO26 student (real-time)", (W + GAP + 6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)
        cv2.putText(canvas, f"f{i}", (W * 2 + GAP - 70, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), canvas)
    import subprocess, shutil
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                    "-pattern_type", "glob", "-i", str(tmp / "*.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(sec / "sidebyside.mp4")], check=True,
                   capture_output=True)
    shutil.rmtree(tmp)
    print(f"wrote {sec/'sidebyside.mp4'}  ({len(frames)} frames @ {args.fps} fps "
          f"= {len(frames)/args.fps:.0f} s)")

if __name__ == "__main__":
    main()