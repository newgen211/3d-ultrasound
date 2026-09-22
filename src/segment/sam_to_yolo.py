#!/usr/bin/env python3
"""
sam_to_yolo.py — build a YOLO-detect dataset from SAM detections + compression QC.

Label policy (the QC rules, explicit):
  - OPEN frames with detections  -> labeled images (class 0 = vessel)
  - COLLAPSED frames             -> negative images (empty label file) — teaches
                                    the student NOT to hallucinate in collapse
  - EDGE frames                  -> EXCLUDED (unadjudicated; bad labels poison)
  - frames with no compression report: any frame with detections -> labeled;
    detection-free frames -> excluded (can't certify them as true negatives)
  - detections keep their sam_box when present; else a box is synthesized from
    the center + radius (r_mm converted to px, lateral width x1.9 for beam bloom)

Split is BY SECTION (adjacent frames are near-duplicates; frame-level splits leak):
    python3 sam_to_yolo.py --train section_81 section_62 --val section_85 --out yolo_ds

Outputs yolo_ds/{images,labels}/{train,val}/ + dataset.yaml, ready for:
    yolo detect train data=yolo_ds/dataset.yaml model=yolov8n.pt epochs=80 imgsz=640
"""
import argparse, json, shutil, sys
from pathlib import Path

def find_section(name):
    for base in (Path("data/clarius_sessions"), Path(".")):
        p = base / name
        if p.exists():
            return p
    sys.exit(f"section not found: {name}")

def load_report(sec):
    p = sec / "compression_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["frames"]     # ordered like the frames

def process(sec_name, split, out, counts):
    sec = find_section(sec_name)
    dets_path = sec / "sam_detections.json"
    if not dets_path.exists():
        sys.exit(f"{sec_name}: no sam_detections.json — run SAM first")
    jpg_dir = sec / "frames_jpg"
    if not jpg_dir.exists():
        sys.exit(f"{sec_name}: no frames_jpg/ — rerun sam3_track_v2 with --keep-jpg "
                 f"(or scp the folder from Newton)")
    dd = json.loads(dets_path.read_text())["detections"]
    by_frame = {}
    for d in dd:
        by_frame.setdefault(d["frame_index"], []).append(d)
    report = load_report(sec)                       # list or None

    import cv2
    n_img = n_neg = n_det = 0
    for jp in sorted(jpg_dir.glob("*.jpg")):
        i = int(jp.stem)
        dets = by_frame.get(i, [])
        state = report[i]["state"] if (report is not None and i < len(report)) else None

        if dets and state != "EDGE":
            labeled = True                          # positive frame
        elif not dets and state == "COLLAPSED":
            labeled = False                         # certified negative
        else:
            continue                                # EDGE, or uncertified empty

        img = cv2.imread(str(jp)); H, W = img.shape[:2]
        shutil.copy(jp, out / "images" / split / f"{sec_name}_{i:05d}.jpg")
        lines = []
        if labeled:
            for d in dets:
                if d.get("sam_box"):
                    x, y, w, h = d["sam_box"]       # normalized xywh
                    cx, cy = x + w / 2, y + h / 2
                else:
                    cx, cy = d["cx"] / W, d["cy"] / H
                    r_px_ax = (d["r_mm"] / 0.0513) if d.get("r_mm") else 20
                    w = (2 * r_px_ax * 1.9) / W
                    h = (2 * r_px_ax) / H
                lines.append(f"0 {cx:.5f} {cy:.5f} {min(w,1):.5f} {min(h,1):.5f}")
        (out / "labels" / split / f"{sec_name}_{i:05d}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_img += 1; n_det += len(lines)
        if not lines: n_neg += 1
    counts[split].append(f"{sec_name}: {n_img} imgs ({n_neg} negative), {n_det} boxes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="yolo_ds")
    args = ap.parse_args()
    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    counts = {"train": [], "val": []}
    for s in args.train: process(s, "train", out, counts)
    for s in args.val:   process(s, "val", out, counts)
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: vessel\n")
    for sp in ("train", "val"):
        print(f"{sp}:"); [print("  " + l) for l in counts[sp]]
    print(f"wrote {out}/dataset.yaml — train with:\n"
          f"  yolo detect train data={out}/dataset.yaml model=yolov8n.pt "
          f"epochs=80 imgsz=640 fliplr=0.5 flipud=0.0")

if __name__ == "__main__":
    main()