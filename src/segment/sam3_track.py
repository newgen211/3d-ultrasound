#!/usr/bin/env python3
"""
sam3_track.py — per-frame SAM 3 EXEMPLAR segmentation (find ALL vessels), guarded

Fixes over the first exemplar run:
  1. INPUT GUARD on the exemplar. The exemplar box comes from the classical
     detector's best blob on each frame. On some frames that "best blob" was the
     big anechoic band at the image edge -> SAM was told "find 4 mm dark edge
     things" and obliged. Now we reject any exemplar that is too big
     (> MAX_EXEMPLAR_R_MM) or hugging the frame edge (EDGE_MARGIN_FRAC), so SAM
     only ever gets a clean small vessel as the example. Frame skipped if none.
  2. OUTPUT GUARD. Drop returned instances that are too big (> MAX_VESSEL_R_MM)
     or edge-hugging -> removes any residual band detections.
  3. STORE SAM's REAL BOX (out_boxes_xywh) per detection so the overlay can draw
     the actual detection instead of a synthesized circle (lets us judge the true
     lateral extent, which beam-spread blooms).

Note on radius: r_mm = ellipse MINOR axis = the AXIAL (vertical) dimension, which
is the trustworthy one. Lateral width is partly real, partly beam-spread artifact;
the stored sam_box lets you see SAM's actual extent and decide.

    python3 sam3_track.py section_59 --checkpoint ~/ckpt_sam3/sam3.pt --keep-jpg

Box prompts go in NORMALIZED [0,1] xywh (SAM asserts this). Points were pixels —
inconsistent, but that's the API.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

from segment_tube import find_section, load_frame, to_u8, candidates

# --- guards ---
MAX_EXEMPLAR_R_MM = 2.5    # exemplar must be a small clean vessel, not the dark band
MAX_VESSEL_R_MM   = 3.5    # drop output detections bigger than a real vessel (~6 mm dia)
EDGE_MARGIN_FRAC  = 0.06   # reject anything whose centroid is within this frac of the L/R edge


def decode_frames(section, jpg_dir):
    jpg_dir.mkdir(parents=True, exist_ok=True)
    jsons = sorted(section.glob("raw_*.json"))
    if not jsons:
        sys.exit("no frames in section")
    meta_per_frame = []
    for i, jp in enumerate(jsons):
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        f = meta["frame"]
        axial = f["axial_um_per_sample"] / 1000.0
        lateral = f["lateral_um_per_line"] / 1000.0
        if not bp.exists():
            meta_per_frame.append((jp, axial, lateral, None)); continue
        img = to_u8(load_frame(bp, meta))
        cv2.imwrite(str(jpg_dir / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        meta_per_frame.append((jp, axial, lateral, img.shape))
    return meta_per_frame


def exemplar_box(jp, axial, lateral, H, W):
    """Classical detector -> best CLEAN small vessel box (normalized xywh), or None.

    Guarded: reject big blobs and edge-huggers so the dark band can't become the
    exemplar.
    """
    meta = json.loads(jp.read_text())
    cands = candidates(load_frame(jp.with_suffix(".bin"), meta), axial, lateral)
    lo, hi = EDGE_MARGIN_FRAC * W, (1 - EDGE_MARGIN_FRAC) * W
    good = [c for c in cands
            if c["r_mm"] <= MAX_EXEMPLAR_R_MM and lo <= c["cx"] <= hi]
    if not good:
        return None
    c = max(good, key=lambda d: d["quality"])
    (ecx, ecy), (d1, d2), _ = c["ellipse"]
    w, h = max(4.0, float(d1)), max(4.0, float(d2))
    x = max(0.0, min(float(ecx) - w / 2.0, W - 1))
    y = max(0.0, min(float(ecy) - h / 2.0, H - 1))
    w = min(w, W - x); h = min(h, H - y)
    return [x / W, y / H, w / W, h / H]      # NORMALIZED xywh


def mask_metrics(mask, axial, lateral, W):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim > 2:
        m = m.squeeze()
    if m.ndim != 2 or m.sum() < 10:
        return None
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_mm = float("nan")
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            cmm = c.astype(np.float32) * np.array([lateral, axial], np.float32)
            (_, _), (e1, e2), _ = cv2.fitEllipse(cmm)
            r_mm = min(e1, e2) / 2.0
    return (float(cx), float(cy),
            float((cx - W / 2) * lateral), float(cy * axial), r_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-jpg", action="store_true")
    args = ap.parse_args()

    section = find_section(args.section)
    jpg_dir = section / "frames_jpg"
    print(f"{section.name}: decoding frames -> {jpg_dir}")
    meta_per_frame = decode_frames(section, jpg_dir)
    n = len(meta_per_frame)

    import torch
    from sam3.model_builder import build_sam3_video_predictor
    build_kw = {}
    if args.checkpoint:
        build_kw["checkpoint_path"] = args.checkpoint
    predictor = build_sam3_video_predictor(**build_kw)

    resp = predictor.handle_request(request=dict(type="start_session",
                                                 resource_path=str(jpg_dir)))
    session_id = resp["session_id"]

    detections = []
    skipped_no_exemplar = 0
    dropped_guard = 0
    printed = False
    for i, (jp, axial, lateral, shape) in enumerate(meta_per_frame):
        if shape is None:
            continue
        H, W = shape
        box = exemplar_box(jp, axial, lateral, H, W)
        if box is None:
            skipped_no_exemplar += 1
            continue

        predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
        resp = predictor.handle_request(request=dict(
            type="add_prompt", session_id=session_id, frame_index=i,
            bounding_boxes=[box], bounding_box_labels=[1],   # positive exemplar -> all like it
        ))
        out = resp.get("outputs", {}) if isinstance(resp, dict) else {}
        masks = out.get("out_binary_masks")
        sam_boxes = out.get("out_boxes_xywh")
        if masks is None:
            if not printed:
                print(f"  [frame {i}] no masks; keys={list(out.keys())}")
                printed = True
            continue
        if not printed:
            print(f"  first frame {i}: exemplar (guarded) -> {len(list(masks))} raw instances; "
                  f"keys={list(out.keys())}")
            printed = True

        lo, hi = EDGE_MARGIN_FRAC * W, (1 - EDGE_MARGIN_FRAC) * W
        for k, mask in enumerate(masks):
            met = mask_metrics(mask, axial, lateral, W)
            if met is None:
                continue
            cx, cy, cx_mm, depth_mm, r_mm = met
            # OUTPUT GUARD
            if (np.isfinite(r_mm) and r_mm > MAX_VESSEL_R_MM) or not (lo <= cx <= hi):
                dropped_guard += 1
                continue
            sb = None
            if sam_boxes is not None and k < len(sam_boxes):
                b = sam_boxes[k]
                sb = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
            detections.append(dict(frame_index=i, stem=jp.stem, inst=k,
                                   cx=cx, cy=cy, cx_mm=cx_mm,
                                   depth_mm=depth_mm, r_mm=r_mm, sam_box=sb))

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))

    frames_with = len({d["frame_index"] for d in detections})
    per_frame = [sum(1 for d in detections if d["frame_index"] == f)
                 for f in {d["frame_index"] for d in detections}]
    rr = np.array([d["r_mm"] for d in detections if np.isfinite(d["r_mm"])])
    print(f"\n{len(detections)} detections across {frames_with}/{n} frames "
          f"(skipped {skipped_no_exemplar} no-exemplar, dropped {dropped_guard} by guard)")
    if per_frame:
        print(f"instances per frame: median {int(np.median(per_frame))}, max {max(per_frame)}")
    if rr.size:
        print(f"radius (axial): median {np.median(rr):.2f} mm "
              f"(p10 {np.percentile(rr,10):.2f}, p90 {np.percentile(rr,90):.2f})")

    out_path = Path(args.out) if args.out else section / "sam_detections.json"
    out_path.write_text(json.dumps(dict(section=section.name, n_frames=n,
                                        detections=detections), indent=2))
    print(f"wrote {out_path}")

    if not args.keep_jpg:
        shutil.rmtree(jpg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()