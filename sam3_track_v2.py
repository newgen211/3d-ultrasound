#!/usr/bin/env python3
"""
sam3_track_v2.py — per-frame SAM 3 exemplar segmentation, v2.

Changes vs v1 (the audit's marching orders):
  A. DUAL EXEMPLARS: every prompt includes the classical detector's best
     open-vessel box (when present) PLUS audited exemplar boxes for this
     section from exemplars.json (open + crescent, harvested from human
     audit truth). SAM matches BOTH appearance classes.
     Exemplar prompts are placed on their own audited frame index — SAM 3
     PCS matches instances across the prompted frame; we prompt per frame,
     so audited boxes are carried onto each frame as additional positives.
  B. GATE: dark-OR-ring. Keep a mask if interior is anechoic (open lumen,
     v1 rule) OR interior is bright but its boundary annulus is brighter
     still (crescent wall signature). Speckle passes neither.
  C. Every detection carries "morph": "open"|"crescent" (which gate passed).

    python3 sam3_track_v2.py section_85 --checkpoint ~/ckpt_sam3/sam3.pt \
        --exemplars exemplars.json --keep-jpg
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

from segment_tube import find_section, load_frame, to_u8, candidates

MAX_EXEMPLAR_R_MM = 2.5
MAX_VESSEL_R_MM   = 3.5
EDGE_MARGIN_FRAC  = 0.06
CARRY_MAX         = 60
PROB_MIN          = 0.5
DARK_FRAC         = 0.75     # open-lumen gate (v1)
RING_GAIN         = 1.25     # crescent gate: annulus must be this x brighter than interior


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
        cv2.imwrite(str(jpg_dir / f"{i:05d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        meta_per_frame.append((jp, axial, lateral, img.shape))
    return meta_per_frame


def classical_box(jp, axial, lateral, H, W):
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
    return [x / W, y / H, w / W, h / H]


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
    return m, float(cx), float(cy), float((cx - W / 2) * lateral), float(cy * axial), r_mm


def morph_gate(m, gray, frame_med):
    """Return 'open', 'crescent', or None (reject)."""
    interior = float(gray[m > 0].mean())
    if interior < DARK_FRAC * frame_med:
        return "open"
    ring = cv2.dilate(m, np.ones((7, 7), np.uint8)) - m
    if ring.sum() < 10:
        return None
    ring_mean = float(gray[ring > 0].mean())
    if ring_mean > RING_GAIN * interior and ring_mean > frame_med:
        return "crescent"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--exemplars", default="exemplars.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-jpg", action="store_true")
    args = ap.parse_args()

    section = find_section(args.section)
    ex_all = {}
    if Path(args.exemplars).exists():
        ex_all = json.loads(Path(args.exemplars).read_text()).get(section.name, {})
    audit_boxes = [e["box"] for cls in ("open", "crescent")
                   for e in ex_all.get(cls, [])]
    print(f"{section.name}: {len(audit_boxes)} audited exemplar boxes "
          f"({ {c: len(v) for c, v in ex_all.items()} })")

    jpg_dir = section / "frames_jpg"
    meta_per_frame = decode_frames(section, jpg_dir)
    n = len(meta_per_frame)

    from sam3.model_builder import build_sam3_video_predictor
    build_kw = {}
    if args.checkpoint:
        build_kw["checkpoint_path"] = args.checkpoint
    predictor = build_sam3_video_predictor(**build_kw)
    resp = predictor.handle_request(request=dict(type="start_session",
                                                 resource_path=str(jpg_dir)))
    session_id = resp["session_id"]

    detections = []
    stats = dict(no_prompt=0, carried=0, guard=0, prob=0, gate=0,
                 open=0, crescent=0)
    last_cls_box, carry_age = None, 0
    printed = False
    for i, (jp, axial, lateral, shape) in enumerate(meta_per_frame):
        if shape is None:
            continue
        H, W = shape
        cb = classical_box(jp, axial, lateral, H, W)
        if cb is not None:
            last_cls_box, carry_age = cb, 0
        elif last_cls_box is not None and carry_age < CARRY_MAX:
            cb = last_cls_box; carry_age += 1; stats["carried"] += 1
        boxes = ([cb] if cb else []) + audit_boxes
        if not boxes:
            stats["no_prompt"] += 1
            continue

        predictor.handle_request(request=dict(type="reset_session",
                                              session_id=session_id))
        resp = predictor.handle_request(request=dict(
            type="add_prompt", session_id=session_id, frame_index=i,
            bounding_boxes=boxes, bounding_box_labels=[1] * len(boxes)))
        out = resp.get("outputs", {}) if isinstance(resp, dict) else {}
        masks = out.get("out_binary_masks")
        probs = out.get("out_probs")
        sam_boxes = out.get("out_boxes_xywh")
        gray = cv2.imread(str(jpg_dir / f"{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        frame_med = max(float(np.median(gray)), 1.0) if gray is not None else None
        if masks is None or frame_med is None:
            continue
        if not printed:
            print(f"  first frame {i}: {len(boxes)} prompt boxes -> "
                  f"{len(list(masks))} raw instances")
            printed = True

        lo, hi = EDGE_MARGIN_FRAC * W, (1 - EDGE_MARGIN_FRAC) * W
        for k, mask in enumerate(masks):
            met = mask_metrics(mask, axial, lateral, W)
            if met is None:
                continue
            m, cx, cy, cx_mm, depth_mm, r_mm = met
            if (np.isfinite(r_mm) and r_mm > MAX_VESSEL_R_MM) or not (lo <= cx <= hi):
                stats["guard"] += 1; continue
            if probs is not None and k < len(probs) and float(probs[k]) < PROB_MIN:
                stats["prob"] += 1; continue
            morph = morph_gate(m, gray, frame_med)
            if morph is None:
                stats["gate"] += 1; continue
            stats[morph] += 1
            sb = None
            if sam_boxes is not None and k < len(sam_boxes):
                b = sam_boxes[k]
                sb = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
            detections.append(dict(frame_index=i, stem=jp.stem, inst=k,
                                   cx=cx, cy=cy, cx_mm=cx_mm, depth_mm=depth_mm,
                                   r_mm=r_mm, sam_box=sb, morph=morph,
                                   carried=bool(carry_age > 0),
                                   prob=(float(probs[k]) if probs is not None
                                         and k < len(probs) else None)))

    predictor.handle_request(request=dict(type="close_session",
                                          session_id=session_id))

    frames_with = len({d["frame_index"] for d in detections})
    print(f"\n{len(detections)} detections across {frames_with}/{n} frames")
    print(f"  morph: {stats['open']} open, {stats['crescent']} crescent")
    print(f"  dropped: {stats['guard']} guard / {stats['prob']} low-prob / "
          f"{stats['gate']} gate-fail | {stats['no_prompt']} frames unprompted, "
          f"{stats['carried']} carried")
    rr = np.array([d["r_mm"] for d in detections if np.isfinite(d["r_mm"])])
    if rr.size:
        print(f"  radius: median {np.median(rr):.2f} mm")

    out_path = Path(args.out) if args.out else section / "sam_detections_v2.json"
    out_path.write_text(json.dumps(dict(section=section.name, n_frames=n,
                                        version=2, detections=detections),
                                   indent=2))
    print(f"wrote {out_path}")
    if not args.keep_jpg:
        shutil.rmtree(jpg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
