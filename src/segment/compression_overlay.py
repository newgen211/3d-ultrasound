#!/usr/bin/env python3
"""
compression_overlay.py — SHOW the compression: frames with each vessel boxed,
colored green(round) -> red(shut), labeled with its kappa %.

    python3 compression_overlay.py section_85
Writes <section>/compression_overlay.png (presentation contact sheet).
"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.sections import find_section

MIN_LEN, MAX_GAP, MAX_JUMP = 8, 25, 40

def kcolor(k):     # green -> yellow -> red
    k = float(np.clip(k, 0, 1))
    return (0, int(255*(1-k*0.5)), int(255*k)) if k < 0.5 else \
           (0, int(255*(1-k)), 255)

def main():
    sec = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    dets = json.loads((sec/"sam_detections.json").read_text())["detections"]
    rep = json.loads((sec/"compression_report.json").read_text())
    zones = rep["collapse_zones"]; n = rep["n_frames"]
    in_zone = np.zeros(n, bool)
    for a, b in zones: in_zone[a:b+1] = True
    jpg = sec/"frames_jpg"
    meta = json.loads(next(sec.glob("raw_*.json")).read_text())["frame"]
    axial = meta["axial_um_per_sample"]/1000.0
    lateral = meta["lateral_um_per_line"]/1000.0
    sample = cv2.imread(str(next(jpg.glob("*.jpg"))), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape

    # tracking + kappa (same core as compression_score)
    by_frame = {}
    for d in dets: by_frame.setdefault(d["frame_index"], []).append(d)
    tracks, active = [], []
    for f in sorted(by_frame):
        used = set()
        for d in by_frame[f]:
            if d.get("sam_box"):
                x, y, w, h = d["sam_box"]
                cx, cy = (x+w/2)*W, (y+h/2)*H
                w_px, h_px = w*W, h*H
                w_mm, h_mm = w*W*lateral, h*H*axial
            else:
                cx, cy = d["cx"], d["cy"]
                r = d.get("r_mm") or 1.0
                w_mm, h_mm = 2*r*1.9, 2*r
                w_px, h_px = w_mm/lateral, h_mm/axial
            best, bd = None, MAX_JUMP
            for ai, (ti, lf, lx, ly) in enumerate(active):
                if ai in used or f-lf > MAX_GAP: continue
                dd = np.hypot(cx-lx, cy-ly)
                if dd < bd: best, bd = ai, dd
            rec = (f, cx, cy, w_mm, h_mm, w_px, h_px)
            if best is not None:
                ti = active[best][0]; tracks[ti].append(rec)
                active[best] = (ti, f, cx, cy); used.add(best)
            else:
                tracks.append([rec]); active.append((len(tracks)-1, f, cx, cy))
    tracks = [t for t in tracks if len(t) >= MIN_LEN]

    # global baseline from reference capture
    base = 0.73
    try:
        ref = find_section(os.environ.get("KAPPA_REF", "section_81"))
        ras = []
        for d in json.loads((ref/"sam_detections.json").read_text())["detections"]:
            if d.get("sam_box"):
                _, _, w, h = d["sam_box"]
                ras.append((h*H*axial)/max(w*W*lateral, 1e-3))
        if len(ras) > 50: base = float(np.percentile(ras, 75))
    except SystemExit:
        pass

    per_frame = {}    # frame -> list of (box_px, kappa)
    for t in tracks:
        for (f, cx, cy, w_mm, h_mm, w_px, h_px) in t:
            k = float(np.clip(1 - (h_mm/max(w_mm, 1e-3))/base, 0, 0.95))
            per_frame.setdefault(f, []).append(((cx, cy, w_px, h_px), k))

    # ghost positions: where each vessel last stood before each collapse zone
    ghosts = {}          # frame -> list of (cx, cy, half_width_px)
    for a, b in zones:
        for t in tracks:
            pre = [p for p in t if p[0] < a]
            if pre and a - pre[-1][0] < 80:
                _, cx, cy, w_mm, h_mm, _, _ = pre[-1]
                hw = (np.pi * (h_mm / 2.0)) / lateral / 2.0     # pi*r ribbon
                for f in range(a, b + 1):
                    ghosts.setdefault(f, []).append((cx, cy, hw))

    # prefer zone frames that HAVE ghosts, so the story shows
    zone_picks = [f for f in sorted(ghosts) if in_zone[f]][:2] or [110, 250]
    picks = [f for f in ([24, 60] + zone_picks + [470, 520, 650, 725]) if f < n]
    tiles = []
    for f in picks:
        p = jpg/f"{f:05d}.jpg"
        if not p.exists(): continue
        im = cv2.imread(str(p))
        for (cx, cy, w_px, h_px), k in per_frame.get(f, []):
            p1 = (int(cx-w_px/2), int(cy-h_px/2)); p2 = (int(cx+w_px/2), int(cy+h_px/2))
            cv2.rectangle(im, p1, p2, kcolor(k), 2)
            cv2.putText(im, f"{int(k*100)}%", (p1[0], max(14, p1[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, kcolor(k), 2)
        for (gx, gy, hw) in ghosts.get(f, []):
            cv2.line(im, (int(gx - hw), int(gy)), (int(gx + hw), int(gy)),
                     (0, 0, 255), 3)
            cv2.putText(im, "vessel here - SHUT", (int(gx - hw), int(gy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 2)
        banner = "COLLAPSED ZONE" if in_zone[f] else \
                 (f"mean {int(100*np.mean([k for _, k in per_frame[f]]))}% compressed"
                  if per_frame.get(f) else "no vessels tracked")
        col = (0, 0, 255) if in_zone[f] else (255, 255, 255)
        cv2.putText(im, f"f{f}  {banner}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        s = 250/im.shape[1]
        tiles.append(cv2.resize(im, (250, int(im.shape[0]*s))))
    th = max(t.shape[0] for t in tiles)
    sheet = np.full((th+56, 254*len(tiles), 3), 15, np.uint8)
    for i, t in enumerate(tiles):
        sheet[48:48+t.shape[0], i*254+2:i*254+2+t.shape[1]] = t
    cv2.putText(sheet, f"{sec.name} — per-vessel compression (kappa vs gentle-contact "
                f"baseline {base:.2f}) : green=round, red=compressed",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(sec/"compression_overlay.png"), sheet)
    print(f"wrote {sec/'compression_overlay.png'}  ({len(tiles)} frames)")

if __name__ == "__main__":
    main()