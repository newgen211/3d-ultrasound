#!/usr/bin/env python3
"""
compression_score.py — kappa: the graded compression scale (0 = round, 1 = shut).

Fuses three signals per vessel per frame:
  1. ASPECT RATIO vs the vessel's OWN baseline (its p90 ratio = "roundest
     self") — self-referenced, so beam bloom and vessel size cancel.
        kappa_aspect = 1 - ratio/baseline          (clipped to [0, 0.7])
     Aspect can only see down to ~detectability, hence the 0.7 cap.
  2. UNDETECTED inside a COLLAPSED zone (from detect_compression) with the
     track alive on both sides -> kappa = 0.9 (closure inferred).
  3. BRIGHT COAPTATION LINE at the last-known position (Blaze's pi*r window,
     same depth) -> kappa = 1.0 (closure CONFIRMED per-frame).
Per-frame roll-up: mean kappa over active tracks = the sweep's image-derived
force profile. Calibrated in vessel-deformation units, not newtons — rigorous
as a RELATIVE gauge (capture vs capture, region vs region).

    python3 compression_score.py section_85
Outputs: <section>/compression_score.json + compression_kappa.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from segment_tube import find_section

MIN_LEN, MAX_GAP, MAX_JUMP = 8, 25, 40      # tracking: blink-tolerant
BRIGHT_T = 1.5

def main():
    sec = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    dets = json.loads((sec / "sam_detections.json").read_text())["detections"]
    rep = json.loads((sec / "compression_report.json").read_text())
    zones = rep["collapse_zones"]; n = rep["n_frames"]
    in_zone = np.zeros(n, bool)
    for a, b in zones: in_zone[a:b+1] = True
    jpg = sec / "frames_jpg"
    meta = json.loads(next(sec.glob("raw_*.json")).read_text())["frame"]
    axial = meta["axial_um_per_sample"] / 1000.0
    lateral = meta["lateral_um_per_line"] / 1000.0
    sample = cv2.imread(str(next(jpg.glob("*.jpg"))), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape

    # ---- tracking (blink-tolerant) ----
    by_frame = {}
    for d in dets: by_frame.setdefault(d["frame_index"], []).append(d)
    tracks, active = [], []
    for f in sorted(by_frame):
        used = set()
        for d in by_frame[f]:
            if d.get("sam_box"):
                x, y, w, h = d["sam_box"]
                cx, cy = (x+w/2)*W, (y+h/2)*H
                w_mm, h_mm = w*W*lateral, h*H*axial
            else:
                cx, cy = d["cx"], d["cy"]
                r = d.get("r_mm") or 1.0
                w_mm, h_mm = 2*r*1.9, 2*r
            best, bd = None, MAX_JUMP
            for ai, (ti, lf, lx, ly) in enumerate(active):
                if ai in used or f - lf > MAX_GAP: continue
                dd = np.hypot(cx-lx, cy-ly)
                if dd < bd: best, bd = ai, dd
            if best is not None:
                ti = active[best][0]
                tracks[ti].append((f, cx, cy, w_mm, h_mm))
                active[best] = (ti, f, cx, cy); used.add(best)
            else:
                tracks.append([(f, cx, cy, w_mm, h_mm)])
                active.append((len(tracks)-1, f, cx, cy))
    tracks = [t for t in tracks if len(t) >= MIN_LEN]
    print(f"{len(tracks)} tracks (blink-tolerant: gap<={MAX_GAP}, len>={MIN_LEN})")

    # ---- per-track kappa ----
    KAPPA = np.full((len(tracks), n), np.nan)
    # global baseline: the uncompressed reference capture (same phantom/probe)
    base_global = None
    import os
    ref = os.environ.get("KAPPA_REF", "section_81")
    try:
        rsec = find_section(ref)
        rdets = json.loads((rsec / "sam_detections.json").read_text())["detections"]
        ras = []
        for d in rdets:
            if d.get("sam_box"):
                _, _, w, h = d["sam_box"]
                ras.append((h*H*axial) / max(w*W*lateral, 1e-3))
        if len(ras) > 50:
            base_global = float(np.percentile(ras, 75))
            print(f"baseline from {ref}: aspect p75 = {base_global:.2f} "
                  f"({len(ras)} reference detections)")
    except SystemExit:
        pass

    for k, t in enumerate(tracks):
        fr = np.array([p[0] for p in t])
        ar = np.array([p[4]/max(p[3], 1e-3) for p in t])
        base = base_global or float(np.percentile(ar, 90)) or 1.0
        kap = np.clip(1 - ar/base, 0, 0.7)
        KAPPA[k, fr] = kap
        # gaps inside the track span: inferred/confirmed closure
        span0, span1 = fr[0], fr[-1]
        have = np.zeros(n, bool); have[fr] = True
        f = span0
        while f <= span1:
            if not have[f]:
                g0 = f
                while f <= span1 and not have[f]: f += 1
                g1 = f - 1
                if in_zone[g0:g1+1].any():
                    # bright-line probe at last-known position (pi*r window)
                    pre = t[np.searchsorted(fr, g0) - 1]
                    _, cx, cy, w_mm, h_mm = pre
                    r_mm = h_mm / 2.0
                    hw = (np.pi * r_mm) / lateral / 2.0
                    x0, x1 = int(max(0, cx-hw)), int(min(W, cx+hw))
                    y0, y1 = int(max(0, cy-6)), int(min(H, cy+6))
                    scores = []
                    for ff in range(g0, min(g1+1, g0+10)):
                        p = jpg / f"{ff:05d}.jpg"
                        if not p.exists(): continue
                        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        med = float(np.median(im)) or 1.0
                        band = im[y0:y1, x0:x1]
                        if band.size:
                            scores.append(float(band.mean(axis=1).max())/med)
                    bright = scores and np.median(scores) > BRIGHT_T
                    KAPPA[k, g0:g1+1] = 1.0 if bright else 0.9
                else:
                    # benign gap: interpolate
                    a_, b_ = KAPPA[k, g0-1], kap[np.searchsorted(fr, g1+1)] \
                             if np.searchsorted(fr, g1+1) < len(kap) else KAPPA[k, g0-1]
                    KAPPA[k, g0:g1+1] = np.linspace(a_, b_, g1-g0+1)
            else:
                f += 1

    frame_kappa = np.nanmean(KAPPA, axis=0)
    ok = ~np.isnan(frame_kappa)
    print(f"frames scored: {ok.sum()}/{n}  "
          f"median kappa {np.nanmedian(frame_kappa):.2f}  "
          f"p90 {np.nanpercentile(frame_kappa, 90):.2f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    for a, b in zones: ax.axvspan(a, b, color="crimson", alpha=0.15)
    ax.plot(np.where(ok)[0], frame_kappa[ok], lw=1.2, color="k")
    ax.set_ylim(0, 1.05); ax.set_ylabel("kappa (0 round -> 1 shut)")
    ax.set_xlabel("frame")
    ax.axhline(0.7, color="gray", ls="--", lw=0.6)
    ax.set_title(f"{sec.name} — graded compression (image-derived force profile)")
    fig.tight_layout(); fig.savefig(sec / "compression_kappa.png", dpi=130)

    (sec / "compression_score.json").write_text(json.dumps(dict(
        section=sec.name, n_tracks=len(tracks),
        frame_kappa=[None if np.isnan(v) else round(float(v), 3)
                     for v in frame_kappa]), indent=2))
    print(f"wrote compression_score.json + compression_kappa.png")

if __name__ == "__main__":
    main()