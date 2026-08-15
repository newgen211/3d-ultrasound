#!/usr/bin/env python3
"""
detect_compression.py — image-derived contact/compression state, per frame.

The phantom's wall-less vessels collapse under probe pressure: where force
peaks, the anechoic lumens vanish from the image. A single frame can't
distinguish "collapsed" from "no vessel here" — but a SWEEP can: a contiguous
run of vessel-free frames FLANKED by vessel-bearing frames is a collapse zone,
not empty anatomy.

Per-frame signals:
  v  = count of clean vessel candidates (classical detector, exemplar guards)
  nb = near-field band brightness / frame median (contact-coupling proxy)

Sweep-level classification:
  OPEN       vessels visible
  COLLAPSED  vessel-free run bounded by OPEN on both sides
  EDGE       vessel-free at the sweep ends (can't adjudicate — honest unknown)

Outputs: <section>/compression_report.json + compression_timeline.png

    python3 detect_compression.py section_85
This is force-ladder rung (a): the ultrasound as its own contact sensor.
Run it on a start-lift 2.0 capture and a 3.5 capture -> the A/B is automatic.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from segment_tube import find_section, load_frame, to_u8, candidates

MAX_R_MM, EDGE_FRAC = 2.5, 0.06          # same guards as the SAM exemplar
SMOOTH = 5                                # frames, majority vote

def main():
    section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    jsons = sorted(section.glob("raw_*.json"))
    if not jsons:
        sys.exit("no frames")
    # prefer SAM detections (higher recall than the classical counter)
    sam_by_stem = None
    sp = section / "sam_detections.json"
    if sp.exists():
        dd = json.loads(sp.read_text())["detections"]
        sam_by_stem = {}
        for d in dd:
            sam_by_stem[d["stem"]] = sam_by_stem.get(d["stem"], 0) + 1
        print(f"using SAM detections for visibility ({len(dd)} dets)")
    v, nb, ds, stems = [], [], [], []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            continue
        f = meta["frame"]
        axial = f["axial_um_per_sample"] / 1000.0
        lateral = f["lateral_um_per_line"] / 1000.0
        img = load_frame(bp, meta)
        u8 = to_u8(img)
        H, W = u8.shape
        if sam_by_stem is not None:
            v.append(sam_by_stem.get(jp.stem, 0))
        else:
            cands = candidates(img, axial, lateral)
            lo, hi = EDGE_FRAC * W, (1 - EDGE_FRAC) * W
            good = [c for c in cands if c["r_mm"] <= MAX_R_MM and lo <= c["cx"] <= hi]
            v.append(len(good))
        band = u8[: max(1, int(0.15 * H)), :]
        med = float(np.median(u8)) or 1.0
        nb.append(float(band.mean()) / med)
        deep = u8[int(0.4 * H):, :]
        ds.append(float(deep.std()))          # deep structural variance
        stems.append(jp.stem)

    v = np.array(v); nb = np.array(nb); ds = np.array(ds); n = len(v)
    # smoothed visibility -> boolean OPEN
    k = SMOOTH
    vis = np.array([v[max(0, i-k//2):i+k//2+1].mean() > 0.34 for i in range(n)])

    # close visibility gaps shorter than ~1 s — contact force doesn't blink
    MAX_BLINK = 15
    i = 0
    while i < n:
        if not vis[i]:
            j = i
            while j + 1 < n and not vis[j + 1]:
                j += 1
            if (j - i + 1) <= MAX_BLINK and i > 0 and j < n - 1:
                vis[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    state = np.full(n, "EDGE", dtype=object)
    state[vis] = "OPEN"
    # vessel-free runs flanked by OPEN = COLLAPSED
    i = 0
    while i < n:
        if not vis[i]:
            j = i
            while j + 1 < n and not vis[j + 1]:
                j += 1
            if i > 0 and j < n - 1 and vis[i - 1] and vis[j + 1]:
                state[i:j + 1] = "COLLAPSED"
            i = j + 1
        else:
            i += 1

    n_open = int((state == "OPEN").sum())
    n_col = int((state == "COLLAPSED").sum())
    runs = []
    i = 0
    while i < n:
        if state[i] == "COLLAPSED":
            j = i
            while j + 1 < n and state[j + 1] == "COLLAPSED":
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    print(f"{section.name}: {n} frames — OPEN {n_open}, COLLAPSED {n_col} "
          f"({100*n_col/max(n,1):.0f}%), EDGE {n - n_open - n_col}")
    ds_open = float(np.median(ds[state == "OPEN"])) if (state == "OPEN").any() else 1.0
    for a, b in runs:
        z = float(np.median(ds[a:b+1])) / max(ds_open, 1e-6)
        kind = "COUPLING-LOSS (washed — needs MORE contact)" if z < 0.6 \
               else "COLLAPSED (structured — needs LESS force)"
        print(f"  zone {a}-{b} ({b-a+1}f): deep-structure {z:.2f}x open -> {kind}")

    out = dict(section=section.name, n_frames=n,
               collapsed_pct=round(100*n_col/max(n,1), 1),
               collapse_zones=[[int(a), int(b)] for a, b in runs],
               frames=[dict(stem=s, n_vessels=int(vv), nearband=round(float(b), 3),
                            state=str(st)) for s, vv, b, st in zip(stems, v, nb, state)])
    (section / "compression_report.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(v, lw=0.8, color="k"); ax[0].set_ylabel("vessels visible")
    for a, b in runs:
        for x in ax:
            x.axvspan(a, b, color="crimson", alpha=0.25)
    ax[1].plot(nb, lw=0.8, color="steelblue"); ax[1].set_ylabel("near-band / median")
    ax[1].set_xlabel("frame")
    fig.suptitle(f"{section.name} — compression timeline "
                 f"({n_col} collapsed frames, red)")
    fig.tight_layout()
    fig.savefig(section / "compression_timeline.png", dpi=130)
    print(f"wrote {section/'compression_report.json'} + compression_timeline.png")

if __name__ == "__main__":
    main()