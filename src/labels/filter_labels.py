#!/usr/bin/env python3
"""SAM label filter: center-dedupe + temporal-track corroboration.
Reads one detections file and writes the filtered one. Pure numpy.

    python3 src/labels/filter_labels.py section_90              # v5 -> v5f
    python3 src/labels/filter_labels.py section_90 --in v2 --out v4

The --in/--out versions absorb the former filter_labels_v4.py, which was this
same code wired to a different pair."""
import argparse
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.sections import find_section
import numpy as np

DEDUP_PX = 18        # centers closer than this = same vessel, keep best prob
TRACK_PX = 25        # frame-to-frame association radius
MIN_TRACK = 5        # detections must belong to a track this long
MAX_GAP = 3          # blink tolerance inside a track
R_MIN_MM = 0.7       # fragment-mask floor: drop slivers smaller than half the target vessel

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("sections", nargs="+")
ap.add_argument("--in", dest="in_ver", default="v5", help="input detections version (default v5)")
ap.add_argument("--out", dest="out_ver", default="v5f", help="output version (default v5f)")
args = ap.parse_args()
IN_VER, OUT_VER = args.in_ver, args.out_ver

for sec_name in args.sections:
    sec = find_section(sec_name)
    dd = json.loads((sec / ("sam_detections_%s.json" % IN_VER)).read_text())
    dets = dd["detections"]

    # -- pass 1: per-frame center dedupe
    by_frame = {}
    n_tiny = len([d for d in dets if d.get("r_mm") and d["r_mm"] < R_MIN_MM])
    dets = [d for d in dets if not (d.get("r_mm") and d["r_mm"] < R_MIN_MM)]
    for d in dets:
        by_frame.setdefault(d["frame_index"], []).append(d)
    deduped = {}
    n_dup = 0
    for fi, ds in by_frame.items():
        ds = sorted(ds, key=lambda d: -d.get("prob", 0))
        kept = []
        for d in ds:
            if any((d["cx"]-k["cx"])**2 + (d["cy"]-k["cy"])**2 < DEDUP_PX**2 for k in kept):
                n_dup += 1
                continue
            kept.append(d)
        deduped[fi] = kept

    # -- pass 2: greedy temporal tracks with blink tolerance
    tracks = []          # each: {"last_fi", "cx", "cy", "members":[det,...]}
    for fi in sorted(deduped):
        for d in deduped[fi]:
            best = None
            for t in tracks:
                if fi - t["last_fi"] > MAX_GAP + 1:
                    continue
                if (d["cx"]-t["cx"])**2 + (d["cy"]-t["cy"])**2 < TRACK_PX**2:
                    best = t
                    break
            if best is None:
                tracks.append({"last_fi": fi, "cx": d["cx"], "cy": d["cy"], "members": [d]})
            else:
                best["last_fi"] = fi
                best["cx"], best["cy"] = d["cx"], d["cy"]
                best["members"].append(d)

    survivors = []
    n_flicker = 0
    for t in tracks:
        if len(t["members"]) >= MIN_TRACK:
            survivors.extend(t["members"])
        else:
            n_flicker += len(t["members"])

    survivors.sort(key=lambda d: (d["frame_index"], -d.get("prob", 0)))
    out = dict(dd); out["detections"] = survivors
    (sec / ("sam_detections_%s.json" % OUT_VER)).write_text(json.dumps(out))
    print(f"{sec_name}: {len(dets)} -> {len(survivors)}  "
          f"(-{n_tiny} tiny, -{n_dup} duplicates, -{n_flicker} flickers)")
