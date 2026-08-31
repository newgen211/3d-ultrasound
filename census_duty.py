import glob, json, sys
from pathlib import Path
sys.path.insert(0, "src/segment")
from segment_tube import candidates, load_frame

for sec in ["section_106", "section_108", "section_109", "section_111"]:
    bins = sorted(glob.glob(f"data/clarius_sessions/{sec}/*.bin"))
    hit = tot = 0
    for b in bins:
        jp = b.replace(".bin", ".json")
        if not Path(jp).exists():
            continue
        meta_full = json.load(open(jp))
        meta = meta_full["frame"]
        img = load_frame(Path(b), meta_full)
        axial = meta["axial_um_per_sample"] / 1000.0
        lateral = meta["lateral_um_per_line"] / 1000.0
        tot += 1
        if candidates(img, axial, lateral):
            hit += 1
    if tot:
        print(f"{sec}: {hit}/{tot} = {100*hit/tot:.0f}% duty")
    else:
        print(sec, "no frames")
