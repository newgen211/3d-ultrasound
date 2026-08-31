import glob, json, sys
from pathlib import Path
sys.path.insert(0, "src/segment")
from segment_tube import candidates, load_frame

for sec in ["section_106", "section_108", "section_109", "section_111"]:
    bins = sorted(glob.glob(f"data/clarius_sessions/{sec}/*.bin"))
    vis = []
    for b in bins:
        jp = b.replace(".bin", ".json")
        if not Path(jp).exists():
            continue
        meta_full = json.load(open(jp))
        meta = meta_full["frame"]
        img = load_frame(Path(b), meta_full)
        vis.append(1 if candidates(img, meta["axial_um_per_sample"]/1000.0,
                                   meta["lateral_um_per_line"]/1000.0) else 0)
    n = len(vis)
    d = n // 10
    deciles = " ".join(f"{100*sum(vis[i*d:(i+1)*d])/d:3.0f}" for i in range(10))
    # longest consecutive miss run
    gap = mx = 0
    for v in vis:
        gap = 0 if v else gap + 1
        mx = max(mx, gap)
    print(f"{sec}: deciles% [{deciles}]  longest_miss_run={mx} frames")
