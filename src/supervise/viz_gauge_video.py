import json, glob, sys, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8

sec = sys.argv[1] if len(sys.argv) > 1 else "section_106"
rows = {r["frame"]: r for r in
        (json.loads(l) for l in open(f"data/clarius_sessions/{sec}/gauge_replay.jsonl"))}
raws = sorted(glob.glob(f"data/clarius_sessions/{sec}/raw_*.json"))
COL = {"GOOD": (80, 220, 80), "SQUEEZING": (0, 200, 255),
       "PRESSED": (60, 60, 255), "WASHED": (255, 200, 0),
       "MIGRATE": (255, 0, 255), "ACQ": (160, 160, 160), "LOST": (0, 0, 0)}
mf0 = json.load(open(raws[0]))
u80 = to_u8(load_frame(Path(raws[0].replace(".json", ".bin")), mf0))
H, W = u80.shape

vw, outname = None, None
for name, fourcc in [(f"gauge_{sec}.mp4", "avc1"),
                     (f"gauge_{sec}.mp4", "mp4v"),
                     (f"gauge_{sec}.avi", "MJPG")]:
    cand = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*fourcc), 20, (W, H))
    if cand.isOpened():
        vw, outname = cand, name
        break
    cand.release()
if vw is None:
    sys.exit("no working codec — tell Claude")

for i, jp in enumerate(raws):
    mf = json.load(open(jp))
    u8 = to_u8(load_frame(Path(jp.replace(".json", ".bin")), mf))
    fr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    r = rows.get(i)
    if r:
        c = COL.get(r["state"], (200, 200, 200))
        if r["cx"] is not None:
            cv2.rectangle(fr, (int(r["cx"]-r["ww"]), int(r["cy"]-r["hw"])),
                          (int(r["cx"]+r["ww"]), int(r["cy"]+r["hw"])), c, 2)
        txt = f"f{i} {r['state']}" + \
              (f" r={r['ratio']}" if r["ratio"] is not None else "") + \
              (f"  [{r['truth']}]" if r["truth"] else "")
        cv2.putText(fr, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    vw.write(fr)
vw.release()
sz = os.path.getsize(outname)
if sz < 100_000:
    sys.exit(f"{outname} only {sz} bytes — writer produced nothing, tell Claude")
print(f"wrote {os.path.abspath(outname)}  ({sz/1e6:.1f} MB, {len(raws)} frames)")
