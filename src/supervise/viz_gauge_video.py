"""Overlay video of a gauge replay: the ROI box coloured by state."""
import json, glob, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame, to_u8
from us3d.sections import find_section
from us3d.video import finalize, fit_frame, open_writer

sec = find_section(sys.argv[1] if len(sys.argv) > 1 else "section_106")
rows = {r["frame"]: r for r in
        (json.loads(l) for l in open(sec / "gauge_replay.jsonl"))}
raws = sorted(glob.glob(str(sec / "raw_*.json")))
COL = {"GOOD": (80, 220, 80), "SQUEEZING": (0, 200, 255),
       "PRESSED": (60, 60, 255), "WASHED": (255, 200, 0),
       "MIGRATE": (255, 0, 255), "ACQ": (160, 160, 160), "LOST": (0, 0, 0)}
mf0 = json.load(open(raws[0]))
u80 = to_u8(load_frame(Path(raws[0].replace(".json", ".bin")), mf0))
H, W = u80.shape

vw, outpath = open_writer("gauge_%s" % sec.name, (W, H), fps=20)

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
    vw.write(fit_frame(fr, (W, H)))
finalize(vw, outpath)
print("wrote %s  (%.1f MB, %d frames)"
      % (outpath, outpath.stat().st_size / 1e6, len(raws)))
