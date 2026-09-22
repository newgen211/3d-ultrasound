import json, glob, sys, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "src/segment")
from segment_tube import load_frame, to_u8

sec_name = sys.argv[1]
sec = Path(f"data/clarius_sessions/{sec_name}")
raws = sorted(glob.glob(str(sec / "raw_*.json")))
dets = json.loads((sec / "sam_detections.json").read_text())["detections"]
byf = {}
for d in dets:
    byf.setdefault(d["frame_index"], []).append(d)
ks = json.loads((sec / "compression_score.json").read_text())
kappa = {i: v for i, v in enumerate(ks.get("frame_kappa", []))}

mf0 = json.load(open(raws[0]))
u80 = to_u8(load_frame(Path(raws[0].replace(".json", ".bin")), mf0))
H, W = u80.shape
vw, outname = None, None
for name, fcc in [(f"kappa_{sec_name}.mp4", "avc1"),
                  (f"kappa_{sec_name}.mp4", "mp4v"),
                  (f"kappa_{sec_name}.avi", "MJPG")]:
    c = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*fcc), 20, (W, H))
    if c.isOpened(): vw, outname = c, name; break
    c.release()
if vw is None: sys.exit("no codec")

for i, jp in enumerate(raws):
    mf = json.load(open(jp))
    u8 = to_u8(load_frame(Path(jp.replace(".json", ".bin")), mf))
    fr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    for d in byf.get(i, []):
        bx = d.get("sam_box")
        if bx:
            x0, y0, x1, y1 = [int(v) for v in bx]
            cv2.rectangle(fr, (x0, y0), (x1, y1), (80, 220, 80), 2)
    k = kappa.get(i)
    if k is not None and not (isinstance(k, float) and np.isnan(k)):
        bw = int(k * (W - 20))
        col = (60, 60, 255) if k > 0.5 else (0, 200, 255) if k > 0.2 else (80, 220, 80)
        cv2.rectangle(fr, (10, H-18), (10+bw, H-8), col, -1)
        cv2.putText(fr, f"k={k:.2f}", (10, H-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.putText(fr, f"{sec_name} f{i}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    vw.write(fr)
vw.release()
sz = os.path.getsize(outname)
if sz < 100_000: sys.exit(f"{outname} empty — tell Claude")
print(f"wrote {os.path.abspath(outname)} ({sz/1e6:.1f} MB)")
