import json, os, numpy as np
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = _REPO_ROOT / "data"
rows = sorted([json.loads(l) for l in open(os.path.join(DATA, "pose_logs", "old_probe_pose_log.jsonl"))], key=lambda r: r["t_ns"])
id0 = [r for r in rows if r.get("id")==0]
t0 = id0[0]["t_ns"]
TRIM_S = 28.5                                   # cut the lift-off (move to where pos ramps)
insweep = [r for r in id0 if (r["t_ns"]-t0)/1e9 <= TRIM_S]
rx_med = np.median([r["coords"][3] for r in insweep])
kept = [r for r in insweep if abs(r["coords"][3] - rx_med) < 30]   # drop rx flips
with open(os.path.join(DATA, "pose_logs", "sweep_clean.jsonl"), "w") as f:
    for r in kept: f.write(json.dumps(r) + "\n")
print(f"kept {len(kept)}/{len(id0)}  (trim >{TRIM_S}s + drop rx flips, baseline rx={rx_med:.0f})")