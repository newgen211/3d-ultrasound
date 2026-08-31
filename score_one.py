#!/usr/bin/env python3
"""score_one.py CONF MODEL.pt [LABEL] — one checkpoint vs open audit truth,
reusing score_models.py's score()."""
import sys
import score_models as sm
conf = float(sys.argv[1]); path = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else path
sm.CONF = conf
per = {}
P, R, F, D, tp, fp, fn = sm.score(path, "open", per)
print(f"{label} vs open truth:  P {P:.3f}  R {R:.3f}  F1 {F:.3f}")
for s in sorted(per): print(f"  {s:<14} {per[s]}")
