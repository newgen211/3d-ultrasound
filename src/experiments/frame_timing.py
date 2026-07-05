#!/usr/bin/env python3
"""
frame_timing.py — how fast were frames actually arriving?

Reads host_timestamp_ns from a recorded section's sidecars and reports the real
frame rate, jitter, and hiccups. No hardware needed — runs on any session you
already captured. This answers: is "0.5 s per frame" a capture-rate problem?

    python3 frame_timing.py             # latest section
    python3 frame_timing.py section_22
"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2] / "data" / "clarius_sessions"
arg = sys.argv[1] if len(sys.argv) > 1 else None
if arg:
    section = Path(arg) if Path(arg).exists() else root / arg
else:
    secs = sorted([d for d in root.iterdir() if d.name.startswith("section_")],
                  key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
    section = secs[-1]

ts = []
for jp in sorted(Path(section).glob("raw_*.json")):
    t = json.loads(jp.read_text()).get("host_timestamp_ns")
    if t:
        ts.append(t)
ts = np.array(sorted(ts), float)
if len(ts) < 2:
    sys.exit("❌ need ≥2 timestamped frames")

dt_ms = np.diff(ts) / 1e6          # gap between consecutive frames, ms
fps = 1000.0 / dt_ms
med = float(np.median(dt_ms))

print(f"{Path(section).name}: {len(ts)} frames over {(ts[-1]-ts[0])/1e9:.1f} s")
print(f"  rate   median {np.median(fps):5.1f} fps   (min {fps.min():.1f}, max {fps.max():.1f})")
print(f"  gap    median {med:6.1f} ms    (min {dt_ms.min():.1f}, max {dt_ms.max():.1f})")
print(f"  hiccups: {(dt_ms > 3*med).sum()} gaps > 3x median\n")

if med > 200:
    print(f"⚠️  ~{med:.0f} ms/frame ({1000/med:.1f} fps). THIS is the bottleneck.")
    print("    B-mode normally streams 20–40 fps — this is a capture/config issue,")
    print("    not a control problem, and fixing it is free speed.")
elif med > 50:
    print(f"→ {med:.0f} ms/frame (~{1000/med:.0f} fps). Slower than ideal but usable.")
else:
    print(f"✅ {med:.0f} ms/frame (~{1000/med:.0f} fps). Capture is fast —")
    print("    your 0.5 s is downstream (cobot settle or processing), not the camera.")