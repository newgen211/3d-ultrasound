#!/usr/bin/env python
"""
view_sweep.py — visualize a completed sweep

Reads a data/clarius_sessions/section_<N>/ folder, decodes each raw_<ts>.bin into
an image, and shows them alongside their IMU data.

Usage:
    python view_sweep.py                          # picks the latest section
    python view_sweep.py section_3                # specific section by name
    python view_sweep.py data/clarius_sessions/section_3   # or full path

Outputs:
    A PNG montage of all frames in the sweep, with IMU quaternion as text.
    Saves to <section_dir>/preview_montage.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame
from us3d.sections import find_section


def main():
    section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"📂 Reading {section}")

    # find all raw frames
    jsons = sorted(section.glob("raw_*.json"))[::max(1, len(list(section.glob("raw_*.json"))) // 40)]
    if not jsons:
        print("❌ No raw_*.json files found")
        sys.exit(1)
    print(f"   {len(jsons)} frames")

    # load them
    frames = []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            print(f"  ⚠️  Missing {bp.name}")
            continue
        try:
            img = load_frame(bp, meta)
        except Exception as e:
            print(f"  ⚠️  Failed to decode {bp.name}: {e}")
            continue
        frames.append((meta, img, bp.name))

    if not frames:
        print("❌ No frames decoded")
        sys.exit(1)

    # build a montage — N frames in a grid
    n = len(frames)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5), squeeze=False)
    fig.suptitle(f"Sweep: {section.name}  ({n} frames)", fontsize=14, fontweight="bold")

    # first frame timestamp for relative time display
    t0 = frames[0][0]["probe_timestamp_ns"]

    for i, (meta, img, name) in enumerate(frames):
        ax = axes[i // cols][i % cols]
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])

        # title: relative time + frame size
        t_rel_ms = (meta["probe_timestamp_ns"] - t0) / 1e6
        n_imu = meta.get("imu_sample_count", 0)
        f = meta["frame"]
        title = f"t={t_rel_ms:+.0f} ms  ({f['lines']}×{f['samples']})  imu={n_imu}"

        # quaternion from first IMU sample if available
        if meta.get("imu_samples"):
            s = meta["imu_samples"][0]
            qw, qx, qy, qz = s.get("qw"), s.get("qx"), s.get("qy"), s.get("qz")
            if None not in (qw, qx, qy, qz):
                title += f"\nq=({qw:+.2f},{qx:+.2f},{qy:+.2f},{qz:+.2f})"
        ax.set_title(title, fontsize=8)

    # hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout()
    out = section / "preview_montage.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"💾 Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()