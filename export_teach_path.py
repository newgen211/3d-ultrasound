#!/usr/bin/env python3
"""export_teach_path.py — taught sweep path(s) -> .ply/.npz in cobot base frame (mm)."""
import argparse, json
from pathlib import Path
import numpy as np

COLORS = [(230, 60, 90), (60, 130, 230), (60, 200, 120),
          (240, 180, 40), (180, 90, 220)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="teach_paths")
    args = ap.parse_args()

    skills = {}
    for lp in args.logs:
        rows = [json.loads(l) for l in open(lp) if l.strip()]
        rows = [r for r in rows if r.get("coords")]
        if not rows:
            print(f"!! {lp}: no coords — skipped"); continue
        t = np.array([r["t_ns"] for r in rows], float)
        order = np.argsort(t)
        C = np.array([rows[i]["coords"] for i in order], float)
        t = (t[order] - t[order][0]) / 1e9
        name = Path(lp).stem.replace("sweep_teach_", "").replace("sweep_teach", "flat")
        skills[name] = dict(positions=C[:, :3], euler_deg=C[:, 3:6], t_s=t)
        span = C[:, :3].max(0) - C[:, :3].min(0)
        print(f"{name}: {len(C)} samples, {t[-1]:.0f}s, "
              f"extent [{span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f}] mm")

    flat = {}
    for k, v in skills.items():
        for f, a in v.items():
            flat[f"{k}__{f}"] = a
    flat["names"] = np.array(list(skills.keys()))
    np.savez(args.out + ".npz", **flat)

    total = sum(len(v["positions"]) for v in skills.values())
    with open(args.out + ".ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {total}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for si, (k, v) in enumerate(skills.items()):
            c = np.array(COLORS[si % len(COLORS)], float)
            n = len(v["positions"])
            for i, p in enumerate(v["positions"]):
                b = 0.35 + 0.65 * (i / max(n - 1, 1))
                r_, g_, b_ = (c * b).astype(int)
                f.write(f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {r_} {g_} {b_}\n")
    print(f"wrote {args.out}.ply + {args.out}.npz ({total} pts, {len(skills)} skills)")

if __name__ == "__main__":
    main()
