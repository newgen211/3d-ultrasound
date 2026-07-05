#!/usr/bin/env python3
"""
segment_tube.py — find AND track the anechoic vessel in B-mode (texture-aware)

Detection: the vessel is dark AND smooth (anechoic = no speckle), while tissue
is dark but speckled. Threshold on low local-variance + low intensity, fill
reflection holes, keep blobs that are SOLID and not-too-elongated (round AND
obliquely-cut vessels survive). Radius = ellipse MINOR axis (true radius on an
oblique cut; the area-radius over-reads).

Tracking (gated, bidirectional): there can be several smooth-dark blobs per
frame. We estimate the vessel's depth/lateral/size trajectory in a FORWARD pass
and a BACKWARD pass, fuse them, then per frame accept only the candidate that
falls inside a gate around that trajectory. A candidate that is off-region or
off-size is rejected -> the frame is a clean MISS instead of a wrong point
(a wrong point plants an outlier; a miss just gets bridged by smoothing).

    python3 segment_tube.py            # latest section
    python3 segment_tube.py section_50

Output: tube_seg_overlay.png in the section folder.
"""

import json
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- detection tunables ---
MIN_R_MM, MAX_R_MM = 0.6, 6.0
MIN_SOLIDITY = 0.80      # contour area / convex-hull area (drops fragmented/concave noise)
MAX_ELONG = 4.0          # major/minor axis ratio cap (drops wall-reflection slivers)
TEX_WIN = 9
SMOOTH_PCT = 22
DARK_PCT = 38
OPEN_K = 5               # morphological open kernel (despeckle)
CLOSE_K = 7              # morphological close kernel (bridge reflection notches)
TOP_CROP_MM = 3.0
BOT_CROP_MM = 2.0
# --- tracking tunables ---
DEPTH_SIGMA_MM = 3.0     # soft weighting for "near the vessel's depth"
LAT_SIGMA_MM = 7.0       # soft weighting for lateral drift frame-to-frame
R_SIGMA_MM = 0.8         # soft weighting for size consistency
DEPTH_GATE_MM = 5.0      # HARD reject: candidate farther than this in depth -> not the vessel
LAT_GATE_MM = 10.0       # HARD reject: lateral gate around the tracked position
R_GATE_MM = 1.2          # HARD reject: size gate around the tracked radius
EMA = 0.30               # trajectory responsiveness
SAMPLE = 12


def find_section(arg):
    root = Path(__file__).resolve().parents[2] / "data" / "clarius_sessions"
    if arg:
        return Path(arg) if Path(arg).exists() else root / arg
    secs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("section_")],
                  key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
    return secs[-1]


def load_frame(bin_path, meta):
    f = meta["frame"]
    lines, samples, bps = f["lines"], f["samples"], f["bps"]
    jpg = f.get("jpg_size", 0)
    raw = bin_path.read_bytes()
    if jpg > 0:
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(raw)).convert("L")).astype(np.float32)
    dtype = np.uint8 if bps == 8 else np.uint16
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size != lines * samples:
        usable = (arr.size // lines) * lines
        arr = arr[:usable]
        samples = usable // lines
    return arr.reshape(lines, samples).T.astype(np.float32)


def to_u8(img):
    g = img - img.min()
    return (255 * g / max(1.0, g.max())).astype(np.uint8)


def fill_holes(mask):
    """Fill enclosed holes (bright reflections inside the dark void)."""
    h, w = mask.shape
    inv = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 255)          # flood background from a corner
    holes = cv2.bitwise_not(inv)                 # background that couldn't be reached = holes
    return mask | holes


def candidates(img, axial_mm, lateral_mm):
    """All valid smooth-dark blobs in one frame (no tracking yet)."""
    H, W = img.shape
    g = cv2.GaussianBlur(to_u8(img).astype(np.float32), (3, 3), 0)
    k = TEX_WIN
    mean = cv2.blur(g, (k, k))
    std = np.sqrt(np.clip(cv2.blur(g * g, (k, k)) - mean * mean, 0, None))
    top, bot = int(TOP_CROP_MM / axial_mm), H - int(BOT_CROP_MM / axial_mm)
    reg = np.zeros((H, W), bool); reg[top:max(top + 1, bot), :] = True
    mask = ((std < np.percentile(std[reg], SMOOTH_PCT)) &
            (mean < np.percentile(mean[reg], DARK_PCT)) & reg).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K)))
    mask = fill_holes(mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = np.array([lateral_mm, axial_mm], np.float32)
    out = []
    for c in cnts:
        if len(c) < 5:
            continue
        cmm = c.astype(np.float32) * scale
        area = cv2.contourArea(cmm)
        if area <= 0:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-9
        solidity = cv2.contourArea(c) / hull_area     # ratio: anisotropy cancels, use px
        if solidity < MIN_SOLIDITY:
            continue
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(c)        # pixels: drawing + centroid
        (_, _), (e1, e2), _ = cv2.fitEllipse(cmm)          # mm: true axis lengths
        minor, major = min(e1, e2), max(e1, e2)
        r_mm = minor / 2.0                                 # minor axis = true vessel radius
        if not (MIN_R_MM <= r_mm <= MAX_R_MM):
            continue
        elong = major / (minor + 1e-9)
        if elong > MAX_ELONG:                              # sliver / wall streak
            continue
        per = cv2.arcLength(cmm, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0
        out.append(dict(cx=cx, cy=cy, r_mm=r_mm, circ=circ, solidity=solidity,
                        elong=elong, quality=area * solidity,
                        cx_mm=(cx - W / 2) * lateral_mm, depth_mm=cy * axial_mm,
                        ellipse=((cx, cy), (d1, d2), ang)))
    return out


# ---------- gated bidirectional tracker ----------

def _ndist(d, td, tc, tr):
    """soft normalized distance from a candidate to the tracked (depth,cx,r)."""
    return (((d["depth_mm"] - td) / DEPTH_SIGMA_MM) ** 2 +
            ((d["cx_mm"] - tc) / LAT_SIGMA_MM) ** 2 +
            ((d["r_mm"] - tr) / R_SIGMA_MM) ** 2)


def _in_gate(d, td, tc, tr):
    """hard accept window around the tracked region + size."""
    return (abs(d["depth_mm"] - td) <= DEPTH_GATE_MM and
            abs(d["cx_mm"] - tc) <= LAT_GATE_MM and
            abs(d["r_mm"] - tr) <= R_GATE_MM)


def track(frames_cands):
    """Forward + backward gated tracking; fuse; final gated pick per frame."""
    N = len(frames_cands)
    best = [max(c, key=lambda d: d["quality"]) if c else None for c in frames_cands]
    a_d = float(np.nanmedian([b["depth_mm"] if b else np.nan for b in best]))
    a_c = float(np.nanmedian([b["cx_mm"] if b else np.nan for b in best]))
    a_r = float(np.nanmedian([b["r_mm"] if b else np.nan for b in best]))
    if not np.isfinite(a_d):
        return [None] * N

    def run(order):
        td, tc, tr = a_d, a_c, a_r
        est = np.full((N, 3), np.nan)
        for i in order:
            gated = [d for d in frames_cands[i] if _in_gate(d, td, tc, tr)]
            if gated:
                b = min(gated, key=lambda d: _ndist(d, td, tc, tr))
                td = (1 - EMA) * td + EMA * b["depth_mm"]
                tc = (1 - EMA) * tc + EMA * b["cx_mm"]
                tr = (1 - EMA) * tr + EMA * b["r_mm"]
            est[i] = (td, tc, tr)                  # update if matched, else coast
        return est

    fwd = run(range(N))
    bwd = run(range(N - 1, -1, -1))
    traj = np.nanmean(np.stack([fwd, bwd]), axis=0)   # fuse both directions

    picks = []
    for i in range(N):
        td, tc, tr = traj[i]
        gated = [d for d in frames_cands[i] if _in_gate(d, td, tc, tr)]
        picks.append(min(gated, key=lambda d: _ndist(d, td, tc, tr)) if gated else None)
    return picks


def main():
    section = find_section(sys.argv[1] if len(sys.argv) > 1 else None)
    jsons = sorted(Path(section).glob("raw_*.json"))
    if not jsons:
        sys.exit("no frames")
    f0 = json.loads(jsons[0].read_text())["frame"]
    axial_mm = f0["axial_um_per_sample"] / 1000.0
    lateral_mm = f0["lateral_um_per_line"] / 1000.0
    print(f"{Path(section).name}: {len(jsons)} frames, "
          f"scale {axial_mm:.3f} (axial) x {lateral_mm:.3f} (lateral) mm/px")

    # ---- pass 1: candidates per frame ----
    frames = []  # (json_path, [candidates])
    for jp in jsons:
        meta = json.loads(jp.read_text())
        bp = jp.with_suffix(".bin")
        if not bp.exists():
            frames.append((jp, []))
            continue
        try:
            cands = candidates(load_frame(bp, meta), axial_mm, lateral_mm)
        except Exception:
            cands = []
        frames.append((jp, cands))

    # ---- pass 2: gated bidirectional track ----
    picks = track([c for _, c in frames])

    hits = sum(p is not None for p in picks)
    radii = [p["r_mm"] for p in picks if p]
    n = len(frames)
    print(f"detected in {hits}/{n} frames ({100*hits/n:.0f}%)")
    if radii:
        r = np.array(radii)
        print(f"radius: median {np.median(r):.2f} mm  "
              f"(p10 {np.percentile(r,10):.2f}, p90 {np.percentile(r,90):.2f})  "
              f"-> diameter ~{2*np.median(r):.2f} mm")

    # ---- montage ----
    step = max(1, n // SAMPLE)
    idxs = list(range(0, n, step))[:SAMPLE]
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, i in zip(np.ravel(axs), idxs):
        jp, _ = frames[i]; det = picks[i]
        meta = json.loads(jp.read_text())
        vis = cv2.cvtColor(to_u8(load_frame(jp.with_suffix(".bin"), meta)), cv2.COLOR_GRAY2BGR)
        if det:
            cv2.ellipse(vis, det["ellipse"], (0, 255, 0), 2)
            cv2.circle(vis, (int(det["cx"]), int(det["cy"])), 3, (0, 0, 255), -1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{jp.stem}\n{('r=%.1f mm' % det['r_mm']) if det else 'MISS'}", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axs)[len(idxs):]:
        ax.axis("off")
    out = Path(section) / "tube_seg_overlay.png"
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()