#!/usr/bin/env python3
"""
capture_workspace_cloud.py — RealSense point cloud of the scan target,
expressed in the COBOT BASE FRAME. Feeds the Isaac Sim environment: spawn the
phantom where it truly sits, generate alignment targets, plan sweep lines —
all in the same coordinates the real arm executes in.

Prereq: cam_to_base.json written by guided_sweep_reader (the camera->base
similarity it already solves every run: p_base = s * R @ p_cam + t).

    sudo $(which python) capture_workspace_cloud.py \
        --transform src/calibration/cam_to_base.json \
        --out workspace_cloud

Outputs: workspace_cloud.ply (viewable in Isaac/Meshlab/CloudCompare)
         workspace_cloud.npz (points_base [N,3] mm, colors [N,3] uint8)

Crop defaults fit the phantom workspace (x 20..320, y 60..320, z 60..260 mm
in base frame) — tune with flags. Points are voxel-downsampled to --voxel mm.

Singularity note for the RL side: this cloud is WHERE things are, not HOW to
reach them. Reachability/singularity is handled the way the real system does
it — joint-space solving with FK verification (shift_joint_path) — so
targets generated from this cloud should be validated through the same gate
before anyone trusts them on hardware.
"""
import argparse, json, time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import CALIB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default=str(CALIB / "cam_to_base.json"))
    ap.add_argument("--out", default="workspace_cloud")
    ap.add_argument("--frames", type=int, default=30, help="depth frames to median")
    ap.add_argument("--voxel", type=float, default=2.0, help="downsample (mm)")
    ap.add_argument("--xmin", type=float, default=20);  ap.add_argument("--xmax", type=float, default=320)
    ap.add_argument("--ymin", type=float, default=60);  ap.add_argument("--ymax", type=float, default=320)
    ap.add_argument("--zmin", type=float, default=60);  ap.add_argument("--zmax", type=float, default=260)
    args = ap.parse_args()

    T = json.load(open(args.transform))
    if "T_cam_to_base" in T:                     # bridge.json (rigid, s=1)
        M = np.array(T["T_cam_to_base"], float)
        M = np.linalg.inv(M)      # bridge stores base->camera; we need camera->base
        R, t, s = M[:3, :3], M[:3, 3], 1.0
        print(f"cam->base loaded from BRIDGE (rigid, s=1, "
              f"median_err {T.get('median_err_mm', '?'):.1f} mm)")
        print("!! bridge transform is a SNAPSHOT — only valid if the camera "
              "has not moved since it was solved. Prefer the reader's fresh "
              "similarity (cam_to_base.json) once available.")
    else:                                        # cam_to_base.json (similarity)
        R = np.array(T["R"], float); t = np.array(T["t"], float); s = float(T["s"])
        print(f"cam->base loaded (similarity, s={s:.4f}, "
              f"solved {T.get('solved','?')})")
        print("!! confirm this transform is FRESH (rerun the reader if the "
              "camera or desk markers moved since)")

    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    dscale = prof.get_device().first_depth_sensor().get_depth_scale()

    print(f"camera up, averaging {args.frames} depth frames ...")
    for _ in range(30):
        pipe.wait_for_frames()                      # longer warmup (emitter/AE)
    depths, color = [], None
    use_align = True
    for k in range(args.frames):
        raw = pipe.wait_for_frames()
        fr = align.process(raw) if use_align else raw
        df = fr.get_depth_frame()
        d = np.asanyarray(df.get_data()).astype(np.float32)
        valid = int((d > 0).sum())
        if k < 3:
            print(f"  frame {k}: {valid} valid depth px "
                  f"({'aligned' if use_align else 'RAW'})")
        if k == 2 and use_align and valid == 0:
            print("  aligned depth empty — falling back to RAW depth stream")
            use_align = False
            intr = raw.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
            depths = []
            continue
        d[d == 0] = np.nan
        depths.append(d)
        cf = fr.get_color_frame()
        if cf:
            color = np.asanyarray(cf.get_data())
    pipe.stop()
    if not depths or not np.isfinite(np.stack(depths)).any():
        raise SystemExit("!! zero valid depth on BOTH aligned and raw paths — "
                         "camera-level: replug USB (try the other port), check "
                         "nothing within 28 cm, confirm track_probe works.")
    depth = np.nanmedian(np.stack(depths), axis=0) * dscale * 1000.0   # mm

    # deproject every valid pixel
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ok = np.isfinite(depth) & (depth > 100) & (depth < 2000)
    z = depth[ok]
    x = (u[ok] - intr.ppx) / intr.fx * z
    y = (v[ok] - intr.ppy) / intr.fy * z
    pts_cam = np.stack([x, y, z], axis=1)
    cols = color[ok][:, ::-1]                       # BGR->RGB

    # camera -> base (similarity)
    pts = (s * (R @ pts_cam.T)).T + t

    # crop to workspace
    if len(pts) == 0:
        raise SystemExit("!! zero valid depth pixels — camera blocked, "
                         "face-down, <28 cm from a surface, or on a USB2 "
                         "link. Nothing written.")
    m = ((pts[:, 0] > args.xmin) & (pts[:, 0] < args.xmax) &
         (pts[:, 1] > args.ymin) & (pts[:, 1] < args.ymax) &
         (pts[:, 2] > args.zmin) & (pts[:, 2] < args.zmax))
    pts, cols = pts[m], cols[m]
    print(f"{m.sum()} points in workspace crop")

    # ---- SELF-RECTIFY: the desk is flat & horizontal, its base-frame z is
    # TCP-touched truth. Fit the dominant low plane, rotate it level, pin it.
    # Cancels the bridge's rotation error using ground truth it never had.
    try:
        calib = json.load(open("src/calibration/Guided_sweep_calib.json"))
        desk_z = float(np.mean([calib["desk_markers_base_mm"]["1"][2],
                                calib["desk_markers_base_mm"]["2"][2]]))
    except Exception:
        desk_z = None
    zlo = np.percentile(pts[:, 2], 5)
    slab = pts[(pts[:, 2] > zlo - 5) & (pts[:, 2] < zlo + 25)]
    if len(slab) > 500:
        ctr = slab.mean(0)
        n = np.linalg.svd(slab - ctr, full_matrices=False)[2][2]
        if n[2] < 0: n = -n
        tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        axis = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(axis) > 1e-8:
            axis = axis / np.linalg.norm(axis)
            ang = np.arccos(np.clip(n[2], -1, 1))
            Kx = np.array([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            Rl = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
            pts = (pts - ctr) @ Rl.T + ctr
        if desk_z is not None:
            slab2 = pts[(pts[:, 2] > np.percentile(pts[:, 2], 2)) &
                        (pts[:, 2] < np.percentile(pts[:, 2], 2) + 20)]
            pts[:, 2] += desk_z - float(np.median(slab2[:, 2]))
            print(f"self-rectified: desk tilt {tilt:.1f} deg removed, "
                  f"plane pinned to touched z={desk_z:.1f}")
        else:
            print(f"self-rectified: desk tilt {tilt:.1f} deg removed "
                  f"(no calib z found — height unpinned)")

    # voxel downsample
    vox = np.floor(pts / args.voxel).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    pts, cols = pts[idx], cols[idx]
    print(f"{len(pts)} points after {args.voxel} mm voxel downsample")

    np.savez(args.out + ".npz", points_base=pts.astype(np.float32),
             colors=cols.astype(np.uint8),
             frame="cobot_base_mm", solved=T.get("solved", ""))
    with open(args.out + ".ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {c[0]} {c[1]} {c[2]}\n")
    print(f"wrote {args.out}.ply + {args.out}.npz  (units: mm, frame: cobot base)")


if __name__ == "__main__":
    main()