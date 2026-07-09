#!/usr/bin/env python3
"""
guided_sweep_reader.py — Mac-side vision brain for the guided sweep. PRINT ONLY.

Chain:
  ids 1,2 (desk, fixed, TCP-touched)  -> camera->base transform
  id 3   (container, position ONLY)   -> phantom anchor in base coords
  scan start = id3_base + fixed offset (base frame; container translates, never rotates)
  waypoints  = start + k*step * fixed sweep axis (base frame, from drag-teach)

Nothing is sent to the arm. Output is the exact [x,y,z,rx,ry,rz] list the
receiver would take, printed as JSON so it can be eyeballed / tape-measure
checked first.

Bootstrap order (fields in guided_sweep_calib.json):
  1. Fill desk_markers_base_mm from TCP-touch (get_coords() over each center).
  2. Run this script. If id3_to_scan_start_base_mm is null it prints id3's
     base position and stops -> offset = (TCP-touched scan start) - (that).
  3. Fill sweep_axis_base + probe_orientation from the drag-teach endpoints.
  4. Run again -> full waypoint list.

Usage:
  python guided_sweep_reader.py                 # uses guided_sweep_calib.json
  python guided_sweep_reader.py --calib my.json
"""

import argparse, json, sys, time
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------------------------------------------------------- aruco compat
def make_detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):          # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return lambda img: det.detectMarkers(img)[:2]
    params = cv2.aruco.DetectorParameters_create()   # legacy API
    return lambda img: cv2.aruco.detectMarkers(img, d, parameters=params)[:2]

def marker_pose(corners_px, size_mm, K, dist):
    """Single-marker pose via IPPE_SQUARE. Returns (p_cam_mm, R_cam)."""
    s = size_mm / 2.0
    obj = np.array([[-s,  s, 0], [ s,  s, 0],
                    [ s, -s, 0], [-s, -s, 0]], dtype=np.float64)  # aruco corner order
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.reshape(4, 2).astype(np.float64),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.reshape(3), R

# ---------------------------------------------------------------- realsense
def _start(pipe, cfg):
    prof = pipe.start(cfg)
    for _ in range(15):                  # warmup, same as track_probe.py
        pipe.wait_for_frames(10000)
    return prof

def open_camera():
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        raise SystemExit("no RealSense found — check USB / sudo, or replug.")

    # exactly track_probe.py's known-good config: on macOS the color stream
    # alone can fail to power up ("failed to set power state"); enabling
    # depth alongside it is what makes it start reliably.
    both = rs.config()
    both.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    both.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    default = rs.config()

    prof = None
    for attempt, cfg in enumerate([both, default, default]):
        pipe = rs.pipeline()
        try:
            prof = _start(pipe, cfg)
            break
        except RuntimeError as e:
            try:
                pipe.stop()
            except RuntimeError:
                pass
            if attempt == 0:
                print("depth+color config failed — trying device default profile ...")
            elif attempt == 1:
                print(f"default profile failed too ({e}) — hardware reset, last try ...")
                ctx.query_devices()[0].hardware_reset()
                time.sleep(6)
            else:
                raise SystemExit("camera won't start — unplug/replug the RealSense, "
                                 "confirm track_probe.py works, then rerun.")
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"color stream: {intr.width}x{intr.height}"
          + ("" if (intr.width, intr.height) == (640, 480) else
             "  !! NOT the calibrated 640x480 profile — replug the camera and "
             "rerun rather than trusting this session's numbers."))
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    align = rs.align(rs.stream.color)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    return pipe, K, dist, intr, align, depth_scale

def depth_center_mm(depth_img, intr, depth_scale, u, v, half=3):
    """Marker-center 3D position from the DEPTH SENSOR (mm, camera frame).
    PnP z from a small planar marker is scale-biased (a few % — the reason
    track_probe.py fuses depth); the sensor measures z directly."""
    h, w = depth_img.shape
    ui = int(np.clip(u, half, w - half - 1)); vi = int(np.clip(v, half, h - half - 1))
    patch = depth_img[vi-half:vi+half+1, ui-half:ui+half+1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    z_m = float(np.median(valid)) * depth_scale
    p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z_m)
    return np.array(p, dtype=np.float64) * 1000.0

# ---------------------------------------------------------------- geometry
def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("degenerate vector")
    return v / n

def camera_to_base(p1c, p2c, n_c, p1b, p2b):
    """
    Build s,R,t (base <- camera, SIMILARITY transform) from the desk markers.
    The scale s = touched/seen absorbs BOTH the depth sensor's small scale
    bias AND the cobot's kinematic scale error — all commanding happens in
    the arm's believed frame, so camera measurements must be shrunk/stretched
    into it. Assumes: desk horizontal, cobot base z vertical.
      base triad:  z = world up, x = id1->id2 projected horizontal
      cam  triad:  z = desk normal (marker z axes), x = id1->id2 projected
    """
    s = np.linalg.norm(p2b - p1b) / np.linalg.norm(p2c - p1c)

    zb = np.array([0.0, 0.0, 1.0])
    xb = unit((p2b - p1b) - np.dot(p2b - p1b, zb) * zb)
    yb = np.cross(zb, xb)
    B = np.column_stack([xb, yb, zb])

    zc = unit(n_c)
    v = p2c - p1c
    xc = unit(v - np.dot(v, zc) * zc)
    yc = np.cross(zc, xc)
    C = np.column_stack([xc, yc, zc])

    R = B @ C.T
    t = 0.5 * ((p1b - R @ (s * p1c)) + (p2b - R @ (s * p2c)))
    # residual = how far apart the two markers' implied translations are
    resid = np.linalg.norm((p1b - R @ (s * p1c)) - (p2b - R @ (s * p2c)))
    return s, R, t, resid

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="guided_sweep_calib.json")
    args = ap.parse_args()

    cfg = json.load(open(args.calib))
    for mid in ("1", "2"):
        if cfg["desk_markers_base_mm"][mid] is None:
            sys.exit(f"desk_markers_base_mm[{mid}] not set — TCP-touch it first.")
    p1b = np.array(cfg["desk_markers_base_mm"]["1"], dtype=np.float64)
    p2b = np.array(cfg["desk_markers_base_mm"]["2"], dtype=np.float64)
    sizes = {int(k): float(v) for k, v in cfg["marker_size_mm"].items()}
    N = int(cfg.get("n_frames_average", 30))

    detect = make_detector(cfg["aruco_dict"])
    pipe, K, dist, intr, align, depth_scale = open_camera()
    print(f"camera up, averaging over {N} frames ...")

    # accumulate per-id positions (mm, camera frame) + desk-normal samples
    acc = {1: [], 2: [], 3: []}
    desk_pts = []                                    # depth-deprojected marker corners
    x3_axes = []                                     # id3 in-plane axis samples
    t0 = time.time()
    try:
        while min(len(acc[1]), len(acc[2]), len(acc[3])) < N:
            if time.time() - t0 > 30:
                sys.exit(f"timeout: seen counts "
                         f"{ {i: len(acc[i]) for i in acc} } — check visibility.")
            frames = align.process(pipe.wait_for_frames(10000))
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            corners, ids = detect(img)
            if ids is None:
                continue
            for c, i in zip(corners, ids.flatten()):
                i = int(i)
                if i not in acc or i not in sizes:
                    continue
                # position: depth sensor (scale-true)
                u, v = c.reshape(4, 2).mean(axis=0)
                p = depth_center_mm(depth_img, intr, depth_scale, u, v)
                if p is None:
                    continue
                acc[i].append(p)
                if i in (1, 2):
                    # desk-plane samples: deproject all 4 corners via DEPTH.
                    # (PnP out-of-plane tilt on a small planar marker is the
                    # ambiguity-prone DOF — 5-10 deg errors rotate the whole
                    # base frame and made id3 viewpoint-dependent. Depth
                    # corners fitted as a plane don't have that failure.)
                    for (cu, cv) in c.reshape(4, 2):
                        q = depth_center_mm(depth_img, intr, depth_scale, cu, cv, half=2)
                        if q is not None:
                            desk_pts.append(q)
                elif i == 3:
                    _, R = marker_pose(c, sizes[i], K, dist)
                    if R is not None:
                        x3_axes.append(R[:, 0])      # for the yaw guard only
    finally:
        pipe.stop()

    p1c = np.mean(acc[1], axis=0)
    p2c = np.mean(acc[2], axis=0)

    # desk normal: SVD plane fit over all depth-deprojected corner points
    pts = np.asarray(desk_pts)
    if len(pts) < 24:
        sys.exit(f"only {len(pts)} desk-plane points — depth not landing on "
                 f"the markers; check range/occlusion.")
    ctr = pts.mean(axis=0)
    _, sv, Vt = np.linalg.svd(pts - ctr)
    n_c = Vt[2]
    if np.dot(n_c, -ctr) < 0:                        # sign: toward the camera
        n_c = -n_c
    plane_rms = float(np.sqrt(np.mean(((pts - ctr) @ n_c) ** 2)))
    print(f"desk plane: {len(pts)} pts, fit RMS {plane_rms:.2f} mm")
    if plane_rms > 3.0:
        sys.exit("desk plane fit RMS > 3 mm — depth points aren't coplanar "
                 "(marker not flat / depth artifacts). Fix before trusting.")

    # scale report. Camera (depth-fused) vs caliper truth is the sensor's
    # bias; camera vs touched is what the similarity fit will absorb (sensor
    # bias + cobot kinematic scale together). Hard-fail only if structural.
    seen = np.linalg.norm(p2c - p1c)
    truth = np.linalg.norm(p2b - p1b)
    gap = seen - truth
    print(f"desk baseline: camera {seen:.1f} mm vs touched {truth:.1f} mm "
          f"(gap {gap:+.1f} mm, fit scale s={truth/seen:.4f})")
    if abs(gap) > 20.0:
        sys.exit("baseline gap > 20 mm — beyond plausible sensor+kinematic "
                 "error; check marker size / calib values / detection.")

    s, R, t, resid = camera_to_base(p1c, p2c, n_c, p1b, p2b)
    print(f"camera->base solved (similarity), internal residual {resid:.2f} mm")

    to_base = lambda pc: R @ (s * pc) + t

    # id3 -> phantom anchor (position only; container never rotates)
    p3b = to_base(np.mean(acc[3], axis=0))
    print(f"id3 in base frame: [{p3b[0]:.1f}, {p3b[1]:.1f}, {p3b[2]:.1f}] mm")

    # rotation tripwire: id3's in-plane yaw in the base frame. We never USE
    # id3's orientation in the chain, but in-plane yaw is flip-stable, so it
    # works as a "did the container get rotated" alarm.
    x3 = R @ unit(np.mean(x3_axes, axis=0))          # id3 x-axis in base frame
    yaw = float(np.degrees(np.arctan2(x3[1], x3[0])))
    exp_yaw = cfg.get("id3_expected_yaw_deg")
    if exp_yaw is None:
        print(f"id3 yaw in base: {yaw:.1f} deg — store this as "
              f"id3_expected_yaw_deg to arm the rotation guard.")
    else:
        drift = (yaw - float(exp_yaw) + 180.0) % 360.0 - 180.0
        print(f"id3 yaw: {yaw:.1f} deg (ref {exp_yaw:.1f}, drift {drift:+.1f})"
              + ("  <-- CONTAINER ROTATED? sweep axis is stale." if abs(drift) > 5 else ""))

    off = cfg.get("id3_to_scan_start_base_mm")        # [dx, dy] — x/y ONLY
    z0 = cfg.get("scan_start_z_base_mm")              # fixed arm-frame constant
    if off is None or z0 is None:
        print("\nid3_to_scan_start_base_mm ([dx,dy]) or scan_start_z_base_mm "
              "not set. Offset x/y = touched start x/y - id3 x/y above; z0 = "
              "the touched start z. (Container translates on the DESK — 2-DOF."
              " The camera anchors x/y; z is a constant, immune to depth noise.)")
        return

    start = np.array([p3b[0] + off[0], p3b[1] + off[1], float(z0)])

    # hand the anchor to make_waypoints_from_log.py automatically
    import os
    anchor_path = os.path.join(os.path.dirname(os.path.abspath(args.calib)),
                               "vision_anchor.json")
    with open(anchor_path, "w") as f:
        json.dump({"start_x": round(float(start[0]), 2),
                   "start_y": round(float(start[1]), 2),
                   "id3_base": [round(float(v), 2) for v in p3b],
                   "written": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    print(f"anchor written: {anchor_path}  (start x/y "
          f"{start[0]:.1f}, {start[1]:.1f})")

    axis = cfg.get("sweep_axis_base")
    rot = cfg.get("probe_orientation_rxryrz_deg")
    if axis is None or rot is None:
        print(f"\nscan start in base: [{start[0]:.1f}, {start[1]:.1f}, "
              f"{start[2]:.1f}]  (sweep axis / probe orientation not set yet — "
              f"fill from drag-teach endpoints)")
        return

    axis = unit(np.array(axis, dtype=np.float64))
    step = float(cfg["waypoint_step_mm"])
    length = float(cfg["sweep_length_mm"])
    n_wp = int(np.floor(length / step)) + 1

    print(f"\n{n_wp} waypoints (step {step} mm, length {length} mm), "
          f"orientation {rot} — PRINT ONLY, nothing sent:\n")
    for k in range(n_wp):
        p = start + k * step * axis
        coords = [round(float(p[0]), 1), round(float(p[1]), 1),
                  round(float(p[2]), 1)] + [float(r) for r in rot]
        print(json.dumps({"coords": coords, "speed": 25}))

if __name__ == "__main__":
    main()