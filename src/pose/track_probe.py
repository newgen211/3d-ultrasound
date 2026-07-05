# sudo $(which python) src/pose/track_probe.py 
import time, json, os
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA       = _REPO_ROOT / "data"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_MM  = 25.0
TARGET_ID  = 0
# Anchor output to the repo's data/ folder, regardless of where this is launched.
LOG_PATH   = os.path.join(DATA, "pose_logs", "probe_pose_log.jsonl")
WIDTH, HEIGHT, FPS = 640, 480, 30

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    return R_inv, -R_inv @ tvec.reshape(3)

def depth_at(depth_img, u, v, scale, half=3):
    h, w = depth_img.shape
    u = int(np.clip(u, half, w-half-1)); v = int(np.clip(v, half, h-half-1))
    patch = depth_img[v-half:v+half+1, u-half:u+half+1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid) * scale * 1000.0) if valid.size else None

def geodesic(Ra, Rb):
    """angle (rad) between two rotation matrices."""
    d = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(d, -1.0, 1.0)))

_h = MARKER_MM / 2.0
OBJ = np.array([[-_h,_h,0],[_h,_h,0],[_h,-_h,0],[-_h,-_h,0]], dtype=np.float32)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]], dtype=np.float32)
dist = np.array(intr.coeffs, dtype=np.float32)
print(f"Intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f}  depth_scale={depth_scale}")

adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
try:
    detector = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    def detect(g): return detector.detectMarkers(g)
except AttributeError:
    params = cv2.aruco.DetectorParameters_create()
    def detect(g): return cv2.aruco.detectMarkers(g, adict, parameters=params)

def solve_consistent(corners, prevR):
    """Pick the ArUco pose consistent with the previous frame's rotation.
       A single planar marker has two valid solutions; the solver flips between
       them frame-to-frame. solvePnPGeneric returns BOTH, and we choose the one
       closest to the last accepted rotation -> kills the flip."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        OBJ, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n == 0:
        return None
    errs = np.asarray(errs).ravel()
    cands = []
    for k in range(n):
        R, _ = cv2.Rodrigues(rvecs[k])
        cands.append((R, rvecs[k], tvecs[k], float(errs[k])))
    if prevR is None:
        R, rvec, tvec, _ = min(cands, key=lambda c: c[3])              # first frame: lowest reproj error
    else:
        R, rvec, tvec, _ = min(cands, key=lambda c: geodesic(prevR, c[0]))  # else: closest to last
    return R, rvec, tvec

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "a")
buf_xy, buf_az, buf_dz = [], [], []
prev_R = {}   # id -> last accepted rotation matrix (flip suppression)

try:
    print("Warming up...")
    for _ in range(15):
        pipe.wait_for_frames()
    print("Live view open. Press q (or Ctrl-C) to stop.")
    while True:
        frames = align.process(pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth: continue
        img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_ns = time.time_ns()

        corners, ids, _ = detect(gray)
        seen = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                i = int(i)
                sol = solve_consistent(c[0], prev_R.get(i))
                if sol is None: continue
                R, rvec, tvec = sol
                prev_R[i] = R
                ax, ay, az = tvec.reshape(3)
                rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
                ctr = c[0].mean(axis=0)
                dz = depth_at(depth_img, ctr[0], ctr[1], depth_scale)
                z_fused = dz if dz is not None else float(az)
                coords = [float(ax), float(ay), z_fused, float(rpy[0]), float(rpy[1]), float(rpy[2])]
                log.write(json.dumps({"t_ns":t_ns,"id":i,"coords":coords,
                                      "aruco_z":float(az),"depth_z":dz})+"\n")
                seen[i] = (float(ax), float(ay), float(az), dz)
                cv2.drawFrameAxes(img, K, dist, rvec, tvec, MARKER_MM*0.5)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

        if TARGET_ID in seen:
            ax, ay, az, dz = seen[TARGET_ID]
            buf_xy.append((ax,ay)); buf_xy = buf_xy[-30:]
            buf_az.append(az); buf_az = buf_az[-30:]
            if dz is not None: buf_dz.append(dz); buf_dz = buf_dz[-30:]
            jxy = np.array(buf_xy).std(axis=0) if len(buf_xy)>5 else [0,0]
            jaz = float(np.std(buf_az)) if len(buf_az)>5 else 0.0
            jdz = float(np.std(buf_dz)) if len(buf_dz)>5 else 0.0
            dz_txt = f"{dz:.0f}" if dz is not None else "--"
            cv2.putText(img, f"id{TARGET_ID}  aruco_z={az:.0f}  depth_z={dz_txt} mm",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(img, f"z-jitter  aruco {jaz:.2f}   depth {jdz:.2f} mm",
                        (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(img, f"xy-jitter {jxy[0]:.2f} {jxy[1]:.2f} mm",
                        (10,86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(img, f"id{TARGET_ID} not seen  visible={sorted(seen)}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("track_probe", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
except KeyboardInterrupt:
    pass
finally:
    log.close(); pipe.stop(); cv2.destroyAllWindows()