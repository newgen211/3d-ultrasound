#!/usr/bin/env python3
"""
pose_experiment.py — is the RealSense marker pose good enough to beat the arm?

Measures two things that decide the architecture (both frame-independent, so they
compare directly against the cobot's numbers):

  JITTER       — hold the probe DEAD STILL; how much does the reported pose wobble?
  DISPLACEMENT — move a KNOWN distance; does the marker read it back accurately?

Run on the Mac (RealSense). Marker rigid on the probe, arm holding it still.

    conda activate realsense
    python3 pose_experiment.py

Keys:
    (just hold still)  live readout shows rolling jitter — let it settle
    m                  mark current pose; prints distance from the previous mark
    r                  reset jitter stats
    q                  quit + print summary

Compare the jitter (mm) here against the cobot's get_coords jitter (Pi snippet
in the chat). Lower jitter + accurate known-move = the camera is the better pose
source. If the marker jitters several mm, the bet doesn't hold — rethink.
"""

import sys
import time
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

MARKER_MM = 40.0
MARKER_ID = 0
WIDTH, HEIGHT, FPS = 1280, 720, 30
WIN = 90                 # rolling jitter window (samples)
MARK_AVG = 30            # frames averaged when you press 'm'


def make_detector():
    try:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return lambda g: det.detectMarkers(g)
    except AttributeError:
        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters_create()
        return lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]], float)
    dist = np.array(intr.coeffs, float)
    detect = make_detector()
    h = MARKER_MM / 2.0
    objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)

    buf = deque(maxlen=WIN)          # recent translations (mm) for rolling jitter
    rbuf = deque(maxlen=WIN)         # recent euler (deg)
    recent = deque(maxlen=MARK_AVG)  # for averaged marks
    last_mark = None
    seen = total = 0
    print("📏 hold still — watch jitter settle. 'm' to mark, 'r' reset, 'q' quit.")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            total += 1
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)

            pose = None
            if ids is not None and MARKER_ID in ids:
                i = list(ids.flatten()).index(MARKER_ID)
                ok, rvec, tvec = cv2.solvePnP(objp, corners[i][0], K, dist)
                if ok:
                    seen += 1
                    t = tvec.ravel()
                    eul = Rotation.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler("xyz", degrees=True)
                    buf.append(t); rbuf.append(eul); recent.append(t)
                    pose = (rvec, tvec, t, eul)

            # rolling jitter (std over window)
            jt = np.array(buf)
            jr = np.array(rbuf)
            tj = jt.std(0) if len(jt) > 5 else np.zeros(3)
            tj_mag = float(np.sqrt((tj ** 2).sum()))
            rj = jr.std(0) if len(jr) > 5 else np.zeros(3)

            # draw
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
            if pose:
                cv2.drawFrameAxes(img, K, dist, pose[0], pose[1], MARKER_MM)
            det_rate = 100 * seen / max(1, total)
            lines = [
                f"{'TRACKING' if pose else 'NO MARKER'}   det {det_rate:.0f}%",
                f"jitter  T {tj[0]:.2f},{tj[1]:.2f},{tj[2]:.2f} mm  |T|={tj_mag:.2f}",
                f"jitter  R {rj[0]:.2f},{rj[1]:.2f},{rj[2]:.2f} deg",
            ]
            if pose:
                lines.append(f"pos {pose[2][0]:.1f},{pose[2][1]:.1f},{pose[2][2]:.1f} mm")
            for n, ln in enumerate(lines):
                cv2.putText(img, ln, (12, 30 + 26 * n), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0) if pose else (0, 0, 255), 2)
            cv2.imshow("pose_experiment (m=mark r=reset q=quit)", img)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                buf.clear(); rbuf.clear(); seen = total = 0
                print("↺ reset")
            elif k == ord("m"):
                if len(recent) < 5:
                    print("  (no marker — can't mark)")
                else:
                    p = np.array(recent).mean(0)
                    if last_mark is not None:
                        d = float(np.linalg.norm(p - last_mark))
                        print(f"📍 mark {p.round(1)} mm   →  distance from last: {d:.2f} mm")
                    else:
                        print(f"📍 mark {p.round(1)} mm  (move a known distance, press m again)")
                    last_mark = p
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        if len(buf) > 5:
            tj = np.array(buf).std(0)
            print(f"\n=== summary ===")
            print(f"translation jitter: {tj[0]:.2f}, {tj[1]:.2f}, {tj[2]:.2f} mm  "
                  f"(|T|={np.sqrt((tj**2).sum()):.2f} mm)")
            print(f"detection rate:     {100*seen/max(1,total):.0f}%")
            print("compare |T| against the cobot get_coords jitter.")


if __name__ == "__main__":
    main()