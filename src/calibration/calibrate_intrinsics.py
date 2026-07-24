#!/usr/bin/env python3
"""
calibrate_intrinsics.py — custom RealSense color intrinsics (A3 experiment #1).

Captures checkerboard views at the OPERATING PROFILE (640x480 depth+color —
calibrate the profile you fly), runs cv2.calibrateCamera with sub-pixel
corners, and writes intrinsics.json. Compares against factory intrinsics.

Run (Mac, realsense env, sudo like track_probe):
    sudo $(which python) calibrate_intrinsics.py
Keys:  SPACE = capture view   q = finish & calibrate
Aim for 20-30 views: corners of the frame, near/far, tilted every which way.
"""

import json, sys, time
import numpy as np
import cv2
import pyrealsense2 as rs

import argparse, subprocess
ap = argparse.ArgumentParser()
ap.add_argument("--screen", action="store_true",
                help="display the checkerboard fullscreen on THIS laptop and "
                "auto-capture while you pose it — no keyboard needed")
ap.add_argument("--views", type=int, default=25, help="auto-capture view count")
args = ap.parse_args()

ROWS, COLS = 6, 9          # inner corners
SQUARE_MM = 24.0           # printed-board default; --screen mode prompts you

def say(txt):
    try: subprocess.Popen(["say", txt])
    except Exception: print(f"[{txt}]")

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
for _ in range(15):
    pipe.wait_for_frames()
fintr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(f"factory: fx={fintr.fx:.2f} fy={fintr.fy:.2f} "
      f"cx={fintr.ppx:.2f} cy={fintr.ppy:.2f} dist={list(fintr.coeffs)}")

objp = np.zeros((ROWS*COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * (SQUARE_MM / 1000.0)
objpoints, imgpoints = [], []
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
recent = []          # last few corner sets, for the stillness gate
prev_views = []      # captured corner sets, for the novelty gate (--screen)

if args.screen:
    # draw the board ourselves, sized to the actual display — flat and square
    # by manufacture. macOS OpenCV often ignores the fullscreen flag, so we
    # also size the image to the screen and pin the window at the origin.
    try:
        import tkinter
        _tk = tkinter.Tk(); SW, SH = _tk.winfo_screenwidth(), _tk.winfo_screenheight()
        _tk.destroy()
    except Exception:
        SW, SH = 1440, 900
    cell = min((SW - 80) // (COLS + 1), (SH - 80) // (ROWS + 1))
    bw, bh = (COLS + 1) * cell, (ROWS + 1) * cell
    board = np.full((SH, SW), 255, np.uint8)
    ox, oy = (SW - bw) // 2, (SH - bh) // 2
    for r in range(ROWS + 1):
        for cc in range(COLS + 1):
            if (r + cc) % 2 == 0:
                board[oy + r*cell : oy + (r+1)*cell,
                      ox + cc*cell : ox + (cc+1)*cell] = 0
    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("board", SW, SH)
    cv2.moveWindow("board", 0, 0)
    cv2.imshow("board", board); cv2.waitKey(500)
    SQUARE_MM = float(input(
        "Caliper ONE black square ON THE SCREEN and enter its size in mm: "))
    objp = np.zeros((ROWS*COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * (SQUARE_MM / 1000.0)
    say("Calibration armed. Pose the laptop. Hold still to capture.")
    print("Auto-capture armed: pose the laptop toward the camera; each capture "
          "needs stillness AND a genuinely new pose. Listen for the count.")

print("SPACE=capture  q=calibrate. Get 20-30 varied views.")
try:
    while True:
        frames = pipe.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        disp = img.copy()
        still = False
        if found:
            cv2.drawChessboardCorners(disp, (COLS, ROWS), corners, found)
            recent.append(corners.reshape(-1, 2)); recent = recent[-6:]
            if len(recent) == 6:
                drift = max(np.abs(recent[-1] - r).max() for r in recent[:-1])
                still = drift < 0.4          # px over ~0.2 s — truly motionless
        else:
            recent = []
        status = "no board" if not found else ("STILL — SPACE to capture" if still
                                               else "hold still...")
        cv2.putText(disp, f"views: {len(objpoints)}  {status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if still else ((0, 200, 255) if found else (0, 0, 255)), 2)
        if not args.screen:
            cv2.imshow("intrinsics", disp)
        k = cv2.waitKey(1) & 0xFF
        auto_fire = False
        if args.screen and found and still:
            flat = corners.reshape(-1)
            novel = all(np.abs(flat - p).mean() > 12.0 for p in prev_views) \
                    if prev_views else True
            if novel:
                auto_fire = True
        if (k == ord(' ') or auto_fire) and found and still:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            objpoints.append(objp); imgpoints.append(corners)
            prev_views.append(corners.reshape(-1))
            print(f"  captured view {len(objpoints)}")
            if args.screen:
                say(str(len(objpoints)))
                recent = []                       # force fresh stillness next pose
                if len(objpoints) >= args.views:
                    say("Done. Solving.")
                    break
            time.sleep(0.3)
        elif k == ord('q'):
            break
finally:
    pipe.stop(); cv2.destroyAllWindows()

if len(objpoints) < 12:
    sys.exit(f"only {len(objpoints)} views — need >=12 (20-30 better). Rerun.")

def solve(op, ip):
    # seed with factory + lock aspect ratio: the sensor has square pixels
    # (factory fx/fy ratio 1.0009). Removes the degenerate fx-vs-fy direction
    # that under-tilted view sets let the solver wander down.
    K0 = np.array([[fintr.fx, 0, fintr.ppx], [0, fintr.fy, fintr.ppy], [0, 0, 1]])
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    rms, K, dist, rv, tv = cv2.calibrateCamera(op, ip, gray.shape[::-1],
                                               K0.copy(), None, flags=flags)
    per = []
    for i in range(len(op)):
        proj, _ = cv2.projectPoints(op[i], rv[i], tv[i], K, dist)
        per.append(float(np.sqrt(np.mean((proj - ip[i]) ** 2))))
    return rms, K, dist, per

rms, K, dist, per = solve(objpoints, imgpoints)
print("\nper-view error (px):", [f"{e:.2f}" for e in per])
good = [i for i, e in enumerate(per) if e < 1.5]
if len(good) < len(per) and len(good) >= 12:
    print(f"dropping {len(per)-len(good)} outlier view(s), re-solving...")
    objpoints = [objpoints[i] for i in good]
    imgpoints = [imgpoints[i] for i in good]
    rms, K, dist, per = solve(objpoints, imgpoints)
    print("per-view after drop:", [f"{e:.2f}" for e in per])
print(f"\ncalibration RMS reprojection: {rms:.3f} px  ({len(objpoints)} views)")
print(f"custom:  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
print(f"factory: fx={fintr.fx:.2f} fy={fintr.fy:.2f} cx={fintr.ppx:.2f} cy={fintr.ppy:.2f}")
print(f"fx delta: {100*(K[0,0]-fintr.fx)/fintr.fx:+.2f}%  "
      f"fy delta: {100*(K[1,1]-fintr.fy)/fintr.fy:+.2f}%")
if rms > 0.8:
    print("!! RMS > 0.8 px — board not flat, bad prints, or too few varied views. "
          "Fix and rerun before trusting.")

out = {"width": 640, "height": 480,
       "K": K.tolist(), "dist": dist.ravel().tolist(),
       "rms_px": float(rms), "n_views": len(objpoints),
       "square_mm": SQUARE_MM, "captured": time.strftime("%Y-%m-%d %H:%M:%S")}
with open("intrinsics.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote intrinsics.json — track_probe loads this when present.")