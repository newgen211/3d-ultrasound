# CHEATSHEET — Robotic 3D Ultrasound Pipeline
*(one page; update the day a script changes, or it doesn't merge)*

## Network / machines
- Pi: `ssh er@192.168.196.134`  (ZeroTier `MyCobot320`; Mac = 192.168.196.41)
- Pi clock shows UTC — epoch ns are NTP-synced to the Mac (measured −0.2 s)
- Clarius = its own WiFi AP → joining it kills Mac↔Pi. Use tmux + `--delay`.

## AUTONOMOUS SWEEP (the whole loop)
```
# Mac — anchor (camera at tape marks; MUST print "color stream: 640x480"):
sudo $(which python) src/calibration/guided_sweep_reader.py --calib src/calibration/Guided_sweep_calib.json
#   ignore its waypoint printout (vestigial); it writes vision_anchor.json

# Mac — solve (self-gating: align RMS ~0.36, FK <=2.5mm, steps <3°):
python shift_joint_path.py sweep_teach.jsonl > shifted_sweep.jsonl
scp shifted_sweep.jsonl er@192.168.196.134:~/Documents/ultrasound-cobot/pose_logs/

# Mac — probe tracker (leave running; local, WiFi-independent):
sudo $(which python) src/pose/track_probe.py

# Pi — execute (INSIDE tmux; --rate 2 = slow/quality cadence):
tmux new -s sweep
python3 execute_sweep.py shifted_sweep.jsonl --speed 25 --rate 2 --delay 45
#   during countdown: Mac -> Clarius WiFi, start pysidecaster capture
#   after: Mac -> lab WiFi, `tmux attach -t sweep`, Enter to home
#   outputs: pose_logs/exec_<stamp>.jsonl + exec_<stamp>_meta.json
```

## POST-CAPTURE
```
mv data/pose_logs/probe_pose_log.jsonl data/pose_logs/sec<N>_cam.jsonl   # snapshot!
python src/pose/trim_section.py section_<N> <exec.jsonl> <meta.json> data/pose_logs/sec<N>_cam.jsonl
#   check: offset ~0.0s, lag ~ your --delay, keep count sane  ->  rerun --apply
python smooth_cam_log.py data/pose_logs/sec<N>_cam.jsonl data/pose_logs/sec<N>_cam_smooth.jsonl
python src/pose/merge_poses_cam.py section_<N> data/pose_logs/sec<N>_cam_smooth.jsonl
python src/reconstruct/reconstruct_handeye.py section_<N>       # uses root handeye.json = section_60's 2.20mm
# drift audit (expect <1mm taped): id3-during-capture one-liner (handoff 07-09)
```

## VESSELS
```
python src/reconstruct/view_planning_3d.py section_<N> \
  --handeye data/clarius_sessions/section_60/handeye.json \
  --eps 3.5 --n-targets 2 --split-lateral --present
python src/reconstruct/render_tubes_3d.py section_<N> --handeye data/clarius_sessions/section_60/handeye.json
# no --aniso  =>  prints measured per-axis noise (the honest numbers)
```

## CALIBRATION (rare — only when geometry changes)
```
# desk-marker touch (probe MOUNTED, same wrist orientation both touches):
python3 touch_calib.py                    # Pi; gimbal-safe version
# scan-start touch -> offset = start_xy - id3_xy;  z = touched constant
# drag-teach a sweep -> pose_logger; that log IS the skill (commit it)
```

## GOLDEN RULES
1. **Merge exec_/cam logs, never teach or shifted logs** (stale timestamps).
2. **Smooth for reconstruction, never for benchmarks.**
3. Camera must open **640x480 depth+color** — replug if 720p fallback.
4. Marker on probe is **rigid** — move it = redo hand-eye.
5. Container: taped during sweeps; free to translate BETWEEN sessions
   (toward base / sideways; NOT further away — reach boundary).
6. One serial owner on the Pi at a time (stage/receiver/executor: sequential).
7. Snapshot `probe_pose_log.jsonl` after every capture (it appends forever).
8. Gates refuse for a reason — never bypass an abort, diagnose it.

## FILE MAP
| File | Role |
|---|---|
| `guided_sweep_reader.py` | perceive: markers -> vision_anchor.json |
| `shift_joint_path.py` + `mycobot_320_pi.urdf` | plan: taught skill + anchor -> verified joints |
| `execute_sweep.py` (Pi) | act: joint playback + exec log + manifest |
| `trim_section.py` | drop hover/retract frames (clock-aligned) |
| `smooth_cam_log.py` | pose smoothing pre-merge (servo sweeps) |
| `merge_poses_cam.py` | camera poses -> frame sidecars |
| `reconstruct_handeye.py` | posed frames -> volume + MIPs |
| `sweep_teach*.jsonl` | the taught skill(s) — permanent assets |
| `handeye.json` (root) | = section_60's 2.20 mm (keep it so) |
