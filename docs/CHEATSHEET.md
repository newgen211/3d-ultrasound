# CHEATSHEET — Robotic 3D Ultrasound Pipeline
*(one page; update the day a script changes, or it doesn't merge)*

## Network / machines
- Pi: `ssh er@192.168.196.134`  (ZeroTier `MyCobot320`; Mac = 192.168.196.41)
- Pi clock shows UTC — epoch ns are NTP-synced to the Mac (measured −0.2 s)
- Clarius = its own WiFi AP → joining it kills Mac↔Pi. Use tmux + `--delay`.

## AUTONOMOUS SWEEP (the whole loop)
```
# Mac — anchor (camera at tape marks; MUST print "color stream: 640x480"):
sudo PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/realsense/bin/python \
  src/calibration/guided_sweep_reader.py     # --calib defaults to calib/guided_sweep_calib.json
#   ignore its waypoint printout (vestigial); it writes calib/vision_anchor.json

# Mac — solve (self-gating: align RMS ~0.36, FK <=2.5mm, steps <3°):
cd ultrasound-cobot && python shift_joint_path.py sweep_teach.jsonl > shifted_sweep.jsonl; cd ..
#   (that subproject runs from inside its own folder)
scp shifted_sweep.jsonl er@192.168.196.134:~/Documents/ultrasound-cobot/pose_logs/

# Mac — probe tracker (leave running; local, WiFi-independent):
sudo PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/realsense/bin/python src/pose/track_probe.py

# Pi — execute (INSIDE tmux; --rate 2 = slow/quality cadence):
tmux new -s sweep
python3 execute_sweep.py shifted_sweep.jsonl --speed 25 --rate 2 --delay 45
#   execute_sweep.py lives ONLY on the Pi - it is in neither git repo.
#   outputs/pi_mirror/apply_offsets.py imports it, so that mirror is not
#   runnable off the Pi either. Copy it into the mirror next Pi session.
#   during countdown: Mac -> Clarius WiFi, start pysidecaster capture
#   after: Mac -> lab WiFi, `tmux attach -t sweep`, Enter to home
#   outputs: pose_logs/exec_<stamp>.jsonl + exec_<stamp>_meta.json
```

## POST-CAPTURE
```
mv data/pose_logs/probe_pose_log.jsonl data/pose_logs/sec<N>_cam.jsonl   # snapshot!
python src/pose/trim_section.py section_<N> <exec.jsonl> <meta.json> data/pose_logs/sec<N>_cam.jsonl
#   check: offset ~0.0s, lag ~ your --delay, keep count sane  ->  rerun --apply
python src/pose/smooth_cam_log.py data/pose_logs/sec<N>_cam.jsonl data/pose_logs/sec<N>_cam_smooth.jsonl
python src/pose/merge_poses_cam.py section_<N> data/pose_logs/sec<N>_cam_smooth.jsonl
python src/reconstruct/reconstruct_handeye.py section_<N>       # uses calib/handeye.json = section_60's 2.20mm
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

## SUPERVISION + SCORING (E5 / paper era)
```
# offline: replay the contact gauge over a captured flight
python src/supervise/gauge.py section_127 [gold_n.pt]      # -> <section>/gauge_replay.jsonl
python src/supervise/viz_gauge_video.py section_127        # -> outputs/videos/

# live: the same algorithm, sending dz to the Pi (--shadow = decide, send nothing)
python src/supervise/supervisor_v3.py --section section_143 --shadow

# the frozen paper numbers — these must reproduce byte for byte, in `clarius`:
/opt/anaconda3/envs/clarius/bin/python src/audit/score_bench.py     | diff paper/ref_score_bench_stdout.txt -
/opt/anaconda3/envs/clarius/bin/python src/audit/score_classical.py | diff paper/ref_score_classical_stdout.txt -

# E5 figures (--out keeps the committed ones untouched while checking)
python src/viz/make_e5_figs.py --out /tmp/figs

# E1 repeatability (centerlines across identical-command flights 106/108/109/111)
python src/experiments/shape_repeat.py
python src/experiments/rescue_111.py        # undoes the flight-111 camera bump
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
9. Scripts run from any directory now. Point US3D_DATA at a copy before any
   experiment that writes into a section — `data/` has no backup.

## FILE MAP
| File | Role |
|---|---|
| `src/calibration/guided_sweep_reader.py` | perceive: markers -> calib/vision_anchor.json |
| `ultrasound-cobot/shift_joint_path.py` + `calib/mycobot_320_pi.urdf` | plan: taught skill + anchor -> verified joints |
| `execute_sweep.py` (Pi) | act: joint playback + exec log + manifest |
| `src/pose/trim_section.py` | drop hover/retract frames (clock-aligned) |
| `src/pose/smooth_cam_log.py` | pose smoothing pre-merge (servo sweeps) |
| `src/pose/merge_poses_cam.py` | camera poses -> frame sidecars |
| `src/reconstruct/reconstruct_handeye.py` | posed frames -> volume + MIPs |
| `sweep_teach*.jsonl` | the taught skill(s) — permanent assets |
| `calib/handeye.json` | = section_60's 2.20 mm (keep it so) |
