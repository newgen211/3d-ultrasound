# CHEATSHEET — Robotic 3D Ultrasound Pipeline
*(one page; update the day a script changes, or it doesn't merge)*

## Network / machines
- Pi: `ssh er@192.168.196.134`  (ZeroTier `MyCobot320`; Mac = 192.168.196.41)
- Pi wall clock is Asia/Shanghai (UTC+8), so exec_<stamp> names are CST; the epoch ns
  inside the logs are NTP-synced to the Mac (+0.6 s on 22 Sep; trim correlates it out)
- Clarius = its own WiFi AP. The Pi joins it on wlan0 (eth0 keeps ZeroTier); the Mac never does.

## ONE-BUTTON SWEEP (the whole loop)
```
sudo chown -R $USER calib/          # once; the reader used to leave root-owned files
make sweep                          # = fly.py: gates -> anchor -> solve -> deploy -> tracker
                                    #   -> Pi flight (capture + arm) -> fetch -> verify -> make post
TEACH=sweep_teach_up2.jsonl make sweep      # other taught skill (default: clamp15)
#   everything for the run: runs/<stamp>_sec<N>/{fly.log,solve.log,anchor.json,manifest.json}
#   one sudo prompt up front (reader + tracker run as root with the realsense python)
```
Stage by stage, if you need to redo one (each is skipped when its artifact is newer):
```
make post SEC=<N>                   # trim -> smooth -> merge -> reconstruct -> vessels
make trim|smooth|merge|reconstruct|vessels SEC=<N>
make clean-stamps SEC=<N>           # force trim + merge to rerun
#   inputs: data/clarius_sessions/section_<N>/{exec.jsonl,exec_meta.json}  data/pose_logs/sec<N>_cam.jsonl
#   trim refuses (exit 1) on: weak correlation, NaN peak, under half the frames kept,
#   or |Pi->Mac offset| > 5 s (--max-offset widens it for a deliberately drifted clock)
```
Pi smoke test, probe awake, before the first real flight (see ONE_BUTTON.md):
```
ssh er@192.168.196.134 'cd ~/Documents/ultrasound-cobot && python3.10 cast_headless.py --section 900 --ip 192.168.1.1 --port 5828 --seconds 10'
#   pass = raw_*.bin/.json on disk under clarius_sessions/section_900 and imu_sample_count > 0
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
| `fly.py` + `Makefile` | the button: one acquisition, then the file-keyed post chain |
| `run_sweep.py` | `preflight()` gates (imported by fly.py); its own CLI is the manual flow |
| `src/capture/cast_capture.py` | capture core shared by the GUI and `cast_headless.py` (Pi) |
| `ultrasound-cobot/flight.sh` | Pi: probe WiFi -> headless capture -> arm -> pair exec log |
| `src/calibration/guided_sweep_reader.py` | perceive: markers -> calib/vision_anchor.json |
| `ultrasound-cobot/shift_joint_path.py` + `calib/mycobot_320_pi.urdf` | plan: taught skill + anchor -> verified joints |
| `ultrasound-cobot/execute_sweep.py` | act: joint playback + exec log + manifest (`--no-prompt` for flight.sh) |
| `src/pose/trim_section.py` | drop hover/retract frames (clock-aligned) |
| `src/pose/smooth_cam_log.py` | pose smoothing pre-merge (servo sweeps) |
| `src/pose/merge_poses_cam.py` | camera poses -> frame sidecars |
| `src/reconstruct/reconstruct_handeye.py` | posed frames -> volume + MIPs |
| `sweep_teach*.jsonl` | the taught skill(s) — permanent assets |
| `calib/handeye.json` | = section_60's 2.20 mm (keep it so) |
