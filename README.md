# Robotic 3D Ultrasound

A cobot arm sweeps a Clarius ultrasound probe over a phantom while a RealSense
camera tracks the probe pose (ArUco marker); per-frame poses are fused with the
B-mode frames into metric 3D reconstructions, and vessels are segmented and
scored against human ground truth.

## Pipeline

Run the stages in order; each consumes the previous stage's output.

| Stage | Where | What it does |
|---|---|---|
| **1. capture** | `run_sweep.py` (orchestrator) + `src/capture/pysidecaster.py` | Fly a sweep and record raw probe frames + IMU + timestamps into `data/clarius_sessions/section_<N>/`. `run_sweep.py` refuses to fly if any recorder is dead. |
| **2. pose** | `src/pose/` | `track_probe.py` tracks the probe marker (Mac camera). `smooth_cam_log` → `trim_section` → `merge_poses_cam` attach a camera/cobot pose to every frame. |
| **3. calibration** | `src/calibration/` → `calib/` | `calibrate_intrinsics`, `calibrate_handeye` (flange→image), and `solve_bridge` (camera↔cobot, `AX=XB`). Outputs land in `calib/`. |
| **4. segment** | `src/segment/` | Vessel segmentation — classical + SAM 3 (`sam3_track*.py`), gated to open/crescent lumen. |
| **5. reconstruct** | `src/reconstruct/` | Posed frames → metric 3D volume + MIPs (`reconstruct_handeye`), multi-view compounding (`compound_handeye`), vessel clustering/rendering. |
| **audit** (cross-cutting) | `src/audit/` | Human vessel-circle truth (`make_audit_set` → `audit_tool`), scoring teacher/student vs truth (`score_audit`), and exemplar harvest for SAM (`harvest_exemplars`). |
| **arm repo** (separate) | `ultrasound-cobot/` | **Its own git repo** — Raspberry-Pi arm scripts (pose logging, sweep playback, joint-path shifting). Deploys to the Pi; not part of this repo's history. |

Day-to-day commands live in [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md).

## The cwd rule

**Run everything from the repo root.** Scripts anchor their data and calibration
paths to the repo root (`data/`, `calib/`), so `python src/pose/track_probe.py`
works from the root regardless of where the file lives.

The one exception is `ultrasound-cobot/` — that subproject runs from **inside its
own folder** (`cd ultrasound-cobot && python …`), since its scripts and data sit
together and it deploys to the Pi.

## Directory map

```
run_sweep.py            capture orchestrator (root entry point)
src/
  capture/              pysidecaster.py — Clarius frame capture
  pose/                 track_probe, merge_poses*, smooth_cam_log, trim_section, …
  calibration/          calibrate_handeye, calibrate_intrinsics, digitize_beads,
                        vet_frames, guided_sweep_reader, make_hold_log, solve_bridge
  segment/              vessel_*, segment_*, sam3_track*, compression_*  (+ sam3/ = vendored Meta SAM3)
  reconstruct/          reconstruct_handeye, reconstruct_volume, compound_handeye,
                        render_tubes_3d, view_*_3d, cluster_vessels
  audit/                make_audit_set, audit_tool, score_audit, harvest_exemplars
  handoff_tools/        export_teach_path
  viz/                  view_sweep*, minip, view_camera, pyimu
  experiments/          frame_timing, pose_benchmark, pose_experiment
calib/                  handeye.json, bridge.json, intrinsics.json, mycobot_320_pi.urdf
data/                   clarius_sessions/ (captures), pose_logs/ (logs)      [gitignored]
audit/                  audit_truth.json + audit_manifest.json (human truth; frames ignored)
models/                 *.pt weights
media/                  figures (*.png tracked); point clouds (*.ply/*.npz ignored)
handoff/                curated artifacts for teammates (see handoff/README.md)
attic/                  superseded bundles / one-offs — kept, never deleted
docs/                   CHEATSHEET + plans + references
ultrasound-cobot/       SEPARATE git repo — Raspberry-Pi arm scripts
```
