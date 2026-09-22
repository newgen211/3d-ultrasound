# Robotic 3D Ultrasound

A cobot arm sweeps a Clarius ultrasound probe over a phantom while a RealSense
camera tracks the probe pose (ArUco marker); per-frame poses are fused with the
B-mode frames into metric 3D reconstructions, and vessels are segmented and
scored against human ground truth.

## Getting a working copy

```bash
git clone --recurse-submodules https://github.com/newgen211/3d-ultrasound.git
# already cloned without it:
git submodule update --init
```

Two submodules: `src/segment/sam3` (Meta's SAM 3, pinned) and
`ultrasound-cobot` (the Raspberry-Pi arm scripts, its own repo).

Scripts find the repo themselves, so **you can run them from anywhere**:

```bash
python src/reconstruct/reconstruct_handeye.py section_106
python /abs/path/to/src/audit/score_bench.py          # also fine
```

They do it with a three-line `sys.path` bootstrap onto `src/`, then import the
shared `us3d` package. There is no install step, which is what keeps them
working across six conda envs and under `sudo`. `pyproject.toml` exists to
record dependencies, not because anything needs installing.

Two environment variables redirect the mutable trees, which is how you test
without touching 9.5 GB of irreplaceable captures:

```bash
US3D_DATA=/tmp/scratch/data  python src/supervise/gauge.py section_127
US3D_OUTPUTS=/tmp/scratch    python src/viz/viz_zones.py section_118
```

## Environments

Not everything runs in one env; `environment/*.yml` recreates each.

| Env | Python | Runs |
|---|---|---|
| `clarius` | 3.10 | Clarius capture (`src/capture/pysidecaster.py`), and the scorers. **PySide6 is pinned to 6.5.3** to match the Qt inside `libcast.dylib`; a mismatch segfaults. The frozen numbers in `paper/` were produced here. |
| `realsense` | 3.11 | Probe tracking and camera calibration. The only env with `pyrealsense2`. |
| `sam3` | 3.12 | The SAM 3 teacher; also needs `pip install -e src/segment/sam3`. |
| `yolo_env` | 3.9 | Detector training (older ultralytics line). |
| `3d_ultrasound` | 3.11 | Reconstruction and figures; no hardware, no detector. |

`sudo` ignores the active env, so give it the interpreter by path:

```bash
sudo PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/realsense/bin/python src/pose/track_probe.py
```

## Pipeline

| Stage | Where | What it does |
|---|---|---|
| **1. capture** | `run_sweep.py` + `src/capture/pysidecaster.py` | Fly a sweep and record raw frames + IMU + timestamps into `data/clarius_sessions/section_<N>/`. `run_sweep.py` refuses to fly if any recorder is dead. |
| **2. pose** | `src/pose/` | `track_probe.py` tracks the probe marker; `smooth_cam_log` → `trim_section` → `merge_poses_cam` attach a pose to every frame. |
| **3. calibration** | `src/calibration/` → `calib/` | `calibrate_intrinsics`, `calibrate_handeye` (flange→image), `solve_bridge` (camera↔cobot, `AX=XB`). |
| **4. segment** | `src/segment/` | Vessel segmentation, classical and SAM 3, gated to open/crescent lumen. |
| **5. reconstruct** | `src/reconstruct/` | Posed frames → metric volume + MIPs, multi-view compounding, vessel clustering. |
| **audit / score** | `src/audit/` | Human truth (`make_audit_set` → `audit_tool`), then `score_bench` / `score_classical` against it. |
| **supervise** | `src/supervise/` | The contact gauge: `gauge.py` replays it offline, `supervisor_v3.py` runs it live and sends `dz` to the Pi. |
| **labels** | `src/labels/` | Human annotation (`annotate`, `reclass`) and the SAM label filters. |
| **arm** | `ultrasound-cobot/` | **Its own repo**, runs from inside its own folder, deploys to the Pi. |

Day-to-day commands: [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md).

## Layout

```
run_sweep.py          capture orchestrator, the one script at the root
src/us3d/             shared library: paths, sections, frames, handeye, video, tube
src/<area>/           scripts, always exactly two directories under the root
  capture pose calibration segment reconstruct audit supervise labels viz
  experiments handoff_tools
calib/                handeye.json, bridge.json, intrinsics.json, urdf
data/                 captures and pose logs                      [gitignored]
audit/                audit_truth.json, audit_manifest.json, exemplars.json
models/               *.pt weights (mostly untracked, see models/README.md)
outputs/              generated: figures/, videos/, exec_logs/, pi_mirror/
paper/                frozen figures and scoring references, see paper/README.md
handoff/              curated copies for collaborators
media/  docs/  attic/  environment/
```

`audit/audit_truth.json` is canonical; `handoff/` is a curated snapshot.
Anything under `data/` is unrecoverable: it is gitignored, and there is no
backup on this machine.
