# Robotic 3D Ultrasound

A low-cost robotic ultrasound system: a **myCobot 320** arm sweeps a **Clarius PAL
HD3** probe over a phantom, a **RealSense + ArUco marker** measures probe pose, and
the captured B-mode frames are reconstructed into a metrically accurate 3D volume
for measurement in 3D Slicer. Phantom first, real anatomy later.

> **New here? Read the docs in this order:**
> 1. [docs/phase-1.md](docs/phase-1.md) — what we're trying to do *right now*
> 2. [docs/master-plan.md](docs/master-plan.md) — the full plan, status, and critical path
> 3. [docs/project-reference.md](docs/project-reference.md) — hardware/software, calibration & validation reference

---

## ⚠️ How to run (read this first)

**Always run scripts from the project root** (this directory). Every script
resolves data paths relative to the current working directory, *not* to its own
location — so launch from here:

```bash
cd "/Users/daryldocteur/Projects/Laura lab/3d Ultrasound"
python src/<stage>/<script>.py [section_N | path] [flags]
```

**Environment / SDK constraints** (hard-won, see pitfalls P13–P14 in the master plan):

- **NumPy < 2** and **PySide6 6.5.3** for the Clarius capture path.
- The matched Clarius binaries **`libcast.dylib`** + **`pyclariuscast.so`** must
  stay in the project **root** (they are loaded from the working directory). Don't
  move them.
- Keep the project **local** — never in iCloud (it dehydrates the binaries).
- RealSense scripts need the `realsense` env + a connected camera.
- The Clarius app must be running and connected to the probe for capture.

---

## Pipeline at a glance

```
 capture            pose                 calibrate            reconstruct          segment / validate
 ───────            ────                 ─────────            ───────────          ──────────────────
 pysidecaster  →    merge_poses     →    calibrate_handeye →  reconstruct_handeye → segment_tube
 (frames+IMU)       (cobot pose)         (flange→image)       (metric 3D volume)    vessel_centerline
                    merge_poses_cam      [handeye.json]       compound_handeye      vessel_to_slicer
                    (RealSense pose)                          (multi-sweep merge)   → 3D Slicer
```

---

## Repository layout

```
3d Ultrasound/
├── README.md                  ← you are here (project map + how to run)
├── docs/                      ← the "why", the plan, the status
│   ├── phase-1.md
│   ├── master-plan.md
│   └── project-reference.md
├── src/                       ← all code, grouped by pipeline stage
│   ├── capture/               ← Clarius frame capture
│   ├── calibration/           ← image scale + hand-eye
│   ├── pose/                  ← marker tracking + timestamp-merge poses into frames
│   ├── reconstruct/           ← posed frames → 3D voxel volume
│   ├── segment/               ← vessel detection / centerline / Slicer export (KEEP TOGETHER)
│   ├── viz/                   ← viewers and projections
│   └── experiments/           ← latency / pose-source measurements
├── data/                      ← all data (gitignored)
│   ├── clarius_sessions/      ← captured sweeps: section_N/ (raw frames, sidecars, volumes)
│   ├── pose_logs/             ← cobot + RealSense pose logs (*.jsonl)
│   ├── realsense/             ← color.png / depth.png snapshots
│   └── freehand/              ← early freehand-reconstruction experiments + recordings
├── outputs/                   ← generated figures & screenshots
│   ├── figures/
│   └── screenshots/
├── media/                     ← phantom photos, demo videos (gitignored)
├── handeye.json               ← current best hand-eye (flange→image), RMS 2.86 mm
├── libcast.dylib              ← Clarius SDK binary (must stay at root)
├── pyclariuscast.so           ← Clarius SDK binding (must stay at root)
├── pri.pem  /  tls.crt        ← Clarius TLS material
└── .gitignore
```

> **Why some things stay at root:** `handeye.json` is the default fallback
> calibration that 5 scripts read; the SDK binaries and TLS files are loaded from
> the working directory by the capture scripts. Moving them would break those
> scripts, so they stay put.

---

## Script index

Status: ✅ done/validated · 🟡 prototype · 🧪 experiment/one-off · ⚪ superseded (kept for reference).
All commands assume you are in the project root. `section_N` can be a bare name
(resolved under `data/clarius_sessions/`) or a full path.

### `src/capture/` — capture

| Script | | Purpose | Output |
| --- | --- | --- | --- |
| [pysidecaster.py](src/capture/pysidecaster.py) | ✅ | Clarius Cast SDK GUI: streams B-mode frames + IMU + timestamps. `python src/capture/pysidecaster.py` | `data/clarius_sessions/section_N/` (raw_*.bin/.json, proc_*.png, manifest/connection.json) |

### `src/calibration/` — calibration

| Script | | Purpose | Output |
| --- | --- | --- | --- |
| [digitize_beads.py](src/calibration/digitize_beads.py) | ✅ | Interactive: click the calibration bead in each frame. `python src/calibration/digitize_beads.py section_N` | `section_N/handeye_clicks.json` |
| [calibrate_handeye.py](src/calibration/calibrate_handeye.py) | ✅ | Solve flange→image hand-eye from the clicks (tries all Euler conventions). `python src/calibration/calibrate_handeye.py section_N` | `section_N/handeye.json` (current best copied to root `handeye.json`, RMS 2.86 mm) |

### `src/pose/` — pose tracking & merge

| Script | | Purpose | Output |
| --- | --- | --- | --- |
| [track_probe.py](src/pose/track_probe.py) | ✅ | RealSense ArUco marker tracker (flip-suppressed 6-DoF + depth fusion). `python src/pose/track_probe.py` | appends `data/pose_logs/probe_pose_log.jsonl` |
| [merge_poses.py](src/pose/merge_poses.py) | ✅ | Nearest-timestamp merge of **cobot** poses into frame sidecars. `python src/pose/merge_poses.py section_N data/pose_logs/pose_log.jsonl` | writes `cobot_pose` into each `raw_*.json` |
| [merge_poses_cam.py](src/pose/merge_poses_cam.py) | ✅ | Same, but merges **RealSense marker** poses (into the same `cobot_pose` field). `python src/pose/merge_poses_cam.py section_N data/pose_logs/probe_pose_log.jsonl` | updates `raw_*.json` |
| [quckirun.py](src/pose/quckirun.py) | 🧪 | One-off pose-log cleaner (trim lift-off, drop rx flips). `python src/pose/quckirun.py` | `data/pose_logs/sweep_clean.jsonl` |

### `src/reconstruct/` — reconstruction

| Script | | Purpose | Output |
| --- | --- | --- | --- |
| [reconstruct_handeye.py](src/reconstruct/reconstruct_handeye.py) | ✅ **canonical** | Full 6-DoF metric reconstruction (`p_world = T + Rf·(R_X·p_img + t_X)`). `python src/reconstruct/reconstruct_handeye.py section_N` | `section_N/volume_handeye.nii.gz`, `.npy`, `_mips.png` |
| [compound_handeye.py](src/reconstruct/compound_handeye.py) | 🟡 | Multi-sweep shared-grid **max-merge** compounding (the multi-angle unlock, task D2). `python src/reconstruct/compound_handeye.py section_A section_B …` | `data/clarius_sessions/compound_*/volume_compound.*` |
| [reconstruct_volume.py](src/reconstruct/reconstruct_volume.py) | ⚪ | Older non-hand-eye reconstructor (PCA / assumed span). Superseded by `reconstruct_handeye.py`. | `section_N/volume.*` |

### `src/segment/` — vessel segmentation *(these 4 import each other — keep co-located)*

| Script | | Purpose | Output |
| --- | --- | --- | --- |
| [segment_tube.py](src/segment/segment_tube.py) | 🟡 | Detect/track the anechoic vessel, measure radius. Also the shared module (`candidates`, `load_frame`, `find_section`, `track`). `python src/segment/segment_tube.py section_N` | `section_N/tube_seg_overlay.png` |
| [vessel_centerline.py](src/segment/vessel_centerline.py) | 🟡 | Project per-frame centroids → 3D centerline; report straightness. | `section_N/vessel_centerline.png` |
| [vessel_tube.py](src/segment/vessel_tube.py) | 🟡 | Centerline → smoothed 3D tube mesh; clean spread/length. | `section_N/vessel_tube.png` |
| [vessel_to_slicer.py](src/segment/vessel_to_slicer.py) | 🟡 | Export Slicer markups curve + tube model in base-frame mm. | `section_N/vessel_centerline.mrk.json`, `vessel_tube.vtk` |

### `src/viz/` — viewers & projections

| Script | | Purpose |
| --- | --- | --- |
| [minip.py](src/viz/minip.py) | ✅ | MinIP/MIP projections of `volume_handeye.npy` (MinIP shows dark/anechoic tubes) → `volume_minip.png`. |
| [view_camera.py](src/viz/view_camera.py) | ✅ | Live RealSense color preview (sanity check). |
| [view_sweep.py](src/viz/view_sweep.py) | ⚪ | Frame-montage of a sweep → `preview_montage.png`. |
| [view_sweep_3d.py](src/viz/view_sweep_3d.py) | ⚪ | IMU-only 3D frame-stack preview (not metric) → `preview_3d.png`. |
| [pyimu.py](src/viz/pyimu.py) | 🧪 | Qt3D demo rotating a model by the probe IMU. (Needs `scanner.obj`/`.mtl`, not in repo; no entry point — demo only.) |

### `src/experiments/` — measurements

| Script | | Purpose |
| --- | --- | --- |
| [frame_timing.py](src/experiments/frame_timing.py) | ✅ | Capture fps / jitter / hiccups from a section's timestamps. `python src/experiments/frame_timing.py section_N` |
| [pose_benchmark.py](src/experiments/pose_benchmark.py) | 🧪 | Compare cobot vs camera pose source (static jitter + Kabsch-aligned RMS). `python src/experiments/pose_benchmark.py data/pose_logs/cobot_log.jsonl data/pose_logs/probe_pose_log.jsonl` → `outputs/figures/pose_benchmark.png` |
| [pose_experiment.py](src/experiments/pose_experiment.py) | 🧪 | Live RealSense marker jitter + known-displacement accuracy test. |

---

## Data layout

**A captured session** — `data/clarius_sessions/section_N/`:

| File | What |
| --- | --- |
| `raw_<ts>.bin` | Raw polar B-mode frame bytes |
| `raw_<ts>.json` | Per-frame sidecar: dims, scale (`axial_um_per_sample`, `lateral_um_per_line`), IMU samples, `host_timestamp_ns`, `probe_timestamp_ns`. The pose-merge step adds a `cobot_pose` field. |
| `proc_<ts>.png` | Scan-converted display image |
| `manifest.json`, `connection.json` | Sweep + connection metadata |
| `volume_handeye.{nii.gz,npy}`, `*_mips.png` | Reconstruction outputs (after `reconstruct_handeye.py`) |
| `handeye.json`, `handeye_clicks.json` | Present in calibration sessions (e.g. `section_28`) |

**Pose logs** — `data/pose_logs/`: `pose_log.jsonl` (cobot, copied from the Pi),
`probe_pose_log.jsonl` (RealSense marker), `cobot_log.jsonl`, `old_probe_pose_log.jsonl`,
`sweep_clean.jsonl` (cleaned), `center_pose.jsonl`.

---

## Where the project stands

Capture, pose-merge, hand-eye calibration, and single-sweep metric reconstruction
are **done and validated** (12 mm tube confirmed in Slicer on `section_22`). The
open work is multi-sweep compounding, vessel segmentation on a representative
phantom, detect-and-plan autonomy, and force control. Full status table and the
D1–D10 work plan are in [docs/master-plan.md](docs/master-plan.md) §5 and §7.

**Don't break these (project gotchas):**

- Run from the project root; paths are cwd-relative.
- Keep `libcast.dylib`, `pyclariuscast.so`, `handeye.json` at the root.
- Keep the four `src/segment/` modules co-located (they import each other).
- NumPy < 2, PySide6 6.5.3, matched Cast binaries, no worker threads, no iCloud.
