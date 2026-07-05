# Robotic 3D Ultrasound Reconstruction — Project Reference

A working reference document for the project. Phases are ordered, not time-bound.

> File paths in this document use bare script names. For where each script now
> lives in the repo, see the root [`README.md`](../README.md) file map.

---

## 1. Project Overview

**Goal:** A cobot arm performs an automated ultrasound sweep, producing an
accurate, metrically scaled 3D reconstruction of the scanned region.

**In scope (now):**

- Capture B-mode frames + probe pose from the Clarius HD3 and cobot arm
- Calibrate the system (image scale, hand-eye, timestamp sync)
- Produce a 3D voxel volume from a robot-driven sweep
- Validate against a known phantom

**Anatomical targets:**

- Phantom first (tubes in gel, known geometry)
- Radial nerve next (superficial, repeatable, small target)
- More complex anatomy later

Future scope (segmentation, autonomous target search, diagnostic outputs,
live-patient scanning) is captured in Phase 7.

---

## 2. Hardware Stack

**Clarius PAL HD3 (probe):**

- 192 piezoelectric elements, 8 beamformers
- Embedded 9-DOF IMU (separate accelerometer/gyroscope and magnetometer)
- IMU sensor is offset from the imaging plane center — exact offset is
  HD3-specific and listed in `clariusdev/motion`
- Connects via WiFi Direct or local WiFi router

**Cobot arm:**

- Provides primary pose data via end-effector kinematics
- Needs Python (or other) SDK access for motion control and pose readback
- A rigid probe mount (custom 3D-printed fixture, ideally) attaches the Clarius
  to the end-effector

**Phantom (build first):**

- Plastic tubes (3–10 mm diameter, aquarium tubing or similar) suspended in
  unflavored gelatin or ballistic gel in a sealed container
- Include features at known spacings for measurement
- Cheap, single-evening build
- Required for any validation

---

## 3. Software Stack

**Clarius access:**

- `clariusdev/cast` (GitHub) — Python Cast API for real-time streaming of B-mode
  frames + IMU + timestamps
- `clariusdev/motion` (GitHub) — IMU axis conventions, probe-specific sensor
  offsets, motion examples
- The Clarius app must be running and connected to the probe for the Cast API to work
- IMU streams at 200 Hz; imaging at 15–30 FPS; each frame is tagged with multiple
  IMU samples

**To verify with team:**

- Whether the lab has the 3D Positional Data Package and/or Raw Data Package
  (Clarius's paid research add-ons)

**Recommended tooling:**

- **PLUS Toolkit** — open-source library for tracked ultrasound, integrates with
  3D Slicer, supports Clarius, has built-in hand-eye calibration. Strongly worth
  using rather than reimplementing.
- **3D Slicer** — visualization and analysis
- **ITK-SNAP** — alternative volume viewer
- **nibabel** (Python) — NIfTI export
- **NumPy / SciPy** — voxel operations

---

## 4. Pipeline Architecture

```
Cobot arm pose (end-effector, t)
        │
        ▼
[Hand-eye transform]
        │
        ▼
Image plane pose in world (t)
        │
        ▼
For each pixel (i, j) in frame at t:
   3D position = pose × [i·sx, 0, j·sz]
        │
        ▼
Deposit intensity into voxel grid (with hit count)
        │
        ▼
Compound: weighted average / max intensity
        │
        ▼
Fill gaps (trilinear interpolation)
        │
        ▼
Export NIfTI with correct voxel sizes + orientation
```

---

## 5. Phase Plan

### Phase 1 — Plumbing

- [ ] Stream and log Clarius frames + IMU + timestamps to disk (Python Cast API)
- [ ] Verify timestamps are monotonic and IMU is non-zero
- [ ] Control the cobot arm via SDK: move, read end-effector pose
- [ ] Log arm pose + timestamp to disk at fixed rate
- [ ] Write a slow linear sweep motion (target ~1–2 cm/s, 2–5 cm total)
- [ ] Build the phantom

### Phase 2 — Calibrations (one-time, reused across sessions)

- [ ] **mm/pixel scale** (axial + lateral) — image a wire or bead at known depth,
  count pixels per mm
- [ ] **Timestamp synchronization** — align Clarius clock with cobot clock
  - Quick method: tap the probe sharply, find the IMU spike, align to arm motion
    trace, compute offset
  - Better: NTP sync between systems
- [ ] **Hand-eye calibration** — rigid transform from end-effector frame to image
  plane frame
  - Image a single bead or crossed wires from 5–10 distinct arm poses
  - Solve for the transform that makes the feature's 3D position consistent across views
  - Use PLUS Toolkit's implementation if possible
- [ ] **IMU bias estimation** (optional, for sanity check) — 30–60 s still
  recording to estimate gyro bias

### Phase 3 — First Reconstruction

- [ ] Robot-driven sweep over the phantom
- [ ] For each frame: compute image plane pose in world (arm pose × hand-eye)
- [ ] Build voxel grid (initial voxel sizes: lateral mm/px, axial mm/px,
  elevational 0.5–1.0 mm)
- [ ] For each pixel: transform to world coords, deposit into nearest voxel,
  track hit-count
- [ ] Compound: weighted average across deposits
- [ ] Fill gaps with trilinear interpolation
- [ ] Export NIfTI with correct voxel sizes and orientation
- [ ] Open in 3D Slicer, inspect visually

### Phase 4 — Validation

- [ ] Measure phantom features in the reconstruction
- [ ] Compare to known ground truth dimensions
- [ ] Target: < 2 mm error over 50 mm sweep
- [ ] Repeat sweep — check repeatability (centroid error, Dice score)
- [ ] Compare robot-driven reconstruction against a manual freehand sweep of the
  same phantom
- [ ] Tune voxel size, compounding rule (mean vs max-intensity), elevational scale

### Phase 5 — Real Anatomy

- [ ] Repeat the pipeline on the radial nerve
- [ ] Compare visual quality and measurements to clinical 2D ultrasound view
- [ ] Document any artifacts (motion, shadowing, dropout)

### Phase 6 — Sweep Path Expansion

- [ ] Test arc sweep (rotational, useful for curved anatomy)
- [ ] Test raster sweep (multiple parallel passes for wider coverage)
- [ ] Compound across multiple sweeps for noise reduction

### Phase 7 — Long-Term Extensions (out of scope now)

Each bullet is a substantial sub-project on its own:

- [ ] **Anatomical segmentation** — train or adapt a model (e.g., U-Net) for the
  chosen target
- [ ] **Force/torque feedback** — add an F/T sensor for safe skin contact and
  contact-pressure control
- [ ] **Patient body detection** — add a depth or RGB camera for surface and
  landmark recognition
- [ ] **Autonomous target search** — given a rough body region, find target
  anatomy automatically (heuristic, then learned policies)
- [ ] **Real-time orientation optimization** — adjust probe angle dynamically
  based on image quality
- [ ] **Diagnostic classification** — task-specific outputs for the target anatomy
- [ ] **Clinical validation** — IRB approval, healthy volunteers, then patients

---

## 6. Calibrations — Quick Reference

| Calibration | Purpose | Method | When |
| --- | --- | --- | --- |
| mm/pixel (axial + lateral) | Physical scale of image | Image wire/bead at known depth | Once per probe/preset |
| Timestamp sync | Align Clarius and cobot clocks | Probe tap + IMU spike, or NTP | Each session start |
| Hand-eye transform | Image plane position in arm frame | Bead from 5–10 arm poses, solve | Once per probe mount |
| IMU bias | Correct gyro drift | 30–60 s stationary | Each session start (optional) |
| Elevational scale check | Verify reconstruction distance fidelity | Sweep over known 10–20 mm feature | After each major change |

---

## 7. Validation Targets

| Metric | Target | Method |
| --- | --- | --- |
| Tube diameter error | ≤ 2 mm | Measure in reconstructed volume |
| Spatial error over 50 mm sweep | ≤ 2 mm | Compare known feature spacings |
| Repeatability between sweeps | Visual + centroid agreement | Two sweeps of same phantom |
| Cumulative drift (if checking IMU) | < 2–3 mm | Compare IMU-only vs arm-pose reconstruction |
| Quality vs manual baseline | Comparable or better | Robot sweep vs freehand sweep of same target |

---

## 8. Standing Questions (Confirm With Supervisor)

**Project scope and definition:**

- [ ] One-sentence definition of done — what specific measurable outcome counts as success?
- [ ] Repeatability requirement — across sessions, days, operators?
- [ ] Publication, demo, or grant deadlines driving timeline?
- [ ] How far does the long-term roadmap extend — just 3D reconstruction, or
  eventual autonomous scanning?

**Hardware and resources:**

- [ ] Access to Clarius 3D Positional Data Package and Raw Data Package
- [ ] Existing probe-to-arm mount, or design and 3D print from scratch?
- [ ] CAD model of cobot end-effector available?
- [ ] Cobot model and SDK details

**Future scope clarification:**

- [ ] Once segmentation is in scope, which anatomy first?
- [ ] Is live-human scanning ever in scope? If so, what's the IRB pathway?

---

## 9. Risks and Things That Kill Projects Like This

- **Timestamp sync errors** — silent killer. Misaligned poses produce blurry
  reconstructions that look "almost right" and waste weeks. Validate sync early
  and often.
- **Probe mount flex** — any mechanical compliance between the arm and the probe
  invalidates the hand-eye calibration. Make the mount rigid.
- **Skipping the phantom** — without ground truth, you can't tell good
  reconstruction from bad. Build the phantom before writing reconstruction code.
- **Scope creep** — the long-term roadmap is tempting (segmentation, autonomy,
  diagnosis). Stay disciplined: phantom volume with measured error first, anything
  else second.
- **No artifacts at check-ins** — every weekly check-in should produce a
  screenshot, a measurement, or a number. "I'm working on it" for multiple weeks
  is a sign of stalling.

---

## 10. References and Resources

**Clarius:**

- `clariusdev/cast` — https://github.com/clariusdev/cast
- `clariusdev/motion` — https://github.com/clariusdev/motion
- Clarius research toolkit page — https://clarius.com/scanners/research/

**3D Ultrasound tooling:**

- PLUS Toolkit — https://plustoolkit.github.io/
- 3D Slicer — https://www.slicer.org/

**Background reading:**

- Su et al., *A fully autonomous robotic ultrasound system for thyroid scanning.*
  Nature Communications 15, 4004 (2024). https://doi.org/10.1038/s41467-024-48421-y
- Freehand 3D ultrasound review — https://pmc.ncbi.nlm.nih.gov/articles/PMC5385255/

**Local files:**

- `pysidecaster.py` — local Clarius reference script (now at
  [`src/capture/pysidecaster.py`](../src/capture/pysidecaster.py))

---

## 11. Glossary

- **B-mode** — standard grayscale ultrasound image (Brightness mode)
- **Hand-eye calibration** — rigid transform between two coordinate frames,
  typically end-effector ↔ image plane
- **Voxel** — 3D pixel; volumetric element
- **NIfTI** — Neuroimaging Informatics Technology Initiative; standard 3D medical
  image file format (`.nii`)
- **IMU** — Inertial Measurement Unit; accelerometer + gyroscope (+ magnetometer
  for 9-DOF)
- **6-DoF pose** — position (x, y, z) + orientation (roll, pitch, yaw)
- **Compounding** — combining multiple frames covering the same voxel into a
  single value
