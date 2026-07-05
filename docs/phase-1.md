# Phase 1 — Immediate Focus

> **The near-term ask (per supervisor):** get the cobot arm to scan a phantom and
> produce usable data. That's it. Everything downstream (autonomy, segmentation,
> path planning, force control) is later — see [`master-plan.md`](master-plan.md).

This is the disciplined, ship-an-artifact version of **Phase 1 — Plumbing** from
[`project-reference.md`](project-reference.md). The whole point of Phase 1 is a
**working scan of the phantom by the arm**, captured to disk and reconstructable.

## Definition of done for Phase 1

A single robot-driven sweep over a phantom that yields:

1. A `data/clarius_sessions/section_N/` folder of B-mode frames + per-frame
   sidecars (timestamps, scale, IMU) — **already working** via
   [`pysidecaster.py`](../src/capture/pysidecaster.py).
2. A cobot pose log merged into those frames — **already working** via
   [`merge_poses.py`](../src/pose/merge_poses.py).
3. A 3D volume you can open in 3D Slicer — **already working** via
   [`reconstruct_handeye.py`](../src/reconstruct/reconstruct_handeye.py)
   (validated on `section_22`: 12 mm tube confirmed).

So the capture/pose/reconstruct plumbing is **done and validated**. What remains
for a clean Phase 1 deliverable on the *intended* phantom:

## Checklist (what's left to actually close Phase 1)

- [ ] **Build / receive the representative phantom.** The Echonect arm model
      (12 vessels, 2–6 mm) is ~2 weeks out; until then, a thin water-filled tube in
      speckled gelatin is the right interim target (a clean anechoic circle, no
      shadow — unlike the vinyl tube in `section_22`).
- [ ] **Mount the probe rigidly to the end-effector.** Any flex invalidates the
      hand-eye calibration (pitfall P3).
- [ ] **Run a slow, steady robot sweep** over the phantom (~1–2 cm/s, 2–5 cm
      total) and capture it.
- [ ] **Merge poses and reconstruct**, open in Slicer, eyeball that the geometry
      looks right.
- [ ] **Produce an artifact** — a screenshot + a measured number — for the
      check-in. (Every check-in should produce a screenshot, a measurement, or a
      number.)

## How to run the Phase-1 pipeline

All commands run **from the project root** (see the root [`README.md`](../README.md)):

```bash
# 1. Capture (Clarius app connected; SDK env active) — writes data/clarius_sessions/section_N/
python src/capture/pysidecaster.py

# 2. Merge cobot poses into the frames (after copying pose_log.jsonl from the Pi)
python src/pose/merge_poses.py section_N data/pose_logs/pose_log.jsonl

# 3. Reconstruct the metric volume -> opens in 3D Slicer
python src/reconstruct/reconstruct_handeye.py section_N
```

## Note on the phrasing

The original note read *"all she wants is a phantom that scans the arm."* The
Echonect phantom is itself an **arm model**, so the deliverable is read as: **the
robot arm scanning the (arm-shaped) phantom and producing a reconstruction.** If
the supervisor meant something narrower (e.g. just *capture working*, no
reconstruction needed), confirm at the next check-in — but the pipeline above
already covers both readings.
