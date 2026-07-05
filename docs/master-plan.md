# Autonomous Robotic 3D Ultrasound — Master Plan

From scratch → where we are → where we need to be. Architecture, work plan,
pitfalls, risks, timeframes, and a critical-path summary at the end.

> File paths in this document use bare script names (e.g. `reconstruct_handeye.py`).
> For where each script now lives in the repo, see the root
> [`README.md`](../README.md) file map.

---

## 0. Purpose of this document

One source of truth for the autonomous ultrasound project. It exists so that at
any point you (or a teammate, or a future you) can answer: what are we building,
why, what's done, what's next, what will bite us, and how long it takes. Phases
are ordered by dependency, not calendar.

---

## 1. The goal, stated plainly

A low-cost robotic system that autonomously scans a tubular structure (a
vessel/tube in a phantom), plans a path along it, executes that path while
holding probe contact, and builds an accurate 3D ultrasound volume of it.
Phantom first, real anatomy later. "Accurate" means metrically correct enough to
measure in 3D Slicer — target ~2–3 mm, matching the arm's realistic accuracy floor.

The autonomy lives in **detecting the vessel in the ultrasound and planning where
to scan** — not in a blind pre-set sweep. The acquisition itself is open-loop:
survey → detect/pick vessel → plan path → execute → reconstruct (see the
2026-06-12 changelog for why we dropped real-time servoing).

---

## 2. The thesis (why this is a paper, not a tutorial)

Open-loop sweep-and-merge 3D ultrasound is a solved, 30-year-old technique —
there is nothing novel left there. Autonomous image-guided tube-following has been
done (Jiang et al. 2022, the benchmark: real-time U-Net vessel segmentation,
auto-centering, probe oriented normal to the vessel, radius estimation, ~1.95 mm /
3.3° error on a gel phantom). But nearly all of it runs on research-grade arms
(Franka, KUKA) with reliable kinematics and proper force/torque sensing.

The open niche — and your contribution — is the **low-cost, imprecise platform**:
a ~$700 myCobot 320, a Clarius portable probe, a RealSense, and an off-the-shelf
phantom. The literature explicitly names unreliable robot kinematics as the thing
that wrecks reconstruction accuracy on compliant arms. Your answer is **external
visual pose correction** (RealSense + ArUco marker) that gives a cheap floppy arm
the effective pose accuracy the task needs. The weak arm is not a limitation to
apologize for — it is the premise of the paper:

> "Autonomous tubular-structure ultrasound on a low-cost compliant manipulator
> using external visual pose correction to compensate for unreliable kinematics."

That is reproducible, accessible to other labs, and not crowded.

---

## 0.5 Current understanding & changelog — 2026-06-12

*The latest is authoritative; older sections below are kept for context but read
this first where they conflict.*

**Architecture changed: dropped real-time continuous servoing → open-loop.**
The acquisition is now survey → detect → plan → execute: coarse survey scan,
detect/pick the target vessel, extract its centerline, plan a probe path, execute
it open-loop while the RealSense logs pose per frame, then reconstruct. The novelty
is the detect-and-plan step on a low-cost arm. This removes three big risks at
once — stale-data-in-the-loop, force-loop instability, and laggy closed-loop
oscillation — because nothing is being corrected in real time.

**Latency is largely de-risked.** `section_22` captured at ~20 fps (50 ms/frame)
— capture is not the bottleneck. The "0.5 s" is downstream cobot settle time,
which only matters per move command; in open-loop execution you issue smooth moves
and frames stream at 20 fps throughout, so it mostly falls out of the critical
path. (Still worth measuring the cobot-settle number when the arm is reconnected —
it's now a tuning value, not a risk.)

**Multi-angle is re-confirmed as a goal, not scope creep.** It's the fix for
acoustic shadowing: a different angle puts the shadow elsewhere, and compounding
(max-intensity, multi-sweep — the D2 task) fills in what each view missed.

**Hardware identified** (arrives ~2 weeks; gelatin until then):

- **Phantom:** Echonect arm model — 12 vessels, 2–6 mm dia., US-compatible
  polymer, curved arm contour.
- **Force sensor:** Alpha MF01A FSR, 0.3–10 N, single-axis. Needs an ADC (ADS1115
  over I²C to the Pi). Coarse → "maintain a contact band + abort if too hard," not
  precision force control. 0.3 N floor may already compress a vessel.

**New constraints these introduce:**

- **Pose error vs. vessel size is the accuracy crux.** Hand-eye RMS is 2.86 mm;
  vessels are 2–6 mm. You can probably measure the largest vessels; the 2 mm ones
  are at/below the noise floor. Demo 1 targets the largest vessel.
- **Twelve vessels → a selection rule is needed** (largest / user-tap / nearest).
  The survey scan makes this easy: map them all, then pick.
- **Curved surface → defer to the flat top of the phantom for demo 1.**

**Segmentation reality check (`section_22`):** classical detector hit only 18%,
because the clear vinyl tube (thick wall / likely air) shadows, so tube+shadow
reads as a vertical dark column, not a clean circle (MinIP confirmed). `section_22`
is an unrepresentative segmentation target — don't tune on it. The fix is a thin
water-filled tube in speckled gelatin (clean anechoic circle, no shadow, resembles
the real 2–6 mm vessels). Build pending.

**Demo 1 scope (locked):** largest vessel, flat top, fixed probe orientation,
light fixed force. Every constraint dropped later is a paper improvement.

**New tube-material learning:** clear vinyl / thick-walled / air-filled tubing
reflects hard and shadows; thin, soft, water-filled tubing images clean.

---

## 3. System architecture

### 3.1 The components and their jobs

| Component | Role | Notes |
| --- | --- | --- |
| myCobot 320-Pi | Moves the probe | Cheap, compliant, imprecise. Serial `/dev/ttyAMA0` @115200. |
| Clarius PAL HD3 | Produces B-mode frames + scale + IMU | SDK on the Mac. Scale embedded per frame. |
| RealSense D435i | External pose of the probe (via marker) | RGB + factory intrinsics. Runs on the Mac. |
| ArUco marker | Rigid tag on the probe | Marker pose = probe pose. Must be rigid. |
| Force sensor (incoming) | Contact force for pressure control | Enables admittance control. |
| Mac | Capture + reconstruction + control host | Same clock for Clarius frames and RealSense poses. |
| 3D Slicer | Validation / measurement | Ground-truth comparison. |

### 3.2 The key structural insight — two pose sources, two jobs

These are separate and must not be conflated:

- **Control loop** (keep tube centered, hold force, advance along it) works in
  *relative* image-space and force-space. It issues small relative moves to the
  cobot and never needs accurate global pose. The cheap arm is fine here.
- **Reconstruction** needs *accurate absolute pose* per frame to place each slice
  correctly in 3D. That is the RealSense marker's job, not the arm's.

> The cheap arm controls; the camera measures. Neither has to be good at the
> other's job. This split is what makes a low-cost arm viable.

### 3.3 The open-loop flow (survey → plan → execute)

*(Superseded the earlier real-time correction loop — see the 2026-06-12 changelog.)*

```
  SURVEY                PLAN (offline, between moves)        EXECUTE (open-loop)
  ──────                ────────────────────────────         ───────────────────
  coarse scan over  →   segment vessels in survey       →    drive the planned path
  the region            pick target (largest / tap)          at slow steady speed
        │               extract centerline                        │
        ▼               generate probe path                       ▼
  [frames + RealSense                                       [frames stream @ ~20 fps
   pose logged]                                              + RealSense pose logged]
                                                                  │
                              FSR keeps contact in a band         ▼
                              (abort if force too high)     reconstruct vessel volume
                                                            → measure in Slicer
```

Nothing is corrected in real time, so the ~0.5 s cobot-settle latency only costs a
little at the start/end of moves — it is no longer a stability risk. Frames are
timestamp-merged to poses exactly as today (`merge_poses.py`).

---

## 4. PART A — Foundation (built from scratch, COMPLETE)

This is what already exists, in dependency order. If starting from zero, this is
the order to build it; for you it is done and validated.

- **A1. Capture pipeline (Phase 1, done).** `pysidecaster.py` streams synchronized
  B-mode frames from the Clarius SDK, writes per-frame raw_*.bin + raw_*.json
  sidecars with `host_timestamp_ns`, embedded scale (`axial_um_per_sample`,
  `lateral_um_per_line`), and IMU samples. Hard-won SDK constraints are documented
  (matched binary pairs, NumPy <2, PySide6 6.5.3, local files only, no worker
  threads on SDK calls).
- **A2. Cobot integration + pose logging (Phase 2, done).** Poses logged on the Pi
  to `pose_log.jsonl` (`t_ns`, coords, angles), copied to the Mac. `merge_poses.py`
  matches each frame's `host_timestamp_ns` to the nearest pose `t_ns` within 100 ms
  and writes `cobot_pose` into each sidecar.
- **A3. Hand-eye calibration (Phase 2, done).** `calibrate_handeye.py` on the
  `section_28` bead phantom: RMS 2.86 mm, Euler convention xyz (confirmed
  empirically), flange→image translation ≈ [+31.0, +60.6, +48.2] mm. This is the
  arm's realistic accuracy floor — accepted as a defensible result.
- **A4. Metric reconstruction (Phase 2, done, validated).** `reconstruct_handeye.py`
  places every pixel at its true world location using full 6-DOF pose:
  `p_world = T + Rf @ (R_X @ p_image + t_X)`. Validated on tube phantom
  `section_22` — 12 mm tube diameter confirmed in Slicer, all three axes correct,
  depth anomaly correctly diagnosed as acoustic shadowing (not scale error).
  Per-frame orientation is already handled here — this was a prior open question
  and it is closed.
- **A5. Superseded paths (keep for reference, don't build on).**
  `reconstruct_volume.py` uses R=identity + PCA sweep-axis placement
  (`--orient cobot` is experimental relative-tilt only); `view_sweep_3d.py` is an
  IMU-only preview. Both predate the hand-eye work. `reconstruct_handeye.py` is the
  canonical reconstructor.

---

## 5. PART B — Where we are now (exact status)

| Piece | Status | Reality |
| --- | --- | --- |
| Capture | ✅ Done | Solid, validated. |
| Pose logging + timestamp merge | ✅ Done | 100 ms tolerance works. |
| Hand-eye (flange→image) | ✅ Done | 2.86 mm RMS, xyz. |
| Single-sweep metric reconstruction | ✅ Done | Full pose, validated on `section_22`. |
| Live incremental reconstruction | 🟡 Built, untested on hardware | `live_reconstruct.py` — nearest-voxel splat, throttled MIP view, replay mode. |
| RealSense marker tracking | 🟡 Built, not calibrated | `track_probe.py` — logs marker pose in merge_poses format. Needs marker→image hand-eye. |
| Cobot motion control | 🟡 Re-scoped | Jog/sweep controller scrapped; open-loop path execution instead (no real-time loop). |
| Multi-sweep / multi-angle compounding | 🔴 Not built | `reconstruct_handeye.py` does ONE sweep into ONE grid with averaging. Needed for shadow recovery. |
| Tube segmentation | 🟡 Prototyped, target was bad | `segment_tube.py` built; 18% on `section_22` (shadowing vinyl tube — unrepresentative). Re-test on water-tube gelatin. |
| Vessel detection/path planning | 🔴 Not started | The autonomy/novelty step. |
| Force contact control | 🔴 Not started | FSR + ADS1115 ADC; coarse contact band. Sensor ~2 weeks out. |
| Full autonomous run | 🔴 Not started | The integration target. |

**Known hard facts that shape everything:**

- Capture ≈ 20 fps (50 ms/frame) — measured on `section_22`, healthy. The "0.5 s"
  is downstream cobot settle, and open-loop execution keeps it off the critical
  path. (Measure the settle number precisely when the arm is reconnected.)
- The arm is compliant and imprecise; servos sag when released; serial is
  single-threaded (no concurrent `get_coords`/`send_coords`).
- Pose error (2.86 mm) vs. vessel size (2–6 mm) is the accuracy crux — demo targets
  the largest vessel.
- IMU-on reconstruction can degrade nearly-straight sweeps (noise > benefit).

---

## 6. PART C — Where we need to be (end state)

**The demo:** the arm autonomously finds the phantom's vessel, follows it end to
end at slow speed (~1–3 mm/s), continuously re-centering it in the image, holding a
target contact force, and orienting roughly normal to it — while a 3D volume of the
vessel builds live on screen with a radius readout. Afterward the volume is exported
and measured in 3D Slicer against the phantom's known geometry.

**Success metrics (the bar):**

- Vessel followed end-to-end without losing it.
- Reconstructed vessel radius/diameter within ~2–3 mm of ground truth in Slicer.
- Probe orientation within a few degrees of vessel-normal (Jiang's bar: ~3°).
- Repeatable across ≥3 runs (Phase 4 repeatability).

**The paper:** the platform + the external-vision-correction method + the phantom
results, positioned against Jiang et al. as "comparable autonomy and accuracy on an
order-of-magnitude cheaper, compliant platform."

---

## 7. PART D — The work plan (detailed)

Each task: what · why · how · depends on · pitfalls · risk · rough time. Grouped by
what unblocks it. Time assumes a part-time research-student pace (~10–15 hr/week)
and is calendar time including the inevitable debugging.

### GROUP 1 — Unblocked, do now (pure software + existing hardware)

**D1. Measure real loop latency.**
- *Why:* Your whole story is "we made a slow, imprecise arm work." 0.5 s is a
  guess; you need the real number and its breakdown.
- *How:* Time the two legs separately — (a) Clarius frame-request → frame-in-hand,
  (b) `send_coords` → arm settled within tolerance. Log 100 cycles, report
  mean/median/max.
- *Depends on:* nothing.
- *Pitfalls:* the two legs have different fixes; don't average them into one
  meaningless number. Settle-time depends on move size — measure at the small
  increments servoing will actually use.
- *Risk:* Low effort, high information.
- *Time:* 1–2 days.

**D2. Multi-sweep shared-grid compounding (the real multi-angle unlock).**
- *Why:* `reconstruct_handeye.py` reconstructs ONE sweep into ONE grid with
  per-frame averaging. Multi-angle requires multiple sweeps merged into ONE shared
  voxel grid, and max-intensity (or similar) merging instead of averaging —
  averaging washes out angle-dependent reflections, which is the whole point of
  multi-angle.
- *How:* Extend the reconstructor to (1) accept N sections, (2) compute a shared
  world-frame bounding box across all, (3) splat all frames into the same grid,
  (4) merge per-voxel by max (or weighted-max) across sweeps. All sweeps already
  share the cobot base frame via the same hand-eye, so registration is free.
- *Depends on:* nothing (uses existing posed data).
- *Pitfalls:* averaging vs max changes the result character — validate against a
  known target. Memory note about "R=identity" was about the old path; per-frame R
  is already correct in handeye recon, so this is purely the merge step.
- *Risk:* Medium effort, low risk — well-defined.
- *Time:* 3–5 days.
- *(Status: `compound_handeye.py` is the prototype for this.)*

**D3. RealSense marker→image hand-eye calibration.**
- *Why:* The RealSense is your accurate pose source; the marker pose means nothing
  for reconstruction until you know marker→image-plane.
- *How:* Same `calibrate_handeye.py` routine you ran for `section_28`, but feed
  marker poses (from `track_probe.py`) instead of flange coords, over the bead
  phantom. Validate by reconstructing a hand-sweep and comparing to the cobot-pose
  version.
- *Depends on:* RealSense mounted, marker rigidly on probe, `track_probe.py` running.
- *Pitfalls:* marker must be rigid (P3); `MARKER_MM` must be measured, not nominal
  (P4); camera must see the marker through the whole calibration motion (P5
  occlusion).
- *Risk:* Medium — calibration quality gates reconstruction accuracy.
- *Time:* 3–5 days (including a re-do or two).

**D4. Control-loop skeleton with stubs.**
- *Why:* Get the architecture, timing, and arm command path working before the
  phantom and sensors land.
- *How:* Implement the step → frame → `segment()` → `force()` → correction →
  command loop with `segment()` and `force()` as stubs returning canned values.
  Run it against the real arm doing trivial corrections. Confirm the loop holds a
  stable rate at your measured latency.
- *Depends on:* D1 (latency), arm.
- *Pitfalls:* single-threaded serial (P6) — frame grab, pose read, and arm command
  must be sequenced, not threaded. Loop-rate instability if any call blocks.
- *Risk:* Medium — this is where real-time reality hits.
- *Time:* 1 week.

**D5. Validate `live_reconstruct.py` on hardware.**
- *Why:* It's built but only replay-tested. Confirm it keeps up at real frame rates
  and renders live.
- *How:* Run against a live (hand or arm) sweep; confirm the throttled MIP view
  updates and the volume matches the batch version.
- *Depends on:* D3 (for accurate live pose).
- *Time:* 2–3 days.

### GROUP 2 — Gated on the Amazon phantom

**D6. Tube/vessel segmentation.**
- *Why:* The control loop's `segment()` stub needs to become real — find the
  vessel, return centroid + radius per frame.
- *How:* Start classical — the vessel is anechoic (dark); threshold + contour +
  ellipse fit gives centroid and radius fast and explainably. Escalate to a small
  U-Net only if classical fails on the realistic phantom (Jiang used U-Net; you can
  cite that path).
- *Depends on:* phantom.
- *Pitfalls:* the realistic arm phantom is much harder than a clean gel tube —
  speckle, tissue-mimicking clutter, multiple structures, the vessel may collapse
  under probe pressure (P7). Classical may not generalize; budget for the U-Net
  fallback. Must run inside the 0.5 s budget (P8).
- *Risk:* High — this is the perception risk and the most likely place to stall.
- *Time:* 1–3 weeks (classical fast; U-Net path longer, needs labeled frames).

**D7. Vessel detection + path planning (replaces real-time servoing).**
- *Why:* The autonomy/novelty — from the survey, decide what to scan and where to
  move, offline, between motions. No real-time correction loop.
- *How:* From the survey volume/frames, pick the target vessel (largest / tap /
  nearest), extract its centerline, and generate a probe path along it. Execute
  open-loop; RealSense logs pose.
- *Depends on:* D6, survey scan.
- *Pitfalls:* centerline extraction failing on a curved/branching vessel; selecting
  the wrong vessel among the twelve (P10). Path that drifts off the vessel because
  the plan was built from a noisy survey.
- *Risk:* Medium — much lower than the old real-time-servoing version, since latency
  and loop stability are no longer in play.
- *Time:* 2–3 weeks.
- *(Status: `vessel_centerline.py` / `vessel_tube.py` / `vessel_to_slicer.py` are
  the centerline-extraction / Slicer-export prototypes for this.)*

### GROUP 3 — Gated on the force sensor

**D8. Admittance contact control.**
- *Why:* Ultrasound image quality depends on contact pressure; too little = no
  image, too much = deformation/pain (and deformed vessel).
- *How:* Read normal force, command small axial (push-in / back-off) corrections to
  hold a target force. Write and unit-test the controller now against the D4 stub;
  go live when the sensor arrives.
- *Depends on:* force sensor, D4.
- *Pitfalls:* the compliant arm makes force control sloppy (P11); target force too
  high deforms the (soft) phantom vessel and biases the radius (P7); integrate force
  and visual corrections without them fighting (P12 — decouple axes: force on z,
  vision on lateral/yaw).
- *Risk:* Medium–high.
- *Time:* 1–2 weeks after sensor arrives.

### GROUP 4 — Integration & validation

**D9. Full autonomous loop.**
- *Why:* The demo.
- *How:* Compose D6 (segment) + D7 (servo) + D8 (force) + D3/D5 (pose + live volume)
  into the loop from §3.3. Run end-to-end on the phantom vessel.
- *Depends on:* D6, D7, D8, D5.
- *Pitfalls:* everything that worked in isolation interacting badly; rate collapse
  when all components run together (P8); recovery when the vessel is lost mid-scan
  (P10).
- *Risk:* High — integration always surfaces new failures.
- *Time:* 2–3 weeks.

**D10. Repeatability + Slicer validation (Phase 4/5).**
- *Why:* A single lucky run isn't a result. The paper needs error bars.
- *How:* ≥3 runs on the same vessel; measure reconstructed radius/length vs. ground
  truth in Slicer; report mean ± std and orientation error. Compare to Jiang's
  ~2 mm / 3.3°.
- *Depends on:* D9.
- *Time:* 1 week.

---

## 8. Pitfalls & setbacks (cross-cutting catalog)

| # | Pitfall | Consequence | Mitigation |
| --- | --- | --- | --- |
| P1 | 2 Hz loop from 0.5 s latency | Smooth servoing impossible | Step-and-correct; cap speed to ~1–3 mm/s so the vessel can't leave the plane per cycle |
| P2 | Latency creep as components stack | Loop rate collapses under load | Budget per-component time; keep segmentation inside the frame budget |
| P3 | Marker not rigid | Pose drifts, reconstruction smears | Glue/mount flat and solid; re-verify after any knock |
| P4 | `MARKER_MM` wrong | Translation scale error | Measure the printed black square; don't trust nominal size |
| P5 | Marker occlusion / motion blur | Pose dropouts during motion | Mount where camera sees it through full motion; slow moves reduce blur |
| P6 | Serial single-thread | Corrupted reads if threaded | Sequence all arm I/O in one thread (learned in Phase 1/2) |
| P7 | Soft vessel deforms under pressure | Biased radius, collapsed lumen | Tune force low; note deformation in results |
| P8 | Segmentation too slow | Blows the 0.5 s budget | Classical first; optimize/quantize U-Net only if needed |
| P9 | Servo oscillation on laggy loop | Probe hunts, never settles | Conservative gains; rate-limit corrections |
| P10 | Vessel lost (curve/branch/dropout) | Scan fails mid-run | Detect "no vessel," halt + search small neighborhood before giving up |
| P11 | Compliant arm = sloppy force control | Force overshoot/contact loss | Slow admittance gains; accept a tolerance band, not a setpoint |
| P12 | Force vs vision fight | Unstable combined motion | Decouple axes (force→z, vision→lateral/yaw) |
| P13 | Clarius SDK fragility | Capture breaks | Matched binaries, NumPy <2, PySide6 6.5.3, local files, no worker threads |
| P14 | iCloud dehydrates binaries | Mysterious load failures | Project stays local, never in iCloud |
| P15 | Calibration error stacks (marker→image + arm) | Reconstruction off | Validate each transform independently before trusting the chain |
| P16 | Scope creep toward "real anatomy" | Demo never ships | Lock scope to phantom vessel; anatomy is future work |

---

## 9. Risk register (top risks)

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Segmentation doesn't generalize to realistic phantom | High | High | Classical baseline first; U-Net fallback with labeled frames; still the #1 stall point |
| Pose error (2.86 mm) too large vs. 2–6 mm vessels | High | High | Target the largest vessel; accept qualitative result if needed; the pose experiment decides |
| Camera not actually better than the arm (the core bet) | Medium | High | Pose experiment (static + known-move) before committing; if false, rethink pose source |
| Marker tracking accuracy / rigidity insufficient | Medium | High | Validate against cobot-pose recon; mount rigid; measure `MARKER_MM` |
| Hardware (phantom/force sensor) arrives late | Medium | Medium | Front-load Group 1; it's all unblocked now |
| Vessel selection / path planning off (12 vessels) | Medium | Medium | Survey-first; explicit selection rule; sanity-check the planned path |
| FSR too coarse / compresses vessel | Medium | Medium | Contact-band not setpoint; compression test sets the safe-force ceiling |
| Integration surfaces emergent failures | Medium | Medium | Integrate incrementally, not big-bang |
| 2 Hz loop / closed-loop instability | — | — | Retired — open-loop acquisition removes this entirely |

---

## 10. Timeframes (rough, calendar)

Assumes ~10–15 hr/week and that hardware arrives in the next few weeks. These are
honest ranges with debugging included, not best-case.

| Phase | Tasks | Estimate |
| --- | --- | --- |
| Now (unblocked) | D1, D2, D3, D4, D5 | 3–5 weeks |
| Phantom arrives | D6, D7 | 3–7 weeks |
| Force sensor arrives | D8 | 1–2 weeks (overlaps) |
| Integration | D9, D10 | 3–4 weeks |
| **Total to demo** | | **~10–16 weeks of focused work** |
| Paper draft | runs alongside, each phase = a figure | start writing methods after D3 |

The hardware waits overlap the unblocked software, so calendar time is shorter than
the sum if you front-load Group 1 now.

---

## 11. Summary — critical path, steps, flowchart

**The one-line version**

> Make a cheap, imprecise arm autonomously follow a vessel and reconstruct it
> accurately, by letting the ultrasound image steer it and an external camera
> measure it.

**Critical path (do in this order)**

```
NOW (unblocked, front-load all of it):
  D1  Measure real loop latency  ──┐
  D2  Multi-sweep max-merge       │  (independent, parallelizable)
  D3  RealSense marker hand-eye   │
  D4  Control-loop skeleton (stubs)┘
  D5  Validate live recon on hardware

GATED — phantom arrives:
  D6  Tube segmentation  ──►  D7  Visual servoing (the follow)
              │                        ▲
              ▼                        │  HIGHEST-RISK STRETCH
GATED — force sensor arrives:          │
  D8  Admittance contact control ──────┘

INTEGRATE:
  D9  Full autonomous loop  ──►  D10  Repeatability + Slicer validation
                                        │
                                        ▼
                                    DEMO + PAPER
```

**The autonomous loop (the heart of it)**

```
   step forward  →  grab frame  →  segment vessel  →  read force
        ▲                                                 │
        │                                                 ▼
        └──  command correction  ←  compute: re-center + hold force + orient normal
                     │
                     └─►  (frame + RealSense pose)  →  live 3D volume
```

**What to do this week**

- **D1** — measure the latency for real (1–2 days, unblocks honest planning).
- **D4 skeleton** — stand up the loop with stubs against the live arm.
- **D2** — multi-sweep max-merge (clears the multi-angle blocker, pure software).
- **D3** — once the marker is mounted and `track_probe.py` runs, do the
  marker→image calibration.

Everything in Group 1 is unblocked today. The phantom and force sensor gate the
high-risk perception and contact work — so the smart move is to finish all the
software groundwork now, so that when the boxes arrive you're dropping real
components into a loop that already runs.
