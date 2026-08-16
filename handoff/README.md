# handoff/ — curated artifacts for teammates

A stable pull-point. Each file here is a **copy** of a work product that lives
(and is regenerated) elsewhere in the repo, frozen so collaborators have one
place to grab the current best. When the upstream changes, re-copy it here.

| File | What it is | Frame / units | Provenance |
|---|---|---|---|
| `sim_handoff.json` | Sim bundle: workspace cloud + teach paths + notes | `cobot_base_mm` | 2026-08-11 (touch truth refreshed, residual 0.12 mm) |
| `workspace_cloud_clean.ply` | Cleaned workspace point cloud (4605 pts, xyz + rgb) | cobot base (mm) | 2026-08-11 |
| `clamp15_taught_path.json` | The canonical clamped taught sweep (316 samples: `xyz_mm`, `euler_deg`, `joint_angles_deg`) | `cobot_base (mm, xyz-euler deg)` | 2026-08-15 |
| `exemplars.json` | SAM open/crescent exemplar boxes, per section | image-normalized `xywh` | 2026-08-13 |
| `audit_truth.json` | Human vessel-circle ground truth (160 frames) | image pixels `[cx, cy, r]` | 2026-08-13 |

## `clamp15_taught_path.json` vs the "up2" launch files — do not confuse them

- **`clamp15_taught_path.json`** is the *taught sweep itself* — the drag-recorded
  skill (max 1.5 mm indent below the touched phantom surface) that flew sections
  92 / 94 / 106 / 108 / 109 / 111 at 0% collapse. This is the artifact to use for
  replay and analysis.
- **`shifted_*_up2.jsonl`** (e.g. `shifted_flat_up2.jsonl`) are *derived launch
  files* — a flat path lifted +2 mm for one specific flight, **not** a taught
  skill. The populated launch files live in `ultrasound-cobot/`; the copy at the
  outer repo root is a stale/empty stub. If you need a path to reason about, use
  `clamp15_taught_path.json`, never the `up2` stubs.
