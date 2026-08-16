# data/pose_logs/ — log species

Most of `data/` is gitignored (large / regenerable); this README is
force-tracked so the layout is documented. Four kinds of log live here:

| Species | Written by | Clock | What it is |
|---|---|---|---|
| `probe_pose_log.jsonl` | `src/pose/track_probe.py` (Mac) | Mac | Live camera ArUco tracking of the probe marker. Appends forever across sessions — snapshot it after each capture. |
| `pi_pose_log.jsonl` | `ultrasound-cobot/pose_logger.py` (Pi) | Pi | Arm-side pose + joint-angle log recorded on the Raspberry Pi during a drag or flight. |
| `exec_<stamp>.jsonl` + `exec_<stamp>_meta.json` | arm executor (Pi) | Pi | Per-flight playback telemetry: the poses actually commanded, plus a manifest (`events.playback_start/end`, source log, poses played). One pair per flight. |
| `section_<N>_cam.jsonl` | renamed post-flight from a `probe_pose_log` slice | Mac | The camera log for one capture, sliced/renamed to the section it belongs to. Feeds `src/pose/merge_poses_cam.py`. |

**Clocks differ** (Pi vs Mac, unsynced). Alignment is recovered from the motion
itself by speed-profile cross-correlation (`src/pose/trim_section.py`,
`src/calibration/solve_bridge.py`) — never by assuming the clocks are in sync.
