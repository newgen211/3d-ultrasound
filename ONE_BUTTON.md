# One-button sweep

`make sweep` acquires one section end to end; `make post SEC=N` reruns the post chain.

## What each file does

| File | Runs on | Does |
|---|---|---|
| `fly.py` | Mac | gates, anchor read, solve, deploy launch file, start tracker, one ssh to the Pi, fetch, verify, then `make post` |
| `Makefile` | Mac | `sweep` and the file-keyed post chain: trim, smooth, merge, reconstruct, vessels |
| `run_sweep.py` | Mac | `preflight(launch, anchor, solver_report)` used by fly.py; the manual CLI still works |
| `src/capture/cast_capture.py` | both | capture core: SDK libs, callbacks, frame store, on-disk format |
| `src/capture/cast_headless.py` | Pi | capture without a window into `clarius_sessions/section_N` |
| `src/capture/pysidecaster.py` | Mac | the GUI, now a thin layer over cast_capture |
| `ultrasound-cobot/flight.sh` | Pi | probe WiFi on wlan0, headless capture, `execute_sweep --no-prompt`, stop, pair exec log |
| `ultrasound-cobot/probe.env` | Pi | `PROBE_CON`, `PROBE_IP`, `PROBE_PORT` (gitignored; the WiFi password lives only in nmcli) |

Interpreters: `realsense` env for anchor, solve, tracker, trim, smooth, merge; `clarius` env for
reconstruct and the vessel scripts. On the Pi, `python3.10` (built from source, `make altinstall`, numpy<2) runs the capture
and system `python3` runs the arm.

The Pi runs copies of `cast_capture.py` and `cast_headless.py` from the ultrasound-cobot repo
so they deploy by git: `make pi-sync` refreshes them from `src/capture/`, commit, push, then
`git fetch && git checkout one-button` on the Pi. The Cast binaries (`libcast.so`,
`pyclariuscast.so`, 12.2.0 aarch64 python310) are not in git; they sit in the Pi repo dir, with a
focal-built `libstdc++.so.6` and `libgcc_s.so.1` in `sdk_lib/` because that Cast build wants GCC 12's
libstdc++ (GLIBCXX_3.4.30) and Ubuntu 20.04 ships GCC 9. `flight.sh` puts `sdk_lib/` on
`LD_LIBRARY_PATH`; the system libraries are untouched. `python3.10` is a source build (`make altinstall`,
no deadsnakes arm64 packages exist for focal).

## Run folder

```
runs/<MMDD_HHMM>_sec<N>/
  manifest.json     sec, stamp, launch, teach + hash, pi_clock_offset_s, per-stage exit codes,
                    cam_log, frames_pi, frames_mac, state (started -> acquired -> done)
  fly.log           every stage's output
  solve.log         shift_joint_path's report (the '#' lines; preflight audits these)
  anchor.json       the vision anchor the flight was solved against
  shifted_<stamp>.jsonl   the launch file that flew
  track_probe.out   tracker output
```

## Smoke steps, in order

1. **Headless capture, 10 s, probe awake, no arm.** On the Pi:
   `cd ~/Documents/ultrasound-cobot && LD_LIBRARY_PATH=$PWD/sdk_lib python3.10 cast_headless.py --section 900 --ip 192.168.1.1 --port 5828 --seconds 10`
   Pass: `raw_*.bin` and `raw_*.json` under `clarius_sessions/section_900`, and `imu_sample_count > 0`
   in the first json. Then `rm -r clarius_sessions/section_900`.
2. **flight.sh in the air** with a launch file that has already flown (for example
   `pose_logs/shifted_clamp15_0813.jsonl`), nothing under the probe:
   `./flight.sh pose_logs/shifted_clamp15_0813.jsonl 901`
   Pass: it ends with `section_901 frames=... exec=...` and `clarius_sessions/section_901/exec.jsonl`
   exists. Then delete section_901.
3. **`make sweep` on the phantom.** Pass: `runs/<stamp>_sec<N>/manifest.json` says `done` and
   `data/clarius_sessions/section_<N>/recovered_tubes_3d.png` exists.

## Fallback

If `cast_headless` cannot initialise on the Pi (SDK init or connect fails), capture stays on the
Mac over the shared probe LAN: run `pysidecaster.py` there as before, and change only `flight.sh`,
dropping its capture block so it does network, arm, and the exec-log pairing. `fly.py` then fetches
nothing and the post chain reads the Mac's own section folder.
