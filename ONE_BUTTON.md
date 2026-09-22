# One-button sweep

`make sweep` acquires one section end to end; `make post SEC=N` reruns the post chain.

## What each file does

| File | Runs on | Does |
|---|---|---|
| `fly.py` | Mac | anchor read, solve, start tracker, gates, deploy, join the probe network, capture, one ssh to the Pi for the flight, rejoin the lab, fetch the exec log, then `make post` |
| `Makefile` | Mac | `sweep` and the file-keyed post chain: trim, smooth, merge, reconstruct, vessels |
| `run_sweep.py` | Mac | `preflight(launch, anchor, solver_report)` used by fly.py; the manual CLI still works |
| `src/capture/cast_capture.py` | both | capture core: SDK libs, callbacks, frame store, on-disk format |
| `src/capture/cast_headless.py` | Mac | capture without a window, straight into `data/clarius_sessions/section_N`. Runs a Qt event loop: the SDK delivers frames through it, and `--interval-ms` caps the rate (50 ms means at most 20/s) |
| `src/capture/pysidecaster.py` | Mac | the GUI, now a thin layer over cast_capture |
| `ultrasound-cobot/flight.sh` | Pi | `net`: probe WiFi on wlan0, prints its address. `fly`: `execute_sweep --no-prompt`, prints the exec log |
| `ultrasound-cobot/probe.env` | Pi | `PROBE_CON`, `PROBE_IP`, `PROBE_PORT` (gitignored; the WiFi password lives only in nmcli) |

Interpreters: `realsense` env for anchor, solve, tracker, trim, smooth, merge; `clarius` env for
capture, reconstruct and the vessel scripts. On the Pi, system `python3` runs the arm.

## Where capture runs, and why

Capture runs on the **Mac**. The Cast SDK aborts with `Failed to link shader` the moment imaging
starts on the Pi: its scan-conversion renderer needs an OpenGL context that Ubuntu 20.04 aarch64
cannot give it. That was tested against every Qt platform (offscreen, minimal, vnc, xcb), with
`LIBGL_ALWAYS_SOFTWARE` and `QT_OPENGL=software`, and under `xvfb-run` with a real GLX context.
The vendor's own `pycaster.py` connects fine there, so the SDK loads and talks to the probe; only
rendering fails.

So the Mac joins the probe's network for the flight, captures locally, and reaches the Pi at its
address on that same network (`flight.sh net` prints it). Afterwards it rejoins the lab network and
fetches only the exec log. One button is unchanged; only where the frames land differs.

Joining needs the password handed to `networksetup`, even for a network macOS already knows, or it
fails with error -3900. `fly.py` reads it from the System keychain and keeps it out of the run log.
If it is not there, join the probe network once by hand and it will be. Coming back is the other
way round: the lab network is enterprise auth, which `networksetup` cannot drive, so `fly.py`
cycles the Wi-Fi radio and waits for the Pi to answer over ZeroTier again.

`cast_capture.py` and `cast_headless.py` are still deployed to the Pi and `make pi-sync` still
refreshes them, against the day the Pi can render: restoring Pi-side capture then means putting the
capture block back into `flight.sh` and pointing `fly.py` at it.

Everything deploys by git: commit, push, then `git fetch && git reset --hard origin/one-button` on
the Pi. Nothing is scp'd. The Pi also carries the Cast binaries and a `sdk_lib/` with a focal-built
`libstdc++`, both gitignored and both unused while capture lives on the Mac.

## Run folder

```
runs/<MMDD_HHMM>_sec<N>/
  manifest.json     sec, stamp, launch, teach + hash, pi_clock_offset_s, per-stage exit codes,
                    cam_log, frames_pi, frames_mac, state (started -> acquired -> done)
  fly.log           every stage's output
  solve.log         shift_joint_path's report (the '#' lines; preflight audits these)
  anchor.json       the vision anchor the flight was solved against
  shifted_<stamp>.jsonl   the launch file that flew
  track_probe.out   tracker output, plus track_probe.2.out and .3.out if the
                    camera needed retrying (the manifest records
                    tracker_warmup_s, tracker_attempts, and any leftover
                    tracker it had to kill first)
```

## Smoke steps, in order

1. **Headless capture, 10 s, probe awake, no arm.** Join the probe network on the Mac, then from
   the repo root:
   `/opt/anaconda3/envs/clarius/bin/python src/capture/cast_headless.py --section 900 --ip 192.168.1.1 --port 5828 --root data/clarius_sessions --seconds 10`
   Pass: `raw_*.bin` and `raw_*.json` under `data/clarius_sessions/section_900`, and
   `imu_sample_count > 0` in the first json. Then `rm -r data/clarius_sessions/section_900` and
   rejoin the lab network.
2. **flight.sh in the air** with a launch file that has already flown (for example
   `pose_logs/shifted_clamp15_0813.jsonl`), nothing under the probe. On the Pi:
   `./flight.sh net` then `./flight.sh fly pose_logs/shifted_clamp15_0813.jsonl`
   Pass: `net` prints `WLAN0=<ip>` and `fly` ends with `EXEC=pose_logs/exec_<stamp>.jsonl`.
3. **`make sweep` on the phantom.** Pass: `runs/<stamp>_sec<N>/manifest.json` says `done` and
   `data/clarius_sessions/section_<N>/recovered_tubes_3d.png` exists.

## If capture fails on the Mac

Fall back to the GUI: join the probe network, run `src/capture/pysidecaster.py` in the `clarius`
env, press Start Scan when the arm reaches the hover, and Stop Scan at the end. Then run the Pi
half by hand (`flight.sh net`, `flight.sh fly <launch>`), copy the exec pair into the section dir as
`exec.jsonl` and `exec_meta.json`, and run `make post SEC=<N>`.
