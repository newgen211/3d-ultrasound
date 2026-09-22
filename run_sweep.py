#!/usr/bin/env python3
"""
run_sweep.py — the capture orchestrator. Refuses to fly dataless.

Exists because of 2026-07-30: a tilted autonomous sweep flew perfectly and
recorded NOTHING (pysidecaster and track_probe weren't running). This wrapper
makes that impossible: it verifies every leg of the recording chain before,
DURING, and after the flight, and aborts loudly the moment a leg is dead.

    python3 run_sweep.py shifted_tiltA.jsonl
    python3 run_sweep.py shifted_flat.jsonl --delay 45 --no-scp

What it checks (Mac side, no trust required):
  PRE   anchor freshness (<30 min), launch-file header (shift/FK/lift echo),
        camera log ALIVE (file growing right now), disk space, baseline
        snapshot of existing sections
  LIVE  a NEW section directory appears and its frame count GROWS during the
        capture window — the anti-dataless guarantee. No frames -> ABORT
        instructions while the arm is still hovering.
  POST  frame total, camera-log growth over the window, next-command cheat
        sheet for the post chain (trim/fix/smooth/merge/reconstruct)

It does NOT drive the Pi (execute_sweep stays manual in tmux — deliberate:
a human hand near the e-stop is part of the protocol). It scp's the launch
file and tells you exactly what to type, then supervises the recording side.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from us3d.paths import CALIB, POSE_LOGS, SESSIONS

PI = "er@192.168.196.134"
PI_DIR = "~/Documents/ultrasound-cobot/pose_logs/"
CAM_LOG = POSE_LOGS / "probe_pose_log.jsonl"
ANCHOR = CALIB / "vision_anchor.json"
ANCHOR_MAX_AGE_MIN = 30

def fail(msg):
    print(f"\n\u274c  {msg}")
    sys.exit(1)

def ok(msg):
    print(f"\u2705  {msg}")

def warn(msg):
    print(f"\u26a0\ufe0f   {msg}")

def confirm(msg):
    input(f"\u25b6  {msg}  [Enter to confirm] ")

def file_growing(p, wait=2.0):
    if not p.exists():
        return False
    a = p.stat().st_size
    time.sleep(wait)
    return p.stat().st_size > a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("launch", help="shifted_*.jsonl from shift_joint_path")
    ap.add_argument("--delay", type=int, default=45)
    ap.add_argument("--rate", type=int, default=2)
    ap.add_argument("--speed", type=int, default=25)
    ap.add_argument("--no-scp", action="store_true", help="skip pushing the launch file")
    ap.add_argument("--grace", type=int, default=20,
                    help="extra s after delay before declaring the capture dead")
    args = ap.parse_args()

    print("\n=== PRE-FLIGHT ===")
    lp = Path(args.launch)
    if not lp.exists():
        fail(f"launch file not found: {lp}")

    # 1. launch-file header audit (shift_joint_path writes '#' header lines)
    header = [l for l in lp.read_text().splitlines() if l.startswith("#")]
    hdr = " ".join(header)
    m = re.search(r"shift dx=([+-][\d.]+) dy=([+-][\d.]+)", hdr)
    if m:
        dx, dy = float(m.group(1)), float(m.group(2))
        mag = (dx * dx + dy * dy) ** 0.5
        ok(f"launch header: shift dx={dx:+.1f} dy={dy:+.1f} (|{mag:.1f}| mm)")
        if mag > 40:
            fail("shift magnitude > 40 mm — outside the validated envelope. "
                 "Re-place the container or re-run the reader.")
    else:
        warn("no shift line in launch header — was this file made by shift_joint_path?")
    m = re.search(r"FK check worst: ([\d.]+) mm", hdr)
    if m:
        ok(f"launch header: FK worst {float(m.group(1)):.2f} mm")
    if "start-lift" in hdr:
        ok("launch header: start-lift present "
           + re.search(r"start-lift: ([^#]+?) at touchdown", hdr).group(0)[:40])
    if "anchor is" in hdr and "h old" in hdr:
        fail("launch file was solved against a STALE anchor — re-run the "
             "reader and re-solve before flying.")

    # 2. anchor freshness (independent of the launch file)
    if ANCHOR.exists():
        a = json.loads(ANCHOR.read_text())
        try:
            age_min = (time.time() - time.mktime(
                time.strptime(a["written"], "%Y-%m-%d %H:%M:%S"))) / 60
            if age_min > ANCHOR_MAX_AGE_MIN:
                warn(f"anchor is {age_min:.0f} min old — fine only if the "
                     f"container has not moved. Otherwise re-run the reader.")
            else:
                ok(f"anchor fresh ({age_min:.0f} min)")
        except Exception:
            warn("could not parse anchor timestamp")
    else:
        warn("no vision_anchor.json in calib/ - run guided_sweep_reader.py")

    # 3. camera log ALIVE (track_probe must be running RIGHT NOW)
    print("checking camera log is growing (2 s) ...")
    if not file_growing(CAM_LOG):
        fail(f"{CAM_LOG} is not growing — track_probe is NOT running. "
             f"Start it (sudo, [custom] intrinsics line) and rerun.")
    ok("track_probe alive — camera log growing")

    # 4. disk + baseline sections snapshot
    st = os.statvfs(".")
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < 5:
        fail(f"only {free_gb:.1f} GB free — captures are large.")
    ok(f"disk: {free_gb:.0f} GB free")
    before = {p.name for p in SESSIONS.iterdir() if p.is_dir()}
    ok(f"{len(before)} existing sections snapshotted")

    # 5. physical checklist
    print("\n=== PHYSICAL CHECKLIST ===")
    confirm("Container taped and untouched since the anchor was read?")
    confirm("Gel applied generously along the FULL sweep path?")
    confirm("Workspace clear, e-stop within reach?")

    # 6. push launch file
    if not args.no_scp:
        print(f"\npushing {lp.name} to the Pi ...")
        r = subprocess.run(["scp", str(lp), f"{PI}:{PI_DIR}"], )
        if r.returncode != 0:
            fail("scp failed — network/VPN/password issue.")
        ok("launch file on the Pi")

    # 7. hand off to the human for the Pi side
    print("\n=== PI SIDE (manual, in tmux) ===")
    print(f"  ssh {PI}")
    print(f"  cd ~/Documents/ultrasound-cobot && tmux new -s sweep")
    print(f"  python3 execute_sweep.py {lp.name} --speed {args.speed} "
          f"--rate {args.rate} --delay {args.delay}")
    confirm("execute_sweep launched, arm AT HOVER, delay countdown running?")
    print(f"\nNOW: switch Mac WiFi to the Clarius network and START pysidecaster.")
    print(f"You have ~{args.delay} s. I am watching for frames ...")

    # 8. LIVE: the anti-dataless watch
    t0 = time.time()
    new_sec, n_frames = None, 0
    deadline = args.delay + args.grace
    while True:
        now = {p.name for p in SESSIONS.iterdir() if p.is_dir()}
        fresh = now - before
        if fresh:
            new_sec = SESSIONS / sorted(fresh)[-1]
            n = len(list(new_sec.glob("raw_*.json")))
            if n > n_frames:
                n_frames = n
                print(f"\r\u2705  CAPTURING  {new_sec.name}: {n_frames} frames ",
                      end="", flush=True)
        el = time.time() - t0
        if new_sec is None and el > deadline:
            print()
            fail(f"NO new section after {deadline:.0f} s — pysidecaster is NOT "
                 f"recording. ABORT: Ctrl-C execute_sweep on the Pi / e-stop. "
                 f"The arm must not sweep dataless.")
        if new_sec is not None and el > deadline and n_frames > 50:
            # capture confirmed underway; keep monitoring until frames stop
            last = n_frames
            time.sleep(8)
            n = len(list(new_sec.glob("raw_*.json")))
            if n == last and el > deadline + 60:
                break
            n_frames = max(n_frames, n)
        time.sleep(1.5)

    # 9. POST
    print(f"\n\n=== POST ===")
    ok(f"capture complete: {new_sec.name}, {n_frames} frames")
    if not file_growing(CAM_LOG):
        warn("camera log stopped growing — confirm track_probe survived the run")
    else:
        ok("camera log still alive")
    print("\nSwitch WiFi back to the lab network, let the Pi finish homing, then:")
    sec = new_sec.name
    print(f"""
  mv {CAM_LOG} data/pose_logs/{sec}_cam.jsonl
  scp "{PI}:{PI_DIR}exec_*" data/pose_logs/        # newest pair
  python src/pose/trim_section.py {sec} data/pose_logs/exec_<stamp>.jsonl \\
      data/pose_logs/exec_<stamp>_meta.json data/pose_logs/{sec}_cam.jsonl
  # sane? add --apply, then fix -> smooth -> merge -> reconstruct
""")
    print("\U0001f3c1  flight supervised: no dataless sweeps on my watch.")

if __name__ == "__main__":
    main()
