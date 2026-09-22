#!/usr/bin/env python
"""fly.py: one autonomous acquisition.

gates -> anchor -> solve -> deploy -> tracker -> Pi flight (capture + arm)
-> fetch -> verify -> hand off to `make post`.

Everything for one run lands in runs/<stamp>_sec<N>/: fly.log, solve.log,
anchor.json, the launch file, track_probe.out and manifest.json. The tracker
is always stopped (SIGINT, so it closes its log) and the cam log always
snapshotted, even on failure.

Usage: python fly.py [SEC]          TEACH=<file> overrides the taught skill.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                      # every path below is repo-relative, like the Makefile
sys.path.insert(0, str(ROOT))
from run_sweep import preflight     # noqa: E402  (lp, before) or SystemExit on a failed gate

# ---- config (from the 22 Sep discovery pass) ----
COBOT   = Path("ultrasound-cobot")
TEACH   = Path(os.environ.get("TEACH", "sweep_teach_clamp15.jsonl"))   # up2 stays the E1 paper file
if not TEACH.exists():
    TEACH = COBOT / "pose_logs" / TEACH.name
CALIB   = Path("calib/guided_sweep_calib.json")
ANCHOR  = Path("calib/vision_anchor.json")
LOGS    = Path("data/pose_logs")
SESS    = Path("data/clarius_sessions")
PI      = "er@192.168.196.134"
PI_DIR  = "Documents/ultrasound-cobot"
SPEED, RATE, DELAY, LIFT = 25, 2, 8, 3.5
PY      = "/opt/anaconda3/envs/realsense/bin/python"   # reader, tracker, solver all need this env

stamp = datetime.now().strftime("%m%d_%H%M")
sec = int(sys.argv[1]) if len(sys.argv) > 1 else max(
    [int(d.name[8:]) for d in SESS.glob("section_*") if d.name[8:].isdigit()], default=0) + 1
run = Path("runs") / f"{stamp}_sec{sec}"
run.mkdir(parents=True)
log = open(run / "fly.log", "a")
launch = COBOT / "pose_logs" / f"shifted_{stamp}.jsonl"
manifest = dict(sec=sec, stamp=stamp, launch=str(launch), teach=str(TEACH),
                teach_sha256=hashlib.sha256(TEACH.read_bytes()).hexdigest()[:16] if TEACH.exists() else None,
                stages={})


def note(**kw):
    manifest.update(kw)
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))


def sh(cmd, stage, sudo=False, **kw):
    cmd = (["sudo"] if sudo else []) + list(map(str, cmd))
    print(f"[{stage}] {' '.join(cmd)}", flush=True)
    log.write(f"\n== {stage}\n")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    log.write(p.stdout)
    log.flush()
    print(p.stdout, end="")
    manifest["stages"][stage] = p.returncode
    note()
    if p.returncode:
        raise SystemExit(f"ABORT at {stage} (exit {p.returncode})")
    return p.stdout


def gate(ok, msg):
    if not ok:
        raise SystemExit(f"ABORT: {msg}")


def pi_clock_offset():
    """Pi minus Mac wall clock, seconds. trim_section correlates it out; logged for the record."""
    t0 = time.time()
    p = subprocess.run(["ssh", PI, "python3 -c 'import time;print(time.time())'"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t1 = time.time()
    try:
        return round(float(p.stdout) - (t0 + t1) / 2, 3)
    except ValueError:
        return None


def anchor():
    before = ANCHOR.stat().st_mtime if ANCHOR.exists() else 0
    out = sh([PY, "src/calibration/guided_sweep_reader.py", "--calib", CALIB], "anchor", sudo=True)
    gate("color stream: 640x480" in out and "NOT the calibrated" not in out,
         "camera not at 640x480, replug the RealSense")
    gate(ANCHOR.exists() and ANCHOR.stat().st_mtime > before, "anchor not refreshed (reader timed out?)")
    shutil.copy(ANCHOR, run / "anchor.json")


def solve():
    gate(TEACH.exists(), f"teach file not found: {TEACH}")
    with open(launch, "w") as f:
        p = subprocess.run([PY, "shift_joint_path.py", str(TEACH.resolve()),
                            "--start-lift", str(LIFT), "--anchor", str(ANCHOR.resolve())],
                           cwd=COBOT, stdout=f, stderr=subprocess.PIPE, text=True)
    (run / "solve.log").write_text(p.stderr)      # the solver's '#' report lives on stderr
    log.write("\n== solve\n" + p.stderr)
    print(p.stderr, end="")
    manifest["stages"]["solve"] = p.returncode
    if p.returncode or launch.stat().st_size == 0:
        launch.unlink(missing_ok=True)
        note()
        raise SystemExit("ABORT: solver gate refused")
    shutil.copy(launch, run / launch.name)
    note()
    preflight(launch, ANCHOR, p.stderr)


def deploy():
    sh(["scp", launch, f"{PI}:{PI_DIR}/pose_logs/"], "deploy")
    sh(["ssh", PI, f"test -s {PI_DIR}/pose_logs/{launch.name}"], "deploy-verify")


def start_tracker():
    p = subprocess.Popen(["sudo", PY, "src/pose/track_probe.py"],
                         stdout=open(run / "track_probe.out", "w"), stderr=subprocess.STDOUT)
    time.sleep(6)
    cam = LOGS / "probe_pose_log.jsonl"
    gate(cam.exists() and time.time() - cam.stat().st_mtime < 3, "cam log not growing")
    return p


def stop_tracker(p):
    subprocess.run(["sudo", "kill", "-INT", str(p.pid)])   # SIGINT: track_probe closes its log
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "kill", str(p.pid)])
    cam, snap = LOGS / "probe_pose_log.jsonl", LOGS / f"sec{sec}_cam.jsonl"
    if cam.exists():
        subprocess.run(["sudo", "mv", cam, snap])
        subprocess.run(["sudo", "chown", os.getlogin(), snap])
        note(cam_log=str(snap))


def flight():
    sh(["ssh", "-t", PI, f"cd {PI_DIR} && ./flight.sh pose_logs/{launch.name} {sec} {SPEED} {RATE} {DELAY}"],
       "flight")


def fetch():
    dst = SESS / f"section_{sec}"
    sh(["rsync", "-a", f"{PI}:{PI_DIR}/clarius_sessions/section_{sec}/", f"{dst}/"], "fetch")
    a = int(sh(["ssh", PI, f"ls {PI_DIR}/clarius_sessions/section_{sec} | grep -c '^raw_.*bin'"], "count-pi"))
    b = len(list(dst.glob("raw_*.bin")))
    note(frames_pi=a, frames_mac=b)
    gate(a == b > 0, f"frame count mismatch pi={a} mac={b}")
    gate((dst / "exec.jsonl").exists() and (dst / "exec_meta.json").exists(), "exec pair missing in section")


note(state="started")
sh(["-v"], "sudo", sudo=True)                # one password prompt up front
note(pi_clock_offset_s=pi_clock_offset())
anchor()
solve()
deploy()
tracker = start_tracker()
try:
    flight()
finally:
    stop_tracker(tracker)
fetch()
note(state="acquired")
sh(["make", "post", f"SEC={sec}"], "post")
note(state="done")
print(f"\nsection_{sec} complete. run folder: {run}")
