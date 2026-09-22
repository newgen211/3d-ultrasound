#!/usr/bin/env python
"""fly.py: one autonomous acquisition.

gates -> anchor -> solve -> deploy -> tracker -> probe WiFi -> capture (Mac)
-> Pi flight -> stop capture -> lab WiFi -> fetch exec log -> verify -> make post

Capture runs here, not on the Pi: the Cast SDK needs a GL context the Pi cannot
provide (ONE_BUTTON.md). So the Mac joins the probe's network for the flight and
talks to the Pi at its address on that same network, then rejoins the lab.

Everything for one run lands in runs/<stamp>_sec<N>/: fly.log, solve.log,
anchor.json, capture.log, the launch file and manifest.json. The tracker is
always stopped (SIGINT, so it closes its log), the capture is always stopped,
and the Mac always goes back to the lab network, even on failure.

Usage: python fly.py [SEC]          TEACH=<file> overrides the taught skill.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                      # the SDK's TLS pair lives here, as for the GUI
sys.path.insert(0, str(ROOT))
from run_sweep import preflight     # noqa: E402  (lp, before) or SystemExit on a failed gate

# ---- config ----
COBOT      = Path("ultrasound-cobot")
TEACH      = Path(os.environ.get("TEACH", "sweep_teach_clamp15.jsonl"))
if not TEACH.exists():
    TEACH = COBOT / "pose_logs" / TEACH.name
CALIB      = Path("calib/guided_sweep_calib.json")
ANCHOR     = Path("calib/vision_anchor.json")
LOGS       = Path("data/pose_logs")
SESS       = Path("data/clarius_sessions")
PI_USER    = "er"
PI_ZT      = f"{PI_USER}@192.168.196.134"      # over the lab network / ZeroTier
PI_DIR     = "Documents/ultrasound-cobot"
WIFI_IF    = os.environ.get("WIFI_IF", "en0")
PROBE_SSID = "DIRECT-PALHD3012406A0356"
PROBE_IP, PROBE_PORT = "192.168.1.1", 5828
SPEED, RATE, DELAY, LIFT = 25, 2, 8, 3.5
PY_RS = "/opt/anaconda3/envs/realsense/bin/python"   # reader, tracker, solver
PY_CL = "/opt/anaconda3/envs/clarius/bin/python"     # capture (PySide6 + Cast libs)

stamp = datetime.now().strftime("%m%d_%H%M")
sec = int(sys.argv[1]) if len(sys.argv) > 1 else max(
    [int(d.name[8:]) for d in SESS.glob("section_*") if d.name[8:].isdigit()], default=0) + 1
run = Path("runs") / f"{stamp}_sec{sec}"
run.mkdir(parents=True)
log = open(run / "fly.log", "a")
launch = COBOT / "pose_logs" / f"shifted_{stamp}.jsonl"
manifest = dict(sec=sec, stamp=stamp, launch=str(launch), teach=str(TEACH),
                teach_sha256=hashlib.sha256(TEACH.read_bytes()).hexdigest()[:16] if TEACH.exists() else None,
                capture="mac", stages={})


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


def current_ssid():
    out = subprocess.run(["networksetup", "-getairportnetwork", WIFI_IF],
                         stdout=subprocess.PIPE, text=True).stdout
    return out.split(": ", 1)[1].strip() if ": " in out else None


def join(ssid, want_gateway=None, timeout=45):
    """Join a network and wait until it is actually usable."""
    sh(["networksetup", "-setairportnetwork", WIFI_IF, ssid], f"wifi:{ssid}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        if current_ssid() != ssid:
            continue
        if want_gateway is None:
            return True
        if subprocess.run(["ping", "-c1", "-W2", want_gateway],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return True
    raise SystemExit(f"ABORT: joined {ssid} but {want_gateway or 'the network'} is unreachable")


def pi_clock_offset(host):
    t0 = time.time()
    p = subprocess.run(["ssh", host, "python3 -c 'import time;print(time.time())'"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t1 = time.time()
    try:
        return round(float(p.stdout) - (t0 + t1) / 2, 3)
    except ValueError:
        return None


def anchor():
    before = ANCHOR.stat().st_mtime if ANCHOR.exists() else 0
    out = sh([PY_RS, "src/calibration/guided_sweep_reader.py", "--calib", CALIB], "anchor", sudo=True)
    gate("color stream: 640x480" in out and "NOT the calibrated" not in out,
         "camera not at 640x480, replug the RealSense")
    gate(ANCHOR.exists() and ANCHOR.stat().st_mtime > before, "anchor not refreshed (reader timed out?)")
    shutil.copy(ANCHOR, run / "anchor.json")


def solve():
    gate(TEACH.exists(), f"teach file not found: {TEACH}")
    with open(launch, "w") as f:
        p = subprocess.run([PY_RS, "shift_joint_path.py", str(TEACH.resolve()),
                            "--start-lift", str(LIFT), "--anchor", str(ANCHOR.resolve())],
                           cwd=COBOT, stdout=f, stderr=subprocess.PIPE, text=True)
    (run / "solve.log").write_text(p.stderr)
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


def start_tracker():
    p = subprocess.Popen(["sudo", PY_RS, "src/pose/track_probe.py"],
                         stdout=open(run / "track_probe.out", "w"), stderr=subprocess.STDOUT)
    time.sleep(6)
    cam = LOGS / "probe_pose_log.jsonl"
    gate(cam.exists() and time.time() - cam.stat().st_mtime < 3, "cam log not growing")
    return p


def stop_tracker(p):
    subprocess.run(["sudo", "kill", "-INT", str(p.pid)])
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "kill", str(p.pid)])
    cam, snap = LOGS / "probe_pose_log.jsonl", LOGS / f"sec{sec}_cam.jsonl"
    if cam.exists():
        subprocess.run(["sudo", "mv", cam, snap])
        subprocess.run(["sudo", "chown", os.getlogin(), snap])
        note(cam_log=str(snap))


def start_capture():
    """Headless Cast capture, here on the Mac, writing straight into the section dir."""
    out = open(run / "capture.log", "w")
    p = subprocess.Popen([PY_CL, "src/capture/cast_headless.py", "--section", str(sec),
                          "--ip", PROBE_IP, "--port", str(PROBE_PORT),
                          "--root", str(SESS), "--interval-ms", "50"],
                         stdout=out, stderr=subprocess.STDOUT)
    dst = SESS / f"section_{sec}"
    t0 = time.time()
    while time.time() - t0 < 30:
        time.sleep(1)
        if p.poll() is not None:
            raise SystemExit(f"ABORT: capture died at once, see {run / 'capture.log'}")
        if len(list(dst.glob("raw_*.bin"))) > 0:
            note(capture_started=True)
            return p
    p.send_signal(signal.SIGINT)
    raise SystemExit(f"ABORT: no frames after 30 s, see {run / 'capture.log'}")


def stop_capture(p):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    note(frames=len(list((SESS / f"section_{sec}").glob("raw_*.bin"))))


def fetch_exec(host, exec_path):
    """Only the exec log comes back; the frames are already here."""
    dst = SESS / f"section_{sec}"
    meta = exec_path[:-len(".jsonl")] + "_meta.json"
    sh(["scp", f"{host}:{PI_DIR}/{exec_path}", str(dst / "exec.jsonl")], "fetch-exec")
    sh(["scp", f"{host}:{PI_DIR}/{meta}", str(dst / "exec_meta.json")], "fetch-meta")
    gate((dst / "exec.jsonl").exists() and (dst / "exec_meta.json").exists(), "exec pair missing")
    n = len(list(dst.glob("raw_*.bin")))
    note(frames=n, exec_log=exec_path)
    gate(n > 0, "no frames captured")


lab_ssid = current_ssid()
note(state="started", lab_ssid=lab_ssid, pi_clock_offset_s=pi_clock_offset(PI_ZT))
sh(["-v"], "sudo", sudo=True)                # one password prompt up front
anchor()
solve()
sh(["scp", launch, f"{PI_ZT}:{PI_DIR}/pose_logs/"], "deploy")
sh(["ssh", PI_ZT, f"test -s {PI_DIR}/pose_logs/{launch.name}"], "deploy-verify")

# Bring the Pi onto the probe network first, so we know where to reach it once we
# are on it too and off the lab network.
net = sh(["ssh", PI_ZT, f"cd {PI_DIR} && ./flight.sh net"], "pi-net")
pi_lan = next((l.split("=", 1)[1].strip() for l in net.splitlines() if l.startswith("WLAN0=")), None)
gate(pi_lan, "the Pi did not report its address on the probe network")
note(pi_lan=pi_lan)
pi_probe = f"{PI_USER}@{pi_lan}"

tracker = start_tracker()
capture = None
try:
    join(PROBE_SSID, want_gateway=PROBE_IP)
    capture = start_capture()
    out = sh(["ssh", pi_probe, f"cd {PI_DIR} && ./flight.sh fly pose_logs/{launch.name} "
                               f"{SPEED} {RATE} {DELAY}"], "flight")
    exec_path = next((l.split("=", 1)[1].strip() for l in out.splitlines() if l.startswith("EXEC=")), None)
    gate(exec_path, "the Pi did not report an exec log")
finally:
    if capture is not None:
        stop_capture(capture)
    stop_tracker(tracker)
    if lab_ssid and current_ssid() != lab_ssid:
        try:
            join(lab_ssid)
        except SystemExit as e:
            print(f"  (could not rejoin {lab_ssid}: {e})", flush=True)

fetch_exec(PI_ZT, exec_path)
note(state="acquired")
sh(["make", "post", f"SEC={sec}"], "post")
note(state="done")
print(f"\nsection_{sec} complete. run folder: {run}")
