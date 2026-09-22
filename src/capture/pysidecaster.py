#!/usr/bin/env python
"""
Clarius PAL HD3 — Sweep Capture
================================

GUI tool for capturing raw ultrasound frames + IMU + timestamps for the
robot-driven 3D reconstruction project (Phase 1).

What this saves per sweep:

    data/clarius_sessions/section_<N>/
        manifest.json                  - sweep metadata, frame count, settings
        connection.json                - IP, port, host info
        raw_<probe_ts_ns>.bin          - raw polar frame bytes
        raw_<probe_ts_ns>.json         - per-frame metadata: dims, scale, IMU, host clock
        proc_<probe_ts_ns>.png         - optional scan-converted display image (Save Image)

Run:
    python3 pysidecaster.py             # macOS / Linux
    LD_LIBRARY_PATH=. python3 pysidecaster.py    # if Linux can't find libcast.so

Requires libcast.{so,dylib,dll} + pyclariuscast.so beside cast_capture.py.
"""

import ctypes
import json
import sys
from pathlib import Path
from typing import Final, Optional

# Native libs (libcast + pyclariuscast) live next to this script, regardless of
# where it's launched from. Captured sessions go to the repo's data/ folder
# (anchored below), so output lands in the same place no matter the cwd.
_CWD = Path.cwd()
_LIB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LIB_DIR.parents[1]          # src/capture/ -> repo root
DATA = _REPO_ROOT / "data"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# The capture core (library loading, SDK callbacks, frame store, on-disk format)
# is shared with cast_headless.py so there is exactly one save path.
from cast_capture import Session, host_now, libcast_handle, make_caster, store  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt, Slot  # noqa: E402


# --- Cast userFunction command codes ----------------------------------------
# From clariusdev/cast — see project reference doc §5.4

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14

# remembered connection prefs
PREFS_PATH = Path.home() / ".clarius_last_connect.json"


# --- Qt event plumbing ------------------------------------------------------


class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


class Signaller(QtCore.QObject):
    """Relay SDK callbacks (which run on background threads) to Qt signals."""

    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.usimage = QtGui.QImage()

    def event(self, evt):
        t = evt.type()
        if t == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif t == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif t == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True


signaller = Signaller()  # global — SDK callbacks need module-level access


# --- ImageView: renders B-mode, owns the session directory, saves frames ----


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())
        self.image: Optional[QtGui.QImage] = None

        # one session dir per program run; a new one per Start Scan.
        # Session picks highest existing number + 1, so no folder is ever reused.
        self.session_root = DATA / "clarius_sessions"
        self.session = Session(root=self.session_root, base=_REPO_ROOT)

        # sweep tracking
        self.sweep_active = False
        self.scan_timer: Optional[QtCore.QTimer] = None

    @property
    def section_dir(self) -> Path:
        return self.session.dir

    @property
    def sweep_frame_count(self) -> int:
        return self.session.frames

    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    def saveProcessedImage(self) -> Optional[Path]:
        """Save the scan-converted display image as PNG."""
        _, ts, _ = store.snapshot()
        if ts is None:
            return None
        if self.image is None:
            return None
        filename = self.section_dir / f"proc_{ts}.png"
        self.image.save(str(filename))
        return filename

    def saveRawFrame(self) -> Optional[Path]:
        """
        Save the most recent raw frame as .bin + sidecar .json including IMU.
        Called both on demand (button) and from the sweep timer. force=True keeps
        the GUI's every-tick behaviour: a repeated frame is rewritten, not skipped.
        """
        ts = self.session.save_frame(force=True)
        if ts is None:
            return None
        return self.section_dir / f"raw_{ts}.bin"

    # --- sweep control ------------------------------------------------------

    def startScan(self, interval_ms: int = 50):
        if self.sweep_active:
            return
        # always start a new section on each Start Scan press so we don't mix sweeps
        self.session = Session(root=self.session_root, base=_REPO_ROOT)
        self.sweep_active = True
        self.session.start(interval_ms)

        self.scan_timer = QtCore.QTimer(self)
        self.scan_timer.timeout.connect(self.saveRawFrame)
        self.scan_timer.start(interval_ms)

    def stopScan(self) -> dict:
        if not self.sweep_active:
            return {"frames": 0, "duration_s": 0.0}
        if self.scan_timer is not None:
            self.scan_timer.stop()
            self.scan_timer = None
        self.sweep_active = False
        return self.session.stop()

    # --- rendering ----------------------------------------------------------

    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.cast.setOutputSize(w, h)
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.black)
        self.setSceneRect(0, 0, w, h)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    def drawForeground(self, painter, rect):
        if self.image is not None and not self.image.isNull():
            painter.drawImage(rect, self.image)


# --- main window ------------------------------------------------------------


class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.setWindowTitle("Clarius PAL HD3 — Sweep Capture")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # connection bar
        last = self._load_prefs()
        self.ip = QtWidgets.QLineEdit(last.get("ip", "192.168.1.1"))
        self.ip.setInputMask("000.000.000.000")
        self.port = QtWidgets.QLineEdit(str(last.get("port", "5828")))
        self.port.setInputMask("00000")

        self.conn_btn = QtWidgets.QPushButton("Connect")
        self.run_btn = QtWidgets.QPushButton("Run")
        self.quit_btn = QtWidgets.QPushButton("Quit")

        # capture
        self.start_btn = QtWidgets.QPushButton("Start Scan")
        self.stop_btn = QtWidgets.QPushButton("Stop Scan")
        self.save_proc_btn = QtWidgets.QPushButton("Save Image")
        self.save_raw_btn = QtWidgets.QPushButton("Save Raw Image")

        # control (newly wired)
        self.depth_dec_btn = QtWidgets.QPushButton("Depth −")
        self.depth_inc_btn = QtWidgets.QPushButton("Depth +")
        self.gain_dec_btn = QtWidgets.QPushButton("Gain −")
        self.gain_inc_btn = QtWidgets.QPushButton("Gain +")
        self.bmode_btn = QtWidgets.QPushButton("B-Mode")
        self.cfi_btn = QtWidgets.QPushButton("Color Doppler")
        self.cap_img_btn = QtWidgets.QPushButton("Capture (Probe)")
        self.cap_cine_btn = QtWidgets.QPushButton("Capture Cine (Probe)")

        # wire everything
        self.conn_btn.clicked.connect(self._toggle_connect)
        self.run_btn.clicked.connect(lambda: self._send_cmd(CMD_FREEZE))
        self.quit_btn.clicked.connect(self.shutdown)

        self.start_btn.clicked.connect(self._start_scan)
        self.stop_btn.clicked.connect(self._stop_scan)
        self.save_proc_btn.clicked.connect(self._save_processed)
        self.save_raw_btn.clicked.connect(self._save_raw)

        self.depth_dec_btn.clicked.connect(lambda: self._send_cmd(CMD_DEPTH_DEC))
        self.depth_inc_btn.clicked.connect(lambda: self._send_cmd(CMD_DEPTH_INC))
        self.gain_dec_btn.clicked.connect(lambda: self._send_cmd(CMD_GAIN_DEC))
        self.gain_inc_btn.clicked.connect(lambda: self._send_cmd(CMD_GAIN_INC))
        self.bmode_btn.clicked.connect(lambda: self._send_cmd(CMD_B_MODE))
        self.cfi_btn.clicked.connect(lambda: self._send_cmd(CMD_CFI_MODE))
        self.cap_img_btn.clicked.connect(lambda: self._send_cmd(CMD_CAPTURE_IMAGE))
        self.cap_cine_btn.clicked.connect(lambda: self._send_cmd(CMD_CAPTURE_CINE))

        # ----- layout -----
        self.img = ImageView(cast)
        root = QtWidgets.QVBoxLayout()
        root.addWidget(self.img, stretch=1)

        ip_row = QtWidgets.QHBoxLayout()
        ip_row.addWidget(QtWidgets.QLabel("IP:"))
        ip_row.addWidget(self.ip)
        ip_row.addWidget(QtWidgets.QLabel("Port:"))
        ip_row.addWidget(self.port)
        root.addLayout(ip_row)

        conn_row = QtWidgets.QHBoxLayout()
        conn_row.addWidget(self.conn_btn)
        conn_row.addWidget(self.run_btn)
        conn_row.addWidget(self.quit_btn)
        root.addLayout(conn_row)

        cap_row = QtWidgets.QHBoxLayout()
        cap_row.addWidget(self.start_btn)
        cap_row.addWidget(self.stop_btn)
        cap_row.addWidget(self.save_proc_btn)
        cap_row.addWidget(self.save_raw_btn)
        root.addLayout(cap_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        for b in (self.depth_dec_btn, self.depth_inc_btn,
                  self.gain_dec_btn, self.gain_inc_btn):
            ctrl_row.addWidget(b)
        root.addLayout(ctrl_row)

        mode_row = QtWidgets.QHBoxLayout()
        for b in (self.bmode_btn, self.cfi_btn,
                  self.cap_img_btn, self.cap_cine_btn):
            mode_row.addWidget(b)
        root.addLayout(mode_row)

        central.setLayout(root)

        # signals from SDK callbacks
        signaller.freeze.connect(self._on_freeze)
        signaller.button.connect(self._on_probe_button)
        signaller.image.connect(self.img.updateImage)

        # init SDK rendering
        path = str(_CWD)
        if cast.init(path, 640, 480):
            self.statusBar().showMessage("Initialized — fill in IP/port and click Connect")
        else:
            self.statusBar().showMessage("SDK initialization failed")

        # frame-rate display
        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    # ---- prefs --------------------------------------------------------------

    def _load_prefs(self) -> dict:
        try:
            return json.loads(PREFS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_prefs(self):
        try:
            PREFS_PATH.write_text(json.dumps({
                "ip": self.ip.text(),
                "port": int(self.port.text()),
            }))
        except (ValueError, OSError):
            pass

    # ---- connect / commands -------------------------------------------------

    def _toggle_connect(self):
        if not self.cast.isConnected():
            try:
                p = int(self.port.text())
            except ValueError:
                self.statusBar().showMessage("Invalid port")
                return
            ok = self.cast.connect(self.ip.text(), p, "research")
            if ok:
                self.statusBar().showMessage(f"Connected to {self.ip.text()}:{p}")
                self.conn_btn.setText("Disconnect")
                self._save_prefs()
                self._write_connection_info(self.ip.text(), p)
            else:
                self.statusBar().showMessage(
                    f"Failed to connect to {self.ip.text()}:{self.port.text()}"
                )
        else:
            if self.cast.disconnect():
                self.statusBar().showMessage("Disconnected")
                self.conn_btn.setText("Connect")
            else:
                self.statusBar().showMessage("Failed to disconnect")

    def _send_cmd(self, code: int, arg: int = 0):
        if self.cast.isConnected():
            self.cast.userFunction(code, arg)
        else:
            self.statusBar().showMessage("Not connected")

    def _write_connection_info(self, ip: str, port: int):
        self.img.session.write_connection(ip, port)

    # ---- scan capture -------------------------------------------------------

    def _start_scan(self):
        if not self.cast.isConnected():
            self.statusBar().showMessage("Not connected to probe")
            return
        self.img.startScan(interval_ms=50)
        # connection info also goes into the (new) section folder
        self._write_connection_info(self.ip.text(), int(self.port.text()))
        self.statusBar().showMessage(
            f"Scanning → {self.img.section_dir.name}/ (raw frame every 500 ms)"
        )

    def _stop_scan(self):
        summary = self.img.stopScan()
        self.statusBar().showMessage(
            f"Sweep done: {summary['frames']} frames in {summary['duration_s']} s"
        )

    def _save_processed(self):
        out = self.img.saveProcessedImage()
        if out:
            self.statusBar().showMessage(f"Saved {out.name}")
        else:
            self.statusBar().showMessage("No processed frame available yet")

    def _save_raw(self):
        out = self.img.saveRawFrame()
        if out:
            self.statusBar().showMessage(f"Saved {out.name}")
        else:
            self.statusBar().showMessage("No raw frame available yet")

    # ---- SDK signal handlers -----------------------------------------------

    @Slot(bool)
    def _on_freeze(self, frozen: bool):
        if frozen:
            self.run_btn.setText("Run")
            self.statusBar().showMessage("Image frozen")
        else:
            self.run_btn.setText("Freeze")
            self.statusBar().showMessage(
                "Image running (firewall? check if no image appears)"
            )

    @Slot(int, int)
    def _on_probe_button(self, btn: int, clicks: int):
        self.statusBar().showMessage(f"Probe button {btn} pressed ({clicks} clicks)")

    def _update_fps(self):
        # frames-saved-per-second during a sweep — different from streamed fps
        if self.img.sweep_active:
            self.statusBar().showMessage(
                f"Scanning → {self.img.section_dir.name}/ "
                f"frames={self.img.sweep_frame_count}"
            )

    # ---- shutdown -----------------------------------------------------------

    @Slot()
    def shutdown(self):
        if self.img.sweep_active:
            self.img.stopScan()
        if self.cast.isConnected():
            self.cast.disconnect()
        if sys.platform.startswith("linux"):
            try:
                ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
            except OSError:
                pass
        self.cast.destroy()
        QtWidgets.QApplication.quit()


# --- SDK callbacks (run on background threads, keep them fast) -------------
# The raw-frame and IMU bookkeeping live in cast_capture; only the display path
# and the two Qt relays are the GUI's own.


def on_image(image, width, height, sz, micronsPerPixel, timestamp, angle):
    """Display frame. The IMU that came with it is already in the store."""
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    # deep copy: the SDK's image buffer is invalid after this returns
    signaller.usimage = img.copy()
    QtCore.QCoreApplication.postEvent(signaller, ImageEvent())


def freezeFn(frozen):
    QtCore.QCoreApplication.postEvent(signaller, FreezeEvent(frozen))


def buttonsFn(button, clicks):
    QtCore.QCoreApplication.postEvent(signaller, ButtonEvent(button, clicks))


# --- entry point ------------------------------------------------------------


def main():
    cast = make_caster(on_image=on_image, on_freeze=freezeFn, on_button=buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWidget(cast)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()