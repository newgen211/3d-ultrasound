#!/usr/bin/env python3
"""
reward_signals.py — callable reward module for RL probe control.

One B-mode frame in -> four perception signals out. Same code serves the
sim/training loop and the live controller on the real arm, so the policy
never sees a different signal at deployment than it trained on.

    from reward_signals import RewardSignals
    rs = RewardSignals(model="models/best_regated.pt")
    out = rs.update(gray_frame, axial_mm=0.0513, lateral_mm=0.130)
    # -> dict(centered=..., visible=..., patency=..., coupling=...,
    #         n_vessels=..., tracked=bool)

SIGNALS (all normalized, higher = better, None only where honestly unknown):

  visible   0..1  detector confidence mass this frame. 0 = no vessel seen.
  centered  0..1  1 when the strongest vessel sits on the image centerline,
                  falling to 0 at the lateral edge. None if nothing visible.
  patency   0..1  1 - kappa. 1 = vessel as round as gentle-contact reference,
                  0 = flattened/shut. None if no vessel is tracked (UNKNOWN is
                  not the same as SHUT — the policy must be able to tell the
                  difference; a lost vessel is a `visible` problem, not a
                  pressure reading).
  coupling  0..1  deep-structure energy vs the running open-frame reference.
                  Low = washed-out frame = contact too LIGHT / poor gel.

Note the deliberate opposition: patency wants LESS force, coupling wants
MORE. That tension is why depth control is worth learning rather than
hand-tuning, and it is the core of the 7th DOF.

CAUSAL BY DESIGN: no future frames, no whole-sweep statistics. The kappa
baseline comes from a stored gentle-contact reference constant (measured on
a freehand capture), not from a per-track p90, because a live controller
cannot see the rest of the sweep.
"""

from collections import deque
import numpy as np

# gentle-contact reference: aspect p75 measured on a freehand capture
# (section_81, 1175 detections). Override per probe/phantom if recalibrated.
BASELINE_ASPECT = 0.73

MAX_JUMP_PX = 40      # frame-to-frame association gate
TRACK_TIMEOUT = 25    # frames a track survives unseen
DEEP_FRAC = 0.40      # image fraction considered "deep" for coupling
COUPLING_REF_LEN = 90 # rolling frames used as the open-frame reference


class _Track:
    __slots__ = ("cx", "cy", "aspect", "age", "unseen")

    def __init__(self, cx, cy, aspect):
        self.cx, self.cy, self.aspect = cx, cy, aspect
        self.age, self.unseen = 1, 0


class RewardSignals:
    def __init__(self, model="models/best_regated.pt", conf=0.25,
                 baseline_aspect=BASELINE_ASPECT, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.baseline = float(baseline_aspect)
        self.device = device
        self.tracks = []
        self._deep_ref = deque(maxlen=COUPLING_REF_LEN)

    # ---------- main entry ----------
    def update(self, gray, axial_mm, lateral_mm):
        """gray: HxW uint8 B-mode frame. Returns the signal dict."""
        H, W = gray.shape[:2]
        dets = self._detect(gray, W, H, axial_mm, lateral_mm)

        visible = float(np.clip(sum(d["conf"] for d in dets), 0, 1)) if dets else 0.0

        centered = None
        if dets:
            best = max(dets, key=lambda d: d["conf"])
            offset = abs(best["cx"] - W / 2) / (W / 2)
            centered = float(np.clip(1.0 - offset, 0, 1))

        self._associate(dets)
        patency, tracked = self._patency()
        coupling = self._coupling(gray, bool(dets))

        return dict(visible=visible, centered=centered, patency=patency,
                    coupling=coupling, n_vessels=len(dets), tracked=tracked)

    def reset(self):
        self.tracks.clear()
        self._deep_ref.clear()

    # ---------- pieces ----------
    def _detect(self, gray, W, H, axial_mm, lateral_mm):
        import cv2
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
        kw = dict(conf=self.conf, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(bgr, **kw)[0]
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            w_px, h_px = x2 - x1, y2 - y1
            w_mm, h_mm = w_px * lateral_mm, h_px * axial_mm
            out.append(dict(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                            aspect=h_mm / max(w_mm, 1e-3),
                            conf=float(b.conf[0])))
        return out

    def _associate(self, dets):
        for t in self.tracks:
            t.unseen += 1
        used = set()
        for d in dets:
            best, bd = None, MAX_JUMP_PX
            for i, t in enumerate(self.tracks):
                if i in used or t.unseen > TRACK_TIMEOUT:
                    continue
                dd = float(np.hypot(d["cx"] - t.cx, d["cy"] - t.cy))
                if dd < bd:
                    best, bd = i, dd
            if best is None:
                self.tracks.append(_Track(d["cx"], d["cy"], d["aspect"]))
            else:
                t = self.tracks[best]
                t.cx, t.cy = d["cx"], d["cy"]
                t.aspect = 0.6 * t.aspect + 0.4 * d["aspect"]   # smoothed
                t.age += 1
                t.unseen = 0
                used.add(best)
        self.tracks = [t for t in self.tracks if t.unseen <= TRACK_TIMEOUT]

    def _patency(self):
        live = [t for t in self.tracks if t.unseen == 0 and t.age >= 3]
        if not live:
            return None, False           # UNKNOWN, not "shut"
        kap = [np.clip(1.0 - t.aspect / self.baseline, 0.0, 1.0) for t in live]
        return float(np.clip(1.0 - float(np.mean(kap)), 0, 1)), True

    def _coupling(self, gray, saw_vessel):
        deep = gray[int(DEEP_FRAC * gray.shape[0]):, :]
        e = float(deep.std())
        if saw_vessel:                    # only OPEN frames define the reference
            self._deep_ref.append(e)
        if len(self._deep_ref) < 5:
            return None
        ref = float(np.median(self._deep_ref)) or 1.0
        return float(np.clip(e / ref, 0, 1))


# ---------- convenience: single scalar reward ----------
def combine(sig, w=None):
    """Example scalarization. Terms that are None contribute nothing and their
    weight is redistributed, so 'unknown' never silently reads as 'bad'."""
    w = w or dict(visible=1.0, centered=1.0, patency=1.5, coupling=0.5)
    num = den = 0.0
    for k, wt in w.items():
        v = sig.get(k)
        if v is not None:
            num += wt * float(v)
            den += wt
    return num / den if den else 0.0