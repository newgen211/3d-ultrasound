"""Locating and loading the hand-eye calibration.

Ten copies of this search existed, in two mutually inconsistent forms: half
anchored to the repo root, half relative to the working directory. The
anchored form is the correct one and is what lives here.

Search order: an explicit path, then the section's own handeye.json, then
calib/handeye.json.
"""
import json
import sys
from pathlib import Path

from us3d.paths import CALIB


def find_handeye(section=None, explicit=None):
    """Path to the hand-eye file to use, or exit with an actionable message."""
    if explicit:
        cands = [Path(explicit)]
    else:
        cands = []
        if section is not None:
            cands.append(Path(section) / "handeye.json")
        cands.append(CALIB / "handeye.json")
    for c in cands:
        if c and c.exists():
            return c
    sys.exit(
        "No handeye.json found (looked in: %s). Run "
        "src/calibration/calibrate_handeye.py, copy the result to calib/, or "
        "pass --handeye PATH." % ", ".join(str(c) for c in cands)
    )


def load_handeye(section=None, explicit=None):
    """(parsed hand-eye dict, path it came from)."""
    p = find_handeye(section, explicit)
    with open(p) as fh:
        return json.load(fh), p


def handeye_arrays(he):
    """(R_flange_to_image 3x3, t_flange_to_image_mm 3, convention)."""
    import numpy as np

    return (
        np.array(he["R_flange_to_image"], float),
        np.array(he["t_flange_to_image_mm"], float),
        he["convention"],
    )
