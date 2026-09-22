"""Locating a capture section and enumerating its frames.

Reconciled from the thirteen copies of find_section() that were scattered
across src/ and the root scripts. Semantics follow the fullest variant (the
one in reconstruct_handeye.py / calibrate_handeye.py), plus two additions:

  * a bare number resolves, so "106" means section_106;
  * a name that matches nothing exits here with a clear message, instead of
    returning a path that fails later with a confusing "no frames" error.
"""
import json
import sys
from pathlib import Path

from us3d.paths import SESSIONS


def _section_number(p):
    tail = p.name.split("_")[-1]
    return int(tail) if tail.isdigit() else -1


def list_sections(root=None):
    """Every section_N directory, ordered by number."""
    root = Path(root) if root is not None else SESSIONS
    if not root.exists():
        return []
    return sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("section_")),
        key=_section_number,
    )


def find_section(arg=None, root=None):
    """Resolve a section. No argument means the newest one.

    Accepts a path (absolute or relative to the working directory), a full
    name such as "section_106", or a bare number such as "106".
    """
    root = Path(root) if root is not None else SESSIONS
    if arg is None:
        if not root.exists():
            sys.exit("No clarius_sessions/ folder at %s" % root)
        sections = list_sections(root)
        if not sections:
            sys.exit("No section_N folders in %s" % root)
        return sections[-1]

    arg = str(arg)
    candidates = [Path(arg), root / arg]
    if arg.isdigit():
        candidates.append(root / ("section_" + arg))
    for cand in candidates:
        if cand.exists():
            return cand
    sys.exit("Section folder not found: %s (looked in %s)" % (arg, root))


def raw_sidecars(section):
    """The raw_*.json metadata files of a section, in capture order."""
    return sorted(Path(section).glob("raw_*.json"))


def raw_pairs(section):
    """(json, bin) pairs for a section, skipping any sidecar with no payload."""
    pairs = []
    for jp in raw_sidecars(section):
        bp = jp.with_suffix(".bin")
        if bp.exists():
            pairs.append((jp, bp))
    return pairs


def read_meta(json_path):
    """Parse one raw_*.json sidecar."""
    with open(json_path) as fh:
        return json.load(fh)


def frame_scale(meta):
    """(axial_mm_per_sample, lateral_mm_per_line) from a sidecar."""
    f = meta["frame"] if "frame" in meta else meta
    return f["axial_um_per_sample"] / 1000.0, f["lateral_um_per_line"] / 1000.0
