"""Repo-anchored locations. Standard library only, so this imports in every env.

Every path is derived from this file's own location, so a script works from any
working directory. Two environment variables redirect the mutable trees, which
is how tests and dry runs stay away from the real 9.5 GB of captures:

    US3D_DATA      where data/ lives        (captures, pose logs)
    US3D_OUTPUTS   where outputs/ lives     (figures, videos, logs)
"""
import os
from pathlib import Path

# src/us3d/paths.py -> src/us3d -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = Path(os.environ.get("US3D_DATA") or (REPO_ROOT / "data")).resolve()
SESSIONS = DATA / "clarius_sessions"
POSE_LOGS = DATA / "pose_logs"

OUTPUTS = Path(os.environ.get("US3D_OUTPUTS") or (REPO_ROOT / "outputs")).resolve()

CALIB = REPO_ROOT / "calib"
MODELS = REPO_ROOT / "models"
AUDIT = REPO_ROOT / "audit"
PAPER = REPO_ROOT / "paper"
PAPER_FIGS = PAPER / "figs"
HANDOFF = REPO_ROOT / "handoff"
YOLO_HUMAN = REPO_ROOT / "yolo_ds_human"

AUDIT_TRUTH = AUDIT / "audit_truth.json"
AUDIT_MANIFEST = AUDIT / "audit_manifest.json"
EXEMPLARS = AUDIT / "exemplars.json"


def out_dir(*parts):
    """A directory under outputs/, created on demand. out_dir("videos") -> Path."""
    d = OUTPUTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model(name):
    """Resolve a checkpoint: an existing path as given, else models/<name>."""
    p = Path(name)
    return p if p.exists() else MODELS / name
