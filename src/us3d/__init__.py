"""us3d — shared helpers for the robotic 3D ultrasound pipeline.

Scripts live at src/<area>/<name>.py (two directories under the repo root) and
reach this package with a three-line bootstrap:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from us3d.paths import SESSIONS

That makes a script runnable from any working directory and from any conda
env, with no install step. See docs/CHEATSHEET.md.
"""

__all__ = ["paths", "sections", "frames", "handeye", "video", "tube"]
