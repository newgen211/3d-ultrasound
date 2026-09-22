"""Compatibility shim: the detector now lives in src/us3d/tube.py.

Kept so that callers still doing `sys.path.insert(0, "src/segment")` and
`import segment_tube` keep working while they are migrated.

This rebinds sys.modules to the us3d.tube module object rather than copying
names out of it. That matters: score_classical.py sets `st.TOP_CROP_MM = 0.5`
to produce its NO-CROP arm, and candidates() reads that as a module global. A
`from us3d.tube import *` re-export would give the caller a second module whose
attribute writes never reach candidates(), silently changing a published
number.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import us3d.tube as _tube

sys.modules[__name__] = _tube
