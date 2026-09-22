"""Decoding a captured B-mode frame.

The body of load_frame() is the one from segment_tube.py, which five of the
seven copies in the tree already matched byte for byte. PIL is imported lazily
because the realsense env does not have it and only jpg-mode captures need it.
"""
import numpy as np


def load_frame(bin_path, meta):
    """Decode one raw frame to float32, shaped (depth samples, scan lines)."""
    f = meta["frame"] if "frame" in meta else meta
    lines, samples, bps = f["lines"], f["samples"], f["bps"]
    jpg = f.get("jpg_size", 0)
    raw = open(str(bin_path), "rb").read()

    if jpg > 0:
        import io

        from PIL import Image

        return np.array(Image.open(io.BytesIO(raw)).convert("L")).astype(np.float32)

    if bps == 8:
        dtype = np.uint8
    elif bps == 16:
        dtype = np.uint16
    else:
        raise ValueError("unsupported bits-per-sample %r in %s" % (bps, bin_path))

    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size != lines * samples:
        # A truncated tail happens when a capture is cut mid-frame; keep whole lines.
        usable = (arr.size // lines) * lines
        arr = arr[:usable]
        samples = usable // lines
    return arr.reshape(lines, samples).T.astype(np.float32)


def to_u8(img):
    """Scale an arbitrary-range frame to 0-255 uint8 for display or detection."""
    g = img - img.min()
    return (255 * g / max(1.0, g.max())).astype(np.uint8)


def load_u8(json_path):
    """Convenience: sidecar path -> (uint8 image, metadata dict)."""
    from us3d.sections import read_meta

    meta = read_meta(json_path)
    from pathlib import Path

    return to_u8(load_frame(Path(str(json_path)).with_suffix(".bin"), meta)), meta
