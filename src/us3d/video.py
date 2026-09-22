"""Writing an overlay video.

Both video scripts carried the same codec-fallback ladder and the same size
guard. The guard matters: cv2.VideoWriter silently discards any frame whose
shape does not match the size it was opened with, producing a file that looks
written but plays back empty.
"""
import sys
from pathlib import Path

from us3d.paths import out_dir

# (extension, fourcc), tried in order. avc1 gives the most portable mp4 when
# the local OpenCV has it; MJPG in an avi always works but is much larger.
CODECS = (("mp4", "avc1"), ("mp4", "mp4v"), ("avi", "MJPG"))

MIN_BYTES = 100_000


def open_writer(stem, size_wh, fps=20.0, directory=None):
    """Open a writer for <stem>.<ext>, trying each codec. Returns (writer, path)."""
    import cv2

    directory = Path(directory) if directory is not None else out_dir("videos")
    directory.mkdir(parents=True, exist_ok=True)
    for ext, fourcc in CODECS:
        path = directory / ("%s.%s" % (stem, ext))
        w = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, tuple(size_wh)
        )
        if w.isOpened():
            return w, path
        w.release()
    sys.exit(
        "No working video codec: tried %s"
        % ", ".join("%s/%s" % (f, e) for e, f in CODECS)
    )


def fit_frame(frame, size_wh):
    """Resize a frame if it does not match the writer's size, so it is not dropped."""
    import cv2

    w, h = size_wh
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def finalize(writer, path, min_bytes=MIN_BYTES):
    """Release the writer and fail loudly if the file came out suspiciously small."""
    writer.release()
    size = path.stat().st_size if path.exists() else 0
    if size < min_bytes:
        sys.exit(
            "%s is only %d bytes - the codec accepted frames but wrote nothing "
            "usable. Try another codec." % (path, size)
        )
    return path
