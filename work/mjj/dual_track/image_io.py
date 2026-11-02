"""OpenCV codecs with Python Unicode filesystem I/O on Windows."""
from pathlib import Path
import os
import shutil
import tempfile
import cv2
import numpy as np

def read_image(path, flags=cv2.IMREAD_COLOR):
    if Path(path).suffix.lower() == '.exr':
        fd, name = tempfile.mkstemp(suffix='.exr', dir=Path.cwd())
        os.close(fd)
        try:
            shutil.copyfile(path, name)
            return cv2.imread(Path(name).name, flags)
        finally:
            Path(name).unlink(missing_ok=True)
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)

def write_image(path, image, params=None):
    path = Path(path)
    # OpenCV's OpenEXR encoder requires a filesystem stream; imencode fails.
    # An ASCII relative name also avoids its Windows Unicode path limitation.
    if path.suffix.lower() == '.exr':
        fd, name = tempfile.mkstemp(suffix='.exr', dir=Path.cwd())
        os.close(fd)
        try:
            ok = cv2.imwrite(Path(name).name, image, params or [])
            if ok:
                shutil.copyfile(name, path)
            return bool(ok)
        finally:
            Path(name).unlink(missing_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image, params or [])
    if ok:
        encoded.tofile(str(path))
    return bool(ok)
