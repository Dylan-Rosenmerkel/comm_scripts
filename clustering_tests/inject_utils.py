"""Shared by the injection tests (wide_test.py, mhc_tracks.py): where an injected object may be placed."""
import os
import numpy as np
from scipy import ndimage

# default since 2026-10-25: the whole footprint of an injected object must avoid real events (before: only its
# centre was checked, so a 15-35 px track could cross a real event and every method 'failed' on it).
# INJECT_FOOTPRINT=0 reproduces the earlier placement (and the results saved before that date).
FOOTPRINT_CHECK = os.environ.get("INJECT_FOOTPRINT", "1") != "0"


def footprint_busy(busy, iy, ix, grow=2):
    """True if the object's pixels (iy, ix), grown by `grow` px, touch the busy mask"""
    y0, y1 = max(iy.min() - grow, 0), min(iy.max() + grow + 1, busy.shape[0])
    x0, x1 = max(ix.min() - grow, 0), min(ix.max() + grow + 1, busy.shape[1])
    fp = np.zeros((y1 - y0, x1 - x0), bool)
    fp[iy - y0, ix - x0] = True
    fp = ndimage.binary_dilation(fp, iterations=grow)
    return bool((fp & busy[y0:y1, x0:x1]).any())
