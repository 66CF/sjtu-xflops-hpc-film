"""Stable printed identity for a particle throughout its lifetime."""
import numpy as np


def visible_glyphs(group):
    """Sampling is fixed by ID, never by time, scene, speed or release state."""
    group = np.asarray(group)
    ids = np.arange(len(group), dtype=np.uint64)
    hashed = ((ids * 2654435761) & 4294967295) % 3 != 0
    return ((group == 0) | ((group == 1) & hashed) |
            ((group == 2) & (ids % 4 == 0)) | (group == 3))
