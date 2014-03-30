from __future__ import print_function, division
import numpy as np
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[dist]')


def nearest_point(x, y, pts, mode='random'):
    """ finds the nearest point(s) in pts to (x, y) """
    dists = (pts.T[0] - x) ** 2 + (pts.T[1] - y) ** 2
    fx = dists.argmin()
    mindist = dists[fx]
    other_fx = np.where(mindist == dists)[0]
    if len(other_fx) > 0:
        if mode == 'random':
            np.random.shuffle(other_fx)
            fx = other_fx[0]
        if mode == 'all':
            fx = other_fx
        if mode == 'first':
            fx = fx
    return fx, mindist
