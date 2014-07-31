# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[trig]', DEBUG=False)


np.tau = 2 * np.pi  # tauday.com


def atan2(y, x):
    """ does atan2 but returns from 0 to tau """
    theta = np.arctan2(y, x)  # outputs from -tau/2 to tau/2
    theta[theta < 0] = theta[theta < 0] + np.tau  # map to 0 to tau (keep coords)
    #theta = theta % np.tau
    return theta
