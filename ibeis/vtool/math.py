# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[math]', DEBUG=False)


tau = 2 * np.pi  # References: tauday.com

eps = 1E-9
