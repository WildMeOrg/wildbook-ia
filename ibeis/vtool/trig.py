# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
#import utool
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[trig]')

"""
#if CYTH
cdef np.float64_t TAU
#endif
"""

TAU = 2 * np.pi  # tauday.com


def atan2(y, x):
    """ does atan2 but returns from 0 to TAU
    >>> from vtool.trig import *  # NOQA
    >>> import utool
    >>> np.random.seed(0)
    >>> y = np.random.rand(1000).astype(np.float64)
    >>> x = np.random.rand(1000).astype(np.float64)
    >>> theta = atan2(y, x)
    >>> assert np.all(theta >= 0)
    >>> assert np.all(theta < 2 * np.pi)
    >>> print(utool.hashstr(theta))
    go!su97%eo4qoyq7

    #CYTH_INLINE
    #if CYTH
    cdef:
        np.ndarray[np.float64_t, ndim=1] y
        np.ndarray[np.float64_t, ndim=1] x
        np.ndarray[np.float64_t, ndim=1] theta
    #endif
    """
    theta = np.arctan2(y, x)  # outputs from -TAU/2 to TAU/2
    theta[theta < 0] = theta[theta < 0] + TAU  # map to 0 to TAU (keep coords)
    #theta = theta % TAU
    return theta

import cyth
if cyth.DYNAMIC:
    exec(cyth.import_cyth_execstr(__name__))
else:
    # <AUTOGEN_CYTH>
    # Regen command: python -c "import vtool.trig" --cyth-write
    try:
        if not cyth.WITH_CYTH:
            raise ImportError('no cyth')
        import vtool._trig_cyth
        _atan2_cyth = vtool._trig_cyth._atan2_cyth
        atan2_cyth  = vtool._trig_cyth._atan2_cyth
        CYTHONIZED = True
    except ImportError:
        atan2_cyth = atan2
        CYTHONIZED = False
    # </AUTOGEN_CYTH>
