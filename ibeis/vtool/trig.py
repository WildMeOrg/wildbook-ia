# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[trig]', DEBUG=False)


np.tau = tau = 2 * np.pi  # tauday.com


def atan2(y, x):
    """ does atan2 but returns from 0 to tau
    #if CYTH
    #endif
    """
    theta = np.arctan2(y, x)  # outputs from -tau/2 to tau/2
    theta[theta < 0] = theta[theta < 0] + tau  # map to 0 to tau (keep coords)
    #theta = theta % np.tau
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
        raise ImportError("no cyth")
        CYTHONIZED = True
    except ImportError:
        atan2_cyth = atan2
        CYTHONIZED = False
    # </AUTOGEN_CYTH>
