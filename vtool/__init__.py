from __future__ import print_function, division
import histogram  # NOQA
import linalg     # NOQA
import image      # NOQA
import keypoint   # NOQA
import ellipse    # NOQA
import drawtool   # NOQA
import patch      # NOQA
#from hist import *  # NOQA
#from patch import *  # NOQA
#from ellipse import *  # NOQA


def rrr():
    import imp
    import sys
    mod_list = [
        sys.modules[__name__],
        patch,
        image,
        ellipse,
        keypoint,
        histogram,
    ]
    map(imp.reload, mod_list)
    drawtool.rrr()
