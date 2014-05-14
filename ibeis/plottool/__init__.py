# flake8: noqa
from __future__ import absolute_import, division, print_function

__version__ = '1.0.0.dev1'

import matplotlib as mpl
mpl.use('Qt4Agg')

from . import plot_helpers as ph
from . import plot_helpers
from . import mpl_keypoint
from . import mpl_keypoint as mpl_kp
from . import mpl_sift as mpl_sift
from . import draw_func2
from . import draw_func2 as df2
from . import draw_sv
from . import viz_featrow
from . import viz_keypoints
from . import plots

# The other module shouldn't exist.
# Functions in it need to be organized
from .plots import draw_hist_subbin_maxima
from .draw_func2 import *
from .mpl_keypoint import draw_keypoints
from .mpl_sift import draw_sifts
import utool



def reload_subs():
    rrr()
    df2.rrr()
    plot_helpers.rrr()
    draw_sv.rrr()
    viz_keypoints.rrr()

rrrr = reload_subs

print, print_, printDBG, rrr, profile = utool.inject(__name__, '[plottool]')

