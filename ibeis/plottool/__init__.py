# flake8: noqa
from __future__ import absolute_import, division, print_function

__version__ = '1.0.0.dev1'

# Hopefully this was imported sooner. TODO remove dependency
from guitool import __PYQT__
from plottool import __MPL_INIT__
__MPL_INIT__.init_matplotlib()

import matplotlib as mpl
#mpl.use('Qt4Agg')

from plottool import plot_helpers as ph
from plottool import plot_helpers
from plottool import mpl_keypoint
from plottool import mpl_keypoint as mpl_kp
from plottool import mpl_sift as mpl_sift
from plottool import draw_func2
from plottool import draw_func2 as df2
from plottool import draw_sv
from plottool import viz_featrow
from plottool import viz_keypoints
from plottool import viz_image2
from plottool import plots
from plottool import interact_annotations
from plottool import interact_keypoints
from plottool import interact_multi_image

# The other module shouldn't exist.
# Functions in it need to be organized
from plottool.plots import draw_hist_subbin_maxima
from plottool.draw_func2 import *  # NOQA
from plottool.mpl_keypoint import draw_keypoints
from plottool.mpl_sift import draw_sifts
from plottool import fig_presenter

import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[plottool]')

def reload_subs():
    rrr()
    df2.rrr()
    plot_helpers.rrr()
    draw_sv.rrr()
    viz_keypoints.rrr()
    viz_image2.rrr()
    rrr()

rrrr = reload_subs
