from __future__ import absolute_import, division, print_function

from . import mpl_keypoint  # NOQA
from . import mpl_keypoint as mpl_kp  # NOQA
from . import mpl_sift as mpl_sift  # NOQA
from . import draw_func2  # NOQA
from . import draw_func2 as df2  # NOQA
# The other module shouldn't exist.
# Functions in it need to be organized
from .plots import draw_hist_subbin_maxima  # NOQA
from .draw_func2 import *  # NOQA
from .mpl_keypoint import draw_keypoints  # NOQA
from .mpl_sift import draw_sifts  # NOQA
