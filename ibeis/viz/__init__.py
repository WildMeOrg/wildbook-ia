# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)

#import plottool

from . import viz_helpers
from . import viz_chip
from . import viz_image
from . import viz_matches
from . import viz_name
from . import viz_nearest_descriptors
from . import viz_qres
from . import viz_sver
from . import viz_hough

from . import viz_helpers as vh
from .viz_helpers import draw, kp_info, show_keypoint_gradient_orientations
from .viz_image import show_image
from .viz_chip import show_chip
from .viz_name import show_name
from .viz_matches import show_matches, annotate_matches
from .viz_qres import show_qres, show_qres_top, show_qres_analysis
from .viz_sver import show_sver, _compute_svvars
from .viz_nearest_descriptors import show_nearest_descriptors
from .viz_hough import show_hough_image, show_probability_chip

__LOADED__ = False

def import_subs():
    global __LOADED__
    from . import interact
    __LOADED__ = True

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[viz]')

def reload_subs():
    """ Reloads viz and submodules """
    if not __LOADED__:
        import_subs()
    rrr()
    getattr(viz_chip, 'rrr', lambda: None)()
    getattr(viz_helpers, 'rrr', lambda: None)()
    getattr(viz_image, 'rrr', lambda: None)()
    getattr(viz_matches, 'rrr', lambda: None)()
    getattr(viz_name, 'rrr', lambda: None)()
    getattr(viz_nearest_descriptors, 'rrr', lambda: None)()
    getattr(viz_qres, 'rrr', lambda: None)()
    getattr(viz_sver, 'rrr', lambda: None)()
    getattr(viz_hough, 'rrr', lambda: None)()
    getattr(interact, 'reload_subs', lambda: None)()
    rrr()
rrrr = reload_subs
