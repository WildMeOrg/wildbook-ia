# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)

from . import viz_helpers
from . import viz_image
from . import viz_chip
from . import viz_matches
from . import viz_qres
from . import viz_nearest_descriptors
from . import viz_sver

from . import viz_helpers as vh
from .viz_helpers import draw, kp_info, show_keypoint_gradient_orientations
from .viz_image import show_image
from .viz_chip import show_chip
from .viz_name import show_name
from .viz_matches import show_matches, annotate_matches
from .viz_qres import show_qres, show_qres_top, show_qres_analysis
from .viz_sver import show_sver, _compute_svvars
from .viz_nearest_descriptors import show_nearest_descriptors

__LOADED__ = False

def import_subs():
    global __LOADED__
    from . import interact
    __LOADED__ = True


def reload_subs():
    if not __LOADED__:
        import_subs()
    viz_helpers.rrr()
    viz_name.rrr()
    viz_image.rrr()
    viz_chip.rrr()
    viz_matches.rrr()
    viz_qres.rrr()
    viz_nearest_descriptors.rrr()
    viz_sver.rrr()
    rrr()
    interact.reload_subs()

rrrr = reload_subs
