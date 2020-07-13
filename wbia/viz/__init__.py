# -*- coding: utf-8 -*-

### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function

import utool as ut

ut.noinject(__name__, '[wbia.viz.__init__]', DEBUG=False)

from wbia.viz import viz_chip
from wbia.viz import viz_helpers
from wbia.viz import viz_hough
from wbia.viz import viz_image
from wbia.viz import viz_matches
from wbia.viz import viz_name
from wbia.viz import viz_nearest_descriptors
from wbia.viz import viz_qres
from wbia.viz import viz_sver
from wbia.viz import viz_graph2
from wbia.viz import viz_other

from wbia.viz import viz_helpers as vh
from wbia.viz.viz_helpers import draw, kp_info, show_keypoint_gradient_orientations
from wbia.viz.viz_image import show_image
from wbia.viz.viz_chip import show_chip
from wbia.viz.viz_name import show_name
from wbia.viz.viz_qres import show_qres, show_qres_top, show_qres_analysis
from wbia.viz.viz_sver import show_sver, _compute_svvars
from wbia.viz.viz_nearest_descriptors import show_nearest_descriptors
from wbia.viz.viz_hough import show_hough_image, show_probability_chip
from wbia.viz.viz_other import chip_montage

import utool

print, rrr, profile = utool.inject2(__name__)


__LOADED__ = False


def import_subs():
    global __LOADED__
    from wbia.viz import interact

    __LOADED__ = True


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys

    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import wbia.viz

    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        submodname, fromimports = tup
        submod = getattr(wbia.viz, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(wbia.viz, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """ Reloads wbia.viz and submodules """
    rrr(verbose=verbose)

    def fbrrr(*args, **kwargs):
        """ fallback reload """
        pass

    getattr(viz_chip, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_helpers, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_hough, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_image, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_matches, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_name, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_nearest_descriptors, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_qres, 'rrr', fbrrr)(verbose=verbose)
    getattr(viz_sver, 'rrr', fbrrr)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)


rrrr = reload_subs

IMPORT_TUPLES = [
    ('viz_chip', None),
    ('viz_helpers', None),
    ('viz_hough', None),
    ('viz_image', None),
    ('viz_matches', None),
    ('viz_name', None),
    ('viz_nearest_descriptors', None),
    ('viz_qres', None),
    ('viz_sver', None),
]

"""
Regen Command:
    cd /home/joncrall/code/wbia/wbia/viz
    makeinit.py
"""
