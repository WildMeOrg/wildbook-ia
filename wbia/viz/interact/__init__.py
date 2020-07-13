# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function

import utool as ut

ut.noinject(__name__, '[wbia.viz.interact.__init__]', DEBUG=False)

from wbia.plottool import interact_helpers as ih

from wbia.viz.interact import interact_annotations2
from wbia.viz.interact import interact_chip
from wbia.viz.interact import interact_image
from wbia.viz.interact import interact_matches
from wbia.viz.interact import interact_name
from wbia.viz.interact import interact_qres
from wbia.viz.interact import interact_sver

from wbia.viz.interact.interact_image import ishow_image
from wbia.viz.interact.interact_chip import ishow_chip
from wbia.viz.interact.interact_name import ishow_name
from wbia.viz.interact.interact_sver import ishow_sver

import utool

print, rrr, profile = utool.inject2(__name__, '[wbia.viz.interact]')


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys

    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import wbia.viz.interact

    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(wbia.viz.interact, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(wbia.viz.interact, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """ Reloads wbia.viz.interact and submodules """
    rrr(verbose=verbose)

    def fbrrr(*args, **kwargs):
        """ fallback reload """
        pass

    getattr(interact_annotations2, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_chip, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_image, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_matches, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_name, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_qres, 'rrr', fbrrr)(verbose=verbose)
    getattr(interact_sver, 'rrr', fbrrr)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)


rrrr = reload_subs

IMPORT_TUPLES = [
    ('interact_annotations2', None),
    ('interact_chip', None),
    ('interact_image', None),
    ('interact_matches', None),
    ('interact_name', None),
    ('interact_qres', None),
    ('interact_sver', None),
]

"""
Regen Command:
    cd /home/joncrall/code/wbia/wbia/viz/interact
    makeinit.py
"""
