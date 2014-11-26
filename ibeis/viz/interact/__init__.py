### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function

import utool as ut
ut.noinject(__name__, '[ibeis.viz.interact.__init__]', DEBUG=False)

from plottool import interact_helpers as ih

from ibeis.viz.interact import interact_annotations2
from ibeis.viz.interact import interact_bbox
from ibeis.viz.interact import interact_chip
from ibeis.viz.interact import interact_image
from ibeis.viz.interact import interact_matches
from ibeis.viz.interact import interact_name
from ibeis.viz.interact import interact_qres
from ibeis.viz.interact import interact_qres2
from ibeis.viz.interact import interact_sver

from ibeis.viz.interact.interact_image import ishow_image
from ibeis.viz.interact.interact_chip import ishow_chip
from ibeis.viz.interact.interact_name import ishow_name
from ibeis.viz.interact.interact_qres import ishow_qres
from ibeis.viz.interact.interact_matches import ishow_matches
from ibeis.viz.interact.interact_bbox import iselect_bbox
from ibeis.viz.interact.interact_sver import ishow_sver

import utool

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[ibeis.viz.interact]')


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys
    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import ibeis.viz.interact
    # Implicit reassignment.
    seen_ = set([])
    for submodname, fromimports in IMPORT_TUPLES:
        submod = getattr(ibeis.viz.interact, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(ibeis.viz.interact, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """ Reloads ibeis.viz.interact and submodules """
    rrr(verbose=verbose)
    getattr(interact_annotations2, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_bbox, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_chip, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_image, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_matches, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_name, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_qres, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_qres2, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(interact_sver, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('interact_annotations2', None, False),
    ('interact_bbox', None, False),
    ('interact_chip', None, False),
    ('interact_image', None, False),
    ('interact_matches', None, False),
    ('interact_name', None, False),
    ('interact_qres', None, False),
    ('interact_qres2', None, False),
    ('interact_sver', None, False),
]
"""
Regen Command:
    cd /home/joncrall/code/ibeis/ibeis/viz/interact
    makeinit.py
"""
## flake8: noqa
#from __future__ import absolute_import, division, print_function
#import utool

#from . import interact_image
#from . import interact_chip
#from . import interact_name
#from . import interact_qres
#from . import interact_bbox
#from . import interact_sver
#from . import interact_matches
#from . import interact_annotations2

#print, print_, printDBG, rrr, profile = utool.inject(
#    __name__, '[interact]')

#def reload_subs():
#    """ Reloads interact and submodules """
#    rrr()
#    getattr(interact_bbox, 'rrr', lambda: None)()
#    getattr(interact_chip, 'rrr', lambda: None)()
#    getattr(interact_image, 'rrr', lambda: None)()
#    getattr(interact_matches, 'rrr', lambda: None)()
#    getattr(interact_name, 'rrr', lambda: None)()
#    getattr(interact_qres, 'rrr', lambda: None)()
#    getattr(interact_sver, 'rrr', lambda: None)()
#    rrr()
#rrrr = reload_subs
