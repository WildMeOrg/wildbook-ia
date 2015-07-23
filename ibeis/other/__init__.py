### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool as ut
ut.noinject(__name__, '[ibeis.init.__init__]', DEBUG=False)
from ibeis.other import dbinfo
from ibeis.other import duct_tape
from ibeis.other import optimize_k
#print, print_, printDBG, rrr, profile = ut.inject(
#    __name__, '[ibeis.init.')


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys
    if verbose and '--quiet' not in sys.argv:
        print('other reimport')
    # Self import
    import ibeis.other
    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(ibeis.init. submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(ibeis.init. attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """ Reloads ibeis.init.and submodules """
    #rrr(verbose=verbose)
    def fbrrr(*args, **kwargs):
        """ fallback reload """
        pass
    getattr(dbinfo, 'rrr', fbrrr)(verbose=verbose)
    getattr(duct_tape, 'rrr', fbrrr)(verbose=verbose)
    getattr(optimize_k, 'rrr', fbrrr)(verbose=verbose)
    #rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('dbinfo', None),
    ('duct_tape', None),
    ('optimize_k', None),
]
"""
Regen Command:
    cd /home/joncrall/code/ibeis/ibeis/other
    makeinit.py
"""

