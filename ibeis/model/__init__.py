# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool as ut
ut.noinject(__name__, '[ibeis.model.__init__]', DEBUG=False)

from ibeis.model import Config
from ibeis.model import detect
from ibeis.model import hots
from ibeis.model import preproc

print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[ibeis.model]')


def reload_subs(verbose=True):
    """ Reloads ibeis.model and submodules """
    rrr(verbose=verbose)
    getattr(Config, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(detect, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(hots, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(preproc, 'reload_subs', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('Config', None, False),
    ('detect', None, True),
    ('hots', None, True),
    ('preproc', None, True),
]
"""
Regen Command:
    cd /home/joncrall/code/ibeis/ibeis/model
    makeinit.py
"""
