# flake8: noqa
from __future__ import absolute_import, division, print_function

from . import Config
from . import preproc
from . import hots
from . import detect

#import utool
#print, print_, printDBG, rrr, profile = utool.inject(__name__, '[model]')
def reload_subs():
    """Reloads model and submodules """
    #rrr()
    Config.rrr()
    hots.reload_subs()
    preproc.reload_subs()
    #rrr()
#rrrr = reload_subs
