# flake8: noqa
from __future__ import absolute_import, division, print_function
from . import preproc_chip
from . import preproc_detectimg
from . import preproc_encounter
from . import preproc_feat
from . import preproc_image

#import utool
#print, print_, printDBG, rrr, profile = utool.inject(__name__, '[preproc]')
def reload_subs():
    """Reloads preproc and submodules """
    #rrr()
    getattr(preproc_chip, 'rrr', lambda: None)()
    getattr(preproc_detectimg, 'rrr', lambda: None)()
    getattr(preproc_encounter, 'rrr', lambda: None)()
    getattr(preproc_feat, 'rrr', lambda: None)()
    getattr(preproc_image, 'rrr', lambda: None)()
    #rrr()
#rrrr = reload_subs
