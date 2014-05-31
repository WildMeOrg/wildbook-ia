# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool

from . import interact_image
from . import interact_chip
from . import interact_name
from . import interact_qres
from . import interact_bbox
from . import interact_sver
from . import interact_matches

from plottool import interact_helpers as ih
from .interact_image import ishow_image
from .interact_chip import ishow_chip
from .interact_name import ishow_name
from .interact_qres import ishow_qres
from .interact_matches import ishow_matches
from .interact_bbox import iselect_bbox
from .interact_sver import ishow_sver

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[interact]')

def reload_subs():
    """ Reloads interact and submodules """
    rrr()
    getattr(interact_bbox, 'rrr', lambda: None)()
    getattr(interact_chip, 'rrr', lambda: None)()
    getattr(interact_image, 'rrr', lambda: None)()
    getattr(interact_matches, 'rrr', lambda: None)()
    getattr(interact_name, 'rrr', lambda: None)()
    getattr(interact_qres, 'rrr', lambda: None)()
    getattr(interact_sver, 'rrr', lambda: None)()
    rrr()
rrrr = reload_subs
