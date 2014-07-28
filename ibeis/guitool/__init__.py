# flake8: noqa
from __future__ import absolute_import, division, print_function

__version__ = '1.0.0.dev1'

#from . import __PYQT__

from . import api_item_model
from . import api_table_view
from . import api_tree_view
from . import api_item_widget
from . import stripe_proxy_model

from . import guitool_tables
from . import guitool_dialogs
from . import guitool_decorators
from . import guitool_delegates
from . import guitool_components
from . import guitool_main
from . import guitool_misc
from . import qtype

from .guitool_tables import *
from .guitool_dialogs import *
from .guitool_decorators import *
from .guitool_delegates import *
from .guitool_components import *
from .guitool_main import *
from .guitool_misc import *
from .api_item_model import *
from .api_table_view import *
from .api_tree_view import *
from .api_item_widget import *
from .stripe_proxy_model import *
from .filter_proxy_model import *
from .qtype import *

import utool

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[guitool]')

def reload_subs():
    """Reloads utool and submodules """
    rrr()
    if hasattr(guitool_tables, 'rrr'):
        guitool_tables.rrr()
    if hasattr(guitool_dialogs, 'rrr'):
        guitool_dialogs.rrr()
    if hasattr(guitool_decorators, 'rrr'):
        guitool_decorators.rrr()
    if hasattr(guitool_main, 'rrr'):
        guitool_main.rrr()
    if hasattr(guitool_misc, 'rrr'):
        guitool_misc.rrr()
    if hasattr(api_item_model, 'rrr'):
        api_item_model.rrr()
    if hasattr(guitool_components, 'rrr'):
        guitool_components.rrr()

rrrr = reload_subs
