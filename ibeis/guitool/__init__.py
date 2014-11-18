# flake8: noqa
from __future__ import absolute_import, division, print_function

__version__ = '1.0.0.dev1'

try:
    # try seeing if importing plottool before any guitool things helps
    import plottool
except Exception as ex:
    raise
    #pass

#print('__guitool__1')
from guitool import __PYQT__
#print('__guitool__2')

from guitool import api_item_model
from guitool import api_table_view
from guitool import api_tree_view
from guitool import api_item_widget
from guitool import stripe_proxy_model

from guitool import guitool_tables
from guitool import guitool_dialogs
from guitool import guitool_decorators
from guitool import guitool_delegates
from guitool import guitool_components
from guitool import guitool_main
from guitool import guitool_misc
from guitool import qtype

from guitool.guitool_tables import *
from guitool.guitool_dialogs import *
from guitool.guitool_decorators import *
from guitool.guitool_delegates import *
from guitool.guitool_components import *
from guitool.guitool_main import *
from guitool.guitool_misc import *
from guitool.api_item_model import *
from guitool.api_table_view import *
from guitool.api_tree_view import *
from guitool.api_item_widget import *
from guitool.stripe_proxy_model import *
from guitool.filter_proxy_model import *
from guitool.qtype import *

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
    if hasattr(qtype, 'rrr'):
        qtype.rrr()
    if hasattr(guitool_components, 'rrr'):
        guitool_components.rrr()

rrrr = reload_subs
#print('__guitool__3')
