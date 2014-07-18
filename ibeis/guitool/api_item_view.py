from __future__ import absolute_import, division, print_function
from guitool import qtype
from guitool.api_thumb_delegate import APIThumbDelegate
from guitool.api_button_delegate import APIButtonDelegate
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
#import guitool
#from guitool.guitool_decorators import signal_, slot_
from guitool.guitool_main import get_qtapp
from guitool.guitool_misc import get_view_selection_as_str
import utool
from functools import partial

# Valid API Models
from guitool.stripe_proxy_model import StripeProxyModel
from guitool.filter_proxy_model import FilterProxyModel
from guitool.api_item_model import APIItemModel

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APIItemView]', DEBUG=False)

VERBOSE = utool.VERBOSE

API_VIEW_BASE = QtGui.QAbstractItemView
viewmember = utool.classmember(API_VIEW_BASE)
injectviewinstance = partial(utool.inject_instance, API_VIEW_BASE)


VALID_API_MODELS = (FilterProxyModel, StripeProxyModel, APIItemModel)


class APIItemView(API_VIEW_BASE):
    """
    Trees and Tables implicitly inherit from this class.
    Abstractish class.
    """

    def __init__(view, parent=None):
        API_VIEW_BASE.__init__(view, parent)

#---------------
# Data Manipulation
#---------------


@viewmember
def infer_delegates(view, **headers):
    """ Infers which columns should be given item delegates """
    col_type_list = headers.get('col_type_list', [])
    num_cols = view.model().columnCount()
    num_duplicates = int(num_cols / len(col_type_list))
    col_type_list = col_type_list * num_duplicates
    for colx, coltype in enumerate(col_type_list):
        if coltype in  qtype.QT_PIXMAP_TYPES:
            if utool.VERBOSE:
                print('[view] colx=%r is a PIXMAP' % colx)
            view.setItemDelegateForColumn(colx, APIThumbDelegate(view))
        elif coltype in qtype.QT_BUTTON_TYPES:
            if utool.VERBOSE:
                print('[view] colx=%r is a BUTTON' % colx)
            view.setItemDelegateForColumn(colx, APIButtonDelegate(view))


@viewmember
def set_column_persistant_editor(view, column):
    """ Set each row in a column as persistant """
    num_rows = view.model.rowCount()
    print('view.set_persistant: %r rows' % num_rows)
    for row in xrange(num_rows):
        index  = view.model.index(row, column)
        view.view.openPersistentEditor(index)


@viewmember
def _update_headers(view, **headers):
    """ Mirrors _update_headers in api_item_model """
    # Use headers from model #model = view.model #headers = model.headers
    # Get header info
    col_sort_index = headers.get('col_sort_index', None)
    col_sort_reverse = headers.get('col_sort_reverse', False)
    view.col_hidden_list = headers.get('col_hidden_list', [])
    view.col_name_list = headers.get('col_name_list', [])
    # Call updates
    view._set_sort(col_sort_index, col_sort_reverse)
    view.infer_delegates(**headers)
    #view.infer_delegates_from_model(model=model) #view.resizeColumnsToContents()


@viewmember
def _set_sort(view, col_sort_index, col_sort_reverse=False):
    if col_sort_index is not None:
        order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
        view.sortByColumn(col_sort_index, order)


@viewmember
def hide_cols(view):
    total_num_cols = view.model().columnCount()
    num_cols = len(view.col_hidden_list)
    num_duplicates = int(total_num_cols / num_cols)
    duplicated_hidden_list = view.col_hidden_list * num_duplicates
    for col, hidden in enumerate(duplicated_hidden_list):
        view.setColumnHidden(col, hidden)


#---------------
# Qt Overrides
#---------------


@viewmember
def itemDelegate(view, qindex):
    """ QtOverride: Returns item delegate for this index """
    # Does this even work? TODO: testme
    return API_VIEW_BASE.itemDelegate(view, qindex)


def setModel(view, model):
    """ QtOverride: Returns item delegate for this index """
    assert isinstance(model, VALID_API_MODELS),\
            ('APIItemViews only accepts APIItemModels (or one of its proxys),'
             'received a %r' % type(model))
    # Learn some things about the model before you fully connect it.
    if utool.VERBOSE:
        print('[view] setting model')
    model._rows_updated.connect(view.on_rows_updated)
    #view.infer_delegates_from_model(model=model)
    # TODO: Update headers
    return view.API_VIEW_BASE.setModel(view, model)


#---------------
# Slots
#---------------


@viewmember
def copy_selection_to_clipboard(view):
    """ Copys selected grid to clipboard """
    if VERBOSE:
        print('[guitool] Copying selection to clipboard')
    copy_str = get_view_selection_as_str(view)
    copy_qstr = QtCore.QString(copy_str)
    clipboard = get_qtapp().clipboard()
    if VERBOSE:
        print(copy_str)
    clipboard.setText(copy_qstr)
    if VERBOSE:
        print('[guitool] finished copy')
