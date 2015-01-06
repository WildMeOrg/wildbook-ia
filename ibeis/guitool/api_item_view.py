"""
provides common methods for api_tree_view and api_table_view
"""
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtGui
from guitool.__PYQT__.QtCore import Qt
from six.moves import range
from guitool import qtype
from guitool.api_thumb_delegate import APIThumbDelegate
from guitool.api_button_delegate import APIButtonDelegate
#import guitool
#from guitool.guitool_decorators import signal_, slot_
from guitool.guitool_main import get_qtapp
from guitool.guitool_misc import get_view_selection_as_str
import utool
import utool as ut
from functools import partial

# Valid API Models
from guitool.stripe_proxy_model import StripeProxyModel
from guitool.filter_proxy_model import FilterProxyModel
from guitool.api_item_model import APIItemModel

# DANGER SHOULD UTOOL BE INJECTING HERE?!
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APIItemView]', DEBUG=False)

VERBOSE = utool.VERBOSE or ut.get_argflag(('--verbose-qt', '--verbqt'))

API_VIEW_BASE = QtGui.QAbstractItemView
register_view_method = utool.make_class_method_decorator(API_VIEW_BASE, __name__)

injectviewinstance = partial(utool.inject_instance, classtype=API_VIEW_BASE)


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


@register_view_method
def infer_delegates(view, **headers):
    """ Infers which columns should be given item delegates """
    get_thumb_size = headers.get('get_thumb_size', None)
    col_type_list = headers.get('col_type_list', [])
    num_cols = view.model().columnCount()
    num_duplicates = int(num_cols / len(col_type_list))
    col_type_list = col_type_list * num_duplicates
    for colx, coltype in enumerate(col_type_list):
        if coltype in  qtype.QT_PIXMAP_TYPES:
            if VERBOSE:
                print('[view] colx=%r is a PIXMAP' % colx)
            view.setItemDelegateForColumn(colx, APIThumbDelegate(view, get_thumb_size))
        elif coltype in qtype.QT_BUTTON_TYPES:
            if VERBOSE:
                print('[view] colx=%r is a BUTTON' % colx)
            view.setItemDelegateForColumn(colx, APIButtonDelegate(view))


@register_view_method
def set_column_persistant_editor(view, column):
    """ Set each row in a column as persistant """
    num_rows = view.model.rowCount()
    print('view.set_persistant: %r rows' % num_rows)
    for row in range(num_rows):
        index  = view.model.index(row, column)
        view.view.openPersistentEditor(index)


@register_view_method
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


@register_view_method
def _set_sort(view, col_sort_index, col_sort_reverse=False):
    if col_sort_index is not None:
        order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
        view.sortByColumn(col_sort_index, order)


@register_view_method
def hide_cols(view):
    total_num_cols = view.model().columnCount()
    num_cols = len(view.col_hidden_list)
    num_duplicates = int(total_num_cols / num_cols)
    duplicated_hidden_list = view.col_hidden_list * num_duplicates
    for col, hidden in enumerate(duplicated_hidden_list):
        view.setColumnHidden(col, hidden)


@register_view_method
def select_row_from_id(view, _id, scroll=False, level=0):
    """
    _id is from the iders function (i.e. an ibeis rowid)
    selects the row in that view if it exists


    Ignore:
        from ibeis.gui.guiheaders import (IMAGE_TABLE, IMAGE_GRID, ANNOTATION_TABLE,
                                          NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)
        ibsgwt = back.front

        view   = ibsgwt.views[IMAGE_TABLE]
        model  = ibsgwt.models[IMAGE_TABLE]
        row = model.get_row_from_id(3)
        view.selectRow(row)


        view   = ibsgwt.views[NAMES_TREE]
        model  = ibsgwt.models[NAMES_TREE]

    References:
        http://stackoverflow.com/questions/4967660/selecting-a-row-in-qtreeview-programatically
        http://stackoverflow.com/questions/17912812/how-do-i-select-a-row-in-a-qtreeview-programatically
        http://qt-project.org/doc/qt-4.8/qabstractitemview.html#setSelection
    """
    if isinstance(view, QtGui.QTreeView):
        # TODO find a way to get tree views to select by _id
        # TODO: use level as well
        return None
    else:
        model = view.model()
        row = model.get_row_from_id(_id)
        if row is not None:
            view.selectRow(row)
            if scroll:
                qtindex = model.index(row, 0)
                view.scrollTo(qtindex)
            return row

#---------------
# Qt Overrides
#---------------


#@register_view_method
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
    if VERBOSE:
        print('[view] setting model')
    model._rows_updated.connect(view.on_rows_updated)
    #view.infer_delegates_from_model(model=model)
    # TODO: Update headers
    return view.API_VIEW_BASE.setModel(view, model)


#---------------
# Slots
#---------------


@register_view_method
def copy_selection_to_clipboard(view):
    """ Copys selected grid to clipboard """
    if VERBOSE:
        print('[guitool] Copying selection to clipboard')
    copy_str = get_view_selection_as_str(view)
    #copy_qstr = QtCore.Q__String(copy_str)
    copy_qstr = str(copy_str)
    clipboard = get_qtapp().clipboard()
    if VERBOSE:
        print(copy_str)
    clipboard.setText(copy_qstr)
    if VERBOSE:
        print('[guitool] finished copy')
