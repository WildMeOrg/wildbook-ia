"""
  This module contains functions and clases to get data visualized fast (in
terms of development time)
"""
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtGui, QtCore
from guitool.api_item_model import APIItemModel
from guitool.api_table_view import APITableView
#from guitool import guitool_components as comp
from functools import partial
from six.moves import range
import utool as ut
import six
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[APIItemWidget]', DEBUG=False)


WIDGET_BASE = QtGui.QWidget

VERBOSE_ITEM_WIDGET = ut.get_argflag(('--verbose-item-widget', '--verbiw')) or ut.VERBOSE


def simple_api_item_widget():
    r"""
    Very simple example of basic APIItemWidget widget with CustomAPI

    CommandLine:
        python -m guitool.api_item_widget --test-simple_api_item_widget
        python -m guitool.api_item_widget --test-simple_api_item_widget --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.api_item_widget import *  # NOQA
        >>> import guitool
        >>> guitool.ensure_qapp()  # must be ensured before any embeding
        >>> wgt = simple_api_item_widget()
        >>> ut.quit_if_noshow()
        >>> wgt.show()
        >>> guitool.qtapp_loop(wgt, frequency=100)
    """
    import guitool
    guitool.ensure_qapp()
    col_name_list = ['col1', 'col2']
    col_types_dict = {}
    col_getter_dict = {
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
    }
    col_ider_dict = {}
    col_setter_dict = {}
    editable_colnames = []
    sortby = 'col1'
    get_thumb_size = lambda: 128
    col_width_dict = {}
    col_bgrole_dict = {}

    api = guitool.CustomAPI(
        col_name_list, col_types_dict, col_getter_dict,
        col_bgrole_dict, col_ider_dict, col_setter_dict,
        editable_colnames, sortby, get_thumb_size, True, col_width_dict)
    headers = api.make_headers(tblnice='Simple Example')

    wgt = guitool.APIItemWidget()
    wgt.change_headers(headers)
    #guitool.qtapp_loop(qwin=wgt, ipy=ipy, frequency=loop_freq)
    return wgt


class CustomAPI(object):
    """
    Allows list of lists to be represented as an abstract api table

    # TODO: Rename CustomAPI
    API wrapper around a list of lists, each containing column data
    Defines a single table
    """
    def __init__(self, col_name_list, col_types_dict, col_getter_dict,
                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                 editable_colnames, sortby, get_thumb_size=None,
                 sort_reverse=True, col_width_dict={}, strict=False, **kwargs):
        if VERBOSE_ITEM_WIDGET:
            print('[CustomAPI] <__init__>')
        self.col_width_dict = col_width_dict
        self.col_name_list = []
        self.col_type_list = []
        self.col_getter_list = []
        self.col_setter_list = []
        self.nCols = 0
        self.nRows = 0
        if get_thumb_size is None:
            self.get_thumb_size = lambda: 128
        else:
            self.get_thumb_size = get_thumb_size

        # Hack, maintain the original data
        # FIXME: make more ellegant
        self.orig_data_tup = (col_types_dict, col_getter_dict,
                              col_bgrole_dict, col_ider_dict, col_setter_dict,
                              editable_colnames, sortby, sort_reverse, strict)
        self.orig_kwargs = kwargs
        self.update_column_names(col_name_list)

        #self.parse_column_tuples(col_name_list, col_types_dict, col_getter_dict,
        #                         col_bgrole_dict, col_ider_dict, col_setter_dict,
        #                         editable_colnames, sortby, sort_reverse, strict, **kwargs)
        if VERBOSE_ITEM_WIDGET:
            print('[CustomAPI] </__init__>')

    def update_column_names(self, col_name_list):
        self.parse_column_tuples(col_name_list, *self.orig_data_tup, **self.orig_kwargs)

    def add_column_names(self, new_colnames):
        col_name_list = ut.unique_keep_order2(self.col_name_list + new_colnames)
        self.update_column_names(col_name_list)

    def get_available_colnames(self):
        col_getter_dict = self.orig_data_tup[1]
        return list(col_getter_dict.keys())

    def parse_column_tuples(self,
                            col_name_list,
                            col_types_dict,
                            col_getter_dict,
                            col_bgrole_dict,
                            col_ider_dict,
                            col_setter_dict,
                            editable_colnames,
                            sortby,
                            sort_reverse=True,
                            strict=False,
                            **kwargs):
        """
        parses simple lists into information suitable for making guitool headers
        """
        # Unpack the column tuples into names, getters, and types
        if not strict:
            # slopply colname definitions
            flag_list = [colname in col_getter_dict for colname in col_name_list]
            if not all(flag_list):
                invalid_colnames = ut.list_compress(col_name_list, ut.not_list(flag_list))
                print('[api_item_widget] Warning: colnames=%r have no getters' % (invalid_colnames,))
                col_name_list = ut.list_compress(col_name_list, flag_list)
            # sloppy type inference
            for colname in col_name_list:
                getter_ = col_getter_dict[colname]
                if colname not in col_types_dict:
                    type_ = ut.get_homogenous_list_type(getter_)
                    if type_ is not None:
                        col_types_dict[colname] = type_
        # sloppy kwargs.
        # FIXME: explicitly list col_nice_dict
        col_nice_dict = kwargs.get('col_nice_dict', {})
        self.col_nice_list = [col_nice_dict.get(name, name) for name in col_name_list]

        self.col_name_list = col_name_list
        self.col_type_list = [col_types_dict.get(colname, str) for colname in col_name_list]
        self.col_getter_list = [col_getter_dict.get(colname, str) for colname in col_name_list]  # First col is always a getter
        # Get number of rows / columns
        self.nCols = len(self.col_getter_list)
        self.nRows = 0 if self.nCols == 0 else len(self.col_getter_list[0])  # FIXME
        # Init iders to default and then overwite based on dict inputs
        self.col_ider_list = ut.alloc_nones(self.nCols)
        for colname, ider_colnames in six.iteritems(col_ider_dict):
            try:
                col = self.col_name_list.index(colname)
                # Col iders might have tuple input
                ider_cols = ut.uinput_1to1(self.col_name_list.index, ider_colnames)
                col_ider  = ut.uinput_1to1(lambda c: partial(self.get, c), ider_cols)
                self.col_ider_list[col] = col_ider
                del col_ider
                del ider_cols
                del col
                del colname
            except Exception as ex:
                ut.printex(ex, keys=['colname', 'ider_colnames', 'col', 'col_ider', 'ider_cols'])
                raise
        # Init setters to data, and then overwrite based on dict inputs
        self.col_setter_list = list(self.col_getter_list)
        for colname, col_setter in six.iteritems(col_setter_dict):
            col = self.col_name_list.index(colname)
            self.col_setter_list[col] = col_setter
        # Init bgrole_getters to None, and then overwrite based on dict inputs
        self.col_bgrole_getter_list = [col_bgrole_dict.get(colname, None) for colname in self.col_name_list]
        # Mark edtiable columns
        self.col_edit_list = [name in editable_colnames for name in col_name_list]
        # Mark the sort column index
        if ut.is_str(sortby):
            self.col_sort_index = self.col_name_list.index(sortby)
        else:
            self.col_sort_index = sortby
        self.col_sort_reverse = sort_reverse

    def _infer_index(self, column, row):
        """
        returns the row based on the columns iders.
        This is the identity for the default ider
        """
        ider_ = self.col_ider_list[column]
        if ider_ is None:
            return row
        iderfunc = lambda func_: func_(row)
        return ut.uinput_1to1(iderfunc, ider_)

    def get(self, column, row, **kwargs):
        """
        getters always receive primary rowids, rectify if col_ider is
        specified (row might be a row_pair)
        """
        index = self._infer_index(column, row)
        column_getter = self.col_getter_list[column]
        # Columns might be getter funcs indexable read/write arrays
        try:
            return ut.general_get(column_getter, index, **kwargs)
        except Exception:
            # FIXME: There may be an issue on tuple-key getters when row input is
            # vectorized. Hack it away
            if ut.isiterable(row):
                row_list = row
                return [self.get(column, row_, **kwargs) for row_ in row_list]
            else:
                raise

    def set(self, column, row, val):
        index = self._infer_index(column, row)
        column_setter = self.col_setter_list[column]
        # Columns might be setter funcs or indexable read/write arrays
        ut.general_set(column_setter, index, val)

    def get_bgrole(self, column, row):
        bgrole_getter = self.col_bgrole_getter_list[column]
        if bgrole_getter is None:
            return None
        index = self._infer_index(column, row)
        return ut.general_get(bgrole_getter, index)

    def ider(self):
        return list(range(self.nRows))

    def make_headers(self, tblname='custom_api', tblnice='Custom API'):
        """
        Builds headers for APIItemModel
        """
        headers = {
            'name': tblname,
            'nice': tblname if tblnice is None else tblnice,
            'iders': [self.ider],
            'col_name_list'    : self.col_name_list,
            'col_type_list'    : self.col_type_list,
            'col_nice_list'    : self.col_nice_list,
            'col_edit_list'    : self.col_edit_list,
            'col_sort_index'   : self.col_sort_index,
            'col_sort_reverse' : self.col_sort_reverse,
            'col_getter_list'  : self._make_getter_list(),
            'col_setter_list'  : self._make_setter_list(),
            'col_setter_list'  : self._make_setter_list(),
            'col_bgrole_getter_list' : self._make_bgrole_getter_list(),
            'get_thumb_size'   : self.get_thumb_size,
        }
        return headers

    def _make_bgrole_getter_list(self):
        return [partial(self.get_bgrole, column) for column in range(self.nCols)]

    def _make_getter_list(self):
        return [partial(self.get, column) for column in range(self.nCols)]

    def _make_setter_list(self):
        return [partial(self.set, column) for column in range(self.nCols)]


class APIItemWidget(WIDGET_BASE):
    """ SIMPLE WIDGET WHICH AUTO-CREATES MODEL AND VIEW FOR YOU.
    TODO: Deprecate this except for as an example.
    """

    def __init__(widget, headers=None, parent=None,
                 model_class=APIItemModel,
                 view_class=APITableView,
                 tblnice='APIItemWidget'):
        WIDGET_BASE.__init__(widget, parent)
        # Create vertical layout for the table to go into
        widget.vert_layout = QtGui.QVBoxLayout(widget)
        # Create a ColumnListTableView for the AbstractItemModel
        widget.view = view_class(parent=widget)
        # Instantiate the AbstractItemModel
        # FIXME: It is very bad to give the model a view.
        # Only the view should have a model
        widget.model = model_class(parent=widget.view)
        widget.view.setModel(widget.model)
        widget.vert_layout.addWidget(widget.view)
        widget.tblnice = tblnice
        if headers is not None:
            # Make sure we don't call a subclass method
            APIItemWidget.change_headers(widget, headers)
        widget.connect_signals()
        widget.api = None

    def connect_api(widget, api, autopopulate=True):
        widget.api = api
        if autopopulate:
            widget.refresh_headers()
            #headers = api.make_headers(tblnice=widget.tblnice)
            #widget.change_headers(headers)
            #print(ut.dict_str(headers))

    def change_headers(widget, headers):
        parent = widget.parent()
        # Update headers of both model and view
        widget.model._update_headers(**headers)
        widget.view._update_headers(**headers)
        if parent is None:
            nice = headers.get('nice', 'NO NICE NAME')
            widget.setWindowTitle(nice)

    def connect_signals(widget):
        widget.model._rows_updated.connect(widget.on_rows_updated)
        widget.view.contextMenuClicked.connect(widget.on_contextMenuRequested)

    def on_rows_updated(widget, name, num):
        if VERBOSE_ITEM_WIDGET:
            print('rows updated')
        pass

    def refresh_headers(widget):
        headers = widget.api.make_headers(tblnice=widget.tblnice)
        widget.change_headers(headers)
        #print(ut.dict_str(headers))

    @QtCore.pyqtSlot(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(widget, index, pos):
        print('context request')
        if widget.api is not None:
            print(ut.list_str(widget.api.get_available_colnames()))
            # HACK test
            #widget.api.add_column_names(['qx2_gt_rank', 'qx2_gf_rank', 'qx2_gt_raw_score', 'qx2_gf_raw_score'])
            widget.refresh_headers()
            #widget.change_headers(widget.api.make_headers())
        if VERBOSE_ITEM_WIDGET:
            print('context request')
        pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m guitool.api_item_widget
        python -m guitool.api_item_widget --allexamples
        python -m guitool.api_item_widget --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
