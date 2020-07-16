# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import GUITOOL_PYQT_VERSION
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool.guitool_decorators import signal_, slot_
import utool as ut
from wbia.guitool import api_item_view

(print, rrr, profile) = ut.inject2(__name__)


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch

API_VIEW_BASE = QtWidgets.QTreeView
# API_VIEW_BASE = QtWidgets.QAbstractItemView


class APITreeView(API_VIEW_BASE):
    """
    Tree view of API data.
    Implicitly inherits from APIItemView
    """

    rows_updated = signal_(str, int)
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)
    API_VIEW_BASE = API_VIEW_BASE

    def __init__(view, parent=None):
        # Qt Inheritance
        API_VIEW_BASE.__init__(view, parent)
        # Implicitly inject common APIItemView functions
        api_item_view.injectviewinstance(view)
        view._init_itemview_behavior()
        view._init_tree_behavior()
        view.col_hidden_list = []
        # # view._init_header_behavior()
        # Context menu
        view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(view.on_customMenuRequested)
        # view.cornerButton = guitool_components.newButton(view)
        # view.setCornerWidget(view.cornerButton)
        view._init_api_item_view()

        # view.setUniformRowHeights(True)

    # ---------------
    # Initialization
    # ---------------

    def _init_tree_behavior(view):
        """ Tree behavior

        SeeAlso:
            api_item_view._init_itemview_behavior
        """
        pass

    def _init_header_behavior(view):
        """ Header behavior

        CommandLine:
            python -m wbia.guitool.api_tree_view --test-_init_header_behavior

        Example:
            >>> # ENABLE_DOCTEST
            >>> # xdoctest: +REQUIRES(--gui)
            >>> # TODO figure out how to test these
            >>> from wbia.guitool.api_tree_view import *  # NOQA
            >>> import wbia.guitool as gt
            >>> app = gt.ensure_qapp()
            >>> view = APITreeView()
            >>> view._init_header_behavior()
        """
        # Row Headers
        # Column headers
        header = view.header()
        header.setVisible(True)
        header.setStretchLastSection(True)
        header.setSortIndicatorShown(True)
        header.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE RESIZETOCONTENTS. IT MAKES THINGS VERY SLOW
        # horizontalHeader.setResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # horizontalHeader.setResizeMode(QtWidgets.QHeaderView.Stretch)
        if GUITOOL_PYQT_VERSION == 4:
            header.setResizeMode(QtWidgets.QHeaderView.Interactive)
            # horizontalHeader.setCascadingSectionResizes(True)
            # Columns moveable
            header.setMovable(True)
        else:
            header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            header.setSectionsMovable(True)

    # ---------------
    # Qt Overrides
    # ---------------

    def setModel(view, model):
        """ QtOverride: Returns item delegate for this index """
        api_item_view.setModel(view, model)

    def keyPressEvent(view, event):
        return api_item_view.keyPressEvent(view, event)

    # ---------------
    # Slots
    # ---------------

    @slot_(str, int)
    def on_rows_updated(view, tblname, num):
        # re-emit the model signal
        view.rows_updated.emit(tblname, num)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(view, pos):
        index = view.indexAt(pos)
        view.contextMenuClicked.emit(index, pos)


def testdata_tree_view():
    r"""
    CommandLine:
        python -m wbia.guitool.api_tree_view testdata_tree_view
        python -m wbia.guitool.api_tree_view testdata_tree_view --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> import wbia.guitool as gt
        >>> from wbia.guitool.api_tree_view import *  # NOQA
        >>> wgt = testdata_tree_view()
        >>> view = wgt.view
        >>> rows = view.selectedRows()
        >>> print('rows = %r' % (rows,))
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=wgt)
    """
    import wbia.guitool as gt

    gt.ensure_qapp()
    col_name_list = ['name', 'num_annots', 'annots']
    col_getter_dict = {
        'name': ['fred', 'sue', 'tom', 'mary', 'paul'],
        'num_annots': [2, 1, 3, 5, 1],
    }
    # make consistent data
    grouped_data = [
        [col_getter_dict['name'][index] + '-' + str(i) for i in range(num)]
        for index, num in enumerate(col_getter_dict['num_annots'])
    ]
    flat_data, reverse_list = ut.invertible_flatten1(grouped_data)
    col_getter_dict['annots'] = flat_data

    iders = [list(range(len(col_getter_dict['name']))), reverse_list]

    col_level_dict = {
        'name': 0,
        'num_annots': 0,
        'annots': 1,
    }
    sortby = 'name'

    api = gt.CustomAPI(
        col_name_list=col_name_list,
        col_getter_dict=col_getter_dict,
        sortby=sortby,
        iders=iders,
        col_level_dict=col_level_dict,
    )
    headers = api.make_headers(tblnice='Tree Example')

    wgt = gt.APIItemWidget(view_class=APITreeView)
    wgt.change_headers(headers)

    wgt.menubar = gt.newMenubar(wgt)
    wgt.menuFile = wgt.menubar.newMenu('Dev')

    def wgt_embed(wgt):
        view = wgt.view  # NOQA
        import utool

        utool.embed()

    ut.inject_func_as_method(wgt, wgt_embed)
    wgt.menuFile.newAction(triggered=wgt.wgt_embed)

    return wgt


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.api_tree_view
        python -m wbia.guitool.api_tree_view --allexamples
        python -m wbia.guitool.api_tree_view --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
