"""
This module was never really finished. It is used in some cases
to display the results from a query in a qt window. It needs
some work if its to be re-integrated.

TODO:
    Refresh name table on inspect gui close
"""
from __future__ import absolute_import, division, print_function
from functools import partial
from guitool import qtype, APIItemWidget, APIItemModel, FilterProxyModel, ChangeLayoutContext
from guitool.__PYQT__ import QtGui, QtCore
from ibeis import ibsfuncs
from ibeis.dev import results_organizer
#from ibeis.viz import interact
from ibeis.viz import viz_helpers as vh
from plottool import fig_presenter
#from plottool import interact_helpers as ih
from six.moves import range
#import functools
import guitool
import numpy as np
import six
import utool
#from ibeis import constants as const
import utool as ut
from ibeis.gui import guiexcept
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[inspect_gui]')


MATCHED_STATUS_TEXT  = 'Matched'
REVIEWED_STATUS_TEXT = 'Reviewed'
USE_FILTER_PROXY = False


class CustomFilterModel(FilterProxyModel):
    def __init__(model, headers=None, parent=None, *args):
        FilterProxyModel.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.eid = -1  # negative one is an invalid eid
        model.original_ider = None
        model.sourcemodel = APIItemModel(parent=parent)
        model.setSourceModel(model.sourcemodel)
        print('[ibs_model] just set the sourcemodel')

    def _update_headers(model, **headers):
        def _null_ider(**kwargs):
            return []
        model.original_iders = headers.get('iders', [_null_ider])
        if len(model.original_iders) > 0:
            model.new_iders = model.original_iders[:]
            model.new_iders[0] = model._ider
        headers['iders'] = model.new_iders
        model.sourcemodel._update_headers(**headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_iders[0]()

    def _change_enc(model, eid):
        model.eid = eid
        with ChangeLayoutContext([model]):
            FilterProxyModel._update_rows(model)


class QueryResultsWidget(APIItemWidget):
    """ Window for gui inspection """

    def __init__(qres_wgt, ibs, qaid2_qres, parent=None, callback=None,
                 name_scoring=False, **kwargs):
        if ut.VERBOSE:
            print('[qres_wgt] Init QueryResultsWidget')
        # Uncomment below to turn on FilterProxyModel
        if USE_FILTER_PROXY:
            APIItemWidget.__init__(qres_wgt, parent=parent,
                                    model_class=CustomFilterModel)
        else:
            APIItemWidget.__init__(qres_wgt, parent=parent)
        qres_wgt.button_list = None
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = utool.tic()
        # Set results data
        if USE_FILTER_PROXY:
            qres_wgt.add_checkboxes(qres_wgt.show_new, qres_wgt.show_join, qres_wgt.show_split)
        qres_wgt.set_query_results(ibs, qaid2_qres, name_scoring=name_scoring, **kwargs)
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = lambda: None
        qres_wgt.callback = callback
        qres_wgt.view.setColumnHidden(0, False)
        qres_wgt.view.setColumnHidden(1, False)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

    def add_checkboxes(qres_wgt, show_new, show_join, show_split):
        _CHECK  = partial(guitool.newCheckBox, qres_wgt)
        qres_wgt.button_list = [
            [
                _CHECK('Show New Matches',
                        qres_wgt._check_changed,
                        checked=show_new),

                _CHECK('Show Join Matches',
                        qres_wgt._check_changed,
                        checked=show_join),

                _CHECK('Show Split Matches',
                        qres_wgt._check_changed,
                        checked=show_split),
            ]
        ]

        qres_wgt.buttonBars = []
        for row in qres_wgt.button_list:
            qres_wgt.buttonBars.append(QtGui.QHBoxLayout(qres_wgt))
            qres_wgt.vert_layout.addLayout(qres_wgt.buttonBars[-1])
            for button in row:
                qres_wgt.buttonBars[-1].addWidget(button)

    def update_checkboxes(qres_wgt):
        if qres_wgt.button_list is None:
            return
        show_new = qres_wgt.button_list[0][0].isChecked()
        show_join = qres_wgt.button_list[0][1].isChecked()
        show_split = qres_wgt.button_list[0][2].isChecked()
        if USE_FILTER_PROXY:
            qres_wgt.model.update_filterdict({
                'NEW Match ':   show_new,
                'JOIN Match ':  show_join,
                'SPLIT Match ': show_split,
            })
        qres_wgt.model._update_rows()

    def _check_changed(qres_wgt, value):
        qres_wgt.update_checkboxes()

    def sizeHint(qres_wgt):
        # should eventually improve this to use the widths of the header columns
        return QtCore.QSize(1000, 500)

    def set_query_results(qres_wgt, ibs, qaid2_qres, name_scoring=False, **kwargs):
        print('[qres_wgt] Change QueryResultsWidget data')
        qres_wgt.ibs = ibs
        qres_wgt.qaid2_qres = qaid2_qres
        qres_wgt.qres_api = make_qres_api(ibs, qaid2_qres, name_scoring=name_scoring, **kwargs)
        qres_wgt.update_checkboxes()
        headers = qres_wgt.qres_api.make_headers()
        # super call
        APIItemWidget.change_headers(qres_wgt, headers)

    def connect_signals_and_slots(qres_wgt):
        qres_wgt.view.clicked.connect(qres_wgt._on_click)
        qres_wgt.view.doubleClicked.connect(qres_wgt._on_doubleclick)
        qres_wgt.view.pressed.connect(qres_wgt._on_pressed)
        qres_wgt.view.activated.connect(qres_wgt._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(iqrw, qtindex):
        #print('[qres_wgt] _on_click: ')
        #print('[qres_wgt] _on_click: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)

        if colname == MATCHED_STATUS_TEXT:
            #qres_callback = partial(show_match_at, iqrw, qtindex)
            #review_match_at(iqrw, qtindex, qres_callback=qres_callback)
            review_match_at(iqrw, qtindex)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, qtindex):
        print('[qres_wgt] _on_doubleclick: ')
        print('[qres_wgt] DoubleClicked: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname != MATCHED_STATUS_TEXT:
            return show_match_at(iqrw, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(iqrw, qtindex):
        print('[qres_wgt] _on_pressed: ')
        def _check_for_double_click(iqrw, qtindex):
            threshold = 0.20  # seconds
            distance = utool.toc(iqrw.tt)
            #print('Pressed %r' % (distance,))
            col = qtindex.column()
            model = qtindex.model()
            colname = model.get_header_name(col)
            if distance <= threshold:
                if colname == MATCHED_STATUS_TEXT:
                    iqrw.view.clicked.emit(qtindex)
                    iqrw._on_click(qtindex)
                else:
                    #iqrw.view.doubleClicked.emit(qtindex)
                    iqrw._on_doubleclick(qtindex)
            iqrw.tt = utool.tic()
        _check_for_double_click(iqrw, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(iqrw, qtindex):
        print('Activated: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(iqrw, qtindex, qpos):
        printDBG('[newgui] contextmenu')
        guitool.popup_menu(iqrw, qpos, [
            ('Show feature matches', lambda: show_match_at(iqrw, qtindex)),
            ('Inspect Match Candidates', lambda: review_match_at(iqrw, qtindex)),
            ('Mark as &Reviewed', lambda: mark_pair_as_reviewed(iqrw, qtindex)),
            ('Mark as &True Match.', lambda: mark_pair_as_positive_match(iqrw, qtindex)),
            ('Mark as &False Match.', lambda: mark_pair_as_negative_match(iqrw, qtindex)),
        ])


def mark_pair_as_reviewed(qres_wgt, qtindex):
    """
    Sets the reviewed flag to whatever the current truth status is
    """
    model = qtindex.model()
    qaid  = model.get_header_data('qaid', qtindex)
    daid  = model.get_header_data('aid', qtindex)
    ibs = qres_wgt.ibs
    ibs.mark_annot_pair_as_reviewed(qaid, daid)


def mark_pair_as_positive_match(qres_wgt, qtindex):
    model = qtindex.model()
    qaid  = model.get_header_data('qaid', qtindex)
    daid  = model.get_header_data('aid', qtindex)
    ibs = qres_wgt.ibs
    try:
        status = mark_annot_pair_as_positive_match(ibs, qaid, daid)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match_at(qres_wgt, qtindex)
    except guiexcept.UserCancel:
        print('user canceled positive match')


def mark_pair_as_negative_match(qres_wgt, qtindex):
    model = qtindex.model()
    qaid  = model.get_header_data('qaid', qtindex)
    daid  = model.get_header_data('aid', qtindex)
    ibs = qres_wgt.ibs
    try:
        status = mark_annot_pair_as_negative_match(ibs, qaid, daid)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match_at(qres_wgt, qtindex)
    except guiexcept.UserCancel:
        print('user canceled negative match')


def mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun=False):
    """
    TODO: ELEVATE THIS FUNCTION
    Change into make_task_mark_annot_pair_as_positive_match and it returns what
    needs to be done.

    Need to test several cases:
        uknown, unknown
        knownA, knownA
        knownB, knownA
        unknown, knownA
        knownA, unknown

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  query annotation id
        aid2 (int):  matching annotation id

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_positive_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> status = mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(status)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        if not ut.QUIET:
            print('... _set_annot_name_rowids(aids=%r, nids=%r)' % (aid_list, nid_list))
            print('... names = %r' % (ibs.get_name_texts(nid_list)))
        assert len(aid_list) == len(nid_list), 'list must correspond'
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_MATCH, [1.0])
        # Return the new annots in this name
        _aids_list = ibs.get_name_aids(nid_list)
        _combo_aids_list = [_aids + [aid] for _aids, aid, in zip(_aids_list, aid_list)]
        status = _combo_aids_list
        return status
    print('[marking_match] aid1 = %r, aid2 = %r' % (aid1, aid2))

    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('...images already matched')
        status = None
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...match unknown1 to unknown2 into 1 new name')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids * 2)
        elif not isunknown1 and not isunknown2:
            print('...merge known1 into known2')
            MERGE_NEEDS_INTERACTION  = False
            MERGE_NEEDS_VERIFICATION = True
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            if MERGE_NEEDS_INTERACTION:
                raise guiexcept.NeedsUserInput('confirm merge')
            elif MERGE_NEEDS_VERIFICATION:
                aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
                name1, name2 = ibs.get_annot_names([aid1, aid2])
                msgfmt = ut.codeblock('''
                   Confirm merge of animal {name1} and {name2}
                   {name1} has {num_gt1} annotations
                   {name2} has {num_gt2} annotations
                   ''')
                msg = msgfmt.format(name1=name1, name2=name2,
                                    num_gt1=len(aid1_and_groundtruth),
                                    num_gt2=len(aid2_and_groundtruth),)
                if not guitool.are_you_sure(parent=None, msg=msg, default='Yes'):
                    raise guiexcept.UserCancel('canceled merge')
            status =  _set_annot_name_rowids(aid1_and_groundtruth, [nid2] * len(aid1_and_groundtruth))
        elif isunknown2 and not isunknown1:
            print('...match unknown2 into known1')
            status =  _set_annot_name_rowids([aid2], [nid1])
        elif isunknown1 and not isunknown2:
            print('...match unknown1 into known2')
            status =  _set_annot_name_rowids([aid1], [nid2])
        else:
            raise AssertionError('impossible state')
    return status


def mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun=False):
    """
    TODO: ELEVATE THIS FUNCTION

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        dryrun (bool):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_negative_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> # execute function
        >>> result = mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(result)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        print('... _set_annot_name_rowids(%r, %r)' % (aid_list, nid_list))
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_NOT_MATCH, [1.0])
    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('images are marked as having the same name... we must tread carefully')
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        if len(aid1_groundtruth) == 1 and aid1_groundtruth == [aid2]:
            # this is the only safe case for same name split
            # Change so the names are not the same
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            status = 'error'
            print('There are %d annots in this name. Need more sophisticated split' % (len(aid1_groundtruth)))
            raise guiexcept.NeedsUserInput('non-trivial split')
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...nonmatch unknown1 and unknown2 into 2 new names')
            next_nids = ibs.make_next_nids(num=2)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids)
        elif not isunknown1 and not isunknown2:
            print('...nonmatch known1 and known2... nothing to do (yet)')
            status = None
        elif isunknown2 and not isunknown1:
            print('...nonmatch unknown2 -> newname and known1')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid2], next_nids)
        elif isunknown1 and not isunknown2:
            print('...nonmatch unknown1 -> newname and known2')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            raise AssertionError('impossible state')
    return status


def show_match_at(qres_wgt, qtindex):
    print('interact')
    model = qtindex.model()
    aid  = model.get_header_data('aid', qtindex)
    qaid = model.get_header_data('qaid', qtindex)
    #fig = interact.ishow_matches(qres_wgt.ibs, qres_wgt.qaid2_qres[qaid], aid, mode=1)
    match_interaction = qres_wgt.qaid2_qres[qaid].ishow_matches(qres_wgt.ibs, aid, mode=1)
    fig = match_interaction.fig
    fig_presenter.bring_to_front(fig)


def review_match_at(qres_wgt, qtindex, **kwargs):
    print('review')
    ibs = qres_wgt.ibs
    model = qtindex.model()
    aid1 = model.get_header_data('qaid', qtindex)
    aid2 = model.get_header_data('aid', qtindex)
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    model = qtindex.model()
    #update_callback = model._update
    update_callback = None  # hack (checking if necessary)
    backend_callback = qres_wgt.callback
    #qres_callback = partial(show_match_at, qres_wgt, qtindex)
    qres = qres_wgt.qaid2_qres[aid1]
    review_match(ibs, aid1, aid2, update_callback=update_callback,
                 backend_callback=backend_callback,
                 qres=qres,
                 #qres_callback=qres_callback,
                 **kwargs)


def review_match(ibs, aid1, aid2, update_callback=None, backend_callback=None, **kwargs):
    print('Review match: ' + ibsfuncs.vsstr(aid1, aid2))
    from ibeis.viz.interact import interact_name
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    mvinteract = interact_name.MatchVerificationInteraction(
        ibs, aid1, aid2, fnum=64, update_callback=update_callback,
        backend_callback=backend_callback, **kwargs)
    return mvinteract
    #ih.register_interaction(mvinteract)


class CustomAPI(object):
    """
    Allows list of lists to be represented as an abstract api table

    # TODO: Rename CustomAPI
    API wrapper around a list of lists, each containing column data
    Defines a single table
    """
    def __init__(self, col_name_list, col_types_dict, col_getters_dict,
                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                 editable_colnames, sortby, get_thumb_size=None,
                 sort_reverse=True):
        if ut.VERBOSE:
            print('[CustomAPI] <__init__>')
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
        self.parse_column_tuples(col_name_list, col_types_dict, col_getters_dict,
                                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                                 editable_colnames, sortby, sort_reverse)
        if ut.VERBOSE:
            print('[CustomAPI] </__init__>')

    def parse_column_tuples(self,
                            col_name_list,
                            col_types_dict,
                            col_getters_dict,
                            col_bgrole_dict,
                            col_ider_dict,
                            col_setter_dict,
                            editable_colnames,
                            sortby,
                            sort_reverse=True):
        """
        parses simple lists into information suitable for making guitool headers
        """
        # Unpack the column tuples into names, getters, and types
        self.col_name_list = col_name_list
        self.col_type_list = [col_types_dict.get(colname, str) for colname in col_name_list]
        self.col_getter_list = [col_getters_dict.get(colname, str) for colname in col_name_list]  # First col is always a getter
        # Get number of rows / columns
        self.nCols = len(self.col_getter_list)
        self.nRows = 0 if self.nCols == 0 else len(self.col_getter_list[0])  # FIXME
        # Init iders to default and then overwite based on dict inputs
        self.col_ider_list = utool.alloc_nones(self.nCols)
        for colname, ider_colnames in six.iteritems(col_ider_dict):
            col = self.col_name_list.index(colname)
            # Col iders might have tuple input
            ider_cols = utool.uinput_1to1(self.col_name_list.index, ider_colnames)
            col_ider  = utool.uinput_1to1(lambda c: partial(self.get, c), ider_cols)
            self.col_ider_list[col] = col_ider
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
        if utool.is_str(sortby):
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
        return utool.uinput_1to1(iderfunc, ider_)

    def get(self, column, row, **kwargs):
        """
        getters always receive primary rowids, rectify if col_ider is
        specified (row might be a row_pair)
        """
        index = self._infer_index(column, row)
        column_getter = self.col_getter_list[column]
        # Columns might be getter funcs indexable read/write arrays
        try:
            return utool.general_get(column_getter, index, **kwargs)
        except Exception:
            # FIXME: There may be an issue on tuple-key getters when row input is
            # vectorized. Hack it away
            if utool.isiterable(row):
                row_list = row
                return [self.get(column, row_, **kwargs) for row_ in row_list]
            else:
                raise

    def set(self, column, row, val):
        index = self._infer_index(column, row)
        column_setter = self.col_setter_list[column]
        # Columns might be setter funcs or indexable read/write arrays
        utool.general_set(column_setter, index, val)

    def get_bgrole(self, column, row):
        bgrole_getter = self.col_bgrole_getter_list[column]
        if bgrole_getter is None:
            return None
        index = self._infer_index(column, row)
        return utool.general_get(bgrole_getter, index)

    def ider(self):
        return list(range(self.nRows))

    def make_headers(self, tblname='qres_api', tblnice='Query Results'):
        """
        Builds headers for APIItemModel
        """
        headers = {
            'name': tblname,
            'nice': tblname if tblnice is None else tblnice,
            'iders': [self.ider],
            'col_name_list'    : self.col_name_list,
            'col_type_list'    : self.col_type_list,
            'col_nice_list'    : self.col_name_list,
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


def get_match_status(ibs, aid_pair):
    """ Data role for status column
    FIXME: no other function in this project takes a tuple of scalars as an
    argument. Everything else is written in the context of lists, This function
    should follow the same paradigm, but CustomAPI will have to change.
    """
    aid1, aid2 = aid_pair
    assert not utool.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not utool.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #text  = ibsfuncs.vsstr(aid1, aid2)
    text = ibs.get_match_text(aid1, aid2)
    if text is None:
        raise AssertionError('impossible state inspect_gui')
    return text


def get_reviewed_status(ibs, aid_pair):
    """ Data role for status column
    FIXME: no other function in this project takes a tuple of scalars as an
    argument. Everything else is written in the context of lists, This function
    should follow the same paradigm, but CustomAPI will have to change.
    """
    aid1, aid2 = aid_pair
    assert not utool.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not utool.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #text  = ibsfuncs.vsstr(aid1, aid2)
    annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    return 'Yes' if annotmach_reviewed else 'No'
    #text = ibs.get_match_text(aid1, aid2)
    #if text is None:
    #    raise AssertionError('impossible state inspect_gui')
    #return 'No'


def get_match_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def get_reviewed_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    #truth = ibs.get_annot_pair_truth([aid1], [aid2])[0]
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    lighten_amount = .35 if annotmach_reviewed else .9
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=lighten_amount)
    #truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def get_buttontup(ibs, qtindex):
    """
    helper for make_qres_api
    """
    model = qtindex.model()
    aid1 = model.get_header_data('qaid', qtindex)
    aid2 = model.get_header_data('aid', qtindex)
    truth = ibs.get_match_truth(aid1, aid2)
    truth_color = vh.get_truth_color(truth, base255=True,
                                        lighten_amount=0.35)
    truth_text = ibs.get_match_text(aid1, aid2)
    callback = partial(review_match, ibs, aid1, aid2)
    #print('get_button, aid1=%r, aid2=%r, row=%r, truth=%r' % (aid1, aid2, row, truth))
    buttontup = (truth_text, callback, truth_color)
    return buttontup


def test_inspect_matches(ibs, qaid_list, daid_list):
    """

    Args:
        ibs       (IBEISController):
        qaid_list (list): query annotation id list
        daid_list (list): database annotation id list

    Returns:
        dict: locals_

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> import guitool
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> main_locals = test_inspect_matches(ibs, qaid_list, daid_list)
        >>> main_execstr = ibeis.main_loop(main_locals)
        >>> if ut.show_was_requested():
        >>>     guitool.qtapp_loop()
        >>> print(main_execstr)
        >>> exec(main_execstr)
    """
    from ibeis.viz.interact import interact_qres2  # NOQA
    from ibeis.gui import inspect_gui
    from ibeis.dev import results_all
    allres = results_all.get_allres(ibs, qaid_list)
    tblname = 'qres'
    qaid2_qres = allres.qaid2_qres
    name_scoring = False
    ranks_lt = 5
    # This object is created inside QresResultsWidget
    #qres_api = inspect_gui.make_qres_api(ibs, qaid2_qres)  # NOQA
    # This is where you create the result widigt
    guitool.ensure_qapp()
    print('[inspect_matches] make_qres_widget')
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, ranks_lt=ranks_lt)
    print('[inspect_matches] show')
    qres_wgt.show()
    print('[inspect_matches] raise')
    qres_wgt.raise_()
    #query_review = interact_qres2.Interact_QueryResult(ibs, qaid2_qres)
    #self = interact_qres2.Interact_QueryResult(ibs, qaid2_qres, ranks_lt=ranks_lt)
    print('</inspect_matches>')
    # simulate double click
    #qres_wgt._on_click(qres_wgt.model.index(2, 2))
    #qres_wgt._on_doubleclick(qres_wgt.model.index(2, 0))
    locals_ =  locals()
    return locals_


def make_qres_api(ibs, qaid2_qres, ranks_lt=None, name_scoring=False):
    """
    Builds columns which are displayable in a ColumnListTableWidget

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-make_qres_api

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> import guitool
        >>> from ibeis.viz.interact import interact_qres2  # NOQA
        >>> from ibeis.gui import inspect_gui
        >>> from ibeis.dev import results_all
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:2]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> allres = results_all.get_allres(ibs, qaid_list)
        >>> tblname = 'qres'
        >>> qaid2_qres = allres.qaid2_qres
        >>> name_scoring = False
        >>> ranks_lt = 5
        >>> make_qres_api(ibs, qaid2_qres, ranks_lt, name_scoring)

    """
    if ut.VERBOSE:
        print('[inspect] make_qres_api')
    ibs.cfg.other_cfg.ranks_lt = 2
    ranks_lt = ranks_lt if ranks_lt is not None else ibs.cfg.other_cfg.ranks_lt
    candidate_matches = results_organizer.get_automatch_candidates(
        qaid2_qres, ranks_lt=ranks_lt, name_scoring=name_scoring, ibs=ibs, directed=False)
    # Get extra info
    (qaids, aids, scores, ranks) = candidate_matches

    # Preprocess Thumbs
    USE_MATCH_THUMBS = True
    if USE_MATCH_THUMBS:
        match_thumb_dir = ut.unixjoin(ibs.get_cachedir(), 'match_thumbs')
        ut.ensuredir(match_thumb_dir)
        #match_thumb_fpath_list = []
        #_prog = functools.partial(ut.ProgressIter, lbl='match chip: ', nTotal=len(qaids))

        #for qres, qaid, daid in _prog(zip(ut.dict_take(qaid2_qres, qaids), qaids, aids)):
        #    assert qres.qaid == qaid
        #    match_thumb_fpath_ = ut.unixjoin(match_thumb_dir, 'matchthumb-aid1=%d-aid2=%d.jpg' % ((qaid, daid)))
        #    match_thumb_fpath = qres.dump_match_img(ibs, daid, fpath=match_thumb_fpath_, saveax=True, fnum=32, notitle=True, verbose=False)
        #    match_thumb_fpath_list.append(match_thumb_fpath)

        #thumbtup_list = [(ut.augpath(fpath, 'thumb'), fpath, (128, 128), [], []) far fpath in match_thumb_fpath_list ]
        #print(ut.list_str(thumbtup_list))

        match_thumbtup_cache = {}
        def get_match_thumbtup(index, thumbsize=(128, 128)):
            qaid, daid = qaids[index], aids[index]
            qres = qaid2_qres[qaid]
            match_thumb_fpath_ = ut.unixjoin(match_thumb_dir, 'matchthumb-aid1=%d-aid2=%d.jpg' % ((qaid, daid)))
            if match_thumb_fpath_  not in match_thumbtup_cache:
                match_thumb_fpath = qres.dump_match_img(ibs, daid, fpath=match_thumb_fpath_, saveax=True, fnum=32, notitle=True, verbose=False)
                #match_thumb_fpath_list.append(match_thumb_fpath)
                match_thumbtup_cache[match_thumb_fpath_] = match_thumb_fpath
            fpath = match_thumbtup_cache[match_thumb_fpath_]
            #print(thumbsize)
            if isinstance(thumbsize, int):
                # HACK
                thumbsize = (thumbsize, thumbsize)
            thumbtup = (ut.augpath(fpath, 'thumb'), fpath, thumbsize, [], [])
            return thumbtup

    #ut.view_directory(match_thumb_dir)
    #return

    #opts = np.zeros(len(qaids))
    # Define column information

    # TODO: MAKE A PAIR IDER AND JUST USE EXISTING API_ITEM_MODEL FUNCTIONALITY
    # TO GET THOSE PAIRWISE INDEXES

    RES_THUMB_TEXT = 'ResThumb'
    MATCH_THUMB_TEXT = 'MatchThumb'

    col_name_list = [
        'score',
        REVIEWED_STATUS_TEXT,
        MATCHED_STATUS_TEXT,
        'querythumb',
        RES_THUMB_TEXT,
        'qname',
        'name',
        'rank',
        'qaid',
        'aid',
    ]
    #if ut.is_developer():
    #    pass
    #    col_name_list.insert(2, 'd_nGt')
    #    col_name_list.insert(2, 'q_nGt')
    #    #col_name_list.insert(2, 'q_nGt')

    col_types_dict = dict([
        ('qaid',       int),
        ('aid',        int),
        #('d_nGt',      int),
        #('q_nGt',      int),
        ('review',     'BUTTON'),
        (MATCHED_STATUS_TEXT, str),
        (REVIEWED_STATUS_TEXT, str),
        ('querythumb', 'PIXMAP'),
        (RES_THUMB_TEXT,   'PIXMAP'),
        ('qname',      str),
        ('name',       str),
        ('score',      float),
        ('rank',       int),
        ('truth',      bool),
        ('opt',        int),
    ])

    col_getters_dict = dict([
        ('qaid',       np.array(qaids)),
        ('aid',        np.array(aids)),
        #('d_nGt',      ibs.get_annot_num_groundtruth),
        #('q_nGt',      ibs.get_annot_num_groundtruth),
        ('review',     lambda rowid: get_buttontup),
        (MATCHED_STATUS_TEXT,  partial(get_match_status, ibs)),
        (REVIEWED_STATUS_TEXT,  partial(get_reviewed_status, ibs)),
        ('querythumb', ibs.get_annot_chip_thumbtup),
        (RES_THUMB_TEXT,   ibs.get_annot_chip_thumbtup),
        ('qname',      ibs.get_annot_names),
        ('name',       ibs.get_annot_names),
        ('score',      np.array(scores)),
        ('rank',       np.array(ranks)),
        #('truth',     truths),
        #('opt',       opts),
    ])

    if USE_MATCH_THUMBS:
        col_name_list.insert(col_name_list.index(RES_THUMB_TEXT) + 1, MATCH_THUMB_TEXT)
        col_types_dict[MATCH_THUMB_TEXT] = 'PIXMAP'
        col_getters_dict[MATCH_THUMB_TEXT] = get_match_thumbtup

    #get_status_bgrole_func = partial(get_match_status_bgrole, ibs)
    col_bgrole_dict = {
        MATCHED_STATUS_TEXT : partial(get_match_status_bgrole, ibs),
        REVIEWED_STATUS_TEXT: partial(get_reviewed_status_bgrole, ibs),
        #'aid'    : get_status_bgrole_func,
        #'qaid'   : get_status_bgrole_func,
    }
    # TODO: remove ider dict.
    # it is massively unuseful
    col_ider_dict = {
        MATCHED_STATUS_TEXT     : ('qaid', 'aid'),
        REVIEWED_STATUS_TEXT    : ('qaid', 'aid'),
        #'d_nGt'      : ('aid'),
        #'q_nGt'      : ('qaid'),
        'querythumb' : ('qaid'),
        'ResThumb'   : ('aid'),
        'qname'      : ('qaid'),
        'name'       : ('aid'),
    }
    col_setter_dict = {
        'qname': ibs.set_annot_names,
        'name': ibs.set_annot_names
    }
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'opt']
    sortby = 'score'
    get_thumb_size = lambda: ibs.cfg.other_cfg.thumb_size
    # Insert info into dict
    qres_api = CustomAPI(col_name_list, col_types_dict, col_getters_dict,
                         col_bgrole_dict, col_ider_dict, col_setter_dict,
                         editable_colnames, sortby, get_thumb_size)
    return qres_api


def launch_review_matches_interface(ibs, qres_list, dodraw=False):
    """ TODO: move to a more general function """
    from ibeis.gui import inspect_gui
    import guitool
    guitool.ensure_qapp()
    #backend_callback = back.front.update_tables
    backend_callback = None
    qaid2_qres = {qres.qaid: qres for qres in qres_list}
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, callback=backend_callback)
    if dodraw:
        qres_wgt.show()
        qres_wgt.raise_()
    return qres_wgt


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show

        python -m ibeis.gui.inspect_gui
        python -m ibeis.gui.inspect_gui --allexamples
        python -m ibeis.gui.inspect_gui --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
