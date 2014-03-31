from __future__ import division, print_function
from itertools import izip
import numpy as np
import utool
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QAbstractItemView
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[guitables]', DEBUG=False)

UUID_type = str


# A map from short internal headers to fancy headers seen by the user
fancy_headers = {
    'gid':        'Image Index',
    'nid':        'Name Index',
    'cid':        'Chip ID',
    'aif':        'All Detected',
    'gname':      'Image Name',
    'nRids':       '#ROIs',
    'name':       'Name',
    'nGt':        '#GT',
    'nKpts':      '#Kpts',
    'theta':      'Theta',
    'roi':        'ROI (x, y, w, h)',
    'rank':       'Rank',
    'score':      'Confidence',
    'match_name': 'Matching Name',
}
reverse_fancy = {v: k for (k, v) in fancy_headers.items()}

# A list of default internal headers to display
table_headers = {
    'gids':  ['gid', 'gname', 'nRids', 'aif'],
    'rids':  ['rid', 'name', 'gname', 'nGt', 'nKpts', 'theta'],
    'nids':  ['nid', 'name', 'nRids'],
    'res':   ['rank', 'score', 'name', 'rid']
}

# Lists internal headers whos items are editable
table_editable = {
    'gids':  [],
    'rids':  ['name'],
    'nids':  ['name'],
    'res':   ['name'],
}


def _get_datatup_list(ibs, tblname, index_list, header_order, extra_cols):
    '''
    Used by guiback to get lists of datatuples by internal column names.
    '''
    printDBG('[gui] _get_datatup_list()')
    cols = _datatup_cols(ibs, tblname)
    printDBG('[gui] cols=%r' % cols)
    cols.update(extra_cols)
    printDBG('[gui] cols=%r' % cols)
    unknown_header = lambda indexes: ['ERROR!' for gx in indexes]
    get_tup = lambda header: cols.get(header, unknown_header)(index_list)
    unziped_tups = [get_tup(header) for header in header_order]
    printDBG('[gui] unziped_tups=%r' % unziped_tups)
    datatup_list = [tup for tup in izip(*unziped_tups)]
    printDBG('[gui] datatup_list=%r' % datatup_list)
    return datatup_list


def _datatup_cols(ibs, tblname, cx2_score=None):
    '''
    Returns maps which map which maps internal column names
    to lazy evaluation functions which compute the data (hence the lambdas)
    '''
    printDBG('[gui] _datatup_cols()')
    # Return requested columns
    if tblname == 'nids':
        cols = {
            'nid':   lambda nids: nids,
            'name':  lambda nids: ibs.get_names(nids),
            'nRids':  lambda nids: ibs.get_num_rids_in_nids(nids),
        }
    elif tblname == 'gids':
        cols = {
            'gid':   lambda gids: gids,
            'aif':   lambda gids: ibs.get_image_aifs(gids),
            'gname': lambda gids: ibs.get_image_gnames(gids),
            'nRids':  lambda gids: ibs.get_num_rids_in_gids(gids),
            'unixtime': lambda gids: ibs.get_image_unixtime(gids),
        }
    elif tblname in ['cxs', 'res']:
        np.tau = (2 * np.pi)
        taustr = 'tau' if utool.get_flag('--myway') else '2pi'

        def theta_str(theta):
            'Format theta so it is interpretable in base 10'
            #coeff = (((tau - theta) % tau) / tau)
            coeff = (theta / np.tau)
            return ('%.2f * ' % coeff) + taustr

        cols = {
            'rid':    lambda rids: rids,
            'name':   lambda rids: ibs.get_roi_names(rids),
            'gname':  lambda rids: ibs.get_roi_gname(rids),
            'nGt':    lambda rids: ibs.get_roi_num_groundtruth(rids),
            'nKpts':  lambda rids: ibs.get_roi_nKpts(rids),
            'theta':  lambda rids: map(theta_str, ibs.get_roi_theta(rids)),
            'roi':    lambda rids: map(str, ibs.get_roi_bbox(rids)),
        }
        if tblname == 'res':
            cols.update({
                'rank':   lambda cxs:  range(1, len(cxs) + 1),
            })
    else:
        cols = {}
    return cols


# ----


def make_header_lists(tbl_headers, editable_list, prop_keys=[]):
    col_headers = tbl_headers[:] + prop_keys
    col_editable = [False] * len(tbl_headers) + [True] * len(prop_keys)
    for header in editable_list:
        col_editable[col_headers.index(header)] = True
    return col_headers, col_editable


def _get_table_headers_editable(tblname):
    headers = table_headers[tblname]
    editable = table_editable[tblname]
    printDBG('headers = %r ' % headers)
    printDBG('editable = %r ' % editable)
    col_headers, col_editable = make_header_lists(headers, editable)
    return col_headers, col_editable


def _get_table_datatup_list(ibs, tblname, col_headers, col_editable, extra_cols={},
                            index_list=None, prefix_cols=[]):
    if index_list is None:
        index_list = ibs.get_valid_ids(tblname)
    printDBG('[tables] len(index_list) = %r' % len(index_list))
    # Prefix datatup
    prefix_datatup = [[prefix_col.get(header, 'error')
                       for header in col_headers]
                      for prefix_col in prefix_cols]
    body_datatup = _get_datatup_list(ibs, tblname, index_list,
                                     col_headers, extra_cols)
    datatup_list = prefix_datatup + body_datatup
    return datatup_list


def emit_populate_table(back, tblname, *args, **kwargs):
    printDBG('>>>>>>>>>>>>>>>>>>>>>')
    printDBG('[gui_item_tables] _populate_table(%r)' % tblname)
    col_headers, col_editable = _get_table_headers_editable(tblname)
    printDBG('[gui_item_tables] col_headers = %r' % col_headers)
    printDBG('[gui_item_tables] col_editable = %r' % col_editable)
    datatup_list = _get_table_datatup_list(back.ibs, tblname, col_headers,
                                           col_editable, *args, **kwargs)
    printDBG('[gui_item_tables] datatup_list = %r' % datatup_list)
    row_list = range(len(datatup_list))
    # Populate with fancyheaders.
    col_fancyheaders = [fancy_headers[key]
                        if key in fancy_headers else key
                        for key in col_headers]
    printDBG('[gui] populateSignal.emit(%r, len=%r, len=%r, len=%r, len=%r)' %
             ((tblname, len(col_fancyheaders), len(col_editable), len(row_list),
               len(datatup_list))))
    back.populateSignal.emit(tblname, col_fancyheaders, col_editable,
                             row_list, datatup_list)


def populate_item_table(tbl, col_fancyheaders, col_editable, row_list, datatup_list):
    # TODO: for chip table: delete metedata column
    # RCOS TODO:
    # I have a small right-click context menu working
    # Maybe one of you can put some useful functions in these?
    # RCOS TODO: How do we get the clicked item on a right click?
    # RCOS TODO:
    # The data tables should not use the item model
    # Instead they should use the more efficient and powerful
    # QAbstractItemModel / QAbstractTreeModel

    hheader = tbl.horizontalHeader()

    sort_col = hheader.sortIndicatorSection()
    sort_ord = hheader.sortIndicatorOrder()
    tbl.sortByColumn(0, Qt.AscendingOrder)  # Basic Sorting
    tblWasBlocked = tbl.blockSignals(True)
    tbl.clear()
    tbl.setColumnCount(len(col_fancyheaders))
    tbl.setRowCount(len(row_list))
    tbl.verticalHeader().hide()
    tbl.setHorizontalHeaderLabels(col_fancyheaders)
    tbl.setSelectionMode(QAbstractItemView.SingleSelection)
    tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    tbl.setSortingEnabled(False)
    # Add items for each row and column
    for row in iter(row_list):
        data_tup = datatup_list[row]
        for col, data in enumerate(data_tup):
            item = QtGui.QTableWidgetItem()
            # RCOS TODO: Pass in datatype here.
            # BOOLEAN DATA
            if utool.is_bool(data) or data == 'True' or data == 'False':
                check_state = Qt.Checked if bool(data) else Qt.Unchecked
                item.setCheckState(check_state)
                #item.setData(Qt.DisplayRole, bool(data))
            # INTEGER DATA
            elif utool.is_int(data):
                item.setData(Qt.DisplayRole, int(data))
            # FLOAT DATA
            elif utool.is_float(data):
                item.setData(Qt.DisplayRole, float(data))
            # STRING DATA
            else:
                item.setText(str(data))
            # Mark as editable or not
            if col_editable[col]:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                item.setBackground(QtGui.QColor(250, 240, 240))
            else:
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignHCenter)
            tbl.setItem(row, col, item)

    #printDBG(dbg_col2_dtype)
    tbl.setSortingEnabled(True)
    tbl.sortByColumn(sort_col, sort_ord)  # Move back to old sorting
    tbl.show()
    tbl.blockSignals(tblWasBlocked)
