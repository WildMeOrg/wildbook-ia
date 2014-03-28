from __future__ import division, print_function
from itertools import izip
import numpy as np
import utool


# A map from short internal headers to fancy headers seen by the user
fancy_headers = {
    'gid':        'Image Index',
    'nid':        'Name Index',
    'cid':        'Chip ID',
    'aif':        'All Detected',
    'gname':      'Image Name',
    'nCxs':       '#Chips',
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
    'gids':  ['gid', 'gname', 'nCxs', 'aif'],
    'cids':  ['cid', 'name', 'gname', 'nGt', 'nKpts', 'theta'],
    'nids':  ['nid', 'name', 'nCxs'],
    'res':   ['rank', 'score', 'name', 'cid']
}

# Lists internal headers whos items are editable
table_editable = {
    'gids':  [],
    'cids':  ['name'],
    'nids':  ['name'],
    'res':   ['name'],
}


def _get_datatup_list(ibs, tblname, index_list, header_order, extra_cols):
    '''
    Used by guiback to get lists of datatuples by internal column names.
    '''
    print('[gui] _get_datatup_list()')
    cols = _datatup_cols(ibs, tblname)
    cols.update(extra_cols)
    unknown_header = lambda indexes: ['ERROR!' for gx in indexes]
    get_tup = lambda header: cols.get(header, unknown_header)(index_list)
    unziped_tups = [get_tup(header) for header in header_order]
    datatup_list = [tup for tup in izip(*unziped_tups)]
    return datatup_list


def _datatup_cols(ibs, tblname, cx2_score=None):
    '''
    Returns maps which map which maps internal column names
    to lazy evaluation functions which compute the data (hence the lambdas)
    '''
    print('[gui] _datatup_cols()')
    # Return requested columns
    if tblname == 'nids':
        cols = {
            'nid':   lambda nids: nids,
            'name':  lambda nids: ibs.get_names(nids),
            'nCxs':  lambda nids: ibs.get_num_cids_in_name(nids),
        }
    elif tblname == 'gids':
        cols = {
            'gid':   lambda gids: gids,
            'aif':   lambda gids: ibs.get_image_aifs(gids),
            'gname': lambda gids: ibs.get_image_gnames(gids),
            'nCxs':  lambda gids: ibs.get_num_cids_in_gids(gids),
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
            'cid':    lambda cids: cids,
            'name':   lambda cids: ibs.get_chip_names(cids),
            'gname':  lambda cids: ibs.get_chip_gname(cids),
            'nGt':    lambda cids: ibs.get_chip_num_groundtruth(),
            'nKpts':  lambda cids: ibs.get_chip_nKpts(cids),
            'theta':  lambda cids: map(theta_str, ibs.get_chip_theta(cids)),
            'roi':    lambda cids: map(str, ibs.get_chip_roi(cids)),
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
    print('headers = %r ' % headers)
    print('editable = %r ' % editable)
    col_headers, col_editable = make_header_lists(headers, editable)
    return col_headers, col_editable


def _get_table_datatup_list(ibs, tblname, col_headers, col_editable, extra_cols={},
                            index_list=None, prefix_cols=[]):
    if index_list is None:
        index_list = ibs.get_valid_ids(tblname)
        # Prefix datatup
        prefix_datatup = [[prefix_col.get(header, 'error')
                           for header in col_headers]
                          for prefix_col in prefix_cols]
        body_datatup = _get_datatup_list(ibs, tblname, index_list,
                                         col_headers, extra_cols)
        datatup_list = prefix_datatup + body_datatup
        return datatup_list


def emit_populate_table(back, tblname, *args, **kwargs):
    print('[gui_item_tables] _populate_table(%r)' % tblname)
    col_headers, col_editable = _get_table_headers_editable(tblname)
    datatup_list = _get_table_datatup_list(back.ibs, tblname, col_headers,
                                           col_editable, *args, **kwargs)
    row_list = range(len(datatup_list))
    # Populate with fancyheaders.
    col_fancyheaders = [fancy_headers[key]
                        if key in fancy_headers else key
                        for key in col_headers]
    print('[gui] populateSignal.emit(%r, len=%r, len=%r, len=%r, len=%r)' %
          ((tblname, len(col_fancyheaders), len(col_editable), len(row_list),
            len(datatup_list))))
    back.populateSignal.emit(tblname, col_fancyheaders, col_editable,
                             row_list, datatup_list)
