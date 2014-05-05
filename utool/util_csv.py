from __future__ import absolute_import, division, print_function
import numpy as np
from .util_type import is_list, is_int, is_str, is_float
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[csv]')


def numpy_to_csv(arr, col_labels=None, header='', col_type=None):
    col_list = arr.T.tolist()
    return make_csv_table(col_labels, col_list, header, col_type)


def make_csv_table(column_labels=None, column_list=[], header='',
                   column_type=None):
    """
    Creates a csv table with aligned columns
    """
    if len(column_list) == 0:
        print('[csv] No columns')
        return header
    column_len = [len(col) for col in column_list]
    num_data = column_len[0]
    if num_data == 0:
        #print('[csv.make_csv_table()] No data. (header=%r)' % (header,))
        return header
    if any([num_data != clen for clen in column_len]):
        print('[csv] column_labels = %r ' % (column_labels,))
        print('[csv] column_len = %r ' % (column_len,))
        print('[csv] inconsistent column lengths')
        return header

    if column_type is None:
        column_type = [type(col[0]) for col in column_list]

    csv_rows = []
    csv_rows.append(header)
    csv_rows.append('# NumData %r' % num_data)

    column_maxlen = []
    column_str_list = []

    if column_labels is None:
        column_labels = [''] * len(column_list)

    def _toint(c):
        try:
            if np.isnan(c):
                return 'nan'
        except TypeError as ex:
            print('------')
            print('[csv] TypeError %r ' % ex)
            print('[csv] _toint(c) failed')
            print('[csv] c = %r ' % c)
            print('[csv] type(c) = %r ' % type(c))
            print('------')
            raise
        return ('%d') % int(c)

    for col, lbl, coltype in iter(zip(column_list, column_labels, column_type)):
        if coltype is list or is_list(coltype):
            #col_str = [str(c).replace(',', '<comma>').replace('.', '<dot>') for c in iter(col)]
            col_str = [str(c).replace(',', ' ').replace('.', '<dot>') for c in iter(col)]
        elif coltype is float or is_float(coltype):
            col_str = [('%.2f') % float(c) for c in iter(col)]
        elif coltype is int or is_int(coltype):
            col_str = [_toint(c) for c in iter(col)]
        elif coltype is str or is_str(coltype):
            col_str = [str(c).replace(',', '<comma>') for c in iter(col)]
        else:
            col_str = [str(c) for c in iter(col)]
        col_lens = [len(s) for s in iter(col_str)]
        max_len  = max(col_lens)
        max_len  = max(len(lbl), max_len)
        column_maxlen.append(max_len)
        column_str_list.append(col_str)

    _fmtfn = lambda maxlen: ''.join(['%', str(maxlen + 2), 's'])
    fmtstr = ','.join([_fmtfn(maxlen) for maxlen in column_maxlen])
    csv_rows.append('# ' + fmtstr % tuple(column_labels))
    for row in zip(*column_str_list):
        csv_rows.append('  ' + fmtstr % row)

    csv_text = '\n'.join(csv_rows)
    return csv_text
