from __future__ import absolute_import, division, print_function
# Python
import os
import re
import textwrap
# Science
import numpy as np
from os.path import join, splitext
# Util
from . import util_cplat
from . import util_path
from . import util_num
from . import util_dev
from . import util_cache
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[latex]')

"""
#def ensure_latex_environ():
    #paths = os.environ['PATH'].split(os.pathsep)
    #mpl.rc('font',**{'family':'serif'})
    #mpl.rc('text', usetex=True)
    #mpl.rc('text.latex',unicode=True)
    #mpl.rc('text.latex',preamble='\usepackage[utf8]{inputenc}')
"""


def make_full_document(text):
    doc_preamb = r'''
    \documentclass{article}
    \pagenumbering{gobble}
    '''
    doc_header = r'''
    \begin{document}
    '''
    doc_footer = r'''
    \end{document}
    '''
    return doc_preamb + doc_header + text + doc_footer


def render(input_text, fnum=1):
    import pylab as plt
    import matplotlib as mpl
    verbose = False
    text = make_full_document(input_text)
    cwd = os.getcwd()
    text_dir = join(cwd, 'tmptex')
    text_fname = 'latex_formatter_temp.tex'
    text_fpath = join(text_dir, text_fname)
    pdf_fpath = splitext(text_fpath)[0] + '.pdf'
    jpg_fpath = splitext(text_fpath)[0] + '.jpg'
    try:
        util_path.ensuredir(text_dir, verbose=verbose)
        os.chdir(text_dir)
        util_cache.write_to(text_fpath, text)
        util_cplat.cmd('pdflatex', text_fpath, verbose=verbose)
        assert util_path.checkpath(pdf_fpath, verbose=verbose), 'latex failed'
        util_cplat.cmd('convert', '-density', '300', pdf_fpath, '-quality', '90', jpg_fpath, verbose=verbose)
        assert util_path.checkpath(jpg_fpath, verbose=verbose), 'imgmagick failed'
        tex_img = plt.imread(jpg_fpath)
        # Crop img bbox
        nonwhite_x = np.where(tex_img.flatten() != 255)[0]
        nonwhite_rows = nonwhite_x // tex_img.shape[1]
        nonwhite_cols = nonwhite_x % tex_img.shape[1]
        x1 = nonwhite_cols.min()
        y1 = nonwhite_rows.min()
        x2 = nonwhite_cols.max()
        y2 = nonwhite_rows.max()
        #util.embed()
        cropped = tex_img[y1:y2, x1:x2]
        fig = plt.figure(fnum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cropped, cmap=mpl.cm.gray)
        #plt.show()
        #mpl.rc('text', usetex=True)
        #mpl.rc('font', family='serif')
        #plt.figure()
        #plt.text(9, 3.4, text, size=12)
        #plt.show()
    except Exception as ex:
        print('LATEX ERROR')
        print(text)
        print(ex)
        print('LATEX ERROR')
        pass
    finally:
        os.chdir(cwd)
        if util_path.checkpath(text_dir, verbose=verbose):
            util_path.delete(text_dir)


def latex_multicolumn(data, ncol=2):
    data = escape_latex(data)
    return r'\multicolumn{%d}{|c|}{%s}' % (ncol, data)


def latex_multirow(data, nrow=2):
    return r'\multirow{%d}{*}{|c|}{%s}' % (nrow, data)


def latex_mystats(lbl, data, mode=0):
    stats_ = util_dev.mystats(data)
    max_ = stats_['max']
    min_ = stats_['min']
    mean = stats_['mean']
    std  = stats_['std']
    shape = stats_['shape']

    #int_fmt = lambda num: util.num_fmt(int(num))
    float_fmt = lambda num: util_num.num_fmt(float(num))
    tup_fmt = lambda tup: str(tup)
    fmttup = (float_fmt(min_), float_fmt(max_), float_fmt(mean), float_fmt(std), tup_fmt(shape))
    lll = ' ' * len(lbl)
    if mode == 0:
        prefmtstr = r'''
        {label} stats & min ; max = %s ; %s\\
        {space}       & mean; std = %s ; %s\\
        {space}       & shape = %s \\'''
    if mode == 1:
        prefmtstr = r'''
        {label} stats & min  = $%s$\\
        {space}       & max  = $%s$\\
        {space}       & mean = $%s$\\
        {space}       & std  = $%s$\\
        {space}       & shape = $%s$\\'''
    fmtstr = prefmtstr.format(label=lbl, space=lll)
    latex_str = textwrap.dedent(fmtstr % fmttup).strip('\n') + '\n'
    return latex_str


def latex_scalar(lbl, data):
    return (r'%s & %s\\' % (lbl, util_num.num_fmt(data))) + '\n'


def make_stats_tabular():
    'tabular for dipslaying statistics'
    pass


def ensure_rowvec(arr):
    arr = np.array(arr)
    arr.shape = (1, arr.size)
    return arr


def ensure_colvec(arr):
    arr = np.array(arr)
    arr.shape = (arr.size, 1)
    return arr


def padvec(shape=(1, 1)):
    pad = np.array([[' ' for c in xrange(shape[1])] for r in xrange(shape[0])])
    return pad


def escape_latex(unescaped_latex_str):
    ret = unescaped_latex_str
    ret = ret.replace('#', '\\#')
    ret = ret.replace('%', '\\%')
    ret = ret.replace('_', '\\_')
    return ret


def replace_all(str_, repltups):
    ret = str_
    for ser, rep in repltups:
        ret = re.sub(ser, rep, ret)
    return ret


def make_score_tabular(row_lbls, col_lbls, scores, title=None,
                       out_of=None, bold_best=True,
                       replace_rowlbl=None, flip=False):
    'tabular for displaying scores'
    bigger_is_better = True
    if flip:
        bigger_is_better = not bigger_is_better
        flip_repltups = [('<', '>'), ('score', 'error')]
        col_lbls = [replace_all(lbl, flip_repltups) for lbl in col_lbls]
        if title is not None:
            title = replace_all(title, flip_repltups)
        if out_of is not None:
            scores = out_of - scores

    if replace_rowlbl is not None:
        for ser, rep in replace_rowlbl:
            row_lbls = [re.sub(ser, rep, lbl) for lbl in row_lbls]

    # Abbreviate based on common substrings
    SHORTEN_ROW_LBLS = True
    common_rowlbl = None
    if SHORTEN_ROW_LBLS:
        if isinstance(row_lbls, list):
            row_lbl_list = row_lbls
        else:
            row_lbl_list = row_lbls.flatten().tolist()
        # Split the rob labels into the alg components
        #algcomp_list = [lbl.split(')_') for lbl in row_lbl_list]
        longest = long_substr(row_lbl_list)
        common_strs = []
        while len(longest) > 10:
            common_strs += [longest]
            row_lbl_list = [row.replace(longest, '...') for row in row_lbl_list]
            longest = long_substr(row_lbl_list)
        common_rowlbl = ('...'.join(common_strs)).replace(')_', ')_\n')
        row_lbls = row_lbl_list
        if len(row_lbl_list) == 1:
            common_rowlbl = row_lbl_list[0]
            row_lbls = ['0']

    # Stack into a tabular body
    col_lbls = ensure_rowvec(col_lbls)
    row_lbls = ensure_colvec(row_lbls)
    _0 = np.vstack([padvec(), row_lbls])
    _1 = np.vstack([col_lbls, scores])
    body = np.hstack([_0, _1])
    body = [[str_ for str_ in row] for row in body]
    # Fix things in each body cell
    AUTOFIX_LATEX = True
    FORCE_INT = True
    DO_PERCENT = True
    for r in xrange(len(body)):
        for c in xrange(len(body[0])):
            # In data land
            if r > 0 and c > 0:
                # Force integer
                if FORCE_INT:
                    body[r][c] = str(int(float(body[r][c])))
            # Remove bad formatting;
            if AUTOFIX_LATEX:
                body[r][c] = escape_latex(body[r][c])

    # Bold the best scores
    if bold_best:
        best_col_scores = scores.max(0) if bigger_is_better else scores.min(0)
        rows_to_bold = [np.where(scores[:, colx] == best_col_scores[colx])[0]
                        for colx in xrange(len(scores.T))]
        for colx, rowx_list in enumerate(rows_to_bold):
            for rowx in rowx_list:
                body[rowx + 1][colx + 1] = '\\txtbf{' + body[rowx + 1][colx + 1] + '}'

    # More fixing after the bold is in place
    for r in xrange(len(body)):
        for c in xrange(len(body[0])):
            # In data land
            if r > 0 and c > 0:
                if out_of is not None:
                    body[r][c] = body[r][c] + '/' + str(out_of)
                    if DO_PERCENT:
                        percent = ' = %.1f%%' % float(100 * scores[r - 1, c - 1] / out_of)
                        body[r][c] += escape_latex(percent)

    # Align columns for pretty printing
    body = np.array(body)
    ALIGN_BODY = True
    if ALIGN_BODY:
        new_body_cols = []
        for col in body.T:
            colstrs = map(str, col.tolist())
            collens = map(len, colstrs)
            maxlen = max(collens)
            newcols = [str_ + (' ' * (maxlen - len(str_))) for str_ in colstrs]
            new_body_cols += [newcols]
        body = np.array(new_body_cols).T

    # Build Body (and row layout)
    HLINE_SEP = True
    rowsep = ''
    colsep = '&'
    endl = '\\\\\n'
    hline = r'\hline'
    extra_rowsep_pos_list = [1]
    if HLINE_SEP:
        rowsep = hline + '\n'
    rowstr_list = [colsep.join(row) + endl for row in body]
    # Insert title
    if title is not None:
        tex_title = latex_multicolumn(title, len(body[0])) + endl
        rowstr_list = [tex_title] + rowstr_list
        extra_rowsep_pos_list += [2]
    # Apply an extra hline (for label)
    for pos in sorted(extra_rowsep_pos_list)[::-1]:
        rowstr_list.insert(pos, '')
    tabular_body = rowsep.join(rowstr_list)

    # Build Column Layout
    col_layout_sep = '|'
    col_layout_list = ['l'] * len(body[0])
    extra_collayoutsep_pos_list = [1]
    for pos in  sorted(extra_collayoutsep_pos_list)[::-1]:
        col_layout_list.insert(pos, '')
    col_layout = col_layout_sep.join(col_layout_list)

    tabular_head = (r'\begin{tabular}{|%s|}' % col_layout) + '\n'
    tabular_tail = r'\end{tabular}'

    tabular_str = rowsep.join([tabular_head, tabular_body, tabular_tail])

    if common_rowlbl is not None:
        #tabular_str += escape_latex('\n\nThe following parameters were held fixed:\n' + common_rowlbl)
        pass
    return tabular_str


def _tabular_header_and_footer(col_layout):
    tabular_head = textwrap.dedent(r'\begin{tabular}{|%s|}' % col_layout)
    tabular_tail = textwrap.dedent(r'\end{tabular}')
    return tabular_head, tabular_tail


def long_substr(strlist):
    # Longest common substring
    # http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    substr = ''
    if len(strlist) > 1 and len(strlist[0]) > 0:
        for i in range(len(strlist[0])):
            for j in range(len(strlist[0]) - i + 1):
                if j > len(substr) and is_substr(strlist[0][i:i + j], strlist):
                    substr = strlist[0][i:i + j]
    return substr


def is_substr(find, strlist):
    if len(strlist) < 1 and len(find) < 1:
        return False
    for i in range(len(strlist)):
        if find not in strlist[i]:
            return False
    return True


def tabular_join(tabular_body_list, nCols=2):
    dedent = textwrap.dedent
    tabular_head = dedent(r'''
    \begin{tabular}{|l|l|}
    ''')
    tabular_tail = dedent(r'''
    \end{tabular}
    ''')
    hline = ''.join([r'\hline', '\n'])
    tabular_body = hline.join(tabular_body_list)
    tabular = hline.join([tabular_head, tabular_body, tabular_tail])
    return tabular
