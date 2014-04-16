from __future__ import absolute_import, division, print_function
from . import viz_helpers as vh
import utool
import drawtool.draw_func2 as df2
import numpy as np
from .viz_chip import show_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)


def show_name_of(ibs, cid, **kwargs):
    nid = ibs.get_chip_names(cid)
    show_name(ibs, nid, sel_cids=[cid], **kwargs)


def show_name(ibs, nid, nid2_cids=None, fnum=0, sel_cids=[], subtitle='',
              annote=False, **kwargs):
    print('[viz] show_name nid=%r' % nid)
    nid2_name = ibs.tables.nid2_name
    cid2_nid   = ibs.tables.cid2_nid
    name = nid2_name[nid]
    if not nid2_cids is None:
        cids = nid2_cids[nid]
    else:
        cids = np.where(cid2_nid == nid)[0]
    print('[viz] show_name %r' % ibs.cidstr(cids))
    nRows, nCols = vh.get_square_row_cols(len(cids))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px + 1)
    fig = df2.figure(fnum=fnum, pnum=pnum(0), **kwargs)
    fig.clf()
    # Trigger computation of all chips in parallel
    ibs.refresh_features(cids)
    for px, cid in enumerate(cids):
        show_chip(ibs, cid=cid, pnum=pnum(px), draw_ell=annote, kpts_alpha=.2)
        if cid in sel_cids:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_cid3(ibs, cid)
    if isinstance(nid, np.ndarray):
        nid = nid[0]
    if isinstance(name, np.ndarray):
        name = name[0]

    figtitle = 'Name View nid=%r name=%r' % (nid, name)
    df2.set_figtitle(figtitle)
    #if not annote:
        #title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)
