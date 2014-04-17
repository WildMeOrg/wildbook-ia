from __future__ import absolute_import, division, print_function
from . import viz_helpers as vh
import utool
import drawtool.draw_func2 as df2
import numpy as np
from .viz_chip import show_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)


def show_name_of(ibs, rid, **kwargs):
    nid = ibs.get_roi_names(rid)
    show_name(ibs, nid, sel_rids=[rid], **kwargs)


def show_name(ibs, nid, nid2_rids=None, fnum=0, sel_rids=[], subtitle='',
              annote=False, **kwargs):
    print('[viz] show_name nid=%r' % nid)
    rids = ibs.get_rids_in_nids(nid)
    name = ibs.get_names(nid)
    rids = nid2_rids[nid]
    print('[viz] show_name %r' % ibs.ridstr(rids))
    nRows, nCols = vh.get_square_row_cols(len(rids))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px + 1)
    fig = df2.figure(fnum=fnum, pnum=pnum(0), **kwargs)
    fig.clf()
    # Trigger computation of all chips in parallel
    ibs.get_roi_cids(rids, enfore=True)
    for px, rid in enumerate(rids):
        show_chip(ibs, rid=rid, pnum=pnum(px), draw_ell=annote, kpts_alpha=.2)
        if rid in sel_rids:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_rid3(ibs, rid)
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
