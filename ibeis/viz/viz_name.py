from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
import numpy as np
from ibeis.dev import ibsfuncs
from plottool import plot_helpers as ph
from .viz_chip import show_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)


def show_name_of(ibs, rid, **kwargs):
    nid = ibs.get_roi_names(rid)
    show_name(ibs, nid, sel_rids=[rid], **kwargs)


@utool.indent_func
def show_name(ibs, nid, nid2_rids=None, in_image=True, fnum=0, sel_rids=[], subtitle='',
              annote=False, **kwargs):
    print('[viz] show_name nid=%r' % nid)
    rids = ibs.get_name_rids(nid)
    name = ibs.get_names((nid,))
    ibsfuncs.ensure_roi_data(ibs, rids, chips=(not in_image or annote), feats=annote)
    print('[viz] show_name=%r rids=%r' % (name, rids))
    nRids = len(rids)
    if nRids > 0:
        nRows, nCols = ph.get_square_row_cols(nRids)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        #gs2 = gridspec.GridSpec(nRows, nCols)
        pnum_ = df2.get_pnum_func(nRows, nCols)
        fig = df2.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
        fig.clf()
        # Trigger computation of all chips in parallel
        for px, rid in enumerate(rids):
            show_chip(ibs, rid=rid, pnum=pnum_(px), annote=annote, in_image=in_image)
            if rid in sel_rids:
                ax = df2.gca()
                df2.draw_border(ax, df2.GREEN, 4)
            #plot_rid3(ibs, rid)
        if isinstance(nid, np.ndarray):
            nid = nid[0]
        if isinstance(name, np.ndarray):
            name = name[0]
    else:
        df2.imshow_null(fnum=fnum, **kwargs)

    figtitle = 'Name View nid=%r name=%r' % (nid, name)
    df2.set_figtitle(figtitle)
    #if not annote:
        #title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)
