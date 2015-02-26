from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
import numpy as np
from ibeis import ibsfuncs
from plottool import plot_helpers as ph
from .viz_chip import show_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)


def show_name_of(ibs, aid, **kwargs):
    nid = ibs.get_annot_names(aid)
    show_name(ibs, nid, sel_aids=[aid], **kwargs)


@utool.indent_func
def show_name(ibs, nid, nid2_aids=None, in_image=True, fnum=0, sel_aids=[], subtitle='',
              annote=False, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        nid (?):
        nid2_aids (dict):
        in_image (bool):
        fnum (int):  figure number
        sel_aids (list):
        subtitle (str):
        annote (bool):

    CommandLine:
        python -m ibeis.viz.viz_name --test-show_name

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_name import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid = 1
        >>> nid2_aids = None
        >>> in_image = True
        >>> fnum = 0
        >>> sel_aids = []
        >>> subtitle = ''
        >>> annote = False
        >>> # execute function
        >>> result = show_name(ibs, nid, nid2_aids, in_image, fnum, sel_aids, subtitle, annote)
        >>> # verify results
        >>> print(result)
    """
    print('[viz] show_name nid=%r' % nid)
    aid_list = ibs.get_name_aids(nid)
    name = ibs.get_name_texts((nid,))
    ibsfuncs.ensure_annotation_data(ibs, aid_list, chips=(not in_image or annote), feats=annote)
    print('[viz] show_name=%r aid_list=%r' % (name, aid_list))
    nAids = len(aid_list)
    if nAids > 0:
        nRows, nCols = ph.get_square_row_cols(nAids)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        #gs2 = gridspec.GridSpec(nRows, nCols)
        pnum_ = df2.get_pnum_func(nRows, nCols)
        fig = df2.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
        fig.clf()
        # Trigger computation of all chips in parallel
        for px, aid in enumerate(aid_list):
            show_chip(ibs, aid=aid, pnum=pnum_(px), annote=annote, in_image=in_image)
            if aid in sel_aids:
                ax = df2.gca()
                df2.draw_border(ax, df2.GREEN, 4)
            #plot_aid3(ibs, aid)
        if isinstance(nid, np.ndarray):
            nid = nid[0]
        if isinstance(name, np.ndarray):
            name = name[0]
    else:
        df2.imshow_null(fnum=fnum, **kwargs)

    figtitle = 'Name View nid=%r name=%r' % (nid, name)
    df2.set_figtitle(figtitle)
    #if not annote:
    #    title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_name
        python -m ibeis.viz.viz_name --allexamples
        python -m ibeis.viz.viz_name --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
