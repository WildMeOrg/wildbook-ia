from __future__ import absolute_import, division, print_function
import plottool.draw_func2 as df2
import numpy as np
from ibeis import ibsfuncs
from plottool import plot_helpers as ph
import plottool as pt
import utool as ut
from ibeis.viz import viz_chip
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz]', DEBUG=False)


def show_name_of(ibs, aid, **kwargs):
    nid = ibs.get_annot_names(aid)
    show_name(ibs, nid, sel_aids=[aid], **kwargs)


def testdata_showname():
    import ibeis
    ibs = ibeis.opendb(defaultdb='testdb1')
    name_text = ut.get_argval('--name', type_=str, default='easy')
    nid = ibs.get_name_rowids_from_text(name_text)
    in_image = not ut.get_argflag('--no-inimage')
    index_list = ut.get_argval('--index_list', type_=list, default=None)
    return ibs, nid, in_image, index_list


def testdata_multichips():
    import ibeis
    ibs = ibeis.opendb(defaultdb='testdb1')
    aid_list = ut.get_argval('--aids', type_=list, default=[1, 2, 3])
    in_image = not ut.get_argflag('--no-inimage')
    return ibs, aid_list, in_image

#9108 and 9180


def show_multiple_chips(ibs, aid_list, in_image=True, fnum=0, sel_aids=[],
                        subtitle='', annote=False, **kwargs):
    """
    CommandLine:
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --no-inimage
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=6435,9861,137,6563,9167,12547,9332,12598,13285 --no-inimage --notitle
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=137,6563,12547,9332,12598,13285 --no-inimage --notitle --adjust=.05
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=6563,9332,13285,12598 --no-inimage --notitle --adjust=.05 --rc=1,4
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db PZ_Master0 --aids=1288 --no-inimage --notitle --adjust=.05
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db PZ_Master0 --aids=4020,4839 --no-inimage --notitle --adjust=.05

        python -m ibeis.viz.viz_name --test-show_multiple_chips --db NNP_Master3 --aids=6524,6540,6571,6751 --no-inimage --notitle --adjust=.05 --diskshow

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_name import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list, in_image = testdata_multichips()
        >>> fnum = 0
        >>> sel_aids = []
        >>> subtitle = ''
        >>> annote = False
        >>> result = show_multiple_chips(ibs, aid_list, in_image, fnum, sel_aids, subtitle, annote)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    if fnum is None:
        fnum = pt.next_fnum()

    ibsfuncs.ensure_annotation_data(ibs, aid_list, chips=(not in_image or annote), feats=annote)
    nAids = len(aid_list)
    if nAids == 0:
        df2.imshow_null(fnum=fnum, **kwargs)
    else:
        rc = ut.get_argval('--rc', type_=list, default=None)
        if rc is None:
            nRows, nCols = ph.get_square_row_cols(nAids)
        else:
            nRows, nCols = rc
        notitle = ut.get_argflag('--notitle')
        draw_lbls = not ut.get_argflag('--no-draw_lbls')
        show_chip_kw = dict(annote=annote, in_image=in_image, notitle=notitle, draw_lbls=draw_lbls)
        #print('[viz_name] * r=%r, c=%r' % (nRows, nCols))
        #gs2 = gridspec.GridSpec(nRows, nCols)
        pnum_ = df2.get_pnum_func(nRows, nCols)
        fig = df2.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
        fig.clf()
        # Trigger computation of all chips in parallel
        for px, aid in enumerate(aid_list):
            viz_chip.show_chip(ibs, aid=aid, pnum=pnum_(px), **show_chip_kw)
            if aid in sel_aids:
                ax = df2.gca()
                df2.draw_border(ax, df2.GREEN, 4)
            #plot_aid3(ibs, aid)

        # HACK to show in image and not in image
        DOBOTH = ut.get_argflag('--doboth')
        if DOBOTH:
            show_chip_kw['in_image'] = not show_chip_kw['in_image']
            for px, aid in enumerate(aid_list, start=px + 1):
                viz_chip.show_chip(ibs, aid=aid, pnum=pnum_(px), **show_chip_kw)
                if aid in sel_aids:
                    ax = df2.gca()
                    df2.draw_border(ax, df2.GREEN, 4)


#@ut.indent_func
def show_name(ibs, nid, in_image=True, fnum=0, sel_aids=[], subtitle='',
              annote=False, aid_list=None, index_list=None,  **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        nid (?):
        in_image (bool):
        fnum (int):  figure number
        sel_aids (list):
        subtitle (str):
        annote (bool):

    CommandLine:
        python -m ibeis.viz.viz_name --test-show_name --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_name import *  # NOQA
        >>> ibs, nid, in_image, index_list = testdata_showname()
        >>> fnum = 0
        >>> sel_aids = []
        >>> subtitle = ''
        >>> annote = False
        >>> # execute function
        >>> show_name(ibs, nid, in_image, fnum, sel_aids, subtitle, annote, index_list=index_list)
        >>> ut.show_if_requested()
    """
    print('[viz_name] show_name nid=%r, index_list=%r, aid_list=%r' % (nid, index_list, aid_list))
    if aid_list is None:
        aid_list = ibs.get_name_aids(nid)
    else:
        assert ut.list_all_eq_to(ibs.get_annot_nids(aid_list), nid)

    if index_list is not None:
        aid_list = ut.list_take(aid_list, index_list)

    name = ibs.get_name_texts((nid,))
    print('[viz_name] * name=%r aid_list=%r' % (name, aid_list))

    show_multiple_chips(ibs, aid_list, in_image=in_image, fnum=fnum,
                        sel_aids=sel_aids, annote=annote, **kwargs)

    if isinstance(nid, np.ndarray):
        nid = nid[0]
    if isinstance(name, np.ndarray):
        name = name[0]

    use_figtitle = not ut.get_argflag('--no-figtitle')

    if use_figtitle:
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
