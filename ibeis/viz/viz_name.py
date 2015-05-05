from __future__ import absolute_import, division, print_function
import plottool.draw_func2 as df2
import numpy as np
from ibeis import ibsfuncs
from plottool import plot_helpers as ph
import utool as ut
from . import viz_chip
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

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_1348" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_1348.jpg --dpath figures --caption='viewpoint issue different and viewing conditions' --figsize=11,3 --no-figtitle --notitle

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_1421" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_1421.jpg --dpath figures --caption='Pose issues where the zebras are fighting'

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_1366" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_1366.jpg --dpath figures --caption='Occlusion from another animal, blurry photo'

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_1288" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_1288.jpg --dpath figures --caption='Occlusion from another animal, blurry photo'

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_0370" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_0370.jpg --dpath figures --caption='viewpoint'

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_0453" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_0453.jpg --dpath figures --caption='Viewpoint minor occlusion'

        python -m ibeis.viz.viz_name --test-show_name --name="IBEIS_PZ_0303" --db testdb3 --save ~/latex/crall-candidacy-2015/figures/IBEIS_PZ_0303.jpg --dpath figures --caption='Shadowed'

        python -m ibeis.viz.viz_name --test-show_name --name=IBEIS_PZ_0884 --show --db NNP_Master3 --adjust=[.02,.02,.02] --notitle --index_list=[1,4,5,6] --rc=1,4

        --left=.02 --right=.98 --top=.98 --bottom=.02 --wspace=.05 --hspace=.05

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
    ibsfuncs.ensure_annotation_data(ibs, aid_list, chips=(not in_image or annote), feats=annote)
    print('[viz_name] * name=%r aid_list=%r' % (name, aid_list))
    nAids = len(aid_list)
    if nAids > 0:
        rc = ut.get_argval('--rc', type_=list, default=None)
        if rc is None:
            nRows, nCols = ph.get_square_row_cols(nAids)
        else:
            nRows, nCols = rc
        #print('[viz_name] * r=%r, c=%r' % (nRows, nCols))
        #gs2 = gridspec.GridSpec(nRows, nCols)
        pnum_ = df2.get_pnum_func(nRows, nCols)
        fig = df2.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
        fig.clf()
        # Trigger computation of all chips in parallel
        for px, aid in enumerate(aid_list):
            notitle = ut.get_argflag('--notitle')
            show_chip_kw = dict(annote=annote, in_image=in_image, notitle=notitle)
            viz_chip.show_chip(ibs, aid=aid, pnum=pnum_(px), **show_chip_kw)
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
