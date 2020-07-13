# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import wbia.plottool.draw_func2 as df2
import numpy as np
from wbia.other import ibsfuncs
from wbia.plottool import plot_helpers as ph
import wbia.plottool as pt
import utool as ut
from wbia.viz import viz_chip

(print, rrr, profile) = ut.inject2(__name__)


def show_name_of(ibs, aid, **kwargs):
    nid = ibs.get_annot_names(aid)
    show_name(ibs, nid, sel_aids=[aid], **kwargs)


def testdata_showname():
    import wbia

    ibs = wbia.opendb(defaultdb='testdb1')
    default = None
    if ibs.dbname == 'testdb1':
        default = 'easy'

    name_text = ut.get_argval('--name', type_=str, default=default)
    if name_text is None:
        nid = 1
    else:
        nid = ibs.get_name_rowids_from_text(name_text)
    in_image = not ut.get_argflag('--no-inimage')
    index_list = ut.get_argval('--index_list', type_=list, default=None)
    return ibs, nid, in_image, index_list


def testdata_multichips():
    import wbia

    ibs = wbia.opendb(defaultdb='testdb1')
    nid = ut.get_argval('--nid', type_=int, default=None)
    tags = ut.get_argval('--tags', type_=list, default=None)

    if nid is not None:
        aid_list = ibs.get_name_aids(nid)
    elif tags is not None:
        index = ut.get_argval('--index', default=0)
        aid_list = ibs.filter_aidpairs_by_tags(any_tags=tags)[index]
    else:
        # aid_list = ut.get_argval('--aids', type_=list, default=[1, 2, 3])
        aid_list = wbia.testdata_aids(default_aids=[1, 2, 3], ibs=ibs)

    in_image = not ut.get_argflag('--no-inimage')
    return ibs, aid_list, in_image


# 9108 and 9180


def show_multiple_chips(
    ibs, aid_list, in_image=True, fnum=0, sel_aids=[], subtitle='', annote=False, **kwargs
):
    """
    CommandLine:
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --no-inimage
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=6435,9861,137,6563,9167,12547,9332,12598,13285 --no-inimage --notitle
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=137,6563,12547,9332,12598,13285 --no-inimage --notitle --adjust=.05
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --db NNP_Master3 --aids=6563,9332,13285,12598 --no-inimage --notitle --adjust=.05 --rc=1,4
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --db PZ_Master0 --aids=1288 --no-inimage --notitle --adjust=.05
        python -m wbia.viz.viz_name --test-show_multiple_chips --show --db PZ_Master0 --aids=4020,4839 --no-inimage --notitle --adjust=.05

        python -m wbia.viz.viz_name --test-show_multiple_chips --db NNP_Master3 --aids=6524,6540,6571,6751 --no-inimage --notitle --adjust=.05 --diskshow

        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST -a default:index=0:4 --show
        --aids=1 --doboth --show --no-inimage

        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST --aids=1 --doboth --show --no-inimage
        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST --aids=1 --doboth --rc=2,1 --show --no-inimage
        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST --aids=1 --doboth --rc=2,1 --show --notitle --trydrawline --no-draw_lbls
        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST --aids=1,2 --doboth  --show --notitle --trydrawline

        python -m wbia.viz.viz_name --test-show_multiple_chips --db PZ_MTEST --aids=1,2,3,4,5 --doboth --rc=2,5 --show --chrlbl --trydrawline --qualtitle --no-figtitle --notitle
        --doboth
        --doboth --show

        python -m wbia.viz.viz_name --test-show_multiple_chips --db NNP_Master3 --aids=15419 --doboth --rc=2,1 --show --notitle --trydrawline --no-draw_lbls

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_name import *  # NOQA
        >>> import wbia
        >>> ibs, aid_list, in_image = testdata_multichips()
        >>> if True:
        >>>     import matplotlib as mpl
        >>>     from wbia.scripts.thesis import TMP_RC
        >>>     mpl.rcParams.update(TMP_RC)
        >>> fnum = 0
        >>> sel_aids = []
        >>> subtitle = ''
        >>> annote = False
        >>> fig = show_multiple_chips(ibs, aid_list, in_image, fnum, sel_aids, subtitle, annote)
        >>> ut.quit_if_noshow()
        >>> fig.canvas.draw()
        >>> ut.show_if_requested()
    """
    fnum = pt.ensure_fnum(fnum)
    nAids = len(aid_list)
    if nAids == 0:
        fig = df2.figure(fnum=fnum, pnum=(1, 1, 1), **kwargs)
        df2.imshow_null(fnum=fnum, **kwargs)
        return fig
    # Trigger computation of all chips in parallel
    ibsfuncs.ensure_annotation_data(
        ibs, aid_list, chips=(not in_image or annote), feats=annote
    )

    print('[viz_name] * annot_vuuid=%r' % ((ibs.get_annot_visual_uuids(aid_list),)))
    print('[viz_name] * aid_list=%r' % ((aid_list,)))

    DOBOTH = ut.get_argflag('--doboth')

    rc = ut.get_argval('--rc', type_=list, default=None)
    if rc is None:
        nRows, nCols = ph.get_square_row_cols(nAids * (2 if DOBOTH else 1))
    else:
        nRows, nCols = rc
    notitle = ut.get_argflag('--notitle')
    draw_lbls = not ut.get_argflag('--no-draw_lbls')
    show_chip_kw = dict(
        annote=annote, in_image=in_image, notitle=notitle, draw_lbls=draw_lbls
    )
    # print('[viz_name] * r=%r, c=%r' % (nRows, nCols))
    # gs2 = gridspec.GridSpec(nRows, nCols)
    pnum_ = df2.get_pnum_func(nRows, nCols)
    fig = df2.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
    fig.clf()
    ax_list1 = []
    for px, aid in enumerate(aid_list):
        print('px = %r' % (px,))
        _fig, _ax1 = viz_chip.show_chip(ibs, aid=aid, pnum=pnum_(px), **show_chip_kw)
        print('other_aids = %r' % (ibs.get_annot_contact_aids(aid),))
        ax = df2.gca()
        ax_list1.append(_ax1)
        if aid in sel_aids:
            df2.draw_border(ax, df2.GREEN, 4)
        if ut.get_argflag('--chrlbl') and not DOBOTH:
            ax.set_xlabel('(' + chr(ord('a') - 1 + px) + ')')
        elif ut.get_argflag('--numlbl') and not DOBOTH:
            ax.set_xlabel('(' + str(px + 1) + ')')
        # plot_aid3(ibs, aid)

    # HACK to show in image and not in image
    if DOBOTH:
        # ut.embed()
        # ph.get_plotdat_dict(ax_list1[1])
        # ph.get_plotdat_dict(ax_list2[1])
        ax_list2 = []

        show_chip_kw['in_image'] = not show_chip_kw['in_image']
        start = px + 1
        for px, aid in enumerate(aid_list, start=start):
            _fig, _ax2 = viz_chip.show_chip(ibs, aid=aid, pnum=pnum_(px), **show_chip_kw)
            ax = df2.gca()
            ax_list2.append(_ax2)

            if ut.get_argflag('--chrlbl'):
                ax.set_xlabel('(' + chr(ord('a') - start + px) + ')')
            elif ut.get_argflag('--numlbl'):
                ax.set_xlabel('(' + str(px - start + 1) + ')')

            if ut.get_argflag('--qualtitle'):
                qualtext = ibs.get_annot_quality_texts(aid)
                ax.set_title(qualtext)

            if aid in sel_aids:
                df2.draw_border(ax, df2.GREEN, 4)

        if in_image:
            ax_list1, ax_list2 = ax_list2, ax_list1

        if ut.get_argflag('--trydrawline'):
            # Unfinished
            # ut.embed()
            # Draw lines between corresponding axes
            # References:
            # http://stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib
            import matplotlib as mpl
            import vtool as vt

            # !!!
            # http://matplotlib.org/users/transforms_tutorial.html

            # invTransFigure_fn1 = fig.transFigure.inverted().transform
            # invTransFigure_fn2 = fig.transFigure.inverted().transform
            # print(ax_list1)
            # print(ax_list2)
            assert len(ax_list1) == len(ax_list2)

            for ax1, ax2 in zip(ax_list1, ax_list2):
                # _ = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # bbox1 = (0, 0, _.width * fig.dpi, _.height * fig.dpi)

                # returns in figure coordinates
                # bbox1 = df2.get_axis_bbox(ax=ax1)
                # if bbox1[-1] < 0:
                #    # Weird bug
                #    bbox1 = bbox1[1]
                print('--')
                print('ax1 = %r' % (ax1,))
                print('ax2 = %r' % (ax2,))
                chipshape = ph.get_plotdat(ax1, 'chipshape')
                # _bbox1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # bbox1 = (0, 0, _bbox1.width * fig.dpi, _bbox1.height * fig.dpi)
                bbox1 = (0, 0, chipshape[1], chipshape[0])

                aid_ = ph.get_plotdat(ax2, 'aid')
                aid_list_ = ph.get_plotdat(ax2, 'aid_list')
                index = aid_list_.index(aid_)
                annotation_bbox_list = ph.get_plotdat(ax2, 'annotation_bbox_list')
                bbox2 = annotation_bbox_list[index]

                print('bbox1 = %r' % (bbox1,))
                print('bbox2 = %r' % (bbox2,))

                vert_list1 = np.array(vt.verts_from_bbox(bbox1))
                vert_list2 = np.array(vt.verts_from_bbox(bbox2))

                print('vert_list1 = %r' % (vert_list1,))
                print('vert_list2 = %r' % (vert_list2,))
                # for vx in [0, 1, 2, 3]:
                for vx in [0, 1]:
                    vert1 = vert_list1[vx].tolist()
                    vert2 = vert_list2[vx].tolist()
                    print('  ***')
                    print('  * vert1 = %r' % (vert1,))
                    print('  * vert2 = %r' % (vert2,))

                    coordsA = coordsB = 'data'
                    # coords = 'axes points'
                    # 'axes fraction'
                    # 'axes pixels'
                    # coordsA = 'axes pixels'
                    # coordsB = 'data'
                    # 'figure fraction'
                    # 'figure pixels'
                    # 'figure pixels'
                    # 'figure points'
                    # 'polar'
                    # 'offset points'

                    con = mpl.patches.ConnectionPatch(
                        xyA=vert1,
                        xyB=vert2,
                        coordsA=coordsA,
                        coordsB=coordsB,
                        axesA=ax1,
                        axesB=ax2,
                        linewidth=1,
                        color='k',
                    )
                    # , arrowstyle="-")

                    # ut.embed()
                    # con.set_zorder(None)
                    ax1.add_artist(con)
                    # ax2.add_artist(con)

                    # ut.embed()

                    # verts2.T[1] -= bbox2[-1]
                    # bottom_left1, bottom_right1 = verts1[1:3].tolist()
                    # bottom_left2, bottom_right2 = verts2[1:3].tolist()

                # #transAxes1 = ax1.transData.inverted()
                # transAxes1_fn = ax1.transData.transform
                # transAxes2_fn = ax2.transData.transform

                # transAxes1_fn = ut.identity
                # transAxes2_fn = ut.identity

                # coord_bl1 = transFigure.transform(transAxes1.transform(bottom_left1))
                # coord_br1 = transFigure.transform(transAxes1.transform(bottom_right1))
                # coord_bl1 = invTransFigure_fn1(transAxes1_fn(bottom_left1))
                # print('bottom_left2 = %r' % (bottom_left2,))
                # coord_bl1 = (5, 5)
                # coord_bl2 = invTransFigure_fn2(transAxes2_fn(bottom_left2))
                # print('coord_bl2 = %r' % (coord_bl2,))

                # coord_br1 = invTransFigure_fn1(transAxes1_fn(bottom_right1))
                # coord_br2 = invTransFigure_fn2(transAxes2_fn(bottom_right2))
                # #print('coord_bl1 = %r' % (coord_bl1,))

                # line_coords1 = np.vstack([coord_bl1, coord_bl2])
                # line_coords2 = np.vstack([coord_br1, coord_br2])
                # print('line_coords1 = %r' % (line_coords1,))

                # line1 = mpl.lines.Line2D((line_coords1[0]), (line_coords1[1]), transform=fig.transFigure)
                # line2 = mpl.lines.Line2D((line_coords2[0]), (line_coords2[1]), transform=fig.transFigure)

                # xs1, ys1 = line_coords1.T
                # xs2, ys2 = line_coords2.T

                # linekw = dict(transform=fig.transFigure)
                # linekw = dict()

                # print('xs1 = %r' % (xs1,))
                # print('ys1 = %r' % (ys1,))

                # line1 = mpl.lines.Line2D(xs1, ys1, **linekw)
                # line2 = mpl.lines.Line2D(xs2, ys2, **linekw)  # NOQA
                # shrinkA=5, shrinkB=5, mutation_scale=20, fc="w")

                # ax2.add_artist(con)

                # fig.lines.append(line1)
                # fig.lines.append(line2)

        pass
    return fig


# @ut.indent_func
def show_name(
    ibs,
    nid,
    in_image=True,
    fnum=0,
    sel_aids=[],
    subtitle='',
    annote=False,
    aid_list=None,
    index_list=None,
    **kwargs
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        nid (?):
        in_image (bool):
        fnum (int):  figure number
        sel_aids (list):
        subtitle (str):
        annote (bool):

    CommandLine:

        python -m wbia.viz.viz_name --test-show_name --dpath ~/latex/crall-candidacy-2015 --save 'figures/{name}.jpg' --no-figtitle --notitle --db NNP_Master3 --figsize=9,4 --clipwhite --dpi=180 --adjust=.05 --index_list=[0,1,2,3] --rc=2,4 --append temp_out_figure.tex --name=IBEIS_PZ_0739 --no-draw_lbls --doboth --no-inimage  --diskshow

        python -m wbia.viz.viz_name --test-show_name --no-figtitle --notitle --db NNP_Master3 --figsize=9,4 --clipwhite --dpi=180 --adjust=.05 --index_list=[0,1,2,3] --rc=2,4 --append temp_out_figure.tex --name=IBEIS_PZ_0739 --no-draw_lbls --doboth --no-inimage  --show

        python -m wbia.viz.viz_name --test-show_name --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_name import *  # NOQA
        >>> ibs, nid, in_image, index_list = testdata_showname()
        >>> fnum = 0
        >>> sel_aids = []
        >>> subtitle = ''
        >>> annote = False
        >>> # execute function
        >>> show_name(ibs, nid, in_image, fnum, sel_aids, subtitle, annote, index_list=index_list)
        >>> ut.show_if_requested()
    """
    print(
        '[viz_name] show_name nid=%r, index_list=%r, aid_list=%r'
        % (nid, index_list, aid_list)
    )
    if aid_list is None:
        aid_list = ibs.get_name_aids(nid)
    else:
        assert ut.list_all_eq_to(ibs.get_annot_nids(aid_list), nid)

    if index_list is not None:
        aid_list = ut.take(aid_list, index_list)

    name = ibs.get_name_texts((nid,))
    print('[viz_name] * name=%r aid_list=%r' % (name, aid_list))

    show_multiple_chips(
        ibs,
        aid_list,
        in_image=in_image,
        fnum=fnum,
        sel_aids=sel_aids,
        annote=annote,
        **kwargs
    )

    if isinstance(nid, np.ndarray):
        nid = nid[0]
    if isinstance(name, np.ndarray):
        name = name[0]

    use_figtitle = not ut.get_argflag('--no-figtitle')

    if use_figtitle:
        figtitle = 'Name View nid=%r name=%r' % (nid, name)
        df2.set_figtitle(figtitle)
    # if not annote:
    #    title += ' noannote'
    # gs2.tight_layout(fig)
    # gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    # df2.set_figtitle(title, subtitle)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_name
        python -m wbia.viz.viz_name --allexamples
        python -m wbia.viz.viz_name --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
