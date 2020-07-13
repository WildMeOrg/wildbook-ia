# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import utool as ut
import wbia.plottool as pt
from wbia.plottool import plot_helpers as ph
from wbia.plottool import viz_image2
import numpy as np
from wbia.viz import viz_helpers as vh

(print, rrr, profile) = ut.inject2(__name__, '[viz_img]')


def draw_image_overlay(ibs, ax, gid, sel_aids, draw_lbls=True, annote=True):
    try:
        raise NotImplementedError('use pt.viz_image2.draw_image_overlay')
        # draw chips in the image
        aid_list = ibs.get_image_aids(gid)
        bbox_list = ibs.get_annot_bboxes(aid_list)
        theta_list = ibs.get_annot_thetas(aid_list)
        text_list = vh.get_annot_text(ibs, aid_list, draw_lbls)
        annotation_centers = vh.get_bbox_centers(bbox_list)
        sel_list = [aid in sel_aids for aid in aid_list]

        viz_image2.draw_image_overlay(
            ax, bbox_list, theta_list, text_list, sel_list, draw_lbls, annote
        )
        # Draw all chip indexes in the image
        if annote:
            annotation_iter = zip(bbox_list, theta_list, text_list, sel_list)
            for bbox, theta, lbl, is_sel in annotation_iter:
                viz_image2.draw_chip_overlay(ax, bbox, theta, lbl, is_sel)
        # Put annotation centers in the axis
        ph.set_plotdat(ax, 'annotation_centers', np.array(annotation_centers))
        ph.set_plotdat(ax, 'annotation_bbox_list', bbox_list)
        ph.set_plotdat(ax, 'aid_list', aid_list)
    except Exception as ex:
        ut.printex(
            ex, 'error drawing image overlay', key_list=['ibs', 'ax', 'gid', 'sel_aids']
        )
        raise


def get_annot_annotations(ibs, aid_list, sel_aids=[], draw_lbls=True):
    annotekw = {
        'bbox_list': ibs.get_annot_bboxes(aid_list),
        'theta_list': ibs.get_annot_thetas(aid_list),
        'text_list': vh.get_annot_text(ibs, aid_list, draw_lbls),
        'sel_list': [aid in sel_aids for aid in aid_list],
    }
    return annotekw


def drive_test_script(ibs):
    r"""
    Test script where we drive around and take pictures of animals
    both in a given database and not in a given databse to make sure
    the system works.

    CommandLine:
        python -m wbia.viz.viz_image --test-drive_test_script
        python -m wbia.viz.viz_image --test-drive_test_script --db PZ_MTEST --show
        python -m wbia.viz.viz_image --test-drive_test_script --db GIR_Tanya --show
        python -m wbia.viz.viz_image --test-drive_test_script --db GIR_Master0 --show
        python -m wbia.viz.viz_image --test-drive_test_script --db PZ_Master0 --show
        python -m wbia.viz.viz_image --test-drive_test_script --db PZ_FlankHack --show

        python -m wbia.viz.viz_image --test-drive_test_script --db PZ_FlankHack --show
        python -m wbia.viz.viz_image --test-drive_test_script --dbdir /raid/work2/Turk/GIR_Master --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_image import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb()
        >>> drive_test_script(ibs)
    """
    import wbia

    aid_list = wbia.testdata_aids(a='default:pername=1')
    print('Running with (annot) aid_list = %r' % (aid_list))
    gid_list = ibs.get_annot_gids(aid_list)
    print('Running with (image) gid_list = %r' % (gid_list))
    avuuid_list = ibs.get_annot_visual_uuids(aid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    print(
        'Running with annot_visual_uuid_list = %s'
        % (ut.repr2(zip(aid_list, avuuid_list)))
    )
    print('Running with image_uuid_list = %s' % (ut.repr2(zip(gid_list, guuid_list))))
    for gid, aid in ut.ProgressIter(zip(gid_list, aid_list), lbl='progress '):
        print('\ngid, aid, nid = %r, %r, %r' % (gid, aid, ibs.get_annot_nids(aid),))
        show_image(ibs, gid, annote=False, rich_title=True)
        pt.show_if_requested()


def show_multi_images(ibs, gid_list, fnum=None, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):
        fnum (int):  figure number(default = None)

    CommandLine:
        python -m wbia.viz.viz_image --test-show_multi_images --db NNP_Master3 --gids=7409,7448,4670,7497,7496,7464,7446,7442 --show
        python -m wbia.viz.viz_image --test-show_multi_images --db NNP_Master3 --gids=1,2,3 --show

    Ignore:
        >>> # print to 8 gids sorted by num aids
        >>> import wbia
        >>> ibs = wbia.opendb('NNP_Master3')
        >>> gid_list = ibs.get_valid_gids()
        >>> aids_list = ibs.get_image_aids(gid_list)
        >>> index_list = ut.list_argsort(list(map(len, aids_list)))[::-1]
        >>> gid_list = ut.take(gid_list, index_list[0:8])
        >>> print(','.join(map(str, gid_list)))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_image import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> gid_list = ut.get_argval('--gids', list, default=[1, 2])
        >>> fnum = None
        >>> result = show_multi_images(ibs, gid_list, fnum, draw_lbls=False, notitle=True, sel_aids='all')
        >>> print(result)
        >>> ut.show_if_requested()
    """
    fnum = pt.ensure_fnum(fnum)
    nGids = len(gid_list)
    if nGids == 0:
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), **kwargs)
        pt.imshow_null(fnum=fnum, **kwargs)
        return fig
    # Trigger computation of all chips in parallel
    # ibsfuncs.ensure_annotation_data(ibs, aid_list, chips=(not in_image or annote), feats=annote)

    rc = ut.get_argval('--rc', type_=list, default=None)
    if rc is None:
        nRows, nCols = ph.get_square_row_cols(nGids)
    else:
        nRows, nCols = rc
    # notitle = ut.get_argflag('--notitle')
    # draw_lbls = not ut.get_argflag('--no-draw_lbls')
    # show_chip_kw = dict(annote=annote, in_image=in_image, notitle=notitle, draw_lbls=draw_lbls)
    # print('[viz_name] * r=%r, c=%r' % (nRows, nCols))
    # gs2 = gridspec.GridSpec(nRows, nCols)
    pnum_ = pt.get_pnum_func(nRows, nCols)
    fig = pt.figure(fnum=fnum, pnum=pnum_(0), **kwargs)
    fig.clf()
    for px, gid in enumerate(gid_list):
        print(pnum_(px))
        _fig, _ax1 = show_image(ibs, gid, fnum=fnum, pnum=pnum_(px), **kwargs)
        # ax = pt.gca()
        # if aid in sel_aids:
        #    pt.draw_border(ax, pt.GREEN, 4)
        # if ut.get_argflag('--numlbl') and not DOBOTH:
        #    ax.set_xlabel('(' + str(px + 1) + ')')
        # plot_aid3(ibs, aid)
    pass


# @ut.indent_func
def show_image(
    ibs,
    gid,
    sel_aids=[],
    fnum=None,
    annote=True,
    draw_lbls=True,
    notitle=False,
    rich_title=False,
    pnum=(1, 1, 1),
    **kwargs,
):
    """
    Driver function to show images

    Args:
        ibs (IBEISController):  wbia controller object
        gid (int): image row id
        sel_aids (list):
        fnum (int):  figure number
        annote (bool):
        draw_lbls (bool):

    Returns:
        tuple: (fig, ax)

    CommandLine:
        python -m wbia.viz.viz_image --test-show_image --show
        python -m wbia.viz.viz_image --test-show_image --show --db GZ_ALL
        python -m wbia.viz.viz_image --test-show_image --show --db GZ_ALL --gid 100
        python -m wbia.viz.viz_image --test-show_image --show --db PZ_MTEST --aid 10

        python -m wbia.viz.viz_image --test-show_image --show --db PZ_MTEST --aid 91 --no-annot --rich-title
        python -m wbia.viz.viz_image --test-show_image --show --db GIR_Tanya --aid 1 --no-annot --rich-title

    Example:
        >>> # SLOW_DOCTEST
        >>> # VIZ_TEST
        >>> from wbia.viz.viz_image import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb(ut.get_argval('--db', str, 'testdb1'))
        >>> #gid = ibs.get_valid_gids()[0]
        >>> gid = ut.get_argval('--gid', int, 1)
        >>> aid = ut.get_argval('--aid', int, None)
        >>> if aid is not None:
        >>>    gid = ibs.get_annot_gids(aid)
        >>> sel_aids = []
        >>> fnum = None
        >>> annote = not ut.get_argflag('--no-annot')
        >>> rich_title = ut.get_argflag('--rich-title')
        >>> draw_lbls = True
        >>> # execute function
        >>> (fig, ax) = show_image(ibs, gid, sel_aids, fnum, annote, draw_lbls, rich_title)
        >>> pt.show_if_requested()
    """
    if fnum is None:
        fnum = pt.next_fnum()
    # Read Image
    img = ibs.get_images(gid)
    aid_list = ibs.get_image_aids(gid)
    if sel_aids == 'all':
        sel_aids = aid_list
    annotekw = get_annot_annotations(ibs, aid_list, sel_aids, draw_lbls)
    annotation_centers = vh.get_bbox_centers(annotekw['bbox_list'])
    title = vh.get_image_titles(ibs, gid)
    if rich_title:
        title += ', aids=%r' % (aid_list)
        title += ', db=%r' % (ibs.get_dbname())
    showkw = {
        'title': title,
        'annote': annote,
        'fnum': fnum,
        'pnum': pnum,
    }
    if notitle:
        del showkw['title']
    showkw.update(annotekw)
    fig, ax = viz_image2.show_image(img, **showkw)
    # Label the axis with data
    ph.set_plotdat(ax, 'annotation_centers', annotation_centers)
    ph.set_plotdat(ax, 'aid_list', aid_list)
    ph.set_plotdat(ax, 'viztype', 'image')
    ph.set_plotdat(ax, 'gid', gid)
    return fig, ax


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_image
        python -m wbia.viz.viz_image --allexamples
        python -m wbia.viz.viz_image --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
