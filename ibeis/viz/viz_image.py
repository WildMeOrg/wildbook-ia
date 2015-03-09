from __future__ import absolute_import, division, print_function
from six.moves import zip
import utool as ut
import plottool as pt  # NOQA
import plottool.draw_func2 as df2
from plottool import viz_image2
import numpy as np
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[viz_img]', DEBUG=False)


def draw_image_overlay(ibs, ax, gid, sel_aids, draw_lbls=True, annote=True):
    try:
        # draw chips in the image
        aid_list    = ibs.get_image_aids(gid)
        bbox_list   = ibs.get_annot_bboxes(aid_list)
        theta_list  = ibs.get_annot_thetas(aid_list)
        text_list  = vh.get_annot_text(ibs, aid_list, draw_lbls)
        annotation_centers = vh.get_bbox_centers(bbox_list)
        sel_list    = [aid in sel_aids for aid in aid_list]

        viz_image2.draw_image_overlay(ax, bbox_list, theta_list, text_list, sel_list, draw_lbls, annote)
        # Draw all chip indexes in the image
        if annote:
            annotation_iter = zip(bbox_list, theta_list, text_list, sel_list)
            for bbox, theta, lbl, is_sel in annotation_iter:
                viz_image2.draw_chip_overlay(ax, bbox, theta, lbl, is_sel)
        # Put annotation centers in the axis
        vh.set_ibsdat(ax, 'annotation_centers', np.array(annotation_centers))
        vh.set_ibsdat(ax, 'aid_list', aid_list)
    except Exception as ex:
        ut.printex(ex, key_list=['ibs', 'ax', 'gid', 'sel_aids'])
        raise


def get_annot_annotations(ibs, aid_list, sel_aids=[], draw_lbls=True):
    annotekw = {
        'bbox_list'  : ibs.get_annot_bboxes(aid_list),
        'theta_list' : ibs.get_annot_thetas(aid_list),
        'text_list' : vh.get_annot_text(ibs, aid_list, draw_lbls),
        'sel_list'   : [aid in sel_aids for aid in aid_list],
    }
    return annotekw


def drive_test_script(ibs):
    r"""
    Test script where we drive around and take pictures of animals
    both in a given database and not in a given databse to make sure
    the system works.

    CommandLine:
        python -m ibeis.viz.viz_image --test-drive_test_script
        python -m ibeis.viz.viz_image --test-drive_test_script --db PZ_MTEST --show
        python -m ibeis.viz.viz_image --test-drive_test_script --db GIR_Tanya --show
        python -m ibeis.viz.viz_image --test-drive_test_script --db GIR_Master0 --show
        python -m ibeis.viz.viz_image --test-drive_test_script --db PZ_Master0 --show
        python -m ibeis.viz.viz_image --test-drive_test_script --db PZ_FlankHack --show

        python -m ibeis.viz.viz_image --test-drive_test_script --db PZ_FlankHack --show
        python -m ibeis.viz.viz_image --test-drive_test_script --dbdir /raid/work2/Turk/GIR_Master --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_image import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb()
        >>> drive_test_script(ibs)
    """
    aid_list = ibs.get_one_annot_per_name()
    print('Running with (annot) aid_list = %r' % (aid_list))
    gid_list = ibs.get_annot_gids(aid_list)
    print('Running with (image) gid_list = %r' % (gid_list))
    avuuid_list = ibs.get_annot_visual_uuids(aid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    print('Running with annot_visual_uuid_list = %s' % (ut.list_str(zip(aid_list, avuuid_list))))
    print('Running with image_uuid_list = %s' % (ut.list_str(zip(gid_list, guuid_list))))
    for gid, aid in ut.ProgressIter(zip(gid_list, aid_list), lbl='progress '):
        print('\ngid, aid, nid = %r, %r, %r' % (gid, aid, ibs.get_annot_nids(aid),))
        show_image(ibs, gid, annote=False, rich_title=True)
        pt.show_if_requested()


#@ut.indent_func
def show_image(ibs, gid, sel_aids=[], fnum=None,
               annote=True, draw_lbls=True, rich_title=False, **kwargs):
    """
    Driver function to show images

    Args:
        ibs (IBEISController):  ibeis controller object
        gid (int): image row id
        sel_aids (list):
        fnum (int):  figure number
        annote (bool):
        draw_lbls (bool):

    Returns:
        tuple: (fig, ax)

    CommandLine:
        python -m ibeis.viz.viz_image --test-show_image --show
        python -m ibeis.viz.viz_image --test-show_image --show --db GZ_ALL
        python -m ibeis.viz.viz_image --test-show_image --show --db GZ_ALL --gid 100
        python -m ibeis.viz.viz_image --test-show_image --show --db PZ_MTEST --aid 10

        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --db PZ_MTEST

        python -m ibeis.viz.viz_image --test-show_image --show --db PZ_MTEST --aid 91 --no-annot --rich-title
        python -m ibeis.viz.viz_image --test-show_image --show --db GIR_Tanya --aid 1 --no-annot --rich-title

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.viz.viz_image import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(ut.get_argval('--db', str, 'testdb1'))
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
        fnum = df2.next_fnum()
    # Read Image
    img = ibs.get_images(gid)
    aid_list = ibs.get_image_aids(gid)
    annotekw = get_annot_annotations(ibs, aid_list, sel_aids, draw_lbls)
    annotation_centers = vh.get_bbox_centers(annotekw['bbox_list'])
    title = vh.get_image_titles(ibs, gid)
    if rich_title:
        title += ', aids=%r' % (aid_list)
        title += ', db=%r' % (ibs.get_dbname())
    showkw = {
        'title'      : title,
        'annote'     : annote,
        'fnum'       : fnum,
    }
    showkw.update(annotekw)
    fig, ax = viz_image2.show_image(img, **showkw)
    # Label the axis with data
    vh.set_ibsdat(ax, 'annotation_centers', annotation_centers)
    vh.set_ibsdat(ax, 'aid_list', aid_list)
    vh.set_ibsdat(ax, 'viztype', 'image')
    vh.set_ibsdat(ax, 'gid', gid)
    return fig, ax

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_image
        python -m ibeis.viz.viz_image --allexamples
        python -m ibeis.viz.viz_image --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
