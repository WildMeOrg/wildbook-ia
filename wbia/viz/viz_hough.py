# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
from wbia.viz import viz_helpers as vh
from wbia.algo.detect import randomforest
from os.path import splitext
from wbia.plottool import viz_image2
import wbia.plottool as pt

(print, rrr, profile) = ut.inject2(__name__)


def show_hough_image(ibs, gid, species=None, fnum=None, **kwargs):
    if fnum is None:
        fnum = pt.next_fnum()
    title = 'Hough Image: ' + vh.get_image_titles(ibs, gid)
    print(title)

    if species is None:
        species = ibs.get_
        # .cfg.detect_cfg.species_text
    src_gpath_list = ibs.get_image_detectpaths([gid])
    dst_gpath_list = [splitext(gpath)[0] for gpath in src_gpath_list]
    hough_gpath_list = [gpath + '_' + species + '_hough.png' for gpath in dst_gpath_list]
    # Detect with hough
    config = {
        'output_gpath_list': hough_gpath_list,
    }
    print('-' * 80)
    print('')
    print('WARNING!!!')
    print('Hough image is not used often and not worth putting into depcache.')
    print('This image is computed as needed and not cached to disk.')
    print('')
    print('-' * 80)
    results_list = list(  # NOQA
        randomforest.detect_gpath_list_with_species(
            ibs, src_gpath_list, species, **config
        )
    )
    # Get path
    hough_gpath = hough_gpath_list[0]
    img = vt.imread(hough_gpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax


def show_probability_chip(
    ibs, aid, species=None, fnum=None, config2_=None, blend=False, **kwargs
):
    """
    TODO: allow species override in controller

    CommandLine:
        python -m wbia.viz.viz_hough --exec-show_probability_chip --cnn --show
        python -m wbia.viz.viz_hough --exec-show_probability_chip --cnn --show --db PZ_Master1
        python -m wbia.viz.viz_hough --exec-show_probability_chip --cnn --show --db PZ_Master1 --aid 9970

    Example:
        >>> # SCRIPT
        >>> from wbia.viz.viz_hough import *  # NOQA
        >>> import wbia
        >>> from wbia.viz import viz_chip
        >>> ibs, aid_list, kwargs, config2_ = viz_chip.testdata_showchip()
        >>> fnum = 1
        >>> species = None
        >>> aid = aid_list[0]
        >>> fig, ax = show_probability_chip(ibs, aid, species, fnum, blend=True, **kwargs)
        >>> ut.show_if_requested()
    """
    fnum = pt.ensure_fnum(fnum)
    title = 'Probability Chip: ' + ', '.join(vh.get_annot_text(ibs, [aid], True))
    hough_cpath = ibs.get_annot_probchip_fpath(aid, config2_=config2_)
    img = vt.imread(hough_cpath)
    if blend:
        chip = ibs.get_annot_chips(aid, config2_=config2_)
        img = vt.blend_images_multiply(chip, vt.resize_mask(img, chip))
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.viz.viz_hough
        python -m wbia.viz.viz_hough --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
