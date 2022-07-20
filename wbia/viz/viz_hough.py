# -*- coding: utf-8 -*-
import logging
import utool as ut
import vtool as vt
from wbia.viz import viz_helpers as vh
from wbia.plottool import viz_image2
import wbia.plottool as pt

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


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
