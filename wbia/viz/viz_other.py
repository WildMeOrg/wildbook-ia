# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import wbia.plottool as pt

(print, rrr, profile) = ut.inject2(__name__, '[viz_chip]')


def chip_montage(ibs, qaids, config=None):
    r"""
    CommandLine:
        python -m wbia.viz.viz_other chip_montage --show --db PZ_MTEST
        python -m wbia.viz.viz_other chip_montage --show --db GZ_ALL

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_other import *  # NOQA
        >>> defaltdb = 'seaturtles'
        >>> import wbia
        >>> a = ['default']
        >>> ibs = wbia.opendb(defaultdb=defaltdb)
        >>> ibs, qaids, daids = wbia.testdata_expanded_aids(ibs=ibs, a=a)
        >>> config = None
        >>> chip_montage(ibs, qaids, config)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import vtool as vt

    chip_list = ibs.get_annot_chips(qaids, config2_=config)
    height = 2000
    dsize = (int(height * ut.PHI), height)
    dst = vt.montage(chip_list, dsize)
    pt.imshow(dst)


def image_montage(ibs, gids, config=None):
    r"""
    CommandLine:
        python -m wbia.viz.viz_other image_montage --show --db PZ_Master1
        python -m wbia.viz.viz_other image_montage --show --db GZ_ALL

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_other import *  # NOQA
        >>> defaltdb = 'seaturtles'
        >>> import wbia
        >>> a = ['default']
        >>> ibs = wbia.opendb(defaultdb=defaltdb)
        >>> ibs, qaids, daids = wbia.testdata_expanded_aids(ibs=ibs, a=a)
        >>> config = None
        >>> gids = ibs.get_annot_gids(qaids[0:1000])
        >>> image_montage(ibs, gids, config)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import vtool as vt

    img_list = ibs.get_images(gids, config2_=config)
    height = 2000
    dsize = (int(height * ut.PHI), height)
    dst = vt.montage(img_list, dsize)
    pt.imshow(dst)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.viz.viz_other
        python -m wbia.viz.viz_other --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
