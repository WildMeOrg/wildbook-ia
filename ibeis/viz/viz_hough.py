from __future__ import absolute_import, division, print_function
import utool
from ibeis.viz import viz_helpers as vh
from vtool import image as gtool
from ibeis.model.detect import randomforest
from os.path import splitext
from plottool import viz_image2
from plottool import draw_func2 as df2
from six.moves import map
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_hough]', DEBUG=False)


@utool.indent_func
def show_hough_image(ibs, gid, species=None, fnum=None, **kwargs):
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Hough Image: ' + vh.get_image_titles(ibs, gid)
    print(title)

    if species is None:
        species = ibs.cfg.detect_cfg.species
    use_chunks = ibs.cfg.other_cfg.detect_use_chunks

    src_gpath_list = list(map(str, ibs.get_image_detectpaths([gid])))
    dst_gpath_list = [splitext(gpath)[0] for gpath in src_gpath_list]
    hough_gpath_list = [gpath + '_hough' for gpath in dst_gpath_list]
    randomforest.compute_hough_images(src_gpath_list, hough_gpath_list, species, use_chunks=use_chunks)
    # HACK: THIS SHOULD BE DONE PREVIOUSLY NOT IN PYRF
    hough_gpath = hough_gpath_list[0] + '.png'
    img = gtool.imread(hough_gpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax


@utool.indent_func
def show_probability_chip(ibs, aid, species=None, fnum=None, **kwargs):
    """
    TODO: allow species override in controller

    Example:
        >>> from ibeis.viz.viz_hough import *  # NOQA
        >>> import ibeis
        >>> from ibeis.viz import viz_hough
        >>> ibs = ibeis.opendb('testdb1')
        >>> fnum = 1
        >>> species = None
        >>> aid = ibs.get_valid_aids()[0]
        >>> kwargs = {}
        >>> fig, ax = show_probability_chip(ibs, aid, species, fnum, **kwargs)
        >>> df2.present()
    """
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Probability Chip: ' + ', '.join(vh.get_annot_text(ibs, [aid], True))
    print(title)

    OLD_WAY = True
    if OLD_WAY:
        if species is None:
            species = ibs.cfg.detect_cfg.species
        use_chunks = ibs.cfg.other_cfg.detect_use_chunks

        src_cpath_list = list(map(str, ibs.get_annot_chip_paths([aid])))
        dst_cpath_list = [splitext(gpath)[0] for gpath in src_cpath_list]
        hough_cpath_list = [gpath + '_hough' for gpath in dst_cpath_list]
        randomforest.compute_probability_images(src_cpath_list, hough_cpath_list, species, use_chunks=use_chunks)
        # HACK: THIS SHOULD BE DONE PREVIOUSLY NOT IN PYRF
        hough_cpath = hough_cpath_list[0] + '.png'
    else:
        # Yay abstractions
        hough_cpath = ibs.get_annot_probchip_fpaths(aid)

    img = gtool.imread(hough_cpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax
