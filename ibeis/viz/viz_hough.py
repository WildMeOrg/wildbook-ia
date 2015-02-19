from __future__ import absolute_import, division, print_function
import utool
from ibeis.viz import viz_helpers as vh
from vtool import image as gtool
from ibeis.model.detect import randomforest
from os.path import splitext
from plottool import viz_image2
from plottool import draw_func2 as df2
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_hough]', DEBUG=False)


@utool.indent_func
def show_hough_image(ibs, gid, species=None, fnum=None, **kwargs):
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Hough Image: ' + vh.get_image_titles(ibs, gid)
    print(title)

    if species is None:
        species = ibs.cfg.detect_cfg.species_text
    src_gpath_list = ibs.get_image_detectpaths([gid])
    dst_gpath_list = [splitext(gpath)[0] for gpath in src_gpath_list]
    hough_gpath_list = [gpath + '_' + species + '_hough.png' for gpath in dst_gpath_list]
    # Detect with hough
    config = {
        'output_gpath_list': hough_gpath_list,
    }
    results_list = list(randomforest.detect_gpath_list_with_species(ibs, src_gpath_list, species, **config))  # NOQA
    # Get path
    hough_gpath = hough_gpath_list[0]
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
        >>> from ibeis.viz import viz_hough  # NOQA
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
    hough_cpath = ibs.get_annot_probchip_fpaths(aid)
    img = gtool.imread(hough_cpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax
