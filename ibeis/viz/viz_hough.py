from __future__ import absolute_import, division, print_function
import utool
from ibeis.viz import viz_helpers as vh
from vtool import image as gtool
from ibeis.model.detect import randomforest
from os.path import splitext, exists
from plottool import viz_image2
from plottool import draw_func2 as df2
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_hough]', DEBUG=False)


@utool.indent_func
def show_hough_image(ibs, gid, species, fnum=None, **kwargs):
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Hough Image: ' + vh.get_image_titles(ibs, gid)
    print(title)

    src_gpath_list = list(map(str, ibs.get_image_detectpaths([gid])))
    dst_gpath_list = [splitext(gpath)[0] for gpath in src_gpath_list]
    hough_gpath_list = [gpath + '_hough' for gpath in dst_gpath_list]
    randomforest.compute_hough_images(ibs, src_gpath_list, hough_gpath_list, species)
    hough_gpath = hough_gpath_list[0] + '.png'
    img = gtool.imread(hough_gpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax


@utool.indent_func
def show_probability_chip(ibs, cid, species, fnum=None, **kwargs):
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Hough Chip: ' + ', '.join(vh.get_annot_text(ibs, [cid], True))
    print(title)

    src_cpath_list = list(map(str, ibs.get_chip_paths([cid])))
    dst_cpath_list = [splitext(gpath)[0] for gpath in src_cpath_list]
    hough_cpath_list = [gpath + '_hough' for gpath in dst_cpath_list]
    randomforest.compute_probability_images(ibs, src_cpath_list, hough_cpath_list, species)
    hough_cpath = hough_cpath_list[0] + '.png'
    img = gtool.imread(hough_cpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax
