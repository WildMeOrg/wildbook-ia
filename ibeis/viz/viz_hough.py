from __future__ import absolute_import, division, print_function
import utool
from ibeis.viz import viz_helpers as vh
from vtool import image as gtool
from ibeis.model.detect import randomforest
from plottool import viz_image2
from plottool import draw_func2 as df2
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_hough]', DEBUG=False)


@utool.indent_func
def show_hough(ibs, gid, species, fnum=None, **kwargs):
    if fnum is None:
        fnum = df2.next_fnum()
    title = 'Hough Image: ' + vh.get_image_titles(ibs, gid)
    print(title)
    hough_gpath = randomforest.get_image_hough_gpaths(ibs, [gid], species)[0]
    img = gtool.imread(hough_gpath)
    fig, ax = viz_image2.show_image(img, title=title, fnum=fnum, **kwargs)
    return fig, ax
