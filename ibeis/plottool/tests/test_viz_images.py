#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from plottool_ibeis import viz_image2
from plottool_ibeis import draw_func2 as df2
from plottool_ibeis import plot_helpers as ph
import utool
import numpy as np
from plottool_ibeis.tests.test_helpers import dummy_bbox, imread_many


def test_viz_image(imgpaths):
    nImgs = len(imgpaths)
    assert len(imgpaths) < 20, '%d > 20 out of scope of this test' % nImgs
    tau = np.pi * 2
    fnum = 1
    img_list = imread_many(imgpaths)
    nRows, nCols = ph.get_square_row_cols(nImgs)
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum_ = df2.get_pnum_func(nRows, nCols)

    fig = df2.figure(fnum=fnum, pnum=pnum_(0))
    fig.clf()
    for px, img in enumerate(img_list):
        title = 'test title'
        bbox_list = [dummy_bbox(img),
                     dummy_bbox(img, (-.25, -.25), .1)]
        theta_list = [tau * .7, tau * .9]
        sel_list = [True, False]
        label_list = ['test label', 'lbl2']
        viz_image2.show_image(img,
                              bbox_list=bbox_list,
                              title=title,
                              sel_list=sel_list,
                              label_list=label_list,
                              theta_list=theta_list,
                              fnum=fnum,
                              pnum=pnum_(px))


if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://cthulhu.dyn.wildme.io/public/data/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True)   # test image paths
    test_viz_image(imgpaths)
    exec(df2.present())
