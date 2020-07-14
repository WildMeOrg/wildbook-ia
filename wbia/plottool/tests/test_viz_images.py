#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.plottool import viz_image2
from wbia.plottool import draw_func2 as df2
from wbia.plottool import plot_helpers as ph
import utool
import numpy as np
from wbia.plottool.tests.test_helpers import dummy_bbox, imread_many


def _test_viz_image(imgpaths):
    nImgs = len(imgpaths)
    assert len(imgpaths) < 20, '%d > 20 out of scope of this test' % nImgs
    tau = np.pi * 2
    fnum = 1
    img_list = imread_many(imgpaths)
    nRows, nCols = ph.get_square_row_cols(nImgs)
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    # gs2 = gridspec.GridSpec(nRows, nCols)
    pnum_ = df2.get_pnum_func(nRows, nCols)

    fig = df2.figure(fnum=fnum, pnum=pnum_(0))
    fig.clf()
    for px, img in enumerate(img_list):
        title = 'test title'
        bbox_list = [dummy_bbox(img), dummy_bbox(img, (-0.25, -0.25), 0.1)]
        theta_list = [tau * 0.7, tau * 0.9]
        sel_list = [True, False]
        label_list = ['test label', 'lbl2']
        viz_image2.show_image(
            img,
            bbox_list=bbox_list,
            title=title,
            sel_list=sel_list,
            label_list=label_list,
            theta_list=theta_list,
            fnum=fnum,
            pnum=pnum_(px),
        )


if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://wildbookiarepository.azureedge.net/data/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths = utool.list_images(test_image_dir, fullpath=True)  # test image paths
    _test_viz_image(imgpaths)
    exec(df2.present())
