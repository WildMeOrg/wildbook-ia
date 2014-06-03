#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from plottool import viz_image2
from plottool import draw_func2 as df2
import cv2
import utool
import numpy as np
from plottool.tests.test_helpers import dummy_bbox


def test_viz_image(img_fpath):
    # Read image
    img = cv2.imread(img_fpath)
    tau = np.pi * 2  # tauday.com
    # Create figure
    fig = df2.figure(fnum=1, pnum=(1, 1, 1))
    # Clear figure
    fig.clf()
    # Build parameters
    showkw = {
        'title'      : 'test title',
        'bbox_list'  : [dummy_bbox(img),
                        dummy_bbox(img, (-.25, -.25), .1)],
        'theta_list' : [tau * .7, tau * .9],
        'sel_list'   : [True, False],
        'label_list' : ['test label', 'lbl2'],
    }
    viz_image2.show_image(img, fnum=1, pnum=(1, 1, 1), **showkw)


if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True)   # test image paths
    img_fpath = imgpaths[0]
    test_viz_image(img_fpath)
    exec(df2.present())
