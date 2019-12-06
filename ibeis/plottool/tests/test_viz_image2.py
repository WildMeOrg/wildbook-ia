#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from plottool_ibeis import viz_image2
from plottool_ibeis import draw_func2 as df2
import cv2
import utool
import numpy as np
from plottool_ibeis.tests.test_helpers import dummy_bbox


def test_viz_image(img_fpath):
    # Read image
    img = cv2.imread(img_fpath)
    tau = np.pi * 2  # References: tauday.com
    # Create figure
    fig = df2.figure(fnum=42, pnum=(1, 1, 1))
    # Clear figure
    fig.clf()
    # Build parameters
    bbox_list = [dummy_bbox(img), dummy_bbox(img, (-.25, -.25), .1)]
    showkw = {
        'title'      : 'test axis title',
        # The list of bounding boxes to be drawn on the image
        'bbox_list'  : bbox_list,
        'theta_list' : [tau * .7, tau * .9],
        'sel_list'   : [True, False],
        'label_list' : ['test label', 'lbl2'],
    }
    # Print the keyword arguments to illustrate their format
    print('showkw = ' + utool.repr2(showkw))
    # Display the image in figure-num 42, using a 1x1 axis grid in the first
    # axis. Pass showkw as keyword arguments.
    viz_image2.show_image(img, fnum=42, pnum=(1, 1, 1), **showkw)
    df2.set_figtitle('Test figure title')


if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://cthulhu.dyn.wildme.io/public/data/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True, recursive=False)   # test image paths
    # Get one image filepath to load and display
    img_fpath = imgpaths[0]
    # Run Test
    test_viz_image(img_fpath)
    # Magic exec which displays or puts you into IPython with --cmd flag
    exec(df2.present())
