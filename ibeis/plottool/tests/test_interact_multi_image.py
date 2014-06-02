#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from plottool import interact_multi_image
from plottool import draw_func2 as df2
import utool


def test_interact_multimage(imgpaths):
    iteract_obj = interact_multi_image.MultiImageInteraction(imgpaths,
                                                             max_per_page=6)
    return iteract_obj

if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True)   # test image paths
    iteract_obj = test_interact_multimage(imgpaths)
    exec(df2.present())
