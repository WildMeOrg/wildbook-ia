#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DUPLICATE CODE, DELETE
from __future__ import absolute_import, division, print_function
from wbia.plottool import interact_multi_image
from wbia.plottool import draw_func2 as df2
import utool

# import wbia


def _test_interact_multimage(imgpaths):
    print('len: ', len(imgpaths))
    bboxes_list = [[]] * len(imgpaths)

    bboxes_list[0] = [(-200, -100, 400, 400)]
    print(bboxes_list)
    iteract_obj = interact_multi_image.MultiImageInteraction(
        imgpaths, nPerPage=4, bboxes_list=bboxes_list
    )
    # def test_interact_multimage(imgpaths, gid_list=None, aids_list=None, bboxes_list=None):
    #     img_list = imread_many(imgpaths)
    #     iteract_obj = interact_multi_image.MultiImageInteraction(img_list +
    #                                                              img_list,
    #                                                              gid_list, aids_list, bboxes_list,
    #                                                              nPerPage=6)
    return iteract_obj


if __name__ == '__main__':
    TEST_IMAGES_URL = 'https://wildbookiarepository.azureedge.net/data/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths = utool.list_images(
        test_image_dir, fullpath=True, recursive=False
    )  # test image paths
    iteract_obj = _test_interact_multimage(imgpaths)
    exec(df2.present())
