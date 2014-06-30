#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from plottool import interact_multi_image
from plottool import draw_func2 as df2
import utool
#import ibeis


def test_interact_multimage(imgpaths):
    print("len: ", len(imgpaths))
    bboxes_list = [[]] * len(imgpaths)

    bboxes_list[0] = [(-200, -100, 400, 400)]
    print(bboxes_list)
    iteract_obj = interact_multi_image.MultiImageInteraction(imgpaths, nPerPage=4, bboxes_list=bboxes_list)
# def test_interact_multimage(imgpaths, gid_list=None, aids_list=None, bboxes_list=None):
#     img_list = imread_many(imgpaths)
#     iteract_obj = interact_multi_image.MultiImageInteraction(img_list +
#                                                              img_list,
#                                                              gid_list, aids_list, bboxes_list,
#                                                              nPerPage=6)
    return iteract_obj

if __name__ == '__main__':
    # main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    # ibs = main_locals['ibs']
    # # List of imaGe-ids:
    # gid_list  = ibs.get_valid_gids()
    # # Get a list of lists of ANNOTATION-ids (because each image can have more than one ANNOTATION)
    # aids_list = ibs.get_image_aids(gid_list)
    # # Get the list of lists of bounding boxes
    # bboxes_list = [ibs.get_annotion_bboxes(aids) for aids in aids_list]

    # image_paths = ibs.get_image_paths(gid_list)
    # print("gid_list: ", gid_list)
    # print("aids_list", aids_list)
    # iteract_obj = test_interact_multimage(image_paths, gid_list, aids_list, bboxes_list)


    TEST_IMAGES_URL = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True, recursive=False)   # test image paths
    iteract_obj = test_interact_multimage(imgpaths)
    exec(df2.present())
