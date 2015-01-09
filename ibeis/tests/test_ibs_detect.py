#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
import ibeis
import multiprocessing
#from ibeis.model.detect import randomforest
# IBEIS
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DETECT]')

SPECIAL = utool.get_argflag('--special') or utool.inIPython()


def TEST_DETECT(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    assert False, "Re-do this detect test"
    # print('get_valid_ANNOTATIONS')
    # gid_list = ibs.get_valid_gids()[0:1]
    # if SPECIAL:
    #     gid_list = utool.safe_slice(ibs.get_valid_gids(), 3)
    # #gid_list.extend(ibs.add_images([utool.unixpath('~/Dropbox/Chuck/detect_testimg/testgrevy.jpg')]))
    # species = 'zebra_plains'
    # detectkw = {
    #     'quick': True,
    #     'save_detection_images': SPECIAL,
    #     'save_scales': SPECIAL,
    # }
    # detect_gen = randomforest.detect_gid_list_with_species(ibs, gid_list, species, **detectkw)
    # gid_list2 = []
    # bbox_list2 = []
    # for gid, bboxes, confidences, img_conf in detect_gen:
    #     for bbox in bboxes:
    #         gid_list2.append(gid)
    #         bbox_list2.append(bbox)
            # not using confidence nor img_conf here

    # if SPECIAL:
    #     from plottool import viz_image2, fig_presenter
    #     #from plottool import draw_func2 as df2
    #     for gid in gid_list:
    #         isthisgid = [gid == gid2 for gid2 in gid_list2]
    #         bbox_list = utool.filter_items(bbox_list2, isthisgid)
    #         img = ibs.get_images(gid)
    #         print(bbox_list)
    #         fig = viz_image2.show_image(img, bbox_list=bbox_list)
    #     fig_presenter.present()
    #fig_presenter.all_figures_bring_to_front()
    #ibs.detect_random_forest(gid_list, 'zebra_grevys')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_DETECT, ibs)
    #execstr = utool.execstr_dict(test_locals, 'test_locals')
    #exec(execstr)
    if SPECIAL:
        from plottool import df2
        df2.present()
        raw_input()
