#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
#-----
TEST_NAME = 'TEST_COMPUTE_CHIPS'
#-----
import __testing__
import multiprocessing
from itertools import izip
import utool
import vtool.chip as ctool
# IBEIST
from ibeis.model.jon_recognition import Config
from ibeis.view import viz
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext
def TEST_COMPUTE_CHIPS():
    # Create a HotSpotter API (hs) and GUI backend (back)
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA

    chip_cfg = Config.default_chip_cfg()

    #chip_cfg =

    printTEST('get_valid_ROIS')
    rid_list = ibs.get_valid_rids()
    assert len(rid_list) > 0, 'database rois cannot be empty for ' + TEST_NAME
    print(' * len(rid_list) = %r' % len(rid_list))

    gpath_list = ibs.get_roi_gpaths(rid_list)
    theta_list = ibs.get_roi_thetas(rid_list)
    bbox_list = ibs.get_roi_bboxes(rid_list)

    target_area = chip_cfg['chip_sqrt_area'] ** 2
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, ((w, h) for (x, y, w, h) in bbox_list))

    args_list = [args for args in izip(gpath_list, bbox_list, theta_list, newsize_list)]
    filter_list = ctool.get_filter_list(chip_cfg.to_dict())
    chip_kwargs =  {
        'filter_list': filter_list,
    }
    chip_list = utool.util_parallel.process(ctool.compute_chip, args_list, chip_kwargs, force_serial=True)
    viz.df2.imshow(chip_list[0])

    __testing__.main_loop(main_locals, rungui=RUNGUI)


TEST_COMPUTE_CHIPS.func_name = TEST_NAME


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    TEST_COMPUTE_CHIPS()
    #exec(viz.df2.present())
