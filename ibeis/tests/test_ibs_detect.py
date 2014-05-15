#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
import ibeis
import multiprocessing
# IBEIS
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DETECT]')


def TEST_DETECT(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    print('get_valid_ROIS')
    gid_list = ibs.get_valid_gids()[0:1]
    ibs.detect_random_forest(gid_list, 'zebra_grevys')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_DETECT, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
