#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.tests import __testing__
import multiprocessing
import utool
# IBEIST
from ibeis.model.preproc import preproc_detectimg
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_COMPUTE_DETECTIMG]')


def TEST_COMPUTE_DETECTIMG(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    print('get_valid_ROIS')
    gid_list = ibs.get_valid_gids()
    assert len(gid_list) > 0, 'database rois cannot be empty for TEST_COMPUTE_DETECTIMG'
    print(' * len(gid_list) = %r' % len(gid_list))
    preproc_detectimg.compute_and_write_detectimg(ibs, gid_list)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0')
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = __testing__.run_test(TEST_COMPUTE_DETECTIMG, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
