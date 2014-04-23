#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import __testing__  # Should be imported before any ibeis stuff
import multiprocessing
import utool
printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ADD_IMAGES]')


def TEST_ADD_IMAGES(ibs):
    printTEST('[TEST] GET_TEST_IMAGE_PATHS')
    # The test api returns a list of interesting chip indexes
    gpath_list = __testing__.get_test_gpaths(ndata=None)

    printTEST('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)
    valid_gpaths = list(ibs.get_image_paths(gid_list))
    printTEST('[TEST] valid_gpaths=%r' % valid_gpaths)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0')
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = __testing__.run_test(TEST_ADD_IMAGES, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
