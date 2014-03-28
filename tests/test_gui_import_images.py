#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __testing__  # NOQA
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[test_parallel]')
printTEST = __testing__.printTEST


@__testing__.testcontext
def test_gui_import_images():
    # Create a HotSpotter API (hs) and GUI backend (back)
    printTEST('[TEST] TEST_ADD_IMAGES')
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend

    printTEST('[TEST] GET_TEST_IMAGE_PATHS', True)
    # The test api returns a list of interesting chip indexes
    gpath_list = __testing__.get_test_image_paths(ibs, ndata=None)

    print('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
    gid_list = back.import_images_from_file(gpath_list)

    print('[TEST] gid_list=%r' % gid_list)
    __testing__.main_loop(main_locals)


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_gui_import_images()
