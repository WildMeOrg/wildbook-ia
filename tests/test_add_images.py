#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __testing__
import sys
import multiprocessing
from ibeis.dev import main_api
printTEST = __testing__.printTEST

sys.argv.append('--nogui')


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    try:
        # Create a HotSpotter API (hs) and GUI backend (back)
        printTEST('[TEST] TEST_ADD_IMAGES')
        main_locals = __testing__.main(defaultdb='testdb', nogui=True)
        ibs = main_locals['ibs']    # IBEIS Control
        #back = main_locals['back']  # IBEIS GUI backend

        printTEST('[TEST] GET_TEST_IMAGE_PATHS')
        # The test api returns a list of interesting chip indexes
        gpath_list = __testing__.get_test_image_paths(ibs, ndata=None)

        printTEST('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
        gid_list = ibs.add_images(gpath_list)
        valid_gids = list(ibs.get_valid_gids())
        printTEST('[TEST] valid_gids=%r' % valid_gids)
        valid_gpaths = list(ibs.get_image_paths(valid_gids))
        printTEST('[TEST] valid_gpaths=%r' % valid_gpaths)

        # Run Qt Loop to use the GUI
        printTEST('[TEST] MAIN_LOOP')
        __testing__.main_loop(main_locals, rungui=False)

    except Exception as ex:
        __testing__.handle_exceptions(ex, locals())
