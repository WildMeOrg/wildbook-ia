#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __sysreq__  # NOQA
import __builtin__
import sys
import multiprocessing
from os.path import expanduser
from ibeis.dev import main_api


sys.argv.append('--nogui')
INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    __builtin__.print('\n=============================')
    __builtin__.print(msg)
    if INTERACTIVE:
        raw_input('press enter to continue')


def  get_test_image_paths(ibs, ndata=None):
    if ndata is None:
        test_gpaths = [expanduser('/lena.png')]
    if INTERACTIVE:
        test_gpaths = None
    return test_gpaths


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    try:
        # Create a HotSpotter API (hs) and GUI backend (back)
        print('[TEST] TEST_ADD_IMAGES')
        main_locals = main_api.main(defaultdb='testdb', nogui=True)
        ibs = main_locals['ibs']    # IBEIS Control
        #back = main_locals['back']  # IBEIS GUI backend

        print('[TEST] GET_TEST_IMAGE_PATHS')
        # The test api returns a list of interesting chip indexes
        gpath_list = get_test_image_paths(ibs, ndata=None)

        print('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
        gid_list = ibs.add_images(gpath_list)
        valid_gids = list(ibs.get_valid_gids())
        print('[TEST] valid_gids=%r' % valid_gids)
        valid_gpaths = list(ibs.get_image_paths(valid_gids))
        print('[TEST] valid_gpaths=%r' % valid_gpaths)

        # Run Qt Loop to use the GUI
        print('[TEST] MAIN_LOOP')
        main_api.main_loop(main_locals, rungui=False)

    except Exception as ex:
        print('[TEST] test_add_images FAILED: %s %s' % (type(ex), ex))
        ibs.db.dump()
        if '--strict' in sys.argv:
            raise
