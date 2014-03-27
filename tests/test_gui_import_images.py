#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __sysreq__  # NOQA
import __builtin__
import sys
import multiprocessing
from os.path import expanduser
from ibeis.dev import main_api


INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    __builtin__.print('\n=============================')
    __builtin__.print(msg)
    if INTERACTIVE:
        raw_input('press enter to continue')


def  get_test_image_paths(ibs, nTest=None):
    if nTest is None:
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
        main_locals = main_api.main(defaultdb='NAUTS')
        ibs = main_locals['ibs']    # IBEIS Control
        back = main_locals['back']  # IBEIS GUI backend

        print('[TEST] GET_TEST_IMAGE_PATHS')
        # The test api returns a list of interesting chip indexes
        gpath_list = get_test_image_paths(ibs, nTest=None)

        print('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
        gid_list = back.import_images_from_file(gpath_list)

        # Run Qt Loop to use the GUI
        print('[TEST] MAIN_LOOP')
        main_api.main_loop(main_locals, rungui=True)

    except Exception as ex:
        print('[TEST] test_add_images FAILED: %s %s' % (type(ex), ex))
        ibs.db.dump()
        if '--strict' in sys.argv:
            raise
