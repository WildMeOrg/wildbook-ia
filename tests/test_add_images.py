#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import sys
import __sysreq__  # NOQA
from ibeis.dev import main_api
import multiprocessing


def  get_test_image_paths(ibs, nTest=None):
    if nTest is None:
        test_gpaths = ['lena.png']
    if '-i' in sys.argv or '--interactive' in sys.argv:
        test_gpaths = None
    return test_gpaths


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    try:
        # Create a HotSpotter API (hs) and GUI backend (back)
        main_locals = main_api.main(defaultdb='NAUTS')
        ibs = main_locals['ibs']    # IBEIS Control
        back = main_locals['back']  # IBEIS GUI backend

        # The test api returns a list of interesting chip indexes
        gpath_list = get_test_image_paths(ibs, nTest=None)
        gid_list = back.import_images_from_file(gpath_list)
        # Run Qt Loop to use the GUI
        main_api.main_loop(main_locals)
    except Exception as ex:
        print('[test] test_add_images Handled: %s %s' % (type(ex), ex))
        print('[test] test_add_images FAILED')
        if '--strict' in sys.argv:
            raise
