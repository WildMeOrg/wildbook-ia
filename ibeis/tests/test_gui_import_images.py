#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_IMPORT_IMAGES]')

from __testing__ import printTEST


def TEST_GUI_IMPORT_IMAGES(ibs, back):
    printTEST('[TEST] GET_TEST_IMAGE_PATHS', wait=True)
    # The test api returns a list of interesting chip indexes
    mode = 'FILE'
    if mode == 'FILE':
        gpath_list = __testing__.get_test_gpaths()
        # else:
        #    dir_ = utool.truepath(join(sysres.get_workdir(), 'PZ_MOTHERS/images'))
        #    gpath_list = utool.list_images(dir_, fullpath=True)[::4]
        printTEST('[TEST] IMPORT IMAGES FROM FILE\n * gpath_list=%r' % gpath_list)
        gid_list = back.import_images(gpath_list=gpath_list)
    elif mode == 'DIR':
        dir_ = __testing__.get_testdata_dir()
        printTEST('[TEST] IMPORT IMAGES FROM DIR\n * dir_=%r' % dir_)
        gid_list = back.import_images(dir_=dir_)
    else:
        raise AssertionError('unknown mode=%r' % mode)

    printTEST('[TEST] * len(gid_list)=%r' % len(gid_list))
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = __testing__.run_test(TEST_GUI_IMPORT_IMAGES, ibs, back)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
