#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
import os
sys.path.append(os.path.expanduser('~/code/ibeis/tests'))
#-----
TEST_NAME = 'TEST_GUI_IMPORT_IMAGES'
#-----
import __testing__
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext
def TEST_GUI_IMPORT_IMAGES():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    printTEST('[TEST] GET_TEST_IMAGE_PATHS', wait=True)
    # The test api returns a list of interesting chip indexes
    mode = 'FILE'
    if mode == 'FILE':
        gpath_list = __testing__.get_test_image_paths(ibs, ndata=None)
        # else:
        #    dir_ = utool.truepath(join(params.get_workdir(), 'PZ_MOTHERS/images'))
        #    gpath_list = utool.list_images(dir_, fullpath=True)[::4]
        printTEST('[TEST] IMPORT IMAGES FROM FILE\n * gpath_list=%r' % gpath_list)
        gid_list = back.import_images(gpath_list=gpath_list)
    elif mode == 'DIR':
        dir_ = utool.truepath('~/data/work/PZ_MOTHERS/images')
        printTEST('[TEST] IMPORT IMAGES FROM DIR\n * dir_=%r' % dir_)
        gid_list = back.import_images(dir_=dir_)
    else:
        raise AssertionError('unknown mode=%r' % mode)

    printTEST('[TEST] * len(gid_list)=%r' % len(gid_list))
    __testing__.main_loop(main_locals, rungui=RUNGUI)
    return locals()

TEST_GUI_IMPORT_IMAGES.func_name = TEST_NAME

if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_GUI_IMPORT_IMAGES()
    ibs = test_locals['ibs']
    back = test_locals['back']
    exec(__testing__.execfunc()())
