#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#-----
TEST_NAME = 'TEST_ADD_IMAGES'
#-----
import __testing__  # Should be imported before any ibeis stuff
import sys
import multiprocessing
import utool
printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


@__testing__.testcontext2(TEST_NAME)
def TEST_ADD_IMAGES():
    main_locals = __testing__.main(defaultdb='testdb', nogui=True)
    ibs = main_locals['ibs']    # IBEIS Control
    #back = main_locals['back']  # IBEIS GUI backend

    printTEST('[TEST] GET_TEST_IMAGE_PATHS')
    # The test api returns a list of interesting chip indexes
    gpath_list = __testing__.get_test_image_paths(ibs, ndata=None)

    printTEST('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)
    valid_gpaths = list(ibs.get_image_paths(gid_list))
    printTEST('[TEST] valid_gpaths=%r' % valid_gpaths)

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=False)
    return main_locals


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_ADD_IMAGES()
    exec(test_locals['execstr'])
