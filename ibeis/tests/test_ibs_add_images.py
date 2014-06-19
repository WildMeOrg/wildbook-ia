#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ADD_IMAGES]')


def TEST_ADD_IMAGES(ibs):
    print('[TEST] GET_TEST_IMAGE_PATHS')
    # The test api returns a list of interesting chip indexes
    gpath_list = grabdata.get_test_gpaths(ndata=None)

    print('[TEST] IMPORT IMAGES FROM FILE\ngpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)
    valid_gpaths = list(ibs.get_image_paths(gid_list))
    print('[TEST] valid_gpaths=%r' % valid_gpaths)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_ADD_IMAGES, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
