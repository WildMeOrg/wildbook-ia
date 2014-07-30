#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
from ibeis.control.IBEISControl import IBEISController
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_LOCALIZE_IMAGES]')


def TEST_LOCALIZE_IMAGES(ibs):
    assert isinstance(ibs, IBEISController)
    print('[TEST_LOCALIZE_IMAGES]')

    gid_list = ibs.get_valid_gids()
    uuids_old = ibs.get_image_uuids(gid_list)

    # test that uuids are the same after running localize images
    ibs.localize_images()
    gpath_list = ibs.get_image_paths(gid_list)

    assert gid_list, 'get_valid_gids returned empty'
    assert gpath_list, 'get_image_paths returned empty'

    print('[gpath_list = %r' % (gpath_list,))

    uuids_new = [utool.get_file_uuid(gpath) for gpath in gpath_list]

    assert uuids_old, 'get_file_uuid returned empty'
    assert uuids_new, 'get_image_uuids returned empty'

    print('uuids_old = %r' % (uuids_old,))
    print('uuids_new = %r' % (uuids_new,))
    assert uuids_old == uuids_new, 'regenerated uuids are not the same as the originals'

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_LOCALIZE_IMAGES, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
