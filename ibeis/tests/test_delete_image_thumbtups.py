#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_IMAGE_THUMBTUPS]')


def TEST_DELETE_IMAGE_THUMBTUPS(ibs, back):
    gid_list = ibs.get_valid_gids()
    gid = gid_list[0]
    thumbtup_list = ibs.get_image_thumbtup(gid)
    thumbpath_list = [tup[0] for tup in thumbtup_list]
    gpath_list = [tup[1] for tup in thumbtup_list]
    ibs.delete_image_thumbtup(gid)
    for path in thumbpath_list:
        assert not utool.checkpath(path), "Thumbnail not deleted"
    for path in gpath_list:
        utool.assertpath(path)
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_IMAGE_THUMBTUPS, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
