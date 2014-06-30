#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_IMAGE_THUMBTUPS]')


def TEST_DELETE_IMAGE_THUMBTUPS(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)]*len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    aid_list = ibs.add_annotations(gid_list, bbox_list=bbox_list, name_list=name_list)
    assert len(aid_list) !=0, "No annotations added"
    thumbtup_list = ibs.get_image_thumbtup(gid_list)
    assert utool.is_list(thumbtup_list), "thumbtup_list is not a list"
    thumbpath_list = [tup[0] for tup in thumbtup_list]
    gpath_list = [tup[1] for tup in thumbtup_list]
    ibs.delete_image_thumbtups(gid_list)
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
