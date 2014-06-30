#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ANNOTATION_CHIPS]')


def TEST_DELETE_ANNOTATION_CHIPS(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)] * len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    aid_list = ibs.add_annotations(gid_list, bbox_list=bbox_list, name_list=name_list)
    aid = aid_list[0]
    gid = ibs.get_annotion_gids(aid)
    gthumbtup = ibs.get_image_thumbtup(gid)
    gthumbpath = gthumbtup[0]
    annotion_thumbtup = ibs.get_annotion_chip_thumbtup(aid)
    annotion_thumbpath = annotion_thumbtup[0]
    ibs.delete_annotion_chips(aid)
    aid_list = ibs.get_valid_aids()
    assert aid in aid_list, "Error: RID deleted"
    assert not utool.checkpath(gthumbpath), "Image Thumbnail not deleted"
    assert not utool.checkpath(annotion_thumbpath), "Roi Thumbnail not deleted"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ANNOTATION_CHIPS, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
