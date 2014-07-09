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
    aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list, name_list=name_list)
    assert len(aid_list) != 0, "No annotations"
    aid = aid_list[0]
    gid = ibs.get_annot_gids(aid)
    assert gid is not None, "gid for aid=%r is None" % (aid,)
    gthumbpath = ibs.get_image_thumbpath(gid)
    annotation_thumbpath = ibs.get_annot_chip_thumbpath(aid)
    ibs.delete_annot_chips(aid)
    aid_list = ibs.get_valid_aids()
    assert aid in aid_list, "Error: Annotation deleted"
    assert not utool.checkpath(gthumbpath), "Image Thumbnail not deleted"
    assert not utool.checkpath(annotation_thumbpath), "Roi Thumbnail not deleted"
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
