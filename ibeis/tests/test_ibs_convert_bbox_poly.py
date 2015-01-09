#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool import geometry
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_CONVERT_BBOX_POLY]')


def TEST_CONVERT_BBOX_POLY(ibs):
    print('[TEST] CONVERT_BBOX_POLY')

    gids = ibs.get_valid_gids()
    bbox_list = [(0, 0, 100, 100)]
    aid_list = ibs.add_annots(gids[0:1], bbox_list=bbox_list)
    vert_list = ibs.get_annot_verts(aid_list)
    bbox_list_new = geometry.bboxes_from_vert_list(vert_list)
    assert bbox_list_new == bbox_list, 'Original bbox does not match the returned one'

    bbox_list = [(0, 0, 100, 100)]
    aid_list = ibs.add_annots(gids[1:2], bbox_list=bbox_list)
    vert_list = ibs.get_annot_verts(aid_list)
    vert_list_new = geometry.verts_list_from_bboxes_list(bbox_list)
    assert vert_list_new == vert_list, 'Vertices and their bounding box do not match'

    vert_list = [((0, 50), (50, 100), (100, 50), (50, 0))]
    aid_list = ibs.add_annots(gids[2:3], vert_list=vert_list)
    bbox_list = ibs.get_annot_bboxes(aid_list)
    bbox_list_new = geometry.bboxes_from_vert_list(vert_list)
    assert bbox_list_new == bbox_list, 'Original bbox does not match the returned one'

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=False,
                             allow_newdir=False, delete_ibsdir=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_CONVERT_BBOX_POLY, ibs)
    #execstr = utool.execstr_dict(test_locals, 'test_locals')
    #execstr += '\n' + utool.ipython_execstr()
    #exec(execstr)
