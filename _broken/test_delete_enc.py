#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ENC]')


def TEST_DELETE_ENC(ibs, back):
    from ibeis.other import ibsfuncs
    ibsfuncs.update_special_imagesets(ibs)
    imgsetid_list = ibs.get_valid_imgsetids()
    assert len(imgsetid_list) != 0, "All Image imageset not created"
    imgsetid = imgsetid_list[0]
    ibs.delete_imagesets(imgsetid)
    imgsetid_list = ibs.get_valid_imgsetids()
    assert imgsetid not in imgsetid_list, "imgsetid=%r still exists" % (imgsetid,)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ENC, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
