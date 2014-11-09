#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
from ibeis.dbio import export_subset
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__,
                                                     '[TEST_IBS_EXPORT]')


def TEST_IBS_EXPORT(ibs):
    print('[TEST_IBS_EXPORT]')
    gid_list1 = ibs.get_valid_gids()[0:4]
    new_dbdir = 'testdb_dst'
    # delete for doctest
    ibs_dst = ibeis.opendb('testdb_dst', allow_newdir=True, delete_ibsdir=True)
    assert ibs_dst.get_num_names() == 0
    assert ibs_dst.get_num_images() == 0
    assert ibs_dst.get_num_annotations() == 0
    status = export_subset.transfer_data(ibs, ibs_dst, gid_list1)
    ibs_dst.get_num_annotations() == 4
    ibs_dst.get_num_images() == 4
    ibs_dst.get_num_names() == 1
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    ibs = ibeis.main(defaultdb='testdb1', gui=False)['ibs']
    test_locals = utool.run_test(TEST_IBS_EXPORT, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
