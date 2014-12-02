#!/usr/bin/env python2.7
"""
python ibeis/tests/test_ibs_getters.py

"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
#IBEIS
import numpy as np
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[TEST_IBS_GETTERS]')


def TEST_IBS_GETTERS(ibs=None):
    """
    >>> import ibeis
    >>> main_locals = ibeis.main(defaultdb='testdb0', gui=False)
    >>> ibs = main_locals['ibs']    # IBEIS Control
    >>> test_locals = utool.run_test(TEST_IBS_GETTERS, ibs)
    """
    if ibs is None:
        print('ibs is none')

    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()

    # Ensure we grab an even number of aids
    aid_list   = aid_list[0:(len(aid_list) // 2) * 2]
    aid_scalar = aid_list[0]
    aid_numpy  = np.array(aid_list).reshape((len(aid_list) / 2, 2))

    # tests that scalar, list, and numpy getters are working correctly
    bboxes_list   = ibs.get_annot_bboxes(aid_list)
    bboxes_scalar = ibs.get_annot_bboxes(aid_scalar)
    bboxes_numpy  = ibs.get_annot_bboxes(aid_numpy)

    kpts_list   = ibs.get_annot_kpts(aid_list)
    kpts_scalar = ibs.get_annot_kpts(aid_scalar)
    kpts_numpy  = ibs.get_annot_kpts(aid_numpy)

    # Test lazy (noneager eval)
    fid_list  = ibs.get_annot_feat_rowids(aid_list, eager=True)
    assert isinstance(fid_list, list), 'should be a list'
    utool.DEBUG2 = True
    fid_gen = ibs.get_annot_feat_rowids(aid_list, eager=False)
    utool.DEBUG2 = False
    import types
    assert isinstance(fid_gen, types.GeneratorType)
    fid_list_ = list(fid_gen)
    assert len(list(fid_gen)) == 0, 'generator should be used up'
    assert fid_list == fid_list_, 'sql generators not working'

    #from ibeis.constants import FEATURE_TABLE
    #eager = False
    #colnames = ('feature_sifts',)
    #tblname = FEATURE_TABLE
    #id_iter = fid_list
    #id_colname = 'rowid'
    #desc_list = ibs.dbcache.get(FEATURE_TABLE, ('feature_sifts',), fid_list, nInput=len(fid_list), eager=eager)
    #"""
    #%timeit list(ibs.dbcache.get(FEATURE_TABLE, ('feature_sifts',), fid_list, nInput=len(fid_list), eager=False))
    #%timeit ibs.dbcache.get(FEATURE_TABLE, ('feature_sifts',), fid_list, nInput=len(fid_list), eager=True)
    #"""
    #utool.embed()

    def assert_getter_output(list_, scalar, numpy_, lbl=''):
        item1 = list_[0]
        item2 = scalar
        item3 = numpy_[0, 0]
        assert np.all(np.array(item1) == np.array(item2)), 'getters are broken'
        assert np.all(np.array(item2) == np.array(item3)), 'getters are broken'
        print(lbl + ' passed')

    assert_getter_output(bboxes_list, bboxes_scalar, bboxes_numpy, lbl='bboxes')
    assert_getter_output(kpts_list, kpts_scalar, kpts_numpy, lbl='kpts')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_IBS_GETTERS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
