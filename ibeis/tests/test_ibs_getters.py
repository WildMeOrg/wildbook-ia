#!/usr/bin/env python2.7
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
    if ibs is None:
        print('ibs is none')

    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()

    # Ensure we grab an even number of aids
    aid_list   = aid_list[0:(len(aid_list) // 2) * 2]
    aid_scalar = aid_list[0]
    aid_numpy  = np.array(aid_list).reshape((len(aid_list) / 2, 2))

    # tests that scalar, list, and numpy getters are working correctly
    bboxes_list   = ibs.get_annotation_bboxes(aid_list)
    bboxes_scalar = ibs.get_annotation_bboxes(aid_scalar)
    bboxes_numpy  = ibs.get_annotation_bboxes(aid_numpy)

    kpts_list   = ibs.get_annotation_kpts(aid_list)
    kpts_scalar = ibs.get_annotation_kpts(aid_scalar)
    kpts_numpy  = ibs.get_annotation_kpts(aid_numpy)

    def assert_getter_output(list_, scalar, numpy_, label=''):
        item1 = list_[0]
        item2 = scalar
        item3 = numpy_[0, 0]
        assert np.all(np.array(item1) == np.array(item2)), 'getters are broken'
        assert np.all(np.array(item2) == np.array(item3)), 'getters are broken'
        print(label + ' passed')

    assert_getter_output(bboxes_list, bboxes_scalar, bboxes_numpy, 'bboxes')
    assert_getter_output(kpts_list, kpts_scalar, kpts_numpy, 'kpts')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_IBS_GETTERS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
