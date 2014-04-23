#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
try:
    import __testing__
    printTEST = __testing__.printTEST
except ImportError:
    printTEST = print
    pass
# Python
from os.path import join, exists  # NOQA
import multiprocessing
# Tools
import utool
#IBEIS
from ibeis.dev import params  # NOQA
import numpy as np
from ibeis.model.hots import QueryRequest  # NOQA
from ibeis.model.hots import NNIndex  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[TEST_IBS_GETTERS]')


def TEST_IBS_GETTERS(ibs=None):
    if ibs is None:
        print('ibs is none')

    gid_list = ibs.get_valid_gids()
    rid_list = ibs.get_valid_rids()

    # Ensure even number or rids
    rid_list = rid_list[0:(len(rid_list) // 2) * 2]
    rid_scalar = rid_list[0]
    rid_numpy  = np.array(rid_list).reshape((len(rid_list) / 2, 2))

    # tests that scalar, list, and numpy getters are working correctly
    bboxes_list   = ibs.get_roi_bboxes(rid_list)
    bboxes_scalar = ibs.get_roi_bboxes(rid_scalar)
    bboxes_numpy  = ibs.get_roi_bboxes(rid_numpy)

    kpts_list   = ibs.get_roi_kpts(rid_list)
    kpts_scalar = ibs.get_roi_kpts(rid_scalar)
    kpts_numpy  = ibs.get_roi_kpts(rid_numpy)

    def assert_getter_output(list_, scalar, numpy_, label=''):
        item1 = list_[0]
        item2 = scalar
        item3 = numpy_[0, 0]
        assert np.all(np.array(item1) == np.array(item2)), 'getters are broken'
        assert np.all(np.array(item2) == np.array(item3)), 'getters are broken'
        print(label + ' passed')

    assert_getter_output(bboxes_list, bboxes_scalar, bboxes_numpy, 'bboxes')
    assert_getter_output(kpts_list, kpts_scalar, kpts_numpy, 'kpts')

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    if main_locals is None:
        main_locals.update(locals())
        __testing__.main_loop(main_locals, rungui=False)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb_big')
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = __testing__.run_test(TEST_IBS_GETTERS, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
