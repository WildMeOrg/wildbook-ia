#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from ibeis.model.hots import hots_nn_index
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_HS_INDEX]')


def TEST_HS_INDEX(ibs):
    print('[TEST_HS_INDEX]')
    daid_list = ibs.get_valid_aids()
    nn_index1 = hots_nn_index.HOTSIndex(ibs, daid_list, use_cache=False)
    # Test Cache
    nn_index2 = hots_nn_index.HOTSIndex(ibs, daid_list, use_cache=True)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    ibs = ibeis.main(defaultdb='testdb1', gui=False)['ibs']
    test_locals = utool.run_test(TEST_HS_INDEX, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
