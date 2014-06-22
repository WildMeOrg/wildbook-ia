#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_IBS_INFO]')


def TEST_IBS_INFO(ibs):
    print(ibs.get_infostr())
    ibs.print_tables()
    ibs.print_stats()
    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For win32
    import ibeis
    # Initialize database
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_IBS_INFO, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
