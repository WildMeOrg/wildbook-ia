#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
"""
Tests IBEIS parallel
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
import pyhesaff
from utool import util_parallel
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_PARALLEL]')
#utool.inject_all()


def TEST_PARALLEL():
    gpath_list = grabdata.get_test_gpaths(ndata=10, names=['zebra', 'lena', 'jeff'])
    args_list  = [(gpath,) for gpath in gpath_list]

    @utool.argv_flag_dec
    def print_test_results(result_list):
        for kpts, desc in result_list:
            print('[test] kpts.shape=(%4d, %d), desc.sum=%8d' % (kpts.shape[0],
                                                                 kpts.shape[1],
                                                                 desc.sum()))

    hesaff_kwargs = {
        'scale_min': -1,
        'scale_max': -1,
        'nogravity_hack': False
    }

    with utool.Timer('c++ parallel'):
        kpts_list, desc_list = pyhesaff.detect_kpts_list(gpath_list, **hesaff_kwargs)

    # Run parallel tasks
    @utool.indent_func('[test_task]')
    def run_parallel_task(num_procs=None):
        print('run_parallel_task. num_procs=%r' % None)
        if num_procs is not None:
            util_parallel.close_pool()
            util_parallel.init_pool(num_procs)
        else:
            num_procs = util_parallel.get_default_numprocs()
        msg = 'processing tasks in %s' % ('serial' if num_procs == 1 else
                                          str(num_procs) + '-parallel')
        with utool.Timer(msg):
            result_list = util_parallel.process(pyhesaff.detect_kpts, args_list, hesaff_kwargs)
        print_test_results(result_list)
        return result_list
    run_parallel_task()

    # Compare to serial if needed
    @utool.argv_flag_dec
    def compare_serial():
        print('compare_serial')
        run_parallel_task(1)
    compare_serial()
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    test_locals = utool.run_test(TEST_PARALLEL)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
