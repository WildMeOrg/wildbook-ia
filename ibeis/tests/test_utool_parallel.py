#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
'''
Tests IBEIS parallel
'''
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
import multiprocessing
import utool
import pyhesaff
from utool import util_parallel
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_PARALLEL]')
#utool.inject_all()
printTEST = __testing__.printTEST


def TEST_PARALLEL():

    gpath_list = __testing__.get_test_gpaths()
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

    # Run parallel tasks
    @utool.indent_decor('[test_task]')
    def run_parallel_task(num_procs=None):
        printTEST('run_parallel_task. num_procs=%r' % None)
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
        printTEST('compare_serial')
        run_parallel_task(1)
    compare_serial()
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    test_locals = __testing__.run_test(TEST_PARALLEL)
    execstr = __testing__.main_loop(test_locals)
    exec(execstr)
