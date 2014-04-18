#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
ibeis._preload()
from plottool import draw_func2 as df2
from ibeis.dev import main_helpers
import utool
import multiprocessing
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=True)


@utool.indent_decor('[dev]')
def run_experiments(ibs, qrid_list):
    print('\n')
    print('==========================')
    print('RUN INVESTIGATIONS %s' % ibs.get_dbname())
    print('==========================')
    input_test_list = params.args.tests[:]
    print('[dev] input_test_list = %r' % (input_test_list,))
    # fnum = 1

    valid_test_list = []  # build list for printing in case of failure

    def intest(*args):
        for testname in args:
            valid_test_list.append(testname)
            ret = testname in input_test_list
            if ret:
                input_test_list.remove(testname)
                print('+===================')
                print('| running testname=%s' % testname)
                return ret
        return False

    if intest('info'):
        print(ibs.get_infostr())
    if intest('query'):
        from ibeis.tests.test_query import TEST_QUERY
        TEST_QUERY(ibs)

    # Allow any testcfg to be in tests like:
    # vsone_1 or vsmany_3
    #testcfg_keys = vars(experiment_configs).keys()
    #testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    #for test_cfg_name in testcfg_locals:
        #if intest(test_cfg_name):
            #fnum = experiment_harness.test_configurations(ibs, qrid_list, [test_cfg_name], fnum)

    if intest('help'):
        print('valid tests are:')
        print(''.join(utool.indent_list('\n -t ', valid_test_list)))
        return

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    from ibeis.dev.all_imports import *  # NOQA
    print('\n [DEV] __DEV__\n')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    ibs  = main_locals['ibs']
    back = main_locals['back']

    fnum = 1
    qrid_list = main_helpers.get_test_qrids(ibs)
    run_experiments(ibs, qrid_list)

    df2.present()
    ipy = (not '--gui' in sys.argv) or ('--cmd' in sys.argv)
    execstr = ibeis.main_loop(main_locals, ipy=ipy)
    print('\n[DEV] ENTER EXEC\n')
    #print(execstr)
    exec(execstr)
