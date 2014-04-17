#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
from ibeis.dev.all_imports import *  # NOQA
import ibeis
import utool
from drawtool import draw_func2 as df2
from tests.__testing__ import printTEST  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=True)


def run_investigations(ibs, qrid_list):
    print('\n\n')
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
                print('[dev] ===================')
                print('[dev] running testname=%s' % testname)
                return ret
        return False

    if intest('print-ibs'):
        print(ibs.get_infostr())
    if intest('query'):
        from tests.test_query import TEST_QUERY
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
        print(''.join(util.indent_list('\n -t ', valid_test_list)))
        return

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    print('\n [DEV] __DEV__\n')
    main_locals = main_api.main()
    ibs = main_locals['ibs']
    back = main_locals['back']
    print('\n [DEV] ENTER EXEC \n')

    main_locals = ibeis.main()
    ibs = main_locals['ibs']

    #qrid_list = main_api.get_qrid_list(ibs)
    qrid_list = []
    print('[dev]====================')
    #mf.print_off()  # Make testing slightly faster
    # Big test function. Should be replaced with something
    # not as ugly soon.
    fnum = 1
    run_investigations(ibs, qrid_list)

    execstr = ibeis.main_loop(main_locals, ipy=True)
    exec(df2.present())
    #exec(execstr)
