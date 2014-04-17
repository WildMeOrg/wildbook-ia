#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev.all_imports import *  # NOQA
from itertools import izip
import ibeis
import multiprocessing
import utool
from plottool import draw_func2 as df2
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=True)


@utool.indent_func
def filter_rids(ibs, rid_list, with_hard=True, with_gt=True, with_nogt=True):
    qrid_list = []
    if with_hard:
        notes_list = ibs.get_roi_notes(rid_list)
        qrid_list.extend([rid for (notes, rid) in izip(notes_list, rid_list)
                          if 'hard' in notes.lower().split()])
    if with_gt and not with_nogt:
        gts_list = ibs.get_roi_groundtruth(rid_list)
        qrid_list.extend([rid for (gts, rid) in izip(gts_list, rid_list)
                          if len(gts) > 0])
    if with_gt and with_nogt:
        qrid_list = rid_list
    return qrid_list


@utool.indent_func
def get_test_qrids(ibs):
    """ Function for getting the list of queries to test """
    print('[dev] get_test_qrids()')

    valid_rids = ibs.get_valid_rids()

    # Sample a large pool of query indexes
    histids = None if params.args.histid is None else np.array(params.args.histid)
    if params.args.all_cases:
        print('[dev] all cases')
        qrids_all = filter_rids(ibs, valid_rids, with_gt=True, with_nogt=True)
    elif params.args.all_gt_cases:
        print('[dev] all gt cases')
        qrids_all = filter_rids(ibs, valid_rids, with_hard=True, with_gt=True, with_nogt=False)
    elif params.args.qrid is None:
        print('[dev] did not select cases')
        qrids_all = filter_rids(ibs, valid_rids, with_hard=True, with_gt=False, with_nogt=False)
    else:
        print('[dev] Chosen qrid=%r' % params.args.qrid)
        qrids_all = params.args.qrid

    # Filter only the ones you want from the large pool
    if histids is None:
        qrid_list = qrids_all
    else:
        histids = utool.ensure_iterable(histids)
        print('[dev] Chosen histids=%r' % histids)
        qrid_list = [qrid_list[id_] for id_ in histids]

    if len(qrid_list) == 0:
        msg = '[dev.get_qrids] no qrid_list history'
        print(msg)
        print(valid_rids)
        qrid_list = valid_rids[0:1]
    print('[dev] len(qrid_list) = %d' % len(qrid_list))
    qrid_list = utool.unique_keep_order(qrid_list)
    print('[dev] qrid_list = %r' % qrid_list)
    return qrid_list


@utool.indent_func
def run_experiments(ibs, qrid_list):
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
    print('\n [DEV] __DEV__\n')

    main_locals = ibeis.main()
    ibs = main_locals['ibs']
    back = main_locals['back']

    qrid_list = get_test_qrids(ibs)
    qrid_list = []
    print('[dev]====================')
    fnum = 1
    run_experiments(ibs, qrid_list)

    execstr = ibeis.main_loop(main_locals, ipy=True)
    df2.present()
    print('\n [DEV] ENTER EXEC \n')
    exec(execstr)
