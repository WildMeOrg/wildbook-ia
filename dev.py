#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
# Dev
from _devscript import devcmd,  DEVCMD_FUNCTIONS
import ibeis
if __name__ == '__main__':
    multiprocessing.freeze_support()
    ibeis._preload()
    from ibeis.dev.all_imports import *  # NOQA
from _devcmds_ibeis import *  # NOQA
# Tools
from plottool import draw_func2 as df2
# IBEIS
from ibeis.dev import main_helpers
from ibeis.dev import dbinfo
from ibeis.viz import interact
from ibeis.dev import experiment_configs
from ibeis.dev import experiment_harness
from ibeis.dev import results_all
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=False)

if 'back' not in vars():
    back = None


#--------------------
# RUN DEV EXPERIMENTS
#--------------------


#@utool.indent_decor('[dev]')
@profile
def run_experiments(ibs, qrid_list):
    print('\n')
    print('==========================')
    print('RUN INVESTIGATIONS %s' % ibs.get_dbname())
    print('==========================')
    input_test_list = params.args.tests[:] + params.unknown[:]  # Let unparsed args count towards tests
    print('input_test_list = %r' % (input_test_list,))
    # fnum = 1

    valid_test_list = []  # build list for printing in case of failure
    valid_test_helpstr_list = []  # for printing

    def intest(*args, **kwargs):
        helpstr = kwargs.get('help', '')
        valid_test_helpstr_list.append('   -t ' + ', '.join(args) + helpstr)
        for testname in args:
            valid_test_list.append(testname)
            ret = testname in input_test_list
            if ret:
                input_test_list.remove(testname)
                print('+===================')
                print('| running testname = %s' % (args,))
                return ret
        return False

    valid_test_helpstr_list.append('    # --- Simple Tests ---')

    if intest('export'):
        export(ibs)
    if intest('info'):
        print(ibs.get_infostr())
    if intest('dbinfo'):
        dbinfo.get_dbinfo(ibs)
    if intest('printcfg'):
        printcfg(ibs)
    if intest('tables'):
        ibs.print_tables()
    if intest('imgtbl'):
        ibs.print_image_table()

    valid_test_helpstr_list.append('    # --- Decor Tests ---')

    # Run decorated functions
    for (func_aliases, func) in DEVCMD_FUNCTIONS:
        if intest(*func_aliases):
            func(ibs, qrid_list)

    valid_test_helpstr_list.append('    # --- Config Tests ---')

    # Allow any testcfg to be in tests like: vsone_1 or vsmany_3
    for test_cfg_name in experiment_configs.TEST_NAMES:
        if intest(test_cfg_name):
            test_cfg_name_list = [test_cfg_name]
            fnum = df2.next_fnum()
            experiment_harness.test_configurations(ibs, qrid_list, test_cfg_name_list, fnum)

    valid_test_helpstr_list.append('    # --- Help ---')

    if intest('help'):
        print('valid tests are:')
        print('\n'.join(valid_test_helpstr_list))
        return

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)
    return locals()


#-------------------
# CUSTOM DEV FUNCS
#-------------------


__ALLRES_CACHE__ = {}


def get_allres(ibs, qrid_list):
    allres_uid = ibs.qreq.get_uid()
    try:
        allres = __ALLRES_CACHE__[allres_uid]
    except KeyError:
        qrid2_qres = ibs.query_database(qrid_list)
        allres = results_all.init_allres(ibs, qrid2_qres)
    # Cache save
    __ALLRES_CACHE__[allres_uid] = allres
    return allres


#------------------
# DEV DEVELOPMENT
#------------------


def devfunc(ibs, qrid_list):
    """ Function for developing something """
    allres = get_allres(ibs, qrid_list)
    locals_ = locals()
    #locals_.update(chipmatch_scores(ibs, qrid_list))
    return locals_


@devcmd('desc_dists')
def desc_dists(ibs, qrid_list):
    """ Plots the distances between matching descriptors
    labeled with groundtruth (true/false) data """
    allres = get_allres(ibs, qrid_list)
    # Get the descriptor distances of true matches
    orgtype_list = ['top_false', 'true']
    disttype = 'L2'
    desc_distances_map = allres.get_desc_match_dists(orgtype_list, orgtype_list)
    results_analyzer.print_desc_distances_map(desc_distances_map)
    #true_desc_dists  = desc_distances_map['true']['L2']
    #false_desc_dists = desc_distances_map['false']['L2']
    #scores_list = [false_desc_dists, true_desc_dists]
    scores_list = [desc_distances_map[orgtype][disttype] for orgtype in orgtype_list]
    scores_lbls = orgtype_list
    scores_markers = ['x', 'o--']
    plottool.plots.draw_scores_cdf(scores_list, scores_lbls, scores_markers)
    df2.set_figtitle('Descriptor Distances')
    return locals()


@devcmd('scores')
def chipmatch_scores(ibs, qrid_list):
    allres = get_allres(ibs, qrid_list)
    # Get the descriptor distances of true matches
    orgtype_list = ['false', 'true']
    markers_map = {'false': 'x', 'true': 'o-'}
    cmatch_scores_map = allres.get_chipmatch_scores(orgtype_list)
    results_analyzer.print_chipmatch_scores_map(cmatch_scores_map)
    true_cmatch_scores  = cmatch_scores_map['true']
    false_cmatch_scores = cmatch_scores_map['false']
    scores_list = [cmatch_scores_map[orgtype] for orgtype in orgtype_list]
    scores_lbls = orgtype_list
    scores_markers = [markers_map[orgtype] for orgtype in orgtype_list]
    plottool.plots.draw_scores_cdf(scores_list, scores_lbls, scores_markers)
    df2.set_figtitle('Chipmatch Scores ' + ibs.qreq.get_uid())
    return locals()


@devcmd('gv')
def gvcomp(ibs, qrid_list):
    """
    GV = With gravity vector
    RI = With rotation invariance
    """
    def testcomp(ibs, qrid_list):
        allres = get_allres(ibs, qrid_list)
        for qrid in qrid_list:
            qres = allres.get_qres(qrid)
            interact.ishow_qres(ibs, qres, annote_mode=2)
        return allres
    ibs_GV = ibs
    ibs_RI = ibs.clone_handle(nogravity_hack=True)

    allres_GV = testcomp(ibs_GV, qrid_list)
    allres_RI = testcomp(ibs_RI, qrid_list)
    return locals()


def get_ibslist(ibs):
    ibs_GV  = ibs
    ibs_RI  = ibs.clone_handle(nogravity_hack=True)
    ibs_RIW = ibs.clone_handle(nogravity_hack=True, gravity_weighting=True)
    ibs_list = [ibs_GV, ibs_RI, ibs_RIW]
    return ibs_list


@devcmd('gv_scores')
def compgrav_chipmatch_scores(ibs, qrid_list):
    ibs_list = get_ibslist(ibs)
    for ibs_ in ibs_list:
        chipmatch_scores(ibs_, qrid_list)


#------------------
# DEV MAIN
#------------------

def rundev(main_locals):
    ibs = main_locals['ibs']
    back = main_locals['back']
    fnum = 1
    qrid_list = main_helpers.get_test_qrids(ibs)
    print('test_qrids = %r' % qrid_list)
    print('len(test_qrids) = %d' % len(qrid_list))
    assert len(qrid_list) > 0, 'assert!'
    ibs.prep_qreq_db(qrid_list)

    expt_locals = run_experiments(ibs, qrid_list)

    if '--cmd' in sys.argv:
        exec(utool.execstr_dict(expt_locals, 'expt_locals'))

    if '--devmode' in sys.argv:
        devfunc_locals = devfunc(ibs, qrid_list)
        exec(utool.execstr_dict(devfunc_locals, 'devfunc_locals'))

    if '--nopresent' not in sys.argv:
        df2.present()
    ipy = ('--gui' not in sys.argv) or ('--cmd' in sys.argv)
    main_execstr = ibeis.main_loop(main_locals, ipy=ipy)
    return locals(), main_execstr


if __name__ == '__main__':
    """
        The Developer Script
            A command line interface to almost everything

            -w     # wait / show the gui / figures are visible
            --cmd  # ipython shell to play with variables
            -t     # run list of tests

            Examples:
                ./dev.py -t query -w
    """
    multiprocessing.freeze_support()  # for win32
    #
    # IBEIS Main
    print('++dev')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    #
    # ______________________________
    # + Common variables for IPython
    SNIPPITS = True
    if SNIPPITS:
        # Get snippet variables
        ibs = main_locals['ibs']
        if 'back' in main_locals:
            back = main_locals['back']
            if back is not None:
                front = back.front
                ui = front.ui
        #ibs.dump_tables()
        valid_rids = ibs.get_valid_rids()
        valid_gids = ibs.get_valid_gids()
        valid_nids = ibs.get_valid_nids()
        valid_nid_list  = ibs.get_roi_nids(valid_rids)
        valid_rid_names = ibs.get_roi_names(valid_rids)
        valid_rid_gtrues = ibs.get_roi_groundtruth(valid_rids)
    # L___________________________
    #
    #
    # Development code
    RUN_DEV = True  # RUN_DEV = '__IPYTHON__' in vars()
    if RUN_DEV:
        dev_locals, main_execstr = rundev(main_locals)
        dev_execstr = utool.execstr_dict(dev_locals, 'dev_locals')
        execstr = dev_execstr + '\n' + main_execstr
        exec(execstr)
    # Memory profile
    if '--memprof' in sys.argv:
        utool.print_resource_usage()
        utool.memory_profile()
