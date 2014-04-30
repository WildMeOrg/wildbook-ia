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
# Tools
from plottool import draw_func2 as df2
# IBEIS
from ibeis.dev import main_helpers
from ibeis.viz import interact
from ibeis.dev import experiment_configs
from ibeis.dev import experiment_harness
from ibeis.dev import results_all
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=False)

if not 'back' in vars():
    back = None


@devcmd
def vdd(ibs=None, qrid_list=None):
    utool.view_directory(ibs.get_dbdir())


@devcmd('show')
def show_rids(ibs, qrid_list):
    for rid in qrid_list:
        interact.ishow_chip(ibs, rid, fnum=df2.next_fnum())


@devcmd()
def change_names(ibs, qrid_list):
    #new_name = utool.get_arg('--name', str, default='<name>_the_<species>')
    new_name = utool.get_arg('--name', str, default='glob')
    for rid in qrid_list:
        ibs.print_name_table()
        #(nid,) = ibs.add_names((new_name,))
        ibs.set_roi_props((rid,), 'name', (new_name,))
        ibs.print_name_table()
        ibs.print_roi_table()
    new_nid = ibs.get_name_nids(new_name, ensure=False)
    if back is not None:
        back.select_nid(new_nid)


@devcmd('query')
def query_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        interact.ishow_qres(ibs, qres, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('sver')
def sver_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        rid2 = qres.get_top_rids()[0]
        interact.ishow_sver(ibs, qrid, rid2, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('cfg')
def printcfg(ibs, qrid_list):
    ibs.cfg.printme3()
    print(ibs.cfg.query_cfg.get_uid())


@devcmd('hsdbs')
def list_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import get_unconverted_hsdbs
    get_unconverted_hsdbs()


@devcmd('convert')
def convert_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import injest_unconverted_hsdbs_in_workdir
    injest_unconverted_hsdbs_in_workdir()


@devcmd
def delete_all_feats(ibs, *args):
    ibsfuncs.delete_all_features(ibs)


@devcmd
def delete_all_chips(ibs, *args):
    ibsfuncs.delete_all_chips(ibs)


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
    input_test_list = params.args.tests[:]
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

    if intest('info'):
        print(ibs.get_infostr())
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
            experiment_harness.test_configurations(ibs, qrid_list, [test_cfg_name],  df2.next_fnum())

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


@profile
def dev_main():
    global back
    print('++dev')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    ibs  = main_locals['ibs']
    back = main_locals['back']

    fnum = 1
    qrid_list = main_helpers.get_test_qrids(ibs)
    ibs.prep_qreq_db(qrid_list)

    expt_locals = run_experiments(ibs, qrid_list)

    if '--devmode' in sys.argv:
        devfunc_locals = devfunc(ibs, qrid_list)
        exec(utool.execstr_dict(devfunc_locals, 'devfunc_locals'))

    if not '--nopresent' in sys.argv:
        df2.present()
    ipy = (not '--gui' in sys.argv) or ('--cmd' in sys.argv)
    main_execstr = ibeis.main_loop(main_locals, ipy=ipy)
    return locals(), main_execstr

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    """
        The Developer Script
            A command line interface to almost everything

            -w     # wait / show the gui / figures are visible
            --cmd  # ipython shell to play with variables
            -t     # run list of tests

            Examples:
                ./dev.py -t query -w
    """
    utool.print_resource_usage()
    dev_locals, main_execstr = dev_main()
    dev_execstr = utool.execstr_dict(dev_locals, 'dev_locals')
    execstr = dev_execstr + '\n' + main_execstr

    utool.print_resource_usage()
    exec(execstr)


"""
Snippets:

rid_list = ibs.get_valid_rids()
gid_list = ibs.get_valid_gids()
nid_list = ibs.get_valid_nids()

"""
