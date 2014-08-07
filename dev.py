#!/usr/bin/env python2.7
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
# Dev
from _devscript import devcmd,  DEVCMD_FUNCTIONS
from utool.util_six import get_funcname
from utool.util_six import *  # NOQA
import utool
import ibeis
if __name__ == '__main__':
    multiprocessing.freeze_support()
    ibeis._preload()
    from ibeis.all_imports import *  # NOQA
#utool.util_importer.dynamic_import(__name__, ('_devcmds_ibeis', None),
#                                   developing=True)
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
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=False)

#--------------------
# RUN DEV EXPERIMENTS
#--------------------


#@utool.indent_func('[dev]')
#@profile
def run_experiments(ibs, qaid_list):
    """
    This function runs tests passed in with the -t flag
    """
    print('\n')
    print('[dev] run_experiments')
    print('==========================')
    print('RUN EXPERIMENTS %s' % ibs.get_dbname())
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
            ret2 = testname in params.unknown  # Let unparsed args count towards tests
            if ret or ret2:
                if ret:
                    input_test_list.remove(testname)
                else:
                    ret = ret2
                print('+===================')
                print('| running testname = %s' % (args,))
                return ret
        return False

    valid_test_helpstr_list.append('    # --- Simple Tests ---')

    # Explicit (simple) test functions
    if intest('export'):
        export(ibs)
    if intest('dbinfo'):
        dbinfo.get_dbinfo(ibs)
    if intest('info'):
        print(ibs.get_infostr())
    if intest('printcfg'):
        printcfg(ibs)
    if intest('tables'):
        ibs.print_tables()
    if intest('imgtbl'):
        ibs.print_image_table()

    valid_test_helpstr_list.append('    # --- Decor Tests ---')

    locals_ = locals()

    # Implicit (decorated) test functions
    for (func_aliases, func) in DEVCMD_FUNCTIONS:
        if intest(*func_aliases):
            with utool.Indenter('[dev.' + get_funcname(func) + ']'):
                print('[dev] qid_list=%r' % (qaid_list,))
                ret = func(ibs, qaid_list)
                # Add variables returned by the function to the
                # "local scope" (the exec scop)
                if hasattr(ret, 'items'):
                    for key, val in ret.items():
                        if utool.is_valid_varname(key):
                            locals_[key] = val

    valid_test_helpstr_list.append('    # --- Config Tests ---')

    # Config driven test functions
    # Allow any testcfg to be in tests like: vsone_1 or vsmany_3
    for test_cfg_name in experiment_configs.TEST_NAMES:
        if intest(test_cfg_name):
            test_cfg_name_list = [test_cfg_name]
            fnum = df2.next_fnum()
            experiment_harness.test_configurations(ibs, qaid_list, test_cfg_name_list, fnum)

    valid_test_helpstr_list.append('    # --- Help ---')

    if intest('help'):
        print('valid tests are:')
        print('\n'.join(valid_test_helpstr_list))
        return locals_

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)
    return locals_


#-------------------
# CUSTOM DEV FUNCS
#-------------------


__ALLRES_CACHE__ = {}


def get_allres(ibs, qaid_list):
    print('[dev] get_allres')
    allres_cfgstr = ibs.qreq.get_cfgstr()
    try:
        allres = __ALLRES_CACHE__[allres_cfgstr]
    except KeyError:
        qaid2_qres = ibs.query_all(qaid_list)
        allres = results_all.init_allres(ibs, qaid2_qres)
    # Cache save
    __ALLRES_CACHE__[allres_cfgstr] = allres
    return allres


#------------------
# DEV DEVELOPMENT
#------------------
# This is where you write all of the functions that will become pristine
# and then go in _devcmds_ibeis.py


@devcmd('dists', 'dist', 'desc_dists')
def desc_dists(ibs, qaid_list):
    """ Plots the distances between matching descriptors
        with groundtruth (true/false) data """
    print('[dev] desc_dists')
    allres = get_allres(ibs, qaid_list)
    # Get the descriptor distances of true matches
    orgtype_list = ['top_false', 'true']
    disttype = 'L2'
    orgres2_distmap = results_analyzer.get_orgres_desc_match_dists(allres, orgtype_list)
    results_analyzer.print_desc_distances_map(orgres2_distmap)
    #true_desc_dists  = orgres2_distmap['true']['L2']
    #false_desc_dists = orgres2_distmap['false']['L2']
    #scores_list = [false_desc_dists, true_desc_dists]
    dists_list = [orgres2_distmap[orgtype][disttype] for orgtype in orgtype_list]
    dists_lbls = orgtype_list
    dists_markers = ['x', 'o--']
    plottool.plots.draw_scores_cdf(dists_list, dists_lbls, dists_markers)
    df2.set_figtitle('Descriptor Distance CDF d(x)' + ibs.qreq.get_cfgstr())
    return locals()


@devcmd('scores', 'score')
def annotationmatch_scores(ibs, qaid_list):
    print('[dev] annotationmatch_scores')
    allres = get_allres(ibs, qaid_list)
    # Get the descriptor distances of true matches
    orgtype_list = ['false', 'true']
    orgtype_list = ['top_false', 'top_true']
    #markers_map = {'false': 'o', 'true': 'o-', 'top_true': 'o-', 'top_false': 'o'}
    markers_map = defaultdict(lambda: 'o')
    cmatch_scores_map = results_analyzer.get_orgres_annotationmatch_scores(allres, orgtype_list)
    results_analyzer.print_annotationmatch_scores_map(cmatch_scores_map)
    #true_cmatch_scores  = cmatch_scores_map['true']
    #false_cmatch_scores = cmatch_scores_map['false']
    scores_list = [cmatch_scores_map[orgtype] for orgtype in orgtype_list]
    scores_lbls = orgtype_list
    scores_markers = [markers_map[orgtype] for orgtype in orgtype_list]
    plottool.plots.draw_scores_cdf(scores_list, scores_lbls, scores_markers)
    df2.set_figtitle('Chipmatch Scores ' + ibs.qreq.get_cfgstr())
    return locals()


@devcmd('inspect')
def inspect_matches(ibs, qaid_list):
    print('<inspect_matches>')
    from ibeis.gui import inspect_gui
    from ibeis.viz.interact import interact_qres2  # NOQA
    allres = get_allres(ibs, qaid_list)
    guitool.ensure_qapp()
    tblname = 'qres'
    qaid2_qres = allres.qaid2_qres
    ranks_lt = 5
    # This object is created inside QresResultsWidget
    #qres_api = inspect_gui.make_qres_api(ibs, qaid2_qres)  # NOQA
    # This is where you create the result widigt
    print('[inspect_matches] make_qres_widget')
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, ranks_lt=ranks_lt)
    print('[inspect_matches] show')
    qres_wgt.show()
    print('[inspect_matches] raise')
    qres_wgt.raise_()
    #query_review = interact_qres2.Interact_QueryResult(ibs, qaid2_qres)
    #self = interact_qres2.Interact_QueryResult(ibs, qaid2_qres, ranks_lt=ranks_lt)
    print('</inspect_matches>')
    # simulate double click
    qres_wgt._on_click(qres_wgt.model.index(2, 2))
    #qres_wgt._on_doubleclick(qres_wgt.model.index(2, 0))
    return locals()


@devcmd('gv')
def gvcomp(ibs, qaid_list):
    """
    GV = With gravity vector
    RI = With rotation invariance
    """
    print('[dev] gvcomp')
    assert isinstance(ibs, IBEISControl.IBEISController), 'bad input'  # let jedi know whats up
    def testcomp(ibs, qaid_list):
        allres = get_allres(ibs, qaid_list)
        for qaid in qaid_list:
            qres = allres.get_qres(qaid)
            interact.ishow_qres(ibs, qres,
                                annote_mode=2,
                                in_image=True,
                                figtitle='Qaid=%r %s' % (qres.qaid, qres.cfgstr)
                                )
        return allres
    ibs_GV = ibs
    ibs_RI = ibs.clone_handle(nogravity_hack=True)
    #utool.embed()

    allres_GV = testcomp(ibs_GV, qaid_list)
    allres_RI = testcomp(ibs_RI, qaid_list)
    return locals()


@devcmd('upsize')
@profile
def up_dbsize_expt(ibs, qaid_list):
    """
    Input:
        ibs       - IBEISController object
        qaid_list - list of annotation-ids to query
    Plots the scores/ranks of correct matches while varying the size of the
    database.

    python dev.py -t upsize --db PZ_Mothers --qaid 1:30:3 --cmd
    >>> qaid_list = ibs.get_valid_aids()
    """
    print('updbsize_expt')
    # clamp the number of groundtruth to test
    clamp_gt = utool.get_arg('--clamp-gt', int, 1)
    clamp_ft = utool.get_arg('--clamp-gf', int, 1)

    seed_ = 143039
    np.random.seed(seed_)

    # List of database sizes to test
    num_samp = utool.get_arg('--num-samples', int, 5)
    samp_max = ibs.get_num_names()
    #samp_min = 10  # max(samp_max // len(qaid_list), 10)
    samp_min = 2  # max(samp_max // len(qaid_list), 10)
    sample_dbsizes = utool.sample_domain(samp_min, samp_max, num_samp)
    # Get list of true and false matches for every query annotation
    qaid_trues_list  = ibs.get_annot_groundtruth(qaid_list, noself=True,
                                                 is_exemplar=True)
    qaid_falses_list_ = ibs.get_annot_groundfalse(qaid_list)
    qaid_falses_list = []
    for count, false_aids_ in enumerate(qaid_falses_list_):
        aids_list_ = ibsfuncs.group_annots_by_known_names_nochecks(ibs, false_aids_)
        aids_list = [None for x in range(len(aids_list_))]
        for ix in range(len(aids_list_)):
            arr = np.array(aids_list_[ix]).copy()
            np.random.shuffle(arr)
            aids_list[ix] = arr.tolist()
        sample_aids, _ = utool.sample_zip(aids_list, 1, allow_overflow=True, per_bin=1)
        sample = sample_aids[0]
        qaid_falses_list.append(sample)
    # output containers
    upscores_dict = utool.ddict(lambda: utool.ddict(list))
    # For each query annotation, and its true and false set
    query_iter = zip(qaid_list, qaid_trues_list, qaid_falses_list)
    # Get a rough idea of how many queries will be run
    nGtPerAid = (nGt if (clamp_gt is None or nGt < clamp_gt) else clamp_gt
                 for nGt in map(len, qaid_trues_list))
    nTotal = len(sample_dbsizes) * sum(nGtPerAid)
    count = 0
    # Create a progress marking function
    progkw = {'nTotal': nTotal, 'flushfreq': 20, 'approx': True}
    mark_, end_ = utool.log_progress('[upscale] progress: ',  **progkw)
    for qaid, true_aids, false_aids in query_iter:
        # For each set of false matches (of varying sizes)
        true_sample  = utool.random_sample(false_aids, clamp_gt)
        np.random.shuffle(false_aids)
        for dbsize in sample_dbsizes:
            if dbsize > len(false_aids):

                break
            false_sample = false_aids[:dbsize]
            # For each true match
            for gt_aid in true_sample:
                assert len(false_sample) == len(set(ibs.get_annot_nids(false_sample))), \
                    'num false_aids != false_nids'
                count += 1
                mark_(count)
                # Execute query
                qres = ibs._query_chips([qaid], false_sample + [gt_aid])[qaid]
                # Elicit information
                score = qres.get_gt_scores(gt_aids=[gt_aid])[0]
                # Append result
                upscores_dict[(qaid, gt_aid)]['dbsizes'].append(dbsize)
                upscores_dict[(qaid, gt_aid)]['score'].append(score)
    end_()

    if not utool.get_flag('--noshow'):
        colors = df2.distinct_colors(len(upscores_dict))
        df2.figure(fnum=1, doclf=True, docla=True)
        for ix, ((qaid, gt_aid), upscores) in enumerate(upscores_dict.items()):
            xdata = upscores['dbsizes']
            ydata = upscores['score']
            df2.plt.plot(xdata, ydata, 'o-', color=colors[ix])
        figtitle = 'Effect of Database Size on Match Scores'
        figtitle += '\n' + ibs.get_dbname()
        figtitle += '\n' + ibs.cfg.query_cfg.get_cfgstr()
        df2.set_figtitle(figtitle, font='large')
        df2.set_xlabel('# Annotations in database')
        df2.set_ylabel('Groundtruth Match Scores (annot-vs-annot)')
        df2.dark_background()

    # TODO: Should be separate function. Previous code should be intergrated
    # into the experiment_harness
    locals_ = locals()
    return locals_  # return in dict format for execstr_dict


def get_ibslist(ibs):
    print('[dev] get_ibslist')
    ibs_GV  = ibs
    ibs_RI  = ibs.clone_handle(nogravity_hack=True)
    ibs_RIW = ibs.clone_handle(nogravity_hack=True, gravity_weighting=True)
    ibs_list = [ibs_GV, ibs_RI, ibs_RIW]
    return ibs_list


@devcmd('gv_scores')
def compgrav_annotationmatch_scores(ibs, qaid_list):
    print('[dev] compgrav_annotationmatch_scores')
    ibs_list = get_ibslist(ibs)
    for ibs_ in ibs_list:
        annotationmatch_scores(ibs_, qaid_list)


#------------------
# DEV MAIN
#------------------

def dev_snippets(main_locals):
    """ Common variables for convineince when interacting with IPython """
    print('[dev] dev_snippets')
    species = 'zebra_grevys'
    quick = True
    fnum = 1
    # Get reference to IBEIS Controller
    ibs = main_locals['ibs']
    if 'back' in main_locals:
        # Get reference to GUI Backend
        back = main_locals['back']
        if back is not None:
            # Get reference to GUI Frontend
            front = getattr(back, 'front', None)
            ibswgt = front
            view = ibswgt.views['images']
            model = ibswgt.models['names_tree']
            selection_model = view.selectionModel()
    if ibs is not None:
        #ibs.dump_tables()
        aid_list = ibs.get_valid_aids()
        gid_list = ibs.get_valid_gids()
        nid_list = ibs.get_valid_nids()
        #valid_nid_list   = ibs.get_annot_nids(aid_list)
        #valid_aid_names  = ibs.get_annot_names(aid_list)
        #valid_aid_gtrues = ibs.get_annot_groundtruth(aid_list)
    return locals()


def devfunc(ibs, qaid_list):
    """ Function for developing something """
    print('[dev] devfunc')
    allres = get_allres(ibs, qaid_list)
    locals_ = locals()
    #locals_.update(annotationmatch_scores(ibs, qaid_list))
    return locals_


def run_dev(main_locals):
    print('[dev] run_dev')
    # Get references to controller
    ibs  = main_locals['ibs']
    if ibs is not None:
        # Get aids marked as test cases
        qaid_list = main_helpers.get_test_qaids(ibs)
        print('[run_dev] test_qaids = %r' % qaid_list)
        print('[run_dev] len(test_qaids) = %d' % len(qaid_list))
        # Warn on no test cases
        try:
            assert len(qaid_list) > 0, 'assert!'
        except AssertionError as ex:
            utool.printex(ex, 'len(qaid_list) = 0', iswarning=True)
            #qaid_list = ibs.get_valid_aids()[0]

        if len(qaid_list) > 0 or True:
            # Prepare the IBEIS controller for querys
            if len(qaid_list) > 0:
                ibs.prep_qreq_db(qaid_list)
            # Run the dev experiments
            expt_locals = run_experiments(ibs, qaid_list)
            # Add experiment locals to local namespace
            execstr_locals = utool.execstr_dict(expt_locals, 'expt_locals')
            exec(execstr_locals)
            if '--devmode' in sys.argv:
                # Execute the dev-func and add to local namespace
                devfunc_locals = devfunc(ibs, qaid_list)
                exec(utool.execstr_dict(devfunc_locals, 'devfunc_locals'))

    return locals()


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
    #
    # Run IBEIS Main, create controller, and possibly gui
    print('++dev')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    #utool.set_process_title('IBEIS_dev')

    #
    #
    # Load snippet variables
    SNIPPITS = True
    if SNIPPITS:
        snippet_locals = dev_snippets(main_locals)
        snippet_execstr = utool.execstr_dict(snippet_locals, 'snippet_locals')
        exec(snippet_execstr)

    #
    #
    # Development code
    RUN_DEV = True  # RUN_DEV = '__IPYTHON__' in vars()
    if RUN_DEV:
        dev_locals = run_dev(main_locals)
        dev_execstr = utool.execstr_dict(dev_locals, 'dev_locals')
        exec(dev_execstr)

    #
    #
    # Main Loop (IPython interaction, or some exec loop)
    if '--nopresent' not in sys.argv or '--noshow' in sys.argv:
        df2.present()
    ipy = ('--gui' not in sys.argv) or ('--cmd' in sys.argv)
    main_execstr = ibeis.main_loop(main_locals, ipy=ipy)
    exec(main_execstr)

    #
    #
    # Memory profile
    if '--memprof' in sys.argv:
        utool.print_resource_usage()
        utool.memory_profile()

    print('exiting dev')
