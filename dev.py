#!/usr/bin/env python2.7
"""
DEV SCRIPT

This is a hacky script meant to be run mostly automatically with the option of
interactions.

dev.py is supposed to be a developer non-gui interface into the IBEIS software.
dev.py runs experiments and serves as a scratchpad for new code and quick scripts


CommandLine:
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg score_method:nsum prescore_method:nsum
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg fg_on=True
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
# Dev
from _devscript import devcmd,  DEVCMD_FUNCTIONS, DEVPRECMD_FUNCTIONS
import utool as ut
from utool.util_six import get_funcname
import utool
#from ibeis.model.hots import smk
import plottool
import plottool as pt
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
from ibeis.dev import experiment_configs
from ibeis.dev import experiment_harness
from ibeis.dev import results_all
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]')


#------------------
# DEV DEVELOPMENT
#------------------
# This is where you write all of the functions that will become pristine
# and then go in _devcmds_ibeis.py


@devcmd('tune', 'autotune')
def tune_flann(ibs, qaid_list, daid_list=None):
    r"""
    CommandLine:
        python dev.py -t tune --db PZ_MTEST
        python dev.py -t tune --db GZ_ALL
        python dev.py -t tune --db GIR_Tanya
        python dev.py -t tune --db PZ_Master0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from _devscript import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = func_wrapper()
        >>> # verify results
        >>> print(result)
    """
    all_aids = ibs.get_valid_aids()
    vecs = np.vstack(ibs.get_annot_vecs(all_aids))
    print('Tunning flann for species={species}:'.format(species=ibs.get_database_species(all_aids)))
    tuned_params = vt.tune_flann(vecs,
                                 target_precision=.98,
                                 build_weight=0.05,
                                 memory_weight=0.00,
                                 sample_fraction=0.1)
    tuned_params

    #tuned_params2 = vt.tune_flann(vecs,
    #                              target_precision=.90,
    #                              build_weight=0.001,
    #                              memory_weight=0.00,
    #                              sample_fraction=0.5)
    #tuned_params2


@devcmd('incremental', 'inc')
def incremental_test(ibs, qaid_list, daid_list=None):
    """
    Adds / queries new images one at a time to a clean test database.
    Tests the complete system.

    Args:
        ibs       (list) : IBEISController object
        qaid_list (list) : list of annotation-ids to query

    CommandLine:
        python dev.py -t inc --db PZ_MTEST --qaid 1:30:3 --cmd

        python dev.py --db PZ_MTEST --allgt --cmd

        python dev.py --db PZ_MTEST --allgt -t inc

        python dev.py -t inc --db PZ_MTEST --qaid 1:30:3 --cmd

        python dev.py -t inc --db GZ_ALL --ninit 100 --noqcache

        python dev.py -t inc --db PZ_MTEST --noqcache --interactive-after 40
        python dev.py -t inc --db PZ_Master0 --noqcache --interactive-after 10000 --ninit 400

    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
    """
    from ibeis.model.hots import automated_matcher
    ibs1 = ibs
    num_initial = ut.get_argval('--ninit', type_=int, default=0)
    return automated_matcher.incremental_test(ibs1, num_initial)


@devcmd('scores', 'score')
def annotationmatch_scores(ibs, qaid_list, daid_list=None):
    """
    TODO: plot the difference between the top true score and the next best false score
    CommandLine:
        ib
        python dev.py -t scores --db PZ_MTEST --allgt -w --show
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg fg_on:True
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db PZ_Master0 --allgt -w --show --cfg codename='vsmany' fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show

        python dev.py -t scores --db GZ_ALL --allgt -w --show --cfg codename='vsmany'
        python dev.py -t scores --db PZ_Master0 --allgt -w --show --cfg codename='vsmany'


    """
    print('[dev] annotationmatch_scores')
    #allres = results_all.get_allres(ibs, qaid_list, daid_list)
    #ut.embed()
    #orgres = allres.allorg['rank0_true']
    qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list)
    x_data, y_data = results_all.get_stem_data(ibs, qaid2_qres)
    pt.plots.plot_stems(x_data, y_data)
    pt.present()
    pt.show()
    #locals_ = viz_allres_annotation_scores(allres)
    locals_ = locals()
    return locals_


def viz_allres_annotation_scores(allres):
    """
    chip vs chip scores
    """
    # Get the descriptor distances of true matches
    #orgtype_list = ['false', 'true']
    #orgtype_list = ['top_false', 'top_true']
    orgtype_list = ['rank0_false', 'rank0_true']
    #markers_map = {'false': 'o', 'true': 'o-', 'top_true': 'o-', 'top_false': 'o'}
    markers_map = defaultdict(lambda: 'o')
    cmatch_scores_map = results_analyzer.get_orgres_annotationmatch_scores(allres, orgtype_list)
    results_analyzer.print_annotationmatch_scores_map(cmatch_scores_map)
    #true_cmatch_scores  = cmatch_scores_map['true']
    #false_cmatch_scores = cmatch_scores_map['false']
    scores_list = [cmatch_scores_map[orgtype] for orgtype in orgtype_list]
    scores_lbls = orgtype_list
    scores_markers = [markers_map[orgtype] for orgtype in orgtype_list]
    plottool.plots.plot_sorted_scores(scores_list, scores_lbls, scores_markers)
    df2.set_figtitle('Chip-vs-Chip Scores ' + allres.make_title())
    return locals()


@devcmd('upsize', 'upscale')
def up_dbsize_expt(ibs, qaid_list, daid_list=None):
    """
    Plots the scores/ranks of correct matches while varying the size of the
    database.

    Args:
        ibs       (list) : IBEISController object
        qaid_list (list) : list of annotation-ids to query

    CommandLine:
        python dev.py -t upsize --db PZ_MTEST --qaid 1:30:3 --cmd

    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> #ibs = ibeis.opendb('PZ_FlankHack')
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
    """
    print('updbsize_expt')
    upsizekw = dict(
        num_samp=utool.get_argval('--num-samples', int, 5),
        clamp_gt=utool.get_argval('--clamp-gt', int, 1),
        clamp_gf=utool.get_argval('--clamp-gf', int, 1),
        seed=143039
    )
    upsizetup = ibs.get_upsize_data(qaid_list, daid_list, **upsizekw)
    qaid_list, qaid_trues_list, qaid_false_samples_list, nTotal = upsizetup
    # Create a progress marking function
    progkw = dict(nTotal=nTotal, flushfreq=20, approx=False)
    mark_, end_ = utool.log_progress('[upscale] progress: ',  **progkw)
    count = 0
    # Set up output containers and run test iterations
    upscores_dict = utool.ddict(lambda: utool.ddict(list))
    input_iter = zip(qaid_list, qaid_trues_list, qaid_false_samples_list)
    # For each query annotation runs it as a query multiple times
    # each time it increases the number of false annotation in the database
    # so we can see how a score degrades as the number of false
    # database annotations increases
    for qaid, true_aids, false_aids_samples in input_iter:
        #print('qaid = %r' % (qaid,))
        #print('true_aids=%r' % (true_aids,))
        # For each true match and false sample
        for gt_aid, false_sample in utool.iprod(true_aids, false_aids_samples):
            #print('  gt_aid=%r' % (gt_aid,))
            #print('  false_sample=%r' % (false_sample,))
            mark_(count)
            count += 1
            # Execute query
            daids = false_sample + [gt_aid]
            qres = ibs._query_chips4([qaid], daids)[qaid]
            # Elicit information
            score = qres.get_gt_scores(gt_aids=[gt_aid])[0]
            # Append result
            upscores_dict[(qaid, gt_aid)]['dbsizes'].append(len(false_sample))
            upscores_dict[(qaid, gt_aid)]['score'].append(score)
    end_()

    if not utool.get_argflag('--noshow'):
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
        dumpkw = {
            'subdir'    : 'upsize',
            'quality'   : False,
            'overwrite' : True,
            'verbose'   : 0
        }
        figdir = ibs.get_fig_dir()
        ph.dump_figure(figdir, **dumpkw)

    #---------
    # Find highest
    if False:
        dbsample_index = 1
        line_index = 0

        highscore = 0
        highpair = None
        none_pairs = []
        pair_list  = []
        score_list = []
        for pair, dict_ in six.iteritems(upscores_dict):
            scores = dict_['score']
            if any([s is None for s in scores]):
                none_pairs.append(pair)
            if dbsample_index >= len(scores):
                continue
            score = scores[dbsample_index]
            if score is None:
                continue
            score_list.append(score)
            pair_list.append(pair)

        sorted_tups = sorted(list(zip(score_list, pair_list)))
        print(sorted_tups[0])
        print(sorted_tups[-1])

        qaid, gt_aid = sorted_tups[line_index][1]
        print('qaid = %r' % qaid)
        print('gt_aid = %r' % gt_aid)
        index = qaid_list.index(qaid)
        print(index)
        false_aids_samples = qaid_false_samples_list[index]
        false_sample = false_aids_samples[dbsample_index]
        print(false_sample)
        daids = false_sample + [gt_aid]
        qres = ibs._query_chips4([qaid], daids)[qaid]
        #for score in scores:
        #    if score is None:
        #        continue
        #    if score > highscore:
        #        highpair = pair
        #        highscore = score
        #print(scores)

    # TODO: Should be separate function. Previous code should be intergrated
    # into the experiment_harness
    locals_ = locals()
    return locals_  # return in dict format for execstr_dict


@devcmd('dists', 'dist', 'vecs_dist')
def vecs_dist(ibs, qaid_list, daid_list=None):
    """
    Plots the distances between matching descriptors
        with groundtruth (true/false) data


    True distances are spatially verified descriptor matches
    Top false distances distances are spatially verified descriptor matches

    CommandLine:
        python dev.py -t vecs_dist --db NAUT_test --allgt -w --show
        python dev.py -t vecs_dist --db PZ_MTEST --allgt -w --show
        python dev.py -t vecs_dist --db GZ_ALL --allgt -w --show
    """
    print('[dev] vecs_dist')
    allres = results_all.get_allres(ibs, qaid_list)
    # Get the descriptor distances of true matches
    orgtype_list = ['top_false', 'true']
    disttype = 'L2'
    # Get descriptor match distances
    orgres2_distmap = results_analyzer.get_orgres_desc_match_dists(allres, orgtype_list)
    results_analyzer.print_desc_distances_map(orgres2_distmap)
    #true_desc_dists  = orgres2_distmap['true']['L2']
    #false_desc_dists = orgres2_distmap['false']['L2']
    #scores_list = [false_desc_dists, true_desc_dists]
    dists_list = [orgres2_distmap[orgtype][disttype] for orgtype in orgtype_list]
    dists_lbls = orgtype_list
    dists_markers = ['x--', 'o--']
    pt.plots.plot_sorted_scores(dists_list, dists_lbls, dists_markers)
    df2.set_figtitle('Descriptor Sorted Distance d(x)' + allres.get_cfgstr())
    return locals()


@devcmd('vsone_gt')
def vsone_gt(ibs, qaid_list, daid_list=None):
    """
    dev.py --db PZ_MTEST --allgt --cmd
    """
    cfgdict = dict(fg_on=True)
    allres = results_all.get_allres(ibs, qaid_list, daid_list, cfgdict)
    #orgtype_list = ['top_false', 'top_true']
    org_top_false = allres.get_orgtype('rank0_false')
    top_false_aid_pairs = zip(org_top_false.qaids, org_top_false.aids)
    gtaids_list = ibs.get_annot_groundtruth(qaid_list, daid_list=daid_list)
    # Add groundtruth pairs to query
    qaid2_vsoneaids_ = dict(zip(qaid_list, gtaids_list))
    qaid2_vsoneaids = ut.ddict(list)
    # Add problem cases
    for qaid, daid in top_false_aid_pairs:
        qaid2_vsoneaids[qaid].append(daid)
        qaid2_vsoneaids[qaid].extend(qaid2_vsoneaids_.get(qaid, []))
    cfgdict = dict(codename='vsone')
    qaid2_vsoneqres = {}
    for qaid, vsoneaids in six.iteritems(qaid2_vsoneaids):
        qres = ibs._query_chips4([qaid], vsoneaids, cfgdict=cfgdict)[qaid]
        qaid2_vsoneqres[qaid] = qres
    vsone_allres = results_all.init_allres(ibs, qaid2_vsoneqres)
    viz_allres_annotation_scores(vsone_allres)
    #cmatch_scores_map = results_analyzer.get_orgres_annotationmatch_scores(allres, orgtype_list)


@devcmd('inspect')
def inspect_matches(ibs, qaid_list, daid_list):
    print('<inspect_matches>')
    from ibeis.gui import inspect_gui
    return inspect_gui.test_inspect_matches(ibs, qaid_list, daid_list)


@devcmd('gv')
def gvcomp(ibs, qaid_list, daid_list):
    """
    GV = With gravity vector
    RI = With rotation invariance
    """
    print('[dev] gvcomp')
    assert isinstance(ibs, IBEISControl.IBEISController), 'bad input'  # let jedi know whats up
    def testcomp(ibs, qaid_list):
        allres = results_all.get_allres(ibs, qaid_list)
        for qaid in qaid_list:
            qres = allres.get_qres(qaid)
            qres.ishow_top(ibs, annot_mode=2, in_image=True, figtitle='Qaid=%r %s' % (qres.qaid, qres.cfgstr))
        return allres
    ibs_GV = ibs
    ibs_RI = ibs.clone_handle(nogravity_hack=True)
    #utool.embed()

    allres_GV = testcomp(ibs_GV, qaid_list)
    allres_RI = testcomp(ibs_RI, qaid_list)
    return locals()


def get_ibslist(ibs):
    print('[dev] get_ibslist')
    ibs_GV  = ibs
    ibs_RI  = ibs.clone_handle(nogravity_hack=True)
    ibs_RIW = ibs.clone_handle(nogravity_hack=True, gravity_weighting=True)
    ibs_list = [ibs_GV, ibs_RI, ibs_RIW]
    return ibs_list


@devcmd('gv_scores')
def compgrav_annotationmatch_scores(ibs, qaid_list, daid_list):
    print('[dev] compgrav_annotationmatch_scores')
    ibs_list = get_ibslist(ibs)
    for ibs_ in ibs_list:
        annotationmatch_scores(ibs_, qaid_list)

#--------------------
# RUN DEV EXPERIMENTS
#--------------------


def run_devprecmds():
    input_precmd_list = params.args.tests[:]
    valid_precmd_list = []
    def intest(*args, **kwargs):
        for precmd_name in args:
            valid_precmd_list.append(precmd_name)
            ret = precmd_name in input_precmd_list
            ret2 = precmd_name in params.unknown  # Let unparsed args count towards tests
            if ret or ret2:
                if ret:
                    input_precmd_list.remove(precmd_name)
                else:
                    ret = ret2
                print('+===================')
                print('| running precmd = %s' % (args,))
                return ret
        return False

    # Implicit (decorated) test functions
    for (func_aliases, func) in DEVPRECMD_FUNCTIONS:
        if intest(*func_aliases):
            with utool.Indenter('[dev.' + get_funcname(func) + ']'):
                func()
                print('Exiting after first precommand')
            sys.exit(1)


#@utool.indent_func('[dev]')
def run_devcmds(ibs, qaid_list, daid_list):
    """
    This function runs tests passed in with the -t flag
    """
    print('\n')
    print('[dev] run_devcmds')
    print('==========================')
    print('RUN EXPERIMENTS %s' % ibs.get_dbname())
    print('==========================')
    input_test_list = params.args.tests[:]
    print('input_test_list = %r' % (input_test_list,))
    # fnum = 1

    valid_test_list = []  # build list for printing in case of failure
    valid_test_helpstr_list = []  # for printing

    def mark_test_handled(testname):
        input_test_list.remove(testname)

    def intest(*args, **kwargs):
        helpstr = kwargs.get('help', '')
        valid_test_helpstr_list.append('   -t ' + ', '.join(args) + helpstr)
        for testname in args:
            valid_test_list.append(testname)
            ret = testname in input_test_list
            ret2 = testname in params.unknown  # Let unparsed args count towards tests
            if ret or ret2:
                if ret:
                    mark_test_handled(testname)
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
    if intest('headers', 'schema'):
        ibs.db.print_schema()
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
            funcname = get_funcname(func)
            with utool.Indenter('[dev.' + funcname + ']'):
                with utool.Timer(funcname):
                    #print('[dev] qid_list=%r' % (qaid_list,))
                    # FIXME: , daid_list
                    ret = func(ibs, qaid_list, daid_list)
                    # Add variables returned by the function to the
                    # "local scope" (the exec scop)
                    if hasattr(ret, 'items'):
                        for key, val in ret.items():
                            if utool.is_valid_varname(key):
                                locals_[key] = val

    valid_test_helpstr_list.append('    # --- Config Tests ---')

    # ------
    # RUNS EXPERIMENT HARNESS OVER VALID TESTNAMES SPECIFIED WITH -t
    # ------

    # Config driven test functions
    # Allow any testcfg to be in tests like: vsone_1 or vsmany_3
    test_cfg_name_list = []
    for test_cfg_name in experiment_configs.TEST_NAMES:
        if intest(test_cfg_name):
            test_cfg_name_list.append(test_cfg_name)
    # Hack to allow for very customized harness tests
    for testname in input_test_list[:]:
        if testname.startswith('custom:'):
            test_cfg_name_list.append(testname)
            mark_test_handled(testname)
    if len(test_cfg_name_list):
        fnum = df2.next_fnum()
        # Run Experiments
        experiment_harness.test_configurations(ibs, qaid_list, daid_list, test_cfg_name_list)

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


#------------------
# DEV MAIN
#------------------

def dev_snippets(main_locals):
    """ Common variables for convineince when interacting with IPython """
    print('[dev] dev_snippets')
    species = constants.Species.ZEB_GREVY
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
        #nid_list = ibs.get_valid_nids()
        #valid_nid_list   = ibs.get_annot_name_rowids(aid_list)
        #valid_aid_names  = ibs.get_annot_names(aid_list)
        #valid_aid_gtrues = ibs.get_annot_groundtruth(aid_list)
    return locals()


def get_sortbystr(str_list, key_list, strlbl=None, keylbl=None):
    sortx = key_list.argsort()
    ndigits = max(len(str(key_list.max())), 0 if keylbl is None else len(keylbl))
    keyfmt  = '%' + str(ndigits) + 'd'
    if keylbl is not None:
        header = keylbl + ' --- ' + strlbl
    else:
        header = None

    sorted_strs = ([(keyfmt % key + ' --- ' + str_) for str_, key in zip(str_list[sortx], key_list[sortx])])
    def boxjoin(list_, header=None):
        topline = '+----------'
        botline = 'L__________'
        boxlines = []
        boxlines.append(topline + '\n')
        if header is not None:
            boxlines.append(header + '\n')
            boxlines.append(topline)

        body = utool.indentjoin(list_, '\n | ')
        boxlines.append(body + '\n ')
        boxlines.append(botline + '\n')
        return ''.join(boxlines)
    return boxjoin(sorted_strs, header)


@devcmd('test_feats')
def test_feats(ibs, qaid_list, daid_list=None):
    """
    test_feats shows features using several different parameters

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id

    CommandLine:
        python dev.py -t test_feats --db PZ_MTEST --all --qindex 0 --show -w

    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = [1]
    """
    from ibeis import viz
    from plottool import df2
    from ibeis.dev import experiment_configs
    import utool as ut

    NUM_PASSES = 1 if not utool.get_argflag('--show') else 2
    varyparams_list = [experiment_configs.featparams]

    def test_featcfg_combo(ibs, aid, alldictcomb, count, nKpts_list, cfgstr_list):
        for dict_ in ut.progiter(alldictcomb, lbl='FeatCFG Combo: '):
            # Set ibs parameters to the current config
            for key_, val_ in six.iteritems(dict_):
                ibs.cfg.feat_cfg[key_] = val_
            cfgstr_ = ibs.cfg.feat_cfg.get_cfgstr()
            if count == 0:
                # On first run just record info
                kpts = ibs.get_annot_kpts(aid)
                nKpts_list.append(len(kpts))
                cfgstr_list.append(cfgstr_)
            if count == 1:
                kpts = ibs.get_annot_kpts(aid)
                # If second run happens display info
                cfgpackstr = utool.packstr(cfgstr_, textwidth=80,
                                              breakchars=',', newline_prefix='',
                                              break_words=False, wordsep=',')
                title_suffix = (' len(kpts) = %r \n' % len(kpts)) + cfgpackstr
                viz.show_chip(ibs, aid, fnum=df2.next_fnum(),
                              title_suffix=title_suffix, darken=.8,
                              ell_linewidth=2, ell_alpha=.6)

    alldictcomb = utool.flatten(map(utool.all_dict_combinations, varyparams_list))
    for count in range(NUM_PASSES):
        nKpts_list = []
        cfgstr_list = []
        for aid in qaid_list:
            test_featcfg_combo(ibs, aid, alldictcomb, count, nKpts_list, cfgstr_list)
            #for dict_ in alldictcomb:
        if count == 0:
            nKpts_list = np.array(nKpts_list)
            cfgstr_list = np.array(cfgstr_list)
            print(get_sortbystr(cfgstr_list, nKpts_list, 'cfg', 'nKpts'))


def devfunc(ibs, qaid_list):
    """ Function for developing something """
    print('[dev] devfunc')
    import ibeis  # NOQA
    from ibeis.model import Config  # NOQA
    #from ibeis.model.Config import *  # NOQA
    feat_cfg = Config.FeatureConfig()
    #feat_cfg.printme3()
    print('\ncfgstr..')
    print(feat_cfg.get_cfgstr())
    print(utool.dict_str(feat_cfg.get_hesaff_params()))
    #allres = results_all.get_allres(ibs, qaid_list)
    from ibeis import viz
    aid = 1
    ibs.cfg.feat_cfg.threshold = 16.0 / 3.0
    kpts = ibs.get_annot_kpts(aid)
    print('len(kpts) = %r' % len(kpts))
    from plottool import df2
    from ibeis.dev import experiment_configs
    #varyparams_list = [
    #    #{
    #    #    'threshold': [16.0 / 3.0, 32.0 / 3.0],  # 8.0  / 3.0
    #    #    'numberOfScales': [3, 2, 1],
    #    #    'maxIterations': [16, 32],
    #    #    'convergenceThreshold': [.05, .1],
    #    #    'initialSigma': [1.6, 1.2],
    #    #},
    #    {
    #        #'threshold': [16.0 / 3.0, 32.0 / 3.0],  # 8.0  / 3.0
    #        'numberOfScales': [1],
    #        #'maxIterations': [16, 32],
    #        #'convergenceThreshold': [.05, .1],
    #        #'initialSigma': [6.0, 3.0, 2.0, 1.6, 1.2, 1.1],
    #        'initialSigma': [3.2, 1.6, 0.8],
    #        'edgeEigenValueRatio': [10, 5, 3],
    #    },
    #]
    varyparams_list = [experiment_configs.featparams]

    # low threshold = more keypoints
    # low initialSigma = more keypoints

    nKpts_list = []
    cfgstr_list = []

    alldictcomb = utool.flatten([utool.util_dict.all_dict_combinations(varyparams) for varyparams in featparams_list])
    NUM_PASSES = 1 if not utool.get_argflag('--show') else 2
    for count in range(NUM_PASSES):
        for aid in qaid_list:
            #for dict_ in utool.progiter(alldictcomb, lbl='feature param comb: ', total=len(alldictcomb)):
            for dict_ in alldictcomb:
                for key_, val_ in six.iteritems(dict_):
                    ibs.cfg.feat_cfg[key_] = val_
                cfgstr_ = ibs.cfg.feat_cfg.get_cfgstr()
                cfgstr = utool.packstr(cfgstr_, textwidth=80,
                                        breakchars=',', newline_prefix='', break_words=False, wordsep=',')
                if count == 0:
                    kpts = ibs.get_annot_kpts(aid)
                    #print('___________')
                    #print('len(kpts) = %r' % len(kpts))
                    #print(cfgstr)
                    nKpts_list.append(len(kpts))
                    cfgstr_list.append(cfgstr_)
                if count == 1:
                    title_suffix = (' len(kpts) = %r \n' % len(kpts)) + cfgstr
                    viz.show_chip(ibs, aid, fnum=df2.next_fnum(),
                                   title_suffix=title_suffix, darken=.4,
                                   ell_linewidth=2, ell_alpha=.8)

        if count == 0:
            nKpts_list = np.array(nKpts_list)
            cfgstr_list = np.array(cfgstr_list)
            print(get_sortbystr(cfgstr_list, nKpts_list, 'cfg', 'nKpts'))
    df2.present()
    locals_ = locals()
    #locals_.update(annotationmatch_scores(ibs, qaid_list))
    return locals_


def run_dev(main_locals):
    """ main command """
    print('[dev] --- RUN DEV ---')
    # Get references to controller
    ibs  = main_locals['ibs']
    if ibs is not None:
        # Get aids marked as test cases
        qaid_list = main_helpers.get_test_qaids(ibs)
        daid_list = main_helpers.get_test_daids(ibs, qaid_list)
        print('[run_def] Test Annotations:')
        #print('[run_dev] * qaid_list = %s' % ut.packstr(qaid_list, 80, nlprefix='[run_dev]     '))
        bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
        print('[run_dev] * qaid_list = %s' % bigstr(str(qaid_list)))
        print('[run_dev] * daid_list = %s' % bigstr(str(daid_list)))
        print('[run_dev] * len(qaid_list) = %d' % len(qaid_list))
        print('[run_dev] * len(daid_list) = %d' % len(daid_list))
        print('[run_dev] * intersection = %r' % len(list(set(daid_list).intersection(set(qaid_list)))))
        ibs.temporary_state['daid_list'] = daid_list
        # Warn on no test cases
        try:
            assert len(qaid_list) > 0, 'assert!'
            assert len(daid_list) > 0, 'daid_list!'
        except AssertionError as ex:
            utool.printex(ex, 'len(qaid_list) = 0', iswarning=True)
            utool.printex(ex, 'or len(daid_list) = 0', iswarning=True)
            #qaid_list = ibs.get_valid_aids()[0]

        if len(qaid_list) > 0 or True:
            # Run the dev experiments
            expt_locals = run_devcmds(ibs, qaid_list, daid_list)
            # Add experiment locals to local namespace
            execstr_locals = utool.execstr_dict(expt_locals, 'expt_locals')
            exec(execstr_locals)
        if ut.get_argflag('--devmode'):
            # Execute the dev-func and add to local namespace
            devfunc_locals = devfunc(ibs, qaid_list)
            exec(utool.execstr_dict(devfunc_locals, 'devfunc_locals'))

    return locals()


#-------------
# EXAMPLE TEXT
#-------------

EXAMPLE_TEXT = '''
### DOWNLOAD A TEST DATABASE (IF REQUIRED) ###
python dev.py --t mtest
python dev.py --t nauts
./resetdbs.sh  # FIXME
python ibeis/dbio/ingest_database.py  <- see module for usage

### LIST AVAIABLE DATABASES ###
python dev.py -t list_dbs

### CHOOSE A DATABASE ###
python dev.py --db PZ_Master0 --setdb
python dev.py --db GZ_ALL --setdb
python dev.py --db PZ_MTEST --setdb
python dev.py --db NAUT_test --setdb
python dev.py --db testdb1 --setdb
python dev.py --db seals2 --setdb

### DATABASE INFORMATION ###
python dev.py -t dbinfo
python dev.py --allgt -t best
python dev.py --allgt -t vsone
python dev.py --allgt -t vsmany
python dev.py --allgt -t nsum

### COMPARE TWO CONFIGS ###
python dev.py --allgt -t nsum vsmany vsone
python dev.py --allgt -t nsum vsmany
python dev.py --allgt -t nsum vsmany vsone smk

### VIZ A SET OF MATCHES ###
python dev.py --db PZ_MTEST -t query --qaid 72 110 -w
#python dev.py --allgt -t vsone vsmany
#python dev.py --allgt -t vsone --vz --vh

### RUN A SMALL AMOUNT OF VSONE TESTS ###
python dev.py --allgt -t  vsone --qindex 0:1 --vz --vh --vf --noqcache
python dev.py --allgt --qindex 0:20 --

### DUMP ANALYSIS FIGURES TO DISK ###
python dev.py --allgt -t best --vf --vz --fig-dname query_analysis_easy
python dev.py --allgt -t best --vf --vh --fig-dname query_analysis_hard
python dev.py --allgt -t best --vf --va --fig-dname query_analysis_all

python dev.py --db PZ_MTEST --set-aids-as-hard 27 28 44 49 50 51 53 54 66 72 89 97 110
python dev.py --hard -t best vsone nsum
>>>
'''

#L______________


if __name__ == '__main__':
    """
        The Developer Script
            A command line interface to almost everything

            -w     # wait / show the gui / figures are visible
            --cmd  # ipython shell to play with variables
            -t     # run list of tests

            Examples:
    """

    multiprocessing.freeze_support()  # for win32

    helpstr = ut.codeblock(
        '''
        Dev is meant to be run as an interactive script.

        The dev.py script runs any test you regiter with @devcmd in any combination
        of configurations specified by a Config object.

        Dev caches information in order to get quicker results.  # FIXME: Provide quicker results  # FIXME: len(line)
        ''')

    INTRO_TITLE = 'The dev.py Script'
    INTRO_TEXT = ''.join((ut.bubbletext(INTRO_TITLE, font='cybermedium'), helpstr))

    INTRO_STR = ut.msgblock('dev.py Intro',  INTRO_TEXT)

    EXAMPLE_STR = ut.msgblock('dev.py Examples', ut.codeblock(EXAMPLE_TEXT))

    print(INTRO_STR)
    if ut.get_argflag(('--help', '--verbose')):
        print(EXAMPLE_STR)

    CMD   = '--cmd' in sys.argv
    NOGUI = '--gui' not in sys.argv

    if len(sys.argv) == 1:
        print('Run dev.py with arguments!')
        sys.exit(1)

    # Run Precommands
    run_devprecmds()

    #
    #
    # Run IBEIS Main, create controller, and possibly gui
    print('++dev')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    #utool.set_process_title('IBEIS_dev')

    #
    #
    # Load snippet variables
    SNIPPITS = True and CMD
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
    #if '--nopresent' not in sys.argv or '--noshow' in sys.argv:
    if ut.get_argflag(('--show', '--wshow')):
        df2.present()
    main_execstr = ibeis.main_loop(main_locals, ipy=(NOGUI or CMD))
    exec(main_execstr)

    #
    #
    # Memory profile
    if ut.get_argflag('--memprof'):
        utool.print_resource_usage()
        utool.memory_profile()

    print('exiting dev')
