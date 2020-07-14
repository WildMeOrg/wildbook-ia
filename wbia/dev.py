#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mkinit ~/code/wbia/wbia



DEV SCRIPT

TODO: DEPRICATE

This is a hacky script meant to be run mostly automatically with the option of
interactions.

dev.py is supposed to be a developer non-gui interface into the IBEIS software.
dev.py runs experiments and serves as a scratchpad for new code and quick scripts

TODO:

    Test to find typical "good" descriptor scores.  Find nearest neighbors and
    noramlizers for each feature in a query image.  Based on ground truth and
    spatial verification mark feature matches as true or false.  Visualize the
    feature scores of good matches vs bad matches. Lowe shows the pdf of
    correct matches and the PDF for incorrect matches. We should also show the
    same thing.

Done:
    Cache nearest neighbors so different parameters later in the pipeline dont
    take freaking forever.

CommandLine:
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg score_method:nsum prescore_method:nsum
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg fg_on=True
    python dev.py --wshow -t query --db PZ_MTEST --qaid 110 --cfg
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from wbia._devscript import devcmd, DEVCMD_FUNCTIONS, DEVPRECMD_FUNCTIONS
import utool as ut
from utool.util_six import get_funcname
import utool

# from wbia.algo.hots import smk
import wbia.plottool as pt
import wbia

# import multiprocessing
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     wbia._preload()
# utool.util_importer.dynamic_import(__name__, ('_devcmds_wbia', None),
#                                   developing=True)
from wbia._devscript import devcmd, devprecmd  # NOQA
from os.path import split, join, expanduser  # NOQA
from wbia.plottool import draw_func2 as df2  # NOQA
from wbia import sysres  # NOQA
from wbia.other import ibsfuncs  # NOQA
from wbia.dbio import ingest_hsdb  # NOQA
from wbia._devcmds_wbia import (  # NOQA
    GZ_VIEWPOINT_EXPORT_PAIRS,
    MOTHERS_VIEWPOINT_EXPORT_PAIRS,
    change_names,
    convert_hsdbs,
    delete_all_chips,
    delete_all_feats,
    delete_cache,
    ensure_mtest,
    ensure_nauts,
    ensure_wilddogs,
    export,
    list_dbs,
    list_unconverted_hsdbs,
    openworkdirs_test,
    query_aids,
    show_aids,
    sver_aids,
    vdd,
)

# IBEIS
from wbia.init import main_helpers  # NOQA
from wbia.other import dbinfo  # NOQA
from wbia.expt import experiment_configs  # NOQA
from wbia.expt import harness  # NOQA
from wbia.expt.experiment_drawing import draw_annot_scoresep
from wbia import params  # NOQA

print, rrr, profile = utool.inject2(__name__)


# ------------------
# DEV DEVELOPMENT
# ------------------
# This is where you write all of the functions that will become pristine
# and then go in _devcmds_wbia.py


"""
./dev.py -e print_results --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k:K=1 --intersect_hack
./dev.py -e draw_rank_cmc -t baseline -a baseline --show --db PZ_Master1
./dev.py -e get_dbinfo --db PZ_Master1 --aid_list=baseline
./dev.py -e get_dbinfo --db PZ_MTEST
./dev.py -e get_dbinfo --db PZ_Master1 --aid_list=baseline --hackshow-unixtime --show
./dev.py -e get_dbinfo --db PZ_Master1 --hackshow-unixtime --show
"""
# Quick interface into specific registered doctests
REGISTERED_DOCTEST_EXPERIMENTS = [
    (
        'wbia.expt.experiment_drawing',
        'draw_case_timedeltas',
        ['timedelta_hist', 'timedelta_pie'],
    ),
    ('wbia.expt.experiment_drawing', 'draw_match_cases', ['draw_cases', 'cases']),
    ('wbia.expt.experiment_drawing', 'draw_casetag_hist', ['taghist']),
    ('wbia.expt.old_storage', 'draw_results'),
    ('wbia.expt.experiment_drawing', 'draw_rank_cmc', ['rank_cmc']),
    ('wbia.other.dbinfo', 'get_dbinfo'),
    ('wbia.other.dbinfo', 'latex_dbstats'),
    ('wbia.other.dbinfo', 'show_image_time_distributions', ['db_time_hist']),
    ('wbia.expt.experiment_drawing', 'draw_rank_surface', ['rank_surface']),
    ('wbia.expt.experiment_helpers', 'get_annotcfg_list', ['print_acfg']),
    ('wbia.expt.experiment_printres', 'print_results', ['printres', 'print']),
    ('wbia.expt.experiment_printres', 'print_latexsum', ['latexsum']),
    ('wbia.dbio.export_subset', 'export_annots'),
    (
        'wbia.expt.experiment_drawing',
        'draw_annot_scoresep',
        ['scores', 'scores_good', 'scores_all'],
    ),
]


def _exec_doctest_func(modname, funcname):
    module = ut.import_modname(modname)
    func = module.__dict__[funcname]
    testsrc = ut.get_doctest_examples(func)[0][0]
    exec(testsrc, globals(), locals())


def _register_doctest_precmds():
    from functools import partial

    for tup in REGISTERED_DOCTEST_EXPERIMENTS:
        modname, funcname = tup[:2]
        aliases = tup[2] if len(tup) == 3 else []
        aliases += [funcname]
        _doctest_func = partial(_exec_doctest_func, modname, funcname)
        devprecmd(*aliases)(_doctest_func)


_register_doctest_precmds()


@devcmd('tune', 'autotune')
def tune_flann(ibs, qaid_list, daid_list=None):
    r"""
    CommandLine:
        python dev.py -t tune --db PZ_MTEST
        python dev.py -t tune --db GZ_ALL
        python dev.py -t tune --db GIR_Tanya
        python dev.py -t tune --db PZ_Master0
    """
    all_aids = ibs.get_valid_aids()
    vecs = np.vstack(ibs.get_annot_vecs(all_aids))
    print(
        'Tunning flann for species={species}:'.format(
            species=ibs.get_database_species(all_aids)
        )
    )
    import vtool as vt

    tuned_params = vt.tune_flann(
        vecs,
        target_precision=0.98,
        build_weight=0.05,
        memory_weight=0.00,
        sample_fraction=0.1,
    )
    tuned_params

    # tuned_params2 = vt.tune_flann(vecs,
    #                              target_precision=.90,
    #                              build_weight=0.001,
    #                              memory_weight=0.00,
    #                              sample_fraction=0.5)
    # tuned_params2


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
        >>> # DISABLE_DOCTEST
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
    """
    from wbia.algo.hots import automated_matcher

    ibs1 = ibs
    num_initial = ut.get_argval('--ninit', type_=int, default=0)
    return automated_matcher.incremental_test(ibs1, num_initial)


@devcmd('inspect')
def inspect_matches(ibs, qaid_list, daid_list):
    print('<inspect_matches>')
    from wbia.gui import inspect_gui

    return inspect_gui.test_review_widget(ibs, qaid_list, daid_list)


def get_ibslist(ibs):
    print('[dev] get_ibslist')
    ibs_GV = ibs
    ibs_RI = ibs.clone_handle(nogravity_hack=True)
    ibs_RIW = ibs.clone_handle(nogravity_hack=True, gravity_weighting=True)
    ibs_list = [ibs_GV, ibs_RI, ibs_RIW]
    return ibs_list


@devcmd('gv_scores')
def compgrav_draw_score_sep(ibs, qaid_list, daid_list):
    print('[dev] compgrav_draw_score_sep')
    ibs_list = get_ibslist(ibs)
    for ibs_ in ibs_list:
        draw_annot_scoresep(ibs_, qaid_list)


# --------------------
# RUN DEV EXPERIMENTS
# --------------------


# def run_registered_precmd(precmd_name):
#    # Very hacky way to run just a single registered precmd
#    for (func_aliases, func) in DEVPRECMD_FUNCTIONS:
#        for aliases in func_aliases:
#            ret = precmd_name in input_precmd_list
#            if ret:
#                func()


def run_devprecmds():
    """
    Looks for pre-tests specified with the -t flag and runs them
    """
    # input_precmd_list = params.args.tests[:]
    input_precmd_list = ut.get_argval('-e', type_=list, default=[])
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
            # with utool.Indenter('[dev.' + get_funcname(func) + ']'):
            func()
            print('Exiting after first precommand')
            sys.exit(1)
    if len(input_precmd_list) > 0:
        raise AssertionError('Unhandled tests: ' + repr(input_precmd_list))


# @utool.indent_func('[dev]')
def run_devcmds(ibs, qaid_list, daid_list, acfg=None):
    """
    This function runs tests passed in with the -t flag
    """
    print('\n')
    # print('[dev] run_devcmds')
    print('==========================')
    print('[DEV] RUN EXPERIMENTS %s' % ibs.get_dbname())
    print('==========================')
    input_test_list = params.args.tests[:]
    print('input_test_list = %s' % (ut.repr2(input_test_list),))
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
                print('\n+===================')
                print(' [dev] running testname = %s' % (args,))
                print('+-------------------\n')
                return ret
        return False

    valid_test_helpstr_list.append('    # --- Simple Tests ---')

    # Explicit (simple) test functions
    if intest('export'):
        raise NotImplementedError('export')
    if intest('dbinfo'):
        dbinfo.get_dbinfo(ibs)
    if intest('headers', 'schema'):
        ibs.db.print_schema()
    if intest('info'):
        print(ibs.get_infostr())
    if intest('printcfg'):
        raise NotImplementedError('printcfg')
        # printcfg(ibs)
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
            # with utool.Indenter('[dev.' + funcname + ']'):
            with utool.Timer(funcname):
                # print('[dev] qid_list=%r' % (qaid_list,))
                # FIXME: , daid_list
                if len(ut.get_func_argspec(func).args) == 0:
                    ret = func()
                else:
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
        fnum = pt.next_fnum()
        # Run Experiments
        # backwards compatibility yo
        acfgstr_name_list = {'OVERRIDE_HACK': (qaid_list, daid_list)}
        assert (
            False
        ), 'This way of running tests no longer works. It may be fixed in the future'
        # acfg
        harness.test_configurations(ibs, acfgstr_name_list, test_cfg_name_list)

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


# -------------------
# CUSTOM DEV FUNCS
# -------------------


# ------------------
# DEV MAIN
# ------------------


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
        # ibs.dump_tables()
        annots = ibs.annots()
        images = ibs.images()
        aid_list = ibs.get_valid_aids()
        gid_list = ibs.get_valid_gids()
        # nid_list = ibs.get_valid_nids()
        # valid_nid_list   = ibs.get_annot_name_rowids(aid_list)
        # valid_aid_names  = ibs.get_annot_names(aid_list)
        # valid_aid_gtrues = ibs.get_annot_groundtruth(aid_list)
    return locals()


def get_sortbystr(str_list, key_list, strlbl=None, keylbl=None):
    sortx = key_list.argsort()
    ndigits = max(len(str(key_list.max())), 0 if keylbl is None else len(keylbl))
    keyfmt = '%' + str(ndigits) + 'd'
    if keylbl is not None:
        header = keylbl + ' --- ' + strlbl
    else:
        header = None

    sorted_strs = [
        (keyfmt % key + ' --- ' + str_)
        for str_, key in zip(str_list[sortx], key_list[sortx])
    ]

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


# @devcmd('test_feats')
# def tst_feats(ibs, qaid_list, daid_list=None):
#     """
#     test_feats shows features using several different parameters

#     Args:
#         ibs (IBEISController):
#         qaid_list (int): query annotation id

#     CommandLine:
#         python dev.py -t test_feats --db PZ_MTEST --all --qindex 0 --show -w

#     Example:
#         >>> import wbia
#         >>> ibs = wbia.opendb('testdb1')
#         >>> qaid_list = [1]
#     """
#     from wbia import viz
#     from wbia.expt import experiment_configs
#     import utool as ut

#     NUM_PASSES = 1 if not utool.get_argflag('--show') else 2
#     varyparams_list = [experiment_configs.featparams]

#     def tst_featcfg_combo(ibs, aid, alldictcomb, count, nKpts_list, cfgstr_list):
#         for dict_ in ut.progiter(alldictcomb, lbl='FeatCFG Combo: '):
#             cfgstr_ = ut.repr2(dict_)
#             if count == 0:
#                 # On first run just record info
#                 kpts = ibs.get_annot_kpts(aid)
#                 nKpts_list.append(len(kpts))
#                 cfgstr_list.append(cfgstr_)
#             if count == 1:
#                 kpts = ibs.get_annot_kpts(aid)
#                 # If second run happens display info
#                 cfgpackstr = utool.packstr(cfgstr_, textwidth=80,
#                                               breakchars=',', newline_prefix='',
#                                               break_words=False, wordsep=',')
#                 title_suffix = (' len(kpts) = %r \n' % len(kpts)) + cfgpackstr
#                 viz.show_chip(ibs, aid, fnum=pt.next_fnum(),
#                               title_suffix=title_suffix, darken=.8,
#                               ell_linewidth=2, ell_alpha=.6, config=dict_)

#     alldictcomb = utool.flatten(map(utool.all_dict_combinations, varyparams_list))
#     for count in range(NUM_PASSES):
#         nKpts_list = []
#         cfgstr_list = []
#         for aid in qaid_list:
#             test_featcfg_combo(ibs, aid, alldictcomb, count, nKpts_list, cfgstr_list)
#             #for dict_ in alldictcomb:
#         if count == 0:
#             nKpts_list = np.array(nKpts_list)
#             cfgstr_list = np.array(cfgstr_list)
#             print(get_sortbystr(cfgstr_list, nKpts_list, 'cfg', 'nKpts'))


def run_dev(ibs):
    """
    main developer command

    CommandLine:
        python dev.py --db PZ_Master0 --controlled --print-rankhist
    """
    print('[dev] --- RUN DEV ---')
    # Get reference to controller
    if ibs is not None:
        # Get aids marked as test cases
        if ut.get_argflag('--expanded-aids'):
            ibs, qaid_list, daid_list = main_helpers.testdata_expanded_aids(ibs=ibs)
            print('[run_def] Test Annotations:')
            # print('[run_dev] * qaid_list = %s' % ut.packstr(qaid_list, 80, nlprefix='[run_dev]     '))
        else:
            qaid_list = []
            daid_list = []
        try:
            assert len(qaid_list) > 0, 'assert!'
            assert len(daid_list) > 0, 'daid_list!'
        except AssertionError as ex:
            message = ' (try using command line argument --expanded-aids to enable)'
            utool.printex(ex, 'len(qaid_list) = 0%s' % (message,), iswarning=True)
            utool.printex(ex, 'or len(daid_list) = 0%s' % (message,), iswarning=True)
            # qaid_list = ibs.get_valid_aids()[0]

        if len(qaid_list) > 0 or True:
            # Run the dev experiments
            expt_locals = run_devcmds(ibs, qaid_list, daid_list)
            # Add experiment locals to local namespace
            execstr_locals = utool.execstr_dict(expt_locals, 'expt_locals')
            exec(execstr_locals)

    return locals()


# -------------
# EXAMPLE TEXT
# -------------

EXAMPLE_TEXT = """
### DOWNLOAD A TEST DATABASE (IF REQUIRED) ###
python -m wbia.dev --t mtest
python dev.py --t nauts
./resetdbs.sh  # FIXME
python wbia/dbio/ingest_database.py  <- see module for usage

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

### EXPERIMENTS ###
python dev.py --allgt -t best
python dev.py --allgt -t vsone
python dev.py --allgt -t vsmany
python dev.py --allgt -t nsum

# Newstyle experiments
# commmand             # annot settings            # test settings
python -m wbia.dev    -a default:qaids=allgt      -t best


### COMPARE TWO CONFIGS ###
python dev.py --allgt -t nsum vsmany vsone
python dev.py --allgt -t nsum vsmany
python dev.py --allgt -t nsum vsmany vsone smk

### VARY DATABASE SIZE
python -m wbia.dev -a default:qaids=allgt,dsize=100,qper_name=1,qmin_per_name=1 -t default --db PZ_MTEST
python -m wbia.dev -a candidacy:qsize=10,dsize=100 -t default --db PZ_MTEST --verbtd


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
"""

# L______________


# def run_devmain2():
#    input_test_list = ut.get_argval(('--tests', '-t',), type_=list, default=[])[:]
#    print('input_test_list = %s' % (ut.repr2(input_test_list),))
#    # fnum = 1

#    valid_test_list = []  # build list for printing in case of failure
#    valid_test_helpstr_list = []  # for printing

#    def mark_test_handled(testname):
#        input_test_list.remove(testname)

#    def intest(*args, **kwargs):
#        helpstr = kwargs.get('help', '')
#        valid_test_helpstr_list.append('   -t ' + ', '.join(args) + helpstr)
#        for testname in args:
#            valid_test_list.append(testname)
#            ret = testname in input_test_list
#            ret2 = testname in params.unknown  # Let unparsed args count towards tests
#            if ret or ret2:
#                if ret:
#                    mark_test_handled(testname)
#                else:
#                    ret = ret2
#                print('\n+===================')
#                print(' [dev2] running testname = %s' % (args,))
#                print('+-------------------\n')
#                return ret
#        return False

#    anynewhit = False
#    # Implicit (decorated) test functions
#    print('DEVCMD_FUNCTIONS2 = %r' % (DEVCMD_FUNCTIONS2,))
#    for (func_aliases, func) in DEVCMD_FUNCTIONS2:
#        if intest(*func_aliases):
#            funcname = get_funcname(func)
#            with utool.Timer(funcname):
#                if len(ut.get_func_argspec(func).args) == 0:
#                    func()
#                    anynewhit = True
#                else:
#                    func(ibs, qaid_list, daid_list)
#                    anynewhit = True
#    if anynewhit:
#        sys.exit(1)


def devmain():
    """
    The Developer Script
        A command line interface to almost everything

        -w     # wait / show the gui / figures are visible
        --cmd  # ipython shell to play with variables
        -t     # run list of tests
    """
    from wbia import params

    params.parse_args()

    print('DEVMAIN INIT - PRELOGGING')
    if params.args.logdir is not None:
        sysres.set_logdir(params.args.logdir)
    else:
        ut.start_logging(appname='wbia')
    print('DEVMAIN INIT - POSTLOGGING')

    helpstr = ut.codeblock(
        """
        Dev is meant to be run as an interactive script.

        The dev.py script runs any test you regiter with @devcmd in any combination
        of configurations specified by a Config object.

        Dev caches information in order to get quicker results.  # FIXME: Provide quicker results  # FIXME: len(line)
        """
    )

    INTRO_TITLE = 'The dev.py Script'
    # INTRO_TEXT = ''.join((ut.bubbletext(INTRO_TITLE, font='cybermedium'), helpstr))
    INTRO_TEXT = ut.bubbletext(INTRO_TITLE, font='cybermedium')

    INTRO_STR = ut.msgblock('dev.py Intro', INTRO_TEXT)

    EXAMPLE_STR = ut.msgblock('dev.py Examples', ut.codeblock(EXAMPLE_TEXT))

    if ut.NOT_QUIET:
        print(INTRO_STR)
    if ut.get_argflag(('--help', '--verbose')):
        print(EXAMPLE_STR)

    CMD = ut.get_argflag('--cmd')
    NOGUI = not ut.get_argflag('--gui')

    if len(sys.argv) == 1:
        print('Run dev.py with arguments!')
        sys.exit(1)

    # Run Precommands
    run_devprecmds()

    #
    #
    # Run IBEIS Main, create controller, and possibly gui
    print('++dev')
    main_locals = wbia.main(gui=ut.get_argflag('--gui'))
    # utool.set_process_title('wbia-dev')

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
    RUN_DEV = True
    if RUN_DEV:
        dev_locals = run_dev(main_locals['ibs'])
        dev_execstr = utool.execstr_dict(dev_locals, 'dev_locals')
        exec(dev_execstr)

    command = ut.get_argval('--eval', type_=str, default=None)
    if command is not None:
        result = eval(command, globals(), locals())
        print('result = %r' % (result,))
        # ibs.search_annot_notes('360')

    #
    #
    # Main Loop (IPython interaction, or some exec loop)
    # if '--nopresent' not in sys.argv or '--noshow' in sys.argv:
    ut.show_if_requested()
    if ut.get_argflag(('--show', '--wshow')):
        pt.present()
    main_execstr = wbia.main_loop(main_locals, rungui=not NOGUI, ipy=(NOGUI or CMD))
    exec(main_execstr)

    #
    #
    # Memory profile
    if ut.get_argflag('--memprof'):
        utool.print_resource_usage()
        utool.memory_profile()

    print('exiting dev')


def ggr_random_name_splits():
    """
    CommandLine:
        python -m wbia.viz.viz_graph2 ggr_random_name_splits --show

    Ignore:
        sshfs -o idmap=user lev:/ ~/lev

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> ggr_random_name_splits()
    """
    import wbia.guitool as gt

    gt.ensure_qtapp()
    # nid_list = ibs.get_valid_nids(filter_empty=True)
    import wbia

    dbdir = '/media/danger/GGR/GGR-IBEIS'
    dbdir = (
        dbdir if ut.checkpath(dbdir) else ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
    )
    ibs = wbia.opendb(dbdir=dbdir, allow_newdir=False)

    import datetime

    day1 = datetime.date(2016, 1, 30)
    day2 = datetime.date(2016, 1, 31)

    orig_filter_kw = {
        'multiple': None,
        # 'view': ['right'],
        # 'minqual': 'good',
        'is_known': True,
        'min_pername': 2,
    }
    orig_aids = ibs.filter_annots_general(
        filter_kw=ut.dict_union(
            orig_filter_kw,
            {
                'min_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day1, 0.0)),
                'max_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day2, 1.0)),
            },
        )
    )
    orig_all_annots = ibs.annots(orig_aids)
    orig_unique_nids, orig_grouped_annots_ = orig_all_annots.group(orig_all_annots.nids)
    # Ensure we get everything
    orig_grouped_annots = [
        ibs.annots(aids_) for aids_ in ibs.get_name_aids(orig_unique_nids)
    ]

    # pip install quantumrandom
    if False:
        import quantumrandom

        data = quantumrandom.uint16()
        seed = data.sum()
        print('seed = %r' % (seed,))
        # import Crypto.Random
        # from Crypto import Random
        # quantumrandom.get_data()
        # StrongRandom = Crypto.Random.random.StrongRandom
        # aes.reseed(3340258)
        # chars = [str(chr(x)) for x in data.view(np.uint8)]
        # aes_seed = str('').join(chars)
        # aes = Crypto.Random.Fortuna.FortunaGenerator.AESGenerator()
        # aes.reseed(aes_seed)
        # aes.pseudo_random_data(10)

    orig_rand_idxs = ut.random_indexes(len(orig_grouped_annots), seed=3340258)
    orig_sample_size = 75
    random_annot_groups = ut.take(orig_grouped_annots, orig_rand_idxs)
    orig_annot_sample = random_annot_groups[:orig_sample_size]

    # OOOPS MADE ERROR REDO ----

    filter_kw = {
        'multiple': None,
        'view': ['right'],
        'minqual': 'good',
        'is_known': True,
        'min_pername': 2,
    }
    filter_kw_ = ut.dict_union(
        filter_kw,
        {
            'min_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day1, 0.0)),
            'max_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day2, 1.0)),
        },
    )
    refiltered_sample = [
        ibs.filter_annots_general(annot.aids, filter_kw=filter_kw_)
        for annot in orig_annot_sample
    ]
    is_ok = np.array(ut.lmap(len, refiltered_sample)) >= 2
    ok_part_orig_sample = ut.compress(orig_annot_sample, is_ok)
    ok_part_orig_nids = [x.nids[0] for x in ok_part_orig_sample]

    # Now compute real sample
    aids = ibs.filter_annots_general(filter_kw=filter_kw_)
    all_annots = ibs.annots(aids)
    unique_nids, grouped_annots_ = all_annots.group(all_annots.nids)
    grouped_annots = grouped_annots_
    # Ensure we get everything
    # grouped_annots = [ibs.annots(aids_) for aids_ in ibs.get_name_aids(unique_nids)]

    pop = len(grouped_annots)
    pername_list = ut.lmap(len, grouped_annots)
    groups = wbia.annots.AnnotGroups(grouped_annots, ibs)
    match_tags = [ut.unique(ut.flatten(t)) for t in groups.match_tags]
    tag_case_hist = ut.dict_hist(ut.flatten(match_tags))
    print('name_pop = %r' % (pop,))
    print(
        'Annots per Multiton Name' + ut.repr3(ut.get_stats(pername_list, use_median=True))
    )
    print('Name Tag Hist ' + ut.repr3(tag_case_hist))
    print('Percent Photobomb: %.2f%%' % (tag_case_hist['photobomb'] / pop * 100))
    print('Percent Split: %.2f%%' % (tag_case_hist['splitcase'] / pop * 100))

    # Remove the ok part from this sample
    remain_unique_nids = ut.setdiff(unique_nids, ok_part_orig_nids)
    remain_grouped_annots = [
        ibs.annots(aids_) for aids_ in ibs.get_name_aids(remain_unique_nids)
    ]

    sample_size = 75
    import vtool as vt

    vt.calc_sample_from_error_bars(0.05, pop, conf_level=0.95, prior=0.05)

    remain_rand_idxs = ut.random_indexes(len(remain_grouped_annots), seed=3340258)
    remain_sample_size = sample_size - len(ok_part_orig_nids)
    remain_random_annot_groups = ut.take(remain_grouped_annots, remain_rand_idxs)
    remain_annot_sample = remain_random_annot_groups[:remain_sample_size]

    annot_sample_nofilter = ok_part_orig_sample + remain_annot_sample
    # Filter out all bad parts
    annot_sample_filter = [
        ibs.annots(ibs.filter_annots_general(annot.aids, filter_kw=filter_kw_))
        for annot in annot_sample_nofilter
    ]
    annot_sample = annot_sample_filter

    win = None
    from wbia.viz import viz_graph2

    for annots in ut.InteractiveIter(annot_sample):
        if win is not None:
            win.close()
        win = viz_graph2.make_qt_graph_interface(
            ibs, aids=annots.aids, init_mode='rereview'
        )
        print(win)

    sample_groups = wbia.annots.AnnotGroups(annot_sample, ibs)

    flat_tags = [ut.unique(ut.flatten(t)) for t in sample_groups.match_tags]

    print('Using Split and Photobomb')
    is_positive = ['photobomb' in t or 'splitcase' in t for t in flat_tags]
    num_positive = sum(is_positive)
    vt.calc_error_bars_from_sample(sample_size, num_positive, pop, conf_level=0.95)

    print('Only Photobomb')
    is_positive = ['photobomb' in t for t in flat_tags]
    num_positive = sum(is_positive)
    vt.calc_error_bars_from_sample(sample_size, num_positive, pop, conf_level=0.95)

    print('Only SplitCase')
    is_positive = ['splitcase' in t for t in flat_tags]
    num_positive = sum(is_positive)
    vt.calc_error_bars_from_sample(sample_size, num_positive, pop, conf_level=0.95)
    # gt.qtapp_loop(qwin=win)


r"""
CurrentExperiments:
    # Full best settings run
    ./dev.py -t custom --db PZ_Master0 --allgt --species=zebra_plains
    # Full best settings run without spatial verification
    ./dev.py -t custom:sv_on=False --db PZ_Master0 --allgt --species=zebra_plains

    ./dev.py -t custom --db PZ_Master0 --allgt --species=zebra_plains --hs

    # Check to see if new spatial verification helps
    ./dev.py -t custom:full_homog_checks=False custom:full_homog_checks=True --db PZ_Master0 --allgt --species=zebra_plains
    # Yay it does


    # Look for how many false negatives are in the bottom batch
    ./dev.py -t custom --db PZ_MTEST --species=zebra_plains --print-rankhist
    ./dev.py -t custom --db PZ_MTEST --controlled --print-rankhist
    ./dev.py -t custom --db PZ_Master0 --controlled --print-rankhist

    ./dev.py -t \
            custom \
            custom:rotation_invariance=True,affine_invariance=False \
            custom:rotation_invariance=True,query_rotation_heuristic=True \
            --db PZ_Master0 --controlled --print-rankhist  --print-bestcfg

    ./dev.py -t \
            custom:rotation_invariance=True,affine_invariance=False \
            custom:rotation_invariance=True,query_rotation_heuristic=True \
            --db NNP_Master3 --controlled --print-rankhist  --print-bestcfg


ElephantEarExperiments
    --show --vh
    ./dev.py -t custom:affine_invariance=True --db Elephants_drop1_ears --allgt --print-rankhist
    ./dev.py -t custom:affine_invariance=False --db Elephants_drop1_ears --allgt --print-rankhist
    ./dev.py -t custom:affine_invariance=False,histeq=True --db Elephants_drop1_ears --allgt --print-rankhist
    ./dev.py -t custom:affine_invariance=False,adapteq=True --db Elephants_drop1_ears --allgt --print-rankhist

    ./dev.py -t custom:affine_invariance=False,fg_on=False --db Elephants_drop1_ears --allgt
    ./dev.py -t custom:affine_invariance=False,histeq=True,fg_on=False --db Elephants_drop1_ears --allgt
    ./dev.py -t custom:affine_invariance=False,adapteq=True,fg_on=False --db Elephants_drop1_ears --allgt

    ./dev.py -t elph --db Elephants_drop1_ears --allgt


Sift vs Siam Experiments
    ./dev.py -t custom:feat_type=hesaff+siam128,algorithm=linear custom:feat_type=hesaff+sift --db testdb1 --allgt
    ./dev.py -t custom:feat_type=hesaff+siam128,algorithm=linear custom:feat_type=hesaff+sift --db PZ_MTEST --allgt
    ./dev.py -t custom:feat_type=hesaff+siam128,lnbnn_on=False,fg_on=False,bar_l2_on=True custom:feat_type=hesaff+sift,fg_on=False --db PZ_MTEST --allgt

    ./dev.py -t custom:feat_type=hesaff+siam128 custom:feat_type=hesaff+sift --db PZ_MTEST --allgt --print-rankhist
    ./dev.py -t custom:feat_type=hesaff+siam128 --db PZ_MTEST --allgt --print-rankhist
    ./dev.py -t custom:feat_type=hesaff+sift --db PZ_MTEST --allgt --print-rankhist

    ./dev.py -t custom:feat_type=hesaff+siam128 custom:feat_type=hesaff+sift --db PZ_Master0 --allgt

    ./dev.py -t custom:feat_type=hesaff+siam128 --db testdb1 --allgt



Without SV:
agg rank histogram = {
    (0, 1): 2276,
    (1, 5): 126,
    (5, 50): 99,
    (50, 8624): 108,
    (8624, 8625): 28,
}
With SV:
agg rank histogram = {
    (0, 1): 2300,
    (1, 5): 106,
    (5, 50): 16,
    (50, 8624): 0,
    (8624, 8625): 215,
}

Guesses:
    0 2 2 2 4 4 4 4 0 0
    0 0 4 2 2 4 4 4 2 2
    2 4 4 4 1 1 1 2 2 2
    0 0 1 1 1 2 0 0 1
"""


if __name__ == '__main__':
    """
    """
    devmain()
