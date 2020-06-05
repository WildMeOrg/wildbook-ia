# -*- coding: utf-8 -*-
"""
Runs many queries and keeps track of some results
"""
from __future__ import absolute_import, division, print_function
import sys
import textwrap
import numpy as np  # NOQA
import utool as ut
from wbia.expt import experiment_helpers
from wbia.expt import test_result

print, rrr, profile = ut.inject2(__name__)

NOMEMORY = ut.get_argflag('--nomemory')
TESTRES_VERBOSITY = 2 - (2 * ut.QUIET)
NOCACHE_TESTRES = ut.get_argflag(('--nocache-testres', '--nocache-big'), False)
USE_BIG_TEST_CACHE = (
    not ut.get_argflag(('--no-use-testcache', '--nocache-test'))
    and ut.USE_CACHE
    and not NOCACHE_TESTRES
)
USE_BIG_TEST_CACHE = False
TEST_INFO = True

# dont actually query. Just print labels and stuff
DRY_RUN = ut.get_argflag(('--dryrun', '--dry'))


def run_expt(
    ibs,
    acfg_name_list,
    test_cfg_name_list,
    use_cache=None,
    qaid_override=None,
    daid_override=None,
    initial_aids=None,
):
    r"""
    Loops over annot configs.

    Try and use this function as a starting point to clean up this module.
    The code is getting too untenable.

    CommandLine:
        python -m wbia.expt.harness run_expt --acfginfo
        python -m wbia.expt.harness run_expt --pcfginfo
        python -m wbia.expt.harness run_expt

    Ignore:
        test_cfg_name_list = [p]

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.expt.harness import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> default_acfgstrs = ['ctrl:qsize=20,dpername=1,dsize=10',
        >>>                     'ctrl:qsize=20,dpername=10,dsize=20']
        >>> acfg_name_list = default_acfgstrs
        >>> test_cfg_name_list = ['default:proot=smk', 'default']
        >>> #test_cfg_name_list = ['custom', 'custom:fg_on=False']
        >>> use_cache = False
        >>> testres_list = run_expt(ibs, acfg_name_list, test_cfg_name_list, use_cache)
    """
    print('[harn] run_expt')
    # Generate list of database annotation configurations
    if len(acfg_name_list) == 0:
        raise ValueError('must give acfg name list')

    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
        ibs,
        acfg_name_list,
        qaid_override=qaid_override,
        daid_override=daid_override,
        initial_aids=initial_aids,
        use_cache=use_cache,
    )

    # Generate list of query pipeline param configs
    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
        test_cfg_name_list, ibs=ibs
    )

    cfgx2_lbl = experiment_helpers.get_varied_pipecfg_lbls(cfgdict_list)
    # NOTE: Can specify --pcfginfo or --acfginfo

    if ut.NOT_QUIET:
        ut.colorprint(
            textwrap.dedent(
                """

        [harn]================
        [harn] harness.test_configurations2()"""
            ).strip(),
            'white',
        )
        msg = '[harn] Running %s using %s and %s' % (
            ut.quantstr('test', len(acfg_list) * len(cfgdict_list)),
            ut.quantstr('pipeline config', len(cfgdict_list)),
            ut.quantstr('annot config', len(acfg_list)),
        )
        ut.colorprint(msg, 'white')

    testres_list = []

    nAcfg = len(acfg_list)

    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list) + str(acfg_name_list)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list) + str(acfg_name_list)
    expanded_aids_iter = ut.ProgIter(
        expanded_aids_list,
        lbl='annot config',
        freq=1,
        autoadjust=False,
        enabled=ut.NOT_QUIET,
    )

    for acfgx, (qaids, daids) in enumerate(expanded_aids_iter):
        assert len(qaids) != 0, '[harness] No query annots specified'
        assert len(daids) != 0, '[harness] No database annotas specified'
        acfg = acfg_list[acfgx]
        if ut.NOT_QUIET:
            ut.colorprint('\n---Annot config testnameid=%r' % (testnameid,), 'turquoise')
        subindexer_partial = ut.ProgPartial(
            parent_index=acfgx, parent_length=nAcfg, enabled=ut.NOT_QUIET
        )
        testres_ = make_single_testres(
            ibs,
            qaids,
            daids,
            pipecfg_list,
            cfgx2_lbl,
            cfgdict_list,
            lbl,
            testnameid,
            use_cache=use_cache,
            subindexer_partial=subindexer_partial,
        )
        if DRY_RUN:
            continue
        testres_.acfg = acfg
        testres_.test_cfg_name_list = test_cfg_name_list
        testres_list.append(testres_)
    if DRY_RUN:
        print('DRYRUN: Cannot continue past run_expt')
        sys.exit(0)

    testres = test_result.combine_testres_list(ibs, testres_list)
    # testres.print_results()
    print('Returning Test Result')
    return testres


@profile
def make_single_testres(
    ibs,
    qaids,
    daids,
    pipecfg_list,
    cfgx2_lbl,
    cfgdict_list,
    lbl,
    testnameid,
    use_cache=None,
    subindexer_partial=ut.ProgIter,
):
    """
    CommandLine:
        python -m wbia run_expt
    """
    cfgslice = None
    if cfgslice is not None:
        pipecfg_list = pipecfg_list[cfgslice]

    dbname = ibs.get_dbname()

    # if ut.NOT_QUIET:
    #     print('[harn] Make single testres')

    cfgx2_qreq_ = [
        ibs.new_query_request(qaids, daids, verbose=False, query_cfg=pipe_cfg)
        for pipe_cfg in ut.ProgIter(pipecfg_list, lbl='Building qreq_', enabled=False)
    ]

    if use_cache is None:
        use_cache = USE_BIG_TEST_CACHE

    if use_cache:
        try:
            bt_cachedir = ut.ensuredir((ibs.get_cachedir(), 'BULK_TEST_CACHE2'))
            cfgstr_list = [qreq_.get_cfgstr(with_input=True) for qreq_ in cfgx2_qreq_]
            bt_cachestr = ut.hashstr_arr27(cfgstr_list, ibs.get_dbname() + '_cfgs')
            bt_cachename = 'BULKTESTCACHE2_v2'
            testres = ut.load_cache(bt_cachedir, bt_cachename, bt_cachestr)
            testres.cfgdict_list = cfgdict_list
            testres.cfgx2_lbl = cfgx2_lbl  # hack override
        except IOError:
            pass
        else:
            if ut.NOT_QUIET:
                ut.colorprint('[harn] single testres cache hit... returning', 'turquoise')
            return testres

    if ibs.table_cache:
        # HACK
        prev_feat_cfgstr = None

    cfgx2_cmsinfo = []
    cfgiter = subindexer_partial(
        range(len(cfgx2_qreq_)), lbl='pipe config', freq=1, adjust=False
    )
    # Run each pipeline configuration
    for cfgx in cfgiter:
        qreq_ = cfgx2_qreq_[cfgx]
        cprint = ut.colorprint
        cprint('testnameid=%r' % (testnameid,), 'green')
        cprint(
            'annot_cfgstr = %s' % (qreq_.get_cfgstr(with_input=True, with_pipe=False),),
            'yellow',
        )
        cprint('pipe_cfgstr= %s' % (qreq_.get_cfgstr(with_data=False),), 'turquoise')
        cprint('pipe_hashstr = %s' % (qreq_.get_pipe_hashid(),), 'teal')
        if DRY_RUN:
            continue

        indent_prefix = '[%s cfg %d/%d]' % (
            dbname,
            # cfgiter.count (doesnt work when quiet)
            (cfgiter.parent_index * cfgiter.length) + cfgx,
            cfgiter.length * cfgiter.parent_length,
        )

        with ut.Indenter(indent_prefix):
            # Run the test / read cache
            _need_compute = True
            if use_cache:
                # smaller cache for individual configuration runs
                st_cfgstr = qreq_.get_cfgstr(with_input=True)
                st_cachedir = ut.unixjoin(bt_cachedir, 'small_tests')
                st_cachename = 'smalltest'
                ut.ensuredir(st_cachedir)
                try:
                    cmsinfo = ut.load_cache(st_cachedir, st_cachename, st_cfgstr)
                except IOError:
                    _need_compute = True
                else:
                    _need_compute = False
            if _need_compute:
                assert not ibs.table_cache
                if ibs.table_cache:
                    if len(
                        prev_feat_cfgstr is not None
                        and prev_feat_cfgstr != qreq_.qparams.feat_cfgstr
                    ):
                        # Clear features to preserve memory
                        ibs.clear_table_cache()
                        # qreq_.ibs.print_cachestats_str()
                cm_list = qreq_.execute()
                cmsinfo = test_result.build_cmsinfo(cm_list, qreq_)
                # record previous feature configuration
                if ibs.table_cache:
                    prev_feat_cfgstr = qreq_.qparams.feat_cfgstr
                if use_cache:
                    ut.save_cache(st_cachedir, st_cachename, st_cfgstr, cmsinfo)
        if not NOMEMORY:
            # Store the results
            cfgx2_cmsinfo.append(cmsinfo)
        else:
            cfgx2_qreq_[cfgx] = None
    if ut.NOT_QUIET:
        ut.colorprint('[harn] Completed running test configurations', 'white')
    if DRY_RUN:
        print('ran tests dryrun mode.')
        return
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    # Store all pipeline config results in a test result object
    testres = test_result.TestResult(pipecfg_list, cfgx2_lbl, cfgx2_cmsinfo, cfgx2_qreq_)
    testres.testnameid = testnameid
    testres.lbl = lbl
    testres.cfgdict_list = cfgdict_list
    testres.aidcfg = None
    if use_cache:
        try:
            ut.save_cache(bt_cachedir, bt_cachename, bt_cachestr, testres)
        except Exception as ex:
            ut.printex(ex, 'error saving testres cache', iswarning=True)
            if ut.SUPER_STRICT:
                raise
    return testres


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.harness
        python -m wbia.expt.harness --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
