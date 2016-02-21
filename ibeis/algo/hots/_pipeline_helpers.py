# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[_plh]')


VERB_PIPELINE = ut.get_argflag(('--verb-pipeline', '--verb-pipe')) or ut.VERYVERBOSE
VERB_TESTDATA = ut.get_argflag('--verb-testdata') or ut.VERYVERBOSE


def testrun_pipeline_upto(qreq_, stop_node=None, verbose=True):
    r"""
    Main tester function. Runs the pipeline by mirroring
    `request_ibeis_query_L0`, but stops at a requested breakpoint and returns
    the local variables.

    convinience: runs pipeline for tests
    this should mirror request_ibeis_query_L0

    Ignore:
        >>> # TODO: autogenerate
        >>> # The following is a stub that starts the autogeneration process
        >>> import utool as ut
        >>> from ibeis.algo.hots import pipeline
        >>> source = ut.get_func_sourcecode(pipeline.request_ibeis_query_L0,
        >>>                                 strip_docstr=True, stripdef=True,
        >>>                                 strip_comments=True)
        >>> import re
        >>> source = re.sub(r'^\s*$\n', '', source, flags=re.MULTILINE)
        >>> print(source)
        >>> ut.replace_between_tags(source, '', sentinal)
    """
    from ibeis.algo.hots.pipeline import (
        nearest_neighbors, baseline_neighbor_filter, weight_neighbors,
        build_chipmatches, spatial_verification,
        vsone_reranking, build_impossible_daids_list)

    print('RUN PIPELINE UPTO: %s' % (stop_node,))

    print(qreq_)

    qreq_.lazy_load(verbose=verbose)
    #---
    if stop_node == 'build_impossible_daids_list':
        return locals()
    impossible_daids_list, Kpad_list = build_impossible_daids_list(qreq_)
    #---
    if stop_node == 'nearest_neighbors':
        return locals()
    nns_list = nearest_neighbors(qreq_, Kpad_list, verbose=verbose)
    #---
    if stop_node == 'baseline_neighbor_filter':
        return locals()
    nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list, impossible_daids_list, verbose=verbose)
    #---
    if stop_node == 'weight_neighbors':
        return locals()
    weight_ret = weight_neighbors(qreq_, nns_list, nnvalid0_list, verbose=verbose)
    filtkey_list, filtweights_list, filtvalids_list, filtnormks_list = weight_ret
    #---
    if stop_node == 'filter_neighbors':
        raise AssertionError('no longer exists')
    #---
    if stop_node == 'build_chipmatches':
        return locals()
    cm_list_FILT = build_chipmatches(qreq_, nns_list, nnvalid0_list,
                                     filtkey_list, filtweights_list,
                                     filtvalids_list, filtnormks_list,
                                     verbose=verbose)
    #---
    if stop_node == 'spatial_verification':
        return locals()
    cm_list_SVER = spatial_verification(qreq_, cm_list_FILT, verbose=verbose)
    #---
    if stop_node == 'vsone_reranking':
        return locals()
    if qreq_.qparams.rrvsone_on:
        # VSONE RERANKING
        cm_list_VSONERR = vsone_reranking(qreq_, cm_list_SVER, verbose=verbose)
        cm_list = cm_list_VSONERR
    else:
        cm_list = cm_list_SVER

    assert False, 'unknown stop_node=%r' % (stop_node,)

    #qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


def testdata_pre(stopnode, defaultdb='testdb1', p=['default'],
                 a=['default:qindex=0:1,dindex=0:5'], **kwargs):
    """ New (1-1-2016) generic pipeline node testdata getter

    Args:
        stopnode (str):
        defaultdb (str): (default = u'testdb1')
        p (list): (default = [u'default:'])
        a (list): (default = [u'default:qsize=1,dsize=4'])

    Returns:
        tuple: (ibs, qreq_, args)

    CommandLine:
        python -m ibeis.algo.hots._pipeline_helpers --exec-testdata_pre --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
        >>> stopnode = 'build_chipmatches'
        >>> defaultdb = 'testdb1'
        >>> p = ['default:']
        >>> a = ['default:qindex=0:1,dindex=0:5']
        >>> qreq_, args = testdata_pre(stopnode, defaultdb, p, a)
    """
    import ibeis
    from ibeis.algo.hots import pipeline
    qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb, p=p, a=a, **kwargs)
    locals_ = testrun_pipeline_upto(qreq_, stopnode)
    func = getattr(pipeline, stopnode)
    argnames = ut.get_argnames(func)
    # Hack to ignore qreq_, and verbose
    for ignore in ['qreq_', 'ibs', 'verbose']:
        try:
            argnames.remove(ignore)
        except ValueError:
            pass
    tupname = '_Ret_' + stopnode.upper()
    args = ut.dict_take_asnametup(locals_, argnames, name=tupname)
    return qreq_, args


def get_pipeline_testdata(dbname=None,
                          cfgdict=None,
                          qaid_list=None,
                          daid_list=None,
                          defaultdb='testdb1',
                          cmdline_ok=True,
                          preload=True):
    r"""
    Gets testdata for pipeline defined by tests / and or command line

    DEPRICATE in favor of ibeis.init.main_helpers.testdata_qreq

    Args:
        cmdline_ok : if false does not check command line

    Returns:
        tuple: ibs, qreq_

    CommandLine:
        python -m ibeis.algo.hots._pipeline_helpers --test-get_pipeline_testdata
        python -m ibeis.algo.hots._pipeline_helpers --test-get_pipeline_testdata --daid_list 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.algo.hots._pipeline_helpers --test-get_pipeline_testdata --daids 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.algo.hots._pipeline_helpers --test-get_pipeline_testdata --qaid 41 --db PZ_MTEST
        python -m ibeis.algo.hots._pipeline_helpers --test-get_pipeline_testdata --controlled_daids --qaids=41 --db PZ_MTEST --verb-testdata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots._pipeline_helpers import *
        >>> import ibeis  # NOQA
        >>> from ibeis.algo.hots import _pipeline_helpers as plh
        >>> cfgdict = dict(pipeline_root='vsone', codename='vsone')
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict)
        >>> result = ''
        >>> result += ('daids = %r\n' % (qreq_.get_external_daids(),))
        >>> result += ('qaids = %r' % (qreq_.get_external_qaids(),))
        >>> print('cfgstr %s'  % (qreq_.qparams.query_cfgstr,))
        >>> print(result)

        daids = array([1, 2, 3, 4, 5])
        qaids = array([1])
    """
    import ibeis
    from ibeis.algo.hots import query_request
    # Allow commandline specification if paramaters are not specified in tests
    if cfgdict is None:
        cfgdict = {}

    assert cmdline_ok is True, 'cmdline_ok should always be True'

    if cmdline_ok:
        from ibeis.algo import Config
        # Allow specification of db and qaids/daids
        if dbname is not None:
            defaultdb = dbname
        dbname = ut.get_argval('--db', type_=str, default=defaultdb)

    ibs = ibeis.opendb(defaultdb=dbname)

    default_qaid_list = qaid_list
    default_daid_list = daid_list

    # setup special defautls

    if default_qaid_list is None:
        default_qaid_list = {
            'testdb1' : [1],
            'GZ_ALL'  : [1032],
            'PZ_ALL'  : [1, 3, 5, 9],
        }.get(dbname, [1])

    default_daid_list = ut.get_argval(('--daids', '--daid-list'), type_=list, default=default_daid_list)

    if default_daid_list is None:
        if dbname == 'testdb1':
            default_daid_list = ibs.get_valid_aids()[0:5]
        else:
            default_daid_list = 'all'

    # Use commmand line parsing for custom values

    if cmdline_ok:
        import ibeis
        qaid_list_, daid_list_ = ibeis.testdata_expanded_aids(ibs=ibs, a='default',
                                                              default_qaids=default_qaid_list,
                                                              default_daids=default_daid_list)
        # from ibeis.init import main_helpers
        # qaid_list_ = main_helpers.get_test_qaids(ibs, default_qaids=default_qaid_list)
        # daid_list_ = main_helpers.get_test_daids(ibs, default_daids=default_daid_list, qaid_list=qaid_list_)
        #
        # Allow commond line specification of all query params
        default_cfgdict = dict(Config.parse_config_items(Config.QueryConfig()))
        default_cfgdict.update(cfgdict)
        _orig_cfgdict = cfgdict
        force_keys = set(list(_orig_cfgdict.keys()))
        cfgdict_ = ut.util_arg.argparse_dict(
            default_cfgdict, verbose=not ut.QUIET, only_specified=True,
            force_keys=force_keys)
        #ut.embed()
        if VERB_PIPELINE or VERB_TESTDATA:
            print('[plh] cfgdict_ = ' + ut.dict_str(cfgdict_))
    else:
        qaid_list_ = qaid_list
        daid_list_ = daid_list
        cfgdict_ = cfgdict

    #ibs = ibeis.test_main(db=dbname)

    if VERB_TESTDATA:
        #ibeis.other.dbinfo.print_qd_info(ibs, qaid_list_, daid_list_, verbose=True)
        ibeis.other.dbinfo.print_qd_info(ibs, qaid_list_, daid_list_, verbose=False)

    if 'with_metadata' not in cfgdict:
        cfgdict_['with_metadata'] = True
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list_, daid_list_, cfgdict=cfgdict_)
    if preload:
        qreq_.lazy_load()
    return ibs, qreq_


#+--- OTHER TESTDATA FUNCS ---


def testdata_pre_weight_neighbors(defaultdb='testdb1', qaid_list=[1, 2], daid_list=None, codename='vsmany', cfgdict=None):
    if cfgdict is None:
        cfgdict = dict(codename=codename)
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'weight_neighbors')
    nns_list, nnvalid0_list = ut.dict_take(locals_, ['nns_list', 'nnvalid0_list'])

    # qreq_, args = testdata_pre('weight_neighbors', defaultdb=defaultdb, p=['default:bar_l2_on=True,fg_on=False'])
    return ibs, qreq_, nns_list, nnvalid0_list


def testdata_sparse_matchinfo_nonagg(defaultdb='testdb1', p=['default']):
    qreq_, args = testdata_pre('build_chipmatches', defaultdb=defaultdb, p=p)
    internal_index = 1 if qreq_.qparams.vsone else 0
    # qaid = qreq_.qaids[0]
    # daid = qreq_.daids[1]
    qaid = qreq_.get_external_qaids()[0]
    daid = qreq_.get_external_daids()[1]
    qfx2_idx, qfx2_dist = args.nns_list[internal_index]
    qfx2_valid0         = args.nnvalid0_list[internal_index]
    qfx2_score_list     = args.filtweights_list[internal_index]
    qfx2_valid_list     = args.filtvalids_list[internal_index]
    qfx2_normk          = args.filtnormks_list[internal_index]
    Knorm = qreq_.qparams.Knorm
    args = (qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list, qfx2_normk, Knorm)
    return qreq_, qaid, daid, args


def testdata_pre_baselinefilter(defaultdb='testdb1', qaid_list=None, daid_list=None, codename='vsmany'):
    cfgdict = dict(codename=codename)
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'baseline_neighbor_filter')
    nns_list, impossible_daids_list = ut.dict_take(locals_, ['nns_list', 'impossible_daids_list'])
    return qreq_, nns_list, impossible_daids_list


def testdata_pre_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    #from ibeis.algo import Config
    cfgdict = dict()
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'spatial_verification')
    cm_list = locals_['cm_list_FILT']
    #nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


def testdata_post_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None, codename='vsmany', cfgdict=None):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    #from ibeis.algo import Config
    if cfgdict is None:
        cfgdict = dict(codename=codename)
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'vsone_reranking')
    cm_list = locals_['cm_list_SVER']
    #nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


def testdata_pre_vsonerr(defaultdb='PZ_MTEST', qaid_list=[1], daid_list='all'):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    cfgdict = dict(sver_output_weighting=True, codename='vsmany', rrvsone_on=True)
    # Get pipeline testdata for this configuration
    ibs, qreq_ = get_pipeline_testdata(
        cfgdict=cfgdict, qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cmdline_ok=True)
    qaid_list = qreq_.get_external_qaids().tolist()
    qaid = qaid_list[0]
    #daid_list = qreq_.get_external_daids().tolist()
    if len(ibs.get_annot_groundtruth(qaid)) == 0:
        print('WARNING: qaid=%r has no groundtruth' % (qaid,))
    locals_ = testrun_pipeline_upto(qreq_, 'vsone_reranking')
    cm_list = locals_['cm_list_SVER']
    return ibs, qreq_, cm_list, qaid_list


def testdata_scoring(defaultdb='PZ_MTEST', qaid_list=[1], daid_list='all'):
    from ibeis.algo.hots import vsone_pipeline
    ibs, qreq_, prior_cm = testdata_matching(defaultdb=defaultdb, qaid_list=qaid_list, daid_list=daid_list)
    config = qreq_.qparams
    cm = vsone_pipeline.refine_matches(qreq_, prior_cm, config)
    cm.evaluate_dnids(qreq_.ibs)
    return qreq_, cm


def testdata_matching(*args, **kwargs):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    from ibeis.algo.hots import vsone_pipeline
    from ibeis.algo.hots import scoring
    from ibeis.algo.hots import pipeline  # NOQA
    ibs, qreq_, cm_list, qaid_list  = testdata_pre_vsonerr(*args, **kwargs)
    vsone_pipeline.prepare_vsmany_chipmatch(qreq_, cm_list)
    nNameShortlist = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName  = qreq_.qparams.nAnnotPerNameVsone
    scoring.score_chipmatch_list(qreq_, cm_list, 'nsum')
    vsone_pipeline.prepare_vsmany_chipmatch(qreq_, cm_list)
    cm_shortlist = scoring.make_chipmatch_shortlists(qreq_, cm_list, nNameShortlist, nAnnotPerName)
    prior_cm      = cm_shortlist[0]
    return ibs, qreq_, prior_cm


#L_______


def print_nearest_neighbor_assignments(qvecs_list, nns_list):
    nQAnnots = len(qvecs_list)
    nTotalDesc = sum(map(len, qvecs_list))
    nTotalNN = sum([qfx2_idx.size for (qfx2_idx, qfx2_dist) in nns_list])
    print('[hs] * assigned %d desc (from %d annots) to %r nearest neighbors'
          % (nTotalDesc, nQAnnots, nTotalNN))


def _self_verbose_check(qfx2_notsamechip, qfx2_valid0):
    nInvalidChips = ((True - qfx2_notsamechip)).sum()
    nNewInvalidChips = (qfx2_valid0 * (True - qfx2_notsamechip)).sum()
    total = qfx2_valid0.size
    print('[hs] * self invalidates %d/%d assignments' % (nInvalidChips, total))
    print('[hs] * %d are newly invalided by self' % (nNewInvalidChips))


def _samename_verbose_check(qfx2_notsamename, qfx2_valid0):
    nInvalidNames = ((True - qfx2_notsamename)).sum()
    nNewInvalidNames = (qfx2_valid0 * (True - qfx2_notsamename)).sum()
    total = qfx2_valid0.size
    print('[hs] * nid invalidates %d/%d assignments' % (nInvalidNames, total))
    print('[hs] * %d are newly invalided by nid' % nNewInvalidNames)


def _sameimg_verbose_check(qfx2_notsameimg, qfx2_valid0):
    nInvalidImgs = ((True - qfx2_notsameimg)).sum()
    nNewInvalidImgs = (qfx2_valid0 * (True - qfx2_notsameimg)).sum()
    total = qfx2_valid0.size
    print('[hs] * gid invalidates %d/%d assignments' % (nInvalidImgs, total))
    print('[hs] * %d are newly invalided by gid' % nNewInvalidImgs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots._pipeline_helpers
        python -m ibeis.algo.hots._pipeline_helpers --allexamples
        python -m ibeis.algo.hots._pipeline_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
