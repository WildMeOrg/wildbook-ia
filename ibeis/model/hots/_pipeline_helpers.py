from __future__ import absolute_import, division, print_function
#import six
#import numpy as np
#from ibeis.model.hots import hstypes
import utool as ut
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[_plh]', DEBUG=False)


VERB_PIPELINE = ut.get_argflag(('--verb-pipeline', '--verb-pipe')) or ut.VERYVERBOSE
VERB_TESTDATA = ut.get_argflag('--verb-testdata') or ut.VERYVERBOSE


def testdata_hs2(defaultdb='testdb1', qaids=None, daids=None, cfgdict=None, stop_node=None, argnames=[]):
    """
    CommandLine:
        python -m ibeis.model.hots._pipeline_helpers --test-testdata_hs2

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots._pipeline_helpers import *
        >>> defaultdb = 'testdb1'
        >>> qaids = None
        >>> daids = None
        >>> stop_node = 'build_chipmatches'
        >>> cfgdict = None
        >>> argnames = []
        >>> # execute function
        >>> result = testdata_hs2(defaultdb, qaids, daids, cfgdict, stop_node, argnames)
        >>> # verify results
        >>> print(result)
    """
    from ibeis.model.hots import pipeline
    func = getattr(pipeline, stop_node)
    ibs, qreq_ = get_pipeline_testdata(qaid_list=qaids, daid_list=daids, defaultdb=defaultdb, cfgdict=cfgdict)
    func_argnames = ut.get_func_argspec(func).args
    locals_ = testrun_pipeline_upto(qreq_, stop_node)
    args = ut.dict_take(locals_, func_argnames)
    return args


def testrun_pipeline_upto(qreq_, stop_node=None, verbose=True):
    r"""
    convinience: runs pipeline for tests
    this should mirror request_ibeis_query_L0

    Ignore:
        >>> import utool as ut
        >>> from ibeis.model.hots import pipeline
        >>> source = ut.get_func_sourcecode(pipeline.request_ibeis_query_L0)
        >>> stripsource = source[:]
        >>> stripsource = ut.strip_line_comments(stripsource)
        >>> triplequote1 = ut.TRIPLE_DOUBLE_QUOTE
        >>> triplequote2 = ut.TRIPLE_SINGLE_QUOTE
        >>> docstr_regex1 = 'r?' + triplequote1 + '.* + ' + triplequote1 + '\n    '
        >>> docstr_regex2 = 'r?' + triplequote2 + '.* + ' + triplequote2 + '\n    '
        >>> stripsource = ut.regex_replace(docstr_regex1, '', stripsource)
        >>> stripsource = ut.regex_replace(docstr_regex2, '', stripsource)
        >>> stripsource = ut.strip_line_comments(stripsource)
        >>> print(stripsource)
    """
    from ibeis.model.hots.pipeline import (
        nearest_neighbors, baseline_neighbor_filter, weight_neighbors,
        build_chipmatches, spatial_verification,
        chipmatch_to_resdict, vsone_reranking, build_impossible_daids_list)

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
    filtkey_list, filtweights_list, filtvalids_list = weight_neighbors(qreq_, nns_list, nnvalid0_list, verbose=verbose)
    #---
    if stop_node == 'filter_neighbors':
        raise AssertionError('no longer exists')
    #---
    if stop_node == 'build_chipmatches':
        return locals()
    cm_list_FILT = build_chipmatches(qreq_, nns_list, nnvalid0_list, filtkey_list, filtweights_list, filtvalids_list, verbose=verbose)
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
    #---
    if stop_node == 'chipmatch_to_resdict':
        return locals()
    qaid2_qres = chipmatch_to_resdict(qreq_, cm_list, verbose=verbose)

    assert False, 'unknown stop_node=%r' % (stop_node,)

    #qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


def get_pipeline_testdata(dbname=None,
                          cfgdict=None,
                          qaid_list=None,
                          daid_list=None,
                          defaultdb='testdb1',
                          cmdline_ok=True,
                          preload=True):
    r"""
    Gets testdata for pipeline defined by tests / and or command line

    Args:
        cmdline_ok : if false does not check command line

    Returns:
        tuple: ibs, qreq_

    CommandLine:
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --daid_list 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --daids 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --controlled_daids --qaids=41 --db PZ_MTEST --verb-testdata

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh
        >>> cfgdict = dict(pipeline_root='vsone', codename='vsone')
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict)
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
    from ibeis.model.hots import query_request
    # Allow commandline specification if paramaters are not specified in tests
    if cfgdict is None:
        cfgdict = {}

    assert cmdline_ok is True, 'cmdline_ok should always be True'

    if cmdline_ok:
        from ibeis.model import Config
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
        from ibeis.init import main_helpers
        qaid_list_ = main_helpers.get_test_qaids(ibs, default_qaids=default_qaid_list)
        daid_list_ = main_helpers.get_test_daids(ibs, default_daids=default_daid_list, qaid_list=qaid_list_)
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
    return ibs, qreq_, nns_list, nnvalid0_list


def testdata_sparse_matchinfo_nonagg(defaultdb='testdb1', codename='vsmany'):
    ibs, qreq_, args = testdata_pre_build_chipmatch(defaultdb=defaultdb, codename=codename)
    nns_list, nnvalid0_list, filtkey_list, filtweights_list, filtvalids_list = args
    if qreq_.qparams.vsone:
        interal_index = 1
    else:
        interal_index = 0
    qaid = qreq_.get_external_qaids()[0]
    daid = qreq_.get_external_daids()[1]
    qfx2_idx, _     = nns_list[interal_index]
    qfx2_valid0     = nnvalid0_list[interal_index]
    qfx2_score_list = filtweights_list[interal_index]
    qfx2_valid_list = filtvalids_list[interal_index]
    args = qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list
    return qreq_, qaid, daid, args


def testdata_pre_build_chipmatch(defaultdb='testdb1', codename='vsmany'):
    cfgdict = dict(codename=codename)
    ibs, qreq_ = get_pipeline_testdata(defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'build_chipmatches')
    args = ut.dict_take(locals_, ['nns_list', 'nnvalid0_list', 'filtkey_list', 'filtweights_list', 'filtvalids_list'])
    return ibs, qreq_, args


def testdata_pre_baselinefilter(defaultdb='testdb1', qaid_list=None, daid_list=None, codename='vsmany'):
    cfgdict = dict(codename=codename)
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'baseline_neighbor_filter')
    nns_list, impossible_daids_list = ut.dict_take(locals_, ['nns_list', 'impossible_daids_list'])
    return qreq_, nns_list, impossible_daids_list


def testdata_pre_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None):
    """
        >>> from ibeis.model.hots._pipeline_helpers import *  # NOQA
    """
    #from ibeis.model import Config
    cfgdict = dict()
    ibs, qreq_ = get_pipeline_testdata(
        qaid_list=qaid_list, daid_list=daid_list, defaultdb=defaultdb, cfgdict=cfgdict)
    locals_ = testrun_pipeline_upto(qreq_, 'spatial_verification')
    cm_list = locals_['cm_list_FILT']
    #nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


def get_pipeline_testdata2(defaultdb='testdb1', default_aidcfg_name_list=['default'], default_test_cfg_name_list=['default'], preload=False):
    r"""
    Gets testdata for pipeline defined by tests / and or command line

    Args:
        cmdline_ok : if false does not check command line

    Returns:
        tuple: ibs, qreq_

    CommandLine:
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --daid_list 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --daids 39 --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --qaid 41 --db PZ_MTEST
        python -m ibeis.model.hots._pipeline_helpers --test-get_pipeline_testdata --controlled_daids --qaids=41 --db PZ_MTEST --verb-testdata

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh
        >>> ibs, qreq_list = plh.get_pipeline_testdata2('PZ_MTEST', default_test_cfg_name_list=['candidacy_namescore'])

        daids = array([1, 2, 3, 4, 5])
        qaids = array([1])
    """
    #from ibeis.init import main_helpers
    from ibeis.experiments import experiment_helpers
    from ibeis.model.hots import query_request
    #from ibeis.experiments import experiment_helpers
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_aidcfg_name_list)
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=default_test_cfg_name_list)

    # Generate list of query pipeline param configs
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list)
    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list, ibs=ibs)

    qreq_list = []

    for qaids, daids in expanded_aids_list:
        for query_cfg in pipecfg_list:
            qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids, query_cfg=query_cfg)
            qreq_list.append(qreq_)

    if preload:
        for qreq_ in qreq_list:
            qreq_.lazy_load()
    return ibs, qreq_list


def testdata_pre_sver2(*args, **kwargs):
    """
        >>> from ibeis.model.hots._pipeline_helpers import *  # NOQA
        >>> args = (,)
        >>> kwargs = {}
    """
    #from ibeis.model import Config
    ibs, qreq_list = get_pipeline_testdata2(*args, **kwargs)
    locals_list = [testrun_pipeline_upto(qreq_, 'spatial_verification') for qreq_ in qreq_list]
    cms_list = [locals_['cm_list_FILT'] for locals_ in locals_list]
    #nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_list, cms_list


def testdata_post_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None, codename='vsmany', cfgdict=None):
    """
        >>> from ibeis.model.hots._pipeline_helpers import *  # NOQA
    """
    #from ibeis.model import Config
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
        >>> from ibeis.model.hots._pipeline_helpers import *  # NOQA
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
    from ibeis.model.hots import vsone_pipeline
    ibs, qreq_, prior_cm = testdata_matching(defaultdb=defaultdb, qaid_list=qaid_list, daid_list=daid_list)
    config = qreq_.qparams
    cm = vsone_pipeline.refine_matches(qreq_, prior_cm, config)
    cm.evaluate_dnids(qreq_.ibs)
    return qreq_, cm


def testdata_matching(*args, **kwargs):
    """
        >>> from ibeis.model.hots._pipeline_helpers import *  # NOQA
    """
    from ibeis.model.hots import vsone_pipeline
    from ibeis.model.hots import scoring
    ibs, qreq_, cm_list, qaid_list  = testdata_pre_vsonerr(*args, **kwargs)
    vsone_pipeline.prepare_vsmany_chipmatch(qreq_, cm_list)
    nNameShortlist = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName  = qreq_.qparams.nAnnotPerNameVsOne
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
        python -m ibeis.model.hots._pipeline_helpers
        python -m ibeis.model.hots._pipeline_helpers --allexamples
        python -m ibeis.model.hots._pipeline_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
