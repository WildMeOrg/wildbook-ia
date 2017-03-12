# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
print, rrr, profile = ut.inject2(__name__)


VERB_PIPELINE = ut.get_argflag(('--verb-pipeline', '--verb-pipe')) or ut.VERYVERBOSE
VERB_TESTDATA = ut.get_argflag('--verb-testdata') or ut.VERYVERBOSE


def testrun_pipeline_upto(qreq_, stop_node='end', verbose=True):
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
        # vsone_reranking,
        build_impossible_daids_list)

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
    nns_list = nearest_neighbors(qreq_, Kpad_list, impossible_daids_list,
                                 verbose=verbose)
    #---
    if stop_node == 'baseline_neighbor_filter':
        return locals()
    nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list,
                                             impossible_daids_list,
                                             verbose=verbose)
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

    if stop_node == 'end':
        return locals()
    #---
    # if stop_node == 'vsone_reranking':
    #     return locals()
    # if qreq_.qparams.rrvsone_on:
    #     # VSONE RERANKING
    #     cm_list_VSONERR = vsone_reranking(qreq_, cm_list_SVER, verbose=verbose)
    #     cm_list = cm_list_VSONERR
    # else:
    #     cm_list = cm_list_SVER

    assert False, 'unknown stop_node=%r' % (stop_node,)

    #qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


def testdata_pre(stopnode, defaultdb='testdb1', p=['default'],
                 a=['default:qindex=0:1,dindex=0:5'], **kwargs):
    """
    New (1-1-2016) generic pipeline node testdata getter

    Args:
        stopnode (str): name of pipeline function to be tested
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
    if stopnode == 'end' :
        argnames = ['cm_list_SVER']
    else:
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


#+--- OTHER TESTDATA FUNCS ---


def testdata_sparse_matchinfo_nonagg(defaultdb='testdb1', p=['default']):
    qreq_, args = testdata_pre('build_chipmatches', defaultdb=defaultdb, p=p)
    internal_index = 1 if qreq_.qparams.vsone else 0
    # qaid = qreq_.qaids[0]
    # daid = qreq_.daids[1]
    qaid = qreq_.qaids[0]
    daid = qreq_.daids[1]
    nns                 = args.nns_list[internal_index]
    neighb_idx, neighb_dist = args.nns_list[internal_index]
    neighb_valid0         = args.nnvalid0_list[internal_index]
    neighb_score_list     = args.filtweights_list[internal_index]
    neighb_valid_list     = args.filtvalids_list[internal_index]
    neighb_normk          = args.filtnormks_list[internal_index]
    Knorm = qreq_.qparams.Knorm
    args = (nns, neighb_idx, neighb_valid0, neighb_score_list,
            neighb_valid_list, neighb_normk, Knorm)
    return qreq_, qaid, daid, args


def testdata_pre_baselinefilter(defaultdb='testdb1', qaid_list=None, daid_list=None, codename='vsmany'):
    cfgdict = dict(codename=codename)
    import ibeis
    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb, default_qaids=qaid_list,
                                 default_daids=daid_list, p=p)
    locals_ = testrun_pipeline_upto(qreq_, 'baseline_neighbor_filter')
    nns_list, impossible_daids_list = ut.dict_take(locals_,
                                                   ['nns_list', 'impossible_daids_list'])
    return qreq_, nns_list, impossible_daids_list


def testdata_pre_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    #from ibeis.algo import Config
    cfgdict = dict()
    import ibeis
    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb, default_qaids=qaid_list,
                                 default_daids=daid_list, p=p)
    ibs = qreq_.ibs
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
    import ibeis
    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb, default_qaids=qaid_list, default_daids=daid_list, p=p)
    ibs = qreq_.ibs
    locals_ = testrun_pipeline_upto(qreq_, 'end')
    cm_list = locals_['cm_list_SVER']
    #nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


def testdata_pre_vsonerr(defaultdb='PZ_MTEST', qaid_list=[1], daid_list='all'):
    """
        >>> from ibeis.algo.hots._pipeline_helpers import *  # NOQA
    """
    cfgdict = dict(sver_output_weighting=True, codename='vsmany', rrvsone_on=True)
    import ibeis
    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb, default_qaids=qaid_list, default_daids=daid_list, p=p)
    ibs = qreq_.ibs
    qaid_list = qreq_.qaids.tolist()
    qaid = qaid_list[0]
    #daid_list = qreq_.daids.tolist()
    if len(ibs.get_annot_groundtruth(qaid)) == 0:
        print('WARNING: qaid=%r has no groundtruth' % (qaid,))
    locals_ = testrun_pipeline_upto(qreq_, 'end')
    cm_list = locals_['cm_list_SVER']
    return ibs, qreq_, cm_list, qaid_list


def testdata_scoring(defaultdb='PZ_MTEST', qaid_list=[1], daid_list='all'):
    from ibeis.algo.hots import vsone_pipeline
    ibs, qreq_, prior_cm = testdata_matching(defaultdb=defaultdb, qaid_list=qaid_list, daid_list=daid_list)
    config = qreq_.qparams
    cm = vsone_pipeline.refine_matches(qreq_, prior_cm, config)
    cm.evaluate_dnids(qreq_)
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
