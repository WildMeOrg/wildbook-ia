# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

print, rrr, profile = ut.inject2(__name__)


VERB_PIPELINE = ut.get_argflag(('--verb-pipeline', '--verb-pipe')) or ut.VERYVERBOSE
VERB_TESTDATA = ut.get_argflag('--verb-testdata') or ut.VERYVERBOSE


def testrun_pipeline_upto(qreq_, stop_node='end', verbose=True):
    r"""
    Main tester function. Runs the pipeline by mirroring
    `request_wbia_query_L0`, but stops at a requested breakpoint and returns
    the local variables.

    convinience: runs pipeline for tests
    this should mirror request_wbia_query_L0

    Ignore:
        >>> # TODO: autogenerate
        >>> # The following is a stub that starts the autogeneration process
        >>> import utool as ut
        >>> from wbia.algo.hots import pipeline
        >>> source = ut.get_func_sourcecode(pipeline.request_wbia_query_L0,
        >>>                                 strip_docstr=True, stripdef=True,
        >>>                                 strip_comments=True)
        >>> import re
        >>> source = re.sub(r'^\s*$\n', '', source, flags=re.MULTILINE)
        >>> print(source)
        >>> ut.replace_between_tags(source, '', sentinal)
    """
    from wbia.algo.hots.pipeline import (
        nearest_neighbors,
        baseline_neighbor_filter,
        weight_neighbors,
        build_chipmatches,
        spatial_verification,
        # vsone_reranking,
        build_impossible_daids_list,
    )

    print('RUN PIPELINE UPTO: %s' % (stop_node,))

    print(qreq_)

    qreq_.lazy_load(verbose=verbose)
    # ---
    if stop_node == 'build_impossible_daids_list':
        return locals()
    impossible_daids_list, Kpad_list = build_impossible_daids_list(qreq_)
    # ---
    if stop_node == 'nearest_neighbors':
        return locals()
    nns_list = nearest_neighbors(qreq_, Kpad_list, impossible_daids_list, verbose=verbose)
    # ---
    if stop_node == 'baseline_neighbor_filter':
        return locals()
    nnvalid0_list = baseline_neighbor_filter(
        qreq_, nns_list, impossible_daids_list, verbose=verbose
    )
    # ---
    if stop_node == 'weight_neighbors':
        return locals()
    weight_ret = weight_neighbors(qreq_, nns_list, nnvalid0_list, verbose=verbose)
    filtkey_list, filtweights_list, filtvalids_list, filtnormks_list = weight_ret
    # ---
    if stop_node == 'filter_neighbors':
        raise AssertionError('no longer exists')
    # ---
    if stop_node == 'build_chipmatches':
        return locals()
    cm_list_FILT = build_chipmatches(
        qreq_,
        nns_list,
        nnvalid0_list,
        filtkey_list,
        filtweights_list,
        filtvalids_list,
        filtnormks_list,
        verbose=verbose,
    )
    # ---
    if stop_node == 'spatial_verification':
        return locals()
    cm_list_SVER = spatial_verification(qreq_, cm_list_FILT, verbose=verbose)

    if stop_node == 'end':
        return locals()

    assert False, 'unknown stop_node=%r' % (stop_node,)

    # qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


def testdata_pre(
    stopnode,
    defaultdb='testdb1',
    p=['default'],
    a=['default:qindex=0:1,dindex=0:5'],
    **kwargs,
):
    """
    New (1-1-2016) generic pipeline node testdata getter

    Args:
        stopnode (str): name of pipeline function to be tested
        defaultdb (str): (default = u'testdb1')
        p (list): (default = [u'default:'])
        a (list): (default = [u'default:qsize=1,dsize=4'])
        **kwargs: passed to testdata_qreq_
            qaid_override, daid_override

    Returns:
        tuple: (ibs, qreq_, args)

    CommandLine:
        python -m wbia.algo.hots._pipeline_helpers --exec-testdata_pre --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._pipeline_helpers import *  # NOQA
        >>> stopnode = 'build_chipmatches'
        >>> defaultdb = 'testdb1'
        >>> p = ['default:']
        >>> a = ['default:qindex=0:1,dindex=0:5']
        >>> qreq_, args = testdata_pre(stopnode, defaultdb, p, a)
    """
    import wbia
    from wbia.algo.hots import pipeline

    qreq_ = wbia.testdata_qreq_(defaultdb=defaultdb, p=p, a=a, **kwargs)
    locals_ = testrun_pipeline_upto(qreq_, stopnode)
    if stopnode == 'end':
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


# +--- OTHER TESTDATA FUNCS ---


def testdata_sparse_matchinfo_nonagg(defaultdb='testdb1', p=['default']):
    qreq_, args = testdata_pre('build_chipmatches', defaultdb=defaultdb, p=p)
    internal_index = 1 if qreq_.qparams.vsone else 0
    # qaid = qreq_.qaids[0]
    # daid = qreq_.daids[1]
    qaid = qreq_.qaids[0]
    daid = qreq_.daids[1]
    nns = args.nns_list[internal_index]
    # neighb_idx, neighb_dist = args.nns_list[internal_index]
    neighb_valid0 = args.nnvalid0_list[internal_index]
    neighb_score_list = args.filtweights_list[internal_index]
    neighb_valid_list = args.filtvalids_list[internal_index]
    neighb_normk = args.filtnormks_list[internal_index]
    Knorm = qreq_.qparams.Knorm
    fsv_col_lbls = args.filtkey_list
    args = (
        nns,
        neighb_valid0,
        neighb_score_list,
        neighb_valid_list,
        neighb_normk,
        Knorm,
        fsv_col_lbls,
    )
    return qreq_, qaid, daid, args


def testdata_pre_baselinefilter(
    defaultdb='testdb1', qaid_list=None, daid_list=None, codename='vsmany'
):
    cfgdict = dict(codename=codename)
    import wbia

    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = wbia.testdata_qreq_(
        defaultdb=defaultdb, default_qaids=qaid_list, default_daids=daid_list, p=p
    )
    locals_ = testrun_pipeline_upto(qreq_, 'baseline_neighbor_filter')
    nns_list, impossible_daids_list = ut.dict_take(
        locals_, ['nns_list', 'impossible_daids_list']
    )
    return qreq_, nns_list, impossible_daids_list


def testdata_pre_sver(defaultdb='PZ_MTEST', qaid_list=None, daid_list=None):
    """
        >>> from wbia.algo.hots._pipeline_helpers import *  # NOQA
    """
    # TODO: testdata_pre('sver')
    # from wbia.algo import Config
    cfgdict = dict()
    import wbia

    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = wbia.testdata_qreq_(
        defaultdb=defaultdb, default_qaids=qaid_list, default_daids=daid_list, p=p
    )
    ibs = qreq_.ibs
    locals_ = testrun_pipeline_upto(qreq_, 'spatial_verification')
    cm_list = locals_['cm_list_FILT']
    # nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


def testdata_post_sver(
    defaultdb='PZ_MTEST', qaid_list=None, daid_list=None, codename='vsmany', cfgdict=None,
):
    """
        >>> from wbia.algo.hots._pipeline_helpers import *  # NOQA
    """
    # TODO: testdata_pre('end')
    # from wbia.algo import Config
    if cfgdict is None:
        cfgdict = dict(codename=codename)
    import wbia

    p = 'default' + ut.get_cfg_lbl(cfgdict)
    qreq_ = wbia.testdata_qreq_(
        defaultdb=defaultdb, default_qaids=qaid_list, default_daids=daid_list, p=p
    )
    ibs = qreq_.ibs
    locals_ = testrun_pipeline_upto(qreq_, 'end')
    cm_list = locals_['cm_list_SVER']
    # nnfilts_list   = locals_['nnfilts_list']
    return ibs, qreq_, cm_list


# L_______


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots._pipeline_helpers
        python -m wbia.algo.hots._pipeline_helpers --allexamples
        python -m wbia.algo.hots._pipeline_helpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
