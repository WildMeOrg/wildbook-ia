from __future__ import absolute_import, division, print_function
import six
import numpy as np
from ibeis.model.hots import hstypes
import utool as ut
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[_plh]', DEBUG=False)


def testrun_pipeline_upto(qreq_, stop_node=None, verbose=True):
    r"""
    convinience: runs pipeline for tests
    this should mirror request_ibeis_query_L0

    Ignore:
        >>> import utool as ut
        >>> source = ut.get_func_sourcecode(request_ibeis_query_L0)
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
        filter_neighbors, build_chipmatches, spatial_verification,
        chipmatch_to_resdict, vsone_reranking)

    qreq_.lazy_load(verbose=verbose)
    #---
    if stop_node == 'nearest_neighbors':
        return locals()
    qaid2_nns = nearest_neighbors(qreq_, verbose=verbose)
    #---
    if stop_node == 'baseline_neighbor_filter':
        return locals()
    qaid2_nnvalid0 = baseline_neighbor_filter(qreq_, qaid2_nns, verbose=verbose)
    #---
    if stop_node == 'weight_neighbors':
        return locals()
    qaid2_filtweights = weight_neighbors(qreq_, qaid2_nns, qaid2_nnvalid0, verbose=verbose)
    #---
    if stop_node == 'filter_neighbors':
        return locals()
    qaid2_nnfilts, qaid2_nnfiltagg = filter_neighbors(qreq_, qaid2_nns, qaid2_nnvalid0, qaid2_filtweights, verbose=verbose)
    #---
    if stop_node == 'build_chipmatches':
        return locals()
    qaid2_chipmatch_FILT = build_chipmatches(qreq_, qaid2_nns, qaid2_nnvalid0, qaid2_nnfilts, qaid2_nnfiltagg, verbose=verbose)
    #---
    if stop_node == 'spatial_verification':
        return locals()
    qaid2_chipmatch_SVER = spatial_verification(qreq_, qaid2_chipmatch_FILT, verbose=verbose)
    #---
    if stop_node == 'vsone_reranking':
        return locals()
    if qreq_.qparams.rrvsone_on:
        # VSONE RERANKING
        qaid2_chipmatch_VSONERR = vsone_reranking(qreq_, qaid2_chipmatch_SVER, verbose=verbose)
        qaid2_chipmatch = qaid2_chipmatch_VSONERR
    else:
        qaid2_chipmatch = qaid2_chipmatch_SVER
    #---
    if stop_node == 'chipmatch_to_resdict':
        return locals()
    qaid2_qres = chipmatch_to_resdict(qreq_, qaid2_chipmatch_SVER, verbose=verbose)

    #qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


def get_pipeline_testdata(dbname=None, cfgdict={}, qaid_list=None,
                          daid_list=None):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh
        >>> cfgdict = dict(pipeline_root='vsone', codename='vsone')
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict)
        >>> print(qreq_.qparams.query_cfgstr)
    """
    import ibeis
    from ibeis.model.hots import query_request
    if dbname is None:
        dbname = ut.get_argval('--db', str, 'testdb1')
    ibs = ibeis.opendb(dbname)
    if qaid_list is None:
        if dbname == 'testdb1':
            qaid_list = [1]
        if dbname == 'GZ_ALL':
            qaid_list = [1032]
        if dbname == 'PZ_ALL':
            qaid_list = [1, 3, 5, 9]
        else:
            qaid_list = [1]
    if daid_list is None:
        daid_list = ibs.get_valid_aids()
        if dbname == 'testdb1':
            daid_list = daid_list[0:min(5, len(daid_list))]
    elif daid_list == 'all':
        daid_list = ibs.get_valid_aids()
    ibs = ibeis.test_main(db=dbname)
    if 'with_metadata' not in cfgdict:
        cfgdict['with_metadata'] = True
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict=cfgdict)
    qreq_.lazy_load()
    return ibs, qreq_


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


def identity_filter(qreq_, qaid2_nns):
    """ testing function returns unfiltered nearest neighbors
    this does check that you are not matching yourself
    """
    K = qreq_.qparams.K
    qaid2_valid0 = {}
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        (qfx2_idx, _) = qaid2_nns[qaid]
        qfx2_nnidx = qfx2_idx[:, 0:K]
        qfx2_score = np.ones(qfx2_nnidx.shape, dtype=hstypes.FS_DTYPE)
        qfx2_valid0 = np.ones(qfx2_nnidx.shape, dtype=np.bool)
        # Check that you are not matching yourself
        qfx2_aid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
        qfx2_notsamechip = qfx2_aid != qaid
        qfx2_valid0 = np.logical_and(qfx2_valid0, qfx2_notsamechip)
        qaid2_valid0[qaid] = (qfx2_score, qfx2_valid0)
    return qaid2_valid0


def assert_qaid2_chipmatch(qreq_, qaid2_chipmatch):
    """ Runs consistency check """
    external_qaids = qreq_.get_external_qaids().tolist()
    external_daids = qreq_.get_external_daids().tolist()

    if len(external_qaids) == 1 and qreq_.qparams.pipeline_root == 'vsone':
        nExternalQVecs = qreq_.ibs.get_annot_vecs(external_qaids[0], qreq_=qreq_).shape[0]
        assert qreq_.indexer.idx2_vec.shape[0] == nExternalQVecs, 'did not index query descriptors properly'

    assert external_qaids == list(qaid2_chipmatch.keys()), 'bad external qaids'
    # Loop over internal qaids
    for qaid, chipmatch in qaid2_chipmatch.iteritems():
        (daid2_fm, daid2_fsv, daid2_fk, daid2_score, daid2_H) = chipmatch
        assert len(daid2_fm) == len(daid2_fsv), 'bad chipmatch'
        assert len(daid2_fm) == len(daid2_fk), 'bad chipmatch'
        assert daid2_H is None or len(daid2_fm) == len(daid2_H), 'bad chipmatch'
        daid_list = list(daid2_fm.keys())
        nFeats1 = qreq_.ibs.get_annot_num_feats(qaid, qreq_=qreq_)
        nFeats2_list = np.array(qreq_.ibs.get_annot_num_feats(daid_list, qreq_=qreq_))
        try:
            assert ut.list_issubset(daid_list, external_daids), 'chipmatch must be subset of daids'
        except AssertionError as ex:
            ut.printex(ex, keys=['daid_list', 'external_daids'])
            raise
        for daid in daid_list:
            print('qaid=%r, daid=%r' % (qaid, daid,))
            fm = daid2_fm[daid]
            fsv = daid2_fsv[daid]
            fk = daid2_fk[daid]
            ut.assert_eq(len(fm), len(fsv), 'bad len', 'len(fm)', 'len(fsv)')
            ut.assert_eq(len(fm), len(fk), 'bad len', 'len(fm)', 'len(fk)')
        try:
            fm_list = ut.dict_take(daid2_fm, daid_list)
            fx2s_list = [fm_.T[1] for fm_ in fm_list]
            fx1s_list = [fm_.T[0] for fm_ in fm_list]
            max_fx1_list = np.array([fx1s.max() for fx1s in fx1s_list])
            max_fx2_list = np.array([fx2s.max() for fx2s in fx2s_list])
            ut.assert_lessthan(max_fx2_list, nFeats2_list, 'max feat index must be less than num feats')
            ut.assert_lessthan(max_fx1_list, nFeats1, 'max feat index must be less than num feats')
        except AssertionError as ex:
            ut.printex(ex, keys=[
                'qaid',
                'daid_list',
                'nFeats1',
                'nFeats2_list',
                'max_fx1_list',
                'max_fx2_list', ])
            raise

        #ut.assert_lists_eq(external_daids, daid_list)


#def


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots._pipline_helpers
        python -m ibeis.model.hots._pipline_helpers --allexamples
        python -m ibeis.model.hots._pipline_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
