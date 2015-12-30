# -*- coding: utf-8 -*-
# TODO: ADD COPYRIGHT TAG
# TODO: Restructure
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
from six.moves import zip, range, map
print, rrr, profile = ut.inject2(__name__, '[query_helpers]')


def get_query_components(ibs, qaids):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (?):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.algo.hots.query_helpers --test-get_query_components

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.query_helpers import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaids = ibs.get_valid_aids()
        >>> # execute function
        >>> result = get_query_components(ibs, qaids)
        >>> # verify results
        >>> print(result)
    """
    from ibeis.algo.hots import pipeline
    from ibeis.algo.hots import query_request
    daids = ibs.get_valid_aids()
    cfgdict = dict(with_metadata=True)
    qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids, cfgdict)
    qaid = qaids[0]
    assert len(daids) > 0, '!!! nothing to search'
    assert len(qaids) > 0, '!!! nothing to query'
    qreq_.lazy_load()
    pipeline_locals_ = pipeline.testrun_pipeline_upto(qreq_, None)
    qaid2_nns            = pipeline_locals_['qaid2_nns']
    qaid2_nnvalid0       = pipeline_locals_['qaid2_nnvalid0']
    qaid2_filtweights    = pipeline_locals_['qaid2_filtweights']
    qaid2_nnfilts        = pipeline_locals_['qaid2_nnfilts']
    qaid2_chipmatch_FILT = pipeline_locals_['qaid2_chipmatch_FILT']
    qaid2_chipmatch_SVER = pipeline_locals_['qaid2_chipmatch_SVER']
    qaid2_svtups = qreq_.metadata['qaid2_svtups']
    #---
    qaid2_qres = pipeline.chipmatch_to_resdict(qreq_, qaid2_chipmatch_SVER)
    #####################
    # Testing components
    #####################
    with ut.Indenter('[components]'):
        qfx2_idx, qfx2_dist = qaid2_nns[qaid]
        qfx2_aid = qreq_.indexer.get_nn_aids(qfx2_idx)
        qfx2_fx  = qreq_.indexer.get_nn_featxs(qfx2_idx)
        qfx2_gid = ibs.get_annot_gids(qfx2_aid)  # NOQA
        qfx2_nid = ibs.get_annot_name_rowids(qfx2_aid)  # NOQA
        filtkey_list, qfx2_scores, qfx2_valids = qaid2_nnfilts[qaid]
        qaid2_nnfilt_ORIG    = pipeline.identity_filter(qreq_, qaid2_nns)
        qaid2_chipmatch_ORIG = pipeline.build_chipmatches(qreq_, qaid2_nns, qaid2_nnfilt_ORIG)
        qaid2_qres_ORIG = pipeline.chipmatch_to_resdict(qaid2_chipmatch_ORIG, qreq_)
        qaid2_qres_FILT = pipeline.chipmatch_to_resdict(qaid2_chipmatch_FILT, qreq_)
        qaid2_qres_SVER = qaid2_qres
    #####################
    # Relevant components
    #####################
    qaid = qaids[0]
    qres_ORIG = qaid2_qres_ORIG[qaid]
    qres_FILT = qaid2_qres_FILT[qaid]
    qres_SVER = qaid2_qres_SVER[qaid]

    return locals()


def data_index_integrity(ibs, qreq):
    print('checking qreq.data_index integrity')

    aid_list = ibs.get_valid_aids()
    desc_list = ibs.get_annot_vecs(aid_list)
    fid_list = ibs.get_annot_feat_rowids(aid_list)
    desc_list2 = ibs.get_feat_vecs(fid_list)

    assert all([np.all(desc1 == desc2) for desc1, desc2 in zip(desc_list, desc_list2)])

    dx2_data = qreq.data_index.dx2_data
    check_sift_desc(dx2_data)
    dx2_aid  = qreq.data_index.dx2_aid
    dx2_fx   = qreq.data_index.dx2_fx

    # For each descriptor create a (aid, fx) pair indicating its
    # chip id and the feature index in that chip id.
    nFeat_list = list(map(len, desc_list))
    _dx2_aid = [[aid] * nFeat for (aid, nFeat) in zip(aid_list, nFeat_list)]
    _dx2_fx = [list(range(nFeat)) for nFeat in nFeat_list]

    assert len(_dx2_fx) == len(aid_list)
    assert len(_dx2_aid) == len(aid_list)
    print('... loop checks')

    for count in range(len(aid_list)):
        aid = aid_list[count]
        assert np.all(np.array(_dx2_aid[count]) == aid)
        assert len(_dx2_fx[count]) == desc_list[count].shape[0]
        dx_list = np.where(dx2_aid == aid)[0]
        np.all(dx2_data[dx_list] == desc_list[count])
        np.all(dx2_fx[dx_list] == np.arange(len(dx_list)))
    print('... seems ok')


def check_sift_desc(desc):
    varname = 'desc'
    verbose = True
    if verbose:
        print('%s.shape=%r' % (varname, desc.shape))
        print('%s.dtype=%r' % (varname, desc.dtype))

    assert desc.shape[1] == 128
    assert desc.dtype == np.uint8
    # Checks to make sure descriptors are close to valid SIFT descriptors.
    # There will be error because of uint8
    target = 1.0  # this should be 1.0
    bindepth = 256.0
    L2_list = np.sqrt(((desc / bindepth) ** 2).sum(1)) / 2.0  # why?
    err = (target - L2_list) ** 2
    thresh = 1 / 256.0
    invalids = err >= thresh
    if np.any(invalids):
        print('There are %d/%d problem SIFT descriptors' % (invalids.sum(), len(invalids)))
        L2_range = L2_list.max() - L2_list.min()
        indexes = np.where(invalids)[0]
        print('L2_range = %r' % (L2_range,))
        print('thresh = %r' % thresh)
        print('L2_list.mean() = %r' % L2_list.mean())
        print('at indexes: %r' % indexes)
        print('with errors: %r' % err[indexes])
    else:
        print('There are %d OK SIFT descriptors' % (len(desc),))
    return invalids
