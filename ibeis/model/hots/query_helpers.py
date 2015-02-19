# TODO: ADD COPYRIGHT TAG
# TODO: Restructure
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
from six.moves import zip, range, map
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[query_helpers]')


def get_query_components(ibs, qaids):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (?):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.model.hots.query_helpers --test-get_query_components

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.query_helpers import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaids = ibs.get_valid_aids()
        >>> # execute function
        >>> result = get_query_components(ibs, qaids)
        >>> # verify results
        >>> print(result)
    """
    from ibeis.model.hots import pipeline
    from ibeis.model.hots import query_request
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


#def find_matchable_chips(ibs):
#    """ quick and dirty test to score by number of assignments """
#    import six
#    from . import match_chips3 as mc3
#    from . import matching_functions as pipeline
#    qreq = ibs.qreq
#    qaids = ibs.get_valid_aids()
#    qreq = mc3.prep_query_request(qreq=qreq, qaids=qaids, daids=qaids)
#    mc3.pre_exec_checks(ibs, qreq)
#    qaid2_nns = pipeline.nearest_neighbors(ibs, qaids, qreq)
#    pipeline.rrr()
#    qaid2_nnfilt = pipeline.identity_filter(qaid2_nns, qreq)
#    qaid2_chipmatch_FILT = pipeline.build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq)
#    qaid2_ranked_list = {}
#    qaid2_ranked_scores = {}
#    for qaid, cmtup_old in six.iteritems(qaid2_chipmatch_FILT):
#        (aid2_fm, aid2_fs, aid2_fk) = cmtup_old
#        #aid2_nMatches = {aid: fs.sum() for (aid, fs) in six.iteritems(aid2_fs)}
#        aid2_nMatches = {aid: len(fm) for (aid, fm) in six.iteritems(aid2_fs)}
#        nMatches_list = np.array(aid2_nMatches.values())
#        aid_list      = np.array(aid2_nMatches.keys())
#        sortx = nMatches_list.argsort()[::-1]
#        qaid2_ranked_list[qaid] = aid_list[sortx]
#        qaid2_ranked_scores[qaid] = nMatches_list[sortx]

#    scores_list = []
#    strings_list = []
#    for qaid in qaids:
#        aid   = qaid2_ranked_list[qaid][0]
#        score = qaid2_ranked_scores[qaid][0]
#        strings_list.append('qaid=%r, aid=%r, score=%r' % (qaid, aid, score))
#        scores_list.append(score)
#    sorted_scorestr = np.array(strings_list)[np.array(scores_list).argsort()]
#    print('\n'.join(sorted_scorestr))


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
