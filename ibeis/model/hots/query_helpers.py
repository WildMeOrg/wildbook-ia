# TODO: ADD COPYRIGHT TAG
# TODO: Restructure
from __future__ import absolute_import, division, print_function
import numpy as np
import utool
from itertools import izip
print, print_, printDBG, rrr, profile = utool.inject( __name__, '[query_helpers]')


def get_annotionfeat_nn_index(ibs, qaid, qfx):
    from . import match_chips3 as mc3
    ibs._init_query_requestor()
    qreq = mc3.quickly_ensure_qreq(ibs, [qaid])
    qfx2_desc = ibs.get_annotion_desc(qaid)[qfx:(qfx + 1)]
    (qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm) = qreq.data_index.nn_index2(qreq, qfx2_desc)
    return qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm


def get_query_components(ibs, qaids):
    from . import matching_functions as mf
    from . import match_chips3 as mc3
    ibs._init_query_requestor()
    qreq = ibs.qreq
    #print(ibs.get_infostr())
    daids = ibs.get_recognition_database_aids()
    qaid = qaids[0]
    assert len(daids) > 0, '!!! nothing to search'
    assert len(qaids) > 0, '!!! nothing to query'
    qreq = mc3.prep_query_request(qreq=qreq, qaids=qaids, daids=daids)
    mc3.pre_exec_checks(ibs, qreq)
    qaid2_nns = mf.nearest_neighbors(ibs, qaids, qreq)
    #---
    filt2_weights, filt2_meta = mf.weight_neighbors(ibs, qaid2_nns, qreq)
    #---
    qaid2_nnfilt = mf.filter_neighbors(ibs, qaid2_nns, filt2_weights, qreq)
    #---
    qaid2_chipmatch_FILT = mf.build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq)
    #---
    _tup = mf.spatial_verification(ibs, qaid2_chipmatch_FILT, qreq, dbginfo=True)
    qaid2_chipmatch_SVER, qaid2_svtups = _tup
    #---
    qaid2_qres = mf.chipmatch_to_resdict(ibs, qaid2_chipmatch_SVER, filt2_meta, qreq)
    #####################
    # Testing components
    #####################
    with utool.Indenter('[components]'):
        qfx2_ax, qfx2_dist = qaid2_nns[qaid]
        qfx2_aid = qreq.data_index.ax2_aid[qfx2_ax]
        qfx2_fx  = qreq.data_index.ax2_fx[qfx2_ax]
        qfx2_gid = ibs.get_annotion_gids(qfx2_aid)
        qfx2_nid = ibs.get_annotion_nids(qfx2_aid)
        qfx2_score, qfx2_valid = qaid2_nnfilt[qaid]
        qaid2_nnfilt_ORIG    = mf.identity_filter(qaid2_nns, qreq)
        qaid2_chipmatch_ORIG = mf.build_chipmatches(qaid2_nns, qaid2_nnfilt_ORIG, qreq)
        qaid2_qres_ORIG = mf.chipmatch_to_resdict(ibs, qaid2_chipmatch_ORIG, filt2_meta, qreq)
        qaid2_qres_FILT = mf.chipmatch_to_resdict(ibs, qaid2_chipmatch_FILT, filt2_meta, qreq)
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
    desc_list = ibs.get_annotion_desc(aid_list)
    fid_list = ibs.get_annotion_fids(aid_list)
    desc_list2 = ibs.get_feat_desc(fid_list)

    assert all([np.all(desc1 == desc2) for desc1, desc2 in izip(desc_list, desc_list2)])

    ax2_data = qreq.data_index.ax2_data
    check_sift_desc(ax2_data)
    ax2_aid  = qreq.data_index.ax2_aid
    ax2_fx   = qreq.data_index.ax2_fx

    # For each descriptor create a (aid, fx) pair indicating its
    # chip id and the feature index in that chip id.
    nFeat_list = map(len, desc_list)
    _ax2_aid = [[aid] * nFeat for (aid, nFeat) in izip(aid_list, nFeat_list)]
    _ax2_fx = [range(nFeat) for nFeat in nFeat_list]

    assert len(_ax2_fx) == len(aid_list)
    assert len(_ax2_aid) == len(aid_list)
    print('... loop checks')

    for count in xrange(len(aid_list)):
        aid = aid_list[count]
        assert np.all(np.array(_ax2_aid[count]) == aid)
        assert len(_ax2_fx[count]) == desc_list[count].shape[0]
        ax_list = np.where(ax2_aid == aid)[0]
        np.all(ax2_data[ax_list] == desc_list[count])
        np.all(ax2_fx[ax_list] == np.arange(len(ax_list)))
    print('... seems ok')


def find_matchable_chips(ibs):
    """ quick and dirty test to score by number of assignments """
    from . import match_chips3 as mc3
    from . import matching_functions as mf
    qreq = ibs.qreq
    qaids = ibs.get_valid_aids()
    qreq = mc3.prep_query_request(qreq=qreq, qaids=qaids, daids=qaids)
    mc3.pre_exec_checks(ibs, qreq)
    qaid2_nns = mf.nearest_neighbors(ibs, qaids, qreq)
    mf.rrr()
    qaid2_nnfilt = mf.identity_filter(qaid2_nns, qreq)
    qaid2_chipmatch_FILT = mf.build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq)
    qaid2_ranked_list = {}
    qaid2_ranked_scores = {}
    for qaid, chipmatch in qaid2_chipmatch_FILT.iteritems():
        (aid2_fm, aid2_fs, aid2_fk) = chipmatch
        #aid2_nMatches = {aid: fs.sum() for (aid, fs) in aid2_fs.iteritems()}
        aid2_nMatches = {aid: len(fm) for (aid, fm) in aid2_fs.iteritems()}
        nMatches_list = np.array(aid2_nMatches.values())
        aid_list      = np.array(aid2_nMatches.keys())
        sortx = nMatches_list.argsort()[::-1]
        qaid2_ranked_list[qaid] = aid_list[sortx]
        qaid2_ranked_scores[qaid] = nMatches_list[sortx]

    scores_list = []
    strings_list = []
    for qaid in qaids:
        aid   = qaid2_ranked_list[qaid][0]
        score = qaid2_ranked_scores[qaid][0]
        strings_list.append('qaid=%r, aid=%r, score=%r' % (qaid, aid, score))
        scores_list.append(score)
    sorted_scorestr = np.array(strings_list)[np.array(scores_list).argsort()]
    print('\n'.join(sorted_scorestr))


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
