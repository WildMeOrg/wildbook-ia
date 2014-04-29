# TODO: ADD COPYRIGHT TAG
# TODO: Restructure
from __future__ import absolute_import, division, print_function
import numpy as np
import utool
from itertools import izip
print, print_, printDBG, rrr, profile = utool.inject( __name__, '[query_helpers]')


def get_roifeat_nn_index(ibs, qrid, qfx):
    from . import match_chips3 as mc3
    ibs._init_query_requestor()
    qreq = mc3.quickly_ensure_qreq(ibs, [qrid])
    qfx2_desc = ibs.get_roi_desc(qrid)[qfx:(qfx + 1)]
    (qfx2_rid, qfx2_fx, qfx2_dist, K, Knorm) = qreq.data_index.nn_index2(qreq, qfx2_desc)
    return qfx2_rid, qfx2_fx, qfx2_dist, K, Knorm


def get_query_components(ibs, qrids):
    from . import matching_functions as mf
    from . import match_chips3 as mc3
    ibs._init_query_requestor()
    qreq = ibs.qreq
    #print(ibs.get_infostr())
    drids = ibs.get_recognition_database_rids()
    qrid = qrids[0]
    assert len(drids) > 0, '!!! nothing to search'
    assert len(qrids) > 0, '!!! nothing to query'
    qreq = mc3.prep_query_request(qreq=qreq, qrids=qrids, drids=drids)
    mc3.pre_exec_checks(ibs, qreq)
    qrid2_nns = mf.nearest_neighbors(ibs, qrids, qreq)
    #---
    filt2_weights, filt2_meta = mf.weight_neighbors(ibs, qrid2_nns, qreq)
    #---
    qrid2_nnfilt = mf.filter_neighbors(ibs, qrid2_nns, filt2_weights, qreq)
    #---
    qrid2_chipmatch_FILT = mf.build_chipmatches(qrid2_nns, qrid2_nnfilt, qreq)
    #---
    _tup = mf.spatial_verification(ibs, qrid2_chipmatch_FILT, qreq, dbginfo=True)
    qrid2_chipmatch_SVER, qrid2_svtups = _tup
    #---
    qrid2_res = mf.chipmatch_to_resdict(ibs, qrid2_chipmatch_SVER, filt2_meta, qreq)
    #####################
    # Testing components
    #####################
    with utool.Indenter('[components]'):
        qfx2_ax, qfx2_dist = qrid2_nns[qrid]
        qfx2_rid = qreq.data_index.ax2_rid[qfx2_ax]
        qfx2_fx  = qreq.data_index.ax2_fx[qfx2_ax]
        qfx2_gid = ibs.get_roi_gids(qfx2_rid)
        qfx2_nid = ibs.get_roi_nids(qfx2_rid)
        qfx2_score, qfx2_valid = qrid2_nnfilt[qrid]
        qrid2_nnfilt_ORIG    = mf.identity_filter(qrid2_nns, qreq)
        qrid2_chipmatch_ORIG = mf.build_chipmatches(qrid2_nns, qrid2_nnfilt_ORIG, qreq)
        qrid2_res_ORIG = mf.chipmatch_to_resdict(ibs, qrid2_chipmatch_ORIG, filt2_meta, qreq)
        qrid2_res_FILT = mf.chipmatch_to_resdict(ibs, qrid2_chipmatch_FILT, filt2_meta, qreq)
        qrid2_res_SVER = qrid2_res
    #####################
    # Relevant components
    #####################
    qrid = qrids[0]
    qres_ORIG = qrid2_res_ORIG[qrid]
    qres_FILT = qrid2_res_FILT[qrid]
    qres_SVER = qrid2_res_SVER[qrid]

    return locals()


def data_index_integrity(ibs, qreq):
    print('checking qreq.data_index integrity')

    rid_list = ibs.get_valid_rids()
    desc_list = ibs.get_roi_desc(rid_list)
    fid_list = ibs.get_roi_fids(rid_list)
    desc_list2 = ibs.get_feat_desc(fid_list)

    assert all([np.all(desc1 == desc2) for desc1, desc2 in izip(desc_list, desc_list2)])

    ax2_data = qreq.data_index.ax2_data
    check_sift_desc(ax2_data)
    ax2_rid  = qreq.data_index.ax2_rid
    ax2_fx   = qreq.data_index.ax2_fx

    # For each descriptor create a (rid, fx) pair indicating its
    # chip id and the feature index in that chip id.
    nFeat_list = map(len, desc_list)
    _ax2_rid = [[rid] * nFeat for (rid, nFeat) in izip(rid_list, nFeat_list)]
    _ax2_fx = [range(nFeat) for nFeat in nFeat_list]

    assert len(_ax2_fx) == len(rid_list)
    assert len(_ax2_rid) == len(rid_list)
    print('... loop checks')

    for count in xrange(len(rid_list)):
        rid = rid_list[count]
        assert np.all(np.array(_ax2_rid[count]) == rid)
        assert len(_ax2_fx[count]) == desc_list[count].shape[0]
        ax_list = np.where(ax2_rid == rid)[0]
        np.all(ax2_data[ax_list] == desc_list[count])
        np.all(ax2_fx[ax_list] == np.arange(len(ax_list)))
    print('... seems ok')


def find_matchable_chips(ibs):
    """ quick and dirty test to score by number of assignments """
    from . import match_chips3 as mc3
    from . import matching_functions as mf
    qreq = ibs.qreq
    qrids = ibs.get_valid_rids()
    qreq = mc3.prep_query_request(qreq=qreq, qrids=qrids, drids=qrids)
    mc3.pre_exec_checks(ibs, qreq)
    qrid2_nns = mf.nearest_neighbors(ibs, qrids, qreq)
    mf.rrr()
    qrid2_nnfilt = mf.identity_filter(qrid2_nns, qreq)
    qrid2_chipmatch_FILT = mf.build_chipmatches(qrid2_nns, qrid2_nnfilt, qreq)
    qrid2_ranked_list = {}
    qrid2_ranked_scores = {}
    for qrid, chipmatch in qrid2_chipmatch_FILT.iteritems():
        (rid2_fm, rid2_fs, rid2_fk) = chipmatch
        #rid2_nMatches = {rid: fs.sum() for (rid, fs) in rid2_fs.iteritems()}
        rid2_nMatches = {rid: len(fm) for (rid, fm) in rid2_fs.iteritems()}
        nMatches_list = np.array(rid2_nMatches.values())
        rid_list      = np.array(rid2_nMatches.keys())
        sortx = nMatches_list.argsort()[::-1]
        qrid2_ranked_list[qrid] = rid_list[sortx]
        qrid2_ranked_scores[qrid] = nMatches_list[sortx]

    scores_list = []
    strings_list = []
    for qrid in qrids:
        rid   = qrid2_ranked_list[qrid][0]
        score = qrid2_ranked_scores[qrid][0]
        strings_list.append('qrid=%r, rid=%r, score=%r' % (qrid, rid, score))
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
