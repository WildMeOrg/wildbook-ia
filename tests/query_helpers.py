#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'BUILDQUERY'
#------
import __testing__
import numpy as np
from itertools import izip
import utool
from ibeis.model.hots import QueryRequest  # NOQA
from ibeis.model.hots import NNIndex  # NOQA
from ibeis.model.hots import matching_functions as mf  # NOQA
from ibeis.model.hots import match_chips3 as mc3  # NOQA
from ibeis.model.hots.matching_functions import _apply_filter_scores, progress_func  # NOQA
import test_tools
print, print_, printDBG, rrr, profile = utool.inject( __name__, '[query_helpers]')
printTEST = __testing__.printTEST


def get_query_components(ibs, qcids):
    printTEST('[GET QUERY COMPONENTS]')
    ibs._init_query_requestor()
    qreq = ibs.qreq
    #print(ibs.get_infostr())
    dcids = ibs.get_recognition_database_chips()
    qcid = qcids[0]
    assert len(dcids) > 0, '!!! nothing to search'
    assert len(qcids) > 0, '!!! nothing to query'
    qreq = mc3.prep_query_request(qreq=qreq, qcids=qcids, dcids=dcids)
    mc3.pre_exec_checks(ibs, qreq)
    qcid2_nns = mf.nearest_neighbors(ibs, qcids, qreq)
    #---
    filt2_weights, filt2_meta = mf.weight_neighbors(ibs, qcid2_nns, qreq)
    #---
    qcid2_nnfilt = mf.filter_neighbors(ibs, qcid2_nns, filt2_weights, qreq)
    #---
    qcid2_chipmatch_FILT = mf.build_chipmatches(qcid2_nns, qcid2_nnfilt, qreq)
    #---
    _tup = mf.spatial_verification(ibs, qcid2_chipmatch_FILT, qreq, dbginfo=True)
    qcid2_chipmatch_SVER, qcid2_svtups = _tup
    #---
    qcid2_res = mf.chipmatch_to_resdict(ibs, qcid2_chipmatch_SVER, filt2_meta, qreq)
    #####################
    # Testing components
    #####################
    with utool.Indenter('[components]'):
        qfx2_ax, qfx2_dist = qcid2_nns[qcid]
        qfx2_cid = qreq.data_index.ax2_cid[qfx2_ax]
        qfx2_fx  = qreq.data_index.ax2_fx[qfx2_ax]
        qfx2_gid = ibs.get_chip_gids(qfx2_cid)
        qfx2_nid = ibs.get_chip_nids(qfx2_cid)
        qfx2_score, qfx2_valid = qcid2_nnfilt[qcid]
        qcid2_nnfilt_ORIG    = mf.identity_filter(qcid2_nns, qreq)
        qcid2_chipmatch_ORIG = mf.build_chipmatches(qcid2_nns, qcid2_nnfilt_ORIG, qreq)
        qcid2_res_ORIG = mf.chipmatch_to_resdict(ibs, qcid2_chipmatch_ORIG, filt2_meta, qreq)
        qcid2_res_FILT = mf.chipmatch_to_resdict(ibs, qcid2_chipmatch_FILT, filt2_meta, qreq)
        qcid2_res_SVER = qcid2_res
    #####################
    # Relevant components
    #####################
    qcid = qcids[0]
    qres_ORIG = qcid2_res_ORIG[qcid]
    qres_FILT = qcid2_res_FILT[qcid]
    qres_SVER = qcid2_res_SVER[qcid]

    return locals()


def data_index_integrity(ibs, qreq):
    print('checking qreq.data_index integrity')

    cid_list = ibs.get_valid_cids()
    desc_list = ibs.get_chip_desc(cid_list)
    fid_list = ibs.get_chip_fids(cid_list)
    desc_list2 = ibs.get_feat_desc(fid_list)

    assert all([np.all(desc1 == desc2) for desc1, desc2 in izip(desc_list, desc_list2)])

    ax2_data = qreq.data_index.ax2_data
    test_tools.check_sift_desc(ax2_data)
    ax2_cid  = qreq.data_index.ax2_cid
    ax2_fx   = qreq.data_index.ax2_fx

    # For each descriptor create a (cid, fx) pair indicating its
    # chip id and the feature index in that chip id.
    nFeat_list = map(len, desc_list)
    _ax2_cid = [[cid] * nFeat for (cid, nFeat) in izip(cid_list, nFeat_list)]
    _ax2_fx = [range(nFeat) for nFeat in nFeat_list]

    assert len(_ax2_fx) == len(cid_list)
    assert len(_ax2_cid) == len(cid_list)
    print('... loop checks')

    for count in xrange(len(cid_list)):
        cid = cid_list[count]
        assert np.all(np.array(_ax2_cid[count]) == cid)
        assert len(_ax2_fx[count]) == desc_list[count].shape[0]
        ax_list = np.where(ax2_cid == cid)[0]
        np.all(ax2_data[ax_list] == desc_list[count])
        np.all(ax2_fx[ax_list] == np.arange(len(ax_list)))
    print('... seems ok')


def find_matchable_chips(ibs):
    """ quick and dirty test to score by number of assignments """
    qreq = ibs.qreq
    qcids = ibs.get_valid_cids()
    qreq = mc3.prep_query_request(qreq=qreq, qcids=qcids, dcids=qcids)
    mc3.pre_exec_checks(ibs, qreq)
    qcid2_nns = mf.nearest_neighbors(ibs, qcids, qreq)
    mf.rrr()
    qcid2_nnfilt = mf.identity_filter(qcid2_nns, qreq)
    qcid2_chipmatch_FILT = mf.build_chipmatches(qcid2_nns, qcid2_nnfilt, qreq)
    qcid2_ranked_list = {}
    qcid2_ranked_scores = {}
    for qcid, chipmatch in qcid2_chipmatch_FILT.iteritems():
        (cid2_fm, cid2_fs, cid2_fk) = chipmatch
        #cid2_nMatches = {cid: fs.sum() for (cid, fs) in cid2_fs.iteritems()}
        cid2_nMatches = {cid: len(fm) for (cid, fm) in cid2_fs.iteritems()}
        nMatches_list = np.array(cid2_nMatches.values())
        cid_list      = np.array(cid2_nMatches.keys())
        sortx = nMatches_list.argsort()[::-1]
        qcid2_ranked_list[qcid] = cid_list[sortx]
        qcid2_ranked_scores[qcid] = nMatches_list[sortx]

    scores_list = []
    strings_list = []
    for qcid in qcids:
        cid   = qcid2_ranked_list[qcid][0]
        score = qcid2_ranked_scores[qcid][0]
        strings_list.append('qcid=%r, cid=%r, score=%r' % (qcid, cid, score))
        scores_list.append(score)
    sorted_scorestr = np.array(strings_list)[np.array(scores_list).argsort()]
    print('\n'.join(sorted_scorestr))
