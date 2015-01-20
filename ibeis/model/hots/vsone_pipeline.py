"""
special pipeline for vsone specific functions
"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import vtool as vt
from ibeis.model.hots import neighbor_index
from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
#import pyflann
#from ibeis.model.hots import coverage_image
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
import utool as ut
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[vsonepipe]', DEBUG=False)


def vsone_reranking(qreq_, qaid2_chipmatch):
    """

    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
    """
    vsone_query_pairs = make_vsone_rerank_pairs(qreq_, qaid2_chipmatch)  # NOQA


def make_vsone_rerank_pairs(qreq_, qaid2_chipmatch):
    """
    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> vsone_query_pairs = make_vsone_rerank_pairs(qreq_, qaid2_chipmatch)
        >>> qaid, top_aid_list, top_H_list = vsone_query_pairs[0]
        >>> top_nid_list = ibs.get_annot_name_rowids(top_aid_list)
        >>> assert top_nid_list.index(1) == 0, 'name 1 should be rank 1'
        >>> assert len(top_nid_list) == 5, 'should have 3 names and up to 2 image per name'
    """
    from ibeis.model.hots import pipeline
    vsone_query_pairs = []
    score_method = qreq_.qparams.score_method
    # TODO: paramaterize
    nNameShortlistVsone = 3
    nAnnotPerName = 2
    assert score_method == 'nsum'
    print('vsone reranking. ')
    for qaid, chipmatch in six.iteritems(qaid2_chipmatch):
        daid2_prescore = pipeline.score_chipmatch(qreq_, qaid, chipmatch, score_method)
        daid_list = np.array(daid2_prescore.keys())
        prescore_arr = np.array(daid2_prescore.values())
        nscore_tup = name_scoring.get_one_score_per_name(qreq_.ibs, daid_list, prescore_arr)
        (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        top_aids_list = ut.listclip(sorted_aids, nNameShortlistVsone)
        top_aids_list_ = [ut.listclip(aids, nAnnotPerName) for aids in top_aids_list]
        top_aid_list = ut.flatten(top_aids_list_)
        top_H_list = ut.dict_take(chipmatch.aid2_H, top_aid_list)
        vsone_pair_tup = (qaid, top_aid_list, top_H_list)
        vsone_query_pairs.append(vsone_pair_tup)
    return vsone_query_pairs


def execute_vsone_reranking(qreq_, vsone_query_pairs):
    ibs = qreq_.ibs
    checks = 800
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    match_xy_thresh = .01
    for vsone_pair_tup in vsone_query_pairs:
        (qaid, daid_list, H_list) = vsone_pair_tup
        qvecs = ibs.get_annot_vecs(qaid)
        qkpts = ibs.get_annot_kpts(qaid)
        dvecs_list = ibs.get_annot_vecs(daid_list)
        dkpts_list = ibs.get_annot_kpts(daid_list)
        dfgws_list = ibs.get_annot_fgweights(daid_list)
        dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlen_sqrd(daid_list)
        dinfo_list = zip(dvecs_list, dkpts_list, dfgws_list, dlen_sqrd_list, H_list)

        flann_cachedir = ibs.get_flann_cachedir()
        flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params)

        #preptup = neighbor_index.prepare_index_data(daid_list, dvecs_list, dfgws_list, True)
        #(ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = preptup

        for daid, dinfo in zip(daid_list, dinfo_list):
            K = 7
            dvecs, dkpts, dfgws, dlen_sqrd, H = dinfo
            spatially_constrained_match(flann, dvecs, qkpts, dkpts, H, dlen_sqrd, match_xy_thresh, K)


def spatially_constrained_match(flann, dvecs, qkpts, dkpts, H, dlen_sqrd, match_xy_thresh, K):
    from vtool import constrained_matching
    dfx2_qfx, _dfx2_dist = flann.nn_index(dvecs, num_neighbors=K, checks=800)
    dfx2_dist = np.divide(_dfx2_dist, hstypes.VEC_PSEUDO_MAX_DISTANCE_SQRD)
    constraintup = constrained_matching.spatially_constrain_matches(dlen_sqrd, qkpts, dkpts, H, dfx2_qfx, dfx2_dist, match_xy_thresh)
    (fm_constrained, fm_norm_constrained, match_dist_list, norm_dist_list) = constraintup


def augment_distinctiveness(qreq_):
    pass


def constrained_vsone_matching(qreq_):
    pass


def coverage_scoring(qreq_):
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline
        python -m ibeis.model.hots.vsone_pipeline --allexamples
        python -m ibeis.model.hots.vsone_pipeline --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
