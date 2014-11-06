from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import six
from six.moves import zip
from ibeis import ibsfuncs
#from ibeis.dev import results_organizer
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[resorg]', DEBUG=False)


def get_feat_matches(allres, qaid, aid):
    try:
        qres = allres.qaid2_qres[qaid]
        fm = qres.aid2_fm[aid]
    except KeyError:
        print('Failed qaid=%r, aid=%r' % (qaid, aid))
        raise
    return fm


def print_desc_distances_map(orgres2_distmap):
    print('+-----------------------------')
    print('| DESCRIPTOR MATCHE DISTANCES:')
    for orgtype, distmap in six.iteritems(orgres2_distmap):
        print('| orgtype(%r)' % (orgtype,))
        for disttype, dists in six.iteritems(distmap):
            print('|     disttype(%12r): %s' % (disttype, utool.get_stats_str(dists)))
    print('L-----------------------------')


def print_annotationmatch_scores_map(orgres2_scores):
    print('+-----------------------------')
    print('| CHIPMATCH SCORES:')
    for orgtype, scores in six.iteritems(orgres2_scores):
        print('| orgtype(%r)' % (orgtype,))
        print('|     scores: %s' % (utool.get_stats_str(scores)))
    print('L-----------------------------')


def get_orgres_annotationmatch_scores(allres, orgtype_list=['false', 'true']):
    orgres2_scores = {}
    for orgtype in orgtype_list:
        printDBG('[rr2] getting orgtype=%r distances between sifts' % orgtype)
        orgres = allres.get_orgtype(orgtype)
        ranks  = orgres.ranks
        scores = orgres.scores
        valid_scores = scores[ranks >= 0]  # None is less than 0
        orgres2_scores[orgtype] = valid_scores
    return orgres2_scores


def get_orgres_desc_match_dists(allres, orgtype_list=['false', 'true']):
    """
    computes distances between matching descriptors of orgtypes in allres

    Args:
        allres (?):
        orgtype_list (list):

    Returns:
        dict: orgres2_descmatch_dists mapping from orgtype to distances (ndarrays)

    Example:
        >>> from ibeis.dev.results_analyzer import *  # NOQA
        >>> allres = '?'
        >>> orgtype_list = ['false', 'true']
        >>> orgres2_descmatch_dists = get_orgres_desc_match_dists(allres, orgtype_list)
        >>> print(orgres2_descmatch_dists)
    """
    orgres2_descmatch_dists = {}
    for orgtype in orgtype_list:
        printDBG('[rr2] getting orgtype=%r distances between sifts' % orgtype)
        orgres = allres.get_orgtype(orgtype)
        qaids = orgres.qaids
        aids  = orgres.aids
        try:
            stacked_qvecs, stacked_dvecs = get_matching_descriptors(allres, qaids, aids)
        except Exception as ex:
            orgres.printme3()
            utool.printex(ex)
            raise
        printDBG('[rr2]  * stacked_qvecs.shape = %r' % (stacked_qvecs.shape,))
        printDBG('[rr2]  * stacked_dvecs.shape = %r' % (stacked_dvecs.shape,))
        #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
        #dist_list = ['L1', 'L2', 'hist_isect']
        dist_list = ['L2', 'hist_isect']
        hist1 = np.asarray(stacked_qvecs, dtype=np.float32)
        hist2 = np.asarray(stacked_dvecs, dtype=np.float32)
        distances = utool.compute_distances(hist1, hist2, dist_list)
        orgres2_descmatch_dists[orgtype] = distances
    return orgres2_descmatch_dists


def get_matching_descriptors(allres, qaid_list, daid_list):
    """
    returns a set of matching descriptors from queries to database annotations
    in allres

    Args:
        allres (AllResults): all results object
        qaid_list (list): query annotation id list
        daid_list (list): database annotation id list

    Returns:
        tuple: (stacked_qvecs, stacked_dvecs)

    Example:
        >>> from ibeis.dev.results_analyzer import *  # NOQA
        >>> allres = '?'
        >>> qaid_list = '?'
        >>> daid_list = '?'
        >>> (stacked_qvecs, stacked_dvecs) = get_matching_descriptors(allres, qaid_list, daid_list)
        >>> print((stacked_qvecs, stacked_dvecs))
    """
    ibs = allres.ibs
    qvecs_cache = ibsfuncs.get_annot_vecs_cache(ibs, qaid_list)
    dvecs_cache = ibsfuncs.get_annot_vecs_cache(ibs, daid_list)
    qvecs_list = []
    dvecs_list = []
    for qaid, daid in zip(qaid_list, daid_list):
        try:
            fm = get_feat_matches(allres, qaid, daid)
            if len(fm) == 0:
                continue
        except KeyError:
            continue
        qvecs_m = qvecs_cache[qaid][fm.T[0]]
        dvecs_m = dvecs_cache[daid][fm.T[1]]
        qvecs_list.append(qvecs_m)
        dvecs_list.append(dvecs_m)
    try:
        stacked_qvecs = np.vstack(qvecs_list)
        stacked_dvecs = np.vstack(dvecs_list)
    except Exception as ex:
        utool.printex(ex, '[!!!] get_matching_descriptors', keys=['qaid_list', 'daid_list', 'qvecs_list', 'dvecs_list' 'stacked_qvecs', 'stacked_dvecs'])
        raise
    return stacked_qvecs, stacked_dvecs


#def get_score_stuff_pdfish(allres):
#    """ In development """
#    true_orgres  = allres.get_orgtype('true')
#    false_orgres = allres.get_orgtype('false')
#    orgres = true_orgres
#    orgres = false_orgres
#    def get_interesting_annotationpairs(orgres):
#        orgres2 = results_organizer._score_sorted_ranks_lt(orgres, 2)
