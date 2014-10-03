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
    orgres2_descmatch_dists = {}
    for orgtype in orgtype_list:
        printDBG('[rr2] getting orgtype=%r distances between sifts' % orgtype)
        orgres = allres.get_orgtype(orgtype)
        qaids = orgres.qaids
        aids  = orgres.aids
        try:
            adesc1, adesc2 = get_matching_descriptors(allres, qaids, aids)
        except Exception:
            orgres.printme3()
            raise
        printDBG('[rr2]  * adesc1.shape = %r' % (adesc1.shape,))
        printDBG('[rr2]  * adesc2.shape = %r' % (adesc2.shape,))
        #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
        #dist_list = ['L1', 'L2', 'hist_isect']
        dist_list = ['L2', 'hist_isect']
        hist1 = np.asarray(adesc1, dtype=np.float64)
        hist2 = np.asarray(adesc2, dtype=np.float64)
        distances = utool.compute_distances(hist1, hist2, dist_list)
        orgres2_descmatch_dists[orgtype] = distances
    return orgres2_descmatch_dists


def get_matching_descriptors(allres, qaids, aids):
    ibs = allres.ibs
    qdesc_cache = ibsfuncs.get_annot_desc_cache(ibs, qaids)
    rdesc_cache = ibsfuncs.get_annot_desc_cache(ibs, aids)
    desc1_list = []
    desc2_list = []
    for qaid, aid in zip(qaids, aids):
        try:
            fm = get_feat_matches(allres, qaid, aid)
            if len(fm) == 0:
                continue
        except KeyError:
            continue
        desc1_m = qdesc_cache[qaid][fm.T[0]]
        desc2_m = rdesc_cache[aid][fm.T[1]]
        desc1_list.append(desc1_m)
        desc2_list.append(desc2_m)
    try:
        aggdesc1 = np.vstack(desc1_list)
        aggdesc2 = np.vstack(desc2_list)
    except Exception as ex:
        utool.printex(ex, '[!!!!] get_matching_descriptors', key_list=['qaids', 'aids', 'desc1_list', 'desc2_list' 'aggdesc1', 'aggdesc2'])
        raise
    return aggdesc1, aggdesc2


#def get_score_stuff_pdfish(allres):
#    """ In development """
#    true_orgres  = allres.get_orgtype('true')
#    false_orgres = allres.get_orgtype('false')
#    orgres = true_orgres
#    orgres = false_orgres
#    def get_interesting_annotationpairs(orgres):
#        orgres2 = results_organizer._score_sorted_ranks_lt(orgres, 2)
