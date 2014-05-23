from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[vr2]', DEBUG=False)
# Python
from itertools import izip
# Scientific
import numpy as np
from numpy.linalg import svd
#from numba import autojit
# HotSpotter


def score_chipmatch_csum(chipmatch):
    (_, rid2_fs, _) = chipmatch
    rid2_score = {rid: np.sum(fs) for (rid, fs) in rid2_fs.iteritems()}
    return rid2_score


def score_chipmatch_nsum(ibs, qrid, chipmatch, qreq):
    raise NotImplementedError('nsum')


def score_chipmatch_nunique(ibs, qrid, chipmatch, qreq):
    raise NotImplementedError('nunique')


def enforce_one_name(ibs, rid2_score, chipmatch=None, rid2_chipscore=None):
    """ this is a hack to make the same name only show up once in the top ranked
        list
    """
    if chipmatch is not None:
        (_, rid2_fs, _) = chipmatch
        rid2_chipscore = np.array([np.sum(fs) for fs in rid2_fs])
    # FIXME
    nid2_cxs = ibs.get_nx2_cxs()
    rid2_score = np.array(rid2_score)
    for nid, rids in enumerate(nid2_cxs):
        if len(rids) < 2 or nid <= 1:
            continue
        #print(rids)
        # zero the rids with the lowest csum score
        sortx = rid2_chipscore[rids].argsort()
        rids_to_zero = np.array(rids)[sortx[0:-1]]
        rid2_score[rids_to_zero] = 0
    return rid2_score


def score_chipmatch_pos(ibs, qcx, chipmatch, qreq, rule='borda'):
    (rid2_fm, rid2_fs, rid2_fk) = chipmatch
    K = qreq.cfg.nn_cfg.K
    isWeighted = qreq.cfg.agg_cfg.isWeighted
    # Create voting vectors of top K utilities
    qfx2_utilities = _chipmatch2_utilities(ibs, qcx, chipmatch, K)
    # Run Positional Scoring Rule
    altx2_score, altx2_tnx = positional_scoring_rule(qfx2_utilities, rule, isWeighted)
    # Map alternatives back to chips/names
    rid2_score, nid2_score = get_scores_from_altx2_score(ibs, qcx, altx2_score, altx2_tnx)
    # HACK HACK HACK!!!
    #rid2_score = enforce_one_name_per_cscore(ibs, rid2_score, chipmatch)
    return rid2_score, nid2_score


# chipmatch = qcx2_chipmatch[qcx]
def score_chipmatch_PL(ibs, qcx, chipmatch, qreq):
    K = qreq.cfg.nn_cfg.K
    max_alts = qreq.cfg.agg_cfg.max_alts
    isWeighted = qreq.cfg.agg_cfg.isWeighted
    # Create voting vectors of top K utilities
    qfx2_utilities = _chipmatch2_utilities(ibs, qcx, chipmatch, K)
    qfx2_utilities = _filter_utilities(qfx2_utilities, max_alts)
    # Run Placket Luce Model
    # 1) create placket luce matrix pairwise matrix
    if isWeighted:
        PL_matrix, altx2_tnx = _utilities2_weighted_pairwise_breaking(qfx2_utilities)
    else:
        PL_matrix, altx2_tnx = _utilities2_pairwise_breaking(qfx2_utilities)
    # 2) find the gamma vector which minimizes || Pl * gamma || s.t. gamma > 0
    gamma = _optimize(PL_matrix)
    # Find the probability each alternative is #1
    altx2_prob = _PL_score(gamma)
    #print('[vote] gamma = %r' % gamma)
    #print('[vote] altx2_prob = %r' % altx2_prob)
    # Use probabilities as scores
    rid2_score, nid2_score = get_scores_from_altx2_score(ibs, qcx, altx2_prob, altx2_tnx)
    # HACK HACK HACK!!!
    #rid2_score = enforce_one_name_per_cscore(ibs, rid2_score, chipmatch)
    return rid2_score, nid2_score


def _optimize(M):
    #print('[vote] optimize')
    if M.size == 0:
        return np.array([])
    (u, s, v) = svd(M)
    x = np.abs(v[-1])
    check = np.abs(M.dot(x)) < 1E-9
    if not all(check):
        raise Exception('SVD method failed miserably')
    return x


def _PL_score(gamma):
    #print('[vote] computing probabilities')
    nAlts = len(gamma)
    altx2_prob = np.zeros(nAlts)
    for ax in xrange(nAlts):
        altx2_prob[ax] = gamma[ax] / np.sum(gamma)
    #print('[vote] altx2_prob: '+str(altx2_prob))
    #print('[vote] sum(prob): '+str(sum(altx2_prob)))
    return altx2_prob


def get_scores_from_altx2_score(ibs, qcx, altx2_prob, altx2_tnx):
    nid2_score = np.zeros(len(ibs.tables.nid2_name))
    rid2_score = np.zeros(len(ibs.tables.rid2_rid))
    nid2_cxs = ibs.get_nx2_cxs()
    for altx, prob in enumerate(altx2_prob):
        tnx = altx2_tnx[altx]
        if tnx < 0:  # account for temporary names
            rid2_score[-tnx] = prob
            nid2_score[1] += prob
        else:
            nid2_score[tnx] = prob
            for rid in nid2_cxs[tnx]:
                if rid == qcx:
                    continue
                rid2_score[rid] = prob
    return rid2_score, nid2_score


def _chipmatch2_utilities(ibs, qcx, chipmatch, K):
    """
    returns qfx2_utilities
    fx1 : [(rid_0, tnx_0, fs_0, fk_0), ..., (rid_m, tnx_m, fs_m, fk_m)]
    fx2 : [(rid_0, tnx_0, fs_0, fk_0), ..., (rid_m, tnx_m, fs_m, fk_m)]
                    ...
    fxN : [(rid_0, tnx_0, fs_0, fk_0), ..., (rid_m, tnx_m, fs_m, fk_m)]
    """
    #print('[vote] computing utilities')
    rid2_nx = ibs.tables.rid2_nx
    nQFeats = len(ibs.feats.rid2_kpts[qcx])
    # Stack the feature matches
    (rid2_fm, rid2_fs, rid2_fk) = chipmatch
    rids = np.hstack([[rid] * len(rid2_fm[rid]) for rid in xrange(len(rid2_fm))])
    rids = np.array(rids, np.int)
    fms = np.vstack(rid2_fm)
    # Get the individual feature match lists
    qfxs = fms[:, 0]
    fss  = np.hstack(rid2_fs)
    fks  = np.hstack(rid2_fk)
    qfx2_utilities = [[] for _ in xrange(nQFeats)]
    for rid, qfx, fk, fs in izip(rids, qfxs, fks, fss):
        nid = rid2_nx[rid]
        # Apply temporary uniquish name
        tnx = nid if nid >= 2 else -rid
        utility = (rid, tnx, fs, fk)
        qfx2_utilities[qfx].append(utility)
    for qfx in xrange(len(qfx2_utilities)):
        utilities = qfx2_utilities[qfx]
        utilities = sorted(utilities, key=lambda tup: tup[3])
        qfx2_utilities[qfx] = utilities
    return qfx2_utilities


def _filter_utilities(qfx2_utilities, max_alts=200):
    print('[vote] filtering utilities')
    tnxs = [utool[1] for utils in qfx2_utilities for utool in utils]
    if len(tnxs) == 0:
        return qfx2_utilities
    tnxs = np.array(tnxs)
    tnxs_min = tnxs.min()
    tnx2_freq = np.bincount(tnxs - tnxs_min)
    nAlts = (tnx2_freq > 0).sum()
    nRemove = max(0, nAlts - max_alts)
    print(' * removing %r/%r alternatives' % (nRemove, nAlts))
    if nRemove > 0:  # remove least frequent names
        most_freq_tnxs = tnx2_freq.argsort()[::-1] + tnxs_min
        keep_tnxs = set(most_freq_tnxs[0:max_alts].tolist())
        for qfx in xrange(len(qfx2_utilities)):
            utils = qfx2_utilities[qfx]
            qfx2_utilities[qfx] = [utool for utool in utils if utool[1] in keep_tnxs]
    return qfx2_utilities


def _utilities2_pairwise_breaking(qfx2_utilities):
    print('[vote] building pairwise matrix')
    hstack = np.hstack
    cartesian = utool.cartesian
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    altx2_tnx = utool.unique_keep_order2(tnxs)
    tnx2_altx = {nid: altx for altx, nid in enumerate(altx2_tnx)}
    nUtilities = len(qfx2_utilities)
    nAlts   = len(altx2_tnx)
    altxs   = np.arange(nAlts)
    pairwise_mat = np.zeros((nAlts, nAlts))
    qfx2_porder = [np.array([tnx2_altx[util[1]] for util in utils])
                   for utils in qfx2_utilities]

    def sum_win(ij):
        """ pairiwse wins on off-diagonal """
        pairwise_mat[ij[0], ij[1]] += 1

    def sum_loss(ij):
        """ pairiwse wins on off-diagonal """
        pairwise_mat[ij[1], ij[1]] -= 1

    nVoters = 0
    for qfx in xrange(nUtilities):
        # partial and compliment order over alternatives
        porder = utool.unique_keep_order2(qfx2_porder[qfx])
        nReport = len(porder)
        if nReport == 0:
            continue
        #sys.stdout.write('.')
        corder = np.setdiff1d(altxs, porder)
        # pairwise winners and losers
        pw_winners = [porder[r:r + 1] for r in xrange(nReport)]
        pw_losers = [hstack((corder, porder[r + 1:])) for r in xrange(nReport)]
        pw_iter = izip(pw_winners, pw_losers)
        pw_votes_ = [cartesian((winner, losers)) for winner, losers in pw_iter]
        pw_votes = np.vstack(pw_votes_)
        #pw_votes = [(w,l) for votes in pw_votes_ for w,l in votes if w != l]
        map(sum_win,  iter(pw_votes))
        map(sum_loss, iter(pw_votes))
        nVoters += 1
    #print('')
    PLmatrix = pairwise_mat / nVoters
    # sum(0) gives you the sum over rows, which is summing each column
    # Basically a column stochastic matrix should have
    # M.sum(0) = 0
    #print('CheckMat = %r ' % all(np.abs(PLmatrix.sum(0)) < 1E-9))
    return PLmatrix, altx2_tnx


def _get_alts_from_utilities(qfx2_utilities):
    """ get temp name indexes """
    tnxs = [utool[1] for utils in qfx2_utilities for utool in utils]
    altx2_tnx = utool.unique_keep_order2(tnxs)
    tnx2_altx = {nid: altx for altx, nid in enumerate(altx2_tnx)}
    nUtilities = len(qfx2_utilities)
    nAlts   = len(altx2_tnx)
    altxs   = np.arange(nAlts)
    return tnxs, altx2_tnx, tnx2_altx, nUtilities, nAlts, altxs


def _utilities2_weighted_pairwise_breaking(qfx2_utilities):
    print('[vote] building pairwise matrix')
    tnxs, altx2_tnx, tnx2_altx, nUtilities, nAlts, altxs = _get_alts_from_utilities(qfx2_utilities)
    pairwise_mat = np.zeros((nAlts, nAlts))
    # agent to alternative vote vectors
    qfx2_porder = [np.array([tnx2_altx[utool[1]] for utool in utils]) for utils in qfx2_utilities]
    # agent to alternative weight/utility vectors
    qfx2_worder = [np.array([utool[2] for utool in utils]) for utils in qfx2_utilities]
    nVoters = 0
    for qfx in xrange(nUtilities):
        # partial and compliment order over alternatives
        porder = qfx2_porder[qfx]
        worder = qfx2_worder[qfx]
        _, idx = np.unique(porder, return_inverse=True)
        idx = np.sort(idx)
        porder = porder[idx]
        worder = worder[idx]
        nReport = len(porder)
        if nReport == 0:
            continue
        #sys.stdout.write('.')
        corder = np.setdiff1d(altxs, porder)
        nUnreport = len(corder)
        # pairwise winners and losers
        for r_win in xrange(0, nReport):
            # for each prefered alternative
            i = porder[r_win]
            wi = worder[r_win]
            # count the reported victories: i > j
            for r_lose in xrange(r_win + 1, nReport):
                j = porder[r_lose]
                #wj = worder[r_lose]
                #w = wi - wj
                w = wi
                pairwise_mat[i, j] += w
                pairwise_mat[j, j] -= w
            # count the un-reported victories: i > j
            for r_lose in xrange(nUnreport):
                j = corder[r_lose]
                #wj = 0
                #w = wi - wj
                w = wi
                pairwise_mat[i, j] += w
                pairwise_mat[j, j] -= w
            nVoters += wi
    #print('')
    PLmatrix = pairwise_mat / nVoters
    # sum(0) gives you the sum over rows, which is summing each column
    # Basically a column stochastic matrix should have
    # M.sum(0) = 0
    #print('CheckMat = %r ' % all(np.abs(PLmatrix.sum(0)) < 1E-9))
    return PLmatrix, altx2_tnx
# Positional Scoring Rules


def positional_scoring_rule(qfx2_utilities, rule, isWeighted):
    tnxs, altx2_tnx, tnx2_altx, nUtilities, nAlts, altxs = _get_alts_from_utilities(qfx2_utilities)
    # agent to alternative vote vectors
    qfx2_porder = [np.array([tnx2_altx[util[1]] for util in utils]) for utils in qfx2_utilities]
    # agent to alternative weight/utility vectors
    if isWeighted:
        qfx2_worder = [np.array([util[2] for util in utils]) for utils in qfx2_utilities]
    else:
        qfx2_worder = [np.array([    1.0 for util in utils]) for utils in qfx2_utilities]
    K = max(map(len, qfx2_utilities))
    if rule == 'borda':
        score_vec = np.arange(0, K)[::-1] + 1
    if rule == 'plurality':
        score_vec = np.zeros(K)
        score_vec[0] = 1
    if rule == 'topk':
        score_vec = np.ones(K)
    score_vec = np.array(score_vec, dtype=np.int)
    #print('----')
    #title = 'Rule=%s Weighted=%r ' % (rule, not qfx2_weight is None)
    #print('[vote] ' + title)
    #print('[vote] score_vec = %r' % (score_vec,))
    altx2_score = _positional_score(altxs, score_vec, qfx2_porder, qfx2_worder)
    #ranked_candiates = alt_score.argsort()[::-1]
    #ranked_scores    = alt_score[ranked_candiates]
    #viz_votingrule_table(ranked_candiates, ranked_scores, correct_altx, title, fnum)
    return altx2_score, altx2_tnx


def _positional_score(altxs, score_vec, qfx2_porder, qfx2_worder):
    nAlts = len(altxs)
    altx2_score = np.zeros(nAlts)
    # For each voter
    for qfx in xrange(len(qfx2_porder)):
        partial_order = qfx2_porder[qfx]
        weights       = qfx2_worder[qfx]
        # Loop over the ranked alternatives applying positional/meta weight
        for ix, altx in enumerate(partial_order):
            #if altx == -1: continue
            altx2_score[altx] += weights[ix] * score_vec[ix]
    return altx2_score
