from __future__ import absolute_import, division, print_function
import utool
import six
from six.moves import zip, range, map
import numpy as np
import numpy.linalg as npl
import utool as ut
import vtool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[vr2]', DEBUG=False)


def get_chipmatch_testdata(**kwargs):
    from ibeis.model.hots import pipeline
    cfgdict = {'dupvote_weight': 1.0}
    ibs, qreq_ = pipeline.get_pipeline_testdata('testdb1', cfgdict)
    # Run first four pipeline steps
    locals_ = pipeline.testrun_pipeline_upto(qreq_, 'spatial_verification')
    qaid2_chipmatch = locals_['qaid2_chipmatch_FILT']
    # Get a single chipmatch
    qaid = six.next(six.iterkeys(qaid2_chipmatch))
    chipmatch = qaid2_chipmatch[qaid]
    return ibs, qreq_, qaid, chipmatch


def score_chipmatch_csum(qaid, chipmatch, qreq_):
    """
    score_chipmatch_csum

    Args:
        chipmatch (tuple):

    Returns:
        tuple: aid_list, score_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.voting_rules2 import *  # NOQA
        >>> ibs, qreq_, qaid, chipmatch = get_chipmatch_testdata()
        >>> (aid_list, score_list) = score_chipmatch_csum(qaid, chipmatch, qreq_)
        >>> print(aid_list, score_list)
    """
    (_, aid2_fs, _) = chipmatch
    aid_list = list(six.iterkeys(aid2_fs))
    fs_list  = list(six.itervalues(aid2_fs))
    score_list = [np.sum(fs) for fs in fs_list]
    return (aid_list, score_list)
    #aid2_score = {aid: np.sum(fs) for (aid, fs) in six.iteritems(aid2_fs)}
    #return aid2_score


def score_chipmatch_nsum(qaid, chipmatch, qreq_):
    """
    score_chipmatch_nsum

    Args:
        chipmatch (tuple):

    Returns:
        dict: nid2_score

    CommandLine:
        python dev.py -t nsum_nosv --db GZ_ALL --allgt --noqcache
        python dev.py -t nsum --db GZ_ALL --show --va -w --qaid 1032 --noqcache
        python dev.py -t nsum_nosv --db GZ_ALL --show --va -w --qaid 1032 --noqcache
        qaid=1032_res_gooc+f4msr4ouy9t_quuid=c4f78a6d.npz
        qaid=1032_res_5ujbs8h&%vw1olnx_quuid=c4f78a6d.npz

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.voting_rules2 import *  # NOQA
        >>> ibs, qreq_, qaid, chipmatch = get_chipmatch_testdata()
        >>> (aid_list, score_list) = score_chipmatch_nsum(qaid, chipmatch, qreq_)
    """
    (nid_list, nsum_list) = score_chipmatch_true_nsum(qaid, chipmatch, qreq_)
    # for now apply a hack to return aid scores
    aid2_csum = dict(zip(*score_chipmatch_csum(qaid, chipmatch, qreq_)))
    aids_list = qreq_.ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    aid2_nscore = {}
    daids = np.intersect1d(list(six.iterkeys(aid2_csum)),
                           qreq_.get_external_daids())
    for nid, nsum, aids in zip(nid_list, nsum_list, aids_list):
        aids_ = np.intersect1d(aids, daids)
        if len(aids_) == 1:
            aid2_nscore[aids_[0]] = nsum
        elif len(aids_) > 1:
            csum_arr = np.array([aid2_csum[aid] for aid in aids_])
            sortx = csum_arr.argsort()[::-1]
            # Give the best scoring annotation the score
            aid2_nscore[aids_[sortx[0]]] = nsum
            # All other annotations receive 0 score
            for aid in aids_[sortx[1:]]:
                aid2_nscore[aid] = 0
        else:
            print('warning in voting rules nsum')
    aid_list = list(six.iterkeys(aid2_nscore))
    score_list = list(six.itervalues(aid2_nscore))
    return (aid_list, score_list)
    #raise NotImplementedError('nsum')


def score_chipmatch_true_nsum(qaid, chipmatch, qreq_):
    # Nonhacky version of name scoring
    (_, aid2_fs, _) = chipmatch
    aid_list = list(six.iterkeys(aid2_fs))
    annot_score_list = np.array([fs.sum() for fs in six.itervalues(aid2_fs)])
    annot_nid_list = np.array(qreq_.ibs.get_annot_nids(aid_list))
    unique_nids, groupxs = vtool.group_indicies(annot_nid_list)
    grouped_scores = vtool.apply_grouping(annot_score_list, groupxs)
    nid_list = unique_nids
    score_list = [scores.sum() for scores in grouped_scores]
    return nid_list, score_list


def score_chipmatch_nunique(ibs, qaid, chipmatch, qreq):
    raise NotImplementedError('nunique')


def enforce_one_name(ibs, aid2_score, chipmatch=None, aid2_chipscore=None):
    """
    this is a hack to make the same name only show up once in the top ranked
    list
    """
    if chipmatch is not None:
        (_, aid2_fs, _) = chipmatch
        aid2_chipscore = np.array([np.sum(fs) for fs in aid2_fs])
    # FIXME
    nid_list  = ibs.get_name_aids()
    nid2_aids = {nid: aids for nid, aids in zip(ibs.get_name_aids(nid_list))}
    aid2_score = np.array(aid2_score)
    for nid, aids in enumerate(nid2_aids):
        if len(aids) < 2 or nid <= 1:
            continue
        #print(aids)
        # zero the aids with the lowest csum score
        sortx = aid2_chipscore[aids].argsort()
        aids_to_zero = np.array(aids)[sortx[0:-1]]
        aid2_score[aids_to_zero] = 0
    return aid2_score


def score_chipmatch_PL(ibs, qcx, chipmatch, qreq):
    """
    chipmatch = qcx2_chipmatch[qcx]
    """
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
    aid2_score, nid2_score = get_scores_from_altx2_score(ibs, qcx, altx2_prob, altx2_tnx)
    # HACK HACK HACK!!!
    #aid2_score = enforce_one_name_per_cscore(ibs, aid2_score, chipmatch)
    return aid2_score, nid2_score


def _optimize(M):
    #print('[vote] optimize')
    if M.size == 0:
        return np.array([])
    (u, s, v) = npl.svd(M)
    x = np.abs(v[-1])
    check = np.abs(M.dot(x)) < 1E-9
    if not all(check):
        raise Exception('SVD method failed miserably')
    return x


def _PL_score(gamma):
    #print('[vote] computing probabilities')
    nAlts = len(gamma)
    altx2_prob = np.zeros(nAlts)
    for altx in range(nAlts):
        altx2_prob[altx] = gamma[altx] / np.sum(gamma)
    #print('[vote] altx2_prob: '+str(altx2_prob))
    #print('[vote] sum(prob): '+str(sum(altx2_prob)))
    return altx2_prob


def get_scores_from_altx2_score(ibs, qcx, altx2_prob, altx2_tnx):
    nid2_score = np.zeros(len(ibs.tables.nid2_name))
    aid2_score = np.zeros(len(ibs.tables.aid2_aid))
    nid2_cxs = ibs.get_nx2_cxs()
    for altx, prob in enumerate(altx2_prob):
        tnx = altx2_tnx[altx]
        if tnx < 0:  # account for temporary names
            aid2_score[-tnx] = prob
            nid2_score[1] += prob
        else:
            nid2_score[tnx] = prob
            for aid in nid2_cxs[tnx]:
                if aid == qcx:
                    continue
                aid2_score[aid] = prob
    return aid2_score, nid2_score


def _chipmatch2_utilities(ibs, qcx, chipmatch, K):
    """
    Output: qfx2_utilities - map where qfx is the key and utilities are values

    utilities are lists of tuples
    utilities ~ [(aid, temp_name_index, feature_score, feature_rank), ...]

    fx1 : [(aid_0, tnx_0, fs_0, fk_0), ..., (aid_m, tnx_m, fs_m, fk_m)]
    fx2 : [(aid_0, tnx_0, fs_0, fk_0), ..., (aid_m, tnx_m, fs_m, fk_m)]
                    ...
    fxN : [(aid_0, tnx_0, fs_0, fk_0), ..., (aid_m, tnx_m, fs_m, fk_m)]
    """
    #print('[vote] computing utilities')
    aid2_nx = ibs.tables.aid2_nx
    nQFeats = len(ibs.feats.aid2_kpts[qcx])
    # Stack the feature matches
    (aid2_fm, aid2_fs, aid2_fk) = chipmatch
    aids = np.hstack([[aid] * len(aid2_fm[aid]) for aid in range(len(aid2_fm))])
    aids = np.array(aids, np.int)
    fms = np.vstack(aid2_fm)
    # Get the individual feature match lists
    qfxs = fms[:, 0]
    fss  = np.hstack(aid2_fs)
    fks  = np.hstack(aid2_fk)
    qfx2_utilities = [[] for _ in range(nQFeats)]
    for aid, qfx, fk, fs in zip(aids, qfxs, fks, fss):
        nid = aid2_nx[aid]
        # Apply temporary uniquish name
        tnx = nid if nid >= 2 else -aid
        utility = (aid, tnx, fs, fk)
        qfx2_utilities[qfx].append(utility)
    for qfx in range(len(qfx2_utilities)):
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
        for qfx in range(len(qfx2_utilities)):
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
    for qfx in range(nUtilities):
        # partial and compliment order over alternatives
        porder = utool.unique_keep_order2(qfx2_porder[qfx])
        nReport = len(porder)
        if nReport == 0:
            continue
        #sys.stdout.write('.')
        corder = np.setdiff1d(altxs, porder)
        # pairwise winners and losers
        pw_winners = [porder[r:r + 1] for r in range(nReport)]
        pw_losers = [hstack((corder, porder[r + 1:])) for r in range(nReport)]
        pw_iter = zip(pw_winners, pw_losers)
        pw_votes_ = [cartesian((winner, losers)) for winner, losers in pw_iter]
        pw_votes = np.vstack(pw_votes_)
        #pw_votes = [(w,l) for votes in pw_votes_ for w,l in votes if w != l]
        list(map(sum_win,  iter(pw_votes)))
        list(map(sum_loss, iter(pw_votes)))
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
    for qfx in range(nUtilities):
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
        for r_win in range(0, nReport):
            # for each prefered alternative
            i = porder[r_win]
            wi = worder[r_win]
            # count the reported victories: i > j
            for r_lose in range(r_win + 1, nReport):
                j = porder[r_lose]
                #wj = worder[r_lose]
                #w = wi - wj
                w = wi
                pairwise_mat[i, j] += w
                pairwise_mat[j, j] -= w
            # count the un-reported victories: i > j
            for r_lose in range(nUnreport):
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
    for qfx in range(len(qfx2_porder)):
        partial_order = qfx2_porder[qfx]
        weights       = qfx2_worder[qfx]
        # Loop over the ranked alternatives applying positional/meta weight
        for ix, altx in enumerate(partial_order):
            #if altx == -1: continue
            altx2_score[altx] += weights[ix] * score_vec[ix]
    return altx2_score


def score_chipmatch_pos(ibs, qcx, chipmatch, qreq, rule='borda'):
    """
    Positional Scoring Rule
    """
    (aid2_fm, aid2_fs, aid2_fk) = chipmatch
    K = qreq.cfg.nn_cfg.K
    isWeighted = qreq.cfg.agg_cfg.isWeighted
    # Create voting vectors of top K utilities
    qfx2_utilities = _chipmatch2_utilities(ibs, qcx, chipmatch, K)
    # Run Positional Scoring Rule
    altx2_score, altx2_tnx = positional_scoring_rule(qfx2_utilities, rule, isWeighted)
    # Map alternatives back to chips/names
    aid2_score, nid2_score = get_scores_from_altx2_score(ibs, qcx, altx2_score, altx2_tnx)
    # HACK HACK HACK!!!
    #aid2_score = enforce_one_name_per_cscore(ibs, aid2_score, chipmatch)
    return aid2_score, nid2_score


if __name__ == '__main__':
    """
    python ibeis/model/hots/voting_rules2.py
    python ibeis/model/hots/voting_rules2.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
