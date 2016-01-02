# -*- coding: utf-8 -*-
"""
GOALS:
    1) vsmany
       * works resaonable for very few and very many
       * stars with small k and then k becomes a percent or log percent
       * distinctiveness from different location

    2) 1-vs-1
       * uses distinctiveness and foreground when available
       * start with ratio test and ransac

    3) First N decision are interactive until we learn a good threshold

    4) Always show numbers between 0 and 1 spatial verification is based on
    single best exemplar

       x - build encoder
       x - test encoder
       x - monotonicity (both nondecreasing and strictly increasing)
       x - cache encoder
       x - cache maitainance (deleters and listers)
       o - Incemental learning
       o - Spceies sensitivity

    * Add ability for user to relearn encoder from labeled database.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import re
import utool
import numpy as np
import utool as ut
import vtool as vt
import six  # NOQA
from ibeis.algo.hots import chip_match
print, rrr, profile = utool.inject2(__name__, '[scorenorm]')


def learn_annotscore_normalizer(qreq_, learnkw={}):
    """
    Takes the result of queries and trains a score encoder

    Args:
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters

    Returns:
        vtool.ScoreNormalizer: encoder

    CommandLine:
        python -m ibeis --tf learn_annotscore_normalizer --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.testdata_qreq_(
        >>>     defaultdb='PZ_MTEST', a=['default'], p=['default'])
        >>> encoder = learn_annotscore_normalizer(qreq_)
        >>> ut.quit_if_noshow()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    cm_list = qreq_.ibs.query_chips(qreq_=qreq_)
    tup = get_training_annotscores(qreq_, cm_list)
    tp_scores, tn_scores, good_tn_aidnid_pairs, good_tp_aidnid_pairs = tup
    part_attrs = {
        0: {'aid_pairs': good_tn_aidnid_pairs},
        1: {'aid_pairs': good_tp_aidnid_pairs},
    }
    scores, labels, attrs = vt.flatten_scores(tp_scores, tn_scores,
                                              part_attrs)
    _learnkw = {'monotonize': True}
    _learnkw.update(learnkw)
    # timestamp = ut.get_printable_timestamp()
    encoder = vt.ScoreNormalizer(**_learnkw)
    encoder.fit(scores, labels, attrs=attrs)
    encoder.cfgstr = 'annotscore'
    return encoder


def load_featscore_normalizer(normer_cfgstr):
    r"""
    Args:
        normer_cfgstr (?):

    CommandLine:
        python -m ibeis.algo.hots.scorenorm --exec-load_featscore_normalizer --show
        python -m ibeis.algo.hots.scorenorm --exec-load_featscore_normalizer --show --cfgstr=featscore

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> normer_cfgstr = ut.get_argval('--cfgstr', default='featscore')
        >>> encoder = load_featscore_normalizer(normer_cfgstr)
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    encoder = vt.ScoreNormalizer()
    # qreq_.lnbnn_normer.load(cfgstr=config2_.lnbnn_normer)
    encoder.fuzzyload(partial_cfgstr=normer_cfgstr)
    return encoder


def train_featscore_normalizer():
    r"""
    CommandLine:
        python -m ibeis --tf train_featscore_normalizer --show

        # Write Encoder
        python -m ibeis --tf train_featscore_normalizer --db PZ_MTEST -t best -a default --fsvx=0 --threshx=1 --show

        # Visualize encoder score adjustment
        python -m ibeis --tf TestResult.draw_feat_scoresep --db PZ_MTEST -a timectrl -t best:lnbnn_normer=lnbnn_fg_featscore --show --nocache --nocache-hs

        # Compare ranking with encoder vs without
        python -m ibeis --tf draw_rank_cdf --db PZ_MTEST -a timectrl -t best:lnbnn_normer=[None,lnbnn_fg_0.9__featscore] --show
        python -m ibeis --tf draw_rank_cdf --db PZ_MTEST -a default  -t best:lnbnn_normer=[None,lnbnn_fg_0.9__featscore] --show

        # Compare in ipynb
        python -m ibeis --tf autogen_ipynb --ipynb --db PZ_MTEST -a default -t best:lnbnn_normer=[None,lnbnn_fg_0.9__featscore]

        # Big Test
        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 -a timectrl -t best:lnbnn_normer=[None,qvneocwisnclfaqs],lnbnn_norm_thresh=.5 --show

        # Big Train
        python -m ibeis --tf learn_featscore_normalizer --db PZ_Master1 -a timectrl -t best:K=1 --fsvx=0 --threshx=1 --show
        python -m ibeis --tf train_featscore_normalizer --db PZ_Master1 -a timectrl:has_none=photobomb -t best:K=1 --fsvx=0 --threshx=1 --show --ainfo
        python -m ibeis --tf train_featscore_normalizer --db PZ_Master1 -a timectrl:has_none=photobomb -t best:K=1 --fsvx=0 --threshx=1 --show

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> encoder = train_featscore_normalizer()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    import ibeis
    # TODO: training / loading / general external models
    qreq_ = ibeis.testdata_qreq_(
        defaultdb='PZ_MTEST', a=['default'], p=['default'])
    datakw = dict(
        disttypes_=None,
        namemode=ut.get_argval('--namemode', default=True),
        fsvx=ut.get_argval('--fsvx', type_='fuzzy_subset',
                             default=slice(None, None, None)),
        threshx=ut.get_argval('--threshx', type_=int, default=None),
        thresh=ut.get_argval('--thresh', type_=float, default=.9),
    )
    encoder = learn_featscore_normalizer(qreq_, datakw=datakw)
    encoder.save()
    return encoder


def learn_featscore_normalizer(qreq_, datakw={}, learnkw={}):
    r"""
    Takes the result of queries and trains a score encoder

    Args:
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters

    Returns:
        vtool.ScoreNormalizer: encoder

    CommandLine:
        python -m ibeis --tf learn_featscore_normalizer --show
        python -m ibeis --tf learn_featscore_normalizer --show --fsvx=0 --threshx=1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> import ibeis
        >>> learnkw = {}
        >>> datakw = dict(
        >>>     disttypes_=None,
        >>>     namemode=ut.get_argval('--namemode', default=True),
        >>>     fsvx=ut.get_argval('--fsvx', type_='fuzzy_subset',
        >>>                          default=slice(None, None, None)),
        >>>     threshx=ut.get_argval('--threshx', type_=int, default=None),
        >>>     thresh=ut.get_argval('--thresh', type_=float, default=.9),
        >>> )
        >>> qreq_ = ibeis.testdata_qreq_(
        >>>     defaultdb='PZ_MTEST', a=['default'], p=['default'])
        >>> encoder = learn_featscore_normalizer(qreq_)
        >>> ut.quit_if_noshow()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    cm_list = qreq_.ibs.query_chips(qreq_=qreq_)
    print('learning scorenorm')
    print('datakw = ' + ut.repr3(datakw))
    tp_scores, tn_scores, scorecfg = get_training_featscores(
        qreq_, cm_list, **datakw)
    _learnkw = dict(monotonize=True, adjust=2)
    _learnkw.update(learnkw)
    encoder = vt.ScoreNormalizer(**_learnkw)
    encoder.fit_partitioned(tp_scores, tn_scores, verbose=False)
    # ut.hashstr27(qreq_.get_cfgstr())

    # Maintain regen command info: TODO: generalize and integrate
    encoder._regen_info = {
        'cmd': 'python -m ibeis --tf learn_featscore_normalizer',
        'scorecfg': scorecfg,
        'learnkw': learnkw,
        'datakw': datakw,
        'qaids': qreq_.qaids,
        'daids': qreq_.daids,
        'qreq_cfg': qreq_.get_full_cfgstr(),
        'qreq_regen_info': getattr(qreq_, '_regen_info', {}),
        'timestamp': ut.get_printable_timestamp(),
    }

    scorecfg_safe = scorecfg
    scorecfg_safe = re.sub('[' + re.escape('()= ') + ']', '', scorecfg_safe)
    scorecfg_safe = re.sub('[' + re.escape('+*<>[]') + ']', '_', scorecfg_safe)

    hashid = ut.hashstr27(ut.to_json(encoder._regen_info))
    naidinfo = ('q%s_d%s' % (len(qreq_.qaids), len(qreq_.daids)))
    cfgstr = 'featscore_{}_{}_{}_{}'.format(scorecfg_safe, qreq_.ibs.get_dbname(), naidinfo, hashid)
    encoder.cfgstr = cfgstr
    return encoder


def get_training_annotscores(qreq_, cm_list):
    """
    Returns the annotation scores between each query and the correct groundtruth
    annotations as well as the top scoring false annotations.
    """
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []

    ibs = qreq_.ibs

    trainable = [ibs.get_annot_has_groundtruth(cm.qaid, daid_list=cm.daid_list)
                 for cm in cm_list]
    cm_list_ = ut.compress(cm_list, trainable)

    for cm in cm_list_:
        qaid = cm.qaid
        qnid = ibs.get_annot_name_rowids(cm.qaid)

        nscoretup = cm.get_ranked_nids_and_aids()
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores) = nscoretup

        sorted_ndiff = -np.diff(sorted_nscores.tolist())
        sorted_nids = np.array(sorted_nids)
        is_positive  = sorted_nids == qnid
        is_negative = np.logical_and(~is_positive, sorted_nids > 0)
        # Only take data from results with positive and negative examples
        if not np.any(is_positive) or not np.any(is_negative):
            continue
        gt_rank = np.nonzero(is_positive)[0][0]
        gf_rank = np.nonzero(is_negative)[0][0]
        # Only take correct groundtruth scores
        if gt_rank == 0 and len(sorted_nscores) > gf_rank:
            if len(sorted_ndiff) > gf_rank:
                good_tp_nscores.append(sorted_nscores[gt_rank])
                good_tn_nscores.append(sorted_nscores[gf_rank])
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))
    tp_scores = np.array(good_tp_nscores)
    tn_scores = np.array(good_tn_nscores)
    return tp_scores, tn_scores, good_tn_aidnid_pairs, good_tp_aidnid_pairs


def get_training_featscores(qreq_, cm_list, disttypes_=None, namemode=True,
                            fsvx=slice(None, None, None), threshx=None, thresh=.9):
    """
    Returns the flattened set of feature scores between each query and the
    correct groundtruth annotations as well as the top scoring false
    annotations.

    Args:
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters
        cm_list (list):
        disttypes_ (None): (default = None)
        namemode (bool): (default = True)
        fsvx (slice): (default = slice(None, None, None))
        threshx (None): (default = None)
        thresh (float): only used if threshx is specified (default = 0.9)

    SeeAlso:
        TestResult.draw_feat_scoresep

    Returns:
        tuple: (tp_scores, tn_scores, scorecfg)

    CommandLine:
        python -m ibeis.algo.hots.scorenorm --exec-get_training_featscores

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> import ibeis
        >>> cm_list, qreq_ = ibeis.testdata_cmlist(defaultdb='PZ_MTEST', a=['default:qsize=10'])
        >>> disttypes_ = None
        >>> namemode = True
        >>> fsvx = None
        >>> threshx = 1
        >>> thresh = 0.5
        >>> (tp_scores, tn_scores, scorecfg) = get_training_featscores(
        >>>     qreq_, cm_list, disttypes_, namemode, fsvx, threshx, thresh)
        >>> print(scorecfg)
        lnbnn*fg[fg > 0.5]
    """
    if fsvx is None:
        fsvx = slice(None, None, None)

    fsv_col_lbls = None
    tp_fsvs_list = []
    tn_fsvs_list = []

    trainable = [
        qreq_.ibs.get_annot_has_groundtruth(cm.qaid, daid_list=cm.daid_list)
        for cm in cm_list
    ]
    cm_list_ = ut.compress(cm_list, trainable)

    for cm in ut.ProgIter(cm_list_, lbl='building train featscores',
                          adjust=True, freq=1):
        try:
            if disttypes_ is None:
                # Use precomputed fsv distances
                fsv_col_lbls = cm.fsv_col_lbls
                tp_fsv, tn_fsv = chip_match.get_training_fsv(
                    cm, namemode=namemode)
            else:
                # Investigate independant computed dists
                fsv_col_lbls = disttypes_
                tp_fsv, tn_fsv = chip_match.get_training_desc_dist(
                    cm, qreq_, fsv_col_lbls, namemode=namemode)
            tp_fsvs_list.extend(tp_fsv)
            tn_fsvs_list.extend(tn_fsv)
        except chip_match.UnbalancedExampleException:
            continue
    fsv_tp = np.vstack(tp_fsvs_list)
    fsv_tn = np.vstack(tn_fsvs_list)

    fsv_col_lbls_ = ut.list_take(fsv_col_lbls, fsvx)
    fsv_tp_ = fsv_tp.T[fsvx].T
    fsv_tn_ = fsv_tn.T[fsvx].T

    if threshx is not None:
        tp_scores = fsv_tp_[fsv_tp.T[threshx] > thresh].prod(axis=1)
        tn_scores = fsv_tn_[fsv_tn.T[threshx] > thresh].prod(axis=1)
        threshpart = ('[' + fsv_col_lbls[threshx] + ' > ' + str(thresh) + ']')
        scorecfg = '(%s)%s' % ('*'.join(fsv_col_lbls_), threshpart)
    else:
        tp_scores = fsv_tp_.prod(axis=1)
        tn_scores = fsv_tn_.prod(axis=1)
        scorecfg = '*'.join(fsv_col_lbls_)

    return tp_scores, tn_scores, scorecfg


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.scorenorm
        python -m ibeis.algo.hots.scorenorm --allexamples
        python -m ibeis.algo.hots.scorenorm --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
