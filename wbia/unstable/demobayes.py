# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from wbia.unstable.bayes import make_name_model, temp_model, draw_tree_model

print, rrr, profile = ut.inject2(__name__)


def trytestdata_demo_cfgs():
    alias_keys = {'nA': 'num_annots', 'nN': 'num_names', 'nS': 'num_scores'}
    cfg_list = ut.parse_argv_cfg('--ev', alias_keys=alias_keys)
    return cfg_list


def demo_bayesnet(cfg={}):
    r"""
    Make a model that knows who the previous annots are and tries to classify a new annot

    CommandLine:
        python -m wbia --tf demo_bayesnet --diskshow --verbose --save demo4.png --dpath . --figsize=20,10 --dpi=128 --clipwhite

        python -m wbia --tf demo_bayesnet --ev :nA=3,Sab=0,Sac=0,Sbc=1
        python -m wbia --tf demo_bayesnet --ev :nA=4,Sab=0,Sac=0,Sbc=1,Sbd=1 --show
        python -m wbia --tf demo_bayesnet --ev :nA=4,Sab=0,Sac=0,Sbc=1,Scd=1 --show
        python -m wbia --tf demo_bayesnet --ev :nA=4,Sab=0,Sac=0,Sbc=1,Sbd=1,Scd=1 --show

        python -m wbia --tf demo_bayesnet --ev :nA=3,Sab=0,Sac=0,Sbc=1
        python -m wbia --tf demo_bayesnet --ev :nA=5,rand_scores=True --show

        python -m wbia --tf demo_bayesnet --ev :nA=4,nS=3,rand_scores=True --show --verbose
        python -m wbia --tf demo_bayesnet --ev :nA=5,nS=2,Na=fred,rand_scores=True --show --verbose
        python -m wbia --tf demo_bayesnet --ev :nA=5,nS=5,Na=fred,rand_scores=True --show --verbose
        python -m wbia --tf demo_bayesnet --ev :nA=4,nS=2,Na=fred,rand_scores=True --show --verbose

        python -m wbia.unstable.demobayes --exec-demo_bayesnet \
                --ev =:nA=4,Sab=0,Sac=0,Sbc=1 \
                :Sbd=1 :Scd=1 :Sbd=1,Scd=1 :Sbd=1,Scd=1,Sad=0 \
                --show --present

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> cfg_list = testdata_demo_cfgs()
        >>> print('cfg_list = %r' % (cfg_list,))
        >>> for cfg in cfg_list:
        >>>     demo_bayesnet(cfg)
        >>> ut.show_if_requested()
    """
    cfg = cfg.copy()
    num_annots = cfg.pop('num_annots', 3)
    num_names = cfg.pop('num_names', None)
    num_scores = cfg.pop('num_scores', 2)
    rand_scores = cfg.pop('rand_scores', False)
    method = cfg.pop('method', 'bp')
    other_evidence = {k: v for k, v in cfg.items() if not k.startswith('_')}
    if rand_scores:
        # import randomdotorg
        # import sys
        # r = randomdotorg.RandomDotOrg('ExampleCode')
        # seed = int((1 - 2 * r.random()) * sys.maxint)
        toy_data = get_toy_data_1v1(num_annots, nid_sequence=[0, 0, 1, 0, 1, 2])
        print('toy_data = ' + ut.repr3(toy_data, nl=1))
        (diag_scores,) = ut.dict_take(toy_data, 'diag_scores'.split(', '))
        discr_domain, discr_p_same = learn_prob_score(num_scores)[0:2]

        def discretize_scores(scores):
            # Assign continuous scores to discrete index
            score_idxs = np.abs(1 - (discr_domain / scores[:, None])).argmin(axis=1)
            return score_idxs

        score_evidence = discretize_scores(diag_scores)
    else:
        score_evidence = []
        discr_p_same = None
        discr_domain = None
    model, evidence, query_results = temp_model(
        num_annots=num_annots,
        num_names=num_names,
        num_scores=num_scores,
        score_evidence=score_evidence,
        mode=1,
        other_evidence=other_evidence,
        p_score_given_same=discr_p_same,
        score_basis=discr_domain,
        method=method,
    )


def classify_k(cfg={}):
    """
    CommandLine:
        python -m wbia.unstable.demobayes --exec-classify_k --show --ev :nA=3
        python -m wbia.unstable.demobayes --exec-classify_k --show --ev :nA=3,k=1
        python -m wbia.unstable.demobayes --exec-classify_k --show --ev :nA=3,k=0 --method=approx
        python -m wbia.unstable.demobayes --exec-classify_k --show --ev :nA=10,k=1 --method=approx

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> cfg_list = testdata_demo_cfgs()
        >>> classify_k(cfg_list[0])
        >>> ut.show_if_requested()
    """
    cfg = cfg.copy()
    num_annots = cfg.pop('num_annots', 3)
    num_scores = cfg.pop('num_scores', 2)
    num_iter = cfg.pop('k', 0)
    nid_sequence = np.array([0, 0, 1, 2, 2, 1, 1])
    toy_data = get_toy_data_1v1(num_annots, nid_sequence=nid_sequence)
    force_evidence = None
    force_evidence = 0
    (diag_scores,) = ut.dict_take(toy_data, 'diag_scores'.split(', '))

    # print('diag_scores = %r' % (diag_scores,))
    # diag_labels = pairwise_matches.compress(is_diag)
    # diag_pairs = ut.compress(pairwise_aidxs, is_diag)

    discr_domain, discr_p_same = learn_prob_score(num_scores)[0:2]

    def discretize_scores(scores):
        # Assign continuous scores to closest discrete index
        score_idxs = np.abs(1 - (discr_domain / scores[:, None])).argmin(axis=1)
        return score_idxs

    # Careful ordering is important here
    score_evidence = discretize_scores(diag_scores)
    if force_evidence is not None:
        for x in range(len(score_evidence)):
            score_evidence[x] = 0

    model, evidence, query_results = temp_model(
        num_annots=num_annots,
        num_names=num_annots,
        num_scores=num_scores,
        mode=1,
        score_evidence=score_evidence,
        p_score_given_same=discr_p_same,
        score_basis=discr_domain,
        # verbose=True
    )
    print(query_results['top_assignments'][0])
    toy_data1 = toy_data
    print('toy_data1 = ' + ut.repr3(toy_data1, nl=1))
    num_annots2 = num_annots + 1
    score_evidence1 = [None] * len(score_evidence)
    full_evidence = score_evidence.tolist()

    factor_list = query_results['factor_list']
    using_soft = False
    if using_soft:
        soft_evidence1 = [dict(zip(x.statenames[0], x.values)) for x in factor_list]

    for _ in range(num_iter):
        print('\n\n ---------- \n\n')
        # toy_data1['all_nids'].max() + 1
        num_names_gen = len(toy_data1['all_aids']) + 1
        num_names_gen = toy_data1['all_nids'].max() + 2
        toy_data2 = get_toy_data_1v1(
            1,
            num_names_gen,
            initial_aids=toy_data1['all_aids'],
            initial_nids=toy_data1['all_nids'],
            nid_sequence=nid_sequence,
        )
        (diag_scores2,) = ut.dict_take(toy_data2, 'diag_scores'.split(', '))
        print('toy_data2 = ' + ut.repr3(toy_data2, nl=1))

        score_evidence2 = discretize_scores(diag_scores2).tolist()
        if force_evidence is not None:
            for x in range(len(score_evidence2)):
                score_evidence2[x] = force_evidence
        print('score_evidence2 = %r' % (score_evidence2,))

        if using_soft:
            # Demo with soft evidence
            model, evidence, query_results2 = temp_model(
                num_annots=num_annots2,
                num_names=num_annots2,
                num_scores=num_scores,
                mode=1,
                name_evidence=soft_evidence1,
                # score_evidence=score_evidence1 + score_evidence2,
                score_evidence=score_evidence2,
                p_score_given_same=discr_p_same,
                score_basis=discr_domain,
                # verbose=True,
                hack_score_only=len(score_evidence2),
            )

        if 1:
            # Demo with full evidence
            model, evidence, query_results2 = temp_model(
                num_annots=num_annots2,
                num_names=num_annots2,
                num_scores=num_scores,
                mode=1,
                score_evidence=full_evidence + score_evidence2,
                p_score_given_same=discr_p_same,
                score_basis=discr_domain,
                verbose=True,
            )
        factor_list2 = query_results2['factor_list']
        if using_soft:
            soft_evidence1 = [dict(zip(x.statenames[0], x.values)) for x in factor_list2]
        score_evidence1 += [None] * len(score_evidence2)
        full_evidence = full_evidence + score_evidence2
        num_annots2 += 1
        toy_data1 = toy_data2


def show_toy_distributions(toy_params):
    import vtool as vt
    import wbia.plottool as pt

    pt.ensureqt()
    xdata = np.linspace(0, 8, 1000)
    tp_pdf = vt.gauss_func1d(xdata, **toy_params[True])
    fp_pdf = vt.gauss_func1d(xdata, **toy_params[False])
    pt.plot_probabilities(
        [tp_pdf, fp_pdf],
        ['TP', 'TF'],
        prob_colors=[pt.TRUE_BLUE, pt.FALSE_RED],
        xdata=xdata,
        figtitle='Toy Distributions',
    )


def get_toy_data_1vM(num_annots, num_names=None, **kwargs):
    r"""
    Args:
        num_annots (int):
        num_names (int): (default = None)

    Kwargs:
        initial_aids, initial_nids, nid_sequence, seed

    Returns:
        tuple: (pair_list, feat_list)

    CommandLine:
        python -m wbia.unstable.demobayes --exec-get_toy_data_1vM --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> num_annots = 1000
        >>> num_names = 40
        >>> get_toy_data_1vM(num_annots, num_names)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import vtool as vt

    tup_ = get_toy_annots(num_annots, num_names, **kwargs)
    aids, nids, aids1, nids1, all_aids, all_nids = tup_
    rng = vt.ensure_rng(None)

    # Test a simple SVM classifier
    nid2_nexemp = ut.dict_hist(nids1)
    aid2_nid = dict(zip(aids, nids))

    ut.fix_embed_globals()

    # def add_to_globals(globals_, subdict):
    #    globals_.update(subdict)

    unique_nids = list(nid2_nexemp.keys())

    def annot_to_class_feats2(aid, aid2_nid, top=None):
        pair_list = []
        score_list = []
        nexemplar_list = []
        for nid in unique_nids:
            label = aid2_nid[aid] == nid
            num_exemplars = nid2_nexemp.get(nid, 0)
            if num_exemplars == 0:
                continue
            params = toy_params[label]
            mu, sigma = ut.dict_take(params, ['mu', 'sigma'])
            score_ = rng.normal(mu, sigma, size=num_exemplars).max()
            score = np.clip(score_, 0, np.inf)
            pair_list.append((aid, nid))
            score_list.append(score)
            nexemplar_list.append(num_exemplars)
        rank_list = ut.argsort(score_list, reverse=True)
        feat_list = np.array([score_list, rank_list, nexemplar_list]).T
        sortx = np.argsort(rank_list)
        feat_list = feat_list.take(sortx, axis=0)
        pair_list = np.array(pair_list).take(sortx, axis=0)
        if top is not None:
            feat_list = feat_list[:top]
            pair_list = pair_list[0:top]
        return pair_list, feat_list

    toclass_features = [annot_to_class_feats2(aid, aid2_nid, top=5) for aid in aids]
    aidnid_pairs = np.vstack(ut.get_list_column(toclass_features, 0))
    feat_list = np.vstack(ut.get_list_column(toclass_features, 1))
    score_list = feat_list.T[0:1].T
    lbl_list = [aid2_nid[aid] == nid for aid, nid in aidnid_pairs]

    from sklearn import svm

    # clf1 = svm.LinearSVC()
    print('Learning classifiers')

    clf3 = svm.SVC(probability=True)
    clf3.fit(feat_list, lbl_list)
    # prob_true, prob_false = clf3.predict_proba(feat_list).T

    clf1 = svm.LinearSVC()
    clf1.fit(score_list, lbl_list)

    # Score new annots against the training database
    tup_ = get_toy_annots(
        num_annots * 2, num_names, initial_aids=all_aids, initial_nids=all_nids
    )
    aids, nids, aids1, nids1, all_aids, all_nids = tup_
    aid2_nid = dict(zip(aids, nids))
    toclass_features = [annot_to_class_feats2(aid, aid2_nid) for aid in aids]
    aidnid_pairs = np.vstack(ut.get_list_column(toclass_features, 0))
    feat_list = np.vstack(ut.get_list_column(toclass_features, 1))
    lbl_list = np.array([aid2_nid[aid] == nid for aid, nid in aidnid_pairs])

    print('Running tests')

    score_list = feat_list.T[0:1].T

    tp_feat_list = feat_list[lbl_list]
    tn_feat_list = feat_list[~lbl_list]
    tp_lbls = lbl_list[lbl_list]
    tn_lbls = lbl_list[~lbl_list]
    print('num tp: %d' % len(tp_lbls))
    print('num fp: %d' % len(tn_lbls))

    tp_score_list = score_list[lbl_list]
    tn_score_list = score_list[~lbl_list]

    print('tp_feat' + ut.repr3(ut.get_stats(tp_feat_list, axis=0), precision=2))
    print('tp_feat' + ut.repr3(ut.get_stats(tn_feat_list, axis=0), precision=2))

    print('tp_score' + ut.repr2(ut.get_stats(tp_score_list), precision=2))
    print('tp_score' + ut.repr2(ut.get_stats(tn_score_list), precision=2))

    tp_pred3 = clf3.predict(tp_feat_list)
    tn_pred3 = clf3.predict(tn_feat_list)
    print((tp_pred3.sum(), tp_pred3.shape))
    print((tn_pred3.sum(), tn_pred3.shape))

    tp_score3 = clf3.score(tp_feat_list, tp_lbls)
    tn_score3 = clf3.score(tn_feat_list, tn_lbls)

    tp_pred1 = clf1.predict(tp_score_list)
    tn_pred1 = clf1.predict(tn_score_list)
    print((tp_pred1.sum(), tp_pred1.shape))
    print((tn_pred1.sum(), tn_pred1.shape))

    tp_score1 = clf1.score(tp_score_list, tp_lbls)
    tn_score1 = clf1.score(tn_score_list, tn_lbls)
    print('tp score with rank    = %r' % (tp_score3,))
    print('tn score with rank    = %r' % (tn_score3,))

    print('tp score without rank = %r' % (tp_score1,))
    print('tn score without rank = %r' % (tn_score1,))
    toy_data = {}

    return toy_data


def get_toy_annots(
    num_annots,
    num_names=None,
    initial_aids=None,
    initial_nids=None,
    nid_sequence=None,
    seed=None,
):
    r"""
    Args:
        num_annots (int):
        num_names (int): (default = None)
        initial_aids (None): (default = None)
        initial_nids (None): (default = None)
        nid_sequence (None): (default = None)
        seed (None): (default = None)

    Returns:
        tuple: (aids, nids, aids1, nids1, all_aids, all_nids)

    CommandLine:
        python -m wbia.unstable.demobayes --exec-get_toy_annots

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> num_annots = 1
        >>> num_names = 5
        >>> initial_aids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        >>> initial_nids = np.array([0, 0, 1, 2, 2, 1, 1, 1, 2, 3], dtype=np.int64)
        >>> nid_sequence = np.array([0, 0, 1, 2, 2, 1, 1], dtype=np.int64)
        >>> seed = 0
        >>> (aids, nids, aids1, nids1, all_aids, all_nids) = get_toy_annots(num_annots, num_names, initial_aids, initial_nids, nid_sequence, seed)
        >>> result = ('(aids, nids, aids1, nids1, all_aids, all_nids) = %s' % (ut.repr2((aids, nids, aids1, nids1, all_aids, all_nids), nl=1),))
        >>> print(result)
    """
    import vtool as vt

    if num_names is None:
        num_names = num_annots
    print('Generating toy data with num_annots=%r' % (num_annots,))
    if initial_aids is None:
        assert initial_nids is None
        first_step = True
        initial_aids = []
        initial_nids = []
    else:
        first_step = False
        assert initial_nids is not None

    aids = np.arange(len(initial_aids), num_annots + len(initial_aids))
    rng = vt.ensure_rng(seed)
    if nid_sequence is None:
        nids = rng.randint(0, num_names, num_annots)
    else:
        unused_from_sequence = max(len(nid_sequence) - len(initial_aids), 0)
        if unused_from_sequence == 0:
            nids = rng.randint(0, num_names, num_annots)
        elif unused_from_sequence > 0 and unused_from_sequence < num_annots:
            num_remain = num_annots - unused_from_sequence
            nids = np.append(
                nid_sequence[-unused_from_sequence:],
                rng.randint(0, num_names, num_remain),
            )
        else:
            nids = nid_sequence[-unused_from_sequence]
            nids = np.array(
                ut.take(
                    nid_sequence, range(len(initial_aids), len(initial_aids) + num_annots)
                )
            )

    if first_step:
        aids1 = aids
        nids1 = nids
    else:
        aids1 = initial_aids
        nids1 = initial_nids

    all_nids = np.append(initial_nids, nids)
    all_aids = np.append(initial_aids, aids)
    import utool

    with utool.embed_on_exception_context:
        ut.assert_eq(len(aids), len(nids), 'len new')
        ut.assert_eq(len(aids1), len(nids1), 'len comp')
        ut.assert_eq(len(all_aids), len(all_nids), 'len all')
    return aids, nids, aids1, nids1, all_aids, all_nids


toy_params = {
    True: {'mu': 1.5, 'sigma': 3.0},
    False: {'mu': 0.5, 'sigma': 0.4}
    # True: {'mu': 3.5, 'sigma': 1.1},
    # False: {'mu': .3, 'sigma': .7}
    # 'p': .7},
    # 'p': .2}
}


# @ut.cached_func('_toy_bayes_data3')
def get_toy_data_1v1(num_annots=5, num_names=None, **kwargs):
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes --exec-get_toy_data_1v1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> toy_data = get_toy_data_1v1()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> show_toy_distributions(toy_data['toy_params'])
        >>> ut.show_if_requested()

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> toy_data = get_toy_data_1v1()
        >>> kwargs = {}
        >>> initial_aids = toy_data['aids']
        >>> initial_nids = toy_data['nids']
        >>> num_annots = 1
        >>> num_names = 6
        >>> toy_data2 = get_toy_data_1v1(num_annots, num_names, initial_aids=initial_aids, initial_nids=initial_nids)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> show_toy_distributions(toy_data['toy_params'])
        >>> ut.show_if_requested()

    Ignore:
        >>> num_annots = 1000
        >>> num_names = 400
    """
    import vtool as vt

    tup_ = get_toy_annots(num_annots, num_names, **kwargs)
    aids, nids, aids1, nids1, all_aids, all_nids = tup_
    rng = vt.ensure_rng(None)

    def pairwise_feature(aidx1, aidx2, all_nids=all_nids, toy_params=toy_params):
        if aidx1 == aidx2:
            score = -1
        else:
            # rng = np.random.RandomState(int((aidx1 + 13) * (aidx2 + 13)))
            nid1 = all_nids[int(aidx1)]
            nid2 = all_nids[int(aidx2)]
            params = toy_params[nid1 == nid2]
            mu, sigma = ut.dict_take(params, ['mu', 'sigma'])
            score_ = rng.normal(mu, sigma)
            score = np.clip(score_, 0, np.inf)
        return score

    pairwise_nids = list([tup[::-1] for tup in ut.iprod(nids, nids1)])
    pairwise_matches = np.array([nid1 == nid2 for nid1, nid2 in pairwise_nids])

    pairwise_aidxs = list([tup[::-1] for tup in ut.iprod(aids, aids1)])

    pairwise_features = np.array(
        [pairwise_feature(aidx1, aidx2) for aidx1, aidx2 in pairwise_aidxs]
    )

    # pairwise_scores_mat = pairwise_scores.reshape(num_annots, num_annots)
    is_diag = [r < c for r, c, in pairwise_aidxs]
    diag_scores = pairwise_features.compress(is_diag)
    diag_aidxs = ut.compress(pairwise_aidxs, is_diag)
    import utool

    with utool.embed_on_exception_context:
        diag_nids = ut.compress(pairwise_nids, is_diag)
    diag_labels = pairwise_matches.compress(is_diag)

    # import utool
    # utool.embed()

    toy_data = {
        'aids': aids,
        'nids': nids,
        'all_nids': all_nids,
        'all_aids': all_aids,
        # 'pairwise_aidxs': pairwise_aidxs,
        # 'pairwise_scores': pairwise_scores,
        # 'pairwise_matches': pairwise_matches,
        'diag_labels': diag_labels,
        'diag_scores': diag_scores,
        'diag_nids': diag_nids,
        'diag_aidxs': diag_aidxs,
        'toy_params': toy_params,
    }
    return toy_data


@ut.cached_func('_toy_learn_prob_score5')
def learn_prob_score(num_scores=5, pad=55, ret_enc=False, use_cache=None):
    r"""
    Args:
        num_scores (int): (default = 5)

    Returns:
        tuple: (discr_domain, discr_p_same)

    CommandLine:
        python -m wbia.unstable.demobayes --exec-learn_prob_score --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> num_scores = 2
        >>> (discr_domain, discr_p_same, encoder) = learn_prob_score(num_scores, ret_enc=True, use_cache=False)
        >>> print('discr_p_same = %r' % (discr_p_same,))
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> encoder.visualize()
        >>> ut.show_if_requested()
    """
    num_annots_train = 200
    num_names_train = 5
    toy_data = get_toy_data_1v1(num_annots_train, num_names_train)
    # pairwise_aidxs, pairwise_scores, pairwise_matches = ut.dict_take(
    #    toy_data, 'pairwise_aidxs, pairwise_scores, pairwise_matches'.split(', '))

    diag_scores, diag_labels = ut.dict_take(
        toy_data, 'diag_scores, diag_labels'.split(', ')
    )
    # is_diag = [r < c for r, c, in pairwise_aidxs]
    # diag_scores = pairwise_scores.compress(is_diag)
    # diag_labels = pairwise_matches.compress(is_diag)

    # Learn P(S_{ij} | M_{ij})
    import vtool as vt

    encoder = vt.ScoreNormalizer(reverse=True, monotonize=True, adjust=4,)
    encoder.fit(X=diag_scores, y=diag_labels, verbose=True)

    if False:
        import wbia.plottool as pt

        pt.ensureqt()
        encoder.visualize()
        # show_toy_distributions()

    def discretize_probs(encoder):
        p_tp_given_score = encoder.p_tp_given_score / encoder.p_tp_given_score.sum()
        bins = len(p_tp_given_score) - (pad * 2)
        stride = int(np.ceil(bins / num_scores))
        idxs = np.arange(0, bins, stride) + pad
        discr_p_same = p_tp_given_score.take(idxs)
        discr_p_same = discr_p_same / discr_p_same.sum()
        discr_domain = encoder.score_domain.take(idxs)
        return discr_domain, discr_p_same

    discr_domain, discr_p_same = discretize_probs(encoder)
    if ret_enc:
        return discr_domain, discr_p_same, encoder
    return discr_domain, discr_p_same


def classify_one_new_unknown():
    r"""
    Make a model that knows who the previous annots are and tries to classify a new annot

    CommandLine:
        python -m wbia.unstable.demobayes --exec-classify_one_new_unknown --verbose
        python -m wbia.unstable.demobayes --exec-classify_one_new_unknown --show --verbose --present
        python3 -m wbia.unstable.demobayes --exec-classify_one_new_unknown --verbose
        python3 -m wbia.unstable.demobayes --exec-classify_one_new_unknown --verbose --diskshow --verbose --present --save demo5.png --dpath . --figsize=20,10 --dpi=128 --clipwhite

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = classify_one_new_unknown()
        >>> ut.show_if_requested()
    """
    if False:
        constkw = dict(
            num_annots=5,
            num_names=3,
            name_evidence=[0]
            # name_evidence=[0, 0, 1, 1, None],
            # name_evidence=[{0: .99}, {0: .99}, {1: .99}, {1: .99}, None],
            # name_evidence=[0, {0: .99}, {1: .99}, 1, None],
        )
        temp_model(score_evidence=[1, 0, 0, 0, 0, 1], mode=1, **constkw)

    # from wbia.unstable.demobayes import *
    constkw = dict(num_annots=4, num_names=4,)
    model, evidence = temp_model(
        mode=1,
        # lll and llh have strikingly different
        # probability of M marginals
        score_evidence=[0, 0, 1],
        other_evidence={},
        **constkw
    )


def tst_triangle_property():
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes --exec-test_triangle_property --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = test_triangle_property()
        >>> ut.show_if_requested()
    """
    constkw = dict(num_annots=3, num_names=3, name_evidence=[],)
    temp_model(
        mode=1,
        other_evidence={
            'Mab': False,
            'Mac': False,
            # 'Na': 'fred',
            # 'Nb': 'sue',
        },
        **constkw
    )


def demo_structure():
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_structure --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = demo_structure()
        >>> ut.show_if_requested()
    """
    constkw = dict(score_evidence=[], name_evidence=[], mode=3)
    (model,) = temp_model(num_annots=4, num_names=4, **constkw)
    draw_tree_model(model)


def make_bayes_notebook():
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes --exec-make_bayes_notebook

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = make_bayes_notebook()
        >>> print(result)
    """
    from wbia.templates import generate_notebook

    initialize = ut.codeblock(
        r"""
        # STARTBLOCK
        import os
        os.environ['UTOOL_NO_CNN'] = 'True'
        from wbia.unstable.demobayes import *  # NOQA
        # Matplotlib stuff
        import matplotlib as mpl
        %matplotlib inline
        %load_ext autoreload
        %autoreload
        from IPython.core.display import HTML
        HTML("<style>body .container { width:99% !important; }</style>")
        # ENDBLOCK
        """
    )
    cell_list_def = [
        initialize,
        show_model_templates,
        demo_modes,
        demo_name_annot_complexity,
        # demo_model_idependencies,
        demo_single_add,
        demo_ambiguity,
        demo_conflicting_evidence,
        demo_annot_idependence_overlap,
    ]

    def format_cell(cell):
        if ut.is_funclike(cell):
            header = '# ' + ut.to_title_caps(ut.get_funcname(cell))
            code = (header, ut.get_func_sourcecode(cell, stripdef=True, stripret=True))
        else:
            code = (None, cell)
        return generate_notebook.format_cells(code)

    cell_list = ut.flatten([format_cell(cell) for cell in cell_list_def])
    nbstr = generate_notebook.make_notebook(cell_list)
    print('nbstr = %s' % (nbstr,))
    fpath = 'demobayes.ipynb'
    ut.writeto(fpath, nbstr)
    ut.startfile(fpath)


def show_model_templates():
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes --exec-show_model_templates

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = show_model_templates()
        >>> ut.show_if_requested()
    """
    make_name_model(2, 2, verbose=True, mode=1)
    print('-------------')
    make_name_model(2, 2, verbose=True, mode=2)


def demo_single_add():
    """
    This demo shows how a name is assigned to a new annotation.

    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_single_add --show --present --mode=1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> demo_single_add()
        >>> ut.show_if_requested()
    """
    # Initially there are only two annotations that have a strong match
    name_evidence = [{0: 0.9}]  # Soft label
    name_evidence = [0]  # Hard label
    temp_model(num_annots=2, num_names=5, score_evidence=[1], name_evidence=name_evidence)
    # Adding a new annotation does not change the original probabilites
    temp_model(num_annots=3, num_names=5, score_evidence=[1], name_evidence=name_evidence)
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb. However the probability that Nb matches Nc goes up.
    temp_model(
        num_annots=3, num_names=5, score_evidence=[1, 1], name_evidence=name_evidence
    )
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    temp_model(
        num_annots=3, num_names=5, score_evidence=[1, 1, 1], name_evidence=name_evidence
    )


def demo_conflicting_evidence():
    """
    Notice that the number of annotations in the graph does not affect the
    probability of names.
    """
    # Initialized with two annots. Each are pretty sure they are someone else
    constkw = dict(num_annots=2, num_names=5, score_evidence=[])
    temp_model(name_evidence=[{0: 0.9}, {1: 0.9}], **constkw)
    # Having evidence that they are different increases this confidence.
    temp_model(name_evidence=[{0: 0.9}, {1: 0.9}], other_evidence={'Sab': 0}, **constkw)
    # However,, confusion is introduced if there is evidence that they are the same
    temp_model(name_evidence=[{0: 0.9}, {1: 0.9}], other_evidence={'Sab': 1}, **constkw)
    # When Na is forced to be fred, this doesnt change Nbs evaulatation by more
    # than a few points
    temp_model(name_evidence=[0, {1: 0.9}], other_evidence={'Sab': 1}, **constkw)


def demo_ambiguity():
    r"""
    Test what happens when an annotation need to choose between one of two
    names

    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_ambiguity --show --verbose --present

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = demo_ambiguity()
        >>> ut.show_if_requested()
    """
    constkw = dict(
        num_annots=3,
        num_names=3,
        name_evidence=[0],
        # name_evidence=[],
        # name_evidence=[{0: '+eps'}, {1: '+eps'}, {2: '+eps'}],
    )
    temp_model(score_evidence=[0, 0, 1], mode=1, **constkw)


def demo_annot_idependence_overlap():
    r"""
    Given:
        * an unknown annotation \d
        * three annots with the same name (Fred) \a, \b, and \c
        * \a and \b are near duplicates
        * (\a and \c) / (\b and \c) are novel views

    Goal:
        * If \d matches to \a and \b the probably that \d is Fred should not be
          much more than if \d matched only \a or only \b.

        * The probability that \d is Fred given it matches to any of the 3 annots
           alone should be equal

            P(\d is Fred | Mad=1) = P(\d is Fred | Mbd=1) = P(\d is Fred | Mcd=1)

        * The probability that \d is fred given two matches to any of those two annots
          should be greater than the probability given only one.

            P(\d is Fred | Mad=1, Mbd=1) > P(\d is Fred | Mad=1)
            P(\d is Fred | Mad=1, Mcd=1) > P(\d is Fred | Mad=1)

        * The probability that \d is fred given matches to two near duplicate
          matches should be less than
          if \d matches two non-duplicate matches.

            P(\d is Fred | Mad=1, Mcd=1) > P(\d is Fred | Mad=1, Mbd=1)

        * The probability that \d is fred given two near duplicates should be only epsilon greater than
          a match to either one individually.

            P(\d is Fred | Mad=1, Mbd=1) = P(\d is Fred | Mad=1) + \epsilon

    Method:

        We need to model the fact that there are other causes that create the
        effect of a high score.  Namely, near duplicates.
        This can be done by adding an extra conditional that score depends on
        if they match as well as if they are near duplicates.

        P(S_ij | Mij) --> P(S_ij | Mij, Dij)

        where

        Dij is a random variable indicating if the image is a near duplicate.

        We can model this as an independant variable

        P(Dij) = {True: .5, False: .5}

        or as depending on if the names match.

        P(Dij | Mij) = {'same': {True: .5, False: .5} diff: {True: 0, False 1}}



    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_annot_idependence_overlap --verbose --present --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = demo_annot_idependence_overlap()
        >>> ut.show_if_requested()
    """
    # We will end up making annots a and b fred and c and d sue
    constkw = dict(
        num_annots=4,
        num_names=4,
        name_evidence=[{0: '+eps'}, {1: '+eps'}, {2: '+eps'}, {3: '+eps'}],
        # name_evidence=[{0: .9}, None, None, {1: .9}]
        # name_evidence=[0, None, None, None]
        # name_evidence=[0, None, None, None]
    )
    temp_model(score_evidence=[1, 1, 1, None, None, None], **constkw)
    temp_model(score_evidence=[1, 1, 0, None, None, None], **constkw)
    temp_model(score_evidence=[1, 0, 0, None, None, None], **constkw)


def demo_modes():
    """
    Look at the last result of the different names demo under differet modes
    """
    constkw = dict(
        num_annots=4,
        num_names=8,
        score_evidence=[1, 0, 0, 0, 0, 1],
        # name_evidence=[{0: .9}, None, None, {1: .9}],
        # name_evidence=[0, None, None, 1],
        name_evidence=[0, None, None, None],
        # other_evidence={
        #    'Sad': 0,
        #    'Sab': 1,
        #    'Scd': 1,
        #    'Sac': 0,
        #    'Sbc': 0,
        #    'Sbd': 0,
        # }
    )
    # The first mode uses a hidden Match layer
    temp_model(mode=1, **constkw)
    # The second mode directly maps names to scores
    temp_model(mode=2, **constkw)
    temp_model(mode=3, noquery=True, **constkw)
    temp_model(mode=4, noquery=True, **constkw)


def demo_name_annot_complexity():
    """
    This demo is meant to show the structure of the graph as more annotations
    and names are added.

    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_name_annot_complexity --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> demo_name_annot_complexity()
        >>> ut.show_if_requested()
    """
    constkw = dict(score_evidence=[], name_evidence=[], mode=1)
    # Initially there are 2 annots and 4 names
    (model,) = temp_model(num_annots=2, num_names=4, **constkw)
    draw_tree_model(model)
    # Adding a name causes the probability of the other names to go down
    (model,) = temp_model(num_annots=2, num_names=5, **constkw)
    draw_tree_model(model)
    # Adding an annotation wihtout matches dos not effect probabilities of
    # names
    (model,) = temp_model(num_annots=3, num_names=5, **constkw)
    draw_tree_model(model)
    (model,) = temp_model(num_annots=4, num_names=10, **constkw)
    draw_tree_model(model)
    # Given A annots, the number of score nodes is (A ** 2 - A) / 2
    (model,) = temp_model(num_annots=5, num_names=5, **constkw)
    draw_tree_model(model)
    # model, = temp_model(num_annots=6, num_names=5, score_evidence=[], name_evidence=[], mode=1)
    # draw_tree_model(model)


def demo_model_idependencies():
    """
    Independences of the 3 annot 3 name model

    CommandLine:
        python -m wbia.unstable.demobayes --exec-demo_model_idependencies --mode=1 --num-names=2 --show
        python -m wbia.unstable.demobayes --exec-demo_model_idependencies --mode=2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.unstable.demobayes import *  # NOQA
        >>> result = demo_model_idependencies()
        >>> print(result)
        >>> ut.show_if_requested()
    """
    num_names = ut.get_argval('--num-names', default=3)
    model = temp_model(
        num_annots=num_names, num_names=num_names, score_evidence=[], name_evidence=[]
    )[0]
    # This model has the following independenceis
    idens = model.get_independencies()

    iden_strs = [
        ', '.join(sorted(iden.event1))
        + ' _L '
        + ','.join(sorted(iden.event2))
        + ' | '
        + ', '.join(sorted(iden.event3))
        for iden in idens.independencies
    ]
    print('general idependencies')
    print(ut.align(ut.align('\n'.join(sorted(iden_strs)), '_'), '|'))
    # ut.embed()
    # model.is_active_trail('Na', 'Nb', 'Sab')


# Might not be valid, try and collapse S and M
# xs = list(map(str, idens.independencies))
# import re
# xs = [re.sub(', M..', '', x) for x in xs]
# xs = [re.sub('M..,?', '', x) for x in xs]
# xs = [x for x in xs if not x.startswith('( _')]
# xs = [x for x in xs if not x.endswith('| )')]
# print('\n'.join(sorted(list(set(xs)))))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.unstable.demobayes
        python -m wbia.unstable.demobayes --allexamples
    """
    if ut.VERBOSE:
        print('[hs] demobayes')
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
