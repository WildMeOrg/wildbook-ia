# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
import vtool as vt
import dtool
from six.moves import zip, range  # NOQA
print, rrr, profile = ut.inject2(__name__)


class PairSampleConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('top_gt', 4),
        ut.ParamInfo('mid_gt', 2),
        ut.ParamInfo('bot_gt', 2),
        ut.ParamInfo('rand_gt', 2),
        ut.ParamInfo('top_gf', 3),
        ut.ParamInfo('mid_gf', 2),
        ut.ParamInfo('bot_gf', 1),
        ut.ParamInfo('rand_gf', 2),
    ]


class VsOneAssignConfig(dtool.Config):
    _param_info_list = vt.matching.VSONE_ASSIGN_CONFIG


@profile
def train_pairwise_rf():
    """
    Notes:
        Measures are:

          Local:
            * LNBNN score
            * Foregroundness score
            * SIFT correspondence distance
            * SIFT normalizer distance
            * Correspondence neighbor rank
            * Nearest unique name distances
            * SVER Error

          Global:
            * Viewpoint labels
            * Quality Labels
            * Database Size
            * Number of correspondences
            % Total LNBNN Score

    CommandLine:
        python -m ibeis.algo.hots.script_vsone train_pairwise_rf
        python -m ibeis.algo.hots.script_vsone train_pairwise_rf --db PZ_MTEST --show
        python -m ibeis.algo.hots.script_vsone train_pairwise_rf --db PZ_Master1 --show
        python -m ibeis.algo.hots.script_vsone train_pairwise_rf --db GZ_Master1 --show

    Example:
        >>> from ibeis.algo.hots.script_vsone import *  # NOQA
        >>> train_pairwise_rf()
    """
    # import vtool as vt
    # import ibeis
    import sklearn
    import sklearn.metrics
    import sklearn.model_selection
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import ibeis

    # pd.options.display.max_rows = 10
    pd.options.display.max_rows = 80
    pd.options.display.max_columns = 40
    pd.options.display.width = 160

    # ut.aug_sysargv('--db PZ_MTEST')
    qreq_ = ibeis.testdata_qreq_(
        defaultdb='PZ_MTEST',
        a=':mingt=2,species=primary',
        t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum',
    )
    assert qreq_.qparams.can_match_samename is True
    assert qreq_.qparams.prescore_method == 'csum'

    hyper_params = dtool.Config.from_dict(dict(
        subsample=None,
        pair_sample=PairSampleConfig(),
        vsone_assign=VsOneAssignConfig()),
        tablename='HyperParams'
    )
    if qreq_.qparams.featweight_enabled:
        hyper_params.vsone_assign['weight'] = 'fgweight'

    data = bigcache_features(qreq_, hyper_params)
    simple_scores, X_dict, y, match = data

    X = X_dict['learn(all)']

    match.local_measures.keys()

    print('Building pairwise classifier')
    print('hist(y) = ' + ut.repr4(ut.dict_hist(y)))

    simple_scores_ = simple_scores.copy()
    if True:
        # Remove scores that arent worth reporting
        for k in list(simple_scores_.columns)[:]:
            ignore = [
                'sum(norm_x', 'sum(norm_y',
                'sum(sver_err', 'sum(scale', 'sum(match_dist)',
            ]
            if qreq_.qparams.featweight_enabled:
                ignore.extend(['sum(norm_dist)', 'sum(ratio)', 'sum(lnbnn)',
                               'sum(lnbnn_norm_dist)'])
            flags = [part in k for part in ignore]
            if any(flags):
                del simple_scores_[k]

    # Sort AUC by values
    simple_aucs = pd.DataFrame(dict([
        (k, [sklearn.metrics.roc_auc_score(y, simple_scores_[k])])
        for k in simple_scores_.columns
    ]))
    simple_auc_dict = ut.dzip(simple_aucs.columns, simple_aucs.values[0])
    simple_auc_dict = ut.sort_dict(simple_auc_dict, 'vals', reverse=True)

    # Simple printout of aucs
    # we dont need cross validation because there is no learning here
    print(ut.align(ut.repr4(simple_auc_dict, precision=8), ':'))

    # ---------------
    # Setup cross-validation

    # xvalkw = dict(n_splits=10, shuffle=True,
    xvalkw = dict(n_splits=3, shuffle=True,
                  random_state=np.random.RandomState(42))
    skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
    skf_iter = skf.split(X=np.empty(y.shape), y=y)
    df_results = pd.DataFrame(columns=list(X_dict.keys()))

    rf_params = {
        # 'max_depth': 4,
        'bootstrap': True,
        'class_weight': None,
        'max_features': 'sqrt',
        'missing_values': np.nan,
        'min_samples_leaf': 5,
        'min_samples_split': 2,
        'n_estimators': 256,
        'criterion': 'entropy',
    }
    rf_params.update(verbose=0, random_state=np.random.RandomState(3915904814))
    cv_classifiers = []
    # a RandomForestClassifier is an ensemble of DecisionTreeClassifier(s)
    prog = ut.ProgIter(list(skf_iter), label='skf')
    for count, (train_idx, test_idx) in enumerate(prog):
        y_test = y[test_idx]
        y_train = y[train_idx]
        split_columns = []
        split_aucs = []
        classifiers = {}
        # prog.ensure_newline()
        for name, X in X_dict.items():
            # Learn a random forest classifier using train data
            X_train = X.values[train_idx]
            X_test = X.values[test_idx]
            clf = RandomForestClassifier(**rf_params)
            clf.fit(X_train, y_train)
            classifiers[name] = clf
            # Evaluate using on testing data
            clf_probs = clf.predict_proba(X_test)
            auc_learn = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])
            split_columns.append(name)
            split_aucs.append(auc_learn)
        # Append this folds' classifiers
        cv_classifiers.append(classifiers)
        # Append this fold's results
        newrow = pd.DataFrame([split_aucs], columns=split_columns)
        df_results = df_results.append([newrow], ignore_index=True)

    # change = df[df.columns[2]] - df[df.columns[0]]
    # percent_change = change / df[df.columns[0]] * 100
    # df = df.assign(change=change)
    # df = df.assign(percent_change=percent_change)

    import sandbox_utools as sbut
    print(sbut.to_string_monkey(
        df_results, highlight_cols=np.arange(len(df_results.columns))))

    simple_auc_dict.values()
    simple_keys = list(simple_auc_dict.keys())
    simple_vals = list(simple_auc_dict.values())
    idxs = ut.argsort(list(simple_auc_dict.values()))[::-1][0:3]
    idx_ = simple_keys.index('score_lnbnn_1vM')
    if idx_ in idxs:
        idxs.remove(idx_)
    idxs = [idx_] + idxs
    best_simple_cols = ut.take(simple_keys, idxs)
    best_simple_aucs = ut.take(simple_vals, idxs)

    df_simple = pd.DataFrame([best_simple_aucs], columns=best_simple_cols)

    # Take mean over all classifiers
    df_mean = pd.DataFrame([df_results.mean().values], columns=df_results.columns)
    df_all = pd.concat([df_simple, df_mean], axis=1)

    # Add in the simple scores
    print(sbut.to_string_monkey(df_all, highlight_cols=np.arange(len(df_all.columns))))

    ut.qt4ensure()
    import plottool as pt

    def print_dict_ranks(dict_):
        sorted_ = ut.sort_dict(dict_, 'vals', reverse=True)
        print(ut.align(ut.repr4(sorted_, precision=4), ':'))

    for name, X in X_dict.items():
        # Take average feature importance
        feature_importances = np.mean([
            clf_.feature_importances_
            for clf_ in ut.dict_take_column(cv_classifiers, name)
        ], axis=0)
        _importances = ut.dzip(X.columns, feature_importances)
        importances = ut.map_keys(lambda k: k.replace('norm_x', 'x'), _importances)
        importances = ut.map_keys(lambda k: k.replace('norm_y', 'y'), importances)
        importances = {k: v for k, v in importances.items() if v > .01}
        # Display weight of each individual feature

        print('\n Most Important Features')
        print_dict_ranks(importances)

        # ut.fix_embed_globals()

        # Display weight of groups of features
        def group_importance(item):
            name, keys = item
            num = len(keys)
            weight = sum(ut.take(_importances, keys))
            ave_w = weight / num
            tup = ave_w, weight, num
            # return tup
            df = pd.DataFrame([tup], columns=['ave_w', 'weight', 'num'], index=[name])
            return df

        def apply_grouper(grouper):
            _keys = ut.group_items(X.columns, ut.lmap(grouper, X.columns))
            _weights = pd.concat(ut.lmap(group_importance, _keys.items()))
            _weights = _weights.iloc[_weights['ave_w'].argsort()[::-1]]
            print(_weights)

        print('\nImportance of globals-vs-locals')
        def type_grouper(key):
            if key.startswith('global'):
                return 'global'
            if key.startswith('loc'):
                return 'local'
            if any(key.startswith(p) for p in ['sum', 'mean', 'std', 'len']):
                return 'summary'
        apply_grouper(type_grouper)

        print('\nImportance of locals summaries')
        def summary_grouper(key):
            for p in ['sum', 'std', 'mean']:
                if key.startswith(p):
                    return p
        apply_grouper(summary_grouper)

        print('\nImportance of local measures')
        def local_grouper(key):
            if key.startswith('loc'):
                return key[key.find('(') + 1:-1]
        apply_grouper(local_grouper)

        print('\nImportance of local sorters')
        def sorter_grouper(key):
            if key.startswith('loc'):
                return key[key.find('[') + 1:key.find(',')]
        apply_grouper(sorter_grouper)

        print('\nImportance of local ranks')
        def sorter_grouper(key):
            if key.startswith('loc'):
                return key[key.find(',') + 1:key.find(']')]
        apply_grouper(sorter_grouper)

        pt.wordcloud(importances)
    pt.show_if_requested()

    import utool
    utool.embed()

    # print('rat_sver_rf_auc = %r' % (rat_sver_rf_auc,))
    # columns = ['Method', 'AUC']
    # data = [
    #     ['1vM-LNBNN',       vsmany_lnbnn_auc],

    #     ['1v1-LNBNN',       vsone_sver_lnbnn_auc],
    #     ['1v1-RAT',         rat_auc],
    #     ['1v1-RAT+SVER',    rat_sver_auc],
    #     ['1v1-RAT+SVER+RF', rat_sver_rf_auc],
    # ]
    # table = pd.DataFrame(data, columns=columns)
    # error = 1 - table['AUC']
    # orig = 1 - vsmany_lnbnn_auc
    # import tabulate
    # table = table.assign(percent_error_decrease=(orig - error) / orig * 100)
    # col_to_nice = {
    #     'percent_error_decrease': '% error decrease',
    # }
    # header = [col_to_nice.get(c, c) for c in table.columns]
    # print(tabulate.tabulate(table.values, header, tablefmt='orgtbl'))


def bigcache_features(qreq_, hyper_params):
    dbname = qreq_.ibs.get_dbname()
    vsmany_hashid = qreq_.get_cfgstr(hash_pipe=True, with_input=True)
    features_hashid = ut.hashstr27(vsmany_hashid + hyper_params.get_cfgstr())
    cfgstr = '_'.join(['devcache', str(dbname), features_hashid])

    cacher = ut.Cacher('pairwise_data', cfgstr=cfgstr,
                       appname='vsone_rf_train', enabled=0)
    data = cacher.tryload()
    if not data:
        data = build_features(qreq_, hyper_params)
        cacher.save(data)
    # simple_scores, X_dict, y = data
    return data


def build_features(qreq_, hyper_params):
    import pandas as pd

    # ==================================
    # Compute or load one-vs-one results
    # ==================================
    cached_data, infr = bigcache_vsone(qreq_, hyper_params)

    # =====================================
    # Attempt to train a simple classsifier
    # =====================================

    # setup truth targets
    # TODO: not-comparable
    matches = cached_data['SV_LNBNN']
    # Pass back just one match to play with
    for match in matches:
        if len(match.fm) > 10:
            break

    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches])
    # truth_list = np.array(qreq_.ibs.get_aidpair_truths(*zip(*aid_pairs)))

    # ---------------
    # Try just using simple scores
    # vt.rrrr()
    # for m in matches:
    #     m.rrr(0, reload_module=False)
    simple_scores = pd.DataFrame([
        m._make_local_summary_feature_vector(sum=True, mean=False, std=False)
        for m in matches])

    if True:
        aid_pairs = [(m.annot1['aid'], m.annot2['aid']) for m in matches]
        # TEST ORIGINAL LNBNN SCORE SEP
        infr.graph.add_edges_from(aid_pairs)
        infr.apply_match_scores()
        edge_data = [infr.graph.get_edge_data(u, v) for u, v in aid_pairs]
        lnbnn_score_list = [0 if d is None else d.get('score', 0) for d in edge_data]
        lnbnn_score_list = np.nan_to_num(lnbnn_score_list)
        simple_scores = simple_scores.assign(score_lnbnn_1vM=lnbnn_score_list)

    simple_scores[pd.isnull(simple_scores)] = 0

    # ---------------
    sorters = [
        'ratio', 'lnbnn', 'lnbnn_norm_dist', 'norm_dist', 'match_dist'
    ]
    if qreq_.qparams.featweight_enabled:
        sorters += [
            'weighted_ratio', 'weighted_lnbnn',
        ]

    # keys1 = ['match_dist', 'norm_dist', 'ratio', 'sver_err_xy',
    #          'sver_err_scale', 'sver_err_ori', u'lnbnn_norm_dist', u'lnbnn']
    # keys2 = ['match_dist', 'norm_dist', 'ratio', 'sver_err_xy',
    #          'sver_err_scale', 'sver_err_ori']
    keys1 = list(match.local_measures.keys())

    # Ignore for PZ/GZ
    keys_ignore = ['weighted_lnbnn', 'lnbnn', 'weighted_norm_dist',
                   'fgweights']
    sorters_ignore = ['match_dist', 'ratio']

    keys1 = ut.setdiff(keys1, keys_ignore)
    sorters = ut.setdiff(sorters, sorters_ignore)

    # Try different feature constructions
    print('Building pairwise features')
    pairwise_feats = pd.DataFrame([
        m.make_feature_vector(sorters=sorters, keys=keys1, sl=slice(2, 5))
        for m in ut.ProgIter(matches, label='making pairwise feats')
    ])
    pairwise_feats[pd.isnull(pairwise_feats)] = np.nan

    # pairwise_feats_ratio = pd.DataFrame([
    #     m.make_feature_vector(sorters='ratio', keys=keys2, sl=3)
    #     for m in ut.ProgIter(matches)
    # ])
    # pairwise_feats_ratio[pd.isnull(pairwise_feats)] = np.nan
    # valid_colx = np.where(np.all(pairwise_feats.notnull(), axis=0))[0]
    # valid_cols = pairwise_feats.columns[valid_colx]
    # X_nonan = pairwise_feats[valid_cols].copy()
    X_all = pairwise_feats.copy()
    # X_withnan_ratio = pairwise_feats_ratio

    X_dict = {
        'learn(all)': X_all
    }

    return simple_scores, X_dict, y, match


def vsone_(qreq_, query_aids, data_aids, qannot_cfg, dannot_cfg,
           configured_obj_annots, hyper_params):
    # Do vectorized preload before constructing lazy dicts
    # Then make sure the lazy dicts point to this subset
    unique_obj_annots = list(configured_obj_annots.values())
    for annots in ut.ProgIter(unique_obj_annots, 'vectorized preload'):
        annots.set_caching(True)
        annots.chip_size
        annots.vecs
        annots.kpts
        annots.yaw
        annots.qual
        annots.gps
        annots.time
        if qreq_.qparams.featweight_enabled:
            annots.fgweights
    # annots._internal_attrs.clear()

    # Make convinient lazy dict representations (after loading pre info)
    configured_lazy_annots = ut.ddict(dict)
    for config, annots in configured_obj_annots.items():
        annot_dict = configured_lazy_annots[config]
        for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
            annot = _annot._make_lazy_dict()
            annot_dict[_annot.aid] = annot

    unique_lazy_annots = ut.flatten(
        [x.values() for x in configured_lazy_annots.values()])

    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    for annot in ut.ProgIter(unique_lazy_annots, label='lazy flann'):
        vt.matching.ensure_metadata_flann(annot, flann_params)

    for annot in ut.ProgIter(unique_lazy_annots, 'preload kpts'):
        annot['kpts']

    for annot in ut.ProgIter(unique_lazy_annots, 'normxy'):
        annot['norm_xys'] = (vt.get_xys(annot['kpts']) /
                             np.array(annot['chip_size'])[:, None])
    for annot in ut.ProgIter(unique_lazy_annots, 'preload vecs'):
        annot['vecs']

    # Extract pairs of annot objects (with shared caches)
    lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
    lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

    # TODO: param search over grid
    #     'use_sv': [0, 1],
    #     'use_fg': [0, 1],
    #     'use_ratio_test': [0, 1],
    matches_RAT = [vt.PairwiseMatch(annot1, annot2)
                   for annot1, annot2 in zip(lazy_annots1, lazy_annots2)]

    # Construct global measurements
    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(matches_RAT, label='setup globals'):
        match.add_global_measures(global_keys)

    # Preload flann for only specific annots
    for match in ut.ProgIter(matches_RAT, label='preload FLANN'):
        match.annot1['flann']

    cfgdict = hyper_params.vsone_assign
    if qreq_.qparams.featweight_enabled:
        del cfgdict['weight']
    # Find one-vs-one matches
    # cfgdict = {'checks': 20, 'symmetric': False}
    for match in ut.ProgIter(matches_RAT, label='assign vsone'):
        match.assign(cfgdict=cfgdict)

    # gridsearch_ratio_thresh()
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_ratio_test', {
    #     'ratio_thresh': np.linspace(.6, .7, 50)
    # })
    for match in ut.ProgIter(matches_RAT, label='apply ratio thresh'):
        match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)

    # Add keypoint spatial information to local features
    for match in matches_RAT:
        key_ = 'norm_xys'
        norm_xy1 = match.annot1[key_].take(match.fm.T[0], axis=1)
        norm_xy2 = match.annot2[key_].take(match.fm.T[1], axis=1)
        match.local_measures['norm_x1'] = norm_xy1[0]
        match.local_measures['norm_y1'] = norm_xy1[1]
        match.local_measures['norm_x2'] = norm_xy2[0]
        match.local_measures['norm_y2'] = norm_xy2[1]

        match.local_measures['scale1'] = vt.get_scales(
            match.annot1['kpts'].take(match.fm.T[0], axis=0))
        match.local_measures['scale2'] = vt.get_scales(
            match.annot2['kpts'].take(match.fm.T[1], axis=0))

    # TODO gridsearch over sv params
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_sver', {
    #     'xy_thresh': np.linspace(0, 1, 3)
    # })
    matches_RAT_SV = [
        match.apply_sver(inplace=False)
        for match in ut.ProgIter(matches_RAT, label='sver')
    ]

    # Create another version where we find global normalizers for the data
    qreq_.load_indexer()
    matches_SV_LNBNN = batch_apply_lnbnn(matches_RAT_SV, qreq_)

    if 'weight' in cfgdict:
        for match in matches_SV_LNBNN[::-1]:
            lnbnn_dist = match.local_measures['lnbnn']
            ndist = match.local_measures['lnbnn_norm_dist']
            weights = match.local_measures[cfgdict['weight']]
            match.local_measures['weighted_lnbnn'] = weights * lnbnn_dist
            match.local_measures['weighted_lnbnn_norm_dist'] = weights * ndist
            match.fs = match.local_measures['weighted_lnbnn']

    cached_data = {
        'RAT': matches_RAT,
        'RAT_SV': matches_RAT_SV,
        'SV_LNBNN': matches_SV_LNBNN,
    }
    return cached_data


def batch_apply_lnbnn(matches, qreq_):
    from ibeis.algo.hots import nn_weights
    indexer = qreq_.indexer
    matches_ = [match.copy() for match in matches]
    K = qreq_.qparams.K
    Knorm = qreq_.qparams.Knorm
    normalizer_rule  = qreq_.qparams.normalizer_rule

    print('Stacking vecs for batch matching')
    offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
    stacked_vecs = np.vstack([
        match_.matched_vecs2()
        for match_ in ut.ProgIter(matches_, lablel='stacking matched vecs')
    ])

    vecs = stacked_vecs
    num = (K + Knorm)
    idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192,
                                    label='lnbnn scoring')

    idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
    dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
    iter_ = zip(matches_, idx_list, dist_list)
    prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
    for match_, neighb_idx, neighb_dist in prog:
        qaid = match_.annot2['aid']
        norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
        ndist = vt.take_col_per_row(neighb_dist, norm_k)
        vdist = match_.local_measures['match_dist']
        lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
        lnbnn_dist = np.clip(lnbnn_dist, 0, np.inf)
        match_.local_measures['lnbnn_norm_dist'] = ndist
        match_.local_measures['lnbnn'] = lnbnn_dist
        match_.fs = lnbnn_dist
    return matches_


def bigcache_vsone(qreq_, hyper_params):
    """
    Cached output of one-vs-one matches
    """
    import vtool as vt
    import ibeis
    # Get a set of training pairs
    ibs = qreq_.ibs
    cm_list = qreq_.execute()
    infr = ibeis.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)

    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(rng=np.random.RandomState(42),
                                         **hyper_params.pair_sample)
    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_), directed=False).tolist()
    # TODO: handle non-comparability

    # ======================================
    # Compute one-vs-one scores and local_measures
    # ======================================

    # Prepare lazy attributes for annotations
    qreq_ = infr.qreq_
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    # Remove any pairs missing features
    if dannot_cfg == qannot_cfg:
        unique_annots = ibs.annots(np.unique(np.array(aid_pairs_)), config=dannot_cfg)
        bad_aids = unique_annots.compress(~np.array(unique_annots.num_feats) > 0).aids
        bad_aids = set(bad_aids)
    else:
        annots1_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 0)), config=qannot_cfg)
        annots2_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 1)), config=dannot_cfg)
        bad_aids1 = annots1_.compress(~np.array(annots1_.num_feats) > 0).aids
        bad_aids2 = annots2_.compress(~np.array(annots2_.num_feats) > 0).aids
        bad_aids = set(bad_aids1 + bad_aids2)
    subset_idxs = np.where([not (a1 in bad_aids or a2 in bad_aids)
                            for a1, a2 in aid_pairs_])[0]
    # Keep only a random subset
    if hyper_params.subsample:
        rng = np.random.RandomState(3104855634)
        num_max = hyper_params.subsample
        if num_max < len(subset_idxs):
            subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
            subset_idxs = sorted(subset_idxs)

    # Take the current selection
    aid_pairs = ut.take(aid_pairs_, subset_idxs)
    query_aids = ut.take_column(aid_pairs, 0)
    data_aids = ut.take_column(aid_pairs, 1)

    # Determine a unique set of annots per config
    configured_aids = ut.ddict(set)
    configured_aids[qannot_cfg].update(query_aids)
    configured_aids[dannot_cfg].update(data_aids)

    # Make efficient annot-object representation
    configured_obj_annots = {}
    for config, aids in configured_aids.items():
        annots = ibs.annots(sorted(list(aids)), config=config)
        configured_obj_annots[config] = annots

    annots1 = configured_obj_annots[qannot_cfg].loc(query_aids)
    annots2 = configured_obj_annots[dannot_cfg].loc(data_aids)

    # Get hash based on visual annotation appearence of each pair
    # as well as algorithm configurations used to compute those properties
    qvuuids = annots1.visual_uuids
    dvuuids = annots2.visual_uuids
    qcfgstr = annots1._config.get_cfgstr()
    dcfgstr = annots2._config.get_cfgstr()
    annots_cfgstr = ut.hashstr27(qcfgstr) + ut.hashstr27(dcfgstr)
    vsone_uuids = [
        ut.combine_uuids(uuids, salt=annots_cfgstr)
        for uuids in ut.ProgIter(zip(qvuuids, dvuuids), length=len(qvuuids),
                                 label='hashing ids')
    ]

    # Combine into a big cache for the entire 1-v-1 matching run
    big_uuid = ut.hashstr_arr27(vsone_uuids, '', pathsafe=True)
    cacher = ut.Cacher('vsone', cfgstr=str(big_uuid), appname='vsone_rf_train')

    cached_data = cacher.tryload()
    if cached_data is not None:
        # Caching doesn't work 100% for PairwiseMatch object, so we need to do
        # some postprocessing
        configured_lazy_annots = ut.ddict(dict)
        for config, annots in configured_obj_annots.items():
            annot_dict = configured_lazy_annots[config]
            for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
                annot_dict[_annot.aid] = _annot._make_lazy_dict()

        # Extract pairs of annot objects (with shared caches)
        lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
        lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

        # Create a set of PairwiseMatches with the correct annot properties
        matches = [vt.PairwiseMatch(annot1, annot2)
                   for annot1, annot2 in zip(lazy_annots1, lazy_annots2)]

        # Updating a new matches dictionary ensure the annot1/annot2 properties
        # are set correctly
        for key, cached_matches in list(cached_data.items()):
            fixed_matches = [match.copy() for match in matches]
            for fixed, internal in zip(fixed_matches, cached_matches):
                dict_ = internal.__dict__
                ut.delete_dict_keys(dict_, ['annot1', 'annot2'])
                fixed.__dict__.update(dict_)
            cached_data[key] = fixed_matches
    else:
        cached_data = vsone_(qreq_, query_aids, data_aids, qannot_cfg,
                             dannot_cfg, configured_obj_annots, hyper_params)
        cacher.save(cached_data)
    return cached_data, infr


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.script_vsone
        python -m ibeis.algo.hots.script_vsone --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
