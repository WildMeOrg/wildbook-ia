# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, range  # NOQA
print, rrr, profile = ut.inject2(__name__)


def request_pairwise_matches():
    pass


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

    Example:
        >>> from ibeis.algo.hots.script_vsone import *  # NOQA
        >>> train_pairwise_rf()
    """
    import vtool as vt
    import ibeis
    import sklearn
    import sklearn.metrics
    import sklearn.model_selection
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    # ibs = ibeis.opendb('PZ_MTEST')
    ibs = ibeis.opendb('PZ_Master1')
    # ibs = ibeis.opendb('GZ_Master1')

    aids = ibeis.testdata_aids(a=':mingt=2,species=primary', ibs=ibs)

    # ===========================
    # Get a set of training pairs
    # ===========================
    infr = ibeis.AnnotInference(ibs, aids, autoinit=True)

    infr.exec_matching(cfgdict={
        'can_match_samename': True,
        'K': 4,
        'Knorm': 1,
        'prescore_method': 'csum',
        'score_method': 'csum'
    })
    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(top_gt=4, top_gf=3, rand_gf=2,
                                         rng=np.random.RandomState(42))
    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_), directed=False).tolist()
    query_aids_ = ut.take_column(aid_pairs_, 0)
    data_aids_ = ut.take_column(aid_pairs_, 1)

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
    configured_annot_dict = ut.ddict(dict)
    bad_aids = []
    config_aids_pairs = [(qannot_cfg, query_aids_), (dannot_cfg, data_aids_)]
    for config, aids in ut.ProgIter(config_aids_pairs, label='prepare annots', bs=False):
        annots = ibs.annots(aids, config=config)
        hasfeats = np.array(annots.num_feats) > 0
        aids = annots.compress(hasfeats).aids
        bad_aids.extend(annots.compress(~hasfeats).aids)
        annot_dict = configured_annot_dict[config]
        unique_aids = ut.unique(aids)
        unique_aids = ut.setdiff(unique_aids, annot_dict.keys())
        unique_annots = ibs.annots(unique_aids, config=config)
        for _annot in ut.ProgIter(unique_annots.scalars(), label='unique'):
            annot = _annot._make_lazy_dict()
            flann_params = {'algorithm': 'kdtree', 'trees': 4}
            vt.matching.ensure_metadata_flann(annot, flann_params)
            annot_dict[_annot.aid] = annot

    isgood = [not (a1 in bad_aids or a2 in bad_aids) for a1, a2 in aid_pairs_]
    query_aids = ut.compress(query_aids_, isgood)
    data_aids = ut.compress(data_aids_, isgood)
    aid_pairs = ut.compress(aid_pairs_, isgood)

    # Extract pairs of annot objects (with shared caches)
    annot1_list = ut.take(configured_annot_dict[qannot_cfg], query_aids)
    annot2_list = ut.take(configured_annot_dict[dannot_cfg], data_aids)
    # truth_list = np.array(qreq_.ibs.get_aidpair_truths(*zip(*aid_pairs)))

    def matches_auc(truth_list, match_list, lbl=''):
        score_list = np.array([m.fs.sum() for m in match_list])
        auc = sklearn.metrics.roc_auc_score(truth_list, score_list)
        print('%s auc = %r' % (lbl, auc,))
        return auc

    verbose = True  # NOQA

    # TODO: Cache output of one-vs-one matches
    qvuuids = ibs.get_annot_visual_uuids(query_aids)
    dvuuids = ibs.get_annot_visual_uuids(data_aids)
    qcfgstr = qannot_cfg.get_cfgstr()
    dcfgstr = dannot_cfg.get_cfgstr()
    pair_vuuids = [ut.combine_uuids([q, d]) for q, d in zip(qvuuids, dvuuids)]
    pipe_hashid = qreq_.get_pipe_hashid()
    annots_cfgstr = ut.hashstr27(qcfgstr) + ut.hashstr27(dcfgstr)
    big_uuid = ut.combine_uuids(pair_vuuids, salt=annots_cfgstr + pipe_hashid)
    cacher = ut.Cacher('pairwise-matches-', cfgstr=str(big_uuid))

    cached_data = cacher.tryload()
    if cached_data is not None:

        matches = [vt.PairwiseMatch(annot1, annot2)
                   for annot1, annot2 in zip(annot1_list, annot2_list)]

        for m in ut.ProgIter(matches):
            ut.reload_class(m, verbose=False, reload_module=False)

        matches_RAT = [match.copy() for match in matches]
        matches_RAT_SV = [match.copy() for match in matches]
        matches_SV_LNBNN = [match.copy() for match in matches]
        for match, internal in zip(matches_RAT, cached_data['RAT']):
            match.__dict__.update(internal.__dict__)
        for match, internal in zip(matches_RAT_SV, cached_data['RAT_SV']):
            match.__dict__.update(internal.__dict__)
        for match, internal in zip(matches_SV_LNBNN, cached_data['SV_LNBNN']):
            match.__dict__.update(internal.__dict__)
    else:
        unique_lazy_annots = ut.flatten([x.values() for x in configured_annot_dict.values()])
        for annot in ut.ProgIter(unique_lazy_annots, 'preload kpts'):
            annot['kpts']
        for annot in ut.ProgIter(unique_lazy_annots, 'normxy'):
            annot['norm_xys'] = (vt.get_xys(annot['kpts']) /
                                 np.array(annot['chip_size'])[:, None])
        for annot in ut.ProgIter(unique_lazy_annots, 'preload vecs'):
            annot['vecs']

        for annot in ut.ProgIter(unique_lazy_annots, 'fixup'):
            # annot['rchip'] = annot.getitem('chips', is_eager=False)
            # annot['dlen_sqrd'] = annot.getitem('chip_dlensqrd', is_eager=False)
            annot['rchip_fpath'] = annot.getitem('chip_fpath', is_eager=False)

        matches_RAT = [vt.PairwiseMatch(annot1, annot2)
                       for annot1, annot2 in zip(annot1_list, annot2_list)]

        # Construct global measurements
        global_keys = ['yaw', 'qual', 'gps', 'time']
        for match in ut.ProgIter(matches_RAT, label='setup globals'):
            match.add_global_measures(global_keys)

        # Preload flann for only specific annots
        for match in ut.ProgIter(matches_RAT, label='preload FLANN'):
            match.annot1['flann']

        # Find one-vs-one matches
        cfgdict = {'checks': 20}
        for match in ut.ProgIter(matches_RAT, label='assign vsone'):
            match.assign(cfgdict)

        for match in ut.ProgIter(matches_RAT, label='apply ratio thresh'):
            match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)

        for match in matches_RAT:
            key_ = 'norm_xys'
            norm_xy1 = match.annot1[key_].take(match.fm.T[0], axis=1)
            norm_xy2 = match.annot2[key_].take(match.fm.T[1], axis=1)
            match.local_measures['norm_x1'] = norm_xy1[0]
            match.local_measures['norm_y1'] = norm_xy1[1]
            match.local_measures['norm_x2'] = norm_xy2[0]
            match.local_measures['norm_y2'] = norm_xy2[1]

        # gridsearch_ratio_thresh()

        matches_RAT_SV = [match.apply_sver(inplace=False)
                          for match in ut.ProgIter(matches_RAT, label='sver')]

        # Create another version where we find global normalizers for the data
        qreq_.load_indexer()
        indexer = qreq_.indexer

        def batch_apply_lnbnn(matches):
            # from ibeis.algo.hots import nn_weights
            matches_ = [match.copy() for match in matches]
            # matches_ = [match.rrr(0) for match in matches_]
            K = qreq_.qparams.K
            Knorm = qreq_.qparams.Knorm
            # normalizer_rule  = qreq_.qparams.normalizer_rule
            print('Stacking vecs for batch matching')
            offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
            stacked_vecs = np.vstack([
                match_.matched_vecs2()
                for match_ in ut.ProgIter(matches_, lablel='stacking matched vecs')
            ])
            # vecs_list2 = [stacked_vecs[l:r] for l, r in ut.itertwo(offset_list)]

            vecs = stacked_vecs
            num = (K + Knorm)
            # num = 2000
            idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192, label='lnbnn scoring')

            vdist = np.hstack([
                match_.local_measures.get('match_dist', [])
                for match_ in ut.ProgIter(matches_, lablel='stacking dist')
            ])
            ndist = dists.T[-1]
            lnbnn = ndist - vdist
            lnbnn = np.clip(lnbnn, 0, np.inf)

            ndist_list = [ndist[l:r] for l, r in ut.itertwo(offset_list)]
            lnbnn_list = [lnbnn[l:r] for l, r in ut.itertwo(offset_list)]

            for match_, ndist_, lnbnn_ in zip(matches_, ndist_list, lnbnn_list):
                match_.local_measures['lnbnn_norm_dist'] = ndist_
                match_.local_measures['lnbnn'] = lnbnn_
                match_.fs = lnbnn_

            # idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
            # dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
            # iter_ = zip(matches_, idx_list, dist_list)
            # prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
            # for match_, neighb_idx, neighb_dist in prog:
            #     qaid = match_.annot2['aid']
            #     norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
            #     ndist = vt.take_col_per_row(neighb_dist, norm_k)
            #     vdist = match_.local_measures['match_dist']
            #     lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
            #     # lnbnn_dist = np.clip(lnbnn_dist, 0, np.inf)
            #     match_.local_measures['lnbnn_norm_dist'] = ndist
            #     match_.local_measures['lnbnn'] = lnbnn_dist
            #     match_.fs = lnbnn_dist
            return matches_

        matches = matches_RAT_SV  # NOQA
        matches_SV_LNBNN = batch_apply_lnbnn(matches_RAT_SV)

        if False:
            for m in ut.ProgIter(matches_RAT):
                ut.reload_class(m, verbose=False, reload_module=False)
            for m in ut.ProgIter(matches_RAT_SV):
                ut.reload_class(m, verbose=False, reload_module=False)
            for m in ut.ProgIter(matches_SV_LNBNN):
                ut.reload_class(m, verbose=False, reload_module=False)

        cached_data = {
            'RAT': matches_RAT,
            'RAT_SV': matches_RAT_SV,
            'SV_LNBNN': matches_SV_LNBNN,
        }
        cacher.save(cached_data)

    matches = matches_SV_LNBNN

    # =====================================
    # Attempt to train a simple classsifier
    # =====================================

    # ---------------
    # Try just using simple scores
    S_names = ['ratio', 'lnbnn']
    S_sets = [np.array([m.local_measures[key].sum() for m in matches])
              for key in S_names ]
    S_names += ['vsmany_lnbnn']

    # TEST ORIGINAL LNBNN SCORE SEP
    infr.apply_match_edges()
    infr.apply_match_scores()
    edge_data = [infr.graph.get_edge_data(u, v) for u, v in aid_pairs]
    lnbnn_score_list = [0 if d is None else d.get('score', 0) for d in edge_data]
    S_sets += [lnbnn_score_list]

    # ---------------
    scorers = ['ratio', 'lnbnn', 'lnbnn_norm_dist', 'norm_dist', 'match_dist']
    keys1 = ['match_dist', 'norm_dist', 'ratio', 'sver_err_xy',
             'sver_err_scale', 'sver_err_ori', u'lnbnn_norm_dist', u'lnbnn']
    keys2 = ['match_dist', 'norm_dist', 'ratio', 'sver_err_xy',
             'sver_err_scale', 'sver_err_ori']

    # Try different feature constructions
    print('Building pairwise features')
    pairwise_feats = pd.DataFrame([
        m.make_feature_vector(scorers=scorers, keys=keys1, n_top=3)
        for m in ut.ProgIter(matches)
    ])
    pairwise_feats[pd.isnull(pairwise_feats)] = np.nan

    pairwise_feats_ratio = pd.DataFrame([
        m.make_feature_vector(scorers='ratio', keys=keys2, n_top=3)
        for m in ut.ProgIter(matches)
    ])
    pairwise_feats_ratio[pd.isnull(pairwise_feats)] = np.nan

    valid_colx = np.where(np.all(pairwise_feats.notnull(), axis=0))[0]
    valid_cols = pairwise_feats.columns[valid_colx]

    X_nonan = pairwise_feats[valid_cols].copy()
    X_withnan = pairwise_feats.copy()
    X_withnan_ratio = pairwise_feats_ratio

    X_sets = [X_nonan, X_withnan, X_withnan_ratio]
    X_names = ['nonan', 'withnan', 'withnan_ratio']

    # ---------------
    # Setup target data and cross-validation
    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches])

    # xvalkw = dict(n_splits=10, shuffle=True,
    xvalkw = dict(n_splits=3, shuffle=True,
                  random_state=np.random.RandomState(42))
    skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
    skf_iter = skf.split(X=X_nonan, y=y)
    df_results = pd.DataFrame(columns=S_names + X_names)

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
    rf_params.update(verbose=1, random_state=np.random.RandomState(3915904814))

    # a RandomForestClassifier is an ensemble of DecisionTreeClassifier(s)
    for count, (train_idx, test_idx) in enumerate(skf_iter):
        y_test = y[test_idx]
        y_train = y[train_idx]

        split_columns = []
        split_aucs = []

        for S, name in zip(S_sets, S_names):
            print('name = %r' % (name,))
            score_list = ut.take(S, test_idx)
            auc_score = sklearn.metrics.roc_auc_score(y_test, score_list)
            split_columns.append(name)
            split_aucs.append(auc_score)

        for X, name in zip(X_sets, X_names):
            print('name = %r' % (name,))
            X_train = X.values[train_idx]
            X_test = X.values[test_idx]
            # Train uncalibrated random forest classifier on train data
            clf = RandomForestClassifier(**rf_params)
            clf.fit(X_train, y_train)

            # importances = dict(zip(withnan_cols, clf.feature_importances_))
            # importances = ut.sort_dict(importances, 'vals', reverse=True)
            # print(ut.align(ut.repr4(importances, precision=4), ':'))

            # evaluate on test data
            clf_probs = clf.predict_proba(X_test)
            # log_loss = sklearn.metrics.log_loss(y_test, clf_probs)

            # evaluate on test data
            clf_probs = clf.predict_proba(X_test)
            auc_learn = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])
            split_columns.append(name)
            split_aucs.append(auc_learn)

        newrow = pd.DataFrame([split_aucs], columns=split_columns)
        # print(newrow)
        df_results = df_results.append([newrow], ignore_index=True)

        df = df_results
        change = df[df.columns[2]] - df[df.columns[0]]
        percent_change = change / df[df.columns[0]] * 100
        df = df.assign(change=change)
        df = df.assign(percent_change=percent_change)

    import sandbox_utools as sbut
    print(sbut.to_string_monkey(df, highlight_cols=list(range(len(df.columns) - 2))))
    print(df.mean())

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


def gridsearch_ratio_thresh(match_list, truth_list):
    import vtool as vt
    # Param search for vsone
    import plottool as pt
    pt.qt4ensure()

    import sklearn
    import sklearn.metrics
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10,
                                                  random_state=119372)

    basis = {'ratio_thresh': np.linspace(.6, .7, 50).tolist()}
    grid = ut.all_dict_combinations(basis)
    xdata = np.array(ut.take_column(grid, 'ratio_thresh'))

    def _ratio_thresh(y_true, match_list):
        # Try and find optional ratio threshold
        auc_list = []
        for cfgdict in ut.ProgIter(grid, lbl='gridsearch'):
            y_score = [
                match.fs.compress(match.ratio_test_flags(cfgdict)).sum()
                for match in match_list
            ]
            auc = sklearn.metrics.roc_auc_score(y_true, y_score)
            auc_list.append(auc)
        auc_list = np.array(auc_list)
        return auc_list

    auc_list = _ratio_thresh(truth_list, match_list)
    pt.plot(xdata, auc_list)
    subx, suby = vt.argsubmaxima(auc_list, xdata)
    best_ratio_thresh = subx[suby.argmax()]

    skf_results = []
    y_true = truth_list
    for train_idx, test_idx in skf.split(match_list, truth_list):
        match_list_ = ut.take(match_list, train_idx)
        y_true = truth_list.take(train_idx)
        auc_list = _ratio_thresh(y_true, match_list_)
        subx, suby = vt.argsubmaxima(auc_list, xdata, maxima_thresh=.8)
        best_ratio_thresh = subx[suby.argmax()]
        skf_results.append(best_ratio_thresh)
    print('skf_results.append = %r' % (np.mean(skf_results),))


# def old_vsone_parts():
#     if False:
#         matchesORIG = match_list
#         matches_auc(truth_list, matchesORIG)

#         matches_SV = [match.apply_sver(inplace=False)
#                       for match in ut.ProgIter(matchesORIG, label='sver')]
#         matches_auc(truth_list, matches_SV)

#         matches_RAT = [match.apply_ratio_test(inplace=False)
#                        for match in ut.ProgIter(matchesORIG, label='ratio')]
#         matches_auc(truth_list, matches_RAT)

#         matches_RAT_SV = [match.apply_sver(inplace=False)
#                           for match in ut.ProgIter(matches_RAT, label='sver')]
#         matches_auc(truth_list, matches_RAT_SV)

#     if True:
#         matches_RAT = match_list
#         matches_auc(truth_list, matches_RAT)

#         matches_RAT_SV = [match.apply_sver(inplace=False)
#                           for match in ut.ProgIter(matches_RAT, label='sver')]
#         matches_auc(truth_list, matches_RAT_SV)

#     if False:
#         # Visualize scores
#         score_list = np.array([m.fs.sum() for m in matches_RAT_SV])
#         encoder = vt.ScoreNormalizer()
#         encoder.fit(score_list, truth_list, verbose=True)
#         encoder.visualize()

#     # Fix issue
#     # for match in ut.ProgIter(matches_RAT_SV):
#     #     match.annot1['yaw'] = ibs.get_annot_yaws_asfloat(match.annot1['aid'])
#     #     match.annot2['yaw'] = ibs.get_annot_yaws_asfloat(match.annot2['aid'])
#     # # Construct global measurements
#     # global_keys = ['yaw', 'qual', 'gps', 'time']
#     # for match in ut.ProgIter(match_list, lbl='setup globals'):
#     #     match.global_measures = {}
#     #     for key in global_keys:
#     #         match.global_measures[key] = (match.annot1[key], match.annot2[key])

#     if False:
#         # TEST LNBNN SCORE SEP
#         infr.apply_match_edges()
#         infr.apply_match_scores()
#         edge_to_score = infr.get_edge_attrs('score')

#         lnbnn_score_list = [
#             edge_to_score.get(tup) if tup in edge_to_score
#             else edge_to_score.get(tup[::-1], 0)
#             for tup in ut.lmap(tuple, aid_pairs)
#         ]
#         auc = sklearn.metrics.roc_auc_score(truth_list, lnbnn_score_list)
#         print('auc = %r' % (auc,))

#     if False:
#         nfeats = len(withnan_cols)  # NOQA
#         param_grid = {
#             'bootstrap': [True, False],
#             # 'class_weight': ['balanced', None],
#             # 'criterion': ['gini', 'entropy'],
#             # 'max_depth': [2, 4],
#             'max_features': [int(np.log2(nfeats)), int(np.sqrt(nfeats)), int(np.sqrt(nfeats)) * 2, nfeats],
#             # 'max_features': [1, 3, 10],
#             # 'min_samples_split': [2, 3, 5, 10],
#             # 'min_samples_leaf': [1, 3, 5, 10, 20],
#             # 'n_estimators': [128, 256],
#         }
#         static_params = {
#             'max_depth': 4,
#             # 'bootstrap': False,
#             'class_weight': None,
#             'max_features': 'sqrt',
#             'missing_values': np.nan,
#             'min_samples_leaf': 5,
#             'min_samples_split': 2,
#             'n_estimators': 256,
#             'criterion': 'entropy',
#         }

#         from sklearn.model_selection import GridSearchCV
#         clf = RandomForestClassifier(**static_params)
#         search = GridSearchCV(clf, param_grid=param_grid, n_jobs=4, cv=3,
#                               refit=False, verbose=5)

#         with ut.Timer('GridSearch'):
#             search.fit(X_withnan, y)

#         def report(results, n_top=3):
#             for i in range(1, n_top + 1):
#                 candidates = np.flatnonzero(results['rank_test_score'] == i)
#                 for candidate in candidates:
#                     print('Model with rank: {0}'.format(i))
#                     print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
#                           results['mean_test_score'][candidate],
#                           results['std_test_score'][candidate]))
#                     print('Parameters: {0}'.format(results['params'][candidate]))
#                     print('')

#         results = search.cv_results_
#         report(results, n_top=10)

#         print(ut.sort_dict(search.cv_results_).keys())

#         params = results['params']
#         cols = sorted(param_grid.keys())
#         zX_df = pd.DataFrame([ut.take(p, cols)  for p in params], columns=cols)
#         # zX_df['class_weight'][pd.isnull(zX_df['class_weight'])] = 'none'
#         if 'max_depth' in zX_df.columns:
#             zX_df['max_depth'][pd.isnull(zX_df['max_depth'])] = 10
#         if 'criterion' in zX_df.columns:
#             zX_df['criterion'][zX_df['criterion'] == 'entropy'] = 0
#             zX_df['criterion'][zX_df['criterion'] == 'gini'] = 1
#         if 'class_weight' in zX_df.columns:
#             zX_df['class_weight'][pd.isnull(zX_df['class_weight'])] = 0
#             zX_df['class_weight'][zX_df['class_weight'] == 'balanced'] = 1
#         [(c, zX_df[c].dtype) for c in cols]

#         # zX = pd.get_dummies(zX_df).values.astype(np.float32)
#         zX = zX_df.values.astype(np.float32)
#         zY = mean_test_score = results['mean_test_score']

#         from scipy.stats import mode

#         # from pgmpy.factors.discrete import TabularCPD
#         # TabularCPD('feat', top_feats.shape[0])

#         num_top = 5
#         top_feats = zX.take(zY.argsort()[::-1], axis=0)[0:num_top]
#         print('num_top = %r' % (num_top,))

#         print('Marginalized probabilities over top feature values')
#         uvals = [np.unique(f) for f in top_feats.T]
#         marginal_probs = [[np.sum(f == u) / len(f) for u in us] for us, f in zip(uvals , top_feats.T)]
#         for c, us, mprobs in zip(cols, uvals, marginal_probs):
#             print(c + ' = ' + ut.repr3(ut.dzip(us, mprobs), precision=2))

#         mode_top_zX_ = mode(top_feats, axis=0)
#         mode_top_zX = mode_top_zX_.mode[0]
#         flags = (mode_top_zX_.count == 1)[0]
#         mode_top_zX[flags] = top_feats[0][flags]
#         print('mode')
#         print(ut.repr4(ut.dzip(cols, mode_top_zX)))
#         mean_top_zX = np.mean(top_feats, axis=0)
#         print('mean')
#         print(ut.repr4(ut.dzip(cols, mean_top_zX)))

#         import sklearn.ensemble
#         clf = sklearn.ensemble.RandomForestRegressor(bootstrap=True, oob_score=True)
#         clf.fit(zX, zY)

#         importances = dict(zip(cols, clf.feature_importances_))
#         importances = ut.sort_dict(importances, 'vals', reverse=True)
#         print(ut.align(ut.repr4(importances, precision=4), ':'))

#         mean_test_score

#     # print(df.to_string())

#     # print(df_results)

#     # TODO: TSNE?
#     # http://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
#     # Perform t-distributed stochastic neighbor embedding.
#     # from sklearn import manifold
#     # import matplotlib.pyplot as plt
#     # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#     # trans_data = tsne.fit_transform(feats).T
#     # ax = fig.add_subplot(2, 5, 10)
#     # plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
#     # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
#     # ax.xaxis.set_major_formatter(NullFormatter())
#     # ax.yaxis.set_major_formatter(NullFormatter())
#     # plt.axis('tight')
#     print('--------')


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
