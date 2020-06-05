# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def prepare_annot_pairs(ibs, qaids, daids, qconfig2_, dconfig2_):
    # Prepare lazy attributes for annotations
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    unique_qaids = set(qaids)
    unique_daids = set(daids)

    # Determine a unique set of annots per config
    configured_aids = ut.ddict(set)
    configured_aids[qannot_cfg].update(unique_qaids)
    configured_aids[dannot_cfg].update(unique_daids)

    # Make efficient annot-object representation
    configured_obj_annots = {}
    for config, aids in configured_aids.items():
        annots = ibs.annots(sorted(list(aids)), config=config)
        configured_obj_annots[config] = annots.view()

    # These annot views behave like annot objects
    # but they use the same internal cache
    annots1 = configured_obj_annots[qannot_cfg].view(qaids)
    annots2 = configured_obj_annots[dannot_cfg].view(daids)
    return annots1, annots2


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


def gridsearch_ratio_thresh(matches):
    import sklearn
    import sklearn.metrics
    import vtool as vt

    # Param search for vsone
    import wbia.plottool as pt

    pt.qt4ensure()

    skf = sklearn.model_selection.StratifiedKFold(n_splits=10, random_state=119372)

    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches])

    basis = {'ratio_thresh': np.linspace(0.6, 0.7, 50).tolist()}
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

    auc_list = _ratio_thresh(y, matches)
    pt.plot(xdata, auc_list)
    subx, suby = vt.argsubmaxima(auc_list, xdata)
    best_ratio_thresh = subx[suby.argmax()]

    skf_results = []
    y_true = y
    for train_idx, test_idx in skf.split(matches, y):
        match_list_ = ut.take(matches, train_idx)
        y_true = y.take(train_idx)
        auc_list = _ratio_thresh(y_true, match_list_)
        subx, suby = vt.argsubmaxima(auc_list, xdata, maxima_thresh=0.8)
        best_ratio_thresh = subx[suby.argmax()]
        skf_results.append(best_ratio_thresh)
    print('skf_results.append = %r' % (np.mean(skf_results),))
    import utool

    utool.embed()
