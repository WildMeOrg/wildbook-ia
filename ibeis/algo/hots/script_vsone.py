# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, range  # NOQA
print, rrr, profile = ut.inject2(__name__)


def train_pairwise_rf():
    """
    Notes:
        Meausres are:

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
    # ibs = ibeis.opendb('PZ_MTEST')
    ibs = ibeis.opendb('PZ_Master1')
    aids = ibeis.testdata_aids(a=':mingt=2', ibs=ibs)

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
    aid_pairs = infr._cm_training_pairs(top_gt=4, top_gf=3, rand_gf=2)
    aid_pairs = vt.unique_rows(np.array(aid_pairs), directed=False).tolist()
    query_aids = ut.take_column(aid_pairs, 0)
    data_aids = ut.take_column(aid_pairs, 1)

    # ======================================
    # Compute one-vs-one scores and measures
    # ======================================

    # Prepare lazy attributes for annotations
    qreq_ = infr.qreq_
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)
    configured_annot_dict = ut.ddict(dict)
    config_aids_pairs = [(qannot_cfg, query_aids), (dannot_cfg, data_aids)]
    for config, aids in ut.ProgIter(config_aids_pairs, lbl='prepare annots'):
        annot_dict = configured_annot_dict[config]
        for aid in ut.unique(aids):
            if aid not in annot_dict:
                annot = ibs.get_annot_lazy_dict(aid, config)
                flann_params = {'algorithm': 'kdtree', 'trees': 4}
                vt.matching.ensure_metadata_flann(annot, flann_params)
                annot_dict[aid] = annot
                annot['yaw'] = ibs.get_annot_yaw_texts(aid)
                annot['qual'] = ibs.get_annot_qualities(aid)
                annot['gps'] = ibs.get_annot_image_gps2(aid)
                annot['time'] = ibs.get_annot_image_unixtimes_asfloat(aid)
                del annot['annot_context_options']

    # Extract pairs of annot objects (with shared caches)
    annot1_list = ut.take(configured_annot_dict[qannot_cfg], query_aids)
    annot2_list = ut.take(configured_annot_dict[dannot_cfg], data_aids)
    truth_list = np.array(qreq_.ibs.get_aidpair_truths(*zip(*aid_pairs)))

    verbose = True  # NOQA

    match_list = [vt.PairwiseMatch(annot1, annot2)
                  for annot1, annot2 in zip(annot1_list, annot2_list)]

    # Construct global measurements
    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(match_list, lbl='setup globals'):
        match.global_measures = {}
        for key in global_keys:
            match.global_measures[key] = (match.annot1[key], match.annot2[key])

    # Preload needed attributes
    for match in ut.ProgIter(match_list, lbl='preload'):
        match.annot1['flann']
        match.annot2['vecs']

    # Find one-vs-one matches
    cfgdict = {'checks': 20}
    for match in ut.ProgIter(match_list, lbl='assign vsone'):
        match.assign(cfgdict)

    for match in ut.ProgIter(match_list, lbl='assign vsone'):
        match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)

    # =====================================
    # Use scores as a baseline classifier
    # =====================================

    # Visualize scores
    score_list = np.array([m.fs.sum() for m in match_list])
    encoder = vt.ScoreNormalizer()
    encoder.fit(score_list, truth_list, verbose=True)
    encoder.visualize()

    import sklearn
    import sklearn.metrics
    # gridsearch_ratio_thresh()

    def matches_auc(match_list):
        score_list = np.array([m.fs.sum() for m in match_list])
        auc = sklearn.metrics.roc_auc_score(truth_list, score_list)
        print('auc = %r' % (auc,))
        return auc

    matchesORIG = match_list
    matches_auc(matchesORIG)

    matches_SV = [match.apply_sver(inplace=False)
                  for match in ut.ProgIter(matchesORIG, lbl='sver')]
    matches_auc(matches_SV)

    matches_RAT = [match.apply_ratio_test(inplace=False)
                   for match in ut.ProgIter(matchesORIG, lbl='ratio')]
    matches_auc(matches_RAT)

    matches_RAT_SV = [match.apply_sver(inplace=False)
                      for match in ut.ProgIter(matches_RAT, lbl='sver')]
    matches_auc(matches_RAT_SV)

    # =====================================
    # Attempt to train a simple classsifier
    # =====================================

    import pandas as pd
    import sklearn.model_selection
    from sklearn.ensemble import RandomForestClassifier

    allow_nan = True
    pairwise_feats = pd.DataFrame([m.make_pairwise_constlen_feature('ratio')
                                   for m in matches_RAT_SV])
    pairwise_feats[pd.isnull(pairwise_feats)] = np.nan

    if allow_nan:
        X_withnan = pairwise_feats.values.copy()
    valid_colx = np.where(np.all(pairwise_feats.notnull(), axis=0))[0]
    valid_cols = pairwise_feats.columns[valid_colx]
    X_nonan = pairwise_feats[valid_cols].values.copy()

    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches_RAT_SV])
    # import utool
    # utool.embed()
    rng = np.random.RandomState(42)

    for seed in (rng.rand(5) * 4294967295).astype(np.int):
        print('seed = %r' % (seed,))
        rng = np.random.RandomState(42)
        xvalkw = dict(n_folds=10, shuffle=True, random_state=rng)
        skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
        skf_iter = skf.split(X=X_nonan, y=y)
        df_results = pd.DataFrame(columns=['auc_naive', 'auc_learn_nonan',
                                           'auc_learn_withnan'])

        rng2 = np.random.RandomState(seed)
        # rf_params = dict(n_estimators=256, bootstrap=True, verbose=0, random_state=rng2)
        rf_params = dict(n_estimators=256, bootstrap=False, verbose=0, random_state=rng2)
        for count, (train_idx, test_idx) in enumerate(skf_iter):
            y_test = y[test_idx]
            y_train = y[train_idx]
            if True:
                score_list = np.array(
                    [m.fs.sum() for m in ut.take(matches_RAT_SV, test_idx)])
                auc_naive = sklearn.metrics.roc_auc_score(y_test, score_list)

            if True:
                X_train = X_nonan[train_idx]
                X_test = X_nonan[test_idx]
                # Train uncalibrated random forest classifier on train data
                clf = RandomForestClassifier(**rf_params)
                clf.fit(X_train, y_train)

                # evaluate on test data
                clf_probs = clf.predict_proba(X_test)
                auc_learn_nonan = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])

            if allow_nan:
                X_train = X_withnan[train_idx]
                X_test = X_withnan[test_idx]
                # Train uncalibrated random forest classifier on train data
                clf = RandomForestClassifier(missing_values=np.nan, **rf_params)
                clf.fit(X_train, y_train)

                # evaluate on test data
                clf_probs = clf.predict_proba(X_test)
                # log_loss = sklearn.metrics.log_loss(y_test, clf_probs)
                auc_learn_withnan = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])

            newrow = pd.DataFrame([[auc_naive, auc_learn_nonan,
                                    auc_learn_withnan]],
                                  columns=df_results.columns)
            # print(newrow)
            df_results = df_results.append([newrow], ignore_index=True)

        # print(df_results)

        # TODO: TSNE?
        # http://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
        # Perform t-distributed stochastic neighbor embedding.
        # from sklearn import manifold
        # import matplotlib.pyplot as plt
        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        # trans_data = tsne.fit_transform(feats).T
        # ax = fig.add_subplot(2, 5, 10)
        # plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        # plt.axis('tight')
        print('--------')

        df = df_results
        change = df[df.columns[2]] - df[df.columns[0]]
        percent_change = change / df[df.columns[0]] * 100
        df = df.assign(change=change)
        df = df.assign(percent_change=percent_change)

        import sandbox_utools as sbut
        print(sbut.to_string_monkey(df, highlight_cols=[0, 1, 2]))
        print(df.mean())
    # print(df.to_string())


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


def toydata2(rng):
    from sklearn.datasets import samples_generator
    n_samples = 1000
    n_features = 2
    n_classes = 2
    n_informative = 2
    n_clusters_per_class = int((2 ** n_informative) // n_classes)
    hypercube = False
    samplekw = dict(
        flip_y=0.00,
        class_sep=1.0,
        shift=[-10, 10],
        scale=1.0,
        n_redundant=0,
        n_repeated=0,
        hypercube=hypercube, n_samples=n_samples, n_informative=n_informative,
        n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
        weights=None, shuffle=True, n_features=n_features, random_state=rng)

    X_true, y = samples_generator.make_classification(**samplekw)
    with_extra = ut.get_argflag('--extra')
    # make very informative nan dimension
    if with_extra:
        n_informative_nan = 100
        # extra_x = (rng.randn(n_informative_nan, 2) / 2 + [[12, -8]])
        extra_x = (rng.randn(n_informative_nan, 2) / 2 + [[10, -12]])
        X_true = np.vstack((X_true, extra_x))
        y = np.append(y, [0] * n_informative_nan)

    # Randomly drop datapoints
    X = X_true.copy()
    nanrate = ut.get_argval('--nanrate', default=.01)
    if nanrate:
        # TODO:
        # * informative nan
        # * random nan
        # * random nan + informative nan
        X.ravel()[rng.rand(X.size) < nanrate] = np.nan

    if with_extra:
        if True:
            X.T[1][-n_informative_nan:] = np.nan
        else:
            X.T[0][-n_informative_nan:-n_informative_nan // 2] = np.nan
            X.T[1][-n_informative_nan // 2:] = np.nan
    return X_true, X, y


def toydata1(rng):
    """
    **Description of Plot**

    You'll notice that there are 4 plots. This is necessary to visualize a grid
    with nans. Each plot shows points in the 2-dimensional grid with corners at
    (0, 0) and (40, 40). The top left plot has these coordinates labeled. The
    other 3 plots correspond to the top left grid, but in these plots at least
    one of the dimensions has been "nanned". In the top right the x-dimension
    is "nanned". In the bottom left the y-dimension is "nanned", and in the
    bottom right both dimensions are "nanned". Even though all plots are drawn
    as a 2d-surface only the topleft plot is truly a surface with 2 degrees of
    freedom. The top right and bottom left plots are really lines with 1 degree
    of freedom, and the bottom right plot is actually just a single point with
    0 degrees of freedom.

    In this example I create 10 Gaussian blobs where the first 9 have their
    means laid out in a 3x3 grid and the last one has its mean in the center,
    but I gave it a high standard deviation. I'll refer to the high std cluster
    as 9, and label the other clusters at the grid means (to agree with the
    demo code) like this:

    ```
    6   7   8
    3   4   5
    0   1   2
    ```

    Looking at the top left plot you can see clusters 0, 1, 2, 4, 6, and 8. The
    reason the other cluster do not appear in this grid is because I've set at
    least one of their dimensions to be nan.  Specifically, cluster 3 had its y
    dimension set to nan; cluster 5 and 7 had their x dimension set to nan; and
    cluster 9 had both x and y dimensions set to nan.

    For clusters 3, 5, and 7, I plot "nanned" points as lines along the nanned
    dimension to show that only the non-nan dimensions can be used to
    distinguish these points. I also plot the original position before I
    "nanned" it for visualization purposes, but the learning algorithm never
    sees this. For cluster 9, I only plot the original positions because all of
    this data collapses to a single point [nan, nan].

    Red points are of class 0, and blue points are of class 1. Points in each
    plot represent the training data. The colored background of each plot
    represents the classification surface.
    """
    from sklearn.datasets import samples_generator
    import functools
    step = 20
    n_samples = 100

    blob = functools.partial(samples_generator.make_blobs, n_samples=n_samples,
                             random_state=rng)

    Xy_blobs = [
        (0, blob(centers=[[0 * step, 0 * step]])[0]),
        (1, blob(centers=[[1 * step, 0 * step]])[0]),
        (0, blob(centers=[[2 * step, 0 * step]])[0]),
        (1, blob(centers=[[0 * step, 1 * step]])[0]),
        (0, blob(centers=[[1 * step, 1 * step]])[0]),
        (0, blob(centers=[[2 * step, 1 * step]])[0]),
        (0, blob(centers=[[0 * step, 2 * step]])[0]),
        (1, blob(centers=[[1 * step, 2 * step]])[0]),
        (0, blob(centers=[[2 * step, 2 * step]])[0]),
        (1, blob(centers=[[1 * step, 1 * step]], cluster_std=5)[0]),
    ]
    X_blobs = [Xy[1] for Xy in Xy_blobs]
    X_true = np.vstack(X_blobs)
    y_blobs = [np.full(len(X), y_, dtype=np.int) for y_, X in Xy_blobs]

    # nanify some values
    if True:
        X_blobs[3][:, 1] = np.nan
        X_blobs[7][:, 0] = np.nan
        X_blobs[5][:, 0] = np.nan
        X_blobs[-1][:, :] = np.nan

    X = np.vstack(X_blobs)
    y = np.hstack(y_blobs)
    return X_true, X, y


def get_toydata(rng):
    if ut.get_argflag('--toy1'):
        X_true, X, y = toydata1(rng)
    elif ut.get_argflag('--toy2'):
        X_true, X, y = toydata2(rng)
    return X_true, X, y


def toy_classify_nans():
    r"""
    SeeAlso:
        python -m sklearn.ensemble.tests.test_forest test_multioutput

    CommandLine:
        python -m ibeis toy_classify_nans
        python -m ibeis toy_classify_nans --toy1 --save "rf_nan_toy1.jpg" --figsize=10,10
        python -m ibeis toy_classify_nans --toy2 --save "rf_nan_toy2.jpg" --figsize=10,10
        python -m ibeis toy_classify_nans --toy2 --save "rf_nan_toy3.jpg" --figsize=10,10 --extra
        python -m ibeis toy_classify_nans --toy2 --save "rf_nan_toy4.jpg" --figsize=10,10 --extra --nanrate=0
        python -m ibeis toy_classify_nans --toy2 --save "rf_nan_toy5.jpg" --figsize=10,10 --nanrate=0

    Example:
        >>> from ibeis.algo.hots.script_vsone import *  # NOQA
        >>> result = toy_classify_nans()
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    print('Creating test data')

    X_true, X, y = get_toydata(rng)

    assert len(X) == len(y)

    print('Fitting RF on %d points' % (len(X),))

    grid_basis = dict(  # NOQA
        criterion=['gini', 'entropy'],
    )
    # Train uncalibrated random forest classifier on train data
    clf = RandomForestClassifier(n_estimators=64, random_state=42,
                                 criterion='gini',
                                 missing_values=np.nan, bootstrap=False)
    import pprint
    pprint.pprint(clf.__dict__)
    clf.fit(X, y)
    pprint.pprint(clf.__dict__)
    # indicator, n_nodes_ptr = clf.decision_path(X)
    # clf.fit(X, y)
    # clf_probs = clf.predict_proba(X)

    if ut.show_was_requested():
        # assert n_features == 2
        show_nan_decision_function_2d(X, y, X_true, clf)
        ut.show_if_requested()


def show_nan_decision_function_2d(X, y, X_true, clf):
    import numpy as np

    print('Drawing')

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    plot_step = 1.0
    x_min, x_max = X_true[:, 0].min() - 1, X_true[:, 0].max() + 1
    y_min, y_max = X_true[:, 1].min() - 1, X_true[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    yynan = np.full(yy.shape, fill_value=np.nan)
    xxnan = np.full(yy.shape, fill_value=np.nan)

    # Get prediction surface in the non-nan-zone
    Z_nonnan = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface in the xnan-zone
    Z_xnan = clf.predict_proba(
        np.c_[xxnan.ravel(), yy.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface in the ynan-zone
    Z_ynan = clf.predict_proba(
        np.c_[xx.ravel(), yynan.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface for all-nan-zone
    Z_fullnan = clf.predict_proba(
        np.c_[xxnan.ravel(), yynan.ravel()]).T[1].reshape(xx.shape)

    is_nonnan = np.logical_and(~np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_xnan = np.logical_and(np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_ynan = np.logical_and(~np.isnan(X.T[0]), np.isnan(X.T[1]))
    is_fullnan = np.logical_and(np.isnan(X.T[0]), np.isnan(X.T[1]))

    # Draw surfaces and support points in different axes
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    gs = gridspec.GridSpec(17, 17)
    pnum1 = (gs[0:8,  0:8],)
    pnum2 = (gs[0:8,  8:16],)
    pnum3 = (gs[9:17, 0:8],)
    pnum4 = (gs[9:17, 8:16],)

    fig = plt.figure()

    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.linspace(0, 1))

    color0 = cmap(0 / cmap.N)
    color1 = cmap((cmap.N - 1) / cmap.N)

    def draw_line_segments(pts1, pts2, ax=None, **kwargs):
        import matplotlib as mpl
        if ax is None:
            ax = plt.gca()
        assert len(pts1) == len(pts2), 'unaligned'
        segments = [(xy1, xy2) for xy1, xy2 in zip(pts1, pts2)]
        linewidth = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
        alpha = kwargs.pop('alpha', 1.0)
        line_group = mpl.collections.LineCollection(segments, linewidth,
                                                    alpha=alpha, **kwargs)
        ax.add_collection(line_group)

    def draw_single_nan_lines(X_true, y, flags, nan_dim):
        if not np.any(flags):
            return
        nandim_min = np.nanmin(X_true.T[nan_dim])
        nandim_max = np.nanmax(X_true.T[nan_dim])

        num_dim = 1 - nan_dim  # 2d only
        numdim_pts = X[flags].T[num_dim]

        pts1 = np.empty((flags.sum(), 2))
        pts2 = np.empty((flags.sum(), 2))
        pts1[:, nan_dim] = nandim_min
        pts2[:, nan_dim] = nandim_max
        pts1[:, num_dim] = numdim_pts
        pts2[:, num_dim] = numdim_pts
        y_ = y[flags]
        draw_line_segments(pts1[y_ == 0], pts2[y_ == 0], color=color0, linestyle='-', alpha=1.0)
        draw_line_segments(pts1[y_ == 1], pts2[y_ == 1], color=color1, linestyle='-', alpha=1.0)

    def draw_train_points(X_true, y, flags):
        plt.plot(X_true[flags].T[0][y[flags] == 0], X_true[flags].T[1][y[flags] == 0], 'o', color=color0, markeredgecolor='w')
        plt.plot(X_true[flags].T[0][y[flags] == 1], X_true[flags].T[1][y[flags] == 1], 'o', color=color1, markeredgecolor='w')

    def _contour(Z):
        plt.contourf(xx, yy, Z, cmap=cmap, norm=norm, alpha=1.0)

    fig.add_subplot(*pnum1)
    _contour(Z_nonnan)
    flags = is_nonnan
    draw_train_points(X_true, y, flags)
    plt.title('non-nan decision surface')
    plt.gca().set_aspect('equal')

    fig.add_subplot(*pnum2)
    _contour(Z_xnan)
    flags = is_xnan
    draw_train_points(X_true, y, flags)
    draw_single_nan_lines(X_true, y, flags, 0)
    plt.gca().set_xticks([])
    plt.gca().set_xlabel('nan')

    plt.title('x-nan decision line')
    plt.gca().set_aspect('equal')

    fig.add_subplot(*pnum3)
    _contour(Z_ynan)
    flags = is_ynan
    draw_train_points(X_true, y, flags)
    # make nan-lines
    draw_single_nan_lines(X_true, y, flags, 1)
    plt.title('y-nan decision line')
    plt.gca().set_aspect('equal')
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('nan')

    fig.add_subplot(*pnum4)
    _contour(Z_fullnan)
    flags = is_fullnan
    draw_train_points(X_true, y, flags)
    plt.title('full-nan decision point')
    plt.gca().set_aspect('equal')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xlabel('nan')
    plt.gca().set_ylabel('nan')

    plt.gcf().suptitle('Random Forest With NaN decision criteria')

    gs = gridspec.GridSpec(1, 16)
    subspec = gs[:, -1:]
    cax = plt.subplot(subspec)
    plt.colorbar(sm, cax)
    cax.set_ylabel('probability class 1')

    new_subplotpars = fig.subplotpars.__dict__.copy()
    del new_subplotpars['validate']
    new_subplotpars.update(left=.001, right=.9, top=.9, bottom=.05, hspace=.3, wspace=.1)
    plt.subplots_adjust(**new_subplotpars)


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
