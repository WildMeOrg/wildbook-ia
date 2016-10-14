# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
from six.moves import zip, range  # NOQA
print, rrr, profile = ut.inject2(__name__)


def toy_classify_nans():
    r"""
    SeeAlso:
        python -m sklearn.ensemble.tests.test_forest test_multioutput

    CommandLine:
        python -m ibeis.algo.hots.script_vsone toy_classify_nans
        python -m ibeis.algo.hots.script_vsone toy_classify_nans --show

    Example:
        >>> from ibeis.algo.hots.script_vsone import *  # NOQA
        >>> result = toy_classify_nans()
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets.samples_generator import make_classification

    rng = np.random.RandomState(42)
    print('Creating test data')

    n_samples = 500
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

    X_true, y = make_classification(**samplekw)

    with_extra = True

    # make very informative nan dimension
    if with_extra:
        n_informative_nan = 100
        extra_x = (rng.randn(n_informative_nan, 2) / 2 + [[12, -8]])
        X_true = np.vstack((X_true, extra_x))
        y = np.append(y, [0] * n_informative_nan)

    # Randomly drop datapoints
    X = X_true.copy()
    nan_rate = .01
    if nan_rate:
        # TODO:
        # * informative nan
        # * random nan
        # * random nan + informative nan
        X.ravel()[rng.rand(X.size) < nan_rate] = np.nan

    if with_extra:
        if True:
            X.T[1][-n_informative_nan:] = np.nan
        else:
            X.T[0][-n_informative_nan:-n_informative_nan // 2] = np.nan
            X.T[1][-n_informative_nan // 2:] = np.nan

    print('Fitting')
    # Train uncalibrated random forest classifier on train data
    clf = RandomForestClassifier(n_estimators=512, random_state=42,
                                 missing_values=np.nan, bootstrap=True)
    print(ut.repr2(clf.__dict__))
    clf.fit(X, y)

    # indicator, n_nodes_ptr = clf.decision_path(X)
    # clf.fit(X, y)
    # clf_probs = clf.predict_proba(X)

    if ut.show_was_requested():
        assert n_features == 2
        show_nan_decision_function_2d(X, y, X_true, clf)
        ut.show_if_requested()


def show_nan_decision_function_2d(X, y, X_true, clf):
    import numpy as np
    import plottool as pt
    pt.qt4ensure()
    fig = pt.figure(doclf=True, fnum=1, pnum=(2, 2, 3))

    cmap = pt.plt.cm.RdYlBu

    plot_step = 0.1  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses

    SHOW_COARSE = 0
    SHOW_FINE = 1

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    x_min, x_max = X_true[:, 0].min() - 1, X_true[:, 0].max() + 1
    y_min, y_max = X_true[:, 1].min() - 1, X_true[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    if SHOW_FINE:
        # Plot alpha blend the decision surfaces of the ensemble of classifiers
        # Choose alpha blend level with respect to the number of estimators
        # that are in use (noting that AdaBoost can use fewer estimators
        # than its maximum if it achieves a good enough fit early on)
        # estimator_alpha = 1.0 / len(clf.estimators_)
        Z_maps = []
        for tree in clf.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Z_maps.append(Z)
        # import vtool as vt
        Z_combo = np.sum(Z_maps, axis=0) / len(Z_maps)
        # Z_combo = vt.blend.blend_images_average_stack(Z_maps)

        pt.plt.contourf(xx, yy, Z_combo, cmap=cmap, alpha=.5)

    # Build a coarser grid to plot a set of ensemble classifications
    # to show how these are different to what we see in the decision
    # surfaces. These points are regularly space and do not have a black outline
    if SHOW_COARSE:
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = clf.predict(np.c_[xx_coarser.ravel(),
                                             yy_coarser.ravel()]).reshape(xx_coarser.shape)
        pt.plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap,
                       edgecolors="none")

    pt.plot(X.T[0][y == 0], X.T[1][y == 0], 'r.')
    pt.plot(X.T[0][y == 1], X.T[1][y == 1], 'g.')

    row_has_nan = np.where(np.any(np.isnan(X), axis=1))[0]
    # for i in row_has_nan:
    #     row = X[i]
    #     lbl = y[i]
    #     if not np.isnan(row)[0]:
    #         dim1 = [row[0], row[0]]
    #         dim2 = [np.nanmin(X_true.T[1]), np.nanmax(X_true.T[1])]
    #         m = 'g-' if lbl else 'r-'
    #         pt.plot(dim1, dim2, m)
    #         pt.plot(X_true[i][0], X_true[i][1], 'o' + m)
    #     if not np.isnan(row)[1]:
    #         dim1 = [np.nanmin(X_true.T[0]), np.nanmax(X_true.T[0])]
    #         dim2 = [row[1], row[1]]
    #         m = 'g-' if lbl else 'r-'
    #         pt.plot(dim1, dim2, m)
    #         pt.plot(X_true[i][0], X_true[i][1], 'o' + m)

    # DO NAN PLOTS
    if SHOW_FINE:
        fig = pt.figure(fnum=1, pnum=(2, 2, 1))
        Z_maps = []
        yynan = yy.copy()
        yynan[:] = np.nan
        for tree in clf.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yynan.ravel()])
            Z = Z.reshape(xx.shape)
            Z_maps.append(Z)
        # import vtool as vt
        Z_combo = np.sum(Z_maps, axis=0) / len(Z_maps)
        pt.plt.contourf(xx, yy, Z_combo, cmap=cmap, alpha=.5)

        for i in row_has_nan:
            row = X[i]
            lbl = y[i]
            if not np.isnan(row)[0]:
                dim1 = [row[0], row[0]]
                dim2 = [np.nanmin(X_true.T[1]), np.nanmax(X_true.T[1])]
                m = 'g-' if lbl else 'r-'
                pt.plot(dim1, dim2, m)
                pt.plot(X_true[i][0], X_true[i][1], 'o' + m)

        fig = pt.figure(fnum=1, pnum=(2, 2, 4))
        Z_maps = []
        xxnan = xx.copy()
        xxnan[:] = np.nan
        for tree in clf.estimators_:
            Z = tree.predict(np.c_[xxnan.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Z_maps.append(Z)
        # import vtool as vt
        Z_combo = np.sum(Z_maps, axis=0) / len(Z_maps)
        pt.plt.contourf(xx, yy, Z_combo, cmap=cmap, alpha=.5)

        for i in row_has_nan:
            row = X[i]
            lbl = y[i]
            if not np.isnan(row)[1]:
                dim1 = [np.nanmin(X_true.T[0]), np.nanmax(X_true.T[0])]
                dim2 = [row[1], row[1]]
                m = 'g-' if lbl else 'r-'
                pt.plot(dim1, dim2, m)
                pt.plot(X_true[i][0], X_true[i][1], 'o' + m)

    # Need to show horizontal and vertical nan bar

    pt.bring_to_front(fig)


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
