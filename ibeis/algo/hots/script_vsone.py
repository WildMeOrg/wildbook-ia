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

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    plot_step = 0.1
    x_min, x_max = X_true[:, 0].min() - 1, X_true[:, 0].max() + 1
    y_min, y_max = X_true[:, 1].min() - 1, X_true[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    yynan = np.full(yy.shape, fill_value=np.nan)
    xxnan = np.full(yy.shape, fill_value=np.nan)

    # Get prediction surface in the non-nan-zone
    Z_nonnan = np.sum([
        tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        for tree in clf.estimators_
    ], axis=0) / len(clf.estimators_)

    # Get prediction surface in the xnan-zone
    Z_xnan = np.sum([
        tree.predict(np.c_[xxnan.ravel(), yy.ravel()]).reshape(xx.shape)
        for tree in clf.estimators_
    ], axis=0) / len(clf.estimators_)

    # Get prediction surface in the ynan-zone
    Z_ynan = np.sum([
        tree.predict(np.c_[xx.ravel(), yynan.ravel()]).reshape(xx.shape)
        for tree in clf.estimators_
    ], axis=0) / len(clf.estimators_)

    # Get prediction surface for all-nan-zone
    Z_fullnan = np.sum([
        tree.predict(np.c_[xxnan.ravel(), yynan.ravel()]).reshape(xx.shape)
        for tree in clf.estimators_
    ], axis=0) / len(clf.estimators_)

    is_nonnan = np.logical_and(~np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_xnan = np.logical_and(np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_ynan = np.logical_and(~np.isnan(X.T[0]), np.isnan(X.T[1]))
    is_fullnan = np.logical_and(np.isnan(X.T[0]), np.isnan(X.T[1]))

    # Draw surfaces and support points in different axes
    # pnum1 = (4, 4, (slice(0, 3), slice(0, 3)))
    # pnum2 = (4, 4, (slice(0, 3), slice(3, 4)))
    # pnum3 = (4, 4, (slice(3, 4), slice(0, 3)))
    # pnum4 = (4, 4, (slice(3, 4), slice(3, 4)))
    pnum1 = (2, 2, 1)
    pnum2 = (2, 2, 2)
    pnum3 = (2, 2, 3)
    pnum4 = (2, 2, 4)

    def draw_single_nan_lines(X_true, y, flags, nan_dim):
        nandim_min = np.nanmin(X_true.T[nan_dim][flags])
        nandim_max = np.nanmax(X_true.T[nan_dim][flags])

        num_dim = 1 - nan_dim  # 2d only
        numdim_pts = X[flags].T[num_dim]

        pts1 = np.empty((flags.sum(), 2))
        pts2 = np.empty((flags.sum(), 2))
        pts1[:, nan_dim] = nandim_min
        pts2[:, nan_dim] = nandim_max
        pts1[:, num_dim] = numdim_pts
        pts2[:, num_dim] = numdim_pts
        y_ = y[flags]
        pt.draw_line_segments2(pts1[y_ == 0], pts2[y_ == 0], color='r', linestyle='--', alpha=.8)
        pt.draw_line_segments2(pts1[y_ == 1], pts2[y_ == 1], color='b', linestyle='--', alpha=.8)

    cmap = pt.plt.cm.RdYlBu

    pt.qt4ensure()
    fig = pt.figure(doclf=True, fnum=1, pnum=pnum1)
    pt.plt.contourf(xx, yy, Z_nonnan, cmap=cmap, alpha=.5)
    flags = is_nonnan
    pt.plot(X[flags].T[0][y[flags] == 0], X[flags].T[1][y[flags] == 0], 'ro')
    pt.plot(X[flags].T[0][y[flags] == 1], X[flags].T[1][y[flags] == 1], 'bo')
    pt.plt.title('non-nan decision line')
    pt.plt.gca().set_aspect('equal')

    fig = pt.figure(fnum=1, pnum=pnum2)
    pt.plt.contourf(xx, yy, Z_xnan, cmap=cmap, alpha=.5)
    flags = is_xnan
    pt.plot(X_true[flags].T[0][y[flags] == 0], X_true[flags].T[1][y[flags] == 0], 'ro')
    pt.plot(X_true[flags].T[0][y[flags] == 1], X_true[flags].T[1][y[flags] == 1], 'bo')
    # make nan-lines
    draw_single_nan_lines(X_true, y, flags, 0)
    pt.gca().set_xticks([])
    pt.gca().set_xlabel('nan')

    pt.plt.title('x-nan decision line')
    pt.plt.gca().set_aspect('equal')

    fig = pt.figure(fnum=1, pnum=pnum3)
    pt.plt.contourf(xx, yy, Z_ynan, cmap=cmap, alpha=.5)
    flags = is_ynan
    pt.plot(X_true[flags].T[0][y[flags] == 0], X_true[flags].T[1][y[flags] == 0], 'ro')
    pt.plot(X_true[flags].T[0][y[flags] == 1], X_true[flags].T[1][y[flags] == 1], 'bo')
    # make nan-lines
    draw_single_nan_lines(X_true, y, flags, 1)
    pt.plt.title('y-nan decision point')
    pt.plt.gca().set_aspect('equal')
    pt.gca().set_yticks([])
    pt.gca().set_ylabel('nan')

    fig = pt.figure(fnum=1, pnum=pnum4)
    pt.plt.contourf(xx, yy, Z_fullnan, cmap=cmap, alpha=.5)
    flags = is_fullnan
    pt.plot(X_true[flags].T[0][y[flags] == 0], X_true[flags].T[1][y[flags] == 0], 'ro')
    pt.plot(X_true[flags].T[0][y[flags] == 1], X_true[flags].T[1][y[flags] == 1], 'bo')
    pt.plt.title('full-nan decision surface')
    pt.plt.gca().set_aspect('equal')
    pt.gca().set_xticks([])
    pt.gca().set_yticks([])
    pt.gca().set_xlabel('nan')
    pt.gca().set_ylabel('nan')

    pt.set_figtitle('Random Forest With NaN decision criteria')

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
