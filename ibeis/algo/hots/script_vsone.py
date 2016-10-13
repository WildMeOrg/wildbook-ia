# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import utool as ut
from six.moves import zip, range  # NOQA
print, rrr, profile = ut.inject2(__name__)


def classify_nans():
    r"""
    SeeAlso:
        python -m sklearn.ensemble.tests.test_forest test_multioutput

    CommandLine:
        python -m ibeis.algo.hots.script_vsone classify_nans --show

    Example:
        >>> from ibeis.algo.hots.script_vsone import *  # NOQA
        >>> result = classify_nans()
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets.samples_generator import make_classification

    rng = np.random.RandomState(42)
    X_true, y = make_classification(n_samples=100, n_features=2,
                                    n_informative=2, n_redundant=0,
                                    n_repeated=0, n_classes=2,
                                    n_clusters_per_class=2, weights=None,
                                    flip_y=0.00, class_sep=2.0,
                                    hypercube=True, shift=0.0, scale=1.0,
                                    shuffle=True, random_state=rng)
    # Randomly drop datapoints
    X = X_true.copy()
    X.ravel()[rng.rand(X.size) > .9] = np.nan

    # TODO:
    # * informative nan
    # * random nan
    # * random nan + informative nan

    import plottool as pt
    pt.qt4ensure()
    fig = pt.figure(doclf=True)
    pt.plot(X.T[0][y == 0], X.T[1][y == 0], 'r.')
    pt.plot(X.T[0][y == 1], X.T[1][y == 1], 'g.')

    row_has_nan = np.where(np.any(np.isnan(X), axis=1))[0]
    for i in row_has_nan:
        row = X[i]
        lbl = y[i]
        if not np.isnan(row)[0]:
            dim1 = [row[0], row[0]]
            dim2 = [np.nanmin(X_true.T[1]), np.nanmax(X_true.T[1])]
            m = 'g-' if lbl else 'r-'
            pt.plot(dim1, dim2, m)
            pt.plot(X_true[i][0], X_true[i][1], 'o' + m)
        if not np.isnan(row)[1]:
            dim1 = [np.nanmin(X_true.T[0]), np.nanmax(X_true.T[0])]
            dim2 = [row[1], row[1]]
            m = 'g-' if lbl else 'r-'
            pt.plot(dim1, dim2, m)
            pt.plot(X_true[i][0], X_true[i][1], 'o' + m)

    pt.bring_to_front(fig)

    # X = rng.rand(1000, 64)
    # # invalidate some inputs
    # X.ravel()[rng.rand(X.size) > .9] = np.nan
    # y = rng.rand(1000) > .5

    clf = RandomForestClassifier(random_state=42, missing_values=np.nan,
                                 bootstrap=True)
    clf.fit(X, y)
    indicator, n_nodes_ptr = clf.decision_path(X)

    # Train uncalibrated random forest classifier on train data
    # clf = RandomForestClassifier(n_estimators=250, verbose=0, missing_values=True)
    # clf.fit(X, y)

    clf_probs = clf.predict_proba(X)
    print('clf_probs = %r' % (clf_probs,))


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
