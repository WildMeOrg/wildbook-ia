# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def get_toydata(rng):
    if ut.get_argflag('--toy2'):
        X_true, X, y = toydata2(rng)
    else:
        X_true, X, y = toydata1(rng)
    return X_true, X, y


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
        hypercube=hypercube,
        n_samples=n_samples,
        n_informative=n_informative,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=None,
        shuffle=True,
        n_features=n_features,
        random_state=rng,
    )

    X_true, y = samples_generator.make_classification(**samplekw)
    with_extra = ut.get_argflag('--extra')
    # make very informative nan dimension
    if with_extra:
        n_informative_nan = 100
        # extra_x = (rng.randn(n_informative_nan, 2) / 2 + [[12, -8]])
        extra_x = rng.randn(n_informative_nan, 2) / 2 + [[10, -12]]
        X_true = np.vstack((X_true, extra_x))
        y = np.append(y, [0] * n_informative_nan)

    # Randomly drop datapoints
    X = X_true.copy()
    nanrate = ut.get_argval('--nanrate', default=0.01)
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
            X.T[0][-n_informative_nan : -n_informative_nan // 2] = np.nan
            X.T[1][-n_informative_nan // 2 :] = np.nan
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

    blob = functools.partial(
        samples_generator.make_blobs, n_samples=n_samples, random_state=rng
    )

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


def show_nan_decision_function_2d(X, y, X_true, clf):
    import numpy as np

    print('Drawing')

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    plot_step = 1.0
    x_min, x_max = X_true[:, 0].min() - 1, X_true[:, 0].max() + 1
    y_min, y_max = X_true[:, 1].min() - 1, X_true[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    yynan = np.full(yy.shape, fill_value=np.nan)
    xxnan = np.full(yy.shape, fill_value=np.nan)

    # Get prediction surface in the non-nan-zone
    Z_nonnan = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface in the xnan-zone
    Z_xnan = clf.predict_proba(np.c_[xxnan.ravel(), yy.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface in the ynan-zone
    Z_ynan = clf.predict_proba(np.c_[xx.ravel(), yynan.ravel()]).T[1].reshape(xx.shape)

    # Get prediction surface for all-nan-zone
    Z_fullnan = (
        clf.predict_proba(np.c_[xxnan.ravel(), yynan.ravel()]).T[1].reshape(xx.shape)
    )

    is_nonnan = np.logical_and(~np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_xnan = np.logical_and(np.isnan(X.T[0]), ~np.isnan(X.T[1]))
    is_ynan = np.logical_and(~np.isnan(X.T[0]), np.isnan(X.T[1]))
    is_fullnan = np.logical_and(np.isnan(X.T[0]), np.isnan(X.T[1]))

    # Draw surfaces and support points in different axes
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    gs = gridspec.GridSpec(17, 17)
    pnum1 = (gs[0:8, 0:8],)
    pnum2 = (gs[0:8, 8:16],)
    pnum3 = (gs[9:17, 0:8],)
    pnum4 = (gs[9:17, 8:16],)

    fig = plt.figure()

    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.linspace(0, 1))

    color0 = cmap(0)
    print('color0 = %r' % (color0,))
    color1 = cmap(1.0)
    print('color1 = %r' % (color1,))

    def draw_line_segments(pts1, pts2, ax=None, **kwargs):
        import matplotlib as mpl

        if ax is None:
            ax = plt.gca()
        assert len(pts1) == len(pts2), 'unaligned'
        segments = [(xy1, xy2) for xy1, xy2 in zip(pts1, pts2)]
        linewidth = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
        alpha = kwargs.pop('alpha', 1.0)
        line_group = mpl.collections.LineCollection(
            segments, linewidth, alpha=alpha, **kwargs
        )
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
        draw_line_segments(
            pts1[y_ == 0], pts2[y_ == 0], color=color0, linestyle='-', alpha=1.0
        )
        draw_line_segments(
            pts1[y_ == 1], pts2[y_ == 1], color=color1, linestyle='-', alpha=1.0
        )

    def draw_train_points(X_true, y, flags):
        plt.plot(
            X_true[flags].T[0][y[flags] == 0],
            X_true[flags].T[1][y[flags] == 0],
            'o',
            color=color0,
            markeredgecolor='w',
        )
        plt.plot(
            X_true[flags].T[0][y[flags] == 1],
            X_true[flags].T[1][y[flags] == 1],
            'o',
            color=color1,
            markeredgecolor='w',
        )

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

    plt.title('x-nan decision surface')
    plt.gca().set_aspect('equal')

    fig.add_subplot(*pnum3)
    _contour(Z_ynan)
    flags = is_ynan
    draw_train_points(X_true, y, flags)
    # make nan-lines
    draw_single_nan_lines(X_true, y, flags, 1)
    plt.title('y-nan decision surface')
    plt.gca().set_aspect('equal')
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('nan')

    fig.add_subplot(*pnum4)
    _contour(Z_fullnan)
    flags = is_fullnan
    draw_train_points(X_true, y, flags)
    plt.title('full-nan decision surface')
    plt.gca().set_aspect('equal')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xlabel('nan')
    plt.gca().set_ylabel('nan')

    plt.gcf().suptitle('RandomForestClassifier With NaN decision criteria')

    gs = gridspec.GridSpec(1, 16)
    subspec = gs[:, -1:]
    cax = plt.subplot(subspec)
    plt.colorbar(sm, cax)
    cax.set_ylabel('probability class 1')

    new_subplotpars = fig.subplotpars.__dict__.copy()
    del new_subplotpars['validate']
    new_subplotpars.update(
        left=0.001, right=0.9, top=0.9, bottom=0.05, hspace=1.0, wspace=1.0
    )
    plt.subplots_adjust(**new_subplotpars)


def main():
    r"""
    SeeAlso:
        python -m sklearn.ensemble.tests.test_forest test_multioutput

    CommandLine:
        python -m wbia toy_classify_nans
        python -m wbia toy_classify_nans --toy1 --save "rf_nan_toy1.jpg" --figsize=10,10
        python -m wbia toy_classify_nans --toy2 --save "rf_nan_toy2.jpg" --figsize=10,10
        python -m wbia toy_classify_nans --toy2 --save "rf_nan_toy3.jpg" --figsize=10,10 --extra
        python -m wbia toy_classify_nans --toy2 --save "rf_nan_toy4.jpg" --figsize=10,10 --extra --nanrate=0
        python -m wbia toy_classify_nans --toy2 --save "rf_nan_toy5.jpg" --figsize=10,10 --nanrate=0

    Example:
        >>> # DISABLE_DOCTEST
        >>> result = toy_classify_nans()
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    print('Creating test data')

    X_true, X, y = get_toydata(rng)

    assert len(X) == len(y)

    print('Fitting RF on %d points' % (len(X),))
    # Train uncalibrated random forest classifier on train data
    clf = RandomForestClassifier(
        n_estimators=64,
        random_state=42,
        criterion='gini',
        missing_values=np.nan,
        bootstrap=False,
    )
    # import pprint
    # pprint.pprint(clf.__dict__)
    clf.fit(X, y)
    # pprint.pprint(clf.__dict__)

    show_nan_decision_function_2d(X, y, X_true, clf)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.toy_nan_rf --show
    """
    main()
    import matplotlib.pyplot as plt

    plt.show()
