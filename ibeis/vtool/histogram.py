# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip
# Science
import scipy.signal as spsignal
import numpy as np
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[hist]', DEBUG=False)


TAU = np.pi * 2


def show_ori_image_ondisk(ori_img_fpath=None, weights_img_fpath=None, img_fpath=None):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        ori (?):
        gmag (?):

    CommandLine:
        python -m vtool.histogram --test-show_ori_image_ondisk --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> ori_img_fpath     = ut.get_argval('--fpath-ori', type_=str, default='None')
        >>> weights_img_fpath = ut.get_argval('--fpath-weight', type_=str, default='None')
        >>> patch_img_fpath = ut.get_argval('--fpath-patch', type_=str, default='None')
        >>> result = show_ori_image_ondisk(ori_img_fpath, weights_img_fpath, patch_img_fpath)
        >>> pt.show_if_requested()
    """
    #if img_fpath is not None:
    #    img_fpath = ut.get_argval('--fpath', type_=str, default=ut.grab_test_imgpath('star.png'))
    #    img_fpath = ut.get_argval('--fpath', type_=str, default=ut.grab_test_imgpath('star.png'))
    #    img = vt.imread(img_fpath)
    #    ori_img_fpath     = ut.get_argval('--fpath-ori', type_=str, default=ut.augpath(img_fpath, '_ori'))
    #    weights_img_fpath = ut.get_argval('--fpath-weight', type_=str, default=ut.augpath(img_fpath, '_mag'))
    #    vt.imwrite(ori_img_fpath, vt.patch_ori(*vt.patch_gradient(img)))
    #    vt.imwrite(weights_img_fpath, vt.patch_mag(*vt.patch_gradient(img)))
    import vtool as vt
    if img_fpath is not None and img_fpath != 'None':
        patch = vt.imread(img_fpath, grayscale=True)
    else:
        patch = None
    if ori_img_fpath is not None and ori_img_fpath != 'None':
        gori = vt.imread(ori_img_fpath, grayscale=True)
        gori = TAU * gori / 255.0
    else:
        gori = None
    #print('stats(gori) = ' + ut.get_stats_str(gori.ravel()))
    if weights_img_fpath is not None and weights_img_fpath != 'None':
        weights = vt.imread(weights_img_fpath, grayscale=True)
        weights = weights / 255.0
    else:
        weights = None
    print('show_ori_image_ondisk')
    print(' * ori_img_fpath = %r' % (ori_img_fpath,))
    print(' * weights_img_fpath = %r' % (weights_img_fpath,))
    #import cv2
    #cv2.imread(ori_img_fpath, cv2.IMREAD_UNCHANGED)
    show_ori_image(gori, weights, patch)
    title = ut.get_argval('--title', default='')
    import plottool as pt
    pt.set_figtitle(title)


def show_ori_image(gori, weights, patch, gradx=None, grady=None, fnum=None):
    """
        python -m pyhesaff._pyhesaff --test-test_rot_invar --show --nocpp
    """
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    print('gori.max = %r' % gori.max())
    assert gori.max() <= TAU
    assert gori.min() >= 0
    bgr_ori = pt.color_orimag(gori / TAU, weights, False, encoding='bgr')
    print('bgr_ori.max = %r' % bgr_ori.max())

    #ut.embed()

    bgr_ori = (255 * bgr_ori).astype(np.uint8)
    print('bgr_ori.max = %r' % bgr_ori.max())
    #bgr_ori = np.array(bgr_ori, dtype=np.uint8)
    legend = pt.make_ori_legend_img()
    gorimag_, woff, hoff = pt.stack_images(bgr_ori, legend, vert=False, modifysize=True)
    if patch is None:
        pt.imshow(gorimag_, fnum=fnum)
    else:
        pt.imshow(gorimag_, fnum=fnum, pnum=(2, 1, 1))
        pt.imshow(patch, fnum=fnum, pnum=(2, 2, 3))
        #pt.imshow(patch, fnum=fnum, pnum=(2, 2, 1))
        #gradx, grady = np.cos(gori + TAU / 4.0), np.sin(gori + TAU / 4.0)
        if gradx is not None and grady is not None:
            if weights is not None:
                gradx *= weights
                grady *= weights
            #pt.imshow(bgr_ori, pnum=(2, 2, 4))
            pt.draw_vector_field(gradx, grady, pnum=(2, 2, 4), invert=True)


def show_hist_submaxima(hist_, edges=None, centers=None, maxima_thresh=.8, pnum=(1, 1, 1)):
    r"""
    For C++ to show data

    Args:
        hist_ (?):
        edges (None):
        centers (None):

    CommandLine:
        python -m vtool.histogram --test-show_hist_submaxima --show
        python -m pyhesaff._pyhesaff --test-test_rot_invar --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> hist_ = np.array(list(map(float, ut.get_argval('--hist', type_=list, default=[1, 4, 2, 5, 3, 3]))))
        >>> edges = np.array(list(map(float, ut.get_argval('--edges', type_=list, default=[0, 1, 2, 3, 4, 5, 6]))))
        >>> maxima_thresh = ut.get_argval('--maxima_thresh', type_=float, default=.8)
        >>> centers = None
        >>> # execute function
        >>> show_hist_submaxima(hist_, edges, centers, maxima_thresh)
        >>> pt.show_if_requested()
    """
    #print(repr(hist_))
    #print(repr(hist_.shape))
    #print(repr(edges))
    #print(repr(edges.shape))
    #ut.embed()
    import plottool as pt
    #ut.embed()
    if centers is None:
        centers = hist_edges_to_centers(edges)
    bin_colors = pt.get_orientation_color(centers)
    pt.figure(fnum=pt.next_fnum(), pnum=pnum)
    POLAR = False
    if POLAR:
        pt.df2.plt.subplot(*pnum, polar=True, axisbg='#000000')
    pt.draw_hist_subbin_maxima(hist_, centers, bin_colors=bin_colors, maxima_thresh=maxima_thresh)
    #pt.gca().set_rmax(hist_.max() * 1.1)
    #pt.gca().invert_yaxis()
    #pt.gca().invert_xaxis()
    pt.dark_background()
    #if ut.get_argflag('--legend'):
    #    pt.figure(fnum=pt.next_fnum())
    #    centers_ = np.append(centers, centers[0])
    #    r = np.ones(centers_.shape) * .2
    #    ax = pt.df2.plt.subplot(111, polar=True)
    #    pt.plots.colorline(centers_, r, cmap=pt.df2.plt.get_cmap('hsv'), linewidth=10)
    #    #ax.plot(centers_, r, 'm', color=bin_colors, linewidth=100)
    #    ax.set_rmax(.2)
    #    #ax.grid(True)
    #    #ax.set_title("Angle Colors", va='bottom')
    title = ut.get_argval('--title', default='')
    import plottool as pt
    pt.set_figtitle(title)


def get_histinfo_str(hist, edges):
    # verify results
    centers = hist_edges_to_centers(edges)
    hist_str   = 'hist    = ' + str(hist.tolist())
    center_str = 'centers = ' + str(centers.tolist())
    edge_str   = 'edges   = [' +  ', '.join(['%.2f' % _ for _ in edges]) + ']'
    histinfo_str = hist_str + ut.NEWLINE + center_str + ut.NEWLINE + edge_str
    return histinfo_str


def interpolated_histogram(data, weights, range_, bins, interpolation_wrap=True,
                           DEBUG_ROTINVAR=False):
    r"""
    Follows np.histogram, but does interpolation

    Args:
        range_ (tuple): range from 0 to 1
        bins (?):

    CommandLine:
        python -m vtool.histogram --test-interpolated_histogram

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> data    = np.array([ 0,  1,  2,  3.5,  3,  3,  4,  4])
        >>> weights = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> range_ = (0, 4)
        >>> bins = 5
        >>> interpolation_wrap = False
        >>> # execute function
        >>> hist, edges = interpolated_histogram(data, weights, range_, bins, interpolation_wrap)
        >>> assert np.abs(hist.sum() - weights.sum()) < 1E-9
        >>> assert hist.size == bins
        >>> assert edges.size == bins + 1
        >>> result = get_histinfo_str(hist, edges)
        >>> print(result)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> data    = np.array([ 0,  1,  2,  3.5,  3,  3,  4,  4])
        >>> weights = np.array([4.5, 1., 1., 1., 1., 1., 1., 1.])
        >>> range_ = (-.5, 4.5)
        >>> bins = 5
        >>> interpolation_wrap = True
        >>> # execute function
        >>> hist, edges = interpolated_histogram(data, weights, range_, bins, interpolation_wrap)
        >>> assert np.abs(hist.sum() - weights.sum()) < 1E-9
        >>> assert hist.size == bins
        >>> assert edges.size == bins + 1
        >>> result = get_histinfo_str(hist, edges)
        >>> print(result)

    #Example2:
    #    >>> # ENABLE_DOCTEST
    #    >>> from vtool.histogram import *  # NOQA
    #    >>> # build test data
    #    >>> data    = np.random.rand(10)
    #    >>> weights = np.random.rand(10)
    #    >>> range_ = (0, 1)
    #    >>> bins = np.random.randint(2) + 1 + np.random.randint(2) * 100
    #    >>> interpolation_wrap = True
    #    >>> # execute function
    #    >>> hist, edges = interpolated_histogram(data, weights, range_, bins, interpolation_wrap)
    #    >>> assert np.abs(hist.sum() - weights.sum()) < 1E-9
    #    >>> assert hist.size == bins
    #    >>> assert edges.size == bins + 1
    #    >>> result = get_histinfo_str(hist, edges)
    #    >>> print(result)
    """
    assert bins > 0, 'must have nonzero bins'
    data = np.asarray(data)
    if weights is not None:
        weights = np.asarray(weights)
        assert np.all(weights.shape == data.shape), 'shapes disagree'
        weights = weights.ravel()
    data = data.ravel()
    # Compute bin edges like in np.histogram
    start, stop = float(range_[0]), float(range_[1])
    if start == stop:
        start -= 0.5
        stop += 0.5
    # Find bin edges
    hist_dtype = np.float64
    # Compute bin step size, add one if last bin is the same as the first
    step = (stop - start) / float((bins + interpolation_wrap))
    #edges = [start + i * step for i in range(bins + 1)]
    #centers = hist_edges_to_centers(edges)

    half_step = step / 2.0
    # Find fractional bin center index for each datapoint
    data_offset = start + half_step
    frac_index  = (data - data_offset) / step
    # Find bin center to the left of each datapoint
    left_index = np.floor(frac_index).astype(np.int32)
    # Find bin center to the right of each datapoint
    right_index = left_index + 1
    # Find the fraction of the distiance the right center is away from the datapoint
    right_alpha = (frac_index - left_index)
    left_alpha = 1.0 - right_alpha

    if DEBUG_ROTINVAR:
        print('bins = %r' % bins)
        print('step = %r' % step)
        print('half_step = %r' % half_step)
        print('data_offset = %r' % data_offset)
        TAU = 2 * np.pi
        print("-.5 MOD tau = %r" % (-.5 % TAU,))

    # Handle edge cases
    if interpolation_wrap:
        # when the stop == start (like in orientations)
        left_index  %= bins
        right_index %= bins
    else:
        left_index[left_index < 0] = 0
        right_index[right_index >= bins] = bins - 1

    # Each keypoint votes into its left and right bins
    left_vote  = left_alpha * weights
    right_vote = right_alpha * weights
    hist = np.zeros((bins,), hist_dtype)
    # TODO: can problably do this faster with cumsum
    for index, vote in zip(left_index, left_vote):
        hist[index] += vote
    for index, vote in zip(right_index, right_vote):
        hist[index] += vote

    if interpolation_wrap:
        edges = np.linspace(start, stop, bins + 1, endpoint=False)
    else:
        edges = np.linspace(start, stop, bins + 1, endpoint=True)
    if DEBUG_ROTINVAR:
        import vtool as vt
        assert np.allclose(np.diff(edges), step)
        print(hist.shape)
        print(edges.shape)
        print(vt.kpts_docrepr(hist, 'hist', False))
        print(vt.kpts_docrepr(edges, 'edges', False))

    return hist, edges
    #block = 2 ** 16
    #cumhist = np.zeros(edges.shape, hist_dtype)
    #zero = np.array(0, dtype=np.float64)
    ## Blocking code that is used in numpy
    #for block_index in np.arange(0, len(data), block):
    #    _data = data[block_index:block_index + block]
    #    _weights = weights[block_index:block_index + block]
    #    _sortx = np.argsort(_data)
    #    sorted_data = _data.take(_sortx)
    #    sorted_weights = _weights.take(_sortx)
    #    cumsum_weights = np.concatenate(([zero, ], sorted_weights.cumsum()))
    #    # Find which bin each datapoint belongs in
    #    bin_index = np.r_[
    #        # The first edge will correspond with the first center
    #        sorted_data.searchsorted(edges[:-1], 'left'),
    #        sorted_data.searchsorted(edges[-1], 'right')
    #    ]
    #    cumhist += cumsum_weights[bin_index]
    #hist = np.diff(cumhist)


@profile
def hist_edges_to_centers(edges):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> edges = [-0.79, 0.00, 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.50, 6.28, 7.07]
        >>> # execute function
        >>> centers = hist_edges_to_centers(edges)
        >>> # verify results
        >>> result = str(centers)
        >>> print(result)
        [-0.395  0.395  1.18   1.965  2.75   3.535  4.32   5.105  5.89   6.675]

    Ignore:
        import plottool as pt
        from matplotlib import pyplot as plt
        fig = plt.figure()
        plt.plot(edges, [.5] * len(edges), 'r|-')
        plt.plot(centers, [.5] * len(centers), 'go')
        plt.gca().set_ylim(.49, .51)
        plt.gca().set_xlim(-2, 8)
        pt.dark_background()
        fig.show()
    """
    centers = np.array([(e1 + e2) / 2.0 for (e1, e2) in zip(edges[:-1], edges[1:])])
    return centers


@profile
def wrap_histogram(hist_, edges_, DEBUG_ROTINVAR=False):
    r"""
    Simulates the first and last histogram bin being being adjacent to one another
    by replicating those bins at the last and first positions respectively.

    Args:
        hist_ (ndarray):
        edges_ (ndarray):

    Returns:
        tuple: (hist_wrap, edge_wrap)

    CommandLine:
        python -m vtool.histogram --test-wrap_histogram

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> hist_ = np.array([  8.  ,   0.  ,   0.  ,  34.32,  29.45,   0.  ,   0.  ,   6.73])
        >>> edges_ = np.array([ 0.        ,  0.78539816,  1.57079633,
        ...                    2.35619449,  3.14159265,  3.92699081,
        ...                    4.71238898,  5.49778714,  6.2831853 ])
        >>> # execute function
        >>> (hist_wrap, edge_wrap) = wrap_histogram(hist_, edges_)
        >>> # verify results
        >>> edgewrap_str =  '[' +  ', '.join(['%.2f' % _ for _ in edge_wrap]) + ']'
        >>> histwrap_str = str(hist_wrap.tolist())
        >>> result = histwrap_str + ut.NEWLINE + edgewrap_str
        >>> print(result)
        [6.73, 8.0, 0.0, 0.0, 34.32, 29.45, 0.0, 0.0, 6.73, 8.0]
        [-0.79, 0.00, 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.50, 6.28, 7.07]
    """
    # FIXME; THIS NEEDS INFORMATION ABOUT THE DISTANCE FROM THE LAST BIN
    # TO THE FIRST. IT IS OK AS LONG AS ALL STEPS ARE EQUAL, BUT IT IS NOT
    # GENERAL
    left_step, right_step = np.diff(edges_)[[0, -1]]
    hist_wrap = np.hstack((hist_[-1:], hist_, hist_[0:1]))
    edge_wrap = np.hstack((edges_[0:1] - left_step, edges_, edges_[-1:] + right_step))
    if DEBUG_ROTINVAR:
        import vtool as vt
        print(vt.kpts_docrepr(hist_wrap, 'hist_wrap', False))
        print(vt.kpts_docrepr(edge_wrap, 'edge_wrap', False))

    return hist_wrap, edge_wrap


@profile
def hist_interpolated_submaxima(hist, centers=None, maxima_thresh=.8,
                                DEBUG_ROTINVAR=False):
    r"""
    Args:
        hist (ndarray):
        centers (list):
        maxima_thresh (float):

    Returns:
        tuple: (submaxima_x, submaxima_y)

    CommandLine:
        python -m vtool.histogram --test-hist_interpolated_submaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> maxima_thresh = .8
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> # execute function
        >>> (submaxima_x, submaxima_y) = hist_interpolated_submaxima(hist, centers, maxima_thresh)
        >>> # verify results
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        (array([ 3.0318792]), array([ 37.19208239]))


    Example2:
        >>> from vtool.histogram import *  # NOQA
        >>> hist = [700.413, 1680.93, 823.31, 521.934, 721.44, 777.964, 483.108,
        ...     178.108, 243.603, 1089.18, 2546.99, 306.315, 210.7,
        ...     277.306, 653.927, 805.233, 537.68, 625.399, 738.363,
        ...     1572.03, 406.922, 277.89, 172.177, 387.935, 376.248,
        ...     218.595, 195.206, 222.028, 588.637, 376.1, 193.714,
        ...     175.754, 299.887, 446.97, 275.423, 194.888, 700.413,
        ...     1680.93, ]
        >>> edges = [-0.169816, 0, 0.169816, 0.339631, 0.509447, 0.679263, 0.849078,
        ...     1.01889, 1.18871, 1.35853, 1.52834, 1.69816, 1.86797,
        ...     2.03779, 2.2076, 2.37742, 2.54724, 2.71705, 2.88687,
        ...     3.05668, 3.2265, 3.39631, 3.56613, 3.73595, 3.90576,
        ...     4.07558, 4.24539, 4.41521, 4.58502, 4.75484, 4.92465,
        ...     5.09447, 5.26429, 5.4341, 5.60392, 5.77373, 5.94355,
        ...     6.11336, 6.28318, ]
        >>> hist = np.asarray(hist)
        >>> edges = np.asarray(edges)
        >>> centers = hist_edges_to_centers(edges)
        >>> (submaxima_x, submaxima_y) = hist_interpolated_submaxima(hist, centers, .8)
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        >>> import plottool as pt
        >>> pt.draw_hist_subbin_maxima(hist, centers)
        >>> pt.show_if_requested()


    Ignore:
        import plottool as pt
        pt.draw_hist_subbin_maxima(hist, centers)
    """
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers, maxima_thresh=maxima_thresh)
    if DEBUG_ROTINVAR:
        print('Argmaxima: ')
        print(' * maxima_x = %r' % (maxima_x))
        print(' * maxima_y = %r' % (maxima_y))
        print(' * argmaxima = %r' % (argmaxima))
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
    if DEBUG_ROTINVAR:
        print('Submaxima: ')
        print(' * submaxima_x = %r' % (submaxima_x))
        print(' * submaxima_y = %r' % (submaxima_y))
    return submaxima_x, submaxima_y


@profile
def hist_argmaxima(hist, centers=None, maxima_thresh=.8):
    """

    CommandLine:
        python -m vtool.histogram --test-hist_argmaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> maxima_thresh = .8
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> # execute function
        >>> maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
        >>> # verify results
        >>> result = str((maxima_x, maxima_y, argmaxima))
        >>> print(result)
        (array([ 2.75]), array([ 34.62]), array([4]))

    """
    # FIXME: Not handling general cases
    argmaxima_ = spsignal.argrelextrema(hist, np.greater)[0]  # [0] index because argrelmaxima returns a tuple
    if len(argmaxima_) == 0:
        argmaxima_ = hist.argmax()  # Hack for no maxima
    # threshold maxima to be within a factor of the maximum
    maxima_y = hist[argmaxima_]
    isvalid = maxima_y > maxima_y.max() * maxima_thresh
    argmaxima = argmaxima_[isvalid]
    maxima_y = hist[argmaxima]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    return maxima_x, maxima_y, argmaxima


@profile
def maxima_neighbors(argmaxima, hist, centers=None):
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    y123 = hist[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    return x123, y123


@profile
def interpolate_submaxima(argmaxima, hist, centers=None):
    r"""
    CommandLine:
        python -m vtool.histogram --test-interpolate_submaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> argmaxima = np.array([1, 4])
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> # execute function
        >>> submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
        >>> # verify results
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        (array([ 0.14597723,  3.0318792 ]), array([  9.20251557,  37.19208239]))

    Ignore:
        assert str(interpolate_submaxima(argmaxima, hist, centers)) == str(readable_interpolate_submaxima(argmaxima, hist, centers))
        %timeit interpolate_submaxima(argmaxima, hist, centers)
        %timeit readable_interpolate_submaxima(argmaxima, hist, centers)

    """
    # ~~~TODO Use np.polyfit here instead for readability
    # This turns out to just be faster. Other function is written under
    x123, y123 = maxima_neighbors(argmaxima, hist, centers)
    (y1, y2, y3) = y123
    (x1, x2, x3) = x123
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B     = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    submaxima_x, submaxima_y = np.vstack((xv.T, yv.T))
    return submaxima_x, submaxima_y


def maximum_parabola_point(A, B, C):
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    return xv, yv


def readable_interpolate_submaxima(argmaxima, hist, centers=None):
    x123, y123 = maxima_neighbors(argmaxima, hist, centers)
    coeff_list = [np.polyfit(x123_, y123_, 2) for (x123_, y123_) in zip(x123.T, y123.T)]
    A, B, C = np.vstack(coeff_list).T
    submaxima_x, submaxima_y = maximum_parabola_point(A, B, C)
    #submaxima_points = [maximum_parabola_point(A, B, C) for (A, B, C) in coeff_list]
    #submaxima_x, submaxima_y = np.array(submaxima_points).T
    return submaxima_x, submaxima_y


@profile
def subbin_bounds(z, radius, low, high):
    """
    Gets quantized bounds of a sub-bin/pixel point and a radius.
    Useful for cropping using subpixel points

    Illustration::
        (the bin edges are pipes)
        (the bin centers are pluses)

        Input = {'z': 1.5, 'radius':5.666, 'low':0, 'high':7}
        Output = {'z1':0, 'z2': 7, 'offst': 5.66}

        |   |   |   |   |   |   |   |   |
        |_+_|_+_|_+_|_+_|_+_|_+_|_+_|_+_|
          ^     ^                     ^
          z1    z                     z2
                ,.___.___.___.___.___.   < radius (5.333)
          .---.-,                        < z_offset1 (1.6666)
                ,_.___.___.___.___.___.  < z_offset2 (5.666)
          .---.-,                        < z_offset1 (1.6666)

    Args:
        z (float): center of a circle a 1d pixel array
        radius (float): radius of the circle
        low (int): minimum index of 1d pixel array
        high (int): maximum index of 1d pixel array

    Returns:
        tuple: (iz1, iz2, z_offst) - quantized_bounds and subbin_offset
            iz1 - low radius endpoint
            iz2 - high radius endpoint
            z_offst - subpixel offset
            #Returns: quantized_bounds=(iz1, iz2), subbin_offset

    CommandLine:
        python -m vtool.histogram --test-subbin_bounds

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> z = 1.5
        >>> radius = 5.666
        >>> low = 0
        >>> high = 7
        >>> # execute function
        >>> (iz1, iz2, z_offst) = subbin_bounds(z, radius, low, high)
        >>> # verify results
        >>> result = str((iz1, iz2, z_offst))
        >>> print(result)
        (0, 7, 1.5)
    """
    #print('quan pxl: z=%r, radius=%r, low=%r, high=%r' % (z, radius, low, high))
    # Get subpixel bounds ignoring boundaries
    z1 = z - radius
    z2 = z + radius
    # Quantize and clip bounds
    iz1 = int(max(np.floor(z1), low))
    iz2 = int(min(np.ceil(z2), high))
    # Quantized min radius
    z_offst = z - iz1
    return iz1, iz2, z_offst


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.histogram
        python -m vtool.histogram --allexamples
        python -m vtool.histogram --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
