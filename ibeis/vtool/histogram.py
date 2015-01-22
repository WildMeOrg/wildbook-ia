# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip
# Science
import scipy.signal as spsignal
import numpy as np
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[hist]', DEBUG=False)


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
def wrap_histogram(hist_, edges_):
    r"""
    Simulates the first and last histogram bin being being adjacent to one another
    by replicating those bins at the last and first positions respectively.

    Args:
        hist_ (?):
        edges_ (?):

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
    low, high = np.diff(edges_)[[0, -1]]
    hist_wrap = np.hstack((hist_[-1:], hist_, hist_[0:1]))
    edge_wrap = np.hstack((edges_[0:1] - low, edges_, edges_[-1:] + high))
    return hist_wrap, edge_wrap


@profile
def hist_interpolated_submaxima(hist, centers=None):
    r"""
    Args:
        hist (?):
        centers (None):

    Returns:
        tuple: (submaxima_x, submaxima_y)

    CommandLine:
        python -m vtool.histogram --test-hist_interpolated_submaxima

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> hist = np.array([    6.73, 8.69, 0.00  0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> # execute function
        >>> (submaxima_x, submaxima_y) = hist_interpolated_submaxima(hist, centers)
        >>> # verify results
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        (array([ 0.14597723,  3.0318792 ]), array([  9.20251557,  37.19208239]))

    Ignore:
        import plottool as pt
        pt.draw_hist_subbin_maxima(hist, centers)
    """
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
    return submaxima_x, submaxima_y


@profile
def hist_argmaxima(hist, centers=None):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> hist = np.array([    6.73, 8.69, 0.00  0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> # execute function
        >>> maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
        >>> # verify results
        >>> result = str((maxima_x, maxima_y, argmaxima))
        >>> print(result)
        (array([ 0.39,  2.75]), array([  8.69,  34.62]), array([1, 4]))

    """
    # FIXME: Not handling general cases
    argmaxima = spsignal.argrelextrema(hist, np.greater)[0]  # [0] index because argrelmaxima returns a tuple
    if len(argmaxima) == 0:
        argmaxima = hist.argmax()  # Hack for no maxima
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    maxima_y = hist[argmaxima]
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
        >>> # DISABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> # build test data
        >>> argmaxima = np.array([1, 4])
        >>> hist = np.array([    6.73, 8.69, 0.00  0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
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

    Args:
        z (float):
        radius (float):
        low (int):
        high (int):

    Returns:
        tuple: (iz1, iz2, z_offst) - quantized_bounds and subbin_offset
            #Returns: quantized_bounds=(iz1, iz2), subbin_offset

    CommandLine:
        python -m vtool.histogram --test-subbin_bounds

    Example:
        >>> # DISABLE_DOCTEST
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
