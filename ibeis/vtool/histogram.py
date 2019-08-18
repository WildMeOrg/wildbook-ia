# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import warnings
import scipy.signal
import numpy as np
import utool as ut
import ubelt as ub
from .util_math import TAU


def argsubmax(ydata, xdata=None):
    """
    Finds a single submaximum value to subindex accuracy.
    If xdata is not specified, submax_x is a fractional index.
    Otherwise, submax_x is sub-xdata (essentially doing the index interpolation
    for you)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> ydata = [ 0,  1,  2, 1.5,  0]
        >>> xdata = [00, 10, 20,  30, 40]
        >>> result1 = argsubmax(ydata, xdata=None)
        >>> result2 = argsubmax(ydata, xdata=xdata)
        >>> result = ub.repr2([result1, result2], precision=4, nl=1, nobr=True)
        >>> print(result)
        (2.1667, 2.0208),
        (21.6667, 2.0208),

    Example:
        >>> from vtool.histogram import *  # NOQA
        >>> hist_ = np.array([0, 1, 2, 3, 4])
        >>> centers = None
        >>> maxima_thresh=None
        >>> argsubmax(hist_)
        (4.0, 4.0)
    """
    if len(ydata) == 0:
        raise IndexError('zero length array')
    ydata = np.asarray(ydata)
    xdata = None if xdata is None else np.asarray(xdata)
    submaxima_x, submaxima_y = argsubmaxima(ydata, centers=xdata)
    idx = submaxima_y.argmax()
    submax_y = submaxima_y[idx]
    submax_x = submaxima_x[idx]
    return submax_x, submax_y


def argsubmaxima(hist, centers=None, maxima_thresh=None, _debug=False):
    r"""
    Determines approximate maxima values to subindex accuracy.

    Args:
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels
        maxima_thresh (float): cutoff point for labeing a value as a maxima

    Returns:
        tuple: (submaxima_x, submaxima_y)

    CommandLine:
        python -m vtool.histogram argsubmaxima
        python -m vtool.histogram argsubmaxima --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> maxima_thresh = .8
        >>> hist = np.array([6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> (submaxima_x, submaxima_y) = argsubmaxima(hist, centers, maxima_thresh)
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_hist_subbin_maxima(hist, centers)
        >>> pt.show_if_requested()
        (array([ 3.0318792]), array([ 37.19208239]))
    """
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers, maxima_thresh=maxima_thresh)
    argmaxima = np.asarray(argmaxima)
    if _debug:
        print('Argmaxima: ')
        print(' * maxima_x = %r' % (maxima_x))
        print(' * maxima_y = %r' % (maxima_y))
        print(' * argmaxima = %r' % (argmaxima))
    flags = (argmaxima == 0) | (argmaxima == len(hist) - 1)
    argmaxima_ = argmaxima[~flags]
    submaxima_x_, submaxima_y_ = interpolate_submaxima(argmaxima_, hist, centers)
    if np.any(flags):
        endpts = argmaxima[flags]
        submaxima_x = (np.hstack([submaxima_x_, centers[endpts]])
                       if centers is not None else
                       np.hstack([submaxima_x_, endpts]))
        submaxima_y = np.hstack([submaxima_y_, hist[endpts]])
    else:
        submaxima_y = submaxima_y_
        submaxima_x = submaxima_x_
    if _debug:
        print('Submaxima: ')
        print(' * submaxima_x = %r' % (submaxima_x))
        print(' * submaxima_y = %r' % (submaxima_y))
    return submaxima_x, submaxima_y


def argsubmin2(ydata, xdata=None):
    if len(ydata) == 0:
        raise ValueError('zero length array')
    ydata = np.asarray(ydata)
    xdata = None if xdata is None else np.asarray(xdata)
    submax_x, submax_y = argsubmax2(-ydata, xdata)
    submin_x = submax_x
    submin_y = -submax_y
    return submin_x, submin_y


def argsubmax2(ydata, xdata=None):
    """
    Finds a single submaximum value to subindex accuracy.
    If xdata is not specified, submax_x is a fractional index.
    This version always normalizes x-coordinates.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> ydata = [ 0,  1,  2, 1.5,  0]
        >>> xdata = [00, 10, 20,  30, 40]
        >>> result1 = argsubmax(ydata, xdata=None)
        >>> result2 = argsubmax(ydata, xdata=xdata)
        >>> result = ub.repr2([result1, result2], precision=4, nl=1, nobr=True)
        >>> print(result)
        (2.1667, 2.0208),
        (21.6667, 2.0208),

    Example:
        >>> from vtool.histogram import *  # NOQA
        >>> hist_ = np.array([0, 1, 2, 3, 4])
        >>> centers = None
        >>> thresh_factor = None
        >>> argsubmax(hist_)
        (4.0, 4.0)
    """
    if len(ydata) == 0:
        raise ValueError('zero length array')
    ydata = np.asarray(ydata)
    xdata = None if xdata is None else np.asarray(xdata)
    submaxima_x, submaxima_y = argsubmaxima2(ydata, xdata=xdata,
                                             normalize_x=True)
    idx = submaxima_y.argmax()
    submax_y = submaxima_y[idx]
    submax_x = submaxima_x[idx]
    return submax_x, submax_y


def argsubmaxima2(ydata, xdata=None, thresh_factor=None, normalize_x=True):
    return argsubextrema2('max', ydata, xdata=xdata,
                          thresh_factor=thresh_factor, normalize_x=normalize_x)


def argsubminima2(ydata, xdata=None, thresh_factor=None, normalize_x=True):
    """
    """
    return argsubextrema2('min', ydata, xdata=xdata,
                          thresh_factor=thresh_factor, normalize_x=normalize_x)


def argsubextrema2(op, ydata, xdata=None, thresh_factor=None, normalize_x=True,
                   flat=True):
    r"""
    Determines approximate maxima values to subindex accuracy.

    Args:
        ydata (ndarray): ydata, histogram frequencies
        xdata (ndarray): xdata, histogram labels
        thresh_factor (float): cutoff point for labeing a value as a maxima
        flat (bool): if True allows for flat extrema to be found.

    Returns:
        tuple: (submaxima_x, submaxima_y)

    CommandLine:
        python -m vtool.histogram argsubmaxima
        python -m vtool.histogram argsubmaxima --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> thresh_factor = .8
        >>> ydata = np.array([6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> xdata = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> op = 'min'
        >>> (subextrema_x, subextrema_y) = argsubextrema2(op, ydata, xdata, thresh_factor)
        >>> result = str((subextrema_x, subextrema_y))
        >>> prkeysint(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_hist_subbin_maxima(ydata, xdata)
        >>> pt.show_if_requested()
        (array([ 3.03374251]), array([ 37.2719012]))

    Doctest:
        >>> from vtool.histogram import *  # NOQA
        >>> thresh_factor = .8
        >>> ydata = np.array([1, 1, 1, 2, 1, 2, 3, 2, 4, 1.1, 5, 1.2, 1.1, 1.1, 1.2, 1.1])
        >>> op = 'max'
        >>> thresh_factor = .8
        >>> (subextrema_x, subextrema_y) = argsubextrema2(op, ydata, thresh_factor=thresh_factor)
        >>> result = str((subextrema_x, subextrema_y))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> xdata = np.arange(len(ydata))
        >>> pt.figure(fnum=1, doclf=True)
        >>> pt.plot(xdata, ydata)
        >>> pt.plot(subextrema_x, subextrema_y, 'o')
        >>> ut.show_if_requested()
    """
    if op == 'max':
        cmp = np.greater
        cmp_eq = np.greater_equal
        extreme = np.max
        factor_op = np.multiply
    elif op == 'min':
        cmp = np.less
        cmp_eq = np.less_equal
        extreme = np.min
        factor_op = np.divide
    else:
        raise KeyError(op)

    # absolute extrema
    abs_extreme_y = extreme(ydata)

    if thresh_factor is None:
        thresh_factor = 1.0
        # thresh_factor = None

    if thresh_factor is None:
        thresh_value = None
        flags = np.ones(len(ydata), dtype=np.bool)
    else:
        # Find relative and flat extrema
        thresh_value = factor_op(abs_extreme_y, thresh_factor)
        flags = cmp_eq(ydata, thresh_value)

    # Only consider non-boundary points
    flags[[0, -1]] = False
    mxs = np.where(flags)[0]
    # ydata[np.vstack([mxs - 1, mxs, mxs + 1])]
    lvals, mvals, rvals = ydata[mxs - 1], ydata[mxs], ydata[mxs + 1]
    is_mid_extrema = cmp_eq(mvals, lvals) & cmp_eq(mvals, rvals)
    is_rel_extrema = cmp(mvals, lvals) & cmp(mvals, rvals)
    is_flat_extrema = is_mid_extrema & ~is_rel_extrema

    flat_argextrema = mxs[is_flat_extrema]
    rel_argextrema = mxs[is_rel_extrema]

    # Sublocalize relative extrema
    if len(rel_argextrema) > 0:
        # rel_subextrema_x, rel_subextrema_y = _sublocalize_extrema(
        #     ydata, xdata, rel_argextrema, cmp, normalize_x)
        neighbs = np.vstack((rel_argextrema - 1, rel_argextrema, rel_argextrema + 1))
        y123 = ydata[neighbs]
        if normalize_x or xdata is None:
            x123 = neighbs
        else:
            x123 = xdata[neighbs]

        # Fit parabola around points
        coeff_list = []
        for (x, y) in zip(x123.T, y123.T):
            coeff = np.polyfit(x, y, deg=2)
            coeff_list.append(coeff)

        A, B, C = np.vstack(coeff_list).T

        # Extreme x point is where the derivative is 0
        rel_subextrema_x = -B / (2 * A)
        rel_subextrema_y = C - B * B / (4 * A)

        if xdata is not None and normalize_x:
            # Convert x back to data coordinates if we normalized durring polynimal
            # fitting. Do linear interpoloation.
            rel_subextrema_x = linear_interpolation(xdata, rel_subextrema_x)

        # Check to make sure rel_subextrema is not less extreme than the original
        # extrema (can be the case only if the extrema is incorrectly given)
        # In this case just return what the user wanted as the extrema
        violates = cmp(ydata[rel_argextrema], rel_subextrema_y)
        if np.any(violates):
            warnings.warn(
                'rel_subextrema was less extreme than the measured extrema',
                RuntimeWarning)
            if xdata is not None:
                rel_subextrema_x[violates] = xdata[rel_argextrema[violates]]
            else:
                rel_subextrema_x[violates] = rel_argextrema[violates]
            rel_subextrema_y[violates] = ydata[rel_argextrema[violates]]
    else:
        x123 = None  # for plottool
        coeff_list = []  # for plottool
        rel_subextrema_x, rel_subextrema_y = [], []

    # Check left boundary
    boundary_argextrema = []
    if len(ydata) > 1:
        if thresh_value is None or cmp_eq(ydata[0], thresh_value):
            if cmp_eq(ydata[0], ydata[1]):
                boundary_argextrema.append(0)

    # Check right boundary
    if len(ydata) > 2:
        if thresh_value is None or cmp_eq(ydata[0], thresh_value):
            if cmp_eq(ydata[-1], ydata[-2]):
                boundary_argextrema.append(len(ydata) - 1)

    # Any non-flat mid extrema can be sublocalized
    other_argextrema = np.hstack([boundary_argextrema, flat_argextrema])
    other_argextrema = other_argextrema.astype(np.int)
    other_subextrema_y = ydata[other_argextrema]
    if xdata is None:
        other_subextrema_x = other_argextrema
    else:
        other_subextrema_x = xdata[other_argextrema]

    # Recombine and order all extremas
    subextrema_y = np.hstack([rel_subextrema_y, other_subextrema_y])
    subextrema_x = np.hstack([rel_subextrema_x, other_subextrema_x])
    sortx = subextrema_x.argsort()
    subextrema_x = subextrema_x[sortx]
    subextrema_y = subextrema_y[sortx]
    return subextrema_x, subextrema_y


def _sublocalize_extrema(ydata, xdata, argextrema, cmp, normalize_x=True):
    """
    Assumes argextrema are all relative and not on the boundaries
    """
    # extrema_y = ydata[argextrema]
    # extrema_x = argextrema if xdata is None else xdata[argextrema]

    neighbs = np.vstack((argextrema - 1, argextrema, argextrema + 1))
    y123 = ydata[neighbs]
    if normalize_x or xdata is None:
        x123 = neighbs
    else:
        x123 = xdata[neighbs]

    # Fit parabola around points
    coeff_list = []
    for (x, y) in zip(x123.T, y123.T):
        coeff = np.polyfit(x, y, deg=2)
        coeff_list.append(coeff)

    A, B, C = np.vstack(coeff_list).T

    # Extreme x point is where the derivative is 0
    subextrema_x = -B / (2 * A)
    subextrema_y = C - B * B / (4 * A)

    if xdata is not None and normalize_x:
        # Convert x back to data coordinates if we normalized durring polynimal
        # fitting. Do linear interpoloation.
        subextrema_x = linear_interpolation(xdata, subextrema_x)

    # Check to make sure subextrema is not less extreme than the original
    # extrema (can be the case only if the extrema is incorrectly given)
    # In this case just return what the user wanted as the extrema
    violates = cmp(ydata[argextrema], subextrema_y)
    if np.any(violates):
        warnings.warn(
            'subextrema was less extreme than the measured extrema',
            RuntimeWarning)
        if xdata is not None:
            subextrema_x[violates] = xdata[argextrema[violates]]
        else:
            subextrema_x[violates] = argextrema[violates]
        subextrema_y[violates] = ydata[argextrema[violates]]
    return subextrema_x, subextrema_y


def linear_interpolation(arr, subindices):
    """
    Does linear interpolation to lookup subindex values

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> arr = np.array([0, 1, 2, 3])
        >>> subindices = np.array([0, .1, 1, 1.8, 2, 2.5, 3] )
        >>> subvalues = linear_interpolation(arr, subindices)
        >>> assert np.allclose(subindices, subvalues)
        >>> assert np.allclose(2.3, linear_interpolation(arr, 2.3))
    """
    idx1 = np.floor(subindices).astype(np.int)
    idx2 = np.floor(subindices + 1).astype(np.int)
    idx2 = np.minimum(idx2, len(arr) - 1)
    alpha = idx2 - subindices
    subvalues = arr[idx1] * (alpha) + arr[idx2] * (1 - alpha)
    return subvalues


# subindex_take = linear_interpolation


def hist_argmaxima(hist, centers=None, maxima_thresh=None):
    """
    must take positive only values

    CommandLine:
        python -m vtool.histogram hist_argmaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> maxima_thresh = .8
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
        >>> result = str((maxima_x, maxima_y, argmaxima))
        >>> print(result)
        (array([ 0.39,  2.75]), array([  8.69,  34.62]), array([1, 4]))
    """
    # FIXME: Not handling general cases
    # [0] index because argrelmaxima returns a tuple
    argmaxima_ = scipy.signal.argrelextrema(hist, np.greater)[0]
    if len(argmaxima_) == 0:
        argmaxima_ = hist.argmax()
    if maxima_thresh is not None:
        # threshold maxima to be within a factor of the maximum
        maxima_y = hist[argmaxima_]
        isvalid = maxima_y > maxima_y.max() * maxima_thresh
        argmaxima = argmaxima_[isvalid]
    else:
        argmaxima = argmaxima_
    maxima_y = hist[argmaxima]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    return maxima_x, maxima_y, argmaxima


def hist_argmaxima2(hist, maxima_thresh=.8):
    """
    must take positive only values

    Setup:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA

    GridSearch:
        >>> hist1 = np.array([1, .9, .8, .99, .99, 1.1, .9, 1.0, 1.0])
        >>> hist2 = np.array([1, .9, .8, .99, .99, 1.1, 1.0, 1.0])
        >>> hist2 = np.array([1, .9, .8, .99, .99, 1.1, 1.0])
        >>> hist2 = np.array([1, .9, .8, .99, .99, 1.1, 1.2])
        >>> hist2 = np.array([1, 1.2])
        >>> hist2 = np.array([1, 1, 1.2])
        >>> hist2 = np.array([1])
        >>> hist2 = np.array([])

    Example:
        >>> # ENABLE_DOCTEST
        >>> maxima_thresh = .8
        >>> hist = np.array([1, .9, .8, .99, .99, 1.1, .9, 1.0, 1.0])
        >>> argmaxima = hist_argmaxima2(hist)
        >>> print(argmaxima)
    """
    # FIXME: Not handling general cases
    # [0] index because argrelmaxima returns a tuple
    if len(hist) == 0:
        return np.empty(dtype=np.int)
    comperetor = np.greater
    argmaxima_ = scipy.signal.argrelextrema(hist, comperetor)[0]
    if len(argmaxima_) == 0:
        argmaxima_ = np.array([hist.argmax()])  # Hack for no maxima
    maxval = hist[argmaxima_].max()
    size = len(hist)
    end = size - 1
    # Test if 0 is a maximum point
    if 0 not in argmaxima_ and size > 0:
        start_is_extreme = hist[0] > hist[1]
        if start_is_extreme and hist[0] >= maxval * maxima_thresh:
            argmaxima_ = np.hstack([[0], argmaxima_])
    # Test if end is maximum point
    if end not in argmaxima_ and end > 0:
        #end_is_extreme = np.all(hist[argmaxima_[-1] + 1:(end - 1)] < hist[end] )
        end_is_extreme = hist[end] > hist[end - 1]
        if not end_is_extreme:
            # FIXME: might be a case when end is level
            pass
            #end_is_extreme = np.all(hist[argmaxima_[-1] + 1:(end - 1)] == hist[end] )
        if end_is_extreme and hist[end] >= maxval * maxima_thresh:
            argmaxima_ = np.hstack([argmaxima_, [end]])
    # threshold maxima to be within a factor of the maximum
    maxima_y = hist[argmaxima_]
    isvalid = maxima_y >= maxval * maxima_thresh
    argmaxima = argmaxima_[isvalid]
    return argmaxima


def interpolate_submaxima(argmaxima, hist_, centers=None):
    r"""
    Args:
        argmaxima (ndarray): indicies into ydata / centers that are argmaxima
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels

    FIXME:
        what happens when argmaxima[i] == len(hist_)

    CommandLine:
        python -m vtool.histogram --test-interpolate_submaxima --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> argmaxima = np.array([1, 4, 7])
        >>> hist_ = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist_, centers)
        >>> locals_ = ut.exec_func_src(interpolate_submaxima,
        >>>                            key_list=['x123', 'y123', 'coeff_list'])
        >>> x123, y123, coeff_list = locals_
        >>> res = (submaxima_x, submaxima_y)
        >>> result = ub.repr2(res, nl=1, nobr=True, precision=2, with_dtype=True)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.ensureqt()
        >>> pt.figure(fnum=pt.ensure_fnum(None))
        >>> pt.plot(centers, hist_, '-')
        >>> pt.plot(centers[argmaxima], hist_[argmaxima], 'o', label='argmaxima')
        >>> pt.plot(submaxima_x, submaxima_y, 'b*', markersize=20, label='interp maxima')
        >>> # Extract parabola points
        >>> pt.plt.plot(x123, y123, 'o', label='maxima neighbors')
        >>> xpoints = [np.linspace(x1, x3, 50) for (x1, x2, x3) in x123.T]
        >>> ypoints = [np.polyval(coeff, x_pts) for x_pts, coeff in zip(xpoints, coeff_list)]
        >>> # Draw Submax Parabola
        >>> for x_pts, y_pts in zip(xpoints, ypoints):
        >>>     pt.plt.plot(x_pts, y_pts, 'g--', lw=2)
        >>> pt.show_if_requested()
        np.array([ 0.15,  3.03,  5.11], dtype=np.float64),
        np.array([  9.2 ,  37.19,   0.  ], dtype=np.float64),

    Example:
        >>> hist_ = np.array([5])
        >>> argmaxima = [0]

    """
    if len(argmaxima) == 0:
        return [], []
    argmaxima = np.asarray(argmaxima)
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    # flags = (neighbs[2] > (len(hist_) - 1)) | (neighbs[0] < 0)
    # neighbs = np.clip(neighbs, 0, len(hist_) - 1)
    # if np.any(flags):
    #     # Clip out of bounds positions
    #     neighbs[0, flags] = neighbs[1, flags]
    #     neighbs[2, flags] = neighbs[1, flags]
    y123 = hist_[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    # if np.any(flags):
    #     # Make symetric values so maxima is found exactly in center
    #     y123[0, flags] = y123[1, flags] - 1
    #     y123[2, flags] = y123[1, flags] - 1
    #     x123[0, flags] = x123[1, flags] - 1
    #     x123[2, flags] = x123[1, flags] - 1
    # Fit parabola around points
    coeff_list = [np.polyfit(x123_, y123_, deg=2)
                  for (x123_, y123_) in zip(x123.T, y123.T)]
    A, B, C = np.vstack(coeff_list).T
    submaxima_x, submaxima_y = maximum_parabola_point(A, B, C)

    # Check to make sure submaxima is not less than original maxima
    # (can be the case only if the maxima is incorrectly given)
    # In this case just return what the user wanted as the maxima
    maxima_y = y123[1, :]
    invalid = submaxima_y < maxima_y
    if np.any(invalid):
        if centers is not None:
            submaxima_x[invalid] = centers[argmaxima[invalid]]
        else:
            submaxima_x[invalid] = argmaxima[invalid]
        submaxima_y[invalid] = hist_[argmaxima[invalid]]
    return submaxima_x, submaxima_y


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
        python -m vtool.histogram --test-show_hist_submaxima --dpath figures --save ~/latex/crall-candidacy-2015/figures/show_hist_submaxima.jpg

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.histogram import *  # NOQA
        >>> hist_ = np.array(list(map(float, ut.get_argval('--hist', type_=list, default=[1, 4, 2, 5, 3, 3]))))
        >>> edges = np.array(list(map(float, ut.get_argval('--edges', type_=list, default=[0, 1, 2, 3, 4, 5, 6]))))
        >>> maxima_thresh = ut.get_argval('--maxima_thresh', type_=float, default=.8)
        >>> centers = None
        >>> show_hist_submaxima(hist_, edges, centers, maxima_thresh)
        >>> pt.show_if_requested()
    """
    import plottool as pt
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
    centers = hist_edges_to_centers(edges)
    hist_str   = 'hist    = ' + str(hist.tolist())
    center_str = 'centers = ' + str(centers.tolist())
    edge_str   = 'edges   = [' +  ', '.join(['%.2f' % _ for _ in edges]) + ']'
    histinfo_str = hist_str + ut.NEWLINE + center_str + ut.NEWLINE + edge_str
    return histinfo_str


# def interpolated_wrapped_histogram2():
#     # another stab at it
#     data = np.array([ 0, .5,  2,  3.5,  3,  3,  3.9,  4])
#     weights = np.ones(len(data))

#     n_bins = 4
#     # max_val = np.pi * 2
#     max_val = 4
#     # Define the edges of each bin (we will vote into the middle)
#     # The first and last item are equivalent

#     # bin_centers = np.linspace(0, max_val, n_bins + 1, endpoint=True)
#     bin_centers = np.array([0, 1.5, 2, 3.5, 4])
#     bin_offsets = np.diff(bin_centers) / 2

#     # sort input data
#     sortx = np.argsort(data)
#     sa = data[sortx]


#     right_edges = bin_centers[:-1] + bin_offsets
#     wrapped_right = bin_centers[-1] + bin_offsets[0]

#     # Find data in the wrap-around zone
#     is_right_wrapped = (sa > right_edges[-1])

#     rectified_left = sa[is_right_wrapped] - max_val
#     # rectified_right = max_val - data[is_left_wrapped]

#     right_idx = right_edges.searchsorted(sa)



#     right_edges[-1] - data[]

#     data % right_edges[-1]

#     left_offsets = np.hstack([bin_offsets[-1], bin_offsets])
#     right_offsets = np.hstack([bin_offsets, bin_offsets[0]])

#     left_edges = bin_centers - left_offsets
#     right_edges = bin_centers + right_offsets


#     bin_edges = np.roll(bin_centers[1:] + bin_offsets, 1)


#     first_left_edge =

#     rssign_left = left_edges.searchsorted(sa)
#     assign_right = right_edges.searchsorted(sa)

#     # Find bins to the left and right of each point
#     assign_left  = sa.searchsorted(bin_edges[:-1], 'left')
#     assign_right = sa.searchsorted(bin_edges[-1], 'right')


def interpolated_histogram(data, weights, range_, bins,
                           interpolation_wrap=True, _debug=False):
    r"""
    Follows np.histogram, but does interpolation

    Args:
        data (ndarray):
        weights (ndarray):
        range_ (tuple): range from 0 to 1
        bins (int):
        interpolation_wrap (bool): (default = True)
        _debug (bool): (default = False)

    CommandLine:
        python -m vtool.histogram --test-interpolated_histogram

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> data = np.array([ 0,  1,  2,  3.5,  3,  3,  4,  4])
        >>> weights = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> range_ = (0, 4)
        >>> bins = 5
        >>> interpolation_wrap = False
        >>> hist, edges = interpolated_histogram(data, weights, range_, bins,
        >>>                                      interpolation_wrap)
        >>> assert np.abs(hist.sum() - weights.sum()) < 1E-9
        >>> assert hist.size == bins
        >>> assert edges.size == bins + 1
        >>> result = get_histinfo_str(hist, edges)
        >>> print(result)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> data = np.array([ 0,  1,  2,  3.5,  3,  3,  4,  4])
        >>> weights = np.array([4.5, 1., 1., 1., 1., 1., 1., 1.])
        >>> range_ = (-.5, 4.5)
        >>> bins = 5
        >>> interpolation_wrap = True
        >>> hist, edges = interpolated_histogram(data, weights, range_, bins,
        >>>                                      interpolation_wrap)
        >>> assert np.abs(hist.sum() - weights.sum()) < 1E-9
        >>> assert hist.size == bins
        >>> assert edges.size == bins + 1
        >>> result = get_histinfo_str(hist, edges)
        >>> print(result)
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

    if _debug:
        print('bins = %r' % bins)
        print('step = %r' % step)
        print('half_step = %r' % half_step)
        print('data_offset = %r' % data_offset)
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
    if _debug:
        import vtool as vt
        assert np.allclose(np.diff(edges), step)
        print(hist.shape)
        print(edges.shape)
        print(vt.kpts_docrepr(hist, 'hist', False))
        print(vt.kpts_docrepr(edges, 'edges', False))

    return hist, edges


def hist_edges_to_centers(edges):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> edges = [-0.79, 0.00, 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.50, 6.28, 7.07]
        >>> centers = hist_edges_to_centers(edges)
        >>> result = str(centers)
        >>> print(result)
        [-0.395  0.395  1.18   1.965  2.75   3.535  4.32   5.105  5.89   6.675]
    """
    centers = np.array([(e1 + e2) / 2.0 for (e1, e2) in zip(edges[:-1], edges[1:])])
    return centers


def wrap_histogram(hist_, edges_, _debug=False):
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
        >>> hist_ = np.array([8., 0., 0., 34.32, 29.45, 0., 0., 6.73])
        >>> edges_ = np.array([ 0.        ,  0.78539816,  1.57079633,
        ...                    2.35619449,  3.14159265,  3.92699081,
        ...                    4.71238898,  5.49778714,  6.2831853 ])
        >>> (hist_wrap, edge_wrap) = wrap_histogram(hist_, edges_)
        >>> tup = (hist_wrap.tolist(), edge_wrap.tolist())
        >>> result = ub.repr2(tup, nl=1, nobr=True, precision=2)
        >>> print(result)
        [6.73, 8.00, 0.00, 0.00, 34.32, 29.45, 0.00, 0.00, 6.73, 8.00],
        [-0.79, 0.00, 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.50, 6.28, 7.07],
    """
    # FIXME; THIS NEEDS INFORMATION ABOUT THE DISTANCE FROM THE LAST BIN
    # TO THE FIRST. IT IS OK AS LONG AS ALL STEPS ARE EQUAL, BUT IT IS NOT
    # GENERAL
    left_step, right_step = np.diff(edges_)[[0, -1]]
    hist_wrap = np.hstack((hist_[-1:], hist_, hist_[0:1]))
    edge_wrap = np.hstack((edges_[0:1] - left_step, edges_, edges_[-1:] + right_step))
    if _debug:
        import vtool as vt
        print(vt.kpts_docrepr(hist_wrap, 'hist_wrap', False))
        print(vt.kpts_docrepr(edge_wrap, 'edge_wrap', False))
    return hist_wrap, edge_wrap


def maxima_neighbors(argmaxima, hist_, centers=None):
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    y123 = hist_[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    return x123, y123


def maximum_parabola_point(A, B, C):
    """ Maximum x point is where the derivative is 0 """
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    return xv, yv


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
        >>> z = 1.5
        >>> radius = 5.666
        >>> low = 0
        >>> high = 7
        >>> (iz1, iz2, z_offst) = subbin_bounds(z, radius, low, high)
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


def show_ori_image_ondisk():
    r"""
    CommandLine:
        python -m vtool.histogram --test-show_ori_image_ondisk --show

        python -m vtool.histogram --test-show_ori_image_ondisk --show --patch_img_fpath patches/KP_0_PATCH.png --ori_img_fpath patches/KP_0_orientations01.png --weights_img_fpath patches/KP_0_WEIGHTS.png --grady_img_fpath patches/KP_0_ygradient.png --gradx_img_fpath patches/KP_0_xgradient.png --title cpp_show_ori_ondisk

        python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> result = show_ori_image_ondisk()
        >>> pt.show_if_requested()
    """
    #if img_fpath is not None:
    #    img_fpath = ut.get_argval('--fpath', type_=str, default=ut.grab_test_imgpath('star.png'))
    #    img_fpath = ut.get_argval('--fpath', type_=str, default=ut.grab_test_imgpath('star.png'))
    #    img = vt.imread(img_fpath)
    #    ori_img_fpath     = ut.get_argval('--fpath-ori', type_=str,
    #    default=ut.augpath(img_fpath, '_ori'))
    #    weights_img_fpath = ut.get_argval('--fpath-weight', type_=str,
    #    default=ut.augpath(img_fpath, '_mag'))
    #    vt.imwrite(ori_img_fpath, vt.patch_ori(*vt.patch_gradient(img)))
    #    vt.imwrite(weights_img_fpath, vt.patch_mag(*vt.patch_gradient(img)))
    import vtool as vt
    print('show_ori_image_ondisk')
    def parse_img_from_arg(argstr_):
        fpath = ut.get_argval(argstr_, type_=str, default='None')
        if fpath is not None and fpath != 'None':
            img = vt.imread(fpath, grayscale=True)
            print('Reading %s with stats %s' % (fpath, ut.get_stats_str(img, axis=None)))
        else:
            print('Did not read %s' % (fpath))
            img = None
        return img
    patch = parse_img_from_arg('--patch_img_fpath')
    gori  = parse_img_from_arg('--ori_img_fpath') / 255.0 * TAU
    weights = parse_img_from_arg('--weights_img_fpath') / 255.0
    gradx = parse_img_from_arg('--gradx_img_fpath') / 255.0
    grady = parse_img_from_arg('--grady_img_fpath') / 255.0
    gauss = parse_img_from_arg('--gauss_weights_img_fpath') / 255.0
    #print(' * ori_img_fpath = %r' % (ori_img_fpath,))
    #print(' * weights_img_fpath = %r' % (weights_img_fpath,))
    #print(' * gradx_img_fpath = %r' % (gradx_img_fpath,))
    #print(' * grady_img_fpath = %r' % (grady_img_fpath,))
    #import cv2
    #cv2.imread(ori_img_fpath, cv2.IMREAD_UNCHANGED)
    show_ori_image(gori, weights, patch, gradx, grady, gauss)
    title = ut.get_argval('--title', default='')
    import plottool as pt
    pt.set_figtitle(title)


def show_ori_image(gori, weights, patch, gradx=None, grady=None, gauss=None, fnum=None):
    """
    CommandLine:
        python -m pyhesaff._pyhesaff --test-test_rot_invar --show --nocpp
    """
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    print('gori.max = %r' % gori.max())
    assert gori.max() <= TAU
    assert gori.min() >= 0
    bgr_ori = pt.color_orimag(gori, weights, False, encoding='bgr')
    print('bgr_ori.max = %r' % bgr_ori.max())

    bgr_ori = (255 * bgr_ori).astype(np.uint8)
    print('bgr_ori.max = %r' % bgr_ori.max())
    #bgr_ori = np.array(bgr_ori, dtype=np.uint8)
    legend = pt.make_ori_legend_img()
    #gorimag_, woff, hoff = vt.stack_images(bgr_ori, legend, vert=False, modifysize=True)
    import vtool as vt
    gorimag_, offsets, sftup = vt.stack_images(bgr_ori, legend, vert=False,
                                               modifysize=True,
                                               return_sf=True)
    (woff, hoff) = offsets[1]
    if patch is None:
        pt.imshow(gorimag_, fnum=fnum)
    else:
        pt.imshow(gorimag_, fnum=fnum, pnum=(3, 1, 1), title='colored by orientation')
        #pt.imshow(patch, fnum=fnum, pnum=(2, 2, 1))
        #gradx, grady = np.cos(gori + TAU / 4.0), np.sin(gori + TAU / 4.0)
        if gradx is not None and grady is not None:
            if weights is not None:
                gradx *= weights
                grady *= weights
            pt.imshow(np.array(gradx * 255, dtype=np.uint8), fnum=fnum, pnum=(3, 3, 4))
            pt.imshow(np.array(grady * 255, dtype=np.uint8), fnum=fnum, pnum=(3, 3, 5))
            #pt.imshow(bgr_ori, pnum=(2, 2, 4))
            pt.draw_vector_field(gradx, grady, pnum=(3, 3, 6), invert=True)
        pt.imshow(patch, fnum=fnum, pnum=(3, 1, 3))


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.histogram
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
