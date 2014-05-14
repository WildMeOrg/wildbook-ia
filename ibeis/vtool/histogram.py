# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
# Science
from scipy.signal import argrelextrema
import numpy as np
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[hist]', DEBUG=False)


@profile
def hist_edges_to_centers(edges):
    return np.array([(e1 + e2) / 2 for (e1, e2) in izip(edges[:-1], edges[1:])])


@profile
def wrap_histogram(hist, edges):
    low, high = np.diff(edges)[[0, -1]]
    hist_wrap = np.hstack((hist[-1:], hist, hist[0:1]))
    edge_wrap = np.hstack((edges[0:1] - low, edges, edges[-1:] + high))
    return hist_wrap, edge_wrap


@profile
def hist_interpolated_submaxima(hist, centers=None):
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
    return submaxima_x, submaxima_y


@profile
def hist_argmaxima(hist, centers=None):
    # FIXME: Not handling general cases
    argmaxima = argrelextrema(hist, np.greater)[0]  # [0] index because argrelmaxima returns a tuple
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
    # TODO Use np.polyfit here instead for readability
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


@profile
def subbin_bounds(z, radius, low, high):
    '''
    Gets quantized bounds of a sub-bin/pixel point and a radius.
    Useful for cropping using subpixel points
    Returns: quantized_bounds=(iz1, iz2), subbin_offset

    e.g.
    Illustration: (the bin edges are pipes)
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
                '''
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
