from __future__ import division, print_function
# Standard
from itertools import izip
# Matplotlib
import matplotlib.pyplot as plt
import vtool.histogram as htool
import numpy as np


def draw_hist_subbin_maxima(hist, centers=None):
    # Find maxima
    maxima_x, maxima_y, argmaxima = htool.hist_argmaxima(hist, centers)
    # Expand parabola points around submaxima
    x123, y123 = htool.maxima_neighbors(argmaxima, hist, centers)
    # Find submaxima
    submaxima_x, submaxima_y = htool.interpolate_submaxima(argmaxima, hist, centers)
    xpoints = []
    ypoints = []
    for (x1, x2, x3), (y1, y2, y3) in zip(x123.T, y123.T):
        coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
        x_pts = np.linspace(x1, x3, 50)
        y_pts = np.polyval(coeff, x_pts)
        xpoints.append(x_pts)
        ypoints.append(y_pts)

    plt.plot(centers, hist, 'bo-')            # Draw hist
    plt.plot(maxima_x, maxima_y, 'ro')        # Draw maxbin
    plt.plot(submaxima_x, submaxima_y, 'rx')  # Draw maxsubbin
    for x_pts, y_pts in izip(xpoints, ypoints):
        plt.plot(x_pts, y_pts, 'g--')         # Draw parabola
