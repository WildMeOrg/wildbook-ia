from __future__ import absolute_import, division, print_function
# Standard
from six.moves import zip
from plottool import draw_func2 as df2
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
    for xtup, ytup in zip(x123.T, y123.T):
        (x1, x2, x3) = xtup  # DUPLICATE CODE!!
        (y1, y2, y3) = ytup  # DUPLICATE CODE!!
        coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
        x_pts = np.linspace(x1, x3, 50)
        y_pts = np.polyval(coeff, x_pts)
        xpoints.append(x_pts)
        ypoints.append(y_pts)

    plt.plot(centers, hist, 'bo-')            # Draw hist
    plt.plot(maxima_x, maxima_y, 'ro')        # Draw maxbin
    plt.plot(submaxima_x, submaxima_y, 'rx')  # Draw maxsubbin
    for x_pts, y_pts in zip(xpoints, ypoints):
        plt.plot(x_pts, y_pts, 'g--')         # Draw parabola


def draw_scores_cdf(scores_list,
                    scores_lbls=None,
                    score_markers=None,
                    score_colors=None,
                    fnum=None):
    """
    Input: a list of scores (either chip or descriptor)

    Concatenates and sorts the scores
    Plots them in a CDF with different types of scores labeled
    """
    if scores_lbls is None:
        scores_lbls = [lblx for lblx in xrange(len(scores_list))]
    if score_markers is None:
        score_markers = ['o' for lbl in xrange(len(scores_list))]
    if score_colors is None:
        score_colors = df2.distinct_colors(len(scores_list))[::-1]
    labelx_list = [[lblx] * len(scores_) for lblx, scores_ in enumerate(scores_list)]
    agg_scores  = np.hstack(scores_list)
    agg_labelx  = np.hstack(labelx_list)

    agg_sortx = agg_scores.argsort()

    sorted_scores = agg_scores[agg_sortx]
    sorted_labelx = agg_labelx[agg_sortx]

    if fnum is None:
        fnum = df2.next_fnum()

    df2.figure(fnum=fnum, doclf=True, docla=True)

    for lblx in xrange(len(scores_list)):
        label = scores_lbls[lblx]
        color = score_colors[lblx]
        marker = score_markers[lblx]
        xdata = np.where(sorted_labelx == lblx)[0]
        ydata = sorted_scores[xdata]
        print('[scores_cdf] lblx=%r label=%r, marker=%r' % (lblx, label, marker))
        df2.plot(xdata, ydata, marker, color=color, label=label, alpha=.7)

    df2.set_logyscale_from_data(sorted_scores)

    df2.set_xlabel('sorted scores')
    df2.set_ylabel('scores')
    df2.dark_background()
    df2.set_figtitle('draw_scores_cdf')
    df2.legend(loc='upper left')
    df2.iup()
