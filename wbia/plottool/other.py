# -*- coding: utf-8 -*-
# I'm not quite sure how to organize these functions yet
from __future__ import absolute_import, division, print_function
from six.moves import zip
import numpy as np
import cv2
import matplotlib.pyplot as plt
import vtool.histogram as htool
import utool as ut

ut.noinject(__name__, '[pt.other]')


def color_orimag(gori, gmag):
    # Turn a 0 to 1 orienation map into hsv colors
    gori_01 = (gori - gori.min()) / (gori.max() - gori.min())
    cmap_ = plt.get_cmap('hsv')
    flat_rgb = np.array(cmap_(gori_01.flatten()), dtype=np.float32)
    rgb_ori_alpha = flat_rgb.reshape(np.hstack((gori.shape, [4])))
    rgb_ori = cv2.cvtColor(rgb_ori_alpha, cv2.COLOR_RGBA2RGB)
    hsv_ori = cv2.cvtColor(rgb_ori, cv2.COLOR_RGB2HSV)
    # Desaturate colors based on magnitude
    hsv_ori[:, :, 1] = gmag / 255.0
    hsv_ori[:, :, 2] = gmag / 255.0
    # Convert back to bgr
    bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2RGB)
    return bgr_ori


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
        (x1, x2, x3) = xtup
        (y1, y2, y3) = ytup
        coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
        x_pts = np.linspace(x1, x3, 50)
        y_pts = np.polyval(coeff, x_pts)
        xpoints.append(x_pts)
        ypoints.append(y_pts)

    plt.plot(centers, hist, 'bo-')  # Draw hist
    plt.plot(maxima_x, maxima_y, 'ro')  # Draw maxbin
    plt.plot(submaxima_x, submaxima_y, 'rx')  # Draw maxsubbin
    for x_pts, y_pts in zip(xpoints, ypoints):
        plt.plot(x_pts, y_pts, 'g--')  # Draw parabola
