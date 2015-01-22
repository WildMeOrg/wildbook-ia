#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from six.moves import range
from plottool import draw_func2 as df2
from plottool import mpl_keypoint
from plottool import mpl_sift  # NOQA
import vtool.keypoint as ktool  # NOQA
import numpy as np
import matplotlib as mpl  # NOQA
from itertools import product as iprod
import vtool  # NOQA
import utool

TAU = 2 * np.pi  # References: tauday.com

# Hack these directions to be relative to gravity
RIGHT = ((0 * TAU / 4) - ktool.GRAVITY_THETA) % TAU
DOWN  = ((1 * TAU / 4) - ktool.GRAVITY_THETA) % TAU
LEFT  = ((2 * TAU / 4) - ktool.GRAVITY_THETA) % TAU
UP    = ((3 * TAU / 4) - ktool.GRAVITY_THETA) % TAU


def test_keypoint(xscale=1, yscale=1, ori=DOWN, skew=0):
    # Test Keypoint
    x, y = 0, 0
    iv11, iv21, iv22 = xscale, skew, yscale
    kp = np.array([x, y, iv11, iv21, iv22, ori])

    # Test SIFT descriptor
    sift = np.zeros(128)
    sift[ 0: 8]   = 1
    sift[ 8:16]   = .5
    sift[16:24]   = .0
    sift[24:32]   = .5
    sift[32:40]   = .8
    sift[40:48]   = .8
    sift[48:56]   = .1
    sift[56:64]   = .2
    sift[64:72]   = .3
    sift[72:80]   = .4
    sift[80:88]   = .5
    sift[88:96]   = .6
    sift[96:104]  = .7
    sift[104:112] = .8
    sift[112:120] = .9
    sift[120:128] = 1
    sift = sift / np.sqrt((sift ** 2).sum())
    sift = np.round(sift * 255)

    kpts = np.array([kp])
    sifts = np.array([sift])
    return kpts, sifts


def square_axis(ax, s=3):
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    df2.set_xticks([])
    df2.set_yticks([])


def test_shape(ori=0, skew=0, xscale=1, yscale=1, pnum=(1, 1, 1), fnum=1):
    df2.figure(fnum=fnum, pnum=pnum)
    kpts, sifts = test_keypoint(ori=ori, skew=skew, xscale=xscale, yscale=yscale)
    ax = df2.gca()
    square_axis(ax)
    mpl_keypoint.draw_keypoints(ax, kpts, sifts=sifts, ell_color=df2.ORANGE, ori=True,
                                rect_color=df2.DARK_RED,
                                ori_color=df2.DEEP_PINK, eig_color=df2.PINK,
                                rect=True, eig=True, bin_color=df2.RED,
                                arm1_color=df2.YELLOW, arm2_color=df2.BLACK)

    kptsstr = '\n'.join(ktool.get_kpts_strs(kpts))
    #print(kptsstr)
    df2.upperleft_text(kptsstr)

    title = 'xyscale=(%.1f, %.1f),\n skew=%.1f, ori=%.2ftau' % (xscale, yscale, skew, ori / TAU)
    df2.set_title(title)
    df2.dark_background()
    return kpts, sifts


np.set_printoptions(precision=3)
px_ = 0

THETA1 = DOWN
THETA2 = (DOWN + DOWN + RIGHT) / 3
THETA3 = (DOWN + RIGHT) / 2
THETA4 = (DOWN + RIGHT + RIGHT) / 3
THETA5 = RIGHT

nRows = 2
nCols = 4


def pnum_(px=None):
    global px_
    if px is None:
        px_ += 1
        px = px_
    return (nRows, nCols, px)

MIN_ORI = utool.get_argval('--min-ori', float, DOWN)
MAX_ORI = utool.get_argval('--max-ori', float, DOWN + TAU - .2)

MIN_X = .5
MAX_X = 2

MIN_SWEW = utool.get_argval('--min-skew', float, 0)
MAX_SKEW = utool.get_argval('--max-skew', float, 1)

MIN_Y = .5
MAX_Y = 2

kp_list = []

if __name__ == '__main__':

    for row, col in iprod(range(nRows), range(nCols)):
        #print((row, col))
        alpha = col / (nCols)
        beta  = row / (nRows)
        xsca = (MIN_X    * (1 - alpha)) + (MAX_X    * (alpha))
        ori  = (MIN_ORI  * (1 - alpha)) + (MAX_ORI  * (alpha))
        skew = (MIN_SWEW * (1 - beta))  + (MAX_SKEW * (beta))
        ysca = (MIN_Y    * (1 - beta))  + (MAX_Y    * (beta))

        kpts, sifts = test_shape(pnum=pnum_(),
                                 ori=ori,
                                 skew=skew,
                                 xscale=xsca,
                                 yscale=ysca)
    #print('+----')
    #kp_list.append(kpts[0])
    #S_list = ktool.get_xy_axis_extents(kpts)
    #print('xscale=%r yscale=%r, skew=%r' % (xsca, ysca, skew))
    #print(S_list)

    #scale_factor = 1
    #offset = (0, 0)
    #(_xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s, _oris) = ktool.scaled_kpts(kpts, scale_factor, offset)

    #invVR_aff2Ds = mpl_keypoint.get_invV_aff2Ds(_xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s)
    #aff = invVR_aff2Ds[0]

    #ori = _oris[0]
    #aff2 = mpl.transforms.Affine2D().rotate(-ori)

    #print((aff + aff2).frozen())
    #print((aff2 + aff).frozen())

    #mpl_sift.draw_sift(ax, sift)
    #df2.update()

    exec(df2.present())
