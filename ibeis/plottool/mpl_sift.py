from __future__ import absolute_import, division, print_function
# Standard
from itertools import product as iprod
from six.moves import zip, range
# Science
import numpy as np
# Matplotlib
import matplotlib as mpl
import utool as ut
ut.noinject(__name__, '[pt.mpl_sift]')


TAU = 2 * np.pi  # References: tauday.com
BLACK  = np.array((0.0, 0.0, 0.0, 1.0))
RED    = np.array((1.0, 0.0, 0.0, 1.0))


def _cirlce_rad2xy(radians, mag):
    return np.cos(radians) * mag, np.sin(radians) * mag


def _set_colltup_list_transform(colltup_list, trans):
    for coll_tup in colltup_list:
        for coll in coll_tup:
            coll.set_transform(trans)


def _draw_colltup_list(ax, colltup_list):
    for coll_tup in colltup_list:
        for coll in coll_tup:
            ax.add_collection(coll)


# Create a patch collection with attributes
def _circl_collection(patch_list, color, alpha):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_alpha(alpha)
    coll.set_edgecolor(color)
    coll.set_facecolor('none')
    return coll


def _arm_collection(patch_list, color, alpha, lw):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_alpha(alpha)
    coll.set_color(color)
    coll.set_linewidth(lw)
    return coll


def get_sift_collection(sift, aff=None, bin_color=BLACK, arm1_color=RED,
                        arm2_color=BLACK, arm_alpha=1.0, arm1_lw=0.5,
                        arm2_lw=1.0, circ_alpha=.5, **kwargs):
    """
    Creates a collection of SIFT matplotlib patches

    get_sift_collection

    Args:
        sift (?):
        aff (None):
        bin_color (ndarray):
        arm1_color (ndarray):
        arm2_color (ndarray):
        arm_alpha (float):
        arm1_lw (float):
        arm2_lw (float):
        circ_alpha (float):

    Returns:
        ?: coll_tup

    Example:
        >>> from plottool.mpl_sift import *  # NOQA
        >>> sift = '?'
        >>> aff = None
        >>> bin_color = array([ 0.,  0.,  0.,  1.])
        >>> arm1_color = array([ 1.,  0.,  0.,  1.])
        >>> arm2_color = array([ 0.,  0.,  0.,  1.])
        >>> arm_alpha = 1.0
        >>> arm1_lw = 0.5
        >>> arm2_lw = 1.0
        >>> circ_alpha = 0.5
        >>> coll_tup = get_sift_collection(sift, aff, bin_color, arm1_color, arm2_color, arm_alpha, arm1_lw, arm2_lw, circ_alpha)
        >>> print(coll_tup)
    """
    # global offset scale adjustments
    if aff is None:
        aff = mpl.transforms.Affine2D()
    _kwarm = kwargs.copy()
    _kwarm.update(dict(head_width=1e-10, length_includes_head=False, transform=aff))
    _kwcirc = dict(transform=aff)
    arm_patches1 = []
    arm_patches2 = []
    DSCALE   =  0.25  # Descriptor scale factor
    ARMSCALE =  1.5   # Arm length scale factor
    XYSCALE  =  0.5   # Position scale factor
    XYOFFST  = -0.75  # Position offset
    NORI, NX, NY = 8, 4, 4  # SIFT BIN CONSTANTS
    NBINS = NX * NY
    discrete_ori = (np.arange(0, NORI) * (TAU / NORI))
    # Arm magnitude and orientations
    arm_mag = sift / 255.0
    arm_ori = np.tile(discrete_ori, (NBINS, 1)).flatten()
    # Arm orientation in dxdy format
    arm_dxy = np.array(list(zip(*_cirlce_rad2xy(arm_ori, arm_mag))))
    # Arm locations and dxdy index
    yxt_gen = iprod(range(NY), range(NX), range(NORI))
    # Circle x,y locations
    yx_gen  = iprod(range(NY), range(NX))
    # Draw 8 directional arms in each of the 4x4 grid cells
    arm_args_list = []
    for y, x, t in yxt_gen:
        #print('y=%r, x=%r, t=%r' % (y, x, t))
        index = (y * NX * NORI) + (x * NORI) + (t)
        (dx, dy) = arm_dxy[index]
        arm_x  = (x * XYSCALE) + XYOFFST  # MULTIPLY BY -1 to invert X axis
        arm_y  = (y * XYSCALE) + XYOFFST
        arm_dy = (dy * DSCALE) * ARMSCALE
        arm_dx = (dx * DSCALE) * ARMSCALE
        _args = [arm_x, arm_y, arm_dx, arm_dy]
        arm_args_list.append(_args)
    for _args in arm_args_list:
        arm_patches1.append(mpl.patches.FancyArrow(*_args, **_kwarm))
        arm_patches2.append(mpl.patches.FancyArrow(*_args, **_kwarm))
    # Draw circles around each of the 4x4 grid cells
    circle_patches = []
    for y, x in yx_gen:
        circ_xy = (x * XYSCALE + XYOFFST, y * XYSCALE + XYOFFST)
        circ_radius = DSCALE
        circle_patches += [mpl.patches.Circle(circ_xy, circ_radius, **_kwcirc)]

    circ_coll = _circl_collection(circle_patches,  bin_color, circ_alpha)
    arm1_coll = _arm_collection(arm_patches1, arm1_color, arm_alpha, arm1_lw)
    arm2_coll = _arm_collection(arm_patches2, arm2_color, arm_alpha, arm2_lw)
    coll_tup = (circ_coll, arm2_coll, arm1_coll)
    return coll_tup


def draw_sifts(ax, sifts, invVR_aff2Ds=None, **kwargs):
    """
    Gets sift patch collections, transforms them and then draws them.
    """
    if invVR_aff2Ds is None:
        invVR_aff2Ds = [mpl.transforms.Affine2D() for _ in range(len(sifts))]
    colltup_list = [get_sift_collection(sift, aff, **kwargs)
                    for sift, aff in zip(sifts, invVR_aff2Ds)]
    ax.invert_xaxis()
    _set_colltup_list_transform(colltup_list, ax.transData)
    _draw_colltup_list(ax, colltup_list)
    ax.invert_xaxis()
