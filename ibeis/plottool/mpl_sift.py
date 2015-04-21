from __future__ import absolute_import, division, print_function
# Standard
from itertools import product as iprod
from six.moves import zip, range
# Science
import numpy as np
# Matplotlib
import matplotlib as mpl
import utool as ut
from plottool import color_funcs as color_fns  # NOQA
ut.noinject(__name__, '[pt.mpl_sift]')


TAU = 2 * np.pi  # References: tauday.com
BLACK  = np.array((0.0, 0.0, 0.0, 1.0))
RED    = np.array((1.0, 0.0, 0.0, 1.0))


def testdata_sifts():
    # make random sifts
    randstate = np.random.RandomState(1)
    sifts_float = randstate.rand(1, 128)
    sifts_float = sifts_float / np.linalg.norm(sifts_float)
    sifts_float[sifts_float > .2] = .2
    sifts_float = sifts_float / np.linalg.norm(sifts_float)
    sifts = (sifts_float * 512).astype(np.uint8)
    return sifts


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
                        arm2_color=BLACK, arm_alpha=1.0, arm1_lw=1.0,
                        arm2_lw=2.0, circ_alpha=.5, **kwargs):
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

    CommandLine:
        python -m plottool.mpl_sift --test-get_sift_collection

    Example:
        >>> from plottool.mpl_sift import *  # NOQA
        >>> sift = testdata_sifts()[0]
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
    MULTI_COLORED_ARMS = kwargs.pop('multicolored_arms', False)
    _kwarm = kwargs.copy()
    _kwarm.update(dict(head_width=1e-10, length_includes_head=False, transform=aff, color=[1, 1, 0]))
    _kwcirc = dict(transform=aff)
    arm_patches = []
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
        arm_patch = mpl.patches.FancyArrow(*_args, **_kwarm)
        arm_patches.append(arm_patch)

    #print('len(arm_patches) = %r' % (len(arm_patches),))
    # Draw circles around each of the 4x4 grid cells
    circle_patches = []
    for y, x in yx_gen:
        circ_xy = (x * XYSCALE + XYOFFST, y * XYSCALE + XYOFFST)
        circ_radius = DSCALE
        circle_patches += [mpl.patches.Circle(circ_xy, circ_radius, **_kwcirc)]

    circ_coll = _circl_collection(circle_patches,  bin_color, circ_alpha)
    arm2_coll = _arm_collection(arm_patches, arm2_color, arm_alpha, arm2_lw)

    if MULTI_COLORED_ARMS:
        # Hack in same colorscheme for arms as the sift bars
        ori_colors = color_fns.distinct_colors(16)
        coll_tup = [circ_coll, arm2_coll]
        coll_tup += [_arm_collection(_, color, arm_alpha, arm1_lw)
                     for _, color in zip(ut.ichunks(arm_patches, 8), ori_colors)]
        coll_tup = tuple(coll_tup)
    else:
        # Just use a single color for all the arms
        arm1_coll = _arm_collection(arm_patches, arm1_color, arm_alpha, arm1_lw)
        coll_tup = (circ_coll, arm2_coll, arm1_coll)
    return coll_tup


def draw_sifts(ax, sifts, invVR_aff2Ds=None, **kwargs):
    """
    Gets sift patch collections, transforms them and then draws them.

    CommandLine:
        python -m plottool.mpl_sift --test-draw_sifts --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.mpl_sift import *  # NOQA
        >>> # build test data
        >>> import plottool as pt
        >>> pt.figure(1)
        >>> ax = pt.gca()
        >>> ax.set_xlim(-1.5, 1.5)
        >>> ax.set_ylim(-1.5, 1.5)
        >>> sifts = testdata_sifts()
        >>> invVR_aff2Ds = None
        >>> #kwargs = dict(arm1_lw=1, arm2_lw=2)
        >>> kwargs = dict(multicolored_arms=False)
        >>> # execute function
        >>> result = draw_sifts(ax, sifts, invVR_aff2Ds, **kwargs)
        >>> # verify results
        >>> print(result)
        >>> #pt.dark_background()
        >>> pt.show_if_requested()
    """
    if invVR_aff2Ds is None:
        invVR_aff2Ds = [mpl.transforms.Affine2D() for _ in range(len(sifts))]
    colltup_list = [get_sift_collection(sift, aff, **kwargs)
                    for sift, aff in zip(sifts, invVR_aff2Ds)]
    ax.invert_xaxis()
    _set_colltup_list_transform(colltup_list, ax.transData)
    _draw_colltup_list(ax, colltup_list)
    ax.invert_xaxis()


if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.mpl_sift
        python -m plottool.mpl_sift --allexamples
        python -m plottool.mpl_sift --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
