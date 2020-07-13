# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range
import itertools as it
import numpy as np
import matplotlib as mpl
import utool as ut
from wbia.plottool import color_funcs as color_fns

ut.noinject(__name__, '[pt.mpl_sift]')


TAU = 2 * np.pi  # References: tauday.com
BLACK = np.array((0.0, 0.0, 0.0, 1.0))
RED = np.array((1.0, 0.0, 0.0, 1.0))


def testdata_sifts():
    # make random sifts
    randstate = np.random.RandomState(1)
    sifts_float = randstate.rand(1, 128)
    sifts_float = sifts_float / np.linalg.norm(sifts_float)
    sifts_float[sifts_float > 0.2] = 0.2
    sifts_float = sifts_float / np.linalg.norm(sifts_float)
    sifts = (sifts_float * 512).astype(np.uint8)
    return sifts


# Create a patch collection with attributes
def _circl_collection(patch_list, color, alpha):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_alpha(alpha)
    coll.set_edgecolor(color)
    coll.set_facecolor('none')
    return coll


def get_sift_collection(
    sift,
    aff=None,
    bin_color=BLACK,
    arm1_color=RED,
    arm2_color=BLACK,
    arm_alpha=1.0,
    arm1_lw=1.0,
    arm2_lw=2.0,
    stroke=1.0,
    circ_alpha=0.5,
    fidelity=256,
    scaling=True,
    **kwargs,
):
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
        fidelity (int): quantization factor

    Returns:
        ?: coll_tup

    CommandLine:
        python -m wbia.plottool.mpl_sift --test-get_sift_collection

    Example:
        >>> from wbia.plottool.mpl_sift import *  # NOQA
        >>> sift = testdata_sifts()[0]
        >>> aff = None
        >>> bin_color = np.array([ 0.,  0.,  0.,  1.])
        >>> arm1_color = np.array([ 1.,  0.,  0.,  1.])
        >>> arm2_color = np.array([ 0.,  0.,  0.,  1.])
        >>> arm_alpha = 1.0
        >>> arm1_lw = 0.5
        >>> arm2_lw = 1.0
        >>> circ_alpha = 0.5
        >>> coll_tup = get_sift_collection(sift, aff, bin_color, arm1_color,
        >>>                                arm2_color, arm_alpha, arm1_lw,
        >>>                                arm2_lw, circ_alpha)
        >>> print(coll_tup)
    """
    # global offset scale adjustments
    if aff is None:
        aff = mpl.transforms.Affine2D()
    MULTI_COLORED_ARMS = kwargs.pop('multicolored_arms', False)
    _kwarm = kwargs.copy()
    _kwarm.update(
        dict(head_width=1e-10, length_includes_head=False, transform=aff, color=[1, 1, 0])
    )
    _kwcirc = dict(transform=aff)
    DSCALE = 0.25  # Descriptor scale factor
    XYSCALE = 0.5  # Position scale factor
    XYOFFST = -0.75  # Position offset
    NORI, NX, NY = 8, 4, 4  # SIFT BIN CONSTANTS
    NBINS = NX * NY
    discrete_ori = np.arange(0, NORI) * (TAU / NORI)

    # import utool
    # utool.embed()
    # Arm magnitude and orientations
    # arm_mag = sift / 255.0
    # If given the correct fidelity, each arm will have a max magnitude of 1.0
    # Because the diameter of each circle is 1.0
    arm_mag = sift / (float(fidelity))
    # arm_mag = sift / 512.0  # technically correct
    # arm_mag = sift / 256.0  # but use this instead as it is max bin

    arm_ori = np.tile(discrete_ori, (NBINS, 1)).flatten()

    if scaling and False:
        # Use entropy as a scaling factor to more clearly visualize differences
        p = np.bincount(sift) / len(sift)
        maximum_entropy = -np.log2(1 / len(sift))
        entropy = -np.nansum(p * np.log2(p))
        # Alpha is 1 when entropy is maximum
        # When entropy is maximum, we want to scale things up a bit
        alpha = entropy / maximum_entropy

        max_ = arm_mag.max()
        # Always scale up, but no more than max.
        pt1 = min(max_, 1)  # max_
        pt2 = max(max_, 1)  # 1
        denom = (pt2 * alpha) + (pt1 * (1 - alpha))
        scale_factor = 1 / denom
        arm_mag = arm_mag * scale_factor
    # arm_mag *= 4

    ori_dxy = np.hstack([np.cos(arm_ori)[:, None], np.sin(arm_ori)[:, None]])

    # Arm orientation in dxdy format
    # arm_dx = np.cos(arm_ori) * arm_mag
    # arm_dy = np.sin(arm_ori) * arm_mag
    # arm_dxy = np.hstack([arm_dx[:, None], arm_dy[:, None]])
    # assert np.all(np.isclose(np.sqrt(arm_dy ** 2 + arm_dx ** 2), arm_mag))
    # np.linalg.norm(arm_dxy, axis=1).max()

    # Arm locations and dxdy index
    yxt_gen = it.product(range(NY), range(NX), range(NORI))
    # Circle x,y locations
    yx_gen = it.product(range(NY), range(NX))
    # Draw 8 directional arms in each of the 4x4 grid cells

    # MOVETO = mpl.path.Path.MOVETO
    # LINETO = mpl.path.Path.LINETO
    # STOP = mpl.path.Path.STOP

    arm_patches = []
    for y, x, t in yxt_gen:
        index = (y * NX * NORI) + (x * NORI) + (t)
        (dx, dy) = ori_dxy[index]
        mag = arm_mag[index]
        # (mdx, mdy) = arm_dxy[index]
        arm_x = (x * XYSCALE) + XYOFFST  # MULTIPLY BY -1 to invert X axis
        arm_y = (y * XYSCALE) + XYOFFST
        arm_dy = dy * mag * DSCALE
        arm_dx = dx * mag * DSCALE

        # Move arms a little bit away from the center
        nudge = 0.05
        arm_x = arm_x + dx * (nudge * DSCALE)
        arm_y = arm_y + dy * (nudge * DSCALE)
        arm_dx = arm_dx + dx * (nudge * DSCALE)
        arm_dy = arm_dy + dy * (nudge * DSCALE)

        if 0:
            _args = [arm_x, arm_y, arm_dx, arm_dy]
            arm_patch = mpl.patches.FancyArrow(*_args, **_kwarm)
        else:
            arm_x2 = arm_x + arm_dx
            arm_y2 = arm_y + arm_dy
            pt1 = np.array([arm_x, arm_y])
            pt2 = np.array([arm_x2, arm_y2])
            # Hack a small eps rectangle to make the ends of the line also have
            # a stroke
            eps = 1e-6
            verts = [pt1, pt2, pt2 + eps, pt2 - eps]
            path = mpl.path.Path(verts, closed=True)
            arm_patch = mpl.patches.PathPatch(
                path,
                # joinstyle='bevel',
                # joinstyle='round',
                edgecolor='k',
                transform=aff,
            )
        arm_patches.append(arm_patch)

    # Draw circles around each of the 4x4 grid cells
    circle_patches = []
    for y, x in yx_gen:
        circ_xy = (x * XYSCALE + XYOFFST, y * XYSCALE + XYOFFST)
        circ_radius = DSCALE
        patch = mpl.patches.Circle(circ_xy, circ_radius, **_kwcirc)
        circle_patches.append(patch)

    circ_coll = mpl.collections.PatchCollection(circle_patches)
    circ_coll.set_alpha(circ_alpha)
    circ_coll.set_edgecolor(bin_color)
    circ_coll.set_facecolor('none')

    # arm2_coll = _arm_collection(arm_patches, arm2_color, arm_alpha, arm2_lw)

    # Add stroke instead of another arm
    path_effects = []

    ENABLE_PATH_EFFECTS = 0
    if ENABLE_PATH_EFFECTS:
        from matplotlib import patheffects

        print('stroke = %r' % (stroke,))
        if stroke > 0:
            path_effects.append(
                patheffects.withStroke(linewidth=arm1_lw + stroke, foreground='k')
            )
        path_effects.append(patheffects.Normal())

    if MULTI_COLORED_ARMS:
        # Hack in same colorscheme for arms as the sift bars
        ori_colors = color_fns.distinct_colors(16)
        arm_collections = [
            mpl.collections.PatchCollection(patches)
            for patches in ut.ichunks(arm_patches, 8)
        ]
        for col, color in zip(arm_collections, ori_colors):
            col.set_color(color)
    else:
        # Just use a single color for all the arms
        arm1_coll = mpl.collections.PatchCollection(arm_patches)
        arm1_coll.set_color(arm1_color)
        arm_collections = [arm1_coll]

    for col in arm_collections:
        col.set_alpha(arm_alpha)
        col.set_path_effects(path_effects)
        col.set_linewidth(arm1_lw)

    coll_tup = [circ_coll] + arm_collections
    return coll_tup


def draw_sifts(ax, sifts, invVR_aff2Ds=None, **kwargs):
    """
    Gets sift patch collections, transforms them and then draws them.

    CommandLine:
        python -m wbia.plottool.mpl_sift --test-draw_sifts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.mpl_sift import *  # NOQA
        >>> # build test data
        >>> import wbia.plottool as pt
        >>> pt.figure(1)
        >>> ax = pt.gca()
        >>> ax.set_xlim(-1.1, 1.1)
        >>> ax.set_ylim(-1.1, 1.1)
        >>> sifts = testdata_sifts()
        >>> sifts[:, 0:8] = 0
        >>> invVR_aff2Ds = None
        >>> kwargs = dict(multicolored_arms=False)
        >>> kwargs['arm1_lw'] = 3
        >>> kwargs['stroke'] = 5
        >>> result = draw_sifts(ax, sifts, invVR_aff2Ds, **kwargs)
        >>> ax.set_aspect('equal')
        >>> print(result)
        >>> pt.show_if_requested()
    """
    if invVR_aff2Ds is None:
        invVR_aff2Ds = [mpl.transforms.Affine2D() for _ in range(len(sifts))]
    if isinstance(invVR_aff2Ds, (list, np.ndarray)):
        invVR_aff2Ds = [mpl.transforms.Affine2D(matrix=aff_) for aff_ in invVR_aff2Ds]
    colltup_list = [
        get_sift_collection(sift, aff, **kwargs) for sift, aff in zip(sifts, invVR_aff2Ds)
    ]
    ax.invert_xaxis()

    for coll_tup in colltup_list:
        for coll in coll_tup:
            coll.set_transform(ax.transData)

    for coll_tup in colltup_list:
        for coll in coll_tup:
            ax.add_collection(coll)
    ax.invert_xaxis()


def draw_sift_on_patch(patch, sift, **kwargs):
    import wbia.plottool as pt

    pt.imshow(patch)
    ax = pt.gca()
    half_size = patch.shape[0] / 2
    invVR = np.array([[half_size, 0, half_size], [0, half_size, half_size], [0, 0, 1]])
    invVR_aff2Ds = np.array([invVR])
    sifts = np.array([sift])
    return draw_sifts(ax, sifts, invVR_aff2Ds)


def render_sift_on_patch(patch, sift):
    import wbia.plottool as pt

    with pt.RenderingContext() as render:
        draw_sift_on_patch(patch, sift)
    rendered = render.image
    return rendered


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.mpl_sift
        python -m wbia.plottool.mpl_sift --allexamples
        python -m wbia.plottool.mpl_sift --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
