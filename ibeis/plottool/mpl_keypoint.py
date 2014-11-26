from __future__ import absolute_import, division, print_function
# Standard
from six.moves import zip
# Science
import numpy as np
# Matplotlib
import matplotlib as mpl
# vtool
import vtool.keypoint as ktool
from plottool import mpl_sift
import utool as ut
ut.noinject(__name__, '[pt.mpl_keypoint]')


# TOOD: move to util
def pass_props(dict1, dict2, *args):
    # Passes props from one kwargs dict to the next
    for key in args:
        if key in dict1:
            dict2[key] = dict1[key]


def _draw_patches(ax, patch_list, color, alpha, lw, fcolor='none'):
    # creates a collection from a patch list and sets properties
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_facecolor(fcolor)
    coll.set_alpha(alpha)
    coll.set_linewidth(lw)
    coll.set_edgecolor(color)
    coll.set_transform(ax.transData)
    ax.add_collection(coll)


#----------------------------
def draw_keypoints(ax, kpts, scale_factor=1.0, offset=(0.0, 0.0), rotation=0.0,
                   ell=True, pts=False, rect=False, eig=False, ori=False,
                   sifts=None, siftkw={}, **kwargs):
    """
    draws keypoints extracted by pyhesaff onto a matplotlib axis
    Args:
        ax (?):
        kpts (?):
        scale_factor (float):
        offset (tuple):
        rotation (float):
        ell (bool):
        pts (bool):
        rect (bool):
        eig (bool):
        ori (bool):
        sifts (None):

    Example:
        >>> from plottool.mpl_keypoint import *  # NOQA
        >>> ax
        >>> kpts
        >>> scale_factor=1.0
        >>> offset=(0.0, 0.0)
        >>> rotation=0.0
        >>> ell=True
        >>> pts=False
        >>> rect=False
        >>> eig=False
        >>> ori=False
        >>> sifts=None
        >>>
    """
    # ellipse and point properties
    pts_size       = kwargs.get('pts_size', 2)
    ell_color      = kwargs.get('ell_color', None)
    ell_alpha      = kwargs.get('ell_alpha', 1)
    ell_linewidth  = kwargs.get('ell_linewidth', 2)
    # colors
    pts_color      = kwargs.get('pts_color',  ell_color)
    rect_color     = kwargs.get('rect_color', ell_color)
    eig_color      = kwargs.get('eig_color',  ell_color)
    ori_color      = kwargs.get('ori_color',  ell_color)
    # linewidths
    eig_linewidth  = kwargs.get('eig_linewidth',  ell_linewidth)
    rect_linewidth = kwargs.get('rect_linewidth', ell_linewidth)
    ori_linewidth  = kwargs.get('ori_linewidth',  ell_linewidth)
    # Offset keypoints
    assert len(kpts) > 0, 'cannot draw no keypoints1'
    kpts = ktool.offset_kpts(kpts, offset, scale_factor)
    assert len(kpts) > 0, 'cannot draw no keypoints2'
    # Extract keypoint components
    _xs, _ys = ktool.get_xys(kpts)
    # Build list of keypoint shape transforms from unit circles to ellipes
    invV_aff2Ds = get_invV_aff2Ds(kpts)
    # transformations but with rotations specified
    #RinvV_aff2Ds = get_RinvV_aff2Ds(invV_aff2Ds, _oris)
    try:
        if sifts is not None:
            # SIFT descriptors
            pass_props(kwargs, siftkw, 'bin_color', 'arm1_color', 'arm2_color',
                       'arm1_lw', 'arm2_lw', 'arm_alpha', 'arm_alpha')
            mpl_sift.draw_sifts(ax, sifts, invV_aff2Ds, **siftkw)
        if rect:
            # Bounding Rectangles
            rect_patches = rectangle_actors(invV_aff2Ds)
            _draw_patches(ax, rect_patches, rect_color, ell_alpha, rect_linewidth)
        if ell:
            # Keypoint shape
            ell_patches = ellipse_actors(invV_aff2Ds)
            _draw_patches(ax, ell_patches, ell_color, ell_alpha, ell_linewidth)
        if eig:
            # Shape eigenvectors
            eig_patches = eigenvector_actors(invV_aff2Ds)
            _draw_patches(ax, eig_patches, eig_color, ell_alpha, eig_linewidth)
        if ori:
            # Keypoint orientation
            ori_patches = orientation_actors(kpts)
            _draw_patches(ax, ori_patches, ori_color, ell_alpha, ori_linewidth, ori_color)
        if pts:
            # Keypoint locations
            _draw_pts(ax, _xs, _ys, pts_size, pts_color)
    except ValueError as ex:
        print('\n[mplkp] !!! ERROR %s: ' % str(ex))
        #print('_oris.shape = %r' % (_oris.shape,))
        #print('_xs.shape = %r' % (_xs.shape,))
        #print('_iv11s.shape = %r' % (_iv11s.shape,))
        raise

#----------------------------


def _draw_pts(ax, _xs, _ys, pts_size, pts_color):
    ax.scatter(_xs, _ys, c=pts_color, s=(2 * pts_size), marker='o', edgecolor='none')
    ax.autoscale(enable=False)


def get_invV_aff2Ds(kpts):
    """ Returns matplotlib keypoint transformations (circle -> ellipse) """
    invV_mats = ktool.get_invV_mats(kpts, with_trans=True, with_ori=True)
    invV_aff2Ds = [mpl.transforms.Affine2D(invV) for invV in invV_mats]
    return invV_aff2Ds


def ellipse_actors(invV_aff2Ds):
    # warp unit circles to keypoint shapes
    ell_actors = [mpl.patches.Circle((0, 0), 1, transform=invV)
                  for invV in invV_aff2Ds]
    return ell_actors


def rectangle_actors(RinvV_aff2Ds):
    Rect = mpl.patches.Rectangle
    Arrow = mpl.patches.FancyArrow
    rect_xywh  = (-1, -1), 2, 2
    arw_xydxdy = (-1, -1,  2, 0)
    arw_kw = dict(head_width=.1, length_includes_head=True)
    # warp unit rectangles to keypoint shapes
    rect_actors = [Rect(*rect_xywh, transform=RinvV) for RinvV in RinvV_aff2Ds]
    # an overhead arrow indicates the top of the rectangle
    arw_actors = [Arrow(*arw_xydxdy, transform=RinvV, **arw_kw) for RinvV in RinvV_aff2Ds]
    return rect_actors + arw_actors


def eigenvector_actors(invV_aff2Ds):
    # warps arrows into eigenvector directions
    kwargs = {
        'head_width': .01,
        'length_includes_head': False,
    }
    eig1 = [mpl.patches.FancyArrow(0, 0, 0, 1, transform=invV, **kwargs)
            for invV in invV_aff2Ds]
    eig2 = [mpl.patches.FancyArrow(0, 0, 1, 0, transform=invV, **kwargs)
            for invV in invV_aff2Ds]
    eig_actors = eig1 + eig2
    return eig_actors


def orientation_actors(kpts):
    """ creates orientation actors w.r.t. the gravity vector """
    try:
        # Get xy diretion of the keypoint orientations
        _xs, _ys = ktool.get_xys(kpts)
        _iv11s, _iv21s, _iv22s = ktool.get_invVs(kpts)
        _oris = ktool.get_oris(kpts)
        # mpl's 0 ori == (-tau / 4) w.r.t GRAVITY_THETA
        abs_oris = _oris + ktool.GRAVITY_THETA
        _sins = np.sin(abs_oris)
        _coss = np.cos(abs_oris)
        # The following is essentially
        # invV.dot(R)
        _dxs = _coss * _iv11s
        _dys = _coss * _iv21s +  _sins * _iv22s

        #head_width_list = np.log(_iv11s * _iv22s) / 5
        head_width_list = np.ones(len(_iv11s)) / 10
        kwargs = {
            'length_includes_head': True,
            'shape': 'full',
            'overhang': 0,
            'head_starts_at_zero': False,
        }
        ori_actors = [mpl.patches.FancyArrow(x, y, dx, dy, head_width=hw, **kwargs)
                      for (x, y, dx, dy, hw) in
                      zip(_xs, _ys, _dxs, _dys, head_width_list)]
    except ValueError as ex:
        print('\n[mplkp.2] !!! ERROR %s: ' % str(ex))
        print('_oris.shape = %r' % (_oris.shape,))
        print('x, y, dx, dy = %r' % ((x, y, dx, dy),))
        print('_dxs = %r' % (_dxs,))
        print('_dys = %r' % (_dys,))
        print('_xs = %r' % (_xs,))
        print('_ys = %r' % (_ys,))
        raise

    return ori_actors
