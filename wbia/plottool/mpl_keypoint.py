# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import zip
from wbia.plottool import mpl_sift
import numpy as np
import matplotlib as mpl
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
    # print('new collecitn')
    # print(alpha)
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_facecolor(fcolor)
    coll.set_alpha(alpha)
    coll.set_linewidth(lw)
    coll.set_edgecolor(color)
    # coll.set_transform(ax.transData)
    ax.add_collection(coll)


# ----------------------------
def draw_keypoints(
    ax,
    kpts_,
    scale_factor=1.0,
    offset=(0.0, 0.0),
    rotation=0.0,
    ell=True,
    pts=False,
    rect=False,
    eig=False,
    ori=False,
    sifts=None,
    siftkw={},
    H=None,
    **kwargs
):
    """
    draws keypoints extracted by pyhesaff onto a matplotlib axis

    FIXME: There is probably a matplotlib bug here. If you specify two different
    alphas in a collection, whatever the last alpha was gets applied to
    everything

    Args:
        ax (mpl.Axes):
        kpts (ndarray): keypoints [[x, y, a, c, d, theta], ...]
        scale_factor (float):
        offset (tuple):
        rotation (float):
        ell (bool):
        pts (bool):
        rect (bool):
        eig (bool):
        ori (bool):
        sifts (None):

    References:
        http://stackoverflow.com/questions/28401788/transforms-non-affine-patch

    CommandLine:
        python -m wbia.plottool.mpl_keypoint draw_keypoints --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.mpl_keypoint import *  # NOQA
        >>> from wbia.plottool.mpl_keypoint import _draw_patches, _draw_pts  # NOQA
        >>> import wbia.plottool as pt
        >>> import vtool as vt
        >>> imgBGR = vt.get_star_patch(jitter=True)
        >>> H = np.array([[1, 0, 0], [.5, 2, 0], [0, 0, 1]])
        >>> H = np.array([[.8, 0, 0], [0, .8, 0], [0, 0, 1]])
        >>> H = None
        >>> TAU = 2 * np.pi
        >>> kpts_ = vt.make_test_image_keypoints(imgBGR, scale=.5, skew=2, theta=TAU / 8.0)
        >>> scale_factor=1.0
        >>> #offset=(0.0, -4.0)
        >>> offset=(0.0, 0.0)
        >>> rotation=0.0
        >>> ell=True
        >>> pts=True
        >>> rect=True
        >>> eig=True
        >>> ori=True
        >>> # make random sifts
        >>> sifts = mpl_sift.testdata_sifts()
        >>> siftkw = {}
        >>> kwargs = dict(ori_color=[0, 1, 0], rect_color=[0, 0, 1],
        >>>               eig_color=[1, 1, 0], pts_size=.1)
        >>> w, h = imgBGR.shape[0:2][::-1]
        >>> imgBGR_ = imgBGR if H is None else vt.warpAffine(
        >>>     imgBGR, H, (int(w * .8), int(h * .8)))
        >>> fig, ax = pt.imshow(imgBGR_ * 255)
        >>> draw_keypoints(ax, kpts_, scale_factor, offset, rotation, ell, pts,
        ...                rect, eig, ori, sifts, siftkw, H=H, **kwargs)
        >>> pt.iup()
        >>> pt.show_if_requested()
    """
    import vtool.keypoint as ktool

    if kpts_.shape[1] == 2:
        # pad out structure if only xy given
        kpts = np.zeros((len(kpts_), 6))
        kpts[:, 0:2] = kpts_
        kpts[:, 2] = 1
        kpts[:, 4] = 1
        kpts_ = kpts

    if scale_factor is None:
        scale_factor = 1.0
    # print('[mpl_keypoint.draw_keypoints] kwargs = ' + ut.repr2(kwargs))
    # ellipse and point properties
    pts_size = kwargs.get('pts_size', 2)
    pts_alpha = kwargs.get('pts_alpha', 1.0)
    ell_alpha = kwargs.get('ell_alpha', 1.0)
    ell_linewidth = kwargs.get('ell_linewidth', 2)
    ell_color = kwargs.get('ell_color', None)
    if ell_color is None:
        ell_color = [1, 0, 0]
    # colors
    pts_color = kwargs.get('pts_color', ell_color)
    rect_color = kwargs.get('rect_color', ell_color)
    eig_color = kwargs.get('eig_color', ell_color)
    ori_color = kwargs.get('ori_color', ell_color)
    # linewidths
    eig_linewidth = kwargs.get('eig_linewidth', ell_linewidth)
    rect_linewidth = kwargs.get('rect_linewidth', ell_linewidth)
    ori_linewidth = kwargs.get('ori_linewidth', ell_linewidth)
    # Offset keypoints
    assert len(kpts_) > 0, 'cannot draw no keypoints1'
    kpts = ktool.offset_kpts(kpts_, offset, scale_factor)
    assert len(kpts) > 0, 'cannot draw no keypoints2'
    # Build list of keypoint shape transforms from unit circles to ellipes
    invVR_aff2Ds = get_invVR_aff2Ds(kpts, H=H)
    try:
        if sifts is not None:
            # SIFT descriptors
            pass_props(
                kwargs,
                siftkw,
                'bin_color',
                'arm1_color',
                'arm2_color',
                'arm1_lw',
                'arm2_lw',
                'stroke',
                'arm_alpha',
                'arm_alpha',
                'multicolored_arms',
            )
            mpl_sift.draw_sifts(ax, sifts, invVR_aff2Ds, **siftkw)
        if rect:
            # Bounding Rectangles
            rect_patches = rectangle_actors(invVR_aff2Ds)
            _draw_patches(ax, rect_patches, rect_color, ell_alpha, rect_linewidth)
        if ell:
            # Keypoint shape
            ell_patches = ellipse_actors(invVR_aff2Ds)
            _draw_patches(ax, ell_patches, ell_color, ell_alpha, ell_linewidth)
        if eig:
            # Shape eigenvectors
            eig_patches = eigenvector_actors(invVR_aff2Ds)
            _draw_patches(ax, eig_patches, eig_color, ell_alpha, eig_linewidth)
        if ori:
            # Keypoint orientation
            ori_patches = orientation_actors(kpts, H=H)
            _draw_patches(ax, ori_patches, ori_color, ell_alpha, ori_linewidth, ori_color)
        if pts:
            # Keypoint locations
            _xs, _ys = ktool.get_xys(kpts)
            if H is not None:
                # adjust for homogrpahy
                import vtool as vt

                _xs, _ys = vt.transform_points_with_homography(H, np.vstack((_xs, _ys)))

            pts_patches = _draw_pts(ax, _xs, _ys, pts_size, pts_color, pts_alpha)
            if pts_patches is not None:
                _draw_patches(ax, pts_patches, 'none', pts_alpha, pts_size, pts_color)
    except ValueError as ex:
        ut.printex(ex, '\n[mplkp] !!! ERROR')
        # print('_oris.shape = %r' % (_oris.shape,))
        # print('_xs.shape = %r' % (_xs.shape,))
        # print('_iv11s.shape = %r' % (_iv11s.shape,))
        raise


# ----------------------------


def _draw_pts(ax, _xs, _ys, pts_size, pts_color, pts_alpha=None):
    ptskw = dict(c=pts_color, s=(2 * pts_size), marker='o', edgecolor='none')
    OLD_WAY = False
    # if pts_alpha is not None:
    #    ptskw['alpha'] = pts_alpha
    if OLD_WAY:
        # print(ut.repr2(ptskw))
        ax.scatter(_xs, _ys, **ptskw)
        # FIXME: THIS MIGHT CAUSE ISSUES: UNEXPECTED CALL
        # ax.autoscale(enable=False)
    else:
        pts_patches = [
            mpl.patches.Circle((x, y), radius=(pts_size / 2), fill=True)
            for x, y in zip(_xs, _ys)
        ]
        # print(pts_color)
        return pts_patches


class HomographyTransform(mpl.transforms.Transform):
    """
    References:
        http://stackoverflow.com/questions/28401788/using-homogeneous-transforms-non-affine-with-matplotlib-patches?noredirect=1#comment45156353_28401788
        http://matplotlib.org/users/transforms_tutorial.html
    """

    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, H, axis=None, use_rmin=True):
        mpl.transforms.Transform.__init__(self)
        self._axis = axis
        self._use_rmin = use_rmin
        self.H = H

    def transform_non_affine(self, input_xy):
        """
        The input and output are Nx2 numpy arrays.
        """
        import vtool as vt

        _xys = input_xy.T
        xy_t = vt.transform_points_with_homography(self.H, _xys)
        output_xy = xy_t.T
        return output_xy

    # transform_non_affine.__doc__ = mpl.transforms.Transform.transform_non_affine.__doc__

    def transform_path_non_affine(self, path):
        vertices = path.vertices
        if len(vertices) == 2 and vertices[0, 0] == vertices[1, 0]:
            return mpl.path.Path(self.transform(vertices), path.codes)
        ipath = path.interpolated(path._interpolation_steps)
        return mpl.path.Path(self.transform(ipath.vertices), ipath.codes)

    # transform_path_non_affine.__doc__ = mpl.transforms.Transform.transform_path_non_affine.__doc__


def get_invVR_aff2Ds(kpts, H=None):
    """
    Returns matplotlib keypoint transformations (circle -> ellipse)

    Example:
        >>> # Test CV2 ellipse vs mine using MSER
        >>> import vtool as vt
        >>> import cv2
        >>> import wbia.plottool as pt
        >>> img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='zebra.png'))
        >>> imgBGR = vt.imread(img_fpath)
        >>> imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        >>> mser = cv2.MSER_create()
        >>> regions, bboxs = mser.detectRegions(imgGray)
        >>> region = regions[0]
        >>> bbox = bboxs[0]
        >>> vis = imgBGR.copy()
        >>> vis[region.T[1], region.T[0], :] = 0
        >>> hull = cv2.convexHull(region.reshape(-1, 1, 2))
        >>> cv2.polylines(vis, [hull], 1, (0, 255, 0))
        >>> ell = cv2.fitEllipse(region)
        >>> cv2.ellipse(vis, ell, (255))
        >>> ((cx, cy), (rx, ry), degrees) = ell
        >>> # Convert diameter to radians
        >>> rx /= 2
        >>> ry /= 2
        >>> # Make my version of ell
        >>> theta = np.radians(degrees)  # opencv lives in radians
        >>> S = vt.scale_mat3x3(rx, ry)
        >>> T = vt.translation_mat3x3(cx, cy)
        >>> R = vt.rotation_mat3x3(theta)
        >>> #R = np.eye(3)
        >>> invVR = T.dot(R).dot(S)
        >>> kpts = vt.flatten_invV_mats_to_kpts(np.array([invVR]))
        >>> pt.imshow(vis)
        >>> # MINE IS MUCH LARGER (by factor of 2)) WHY?
        >>> # we start out with a unit circle not a half circle
        >>> pt.draw_keypoints(pt.gca(), kpts, pts=True, ori=True, eig=True, rect=True)
    """
    import vtool.keypoint as ktool

    # invVR_mats = ktool.get_invV_mats(kpts, with_trans=True, with_ori=True)
    invVR_mats = ktool.get_invVR_mats3x3(kpts)
    if H is None:
        invVR_aff2Ds = [mpl.transforms.Affine2D(invVR) for invVR in invVR_mats]
    else:
        invVR_aff2Ds = [HomographyTransform(H.dot(invVR)) for invVR in invVR_mats]
    return invVR_aff2Ds


def ellipse_actors(invVR_aff2Ds):
    # warp unit circles to keypoint shapes
    ell_actors = [
        mpl.patches.Circle((0, 0), 1, transform=invVR) for invVR in invVR_aff2Ds
    ]
    return ell_actors


def rectangle_actors(invVR_aff2Ds):
    Rect = mpl.patches.Rectangle
    Arrow = mpl.patches.FancyArrow
    rect_xywh = (-1, -1), 2, 2
    arw_xydxdy = (-1, -1, 2, 0)
    arw_kw = dict(head_width=0.1, length_includes_head=True)
    # warp unit rectangles to keypoint shapes
    rect_actors = [Rect(*rect_xywh, transform=invVR) for invVR in invVR_aff2Ds]
    # an overhead arrow indicates the top of the rectangle
    arw_actors = [Arrow(*arw_xydxdy, transform=invVR, **arw_kw) for invVR in invVR_aff2Ds]
    return rect_actors + arw_actors


def eigenvector_actors(invVR_aff2Ds):
    # warps arrows into eigenvector directions
    kwargs = {
        'head_width': 0.01,
        'length_includes_head': False,
    }
    eig1 = [
        mpl.patches.FancyArrow(0, 0, 0, 1, transform=invV, **kwargs)
        for invV in invVR_aff2Ds
    ]
    eig2 = [
        mpl.patches.FancyArrow(0, 0, 1, 0, transform=invV, **kwargs)
        for invV in invVR_aff2Ds
    ]
    eig_actors = eig1 + eig2
    return eig_actors


def orientation_actors(kpts, H=None):
    """ creates orientation actors w.r.t. the gravity vector """
    import vtool.keypoint as ktool

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
        _dys = _coss * _iv21s + _sins * _iv22s
        # ut.embed()

        # if H is not None:
        #    # adjust for homogrpahy
        #    import vtool as vt
        #    _xs, _ys = vt.transform_points_with_homography(H, np.vstack((_xs, _ys)))
        #    _dxs, _dys = vt.transform_points_with_homography(H, np.vstack((_dxs, _dys)))

        # head_width_list = np.log(_iv11s * _iv22s) / 5
        head_width_list = np.ones(len(_iv11s)) / 10
        kwargs = {
            'length_includes_head': True,
            'shape': 'full',
            'overhang': 0,
            'head_starts_at_zero': False,
        }
        if H is not None:
            kwargs['transform'] = HomographyTransform(H)

        ori_actors = [
            mpl.patches.FancyArrow(x, y, dx, dy, head_width=hw, **kwargs)
            for (x, y, dx, dy, hw) in zip(_xs, _ys, _dxs, _dys, head_width_list)
        ]
    except ValueError as ex:
        print('\n[mplkp.2] !!! ERROR %s: ' % str(ex))
        print('_oris.shape = %r' % (_oris.shape,))
        # print('x, y, dx, dy = %r' % ((x, y, dx, dy),))
        print('_dxs = %r' % (_dxs,))
        print('_dys = %r' % (_dys,))
        print('_xs = %r' % (_xs,))
        print('_ys = %r' % (_ys,))
        raise

    return ori_actors


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.mpl_keypoint
        python -m wbia.plottool.mpl_keypoint --allexamples
        python -m wbia.plottool.mpl_keypoint --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
