# -*- coding: utf-8 -*-
# LICENCE
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
from six.moves import zip
import numpy as np
from vtool import histogram as htool
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import image as gtool
from vtool import trig
import utool as ut
try:
    import cv2
except ImportError as ex:
    print('ERROR: import cv2 is failing!')
    cv2 = ut.DynStruct()
    cv2.INTER_LANCZOS4 = None
    cv2.INTER_CUBIC = None
    cv2.BORDER_CONSTANT = None
    cv2.BORDER_REPLICATE = None
(print, rrr, profile) = ut.inject2(__name__)


TAU = np.pi * 2  # References: tauday.com


def patch_gradient(patch, ksize=1, gaussian_weighted=False):
    patch_ = np.array(patch, dtype=np.float64)
    gradx = cv2.Sobel(patch_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(patch_, cv2.CV_64F, 0, 1, ksize=ksize)
    if gaussian_weighted:
        gausspatch = gaussian_patch(shape=gradx.shape)
        gausspatch /= gausspatch.max()
        gradx *= gausspatch
        grady *= gausspatch
    return gradx, grady


def patch_mag(gradx, grady):
    return np.sqrt((gradx ** 2) + (grady ** 2))


def patch_ori(gradx, grady):
    """ returns patch orientation relative to the x-axis """
    gori = trig.atan2(grady, gradx)
    return gori


def get_test_patch(key='star', jitter=False):
    r"""
    Args:
        key (str):
        jitter (bool):

    Returns:
        ndarray: patch

    CommandLine:
        python -m vtool.patch --test-get_test_patch --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> key = 'star2'
        >>> jitter = False
        >>> # execute function
        >>> patch = get_test_patch(key, jitter)
        >>> pt.imshow(255 * patch)
        >>> pt.show_if_requested()
    """
    func = {
        'star2': get_star2_patch,
        'cross': get_cross_patch,
        'star': get_star_patch,
        'stripe': get_stripe_patch,
    }[key]
    patch = func(jitter)
    return patch


def make_test_image_keypoints(imgBGR, scale=1.0, skew=0, theta=0, shift=(0, 0)):
    h, w = imgBGR.shape[0:2]
    half_w, half_h = w / 2.0, h / 2.0
    x, y = (half_w - .5) + (w * shift[0]), (half_h - .5) + (h * shift[1])
    a = (half_w) * scale
    c = skew
    d = (half_h) * scale
    theta = theta
    kpts = np.array([[x, y, a, c, d, theta]], np.float32)
    return kpts


def get_no_symbol(variant='symbol', size=(100, 100)):
    r"""
    Returns:
        ndarray: errorimg

    CommandLine:
        python -m vtool.patch --test-get_no_symbol --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> errorimg = get_no_symbol()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(errorimg)
        >>> ut.show_if_requested()
    """
    thickness = 2
    shape = (size[1], size[0], 3)
    errorimg = np.zeros(shape)
    center = (size[0] // 2, size[1] // 2)
    radius = min(center) - thickness
    color_bgr = [0, 0, 255]
    tau = 2 * np.pi
    angle = 45 / 360 * tau
    pt1 = (center[0] - int(np.sin(angle) * radius), center[1] - int(np.cos(angle) * radius))
    pt2 = (center[0] + int(np.sin(angle) * radius), center[1] + int(np.cos(angle) * radius))
    if variant == 'symbol':
        cv2.circle(errorimg, center, radius, color_bgr, thickness)
        cv2.line(errorimg, pt1, pt2, color_bgr, thickness)
    else:
        import vtool as vt
        fontFace = cv2.FONT_HERSHEY_PLAIN
        org = (size[0] * .1, size[1] * .6)
        fontkw = dict(bottomLeftOrigin=False, fontScale=2.5, fontFace=fontFace)
        vt.draw_text(errorimg, 'NaN', org, thickness=2,
                     textcolor_rgb=color_bgr[::-1], **fontkw)
    return errorimg


def get_star_patch(jitter=False):
    """ test data patch """
    _, O = .1, .8
    patch = np.array([
        [_, _, _, O, _, _, _],
        [_, _, _, O, _, _, _],
        [_, _, O, O, O, _, _],
        [O, O, O, O, O, O, O],
        [_, O, O, O, O, O, _],
        [_, _, O, O, O, _, _],
        [_, O, O, O, O, O, _],
        [_, O, _, _, _, O, _],
        [O, _, _, _, _, _, O]])

    if jitter:
        patch += np.random.rand(*patch.shape) * .1
    return patch


def get_star2_patch(jitter=False):
    """ test data patch """
    _, i, O = .1, .8, .5
    patch = np.array([
        [_, _, _, _, _, _, _, O, O, _, _, _, _, _, _, _],
        [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
        [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
        [_, _, _, _, _, O, i, i, i, i, O, _, _, _, _, _],
        [O, O, O, O, O, O, i, i, i, i, O, O, O, O, O, O],
        [O, i, i, i, i, i, i, i, i, i, i, i, i, i, i, O],
        [_, O, i, i, i, i, O, i, i, O, i, i, i, i, O, _],
        [_, _, O, i, i, i, O, i, i, O, i, i, i, O, _, _],
        [_, _, _, O, i, i, O, i, i, O, i, i, O, _, _, _],
        [_, _, _, O, i, i, i, i, i, i, i, i, O, _, _, _],
        [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
        [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
        [_, O, i, i, i, i, i, O, O, i, i, i, i, i, O, _],
        [_, O, i, i, i, O, O, _, _, O, O, i, i, i, O, _],
        [O, i, i, O, O, _, _, _, _, _, _, O, O, i, i, O],
        [O, O, O, _, _, _, _, _, _, _, _, _, _, O, O, O]])

    if jitter:
        patch += np.random.rand(*patch.shape) * .1
    return patch


def get_cross_patch(jitter=False):
    """ test data patch """
    _, O = .1, .8
    patch = np.array([
        [_, _, O, O, O, _, _],
        [_, _, O, O, O, _, _],
        [_, _, O, O, O, _, _],
        [O, O, O, O, O, O, O],
        [O, O, O, O, O, O, O],
        [_, _, O, O, O, _, _],
        [_, _, O, O, O, _, _],
        [_, _, O, O, O, _, _],
        [_, _, O, O, O, _, _]])

    if jitter:
        patch += np.random.rand(*patch.shape) * .1
    return patch


def get_stripe_patch(jitter=False):
    """ test data patch """
    _, O = .1, .8
    patch = np.array([
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _],
        [O, O, O, _, _, _, _]])

    if jitter:
        patch += np.random.rand(*patch.shape) * .1
    return patch


def test_show_gaussian_patches2(shape=(19, 19)):
    r"""
    CommandLine:
        python -m vtool.patch --test-test_show_gaussian_patches2 --show
        python -m vtool.patch --test-test_show_gaussian_patches2 --show --shape=7,7
        python -m vtool.patch --test-test_show_gaussian_patches2 --show --shape=19,19
        python -m vtool.patch --test-test_show_gaussian_patches2 --show --shape=41,41
        python -m vtool.patch --test-test_show_gaussian_patches2 --show --shape=41,7

    References:
        http://matplotlib.org/examples/mplot3d/surface3d_demo.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> import plottool as pt
        >>> shape = ut.get_argval(('--shape',), type_=list, default=[19, 19])
        >>> test_show_gaussian_patches2(shape=shape)
        >>> pt.show_if_requested()
    """
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    import plottool as pt
    import numpy as np
    import matplotlib as mpl
    import vtool as vt
    shape = tuple(map(int, shape))
    print('shape = %r' % (shape,))
    #shape = (27, 27)
    #shape = (7, 7)
    #shape = (41, 41)
    #shape = (5, 5)
    #shape = (3, 3)
    sigma_percent_list = [.1, .3, .5, .6, .7, .8, .9, .95, 1.0]
    #np.linspace(.1, 3, 9)
    ybasis = np.arange(shape[0])
    xbasis = np.arange(shape[1])
    xgrid, ygrid = np.meshgrid(xbasis, ybasis)
    fnum = pt.next_fnum()
    for sigma_percent in pt.param_plot_iterator(sigma_percent_list, fnum=fnum, projection='3d'):
        radius1 = shape[0]
        radius2 = shape[1]
        sigma1 = radius1 * sigma_percent
        sigma2 = radius2 * sigma_percent
        sigma = [sigma1, sigma2]
        gausspatch = vt.gaussian_patch(shape, sigma=sigma)
        #print(gausspatch)
        #pt.imshow(gausspatch * 255)
        pt.plot_surface3d(xgrid, ygrid, gausspatch, rstride=1, cstride=1,
                          cmap=mpl.cm.coolwarm, title='sigma_percent=%.3f' % (sigma_percent,))
    pt.update()
    pt.set_figtitle('2d gaussian kernels')


def show_gaussian_patch(shape, sigma1, sigma2):
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    import matplotlib as mpl
    import plottool as pt
    import vtool as vt
    ybasis = np.arange(shape[0])
    xbasis = np.arange(shape[1])
    xgrid, ygrid = np.meshgrid(xbasis, ybasis)
    sigma = [sigma1, sigma2]
    gausspatch = vt.gaussian_patch(shape, sigma=sigma)
    #print(gausspatch)
    #pt.imshow(gausspatch * 255)
    title = 'ksize=%r, sigma=%r' % (shape, (sigma1, sigma2),)
    pt.plot_surface3d(xgrid, ygrid, gausspatch, rstride=1, cstride=1,
                      cmap=mpl.cm.coolwarm, title=title)


def inverted_sift_patch(sift, dim=32):
    """
    Idea for inverted sift visualization

    CommandLine:
        python -m vtool.patch test_sift_viz --show --name=star
        python -m vtool.patch test_sift_viz --show --name=star2
        python -m vtool.patch test_sift_viz --show --name=cross
        python -m vtool.patch test_sift_viz --show --name=stripe

    Example:
        >>> from vtool.patch import *  # NOQA
        >>> import vtool as vt
        >>> patch = vt.get_test_patch(ut.get_argval('--name', default='star'))
        >>> sift = vt.extract_feature_from_patch(patch)
        >>> siftimg = test_sift_viz(sift)
        >>> # Need to do some image blending
        >>> import plottool as pt
        >>> #pt.imshow(siftimg)
        >>> import plottool as pt
        >>> pt.figure(fnum=1, pnum=(1, 2, 1))
        >>> pt.mpl_sift.draw_sift_on_patch(siftimg, sift)
        >>> pt.figure(fnum=1, pnum=(1, 2, 2))
        >>> patch2 = patch
        >>> patch2 = vt.rectify_to_uint8(patch2)
        >>> patch2 = vt.rectify_to_square(patch2)
        >>> pt.mpl_sift.draw_sift_on_patch(patch2, sift)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    # dim = 21
    pad = dim // 2 + (dim % 2)
    # pad = 0
    blocks = []
    for siftmags in ut.ichunks(sift, 8):
        thetas = np.linspace(0, TAU, 8, endpoint=False)
        # style = 'step'
        style = 'linear'
        block_parts = [gradient_fill(dim, theta, flip=0, style=style) * mag
                       for theta, mag in zip(thetas, siftmags)]
        block = np.add.reduce(block_parts)  # / sum(siftmags)
        # block = block[pad:-pad, pad:-pad]
        # block = gaussian_weight_patch(block, sigma=9)
        blocks.append(block)

    rows = []
    for row_blocks in ut.ichunks(blocks, 4):
        row = vt.stack_image_list(row_blocks, vert=False, overlap=pad)
        rows.append(row)
    siftimg = vt.stack_image_list(rows, vert=True, overlap=pad)
    siftimg /= siftimg.max()
    return siftimg


def gradient_fill(shape, theta=0, flip=False, vert=False, style='linear'):
    """
    FIXME: angle does not work properly

    CommandLine:
        python -m vtool.patch gradient_fill --show

    Example:
        >>> from vtool.patch import *  # NOQA
        >>> import vtool as vt
        >>> shape = (9, 9)
        >>> #style = 'linear'
        >>> style = 'step'
        >>> theta = np.pi / 4
        >>> patch = vt.gradient_fill(shape, theta, style=style)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(vt.rectify_to_uint8(patch))
        >>> ut.show_if_requested()
    """
    if not isinstance(shape, tuple):
        shape = (shape, shape)
    import vtool as vt
    patch = np.zeros(shape)
    if vert:
        vals = np.linspace(0, 1, shape[0])
    else:
        vals = np.linspace(0, 1, shape[1])

    if style == 'linear':
        vals = vals
    elif style == 'step':
        vals = vals > .5

    if flip:
        vals = vals[::-1]

    if vert:
        patch.T[:] = vals
    else:
        patch[:] = vals

    if theta != 0:
        patch = vt.rotate_image(patch, theta, interpolation='linear',
                                border_mode='replicate')

    patch = np.clip(patch, 0, 1)

    return patch


def test_show_gaussian_patches(shape=(19, 19)):
    r"""
    CommandLine:
        python -m vtool.patch --test-test_show_gaussian_patches --show
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=7,7
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=17,17
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=41,41
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=29,29
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=41,7

    References:
        http://matplotlib.org/examples/mplot3d/surface3d_demo.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> import plottool as pt
        >>> shape = ut.get_argval(('--shape',), type_=list, default=[19, 19])
        >>> test_show_gaussian_patches(shape=shape)
        >>> pt.show_if_requested()
    """
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    import plottool as pt
    import numpy as np
    import matplotlib as mpl
    import vtool as vt
    shape = tuple(map(int, shape))
    print('shape = %r' % (shape,))
    #shape = (27, 27)
    #shape = (7, 7)
    #shape = (41, 41)
    #shape = (5, 5)
    #shape = (3, 3)
    sigma = 1.0
    sigma_list = [.1, .5, .825, .925, 1.0, 1.1, 1.2, 1.6, 2.0, 2.2, 3.0, 10.]
    #np.linspace(.1, 3, 9)
    ybasis = np.arange(shape[0])
    xbasis = np.arange(shape[1])
    xgrid, ygrid = np.meshgrid(xbasis, ybasis)
    fnum = pt.next_fnum()
    for sigma in pt.param_plot_iterator(sigma_list, fnum=fnum, projection='3d'):
        gausspatch = vt.gaussian_patch(shape, sigma=sigma)
        #print(gausspatch)
        #pt.imshow(gausspatch * 255)
        pt.plot_surface3d(xgrid, ygrid, gausspatch, rstride=1, cstride=1,
                          cmap=mpl.cm.coolwarm, title='sigma=%.3f' % (sigma,))
    pt.update()
    pt.set_figtitle('2d gaussian kernels')


def gaussian_patch(shape=(7, 7), sigma=1.0):
    """
    another version of the guassian_patch function. hopefully better

    References:
        http://docs.opencv.org/modules/imgproc/doc/filtering.html#getgaussiankernel

    Args:
        shape (tuple):  array dimensions
        sigma (float):

    CommandLine:
        python -m vtool.patch --test-gaussian_patch --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> #shape = (7, 7)
        >>> shape = (24, 24)
        >>> sigma = None  # 1.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> ut.assert_almost_eq(sum_, 1.0)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(vt.norm01(gausspatch) * 255)
        >>> ut.show_if_requested()
    """
    if sigma is None:
        sigma = 0.3 * ((min(shape) - 1) * 0.5 - 1) + 0.8
    if isinstance(sigma, (float)):
        sigma1 = sigma2 = sigma
    else:
        sigma1, sigma2 = sigma
    # see hesaff/src/helpers.cpp : computeCircularGaussMask
    # HACK MAYBE: I think sigma is actually a sigma squared term?
    #sigma1 = np.sqrt(sigma1)
    #sigma2 = np.sqrt(sigma2)
    gauss_kernel_d0 = (cv2.getGaussianKernel(shape[0], sigma1))
    gauss_kernel_d1 = (cv2.getGaussianKernel(shape[1], sigma2))
    gausspatch = gauss_kernel_d0.dot(gauss_kernel_d1.T)
    return gausspatch


#@lru_cache(maxsize=1000)
#def gaussian_patch(width=3, height=3, shape=(7, 7), sigma=None, norm_01=True):
#    """
#    slow function that makes 2d gaussian image patch
#    It is essential that this function is cached!
#    """
#    # Build a list of x and y coordinates
#    half_width  = (width  / 2.0)
#    half_height = (height / 2.0)
#    gauss_xs = np.linspace(-half_width,  half_width,  shape[0])
#    gauss_ys = np.linspace(-half_height, half_height, shape[1])
#    # Iterate over the cartesian coordinate product and get pdf values
#    gauss_xys  = itertools.product(gauss_xs, gauss_ys)
#    gaussvals  = [ltool.gauss2d_pdf(x, y, sigma=sigma, mu=None)
#                  for (x, y) in gauss_xys]
#    # Reshape pdf values into a 2D image
#    gausspatch = np.array(gaussvals, dtype=np.float32).reshape(shape).T
#    if norm_01:
#        # normalize if requested
#        gausspatch -= gausspatch.min()
#        gausspatch /= gausspatch.max()
#    return gausspatch


def get_unwarped_patches(img, kpts):
    r"""
    Returns cropped unwarped (keypoint is still elliptical) patch around a
    keypoint

    Args:
        img (ndarray): array representing an image
        kpts (ndarrays): keypoint ndarrays in [x, y, a, c, d, theta] format
    Returns:
        tuple : (patches, subkpts) - the unnormalized patches from the img
            corresonding to the keypoint

    """
    _xs, _ys = ktool.get_xys(kpts)
    xyexnts = ktool.get_kpts_wh(kpts)
    patches = []
    subkpts = []

    for (kp, x, y, (sfx, sfy)) in zip(kpts, _xs, _ys, xyexnts):
        radius_x = sfx * 1.5
        radius_y = sfy * 1.5
        (chip_h, chip_w) = img.shape[0:2]
        # Get integer grid coordinates to crop at
        ix1, ix2, xm = htool.subbin_bounds(x, radius_x, 0, chip_w)
        iy1, iy2, ym = htool.subbin_bounds(y, radius_y, 0, chip_h)
        # Crop the keypoint out of the image
        patch = img[iy1:iy2, ix1:ix2]
        subkp = kp.copy()  # subkeypoint in patch coordinates
        subkp[0:2] = (xm, ym)
        patches.append(patch)
        subkpts.append(subkp)
    return patches, subkpts


def get_warped_patches(img, kpts, flags=cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REPLICATE, patch_size=41,
                       use_cpp=False):
    r"""
    Returns warped (into a unit circle) patch around a keypoint

    FIXME:
        there is a slight translation difference in the way Python extracts
        patches and the way C++ extracts patches. C++ should be correct.
        TODO: have C++ able to extract color.

    Args:
        img (ndarray[uint8_t, ndim=2]): array representing an image
        kpts (ndarray[float32_t, ndim=2]): list of keypoint ndarrays in
            [[x, y, a, c, d, theta]] format
        flags (long): cv2 interpolation flags
        borderMode (long): cv2 border flags
        patch_size (int): resolution of resulting image patch

    Returns:
        (list, list) : (warped_patches, warped_subkpts) the normalized 41x41
            patches from the img corresonding to the keypoint

    CommandLine:
        python -m vtool.patch --test-get_warped_patches --show --use_cpp
        python -m vtool.patch --test-get_warped_patches --show --use_python

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath('lena.png')
        >>> img = vt.imread(img_fpath)
        >>> use_cpp = ut.get_argflag('--use_cpp')
        >>> kpts, desc = vt.extract_features(img_fpath)
        >>> kpts = kpts[0:1]
        >>> flags = cv2.INTER_LANCZOS4
        >>> borderMode = cv2.BORDER_REPLICATE
        >>> # execute function
        >>> (warped_patches, warped_subkpts) = get_warped_patches(img, kpts, flags, borderMode, use_cpp=use_cpp)
        >>> # verify results
        >>> print(np.array(warped_patches).shape)
        >>> print(ut.repr2(np.array(warped_subkpts), precision=2))
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(warped_patches[0])
        >>> #pt.draw_kpts2(warped_subkpts, pts=True, rect=True)
        >>> pt.set_title('use_cpp = %r' % (use_cpp,))
        >>> pt.show_if_requested()
    """
    # TODO: CLEAN ME
    warped_patches = []
    warped_subkpts = []
    xs, ys = ktool.get_xys(kpts)
    # rotate relative to the gravity vector
    oris = ktool.get_oris(kpts)
    invV_mats = ktool.get_invV_mats(kpts, with_trans=False, ashomog=True)
    V_mats = ktool.invert_invV_mats(invV_mats)
    kpts_iter = zip(xs, ys, V_mats, oris)
    #patch_size = 41  # sf
    #cv2_warp_kwargs = {
    #    #'flags': cv2.INTER_LINEAR,
    #    #'flags': cv2.INTER_NEAREST,
    #    #'flags': cv2.INTER_LANCZOS4,
    #    'borderMode': cv2.BORDER_REPLICATE,
    #}
    #flags = cv2.INTER_CUBIC,
    #borderMode = cv2.BORDER_REPLICATE,
    if use_cpp:
        import pyhesaff
        warped_patches = pyhesaff.extract_patches(img, kpts)
        # FIXME:
        ss = np.sqrt(patch_size) * 3.0
        half_patch_size = patch_size / 2.0
        warped_subkpts.append(np.array((half_patch_size, half_patch_size, ss, 0., ss, 0)))
    else:
        for x, y, V, ori in kpts_iter:
            warped_patch, wkp = intern_warp_single_patch(img, x, y, ori, V,
                                                         patch_size,
                                                         flags=flags,
                                                         borderMode=borderMode)
            ## Build warped keypoints
            #wkp = np.array((patch_size / 2, patch_size / 2, ss, 0., ss, 0))
            warped_patches.append(warped_patch)
            warped_subkpts.append(wkp)
    return warped_patches, warped_subkpts


def intern_warp_single_patch(img, x, y, ori, V,
                             patch_size,
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE):
    r"""
    Sympy:
        # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        from vtool.patch import *  # NOQA
        import sympy
        from sympy.abc import theta
        ori = theta
        x, y, a, c, d, patch_size = sympy.symbols('x y a c d S')
        half_patch_size = patch_size / 2

        def sympy_rotation_mat3x3(radians):
            # TODO: handle array impouts
            sin_ = sympy.sin(radians)
            cos_ = sympy.cos(radians)
            R = np.array(((cos_, -sin_,  0),
                          (sin_,  cos_,  0),
                          (   0,     0,  1),))
            return sympy.Matrix(R)

        kpts = np.array([[x, y, a, c, d, ori]])
        kp = ktool.get_invV_mats(kpts, with_trans=True)[0]
        invV = sympy.Matrix(kp)
        V = invV.inv()
        ss = sympy.sqrt(patch_size) * 3.0
        T = sympy.Matrix(ltool.translation_mat3x3(-x, -y, None))  # Center the patch
        R = sympy_rotation_mat3x3(-ori)  # Rotate the centered unit circle patch
        S = sympy.Matrix(ltool.scale_mat3x3(ss, dtype=None))  # scale from unit circle to the patch size
        X = sympy.Matrix(ltool.translation_mat3x3(half_patch_size, half_patch_size, None))  # Translate back to patch-image coordinates

        sympy.MatMul(X, S, hold=True)

        def add_matmul_hold_prop(mat):
            #import functools
            def matmul_hold(other, hold=True):
                new = sympy.MatMul(mat, other, hold=hold)
                add_matmul_hold_prop(new)
                return new
            #matmul_hold = functools.partial(sympy.MatMul, mat, hold=True)
            setattr(mat, 'matmul_hold', matmul_hold)
        add_matmul_hold_prop(X)
        add_matmul_hold_prop(S)
        add_matmul_hold_prop(R)
        add_matmul_hold_prop(V)
        add_matmul_hold_prop(T)

        M = X.matmul_hold(S).matmul_hold(R).matmul_hold(V).matmul_hold(T)
        #M = X.multiply(S).multiply(R).multiply(V).multiply(T)


        V_full = R.multiply(V).multiply(T)
        sympy.latex(V_full)
        print(sympy.latex(R.multiply(V).multiply(T)))
        print(sympy.latex(X))
        print(sympy.latex(S))
        print(sympy.latex(R))
        print(sympy.latex(invV) + '^{-1}')
        print(sympy.latex(T))
    """
    cv2_warp_kwargs = {
        #'flags': cv2.INTER_LINEAR,
        #'flags': cv2.INTER_NEAREST,
        #'flags': cv2.INTER_CUBIC,
        #'flags': cv2.INTER_LANCZOS4,
        'flags': flags,
        #'borderMode': cv2.BORDER_REPLICATE,
        'borderMode': borderMode
    }
    #ut.embed()
    # FIXME: this works only because of add-hoc reasons.
    # need to more closely follow code in affine.cpp

    half_patch_size = patch_size / 2.0

    OLDWAY = True
    #OLDWAY = True
    if OLDWAY:
        ss = np.sqrt(patch_size) * 3.0
    else:
        # This seems to not work correctly
        mrSize = 3.0 * np.sqrt(3.0)
        sc = np.sqrt(1 / ktool.get_invVR_mats_sqrd_scale(V[None, :])[0])
        s = sc / mrSize
        mrScale = np.ceil(s * mrSize)
        patchImageSize = 2 * mrScale + 1
        imageToPatchScale = patchImageSize / patch_size
        ss = sc

    (h, w) = img.shape[0:2]
    # Translate to origin(0,0) = (x,y)
    T = ltool.translation_mat3x3(-x, -y)  # Center the patch
    # V - reshape and scale the centered patch to the unit circle
    R = ltool.rotation_mat3x3(-ori)  # Rotate the centered unit circle patch
    S = ltool.scale_mat3x3(ss)  # scale from unit circle to the patch size
    X = ltool.translation_mat3x3(half_patch_size, half_patch_size)  # Translate back to patch-image coordinates
    M = X.dot(S).dot(R).dot(V).dot(T)
    # Prepare to warp
    dsize = np.ceil([patch_size, patch_size]).astype(np.int)
    # Warp
    #warped_patch = gtool.warpAffine(img, M, dsize)
    warped_patch = cv2.warpAffine(img, M[0:2], tuple(dsize), **cv2_warp_kwargs)
    # Build warped keypoints
    wkp = np.array((half_patch_size, half_patch_size, ss, 0., ss, 0))

    if not OLDWAY:
        # FIXME: this is still not what is done in affine.cpp
        if imageToPatchScale > 0.4:
            #ksize = (patchImageSize / 2, patchImageSize / 2)
            #sigmaX, sigmaY = (patchImageSize / 2, patchImageSize / 2)
            #ut.embed()
            sigma = imageToPatchScale * 1.5
            GaussianBlurInplace(warped_patch, sigma)
    else:
        sigma = 1.5
        GaussianBlurInplace(warped_patch, sigma)
    return warped_patch, wkp


def generate_to_patch_transforms(kpts, patch_size=41):
    import vtool as vt
    xs, ys = vt.get_xys(kpts)
    # rotate relative to the gravity vector
    oris = vt.get_oris(kpts)
    invV_mats = vt.get_invV_mats(kpts, with_trans=False, ashomog=True)
    V_mats = vt.invert_invV_mats(invV_mats)
    kpts_iter = zip(xs, ys, V_mats, oris)
    # HACK, this is using the old method
    half_patch_size = patch_size / 2.0
    ss = np.sqrt(patch_size) * 3.0

    for x, y, V, ori in kpts_iter:
        T = vt.translation_mat3x3(-x, -y)  # Center the patch
        # V - reshape and scale the centered patch to the unit circle
        R = vt.rotation_mat3x3(-ori)  # Rotate the centered unit circle patch
        S = vt.scale_mat3x3(ss)  # scale from unit circle to the patch size
        X = vt.translation_mat3x3(half_patch_size, half_patch_size)  # Translate back to patch-image coordinates
        M = X.dot(S).dot(R).dot(V).dot(T)
        yield M


def patch_gaussian_weighted_average_intensities(probchip, kpts_):
    """
    """
    import vtool as vt
    patch_size = 41
    M_iter = vt.generate_to_patch_transforms(kpts_, patch_size)
    dsize = np.ceil([patch_size, patch_size]).astype(np.int)
    # Preallocate patch
    patch = np.empty(dsize[::-1], dtype=np.uint8)
    weighted_patch = np.empty(dsize[::-1], dtype=np.float64)
    weight_list = []
    sigma = 0.3 * ((min(patch.shape[0:1]) - 1) * 0.5 - 1) + 0.8
    gauss_kernel_d0 = (cv2.getGaussianKernel(patch.shape[0], sigma))
    gauss_kernel_d1 = (cv2.getGaussianKernel(patch.shape[1], sigma)).T
    warpkw = dict(flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    for M in M_iter:
        patch = cv2.warpAffine(probchip, M[0:2], tuple(dsize), dst=patch, **warpkw)
        vt.GaussianBlurInplace(patch, 1.5)
        weighted_patch = np.divide(patch, 255., out=weighted_patch)
        weighted_patch = np.multiply(weighted_patch, gauss_kernel_d0, out=weighted_patch)
        weighted_patch = np.multiply(weighted_patch, gauss_kernel_d1, out=weighted_patch)
        weight = weighted_patch.sum()
        weight_list.append(weight)
    return weight_list


def gaussian_average_patch(patch, sigma=None, copy=True):
    """

    Args:
        patch (ndarray):
        sigma (float):

    CommandLine:
        python -m vtool.patch --test-gaussian_average_patch

    References:
        http://docs.opencv.org/modules/imgproc/doc/filtering.html#getgaussiankernel

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> patch = get_star_patch()
        >>> #sigma = 1.6
        >>> sigma = None
        >>> result = gaussian_average_patch(patch, sigma)
        >>> print(result)
        0.414210641527

    Ignore:
        import utool as ut
        import plottool as pt
        import vtool as vt
        import cv2
        gauss_kernel_d0 = (cv2.getGaussianKernel(patch.shape[0], sigma))
        gauss_kernel_d1 = (cv2.getGaussianKernel(patch.shape[1], sigma))
        weighted_patch = patch.copy()
        weighted_patch = np.multiply(weighted_patch,   gauss_kernel_d0)
        weighted_patch = np.multiply(weighted_patch.T, gauss_kernel_d1).T
        gaussian_kern2 = gauss_kernel_d0.dot(gauss_kernel_d1.T)
        fig = pt.figure(fnum=1, pnum=(1, 3, 1), doclf=True, docla=True)
        pt.imshow(patch * 255)
        fig = pt.figure(fnum=1, pnum=(1, 3, 2))
        pt.imshow(ut.norm_zero_one(gaussian_kern2) * 255.0)
        fig = pt.figure(fnum=1, pnum=(1, 3, 3))
        pt.imshow(ut.norm_zero_one(weighted_patch) * 255.0)
        pt.update()
    """
    if sigma is None:
        sigma = 0.3 * ((min(patch.shape[0:1]) - 1) * 0.5 - 1) + 0.8
    gauss_kernel_d0 = (cv2.getGaussianKernel(patch.shape[0], sigma))
    gauss_kernel_d1 = (cv2.getGaussianKernel(patch.shape[1], sigma))
    # assert gauss_kernel_d0.dot(gauss_kernel_d1.T).sum() == 1
    if copy:
        weighted_patch = patch.copy()
    else:
        weighted_patch = patch
    weighted_patch = np.multiply(weighted_patch,   gauss_kernel_d0)
    weighted_patch = np.multiply(weighted_patch.T, gauss_kernel_d1).T
    # TODO: use new guass patch weighting function here
    average = weighted_patch.sum()
    return average


def gaussian_weight_patch(patch, sigma=None):
    """
    Applies two one dimensional gaussian operations to a patch which
    effectively weights it by a 2-dimensional gaussian. This is efficient
    because the actually 2-d gaussian never needs to be allocated.

    test_show_gaussian_patches
    """
    #patch = np.ones(patch.shape)
    if sigma is None:
        sigma0 = (patch.shape[0] / 2) * .95
        sigma1 = (patch.shape[1] / 2) * .95
    else:
        sigma0 = sigma1 = sigma
    #sigma0 = (patch.shape[0] / 2) * .5
    #sigma1 = (patch.shape[1] / 2) * .5
    gauss_kernel_d0 = (cv2.getGaussianKernel(patch.shape[0], sigma0))
    gauss_kernel_d1 = (cv2.getGaussianKernel(patch.shape[1], sigma1))
    gauss_kernel_d0 /= gauss_kernel_d0.max()
    gauss_kernel_d1 /= gauss_kernel_d1.max()
    weighted_patch = (patch * gauss_kernel_d0.T) * gauss_kernel_d1
    return weighted_patch


def get_warped_patch(imgBGR, kp, gray=False, flags=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE, patch_size=41):
    r"""
    Returns warped (into a unit circle) patch around a keypoint

    Args:
        img (ndarray): array representing an image
        kpt (ndarray): keypoint ndarray in [x, y, a, c, d, theta] format

    Returns:
        (ndarray, ndarray) : (wpatch, wkp) the normalized 41x41 patches from
            the img corresonding to the keypoint
    """
    kpts = np.array([kp])
    wpatches, wkpts = get_warped_patches(imgBGR, kpts, flags=flags,
                                         borderMode=borderMode,
                                         patch_size=patch_size)
    wpatch = wpatches[0]
    wkp = wkpts[0]
    if gray and len(wpatch.shape) > 2:
        wpatch = gtool.cvt_BGR2L(wpatch)
    return wpatch, wkp


def GaussianBlurInplace(img, sigma, size=None):
    """
    simulates code from helpers.cpp in hesaff

    Args:
        img (ndarray):
        sigma (flaot):

    CommandLine:
        python -m vtool.patch --test-GaussianBlurInplace:0 --show
        python -m vtool.patch --test-GaussianBlurInplace:1 --show


    References;
        http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter4.pdf
        http://en.wikipedia.org/wiki/Scale_space_implementation
        http://www.cse.psu.edu/~rtc12/CSE486/lecture10_6pp.pdf

    Notes:
        The product of the convolution of two Gaussian functions with spread
        sigma is a Gaussian function with spread sqrt(2)*sigma scaled by the
        area of the Gaussian filter

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> import plottool as pt
        >>> img = get_test_patch('star2')
        >>> img_orig = img.copy()
        >>> sigma = .8
        >>> GaussianBlurInplace(img, sigma)
        >>> fig = pt.figure(fnum=1, pnum=(1, 3, 1))
        >>> size = int((2.0 * 3.0 * sigma + 1.0))
        >>> if not size & 1:  # check if even
        >>>     size += 1
        >>> ksize = (size, size)
        >>> fig.add_subplot(1, 3, 1, projection='3d')
        >>> show_gaussian_patch(ksize, sigma, sigma)
        >>> pt.imshow(img_orig * 255, fnum=1, pnum=(1, 3, 2))
        >>> pt.imshow(img * 255, fnum=1, pnum=(1, 3, 3))
        >>> pt.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> # demonstrate cascading smoothing property
        >>> # THIS ISNT WORKING WHY???
        >>> from vtool.patch import *  # NOQA
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> import plottool as pt
        >>> img = get_test_patch('star2')
        >>> img1 = img.copy()
        >>> img2 = img.copy()
        >>> img3 = img.copy()
        >>> img4 = img.copy()
        >>> img_orig = img.copy()
        >>> sigma1 = .6
        >>> sigma2 = .9
        >>> sigma3 = sigma1 + sigma2
        >>> size = 7
        >>> # components
        >>> GaussianBlurInplace(img1, sigma1, size)
        >>> GaussianBlurInplace(img2, sigma2, size)
        >>> # all in one shot
        >>> GaussianBlurInplace(img3, sigma3, size)
        >>> # addative
        >>> GaussianBlurInplace(img4, sigma1, size)
        >>> GaussianBlurInplace(img4, sigma2, size)
        >>> print((img4 - img3).sum())
        >>> ut.quit_if_noshow()
        >>> fig = pt.figure(fnum=1, pnum=(2, 4, 1))
        >>> ksize = (size, size)
        >>> #fig.add_subplot(1, 3, 1, projection='3d')
        >>> fig.add_subplot(2, 4, 1, projection='3d')
        >>> show_gaussian_patch(ksize, sigma1, sigma1)
        >>> fig.add_subplot(2, 4, 2, projection='3d')
        >>> show_gaussian_patch(ksize, sigma2, sigma2)
        >>> fig.add_subplot(2, 4, 3, projection='3d')
        >>> show_gaussian_patch(ksize, sigma3, sigma3)
        >>> pt.imshow(img_orig * 255, fnum=1, pnum=(2, 4, 4))
        >>> pt.imshow(img1 * 255, fnum=1, pnum=(2, 4, 5), title='%r' % (sigma1))
        >>> pt.imshow(img2 * 255, fnum=1, pnum=(2, 4, 6), title='%r' % (sigma2))
        >>> pt.imshow(img3 * 255, fnum=1, pnum=(2, 4, 7), title='%r' % (sigma3))
        >>> pt.imshow(img4 * 255, fnum=1, pnum=(2, 4, 8), title='%r + %r' % (sigma1, sigma2))
        >>> pt.show_if_requested()
    """
    if size is None:
        size = int((2.0 * 3.0 * sigma + 1.0))
        if not size & 1:  # check if even
            size += 1
    else:
        assert size & 1, 'size must be odd'
    ksize = (size, size)
    cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma, dst=img, borderType=cv2.BORDER_REPLICATE)
    return img


def get_unwarped_patch(imgBGR, kp, gray=False):
    """Returns unwarped warped patch around a keypoint

    Args:
        img (ndarray): array representing an image
        kpt (ndarray): keypoint ndarray in [x, y, a, c, d, theta] format
    Returns:
        tuple : (wpatch, wkp) the normalized 41x41 patches from the img corresonding to the keypoint
    """
    kpts = np.array([kp])
    upatches, ukpts = get_unwarped_patches(imgBGR, kpts)
    upatch = upatches[0]
    ukp = ukpts[0]
    if gray:
        upatch = gtool.cvt_BGR2L(upatch)
    return upatch, ukp


def find_kpts_direction(imgBGR, kpts, DEBUG_ROTINVAR=False):
    r"""
    Args:
        imgBGR (ndarray[uint8_t, ndim=2]):  image data in opencv format (blue, green, red)
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=2]: kpts -  keypoints

    CommandLine:
        python -m vtool.patch --test-find_kpts_direction
    """

    ori_list = []
    #gravity_ori = ktool.GRAVITY_THETA
    for kp in kpts:
        new_oris = find_dominant_kp_orientations(imgBGR, kp, DEBUG_ROTINVAR=DEBUG_ROTINVAR)
        # FIXME USE MULTIPLE ORIENTATIONS
        ori = new_oris[0]
        ori_list.append(ori)
    _oris = np.array(ori_list, dtype=kpts.dtype)
    #_oris -= gravity_ori  % TAU  # normalize w.r.t. gravity
    # discard old orientation if they exist
    kpts2 = np.vstack((kpts[:, 0:5].T, _oris)).T
    return kpts2


def draw_kp_ori_steps():
    """
    Shows steps in orientation estimation

    CommandLine:
        python -m vtool.patch --test-draw_kp_ori_steps --show --fname=zebra.png --fx=121
        python -m vtool.patch --test-draw_kp_ori_steps --show --interact
        python -m vtool.patch --test-draw_kp_ori_steps --save ~/latex/crall-candidacy-2015/figures/test_fint_kp_direction.jpg --dpath figures '--caption=visualization of the steps in the computation of the dominant gradient orientations.' --figsize=14,9 --dpi=160 --height=2.65  --left=.04 --right=.96 --top=.95 --bottom=.05 --wspace=.1 --hspace=.1

        python -m vtool.patch --test-draw_kp_ori_steps --dpath ~/latex/crall-candidacy-2015/ --save figures/draw_kp_ori_steps.jpg  --figsize=14,9 --dpi=180 --height=2.65 --left=.04 --right=.96 --top=.95 --bottom=.05 --wspace=.1 --hspace=.1 --diskshow

        python -m vtool.patch --test-draw_kp_ori_steps --dpath ~/latex/crall-candidacy-2015/ --save figures/draw_kp_ori_steps.jpg  --figsize=14,9 --dpi=180  --djust=.04,.05,.1 --diskshow --fname=zebra.png --fx=121

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.patch import *  # NOQA
        >>> draw_kp_ori_steps()
        >>> pt.show_if_requested()
    """
    #from vtool.patch import *  # NOQA
    #import vtool as vt
    # build test data
    import utool as ut
    import plottool as pt
    from six.moves import input
    import vtool as vt

    if True:
        from ibeis.scripts.thesis import TMP_RC
        import matplotlib as mpl
        mpl.rcParams.update(TMP_RC)
    #import vtool as vt
    np.random.seed(0)
    USE_COMMANLINE = True
    if USE_COMMANLINE:
        kpts, vecs, imgBGR = pt.viz_keypoints.testdata_kpts()
        fx = ut.get_argval('--fx', type_=int, default=0)
        kp = kpts[fx]
    else:
        fx = 0
        USE_EXTERN_STAR = False
        if USE_EXTERN_STAR:
            img_fpath = ut.grab_test_imgpath('star.png')
            imgBGR = vt.imread(img_fpath)
            kpts, vecs = vt.extract_features(img_fpath)
            kp = np.array([  3.14742985e+01,   2.95660381e+01,   1.96057682e+01, -5.11199608e-03,   2.05653343e+01,   0.00000000e+00],
                          dtype=np.float32)
        else:
            #imgBGR = get_test_patch('stripe', jitter=True)
            #imgBGR = get_test_patch('star', jitter=True)
            imgBGR = get_test_patch('star2', jitter=True)
            #imgBGR = get_test_patch('cross', jitter=False)
            #imgBGR = cv2.resize(imgBGR, (41, 41), interpolation=cv2.INTER_LANCZOS4)
            imgBGR = cv2.resize(imgBGR, (41, 41), interpolation=cv2.INTER_CUBIC)
            theta = 0  # 3.4  # TAU / 16
            #kpts = make_test_image_keypoints(imgBGR, scale=.9, theta=theta)
            kpts = make_test_image_keypoints(imgBGR, scale=.3, theta=theta, shift=(.3, .1))
            kp = kpts[0]
    bins = 36
    maxima_thresh = .8
    converge_lists = []

    def exec_internals_find_patch_dominant_orientations(patch, bins, maxima_thresh, old_ori):
        # TODO: can use ut.exec_func_src instead
        # <HACKISH>
        # exec source code from find_patch_dominant_orientations to steal its
        # local variables
        # ARGS: patch, bins=36, maxima_thresh=.8, DEBUG_ROTINVAR

        DEBUG_ROTINVAR = False
        globals_ = globals()
        locals_ = locals()
        keys = 'patch, gradx, grady, gmag, gori, hist, centers, gori_weights'.split(', ')
        internal_tup = ut.exec_func_src(find_patch_dominant_orientations, globals_, locals_, key_list=keys, update=True)
        submax_ori_offsets = globals_['submax_ori_offsets']
        new_oris = (old_ori + (submax_ori_offsets - ktool.GRAVITY_THETA)) % TAU
        # sourcecode = ut.get_func_sourcecode(find_patch_dominant_orientations, stripdef=True, stripret=True)
        # six.exec_(sourcecode, globals_, locals_)
        # submax_ori_offsets = locals_['submax_ori_offsets']
        # new_oris = (old_ori + (submax_ori_offsets - ktool.GRAVITY_THETA)) % TAU
        # keys = 'patch, gradx, grady, gmag, gori, hist, centers, gori_weights'.split(', ')
        # internal_tup = ut.dict_take(locals_, keys)
        return new_oris, internal_tup
        # </HACKISH>

    INTERACTIVE_ITERATION = ut.get_argflag('--interact')

    while True:
        patch, wkp = get_warped_patch(imgBGR, kp, gray=True,
                                      #flags=cv2.INTER_LANCZOS4,
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT)
        old_ori = kp[-1]
        # Execute test function
        new_oris, internal_tup = exec_internals_find_patch_dominant_orientations(patch, bins, maxima_thresh, old_ori)
        patch, gradx, grady, gmag, gori, hist, centers, gori_weights = internal_tup
        if INTERACTIVE_ITERATION:
            # Change rotation
            print('new_oris = %r' % (new_oris,))
            print('bins = %r' % (bins,))
            for count, ori in enumerate(new_oris):
                if count >= len(converge_lists):
                    converge_lists.append([])
                converge_lists[count].append(ori)

            # Show any new keypoints that were created
            kpts = [kp.copy() for ori in new_oris]
            for count, ori in enumerate(new_oris):
                kpts[count][-1] = ori
            kpts = np.array(kpts)

            kp[-1] = new_oris[0]

            show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag, gori, hist, centers, gori_weights, fx=fx)
            pt.figure(fnum=2, doclf=True)
            colors = pt.distinct_colors(len(converge_lists))
            print(len(converge_lists))
            for color, converge_list in zip(colors, converge_lists):
                pt.plot(converge_list, 'o-', color=color)
            pt.gca().set_ylim(0, TAU)
            pt.dark_background()
            pt.present()
            #pt.update()
            input('next')
        else:
            show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag, gori, hist, centers, gori_weights, fx=fx)
            # pt.present()
            print('no interaction')
            break


def show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag,
                                      gori, hist, centers, gori_weights,
                                      fx=None):
    import plottool as pt
    import vtool as vt
    # DRAW TEST INFO
    fnum = 1
    pt.figure(fnum=1, doclf=True, docla=True)
    gorimag = pt.color_orimag(gori, gmag=gmag, gmag_is_01=False)
    nRows, nCols = pt.get_square_row_cols(8)
    nRows += 1
    pnum_ = pt.make_pnum_nextgen(nRows, nCols)
    # hack
    imgBGR_ = imgBGR if imgBGR.max() > 1 else imgBGR * 255
    patch_ = patch if patch.max() > 1 else patch * 255
    pt.imshow(imgBGR_, update=True, fnum=fnum, pnum=pnum_(), title='input image')
    colors = pt.distinct_colors(len(kpts))
    if fx is None:
        pt.draw_kpts2(kpts, rect=True, ori=True, ell_color=colors)
    else:
        pt.draw_kpts2(kpts[fx:fx + 1], rect=True, ori=True, colors=[pt.ORANGE])
    pt.imshow(patch_, fnum=fnum, pnum=pnum_(), title='sampled patch')
    def normalize_grad_img(grad_):
        #return np.abs(grad_) * 255
        return vt.norm01(np.abs(grad_)) * 255
        #return vt.norm01(grad_) * 255
    pt.imshow(normalize_grad_img(gradx ** 2), fnum=fnum, pnum=pnum_(), title='gradx ** 2')
    pt.imshow(normalize_grad_img(grady ** 2), fnum=fnum, pnum=pnum_(), title='grady ** 2')
    pt.imshow(normalize_grad_img(gmag), fnum=fnum, pnum=pnum_(), title='mag')
    pt.imshow(normalize_grad_img(gori_weights), fnum=fnum, pnum=pnum_(), title='weighted mag')
    #pt.imshow(ut.norm_zero_one(gori) * 255, fnum=fnum, pnum=pnum_(), title='ori')
    stride = ut.get_argval('--stride', default=1)
    pt.draw_vector_field(gradx, grady, stride=stride, pnum=pnum_(), fnum=fnum,
                         title='gori (vec)')
    pt.imshow(gorimag, fnum=fnum, pnum=pnum_(), title='ori-color')
    if not ut.get_argflag('--noweighted-gori'):
        pt.color_orimag_colorbar(gori * gori_weights)
    else:
        pt.color_orimag_colorbar(gori)
    pt.figure(fnum=fnum, pnum=(nRows, 1, nRows))
    bin_colors = pt.get_orientation_color(centers)
    pt.draw_hist_subbin_maxima(hist, centers, bin_colors=bin_colors,
                               maxima_thresh=.8)
    ax = pt.gca()
    ax.set_xlabel('radians')
    ax.set_ylabel('weight')


def test_ondisk_find_patch_fpath_dominant_orientations(patch_fpath, bins=36,
                                                       maxima_thresh=.8,
                                                       DEBUG_ROTINVAR=True):
    r"""
    Args:
        patch_fpath (?):
        bins (int):
        maxima_thresh (float):

    CommandLine:
        python -m vtool.patch --test-test_ondisk_find_patch_fpath_dominant_orientations

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> patch_fpath = ut.get_argval('--patch-fpath', type_=str, default=ut.grab_test_imgpath('star.png'))
        >>> bins = 36
        >>> maxima_thresh = 0.8
        >>> test_ondisk_find_patch_fpath_dominant_orientations(patch_fpath, bins, maxima_thresh)
        >>> pt.show_if_requested()
    """
    import vtool as vt
    patch = vt.imread(patch_fpath, grayscale=True)
    #submax_ori = submaxima_x[submaxima_y.argmax()]
    #ori_offsets = [submax_ori]  # normalize w.r.t. gravity
    return find_patch_dominant_orientations(patch, bins=bins, maxima_thresh=maxima_thresh, DEBUG_ROTINVAR=DEBUG_ROTINVAR)


def find_patch_dominant_orientations(patch, bins=36, maxima_thresh=.8,
                                     DEBUG_ROTINVAR=False):
    """
    helper
    """
    import plottool as pt
    gradx, grady = patch_gradient(patch, gaussian_weighted=False)
    gori = patch_ori(gradx, grady)
    gmag = patch_mag(gradx, grady)
    gaussian_weighted = True
    # do gaussian weighting correctly
    gori_weights = gaussian_weight_patch(gmag) if gaussian_weighted else gmag
    if DEBUG_ROTINVAR:
        print(ktool.kpts_docrepr(patch[::10, ::10], 'PATCH[::10]', False))
        print(ktool.kpts_docrepr(gradx[::10, ::10], 'gradx[::10]', False))
        print(ktool.kpts_docrepr(grady[::10, ::10], 'grady[::10]', False))
        #print(ktool.kpts_docrepr(patch, 'PATCH', False))
        print(ktool.kpts_docrepr(gori[::10, ::10], 'ORI[::10]', False))
        print(ktool.kpts_docrepr(gmag[::10, ::10], 'MAG[::10]', False))
        print(ktool.kpts_docrepr(gori_weights[::10, ::10], 'WEIGHTS[::10]', False))
    # FIXME: Not taking account to gmag
    #bins = 3
    #bins = 8
    hist, centers = get_orientation_histogram(gori, gori_weights, bins=bins)
    # Find submaxima
    submaxima_x, submaxima_y = htool.argsubmaxima(hist, centers,
                                                  maxima_thresh=maxima_thresh)
    if DEBUG_ROTINVAR:
        htool.show_ori_image(gori, gori_weights / 255.0, patch, gradx, grady)
        pt.set_figtitle('python orimg')
    submax_ori_offsets = submaxima_x
    if DEBUG_ROTINVAR:
        print('submaxima_x, submaxima_y = %r, %r' % (submaxima_x, submaxima_y,))
        htool.show_hist_submaxima(hist, centers=centers)
        pt.set_figtitle('python hist')
        pt.df2.plt.show()
    return submax_ori_offsets


def testdata_patch():
    import plottool as pt
    kpts, vecs, imgBGR = pt.viz_keypoints.testdata_kpts()
    fx = ut.get_argval('--fx', type_=int, default=0)
    kp = kpts[fx]
    patch, wkp = get_warped_patch(imgBGR, kp, gray=True,
                                  #flags=cv2.INTER_LANCZOS4,
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT)
    return patch


def find_dominant_kp_orientations(imgBGR, kp, bins=36, maxima_thresh=.8,
                                  DEBUG_ROTINVAR=False):
    r"""

    References:
        http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf
        page 219

        http://www.cs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf
        page 13.

        Lowe uses a 36-bin histogram of edge orientations weigted by a gaussian
        distance to the center and gradient magintude. He finds all peaks within
        80% of the global maximum. Then he fine tunes the orientation using a
        3-binned parabolic fit. Multiple orientations (and hence multiple
        keypoints) can be returned, but empirically only about 15% will have
        these and they do tend to be important.

    Returns:
        float: ori_offset - offset of current orientation to dominant orientation

    CommandLine:
        python -m vtool.patch --test-find_dominant_kp_orientations

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> np.random.seed(0)
        >>> #imgBGR = get_test_patch('cross', jitter=False)
        >>> img_fpath = ut.grab_test_imgpath('star.png')
        >>> imgBGR = vt.imread(img_fpath)
        >>> kpts, vecs = vt.extract_features(img_fpath)
        >>> assert len(kpts) == 1
        >>> kp = kpts[0]
        >>> print('kp = \n' + (vt.kp_cpp_infostr(kp)))
        >>> bins = 36
        >>> maxima_thresh = .8
        >>> # execute function
        >>> new_oris = find_dominant_kp_orientations(imgBGR, kp, bins,
        >>>                                          maxima_thresh,
        >>>                                          DEBUG_ROTINVAR=True)
        >>> # verify results
        >>> result = 'new_oris = %r' % (new_oris,)
    """
    patch, wkp = get_warped_patch(imgBGR, kp, gray=True,
                                  #flags=cv2.INTER_LANCZOS4,
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT)

    if DEBUG_ROTINVAR:
        pass
        if True:
            import plottool as pt
            fnum = pt.next_fnum()
            pt.imshow(imgBGR, fnum=fnum, pnum=(1, 2, 1))
            pt.draw_kpts2(np.array([kp]), pts=True, rect=True)
            pt.imshow(patch, fnum=fnum, pnum=(1, 2, 2))
            pt.draw_kpts2(np.array([wkp]), pts=True, rect=True)
    # Compute new orientation(s) for this keypoint
    submax_ori_offsets = find_patch_dominant_orientations(patch, bins=bins,
                                                          maxima_thresh=maxima_thresh,
                                                          DEBUG_ROTINVAR=DEBUG_ROTINVAR)
    old_ori = kp[-1]
    new_oris = (old_ori + (submax_ori_offsets + ktool.GRAVITY_THETA)) % TAU
    return new_oris


def get_orientation_histogram(gori, gori_weights, bins=36, DEBUG_ROTINVAR=False):
    r"""
    Args:
        gori (?):
        gori_weights (?):
        bins (int):

    Returns:
        tuple: (hist, centers)

    CommandLine:
        python -m vtool.patch --test-get_orientation_histogram

    Ignore:
        print(vt.kpts_docrepr(gori, 'gori = '))
        print(vt.kpts_docrepr(gori_weights, 'gori_weights = '))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> # build test data
        >>> gori = np.array([[ 0.  ,  0.  ,  3.14,  3.14,  0.  ],
        ...                  [ 4.71,  6.15,  3.13,  3.24,  4.71],
        ...                  [ 4.71,  4.61,  0.5 ,  4.85,  4.71],
        ...                  [ 1.57,  6.28,  3.14,  3.14,  1.57],
        ...                  [ 0.  ,  0.  ,  3.14,  3.14,  0.  ]])
        >>> gori_weights = np.array([[ 0.  ,  0.11,  0.02,  0.13,  0.  ],
        ...                          [ 0.02,  0.19,  0.02,  0.21,  0.02],
        ...                          [ 0.11,  0.16,  0.  ,  0.13,  0.11],
        ...                          [ 0.  ,  0.17,  0.02,  0.19,  0.  ],
        ...                          [ 0.  ,  0.11,  0.02,  0.13,  0.  ]])
        >>> bins = 36
        >>> # execute function
        >>> (hist, centers) = get_orientation_histogram(gori, gori_weights, bins)
        >>> # verify results
        >>> result = str((hist, centers))
        >>> print(result)
    """
    # Get wrapped histogram (because we are finding a direction)
    flat_oris = gori.flatten()
    flat_weights = gori_weights.flatten()
    TAU = np.pi * 2
    range_ = (0, TAU)
    # FIXME: this does not do linear interpolation
    #hist_, edges_ = np.histogram(flat_oris, range=range_, bins=bins, weights=flat_weights)
    # Compute histogram where orientations split weight between bins
    hist_, edges_ = htool.interpolated_histogram(flat_oris, flat_weights,
                                                 range_, bins,
                                                 interpolation_wrap=True,
                                                 _debug=DEBUG_ROTINVAR)
    # Duplicate the first and last edges so neighbor information is contiguous
    hist, edges = htool.wrap_histogram(hist_, edges_, _debug=DEBUG_ROTINVAR)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


# if __name__ == '__main__':
#     """
#     CommandLine:
#         python -m vtool.patch
#         python -m vtool.patch --allexamples
#         python -m vtool.patch --allexamples --noface --nosrc
#     """
#     import multiprocessing
#     multiprocessing.freeze_support()  # for win32
#     import utool as ut  # NOQA
#     ut.doctest_funcs()
if __name__ == '__main__':
    r"""
    CommandLine:
        python -m vtool.patch
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
