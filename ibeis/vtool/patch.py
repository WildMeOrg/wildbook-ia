# LICENCE
from __future__ import absolute_import, division, print_function
# Python
import six  # NOQA
from six.moves import zip
#import itertools
#if six.PY2:
#    from functools32 import lru_cache  # Python2.7 support
#elif six.PY3:
#    from functools import lru_cache  # Python3 only
# Science
import cv2
import numpy as np
# VTool
from vtool import histogram as htool
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import image as gtool
from vtool import trig
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[patch]', DEBUG=False)


# Command line switch
#sys.argv.append('--vecfield')


np.tau = 2 * np.pi  # References: tauday.com


@profile
def patch_gradient(patch, ksize=1, gaussian_weighted=True):
    patch_ = np.array(patch, dtype=np.float64)
    gradx = cv2.Sobel(patch_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(patch_, cv2.CV_64F, 0, 1, ksize=ksize)
    if gaussian_weighted:
        gausspatch = gaussian_patch(shape=gradx.shape)
        gradx *= gausspatch
        grady *= gausspatch
    return gradx, grady


def patch_mag(gradx, grady):
    return np.sqrt((gradx ** 2) + (grady ** 2))


def patch_ori(gradx, grady):
    """ returns patch orientation relative to the x-axis """
    gori = trig.atan2(grady, gradx)
    return gori


def gaussian_average_patch(patch, sigma=None):
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
        >>> # build test data
        >>> patch = np.array([
        ... [.1, .1, .1, .8, .1, .1, .1],
        ... [.1, .1, .1, .8, .1, .1, .1],
        ... [.1, .1, .8, .8, .8, .1, .1],
        ... [.8, .8, .8, .8, .8, .8, .8],
        ... [.1, .8, .8, .8, .8, .8, .1],
        ... [.1, .1, .8, .8, .8, .1, .1],
        ... [.1, .8, .8, .8, .8, .8, .1],
        ... [.1, .8, .1, .1, .1, .8, .1],
        ... [.8, .1, .1, .1, .1, .1, .8]])
        >>> sigma = 1.6
        >>> # execute function
        >>> result = gaussian_average_patch(patch, sigma)
        >>> # verify results
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
    weighted_patch = patch.copy()
    weighted_patch = np.multiply(weighted_patch,   gauss_kernel_d0)
    weighted_patch = np.multiply(weighted_patch.T, gauss_kernel_d1).T
    average = weighted_patch.sum()
    return average


def test_show_gaussian_patches():
    r"""
    CommandLine:
        python -m vtool.patch --test-test_show_gaussian_patches --show

    References:
        http://matplotlib.org/examples/mplot3d/surface3d_demo.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> import plottool as pt
        >>> test_show_gaussian_patches()
        >>> pt.show_if_requested()
    """
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    import plottool as pt
    import numpy as np
    import matplotlib as mpl
    import vtool as vt
    #shape = (27, 27)
    shape = (7, 7)
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
        width (int):
        hight (int):
        shape (tuple):
        sigma (float):
        norm_01 (bool):

    CommandLine:
        python -m vtool.patch --test-gaussian_patch2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.patch import *  # NOQA
        >>> # build test data
        >>> width = 3
        >>> hieght = 3
        >>> shape = (7, 7)
        >>> sigma = 1.0
        >>> norm_01 = False
        >>> # execute function
        >>> gausspatch = gaussian_patch2(width, hight, shape, sigma, norm_01)
        >>> assert gausspatch.sum() == 1.0

    Ignore:
        import plottool as pt
        pt.imshow(gausspatch * 255)
        pt.update()
    """
    gauss_kernel_d0 = (cv2.getGaussianKernel(shape[0], sigma))
    gauss_kernel_d1 = (cv2.getGaussianKernel(shape[1], sigma))
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


@profile
def get_unwarped_patches(img, kpts):
    """ Returns cropped unwarped (keypoint is still elliptical) patch around a keypoint

    Args:
        img (ndarray): array representing an image
        kpts (ndarrays): keypoint ndarrays in [x, y, a, c, d, theta] format
    Returns:
        tuple : (patches, subkpts) - the unnormalized patches from the img corresonding to the keypoint

    """
    _xs, _ys = ktool.get_xys(kpts)
    xyexnts = ktool.get_xy_axis_extents(kpts)
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


@profile
def get_warped_patches(img, kpts):
    """ Returns warped (into a unit circle) patch around a keypoint

    Args:
        img (ndarray): array representing an image
        kpts (ndarrays): keypoint ndarrays in [x, y, a, c, d, theta] format
    Returns:
        tuple : (warped_patches, warped_subkpts) the normalized 41x41 patches from the img corresonding to the keypoint
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
    s = 41  # sf
    for x, y, V, ori in kpts_iter:
        ss = np.sqrt(s) * 3
        (h, w) = img.shape[0:2]
        # Translate to origin(0,0) = (x,y)
        T = ltool.translation_mat3x3(-x, -y)
        R = ltool.rotation_mat3x3(-ori)
        S = ltool.scale_mat3x3(ss)
        X = ltool.translation_mat3x3(s / 2, s / 2)
        M = X.dot(S).dot(R).dot(V).dot(T)
        # Prepare to warp
        dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
        # Warp
        warped_patch = gtool.warpAffine(img, M, dsize)
        # Build warped keypoints
        wkp = np.array((s / 2, s / 2, ss, 0., ss, 0))
        warped_patches.append(warped_patch)
        warped_subkpts.append(wkp)
    return warped_patches, warped_subkpts


@profile
def get_warped_patch(imgBGR, kp, gray=False):
    """Returns warped (into a unit circle) patch around a keypoint

    Args:
        img (ndarray): array representing an image
        kpt (ndarray): keypoint ndarray in [x, y, a, c, d, theta] format
    Returns:
        tuple : (wpatch, wkp) the normalized 41x41 patches from the img corresonding to the keypoint
    """
    kpts = np.array([kp])
    wpatches, wkpts = get_warped_patches(imgBGR, kpts)
    wpatch = wpatches[0]
    wkp = wkpts[0]
    if gray:
        wpatch = gtool.cvt_BGR2L(wpatch)
    return wpatch, wkp


@profile
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


@profile
def get_orientation_histogram(gori):
    # Get wrapped histogram (because we are finding a direction)
    hist_, edges_ = np.histogram(gori.flatten(), bins=8)
    hist, edges = htool.wrap_histogram(hist_, edges_)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


@profile
def find_kpts_direction(imgBGR, kpts):
    ori_list = []
    gravity_ori = ktool.GRAVITY_THETA
    for kp in kpts:
        patch, wkp = get_warped_patch(imgBGR, kp, gray=True)
        gradx, grady = patch_gradient(patch)
        gori = patch_ori(gradx, grady)
        # FIXME: Not taking account to gmag
        hist, centers = get_orientation_histogram(gori)
        # Find submaxima
        submaxima_x, submaxima_y = htool.hist_interpolated_submaxima(hist, centers)
        submax_ori = submaxima_x[submaxima_y.argmax()]
        ori = (submax_ori)  # normalize w.r.t. gravity
        ori_list.append(ori)
    _oris = np.array(ori_list, dtype=kpts.dtype)
    _oris -= gravity_ori  % np.tau  # normalize w.r.t. gravity
    # discard old orientation if they exist
    kpts2 = np.vstack((kpts[:, 0:5].T, _oris)).T
    return kpts2

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.patch
        python -m vtool.patch --allexamples
        python -m vtool.patch --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
