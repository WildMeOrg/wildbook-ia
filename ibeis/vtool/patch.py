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
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[patch]', DEBUG=False)


# Command line switch
#sys.argv.append('--vecfield')


TAU = np.pi * 2  # References: tauday.com


@profile
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
    func = {
        'star': get_star_patch,
        'stripe': get_stripe_patch,
    }[key]
    patch = func(jitter)
    return patch


def make_test_image_keypoints(imgBGR, scale=1.0, skew=0, theta=0):
    h, w = imgBGR.shape[0:2]
    half_w, half_h = w / 2.0, h / 2.0
    x, y = half_w - .5, half_h - .5
    a = (half_w) * scale
    c = skew
    d = (half_h) * scale
    theta = theta
    kpts = np.array([[x, y, a, c, d, theta]], np.float32)
    return kpts


def get_star_patch(jitter=False):
    """ test data patch """
    _, O = .1, .8
    star_patch = np.array([
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
        star_patch += np.random.rand(*star_patch.shape) * .1
    return star_patch


def get_stripe_patch(jitter=False):
    """ test data patch """
    _, O = .1, .8
    star_patch = np.array([
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
        star_patch += np.random.rand(*star_patch.shape) * .1
    return star_patch


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
        >>> patch = get_star_patch()
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
    # TODO: use new guass patch weighting function here
    average = weighted_patch.sum()
    return average


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


def test_show_gaussian_patches(shape=(19, 19)):
    r"""
    CommandLine:
        python -m vtool.patch --test-test_show_gaussian_patches --show
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=7,7
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=17,17
        python -m vtool.patch --test-test_show_gaussian_patches --show --shape=41,41
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
    cv2_warp_kwargs = {
        #'flags': cv2.INTER_LINEAR,
        #'flags': cv2.INTER_NEAREST,
        'flags': cv2.INTER_CUBIC,
        #'flags': cv2.INTER_LANCZOS4,
        'borderMode': cv2.BORDER_REPLICATE,
    }
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
        #warped_patch = gtool.warpAffine(img, M, dsize)
        warped_patch = cv2.warpAffine(img, M[0:2], tuple(dsize), **cv2_warp_kwargs)
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
    if gray and len(wpatch.shape) > 2:
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
def find_kpts_direction(imgBGR, kpts):
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
    gravity_ori = ktool.GRAVITY_THETA
    for kp in kpts:
        ori = find_dominant_kp_orientations(imgBGR, kp)
        ori_list.append(ori)
    _oris = np.array(ori_list, dtype=kpts.dtype)
    _oris -= gravity_ori  % TAU  # normalize w.r.t. gravity
    # discard old orientation if they exist
    kpts2 = np.vstack((kpts[:, 0:5].T, _oris)).T
    return kpts2


def test_find_kp_direction():
    """
    Shows steps in orientation estimation

    CommandLine:
        python -m vtool.patch --test-test_find_kp_direction --show
        python -m vtool.patch --test-test_find_kp_direction --show --interact

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.patch import *  # NOQA
        >>> test_find_kp_direction()
        >>> pt.show_if_requested()
    """
    #from vtool.patch import *  # NOQA
    #import vtool as vt
    # build test data
    import utool as ut
    import vtool as vt
    np.random.seed(0)
    imgBGR = get_test_patch('stripe', jitter=True)
    #imgBGR = get_test_patch('star', jitter=True)
    #imgBGR = cv2.resize(imgBGR, (41, 41), interpolation=cv2.INTER_LANCZOS4)
    imgBGR = cv2.resize(imgBGR, (41, 41), interpolation=cv2.INTER_CUBIC)
    theta = 0  # 3.4  # TAU / 16
    kpts = make_test_image_keypoints(imgBGR, scale=.7, theta=theta)
    kp = kpts[0]
    bins = 32
    globals_ = globals()
    locals_ = locals()
    converge_list = []

    while True:
        # Execute test function
        sourcecode = ut.get_func_sourcecode(find_dominant_kp_orientations, stripdef=True, stripret=True)
        six.exec_(sourcecode, globals_, locals_)
        keys = 'patch, gradx, grady, gmag, gori, hist, centers, gori_weights, ori_offsets'.split(', ')
        patch, gradx, grady, gmag, gori, hist, centers, gori_weights, ori_offsets = ut.dict_take(locals_, keys)
        ori = ori_offsets[0]
        print('ori=%r' % (ori,))
        INTERACTIVE_ITERATION = ut.get_argflag('--interact')
        if INTERACTIVE_ITERATION:
            from six.moves import input
            # Change rotation
            old_ori = locals_['kp'][-1]
            new_ori = (old_ori + (ori - vt.GRAVITY_THETA)) % TAU
            #locals_['bins'] *= 2
            #locals_['bins'] = min(locals_['bins'], 128)
            print('new_ori = %r' % (new_ori,))
            print('bins = %r' % (locals_['bins'],))

            locals_['kp'][-1] = new_ori
            converge_list.append(new_ori)
            import plottool as pt
            if True or len(converge_list) >= 4:
                show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag, gori, hist, centers, gori_weights)
                pt.figure(fnum=2, doclf=True)
                pt.plot(converge_list, 'o-g')
                pt.gca().set_ylim(0, TAU)
                pt.dark_background()
                pt.present()
                #pt.update()
                input('next')
        else:
            show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag, gori, hist, centers, gori_weights)
            break


def show_patch_orientation_estimation(imgBGR, kpts, patch, gradx, grady, gmag, gori, hist, centers, gori_weights):
    import plottool as pt
    # DRAW TEST INFO
    fnum = 1
    pt.figure(fnum=1, doclf=True, docla=True)
    gorimag = pt.color_orimag(gori, gmag, False)
    nRows, nCols = pt.get_square_row_cols(9)
    nRows += 1
    next_pnum = pt.make_pnum_nextgen(nRows, nCols)
    pt.imshow(255 * imgBGR, update=True, fnum=fnum, pnum=next_pnum(), title='orig image')
    pt.draw_kpts2(kpts, rect=True, ori=True)
    pt.imshow(patch * 255, fnum=fnum, pnum=next_pnum(), title='sample')
    pt.imshow((np.abs(gradx)) * 255, fnum=fnum, pnum=next_pnum(), title='gradx')
    pt.imshow((np.abs(grady)) * 255, fnum=fnum, pnum=next_pnum(), title='grady')
    pt.imshow(gmag * 255, fnum=fnum, pnum=next_pnum(), title='mag')
    pt.imshow(gori_weights * 255, fnum=fnum, pnum=next_pnum(), title='weights')
    pt.imshow(ut.norm_zero_one(gori) * 255, fnum=fnum, pnum=next_pnum(), title='ori')
    pt.draw_vector_field(gradx, grady, pnum=next_pnum(), fnum=fnum, title='gori (vec)')
    pt.imshow(gorimag, fnum=fnum, pnum=next_pnum(), title='ori-color')
    pt.color_orimag_colorbar(gori)
    pt.figure(fnum=fnum, pnum=(nRows, 1, nRows))
    pt.draw_hist_subbin_maxima(hist, centers)
    pt.set_xlabel('radians')
    pt.set_ylabel('weight')
    #pt.update()


def find_dominant_kp_orientations(imgBGR, kp, bins=32):
    """

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
        >>> imgBGR = get_test_patch('stripe', jitter=False)
        >>> kpts = make_test_image_keypoints(imgBGR)
        >>> kp = kpts[0]
        >>> # execute function
        >>> ori_offsets = find_dominant_kp_orientations(imgBGR, kp)
        >>> # verify results
        >>> result = str(ori_offsets)
        >>> print(result)
    """
    patch, wkp = get_warped_patch(imgBGR, kp, gray=True)
    gradx, grady = patch_gradient(patch, gaussian_weighted=False)
    gori = patch_ori(gradx, grady)
    gmag = patch_mag(gradx, grady)
    gaussian_weighted = True
    # do gaussian weighting correctly
    gori_weights = gaussian_weight_patch(gmag) if gaussian_weighted else gmag
    # FIXME: Not taking account to gmag
    #bins = 3
    #bins = 8
    hist, centers = get_orientation_histogram(gori, gori_weights, bins=bins)
    # Find submaxima
    submaxima_x, submaxima_y = htool.hist_interpolated_submaxima(hist, centers)
    #ut.embed()
    submax_ori = submaxima_x[submaxima_y.argmax()]
    ori_offsets = [submax_ori]  # normalize w.r.t. gravity
    return ori_offsets


@profile
def get_orientation_histogram(gori, gori_weights, bins=32):
    # Get wrapped histogram (because we are finding a direction)
    flat_oris = gori.flatten()
    flat_weights = gori_weights.flatten()
    TAU = np.pi * 2
    range_ = (0, TAU)
    hist_, edges_ = np.histogram(flat_oris, range=range_, bins=bins, weights=flat_weights)
    hist, edges = htool.wrap_histogram(hist_, edges_)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


def gaussian_weight_patch(patch):
    #patch = np.ones(patch.shape)
    sigma0 = (patch.shape[0] / 2) * .95
    sigma1 = (patch.shape[1] / 2) * .95
    #sigma0 = (patch.shape[0] / 2) * .5
    #sigma1 = (patch.shape[1] / 2) * .5
    gauss_kernel_d0 = (cv2.getGaussianKernel(patch.shape[0], sigma0))
    gauss_kernel_d1 = (cv2.getGaussianKernel(patch.shape[1], sigma1))
    gauss_kernel_d0 /= gauss_kernel_d0.max()
    gauss_kernel_d1 /= gauss_kernel_d1.max()
    weighted_patch = (patch * gauss_kernel_d0.T) * gauss_kernel_d1
    return weighted_patch


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
