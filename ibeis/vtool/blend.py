# -*- coding: utf-8 -*-
# LICENCE: Apache2
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, range  # NOQA
import numpy as np
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[blend]')


def testdata_blend(scale=128):
    import vtool as vt
    img_fpath = ut.grab_test_imgpath('lena.png')
    img1 = vt.imread(img_fpath)
    rng = np.random.RandomState(0)
    img2 = vt.perlin_noise(img1.shape[0:2], scale=scale, rng=rng)[None, :].T
    img1 = vt.rectify_to_float01(img1)
    img2 = vt.rectify_to_float01(img2)
    return img1, img2


def gridsearch_image_function(param_info, test_func, args=tuple(), show_func=None):
    """
    gridsearch for a function that produces a single image
    """
    import plottool as pt
    cfgdict_list, cfglbl_list = param_info.get_gridsearch_input(defaultslice=slice(0, 10))
    fnum = pt.ensure_fnum(None)
    if show_func is None:
        show_func = pt.imshow
    lbl = ut.get_funcname(test_func)
    cfgresult_list = [
        test_func(*args, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl=lbl)
    ]
    onclick_func = None
    ut.interact_gridsearch_result_images(
        show_func, cfgdict_list, cfglbl_list,
        cfgresult_list, fnum=fnum,
        figtitle=lbl, unpack=False,
        max_plots=25, onclick_func=onclick_func)
    pt.iup()


def blend_images(img1, img2, mode='average', **kwargs):
    if mode == 'average':
        return blend_images_average(img1, img2, **kwargs)
    elif mode == 'multiply':
        return blend_images_multiply(img1, img2, **kwargs)
    else:
        raise ValueError('mode = %r' % (mode,))


def blend_images_average(img1, img2, alpha=.5):
    r"""
    Args:
        img1 (ndarray[uint8_t, ndim=2]):  image data
        img2 (ndarray[uint8_t, ndim=2]):  image data
        alpha (float): (default = 0.5)

    Returns:
        ndarray: imgB

    References:
        https://en.wikipedia.org/wiki/Blend_modes

    CommandLine:
        python -m vtool.blend --test-blend_images_average:0 --show
        python -m vtool.blend --test-blend_images_average:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.blend import *  # NOQA
        >>> alpha = 0.8
        >>> img1, img2 = testdata_blend()
        >>> imgB = blend_images_average(img1, img2, alpha)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(imgB)
        >>> ut.show_if_requested()

    Example2:
        >>> # GRIDSEARCH
        >>> from vtool.blend import *  # NOQA
        >>> test_func = blend_images_average
        >>> args = testdata_blend()
        >>> param_info = ut.ParamInfoList('blend_params', [
        ...    ut.ParamInfo('alpha', .8, 'alpha=',
        ...                 varyvals=np.linspace(0, 1.0, 25).tolist()),
        ... ])
        >>> gridsearch_image_function(param_info, test_func, args)
        >>> ut.show_if_requested()
    """
    #assert img1.shape == img2.shape, 'chips must be same shape to blend'
    #imgB = np.zeros(img2.shape, dtype=img2.dtype)
    #assert img1.min() >= 0 and img1.max() <= 1
    #assert img2.min() >= 0 and img2.max() <= 1
    if isinstance(alpha, np.ndarray):
        import vtool as vt
        img1, img2 = vt.make_channels_comparable(img1, img2)
        img1, alpha = vt.make_channels_comparable(img1, alpha)
        img2, alpha = vt.make_channels_comparable(img2, alpha)
        imgB = (img1 * (1.0 - alpha)) + (img2 * (alpha))
    else:
        imgB = (img1 * (1.0 - alpha)) + (img2 * (alpha))
    #assert imgB.min() >= 0 and imgB.max() <= 1
    return imgB


def blend_images_mult_average(img1, img2, alpha=.5):
    r"""
    Args:
        img1 (ndarray[uint8_t, ndim=2]):  image data
        img2 (ndarray[uint8_t, ndim=2]):  image data
        alpha (float): (default = 0.5)

    Returns:
        ndarray: imgB

    References:
        https://en.wikipedia.org/wiki/Blend_modes

    CommandLine:
        python -m vtool.blend --test-blend_images_mult_average:0 --show
        python -m vtool.blend --test-blend_images_mult_average:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.blend import *  # NOQA
        >>> alpha = 0.8
        >>> img1, img2 = testdata_blend()
        >>> imgB = blend_images_mult_average(img1, img2, alpha)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(imgB)
        >>> ut.show_if_requested()

    Example2:
        >>> # GRIDSEARCH
        >>> from vtool.blend import *  # NOQA
        >>> test_func = blend_images_mult_average
        >>> args = testdata_blend()
        >>> param_info = ut.ParamInfoList('blend_params', [
        ...    ut.ParamInfo('alpha', .8, 'alpha=',
        ...                 varyvals=np.linspace(0, 1.0, 25).tolist()),
        ... ])
        >>> gridsearch_image_function(param_info, test_func, args)
        >>> ut.show_if_requested()
    """
    #assert img1.shape == img2.shape, 'chips must be same shape to blend'
    #imgB = np.zeros(img2.shape, dtype=img2.dtype)
    #assert img1.min() >= 0 and img1.max() <= 1
    #assert img2.min() >= 0 and img2.max() <= 1
    imgB = blend_images_average(img1, blend_images_multiply(img1, img2, .5), alpha)
    #assert imgB.min() >= 0 and imgB.max() <= 1
    return imgB


def blend_images_multiply(img1, img2, alpha=0.5):
    r"""
    Args:
        img1 (ndarray[uint8_t, ndim=2]):  image data
        img2 (ndarray[uint8_t, ndim=2]):  image data
        alpha (float): (default = 0.5)

    Returns:
        ndarray: imgB


    References:
        https://en.wikipedia.org/wiki/Blend_modes

    CommandLine:
        python -m vtool.blend --test-blend_images_multiply:0 --show
        python -m vtool.blend --test-blend_images_multiply:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.blend import *  # NOQA
        >>> alpha = 0.8
        >>> img1, img2 = testdata_blend()
        >>> imgB = blend_images_multiply(img1, img2, alpha)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(imgB)
        >>> ut.show_if_requested()

    Example2:
        >>> # GRIDSEARCH
        >>> from vtool.blend import *  # NOQA
        >>> test_func = blend_images_multiply
        >>> args = testdata_blend(scale=128)
        >>> param_info = ut.ParamInfoList('blend_params', [
        ...    ut.ParamInfo('alpha', .8, 'alpha=',
        ...                 varyvals=np.linspace(0, 1.0, 25).tolist()),
        ... ])
        >>> gridsearch_image_function(param_info, test_func, args)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    # rectify type
    img1_ = vt.rectify_to_float01(img1)
    img2_ = vt.rectify_to_float01(img2)
    #assert img1_.min() >= 0 and img1_.max() <= 1
    #assert img2_.min() >= 0 and img2_.max() <= 1
    # apply transform
    #if False and alpha == .5:
    #imgB = img1_ * img2_
    #else:
    #data = [img1_, img2_]
    w1 = 1.0 - alpha + .5
    w2 = alpha + .5
    #weights = [w1, w2]
    #imgB = vt.weighted_geometic_mean(data, weights)
    #imgB = ((img1_ ** w1) * (img2_ ** w2)) ** (1 / (w1 + w2))
    imgB = ((img1_ ** w1) * (img2_ ** w2))
    #imgB = vt.weighted_geometic_mean_unnormalized(data, weights)
    # unrectify
    #assert imgB.min() >= 0 and imgB.max() <= 1
    return imgB


def gridsearch_addWeighted():
    r"""
    CommandLine:
        python -m vtool.blend --test-gridsearch_addWeighted --show

    Example:
        >>> # GRIDSEARCH
        >>> from vtool.blend import *  # NOQA
        >>> gridsearch_addWeighted()
        >>> ut.show_if_requested()
    """
    import cv2
    import vtool as vt
    def test_func(src1, src2, alpha=1.0, **kwargs):
        beta = 1.0 - alpha
        src1 = vt.rectify_to_float01(src1)
        src2 = vt.rectify_to_float01(src2)
        dst = np.empty(src1.shape, dtype=src1.dtype)
        cv2.addWeighted(src1=src1, src2=src2, dst=dst, alpha=alpha, beta=beta, dtype=-1, **kwargs)
        return dst
    img1, img2 = testdata_blend()
    args = img1, img2 = vt.make_channels_comparable(img1, img2)
    param_info = ut.ParamInfoList('blend_params', [
        ut.ParamInfo('alpha', .8,
                     varyvals=np.linspace(0, 1.0, 5).tolist()),
        #ut.ParamInfo('beta', .8,
        ut.ParamInfo('gamma', .0,
                     varyvals=np.linspace(0, 1.0, 5).tolist()),
        #varyvals=[.0],))
        #ut.ParamInfo('gamma', .8, 'alpha=',
        #             varyvals=np.linspace(0, 1.0, 9).tolist()),
    ])
    gridsearch_image_function(param_info, test_func, args)


def gamma_adjust(img, gamma=1.0):
    """
    CommandLine:
        python -m vtool.blend --test-gamma_adjust:0 --show

    Example2:
        >>> # GRIDSEARCH
        >>> from vtool.blend import *  # NOQA
        >>> import vtool as vt
        >>> test_func = gamma_adjust
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.rectify_to_float01(vt.imread(img_fpath))
        >>> args = (img,)
        >>> param_info = ut.ParamInfoList('blend_params', [
        ...    ut.ParamInfo('gamma', .8, 'gamma=',
        ...                 varyvals=np.linspace(.1, 2.5, 25).tolist()),
        ... ])
        >>> gridsearch_image_function(param_info, test_func, args)
        >>> ut.show_if_requested()
    """
    assert img.max() <= 1.0
    assert img.min() >= 0.0
    return img ** gamma


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.blend
        python -m vtool.blend --allexamples
        python -m vtool.blend --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
