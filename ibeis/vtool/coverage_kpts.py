# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range, map, reduce  # NOQA
import cv2
import numpy as np
import utool as ut
import ubelt as ub
from vtool import patch as ptool
from vtool import keypoint as ktool


# TODO: integrate more
COVKPTS_DEFAULT = ut.ParamInfoList('coverage_kpts', [
    ut.ParamInfo('cov_agg_mode' , 'max'),
    ut.ParamInfo('cov_blur_ksize' , (5, 5)),
    ut.ParamInfo('cov_blur_on' , True),
    ut.ParamInfo('cov_blur_sigma' , 5.0),
    ut.ParamInfo('cov_remove_scale' , True),
    ut.ParamInfo('cov_remove_shape' , True),
    ut.ParamInfo('cov_scale_factor' , .3),
    ut.ParamInfo('cov_size_penalty_frac' , .1),
    ut.ParamInfo('cov_size_penalty_on' , True),
    ut.ParamInfo('cov_size_penalty_power' , .5),
])


def make_kpts_heatmask(kpts, chipsize, cmap='plasma'):
    """
    makes a heatmap overlay for keypoints

    CommandLine:
        python -m vtool.coverage_kpts make_kpts_heatmask --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:plottool)
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import vtool as vt
        >>> import pyhesaff
        >>> img_fpath = ut.grab_test_imgpath('carl.png')
        >>> (kpts, vecs) = pyhesaff.detect_feats(img_fpath)
        >>> chip = vt.imread(img_fpath)
        >>> kpts = kpts[0:100]
        >>> chipsize = chip.shape[0:2][::-1]
        >>> heatmask = make_kpts_heatmask(kpts, chipsize)
        >>> img1 = heatmask
        >>> img2 = chip
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> img3 = vt.overlay_alpha_images(heatmask, chip)
        >>> pt.imshow(img3)
        >>> #pt.imshow(heatmask)
        >>> #pt.draw_kpts2(kpts)
        >>> pt.show_if_requested()
    """
    # use a disk instead of a gaussian
    import skimage.morphology
    cov_scale_factor = .25
    radius = min(int((min(chipsize) * cov_scale_factor) // 2) - 1, 50)
    patch = skimage.morphology.disk(radius)
    mask = make_kpts_coverage_mask(kpts, chipsize, resize=True,
                                   cov_size_penalty_on=False,
                                   patch=patch,
                                   cov_scale_factor=cov_scale_factor,
                                   cov_blur_sigma=1.5,
                                   cov_blur_on=True)
    import plottool as pt
    # heatmask = np.ones(tuple(chipsize) + (4,)) * pt.RED
    heatmask = pt.plt.get_cmap(cmap)(mask)
    # conver to bgr
    heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
    # apply alpha channel
    heatmask[:, :, 3] = mask * .5
    return heatmask


def make_heatmask(mask, cmap='plasma'):
    # import vtool as vt
    # use a disk instead of a gaussian
    import plottool as pt
    import vtool as vt
    assert len(mask.shape) == 2
    mask = vt.rectify_to_float01(mask)
    heatmask = pt.plt.get_cmap(cmap)(mask)
    # conver to bgr
    heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
    heatmask[:, :, 3] = mask
    # print('heatmask = {!r}'.format(heatmask))
    return heatmask


def make_kpts_coverage_mask(
        kpts, chipsize,
        weights=None,
        return_patch=False,
        patch=None,
        resize=False,
        out=None,
        cov_blur_on=True,
        cov_disk_hack=None,
        cov_blur_ksize=(17, 17),
        cov_blur_sigma=5.0,
        cov_gauss_shape=(19, 19),
        cov_gauss_sigma_frac=.3,
        cov_scale_factor=.2,
        cov_agg_mode='max',
        cov_remove_shape=False,
        cov_remove_scale=False,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    r"""
    Returns a intensity image denoting which pixels are covered by the input
    keypoints

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        chipsize (tuple): width height of the underlying image

    Returns:
        tuple (ndarray, ndarray): dstimg, patch

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:plottool)
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import vtool as vt
        >>> import plottool as pt
        >>> import pyhesaff
        >>> img_fpath = ut.grab_test_imgpath('carl.png')
        >>> (kpts, vecs) = pyhesaff.detect_feats(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> chipsize = chip.shape[0:2][::-1]
        >>> # execute function
        >>> dstimg, patch = make_kpts_coverage_mask(kpts, chipsize, resize=True, return_patch=True, cov_size_penalty_on=False, cov_blur_on=False)
        >>> # show results
        >>> # xdoctest: +REQUIRES(--show)
        >>> mask = dstimg
        >>> show_coverage_map(chip, mask, patch, kpts)
        >>> pt.show_if_requested()
    """
    if patch is None:
        patch = get_gaussian_weight_patch(cov_gauss_shape, cov_gauss_sigma_frac)
    chipshape = chipsize[::-1]
    # Warp patches onto a scaled image
    dstimg = warp_patch_onto_kpts(
        kpts, patch, chipshape, weights=weights, out=out,
        cov_scale_factor=cov_scale_factor,
        cov_agg_mode=cov_agg_mode,
        cov_remove_shape=cov_remove_shape,
        cov_remove_scale=cov_remove_scale,
        cov_size_penalty_on=cov_size_penalty_on,
        cov_size_penalty_power=cov_size_penalty_power,
        cov_size_penalty_frac=cov_size_penalty_frac
    )
    # Smooth weight of influence
    if cov_blur_on:
        cv2.GaussianBlur(dstimg, ksize=cov_blur_ksize, sigmaX=cov_blur_sigma,
                         sigmaY=cov_blur_sigma, dst=dstimg,
                         borderType=cv2.BORDER_CONSTANT)
    if resize:
        # Resize to original chpsize of requested
        dsize = chipsize
        dstimg = cv2.resize(dstimg, dsize)
    if return_patch:
        return dstimg, patch
    else:
        return dstimg


def warp_patch_onto_kpts(
        kpts, patch, chipshape,
        weights=None,
        out=None,
        cov_scale_factor=.2,
        cov_agg_mode='max',
        cov_remove_shape=False,
        cov_remove_scale=False,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch (ndarray): patch to warp (like gaussian)
        chipshape (tuple):
        weights (ndarray): score for every keypoint

    Kwargs:
        cov_scale_factor (float):

    Returns:
        ndarray: mask

    CommandLine:
        python -m vtool.coverage_kpts --test-warp_patch_onto_kpts
        python -m vtool.coverage_kpts --test-warp_patch_onto_kpts --show
        python -m vtool.coverage_kpts --test-warp_patch_onto_kpts --show --hole
        python -m vtool.coverage_kpts --test-warp_patch_onto_kpts --show --square
        python -m vtool.coverage_kpts --test-warp_patch_onto_kpts --show --square --hole

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import vtool as vt
        >>> import pyhesaff
        >>> img_fpath    = ut.grab_test_imgpath('carl.jpg')
        >>> (kpts, vecs) = pyhesaff.detect_feats(img_fpath)
        >>> kpts = kpts[::15]
        >>> chip = vt.imread(img_fpath)
        >>> chipshape = chip.shape
        >>> weights = np.ones(len(kpts))
        >>> cov_scale_factor = 1.0
        >>> srcshape = (19, 19)
        >>> radius = srcshape[0] / 2.0
        >>> sigma = 0.4 * radius
        >>> SQUARE = ub.argflag('--square')
        >>> HOLE = ub.argflag('--hole')
        >>> if SQUARE:
        >>>     patch = np.ones(srcshape)
        >>> else:
        >>>     patch = ptool.gaussian_patch(shape=srcshape, sigma=sigma) #, norm_01=False)
        >>>     patch = patch / patch.max()
        >>> if HOLE:
        >>>     patch[int(patch.shape[0] / 2), int(patch.shape[1] / 2)] = 0
        >>> # execute function
        >>> dstimg = warp_patch_onto_kpts(kpts, patch, chipshape, weights, cov_scale_factor=cov_scale_factor)
        >>> # verify results
        >>> print('dstimg stats %r' % (ut.get_stats_str(dstimg, axis=None)),)
        >>> print('patch stats %r' % (ut.get_stats_str(patch, axis=None)),)
        >>> #print(patch.sum())
        >>> assert np.all(ut.inbounds(dstimg, 0, 1, eq=True))
        >>> # show results
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> mask = dstimg
        >>> show_coverage_map(chip, mask, patch, kpts)
        >>> pt.show_if_requested()
    """
    import vtool as vt
    #if len(kpts) == 0:
    #    return None
    chip_scale_h = int(np.ceil(chipshape[0] * cov_scale_factor))
    chip_scale_w = int(np.ceil(chipshape[1] * cov_scale_factor))
    if len(kpts) == 0:
        dstimg =  np.zeros((chip_scale_h, chip_scale_w))
        return dstimg
    if weights is None:
        weights = np.ones(len(kpts))
    dsize = (chip_scale_w, chip_scale_h)
    # Allocate destination image
    patch_shape = patch.shape
    # Scale keypoints into destination image
    # <HACK>
    if cov_remove_shape:
        # disregard affine information in keypoints
        # i still dont understand why we are trying this
        (patch_h, patch_w) = patch_shape
        half_width  = (patch_w / 2.0)  # - .5
        half_height = (patch_h / 2.0)  # - .5
        # Center src image
        T1 = vt.translation_mat3x3(-half_width + .5, -half_height + .5)
        # Scale src to the unit circle
        if not cov_remove_scale:
            S1 = vt.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
        # Transform the source image to the keypoint ellipse
        kpts_T = np.array([vt.translation_mat3x3(x, y) for (x, y) in vt.get_xys(kpts).T])
        if not cov_remove_scale:
            kpts_S = np.array([vt.scale_mat3x3(np.sqrt(scale))
                               for scale in vt.get_scales(kpts).T])
        # Adjust for the requested scale factor
        S2 = vt.scale_mat3x3(cov_scale_factor, cov_scale_factor)
        #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
        if not cov_remove_scale:
            M_list = reduce(vt.matrix_multiply, (S2, kpts_T, kpts_S, S1, T1))
        else:
            M_list = reduce(vt.matrix_multiply, (S2, kpts_T, T1))
    # </HACK>
    else:
        M_list = ktool.get_transforms_from_patch_image_kpts(kpts, patch_shape,
                                                            cov_scale_factor)
    affmat_list = M_list[:, 0:2, :]
    weight_list = weights
    # For each keypoint warp a gaussian scaled by the feature score into the image
    warped_patch_iter = warped_patch_generator(
        patch, dsize, affmat_list, weight_list,
        cov_size_penalty_on=cov_size_penalty_on,
        cov_size_penalty_power=cov_size_penalty_power,
        cov_size_penalty_frac=cov_size_penalty_frac)
    # Either max or sum
    if cov_agg_mode == 'max':
        dstimg = vt.iter_reduce_ufunc(np.maximum, warped_patch_iter, out=out)
    elif cov_agg_mode == 'sum':
        dstimg = vt.iter_reduce_ufunc(np.add, warped_patch_iter, out=out)
        # HACK FOR SUM: DO NOT DO THIS FOR MAX
        dstimg[dstimg > 1.0] = 1.0
    else:
        raise AssertionError('Unknown cov_agg_mode=%r' % (cov_agg_mode,))
    return dstimg


def warped_patch_generator(
        patch, dsize, affmat_list, weight_list,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    """
    generator that warps the patches (like gaussian) onto an image with dsize
    using constant memory.

    output must be used or copied on every iteration otherwise the next output
    will clobber the previous

    References:
        http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
    """
    shape = dsize[::-1]
    #warpAffine is weird. If the shape of the dst is the same as src we can
    #use the dst outvar. I dont know why it needs that.  It seems that this
    #will not operate in place even if a destination array is passed in when
    #src.shape != dst.shape.
    patch_h, patch_w = patch.shape
    # If we pad the patch we can use dst
    padded_patch = np.zeros(shape, dtype=np.float32)
    # Prealloc output,
    warped = np.zeros(shape, dtype=np.float32)
    prepad_h, prepad_w = patch.shape[0:2]
    # each score is spread across its contributing pixels
    for (M, weight) in zip(affmat_list, weight_list):
        # inplace weighting of the patch
        np.multiply(patch, weight, out=padded_patch[:prepad_h, :prepad_w] )
        # inplace warping of the padded_patch
        cv2.warpAffine(padded_patch, M, dsize, dst=warped,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
        if cov_size_penalty_on:
            # TODO: size penalty should be based of splitting number of
            # bins in a keypoint over the region that it covers
            total_weight = (warped.sum() ** cov_size_penalty_power) * cov_size_penalty_frac
            if total_weight > 1:
                # Whatever the size of the keypoint is it should
                # contribute a total of 1 score
                np.divide(warped, total_weight, out=warped)
        yield warped


def get_gaussian_weight_patch(gauss_shape=(19, 19), gauss_sigma_frac=.3,
                              gauss_norm_01=True):
    r"""
    2d gaussian image useful for plotting

    Returns:
        ndarray: patch

    CommandLine:
        python -m vtool.coverage_kpts --test-get_gaussian_weight_patch

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> patch = get_gaussian_weight_patch()
        >>> result = str(patch)
        >>> print(result)
    """
    # Perdoch uses roughly .95 of the radius
    radius = gauss_shape[0] / 2.0
    sigma = gauss_sigma_frac * radius
    # Similar to SIFT's computeCircularGaussMask in helpers.cpp
    # uses smmWindowSize=19 in hesaff for patch size. and 1.6 for sigma
    # Create gaussian image to warp
    patch = ptool.gaussian_patch(shape=gauss_shape, sigma=sigma)
    if gauss_norm_01:
        np.divide(patch, patch.max(), out=patch)
    return patch


def get_coverage_kpts_gridsearch_configs():
    """ testing function """
    varied_dict = {
        'cov_agg_mode'           : ['max', 'sum'],
        #'cov_blur_ksize'         : [(19, 19), (5, 5)],
        'cov_blur_ksize'         : [(5, 5)],
        'cov_blur_on'            : [True, False],
        'cov_blur_sigma'         : [5.0],
        'cov_remove_scale'       : [True],
        'cov_remove_shape'       : [False, True],
        'cov_scale_factor'       : [.3],
        'cov_size_penalty_frac'  : [.1],
        'cov_size_penalty_on'    : [True],
        'cov_size_penalty_power' : [.5],
    }
    slice_dict = {
        'cov_scale_factor' : slice(0, 3),
        'cov_agg_mode'     : slice(0, 2),
        'cov_blur_ksize'   : slice(0, 2),
        #'grid_sigma'        : slice(0, 4),
    }
    slice_dict = None
    # Make configuration for every parameter setting
    def constrain_func(cfgdict):
        if cfgdict['cov_remove_shape']:
            cfgdict['cov_remove_scale'] = False
            cfgdict['cov_size_penalty_on'] = False
        if not cfgdict['cov_size_penalty_on']:
            cfgdict['cov_size_penalty_power'] = None
            cfgdict['cov_size_penalty_frac'] = None
        if not cfgdict['cov_blur_on']:
            cfgdict['cov_blur_ksize'] = None
            cfgdict['cov_blur_sigma'] = None
        return cfgdict
    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, constrain_func, slice_dict)
    return cfgdict_list, cfglbl_list


def gridsearch_kpts_coverage_mask():
    """
    testing function

    CommandLine:
        python -m vtool.coverage_kpts --test-gridsearch_kpts_coverage_mask --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_kpts_coverage_mask()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    cfgdict_list, cfglbl_list = get_coverage_kpts_gridsearch_configs()
    kpts, chipsize, weights = testdata_coverage('easy1.png')
    imgmask_list = [
        255 *  make_kpts_coverage_mask(kpts, chipsize, weights,
                                       return_patch=False, **cfgdict)
        for cfgdict in ub.ProgIter(cfgdict_list, label='coverage grid')
    ]
    #NORMHACK = True
    #if NORMHACK:
    #    imgmask_list = [
    #        255 * (mask / mask.max()) for mask in imgmask_list
    #    ]
    fnum = pt.next_fnum()
    ut.interact_gridsearch_result_images(
        pt.imshow, cfgdict_list, cfglbl_list,
        imgmask_list, fnum=fnum, figtitle='coverage image', unpack=False,
        max_plots=25)
    pt.iup()


def testdata_coverage(fname=None):
    """ testing function """
    import vtool as vt
    # build test data
    kpts, vecs = vt.demodata.get_testdata_kpts(fname, with_vecs=True)
    # HACK IN DISTINCTIVENESS
    if fname is not None:
        from ibeis.algo.hots import distinctiveness_normalizer
        cachedir = ub.ensure_app_cache_dir('ibeis', 'distinctiveness_model')
        species = 'zebra_plains'
        dstcnvs_normer = distinctiveness_normalizer.DistinctivnessNormalizer(species, cachedir=cachedir)
        dstcnvs_normer.load(cachedir)
        weights = dstcnvs_normer.get_distinctiveness(vecs)
    else:
        kpts = np.vstack((kpts, [0, 0, 1, 1, 1, 0]))
        kpts = np.vstack((kpts, [0.01, 10, 1, 1, 1, 0]))
        kpts = np.vstack((kpts, [0.94, 11.5, 1, 1, 1, 0]))
        weights = np.ones(len(kpts))
    chipsize = tuple(vt.iceil(vt.get_kpts_image_extent(kpts)[2:4]).tolist())
    return kpts, chipsize, weights


def show_coverage_map(chip, mask, patch, kpts, fnum=None, ell_alpha=.6,
                      show_mask_kpts=False):
    """ testing function """
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    pnum_ = pt.get_pnum_func(nRows=2, nCols=2)
    if patch is not None:
        pt.imshow((patch * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(0), title='patch')
        pt.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(1), title='mask')
    else:
        pt.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=(2, 1, 1), title='mask')
    if show_mask_kpts:
        pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    pt.imshow(chip, fnum=fnum, pnum=pnum_(2), title='chip')
    pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    masked_chip = (chip * mask[:, :, None]).astype(np.uint8)
    pt.imshow(masked_chip, fnum=fnum, pnum=pnum_(3), title='masked chip')
    #pt.draw_kpts2(kpts)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.coverage_kpts
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
