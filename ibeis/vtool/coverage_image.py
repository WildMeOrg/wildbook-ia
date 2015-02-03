from __future__ import absolute_import, division, print_function
from six.moves import zip, range, map  # NOQA
import six
from six import next
import cv2
import numpy as np
import utool as ut
from vtool import patch as ptool
from vtool import keypoint as ktool
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[cov]', DEBUG=False)


def show_coverage_map(chip, mask, patch, kpts, fnum=None, ell_alpha=.6,
                      show_mask_kpts=False):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    pnum_ = pt.get_pnum_func(nRows=2, nCols=2)
    if patch is not None:
        pt.imshow((patch * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(0), title='patch')
    #ut.embed()
    pt.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(1), title='mask')
    if show_mask_kpts:
        pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    pt.imshow(chip, fnum=fnum, pnum=pnum_(2), title='chip')
    pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    masked_chip = (chip * mask[:, :, None]).astype(np.uint8)
    pt.imshow(masked_chip, fnum=fnum, pnum=pnum_(3), title='masked chip')
    #pt.draw_kpts2(kpts)


def iter_reduce_ufunc(ufunc, arr_iter, initial=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    """
    if initial is None:
        try:
            out = next(arr_iter).copy()
        except StopIteration:
            return None
    else:
        out = initial
    for arr in arr_iter:
        ufunc(out, arr, out=out)
    return out


def warped_patch_generator(patch, dsize, affmat_list, weight_list, **kwargs):
    """
    generator that warps the patches (like gaussian) onto an image with dsize using constant memory.

    output must be used or copied on every iteration otherwise the next output will clobber the previous

    References:
        http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
    """
    size_penalty_on = kwargs.get('size_penalty_on', True)
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
        if size_penalty_on:
            size_penalty_power = kwargs.get('size_penalty_power', .5)
            size_penalty_scale = kwargs.get('size_penalty_scale', .1)
            total_weight = (warped.sum() ** size_penalty_power) * size_penalty_scale
            if total_weight > 1:
                # Whatever the size of the keypoint is it should
                # contribute a total of 1 score
                np.divide(warped, total_weight, out=warped)
        yield warped


@profile
def warp_patch_onto_kpts(kpts, patch, chip_shape, fx2_score=None,
                         scale_factor=1.0, mode='max', **kwargs):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch (ndarray): patch to warp (like gaussian)
        chip_shape (tuple):
        fx2_score (ndarray): score for every keypoint
        scale_factor (float):

    Returns:
        ndarray: mask

    CommandLine:
        python -m vtool.coverage_image --test-warp_patch_onto_kpts
        python -m vtool.coverage_image --test-warp_patch_onto_kpts --show
        python -m vtool.coverage_image --test-warp_patch_onto_kpts --show --hole
        python -m vtool.coverage_image --test-warp_patch_onto_kpts --show --square
        python -m vtool.coverage_image --test-warp_patch_onto_kpts --show --square --hole

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import pyhesaff
        >>> img_fpath    = ut.grab_test_imgpath('carl.jpg')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> kpts = kpts[::15]
        >>> chip = vt.imread(img_fpath)
        >>> chip_shape = chip.shape
        >>> fx2_score = np.ones(len(kpts))
        >>> scale_factor = 1.0
        >>> srcshape = (19, 19)
        >>> radius = srcshape[0] / 2.0
        >>> sigma = 0.4 * radius
        >>> SQUARE = ut.get_argflag('--square')
        >>> HOLE = ut.get_argflag('--hole')
        >>> if SQUARE:
        >>>     patch = np.ones(srcshape)
        >>> else:
        >>>     patch = ptool.gaussian_patch(shape=srcshape, sigma=sigma) #, norm_01=False)
        >>>     patch = patch / patch.max()
        >>> if HOLE:
        >>>     patch[int(patch.shape[0] / 2), int(patch.shape[1] / 2)] = 0
        >>> # execute function
        >>> dstimg = warp_patch_onto_kpts(kpts, patch, chip_shape, fx2_score, scale_factor)
        >>> # verify results
        >>> print('dstimg stats %r' % (ut.get_stats_str(dstimg, axis=None)),)
        >>> print('patch stats %r' % (ut.get_stats_str(patch, axis=None)),)
        >>> #print(patch.sum())
        >>> assert np.all(ut.inbounds(dstimg, 0, 1, eq=True))
        >>> # show results
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     mask = dstimg
        >>>     show_coverage_map(chip, mask, patch, kpts)
        >>>     pt.show_if_requested()
    """
    #if len(kpts) == 0:
    #    return None
    chip_scale_h = int(np.ceil(chip_shape[0] * scale_factor))
    chip_scale_w = int(np.ceil(chip_shape[1] * scale_factor))
    if len(kpts) == 0:
        dstimg =  np.zeros((chip_scale_h, chip_scale_w))
        return dstimg
    if fx2_score is None:
        fx2_score = np.ones(len(kpts))
    dsize = (chip_scale_w, chip_scale_h)
    remove_affine_information = kwargs.get('remove_affine_information', False)
    constant_scaling = kwargs.get('constant_scaling', False)
    # Allocate destination image
    patch_shape = patch.shape
    # Scale keypoints into destination image
    if remove_affine_information:
        # disregard affine information in keypoints
        # i still dont understand why we are trying this
        (patch_h, patch_w) = patch_shape
        half_width  = (patch_w / 2.0)  # - .5
        half_height = (patch_h / 2.0)  # - .5
        import vtool as vt
        # Center src image
        T1 = vt.translation_mat3x3(-half_width + .5, -half_height + .5)
        # Scale src to the unit circle
        if not constant_scaling:
            S1 = vt.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
        # Transform the source image to the keypoint ellipse
        kpts_T = np.array([vt.translation_mat3x3(x, y) for (x, y) in vt.get_xys(kpts).T])
        if not constant_scaling:
            kpts_S = np.array([vt.scale_mat3x3(np.sqrt(scale)) for scale in vt.get_scales(kpts).T])
        # Adjust for the requested scale factor
        S2 = vt.scale_mat3x3(scale_factor, scale_factor)
        #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
        if not constant_scaling:
            M_list = reduce(vt.matrix_multiply, (S2, kpts_T, kpts_S, S1, T1))
        else:
            M_list = reduce(vt.matrix_multiply, (S2, kpts_T, T1))
    else:
        M_list = ktool.get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
    affmat_list = M_list[:, 0:2, :]
    weight_list = fx2_score
    # For each keypoint warp a gaussian scaled by the feature score into the image
    warped_patch_iter = warped_patch_generator(
        patch, dsize, affmat_list, weight_list, **kwargs)
    # Either max or sum
    if mode == 'max':
        dstimg = iter_reduce_ufunc(np.maximum, warped_patch_iter)
    elif mode == 'sum':
        dstimg = iter_reduce_ufunc(np.add, warped_patch_iter)
        # HACK FOR SUM: DO NOT DO THIS FOR MAX
        dstimg[dstimg > 1.0] = 1.0
    else:
        raise AssertionError('Unknown mode=%r' % (mode,))
    return dstimg


# TODO: decorator that lets utool know about functions that
# should be configured.
#def configurable_func(

def get_gaussian_weight_patch(gauss_shape=(19, 19), gauss_sigma_frac=.3, gauss_norm_01=True):
    r"""
    2d gaussian image useful for plotting

    Returns:
        ndarray: patch

    CommandLine:
        python -m vtool.coverage_image --test-get_gaussian_weight_patch

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> patch = get_gaussian_weight_patch()
        >>> # verify results
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


@profile
#@ut.memprof
def make_coverage_mask(kpts, chip_shape, fx2_score=None, mode=None,
                       return_patch=True, patch=None, resize=True, **kwargs):
    r"""
    Returns a intensity image denoting which pixels are covered by the input
    keypoints

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        chip_shape (tuple):

    Returns:
        tuple (ndarray, ndarray): dstimg, patch

    CommandLine:
        python -m vtool.coverage_image --test-make_coverage_mask --show
        python -m vtool.coverage_image --test-make_coverage_mask

        python -m vtool.patch --test-test_show_gaussian_patches2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import plottool as pt
        >>> import pyhesaff
        >>> #img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img_fpath = ut.grab_test_imgpath('lena.png')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> chip_shape = chip.shape
        >>> # execute function
        >>> dstimg, patch = make_coverage_mask(kpts, chip_shape)
        >>> # show results
        >>> if ut.show_was_requested():
        >>>     mask = dstimg
        >>>     show_coverage_map(chip, mask, patch, kpts)
        >>>     pt.show_if_requested()
    """
    if patch is None:
        gauss_shape = kwargs.get('gauss_shape', (19, 19))
        gauss_sigma_frac = kwargs.get('gauss_sigma_frac', .3)
        patch = get_gaussian_weight_patch(gauss_shape, gauss_sigma_frac)
    if mode is None:
        mode = 'max'
    cov_scale_factor = kwargs.get('cov_scale_factor', .25)
    cov_blur_ksize = kwargs.get('cov_blur_ksize', (17, 17))
    cov_blur_sigma = kwargs.get('cov_blur_sigma', 5.0)
    dstimg = warp_patch_onto_kpts(
        kpts, patch, chip_shape,
        mode=mode,
        fx2_score=fx2_score,
        scale_factor=cov_scale_factor,
        **kwargs
    )
    if kwargs.get('cov_blur_on', True):
        cv2.GaussianBlur(dstimg, ksize=cov_blur_ksize, sigmaX=cov_blur_sigma, sigmaY=cov_blur_sigma,
                         dst=dstimg, borderType=cv2.BORDER_CONSTANT)
    if resize:
        dsize = tuple(chip_shape[0:2][::-1])
        dstimg = cv2.resize(dstimg, dsize)
    #print(dstimg)
    if return_patch:
        return dstimg, patch
    else:
        return dstimg


def grid_coverage(kpts, chipsize, weights, grid_scale_factor=.3, grid_steps=1):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        chipsize (tuple):
        weights (ndarray):

    CommandLine:
        python -m vtool.coverage_image --test-grid_coverage --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> kpts, chipsize, weights = testdata_coveragegrid()
        >>> grid_scale_factor = .3
        >>> grid_steps = 1
        >>> coverage_gridtup = grid_coverage(kpts, chipsize, weights)
        >>> num_cols, num_rows, subbin_xy_arr, neighbor_bin_centers, neighbor_bin_weights = coverage_gridtup
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     visualize_coverage_grid(num_cols, num_rows, subbin_xy_arr,
        >>>                    neighbor_bin_centers, neighbor_bin_weights)
        >>>     pt.show_if_requested()
    """
    import vtool as vt

    def get_subbin_xy_neighbors(subbin_index00, grid_steps, num_cols, num_rows):
        """ Generate all neighbor of a bin
        subbin_index00 = left and up subbin index
        """
        subbin_index00 = np.floor(subbin_index00).astype(np.int32)
        subbin_x0, subbin_y0 = subbin_index00
        step_list = np.arange(1 - grid_steps, grid_steps + 1)
        offset_list = [
            # broadcast to the shape we will add too
            np.array([xoff, yoff])[:, None]
            for xoff, yoff in list(ut.iprod(step_list, step_list))]
        neighbor_subbin_index_list = [
            np.add(subbin_index00, offset)
            for offset in offset_list
        ]
        # Concatenate all subbin indexes into one array for faster vectorized ops
        neighbor_bin_indicies = np.dstack(neighbor_subbin_index_list).T

        # Clip with no wrapparound
        min_val = np.array([0, 0])
        max_val = np.array([num_cols - 1, num_rows - 1])

        np.clip(neighbor_bin_indicies,
                min_val[None, None, :],
                max_val[None, None, :],
                out=neighbor_bin_indicies)
        return neighbor_bin_indicies

    def compute_subbin_to_bins_dist(neighbor_bin_centers, subbin_xy_arr):
        _tmp = np.subtract(neighbor_bin_centers, subbin_xy_arr.T[None, :])
        neighbor_subbin_sqrddist_arr = np.power(_tmp, 2, out=_tmp).sum(axis=2)
        return neighbor_subbin_sqrddist_arr

    def weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights):
        _gaussweights = vt.gauss_func1d_unnormalized(neighbor_subbin_sqrddist_arr)
        # Each column sums to 1
        np.divide(_gaussweights, _gaussweights.sum(axis=0)[None, :], out=_gaussweights)
        # Scale initial weights by the gaussian falloff
        neighbor_bin_weights = np.multiply(_gaussweights, weights[None, :])
        return neighbor_bin_weights

    # Compute grid size and stride
    chip_w, chip_h = chipsize
    num_rows = vt.iround(grid_scale_factor * chip_h)
    num_cols = vt.iround(grid_scale_factor * chip_w)
    chipstride = np.array((chip_w / num_cols, chip_h / num_rows))
    # Find keypoint subbin locations relative to edges
    xy_arr = vt.get_xys(kpts)
    subbin_xy_arr = np.divide(xy_arr, chipstride[:, None])
    # Find subbin locations relative to centers
    frac_subbin_index = np.subtract(subbin_xy_arr, .5)
    neighbor_bin_xy_indicies = get_subbin_xy_neighbors(frac_subbin_index, grid_steps, num_cols, num_rows)
    # Find centers
    neighbor_bin_centers = np.add(neighbor_bin_xy_indicies, .5)
    # compute distance to neighbors
    neighbor_subbin_sqrddist_arr = compute_subbin_to_bins_dist(neighbor_bin_centers, subbin_xy_arr)
    # scale weights using guassia falloff
    neighbor_bin_weights = weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights)
    # convert to rowcol
    neighbor_bin_indicies = neighbor_bin_rc_indicies = neighbor_bin_xy_indicies[:, :, ::-1]  # NOQA

    coverage_gridtup = (
        num_rows, num_cols, subbin_xy_arr, neighbor_bin_centers,
        neighbor_bin_weights, neighbor_bin_indicies)
    return coverage_gridtup


def get_grid_coverage_mask(kpts, chipsize, weights, grid_scale_factor=.3,
                           grid_steps=1, resize=False, out=None):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        chipsize (tuple):  width, height
        weights (?):
        grid_scale_factor (float):
        grid_steps (int):

    Returns:
        ?: weightgrid

    CommandLine:
        python -m vtool.coverage_image --test-get_grid_coverage_mask

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts, chipsize, weights = testdata_coveragegrid('easy1.png')
        >>> grid_scale_factor = 0.3
        >>> grid_steps = 2
        >>> # execute function
        >>> weightgrid = get_grid_coverage_mask(kpts, chipsize, weights, grid_scale_factor, grid_steps)
        >>> # verify results
        >>> result = str(weightgrid)
        >>> print(result)
    """
    import vtool as vt
    coverage_gridtup = grid_coverage(
        kpts, chipsize, weights, grid_scale_factor=grid_scale_factor, grid_steps=grid_steps)
    gridshape = coverage_gridtup[0:2]
    neighbor_bin_weights, neighbor_bin_indicies = coverage_gridtup[-2:]
    #neighbor_bin_indicies.shape = (steps + 1 * 2, len(kpts), 2)
    oldshape_indicies = neighbor_bin_indicies.shape
    newshape_indicies = (np.prod(oldshape_indicies[0:2]), oldshape_indicies[2])
    neighbor_bin_indicies =  neighbor_bin_indicies.reshape(newshape_indicies).T
    neighbor_bin_weights = neighbor_bin_weights.flatten()
    # Get flat indexing into gridbins
    neighbor_bin_flat_indicies = np.ravel_multi_index(neighbor_bin_indicies, gridshape)
    # Group by bins with weight
    unique_flatxs, grouped_flatxs = vt.group_indicies(neighbor_bin_flat_indicies)
    grouped_weights = vt.apply_grouping(neighbor_bin_weights, grouped_flatxs)
    # FIXME: boundary cases are not handled right because their vote is split
    # into the same bin and is fighting with itself durring the max
    max_weights = list(map(np.max, grouped_weights))
    if out is None:
        weightgrid = np.zeros(gridshape)
    else:
        # outvar specified
        weightgrid = out
        weightgrid[:] = 0
    unique_rows, unique_cols = np.unravel_index(unique_flatxs, gridshape)
    weightgrid[unique_rows, unique_cols] = max_weights
    #flat_weightgrid = np.zeros(np.prod(gridshape))
    #flat_weightgrid[unique_flatxs] = max_weights
    #ut.embed()
    #weightgrid = np.reshape(flat_weightgrid, gridshape)
    if resize:
        weightgrid = cv2.resize(weightgrid, chipsize,
                                interpolation=cv2.INTER_NEAREST)
    return weightgrid


def testdata_coveragegrid(fname=None):
    import vtool as vt
    # build test data
    kpts = vt.dummy.get_testdata_kpts(fname)
    if fname is None:
        kpts = np.vstack((kpts, [0, 0, 1, 1, 1, 0]))
        kpts = np.vstack((kpts, [0.01, 10, 1, 1, 1, 0]))
        kpts = np.vstack((kpts, [0.94, 11.5, 1, 1, 1, 0]))
    chipsize = tuple(vt.iceil(vt.get_kpts_image_extent(kpts)).tolist())
    weights = np.ones(len(kpts))
    return kpts, chipsize, weights


def get_coverage_grid_gridsearch_configs():
    search_basis = {
        'grid_scale_factor': [.1, .25, .5, 1.0],
        'grid_steps': [1, 2, 3, 10],
    }
    param_slice_dict = {
        'grid_scale_factor' : slice(0, 4),
        'grid_steps'        : slice(0, 4),
    }
    varied_dict = {
        key: val[param_slice_dict.get(key, slice(0, 1))]
        for key, val in six.iteritems(search_basis)
    }
    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict)
    return cfgdict_list, cfglbl_list


def gridsearch_coverage_grid_mask():
    """
    CommandLine:
        python -m vtool.coverage_image --test-gridsearch_coverage_grid_mask --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_coverage_grid_mask()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    cfgdict_list, cfglbl_list = get_coverage_grid_gridsearch_configs()
    kpts, chipsize, weights = testdata_coveragegrid('easy1.png')
    gridmask_list = [
        get_grid_coverage_mask(kpts, chipsize, weights, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='coverage grid')
    ]
    gridmask_list = [
        255 * (gridmask / gridmask.max()) for gridmask in gridmask_list
    ]

    fnum = 1
    ut.interact_gridsearch_result_images(
        pt.imshow, cfgdict_list, cfglbl_list,
        gridmask_list, fnum=fnum, figtitle='coverage grid', unpack=False,
        max_plots=25)

    pt.iup()
    #pt.show_if_requested()


def gridsearch_coverage_grid():
    """
    CommandLine:
        python -m vtool.coverage_image --test-gridsearch_coverage_grid --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_coverage_grid()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    kpts, chipsize, weights = testdata_coveragegrid()
    cfgdict_list, cfglbl_list = get_coverage_grid_gridsearch_configs()
    coverage_gridtup_list = [
        grid_coverage(kpts, chipsize, weights, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='coverage grid')
    ]

    fnum = 1
    ut.interact_gridsearch_result_images(
        visualize_coverage_grid, cfgdict_list, cfglbl_list,
        coverage_gridtup_list, fnum=fnum, figtitle='coverage grid', unpack=True,
        max_plots=25)

    pt.iup()


def visualize_coverage_grid(num_rows, num_cols,
                            subbin_xy_arr,
                            neighbor_bin_centers,
                            neighbor_bin_weights,
                            neighbor_bin_indicies,
                            fnum=None, pnum=None):
    import plottool as pt
    import vtool as vt

    if fnum is None:
        fnum = pt.next_fnum()
    fig = pt.figure(fnum, pnum=pnum)
    ax = fig.gca()
    x_edge_indices = np.arange(num_cols)
    y_edge_indices = np.arange(num_rows)
    x_center_indices = vt.hist_edges_to_centers(x_edge_indices)
    y_center_indices = vt.hist_edges_to_centers(y_edge_indices)
    x_center_grid, y_center_grid = np.meshgrid(x_center_indices, y_center_indices)
    ax.set_xticks(x_edge_indices)
    ax.set_yticks(y_edge_indices)
    # Plot keypoint locs
    ax.scatter(subbin_xy_arr[0], subbin_xy_arr[1], marker='o')
    # Plot Weighted Lines to Subbins
    pt_colors = pt.distinct_colors(len(subbin_xy_arr.T))
    for subbin_centers, subbin_weights in zip(neighbor_bin_centers,
                                              neighbor_bin_weights):
        for pt_xys, center_xys, weight, color in zip(subbin_xy_arr.T, subbin_centers,
                                                     subbin_weights, pt_colors):
            # Adjsut weight to alpha for easier visualization
            alpha = weight
            #alpha **= .5
            #min_viz_alpha = .1
            #alpha = alpha * (1 - min_viz_alpha) + min_viz_alpha
            ax.plot(*np.vstack((pt_xys, center_xys)).T, color=color, alpha=alpha, lw=3)
    # Plot Grid Centers
    num_cells = num_cols * num_rows
    grid_alpha = min(.4, max(1 - (num_cells / 500), .1))
    grid_color = [.6, .6, .6, grid_alpha]
    #print(grid_color)
    # Plot grid cetners
    ax.scatter(x_center_grid, y_center_grid, marker='.', color=grid_color,
               s=grid_alpha)

    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows - 1)
    #-----
    pt.dark_background()
    ax.grid(True, color=[.3, .3, .3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.coverage_image
        python -m vtool.coverage_image --allexamples
        python -m vtool.coverage_image --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
