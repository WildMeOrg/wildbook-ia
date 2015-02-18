from __future__ import absolute_import, division, print_function
from six.moves import zip, range, map  # NOQA
#import six
import numpy as np
import utool as ut
import cv2
from vtool import coverage_kpts
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[covgrid]', DEBUG=False)


# TODO: integrate more
COVGRID_DEFAULT = ut.ParamInfoList('coverage_grid', [
    ut.ParamInfo('pxl_per_bin', 10, 'ppb', varyvals=[20, 5, 1]),
    ut.ParamInfo('grid_steps', 3, 'stps', varyvals=[1, 3, 7]),
    ut.ParamInfo('grid_sigma', 1.6, 'sigma', varyvals=[1.0, 1.6]),
])


def make_grid_coverage_mask(kpts, chipsize, weights, pxl_per_bin=4,
                            grid_steps=1, resize=False, out=None, grid_sigma=1.6):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoint
        chipsize (tuple):  width, height
        weights (ndarray[float32_t, ndim=1]):
        pxl_per_bin (float):
        grid_steps (int):

    Returns:
        ndarray: weightgrid

    CommandLine:
        python -m vtool.coverage_grid --test-make_grid_coverage_mask

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_grid import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts, chipsize, weights = coverage_kpts.testdata_coverage('easy1.png')
        >>> pxl_per_bin = 4
        >>> grid_steps = 2
        >>> # execute function
        >>> weightgrid = make_grid_coverage_mask(kpts, chipsize, weights, pxl_per_bin, grid_steps)
        >>> # verify result
        >>> result = str(weightgrid)
        >>> print(result)
    """
    import vtool as vt
    coverage_gridtup = sparse_grid_coverage(
        kpts, chipsize, weights,
        pxl_per_bin=pxl_per_bin,
        grid_steps=grid_steps,
        grid_sigma=grid_sigma
    )
    gridshape = coverage_gridtup[0:2]
    neighbor_bin_weights, neighbor_bin_indicies = coverage_gridtup[-2:]
    oldshape_indicies = neighbor_bin_indicies.shape
    newshape_indicies = (np.prod(oldshape_indicies[0:2]), oldshape_indicies[2])
    neighbor_bin_indicies =  neighbor_bin_indicies.reshape(newshape_indicies).T
    neighbor_bin_weights = neighbor_bin_weights.flatten()
    # Get flat indexing into gridbin
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
    #flat_weightgrid[unique_flatxs] = max_weight
    #ut.embed()
    #weightgrid = np.reshape(flat_weightgrid, gridshape)
    if resize:
        weightgrid = cv2.resize(weightgrid, chipsize,
                                interpolation=cv2.INTER_NEAREST)
    return weightgrid


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
    # Concatenate all subbin indexes into one array for faster vectorized op
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


def weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights, grid_sigma):
    import vtool as vt
    _gaussweights = vt.gauss_func1d_unnormalized(neighbor_subbin_sqrddist_arr, grid_sigma)
    # If uncommented next line ensure each column sums to 1
    #np.divide(_gaussweights, _gaussweights.sum(axis=0)[None, :], out=_gaussweights)
    # Scale initial weights by the gaussian falloff
    neighbor_bin_weights = np.multiply(_gaussweights, weights[None, :])
    return neighbor_bin_weights


def sparse_grid_coverage(kpts, chipsize, weights, pxl_per_bin=.3, grid_steps=1, grid_sigma=1.6):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoint
        chipsize (tuple):
        weights (ndarray):

    CommandLine:
        python -m vtool.coverage_grid --test-sparse_grid_coverage --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_grid import *  # NOQA
        >>> kpts, chipsize, weights = coverage_kpts.testdata_coverage()
        >>> chipsize = (chipsize[0] + 50, chipsize[1])
        >>> pxl_per_bin = 3
        >>> grid_steps = 2
        >>> grid_sigma = 1.6
        >>> coverage_gridtup = sparse_grid_coverage(kpts, chipsize, weights, pxl_per_bin, grid_steps, grid_sigma)
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     show_coverage_grid(*coverage_gridtup)
        >>>     pt.show_if_requested()
    """
    import vtool as vt
    # Compute grid size and stride
    chip_w, chip_h = chipsize
    # find enough rows to fit pxl_per_bin pixels into a grid dimension
    num_rows = max(vt.iround(chip_h / pxl_per_bin), 1)
    num_cols = max(vt.iround(chip_w / pxl_per_bin), 1)
    # stride is roughly equal in each direction, depending on rounding errors
    chipstride = np.array((chip_w / num_cols, chip_h / num_rows))
    # Find keypoint subbin locations relative to edge
    xy_arr = vt.get_xys(kpts)
    subbin_xy_arr = np.divide(xy_arr, chipstride[:, None])
    # Find subbin locations relative to center
    frac_subbin_index = np.subtract(subbin_xy_arr, .5)
    neighbor_bin_xy_indicies = get_subbin_xy_neighbors(frac_subbin_index, grid_steps, num_cols, num_rows)
    # Find center
    neighbor_bin_centers = np.add(neighbor_bin_xy_indicies, .5)
    # compute distance to neighbor
    neighbor_subbin_sqrddist_arr = compute_subbin_to_bins_dist(neighbor_bin_centers, subbin_xy_arr)
    # scale weights using guassia falloff
    neighbor_bin_weights = weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights, grid_sigma)
    # convert to rowcol
    neighbor_bin_indicies = neighbor_bin_rc_indicies = neighbor_bin_xy_indicies[:, :, ::-1]  # NOQA

    coverage_gridtup = (
        num_rows, num_cols, subbin_xy_arr, neighbor_bin_centers,
        neighbor_bin_weights, neighbor_bin_indicies)
    return coverage_gridtup


# VISUALIZATION FUNCS


def show_coverage_grid(num_rows, num_cols, subbin_xy_arr,
                       neighbor_bin_centers, neighbor_bin_weights,
                       neighbor_bin_indicies, fnum=None, pnum=None):
    """
    visualizes the voting scheme on the grid. (not a mask, and no max)
    """
    import plottool as pt
    import vtool as vt
    import matplotlib as mpl

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
    # Plot keypoint loc
    ax.scatter(subbin_xy_arr[0], subbin_xy_arr[1], marker='o')
    # Plot Weighted Lines to Subbin
    pt_colors = pt.distinct_colors(len(subbin_xy_arr.T))
    segment_list = []
    color_list = []
    for subbin_centers, subbin_weights in zip(neighbor_bin_centers,
                                              neighbor_bin_weights):
        for pt_xys, center_xys, weight, color in zip(subbin_xy_arr.T, subbin_centers,
                                                     subbin_weights, pt_colors):
            # Adjsut weight to alpha for easier visualization
            alpha = weight
            INCRESE_ALPHA_VISIBILITY = True
            if INCRESE_ALPHA_VISIBILITY:
                min_viz_alpha = .05
                alpha = alpha * (1.0 - min_viz_alpha) + min_viz_alpha
                alpha **= 1.0
            #pt.plots.colorline(
            segment = np.vstack((pt_xys, center_xys))
            segment_list.append(segment)
            # Alpha becomes part of the colors
            color_list.append(list(color) + [alpha])
            # DO NOT USE PLOT VERY SLOW
            #ax.plot(*segment.T, color=color, alpha=alpha, lw=3)
    ax = pt.gca()
    # Plot all segments in single  line collection for speed
    # solid | dashed | dashdot | dotted
    lc = mpl.collections.LineCollection(segment_list, colors=color_list, linewidth=3, linestyles='solid')
    ax.add_collection(lc)
    # Plot Grid Center
    num_cells = num_cols * num_rows
    grid_alpha = min(.4, max(1 - (num_cells / 500), .1))
    grid_color = [.6, .6, .6, grid_alpha]
    #print(grid_color)
    # Plot grid cetner
    ax.scatter(x_center_grid, y_center_grid, marker='.', color=grid_color,
               s=grid_alpha)

    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows - 1)
    #-----
    pt.dark_background()
    ax.grid(True, color=[.3, .3, .3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


# TESTING FUNCS


def get_coverage_grid_gridsearch_configs():
    #varied_dict = {
    #    'pxl_per_bin': [.05, .3, 1.0],
    #    'grid_steps': [1, 3, 7],
    #    'grid_sigma': [1.0, 1.6],
    #}
    #slice_dict = {
    #    'pxl_per_bin' : slice(0, 3),
    #    'grid_steps'        : slice(0, 3),
    #    'grid_sigma'        : slice(0, 3),
    #}
    cfgdict_list, cfglbl_list = COVGRID_DEFAULT.get_gridsearch_input()
    # Make configuration for every parameter setting
    #cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, slice_dict=slice_dict)
    return cfgdict_list, cfglbl_list


def gridsearch_coverage_grid():
    """
    CommandLine:
        python -m vtool.coverage_grid --test-gridsearch_coverage_grid --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_grid import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_coverage_grid()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fname = None  # 'easy1.png'
    kpts, chipsize, weights = coverage_kpts.testdata_coverage(fname)
    if len(kpts) > 100:
        kpts = kpts[::100]
        weights = weights[::100]
    cfgdict_list, cfglbl_list = get_coverage_grid_gridsearch_configs()
    coverage_gridtup_list = [
        sparse_grid_coverage(kpts, chipsize, weights, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='coverage grid')
    ]

    fnum = 1
    with ut.Timer('plotting gridsearch'):
        ut.interact_gridsearch_result_images(
            show_coverage_grid, cfgdict_list, cfglbl_list,
            coverage_gridtup_list, fnum=fnum, figtitle='coverage grid', unpack=True,
            max_plots=25)

    pt.iup()


def gridsearch_coverage_grid_mask():
    """
    CommandLine:
        python -m vtool.coverage_grid --test-gridsearch_coverage_grid_mask --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_grid import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_coverage_grid_mask()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    cfgdict_list, cfglbl_list = get_coverage_grid_gridsearch_configs()
    kpts, chipsize, weights = coverage_kpts.testdata_coverage('easy1.png')
    gridmask_list = [
        255 *  make_grid_coverage_mask(kpts, chipsize, weights, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='coverage grid')
    ]
    NORMHACK = False
    if NORMHACK:
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.coverage_grid
        python -m vtool.coverage_grid --allexamples
        python -m vtool.coverage_grid --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
