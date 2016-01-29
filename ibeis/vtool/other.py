# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import six
import functools  # NOQA
from six import next
from six.moves import zip, range  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[other]')


def multiaxis_reduce(ufunc, arr, startaxis=0):
    """
    used to get max/min over all axes after <startaxis>

    CommandLine:
        python -m vtool.other --test-multiaxis_reduce

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> arr = (rng.rand(4, 3, 2, 1) * 255).astype(np.uint8)
        >>> ufunc = np.amax
        >>> startaxis = 1
        >>> out_ = multiaxis_reduce(ufunc, arr, startaxis)
        >>> result = out_
        >>> print(result)
        [182 245 236 249]
    """
    num_iters = len(arr.shape) - startaxis
    out_ = ufunc(arr, axis=startaxis)
    for _ in range(num_iters - 1):
        out_ = ufunc(out_, axis=1)
    return out_


def safe_vstack(tup, default_shape=(0,), default_dtype=np.float):
    """ stacks a tuple even if it is empty """
    try:
        return np.vstack(tup)
    except ValueError:
        return np.empty(default_shape, dtype=default_dtype)


def safe_cat(tup, axis=0, default_shape=(0,), default_dtype=np.float):
    """ stacks a tuple even if it is empty """
    try:
        return np.concatenate(tup, axis=axis)
    except ValueError:
        return np.empty(default_shape, dtype=default_dtype)


def median_abs_dev(arr_list, **kwargs):
    """
    References:
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(arr_list - np.median(arr_list, **kwargs)), **kwargs)


def argsort_groups(scores_list, reverse=False, rng=np.random, randomize_levels=True):
    """
    Sorts each group normally, but randomizes order of level values.

    TODO: move to vtool

    Args:
        scores_list (list):
        reverse (bool): (default = True)
        rng (module):  random number generator(default = numpy.random)

    CommandLine:
        python -m ibeis.init.filter_annots --exec-argsort_groups

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> scores_list = [
        >>>     np.array([np.nan, np.nan], dtype=np.float32),
        >>>     np.array([np.nan, 2], dtype=np.float32),
        >>>     np.array([4, 1, 1], dtype=np.float32),
        >>>     np.array([7, 3, 3, 0, 9, 7, 5, 8], dtype=np.float32),
        >>>     np.array([2, 4], dtype=np.float32),
        >>>     np.array([np.nan, 4, np.nan, 8, np.nan, 9], dtype=np.float32),
        >>> ]
        >>> reverse = True
        >>> rng = np.random.RandomState(0)
        >>> idxs_list = argsort_groups(scores_list, reverse, rng)
        >>> #import vtool as vt
        >>> #sorted_scores = vt.ziptake(scores_list, idxs_list)
        >>> #result = 'sorted_scores = %s' % (ut.list_str(sorted_scores),)
        >>> result = 'idxs_list = %s' % (ut.list_str(idxs_list, with_dtype=False),)
        >>> print(result)
        idxs_list = [
            np.array([1, 0]),
            np.array([1, 0]),
            np.array([0, 1, 2]),
            np.array([4, 7, 0, 5, 6, 1, 2, 3]),
            np.array([1, 0]),
            np.array([5, 3, 1, 2, 0, 4]),
        ]

    """
    scores_list_ = [np.array(scores, copy=True).astype(np.float) for scores in scores_list]
    breakers_list = [rng.rand(len(scores)) for scores in scores_list_]
    # replace nan with -inf, or inf randomize order between equal values
    replval = -np.inf if reverse else np.inf
    # Ensure that nans are ordered last
    for scores in scores_list_:
        scores[np.isnan(scores)] = replval
    # The last column is sorted by first with lexsort
    scorebreaker_list = [np.array((breakers, scores))
                         for scores, breakers in zip(scores_list_, breakers_list)]
    if reverse:
        idxs_list = [np.lexsort(scorebreaker)[::-1] for scorebreaker in  scorebreaker_list]
    else:
        idxs_list = [np.lexsort(scorebreaker) for scorebreaker in  scorebreaker_list]
    return idxs_list


def check_sift_validity(sift_uint8, lbl=None, verbose=ut.NOT_QUIET):
    """
    checks if a SIFT descriptor is valid
    """
    if lbl is None:
        lbl = ut.get_varname_from_stack(sift_uint8, N=1)
    print('[checksift] Checking valididty of %d SIFT descriptors. lbl=%s' % (
        sift_uint8.shape[0], lbl))
    is_correct_shape = len(sift_uint8.shape) == 2 and sift_uint8.shape[1] == 128
    is_correct_dtype = sift_uint8.dtype == np.uint8
    if not is_correct_shape:
        print('[checksift]  * incorrect shape = %r' % (sift_uint8.shape,))
    elif verbose:
        print('[checksift]  * correct shape = %r' % (sift_uint8.shape,))

    if not is_correct_dtype:
        print('[checksift]  * incorrect dtype = %r' % (sift_uint8.dtype,))
    elif verbose:
        print('[checksift]  * correct dtype = %r' % (sift_uint8.dtype,))

    num_sifts = sift_uint8.shape[0]
    sift_float01 = sift_uint8 / 512.0

    # Check L2 norm
    sift_norm = np.linalg.norm(sift_float01, axis=1)
    is_normal = np.isclose(sift_norm, 1.0, atol=.04)
    bad_locs_norm = np.where(np.logical_not(is_normal))[0]
    if len(bad_locs_norm) > 0:
        print('[checksift]  * bad norm   = %4d/%d' % (len(bad_locs_norm), num_sifts))
    else:
        print('[checksift]  * correctly normalized')

    # Check less than thresh=.2
    # This check actually is not valid because the SIFT descriptors is
    # normalized after it is thresholded
    #bad_locs_thresh = np.where((sift_float01 > .2).sum(axis=1))[0]
    #print('[checksift]  * bad thresh = %4d/%d' % (len(bad_locs_thresh), num_sifts))
    #if len(bad_locs_thresh) > 0:
    #    above_thresh = sift_float01[(sift_float01 > .2)]
    #    print('[checksift]  * components under thresh = %d' % (sift_float01 <= 2).sum())
    #    print('[checksift]  * components above thresh stats = ' +
    #    ut.get_stats_str(above_thresh, precision=2))

    isok = len(bad_locs_norm) == 0 and is_correct_shape and is_correct_dtype
    if not isok:
        print('[checksift] ERROR. SIFT CHECK FAILED')
    return isok


def pdist_argsort(x):
    """
    Sorts 2d indicies by their distnace matrix output from scipy.spatial.distance

    x = np.array([  3.05555556e-03,   1.47619797e+04,   1.47619828e+04])

    Args:
        x (ndarray):

    Returns:
        ndarray: sortx_2d

    CommandLine:
        python -m vtool.other --test-pdist_argsort

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> x = np.array([ 21695.78, 10943.76, 10941.44, 25867.64, 10752.03, 10754.35, 4171.86, 2.32, 14923.89, 14926.2 ], dtype=np.float64)
        >>> #[ 21025.45583333,    670.60055556,  54936.59111111,  21696.05638889, 33911.13527778,  55607.19166667]
        >>> sortx_2d = pdist_argsort(x)
        >>> result = ('sortx_2d = %s' % (str(sortx_2d),))
        >>> print(result)
        sortx_2d = [(2, 3), (1, 4), (1, 2), (1, 3), (0, 3), (0, 2), (2, 4), (3, 4), (0, 1), (0, 4)]
    """
    OLD = True
    #compare_idxs = [(r, c) for r, c in itertools.product(range(len(x) / 2),
    #range(len(x) / 2)) if (c > r)]
    if OLD:
        import scipy.spatial.distance as spdist
        mat = spdist.squareform(x)
        matu = np.triu(mat)
        sortx_row, sortx_col = np.unravel_index(matu.ravel().argsort(), matu.shape)
        # only take where col is larger than row due to upper triu
        sortx_2d = [(r, c) for r, c in zip(sortx_row, sortx_col) if (c > r)]
    else:
        num_rows = len(x) // 2
        compare_idxs = ut.flatten([[(r, c)  for c in range(r + 1, num_rows)]
                                   for r in range(num_rows)])
        sortx = x.argsort()
        sortx_2d = ut.take(compare_idxs, sortx)
    return sortx_2d


def get_consec_endpoint(consec_index_list, endpoint):
    """
    consec_index_list = consec_cols_list
    endpoint = 0
    """
    for consec_index in consec_index_list:
        if np.any(np.array(consec_index) == endpoint):
            return consec_index


def index_to_boolmask(index_list, maxval=None, hack=False):
    r"""
    Args:
        index_list (ndarray):
        maxval (None): (default = None)

    Kwargs:
        maxval

    Returns:
        ndarray: mask

    CommandLine:
        python -m vtool.other --exec-index_to_boolmask

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> import vtool as vt
        >>> index_list = np.array([(0, 0), (1, 1), (2, 1)])
        >>> maxval = (3, 3)
        >>> mask = vt.index_to_boolmask(index_list, maxval, hack=True)
        >>> result = ('mask =\n%s' % (str(mask.astype(np.uint8)),))
        >>> print(result)
        [[1 0 0]
         [0 1 0]
         [0 1 0]]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> import vtool as vt
        >>> index_list = np.array([0, 1, 4])
        >>> maxval = 5
        >>> mask = vt.index_to_boolmask(index_list, maxval, hack=False)
        >>> result = ('mask = %s' % (str(mask.astype(np.uint8)),))
        >>> print(result)
        mask = [1 1 0 0 1]

    """
    #assert index_list.min() >= 0
    if maxval is None:
        maxval = index_list.max()
    mask = np.zeros(maxval, dtype=np.bool)
    if hack:
        mask.__setitem__(tuple(index_list.T), True)
        #mask.__getitem__(tuple(index_list.T))
    else:
        mask[index_list] = True
    return mask


def get_crop_slices(isfill):
    fill_colxs = [np.where(row)[0] for row in isfill]
    fill_rowxs = [np.where(col)[0] for col in isfill.T]
    nRows, nCols = isfill.shape[0:2]

    filled_columns = intersect1d_reduce(fill_colxs)
    filled_rows = intersect1d_reduce(fill_rowxs)
    consec_rows_list = ut.group_consecutives(filled_rows)
    consec_cols_list = ut.group_consecutives(filled_columns)

    def get_min_consec_endpoint(consec_rows_list, endpoint):
        consec_index = get_consec_endpoint(consec_rows_list, endpoint)
        if consec_index is None:
            return endpoint
        return max(consec_index)

    def get_max_consec_endpoint(consec_rows_list, endpoint):
        consec_index = get_consec_endpoint(consec_rows_list, endpoint)
        if consec_index is None:
            return endpoint + 1
        return min(consec_index)

    consec_rows_top    = get_min_consec_endpoint(consec_rows_list, 0)
    consec_rows_bottom = get_max_consec_endpoint(consec_rows_list, nRows - 1)
    remove_cols_left   = get_min_consec_endpoint(consec_cols_list, 0)
    remove_cols_right  = get_max_consec_endpoint(consec_cols_list, nCols - 1)
    rowslice = slice(consec_rows_top, consec_rows_bottom)
    colslice = slice(remove_cols_left, remove_cols_right)
    return rowslice, colslice


def get_undirected_edge_ids(directed_edges):
    r"""
    Args:
        directed_edges (ndarray[ndims=2]):

    Returns:
        list: edgeid_list

    CommandLine:
        python -m vtool.other --exec-get_undirected_edge_ids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> directed_edges = np.array([[1, 2], [2, 1], [2, 3], [3, 1], [1, 1], [2, 3], [3, 2]])
        >>> edgeid_list = get_undirected_edge_ids(directed_edges)
        >>> result = ('edgeid_list = %s' % (str(edgeid_list),))
        >>> print(result)
        edgeid_list = [0 0 1 2 3 1 1]
    """
    import vtool as vt
    assert len(directed_edges.shape) == 2 and directed_edges.shape[1] == 2
    #flipped = qaid_arr < daid_arr
    flipped = directed_edges.T[0] < directed_edges.T[1]
    # standardize edge order
    edges_dupl = directed_edges.copy()
    edges_dupl[flipped, 0:2] = edges_dupl[flipped, 0:2][:, ::-1]
    edgeid_list = vt.compute_unique_data_ids(edges_dupl)
    return edgeid_list


def find_best_undirected_edge_indexes(directed_edges, score_arr=None):
    r"""
    Args:
        directed_edges (ndarray[ndims=2]):
        score_arr (ndarray):

    Returns:
        list: unique_edge_xs

    CommandLine:
        python -m vtool.other --test-find_best_undirected_edge_indexes

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> directed_edges = np.array([[1, 2], [2, 1], [2, 3], [3, 1], [1, 1], [2, 3], [3, 2]])
        >>> score_arr = np.array([1, 1, 1, 1, 1, 1, 2])
        >>> unique_edge_xs = find_best_undirected_edge_indexes(directed_edges, score_arr)
        >>> result = str(unique_edge_xs)
        >>> print(result)
        [0 3 4 6]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> directed_edges = np.array([[1, 2], [2, 1], [2, 3], [3, 1], [1, 1], [2, 3], [3, 2]])
        >>> score_arr = None
        >>> unique_edge_xs = find_best_undirected_edge_indexes(directed_edges, score_arr)
        >>> result = str(unique_edge_xs)
        >>> print(result)
        [0 2 3 4]
    """
    import vtool as vt
    #assert len(directed_edges.shape) == 2 and directed_edges.shape[1] == 2
    ##flipped = qaid_arr < daid_arr
    #flipped = directed_edges.T[0] < directed_edges.T[1]
    ## standardize edge order
    #edges_dupl = directed_edges.copy()
    #edges_dupl[flipped, 0:2] = edges_dupl[flipped, 0:2][:, ::-1]
    #edgeid_list = vt.compute_unique_data_ids(edges_dupl)
    edgeid_list = get_undirected_edge_ids(directed_edges)
    unique_edgeids, groupxs = vt.group_indices(edgeid_list)
    # if there is more than one edge in a group take the one with the highest score
    if score_arr is None:
        unique_edge_xs_list = [groupx[0] for groupx in groupxs]
    else:
        assert len(score_arr) == len(directed_edges)
        score_groups = vt.apply_grouping(score_arr, groupxs)
        score_argmaxs = [score_group.argmax() for score_group in score_groups]
        unique_edge_xs_list = [
            groupx[argmax] for groupx, argmax in zip(groupxs, score_argmaxs)
        ]
    unique_edge_xs = np.array(sorted(unique_edge_xs_list), dtype=np.int32)
    return unique_edge_xs


def argsort_multiarray(arrays, reverse=False):
    r"""
    Args:
        arrays (ndarray):
        reverse (bool): (default = False)

    Returns:
        ndarray: sortx

    CommandLine:
        python -m vtool.other --exec-argsort_multiarray

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arrays = np.array([[1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 5],
        >>>                    [2, 0, 2, 6, 4, 3, 2, 5, 6, 6, 6],
        >>>                    [1, 1, 0, 2, 3, 4, 5, 6, 7, 7, 7]],)
        >>> reverse = False
        >>> sortx = argsort_multiarray(arrays, reverse)
        >>> result = ('sortx = %s' % (str(sortx),))
        >>> print(result)
        sortx = [ 1  2  0  5  4  3  6  7  8  9 10]
    """
    sorting_records = np.rec.fromarrays(arrays)
    sort_stride = (-reverse * 2) + 1
    sortx = sorting_records.argsort()[::sort_stride]
    return sortx


def unique_rows(arr, directed=True):
    """
    Order or columns does not matter if directed = False
    """
    if directed:
        idx_list = compute_unique_data_ids(arr)
    else:
        idx_list = get_undirected_edge_ids(arr)
    _, unique_rowx = np.unique(idx_list, return_index=True)
    unique_arr = arr.take(unique_rowx, axis=0)
    return unique_arr


def compute_ndarray_unique_rowids_unsafe(arr):
    """
    arr = np.random.randint(2, size=(10000, 10))
    vt.compute_unique_data_ids_(list(map(tuple, arr)))
    len(vt.compute_unique_data_ids_(list(map(tuple, arr))))
    len(np.unique(vt.compute_unique_data_ids_(list(map(tuple, arr)))))

    %timeit vt.compute_unique_data_ids_(list(map(tuple, arr)))
    %timeit compute_ndarray_unique_rowids_unsafe(arr)

    """
    # no checks performed
    void_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    #assert arr.flags['C_CONTIGUOUS']
    arr_void_view = arr.view(void_dtype)
    unique, rowids = np.unique(arr_void_view, return_inverse=True)
    return rowids
    #np.ascontiguousarray(arr).data == arr.data
    #assert arr.data == arr_void_view.data


def unique_row_indexes(arr):
    """ np.unique on rows

    Args:
        arr (ndarray): 2d array

    Returns:
        ndarray: unique_rowx

    References:
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

    CommandLine:
        python -m vtool.other --test-unique_row_indexes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [.534, .432], [.534, .432], [1, 0], [0, 1]])
        >>> unique_rowx = unique_row_indexes(arr)
        >>> result = ('unique_rowx = %s' % (ut.numpy_str(unique_rowx),))
        >>> print(result)
        unique_rowx = np.array([0, 1, 2, 3, 5], dtype=np.int64)

    Ignore:
        %timeit unique_row_indexes(arr)
        %timeit compute_unique_data_ids(arr)
        %timeit compute_unique_integer_data_ids(arr)

    """
    void_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    arr_void_view = np.ascontiguousarray(arr).view(void_dtype)
    _, unique_rowx = np.unique(arr_void_view, return_index=True)
    # cast back to original dtype
    unique_rowx.sort()
    return unique_rowx


def nonunique_row_flags(arr):
    unique_rowx = unique_row_indexes(arr)
    unique_flags = index_to_boolmask(unique_rowx, len(arr))
    nonunique_flags = np.logical_not(unique_flags)
    return nonunique_flags


def nonunique_row_indexes(arr):
    """ rows that are not unique (does not include the first instance of each pattern)

    Args:
        arr (ndarray): 2d array

    Returns:
        ndarray: nonunique_rowx

    SeeAlso:
        unique_row_indexes
        nonunique_row_flags

    CommandLine:
        python -m vtool.other --test-unique_row_indexes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [.534, .432], [.534, .432], [1, 0], [0, 1]])
        >>> nonunique_rowx = unique_row_indexes(arr)
        >>> result = ('nonunique_rowx = %s' % (ut.numpy_str(nonunique_rowx),))
        >>> print(result)
        nonunique_rowx = np.array([4, 6, 7, 8], dtype=np.int64)
    """
    nonunique_flags = nonunique_row_flags(arr)
    nonunique_rowx = np.where(nonunique_flags)[0]
    return nonunique_rowx


def compute_unique_data_ids(data):
    """
    This is actually faster than compute_unique_integer_data_ids it seems

    CommandLine:
        python -m vtool.other --test-compute_unique_data_ids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [.534, .432], [.534, .432], [1, 0], [0, 1]])
        >>> dataid_list = compute_unique_data_ids(data)
        >>> result = 'dataid_list = ' + ut.numpy_str(dataid_list)
        >>> print(result)
        dataid_list = np.array([0, 1, 2, 3, 0, 4, 4, 2, 1], dtype=np.int32)
    """
    # construct a unique id for every edge
    hashable_rows = [tuple(row_.tolist()) for row_ in data]
    dataid_list = np.array(compute_unique_data_ids_(hashable_rows), dtype=np.int32)
    return dataid_list


def compute_unique_data_ids_(hashable_rows, iddict_=None):
    if iddict_ is None:
        iddict_ = {}
    for row in hashable_rows:
        if row not in iddict_:
            iddict_[row] = len(iddict_)
    dataid_list = ut.dict_take(iddict_, hashable_rows)
    return dataid_list


def compute_unique_arr_dataids(arr):
    """ specialized version for speed when arr is an ndarray """
    iddict_ = {}
    hashable_rows = list(map(tuple, arr.tolist()))
    for row in hashable_rows:
        if row not in iddict_:
            iddict_[row] = len(iddict_)
    dataid_list = np.array([iddict_[row] for row in hashable_rows])
    return dataid_list


def compute_unique_integer_data_ids(data):
    r"""
    This is actually slower than compute_unique_data_ids it seems

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> data = np.array([[0, 0], [0, 1], [1, 1], [0, 0], [0, 0], [0, 1], [1, 1], [0, 0], [9, 0]])
        >>> data = np.random.randint(1000, size=(1000, 2))
        >>> # execute function
        >>> result1 = compute_unique_data_ids(data)
        >>> result2 = compute_unique_integer_data_ids(data)
        >>> # verify results
        >>> print(result)

    %timeit compute_unique_data_ids(data)
    %timeit compute_unique_integer_data_ids(data)
    """
    # construct a unique id for every edge
    ncols = data.shape[1]
    # get the number of decimal places to shift
    exp_step = np.ceil(np.log10(data.max()))
    offsets = [int(10 ** (ix * exp_step)) for ix in reversed(range(0, ncols))]
    dataid_list = np.array([
        sum([
            item * offset
            for item, offset in zip(row, offsets)
        ])
        for row in data])
    return dataid_list


@profile
def trytake(list_, index_list):
    return None if list_ is None else list_take_(list_, index_list)


@profile
def list_take_(list_, index_list):
    if isinstance(list_, np.ndarray):
        return list_.take(index_list, axis=0)
    else:
        return ut.take(list_, index_list)


def compress2(arr, flag_list, axis=None, out=None):
    """
    Wrapper around numpy compress that makes the signature more similar to take
    """
    return np.compress(flag_list, arr, axis=axis, out=out)


def take2(arr, index_list, axis=None, out=None):
    """
    Wrapper around numpy compress that makes the signature more similar to take
    """
    return np.take(arr, index_list, axis=axis, out=out)


@profile
def list_compress_(list_, flag_list):
    if isinstance(list_, np.ndarray):
        return list_.compress(flag_list, axis=0)
    else:
        return ut.compress(list_, flag_list)


def index_partition(item_list, part1_items):
    """
    returns two lists. The first are the indecies of items in item_list that
    are in part1_items. the second is the indices in item_list that are not
    in part1_items. items in part1_items that are not in item_list are
    ignored

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> item_list = ['dist', 'fg', 'distinctiveness']
        >>> part1_items = ['fg', 'distinctiveness']
        >>> part1_indexes, part2_indexes = index_partition(item_list, part1_items)
        >>> ut.assert_eq(part1_indexes.tolist(), [1, 2])
        >>> ut.assert_eq(part2_indexes.tolist(), [0])
    """
    part1_indexes_ = [
        item_list.index(item)
        for item in part1_items
        if item in item_list
    ]
    part1_indexes = np.array(part1_indexes_)
    part2_indexes = np.setdiff1d(np.arange(len(item_list)), part1_indexes)
    # FIXME: use dtype np.int_
    part1_indexes = part1_indexes.astype(np.int32)
    part2_indexes = part2_indexes.astype(np.int32)
    return part1_indexes, part2_indexes


# def partition_Nones(item_list):
#     """
#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from vtool.other import *  # NOQA
#         >>> item_list = ['foo', None, None, 'bar']
#         >>> part1_indexes, part2_indexes = partition_Nones(item_list)
#     """
#     # part1_indexes_ = ut.list_where(item_list)
#     part1_indexes_ = [index for index, item in enumerate(item_list) if item is not None]
#     part1_indexes = np.array(part1_indexes_)
#     part2_indexes = np.setdiff1d(np.arange(len(item_list)), part1_indexes)
#     return part1_indexes, part2_indexes


def rebuild_partition(part1_vals, part2_vals, part1_indexes, part2_indexes):
    r"""
    Inverts work done by index_partition

    Args:
        part1_vals (list):
        part2_vals (list):
        part1_indexes (dict):
        part2_indexes (dict):

    CommandLine:
        python -m vtool.other --test-rebuild_partition

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> item_list = ['dist', 'fg', 'distinctiveness']
        >>> part1_items = ['fg', 'distinctiveness']
        >>> part1_indexes, part2_indexes = index_partition(item_list, part1_items)
        >>> part1_vals = ut.take(item_list, part1_indexes)
        >>> part2_vals = ut.take(item_list, part2_indexes)
        >>> val_list = rebuild_partition(part1_vals, part2_vals, part1_indexes, part2_indexes)
        >>> assert val_list == item_list, 'incorrect inversin'
        >>> print(val_list)
    """
    val_list = [None] * (len(part1_indexes) + len(part2_indexes))
    for idx, val in zip(part1_indexes, part1_vals):
        val_list[idx] = val
    for idx, val in zip(part2_indexes, part2_vals):
        val_list[idx] = val
    return val_list


def weighted_average_scoring(fsv, weight_filtxs, nonweight_filtxs):
    r"""
    does \frac{\sum_i w^f_i * w^d_i * r_i}{\sum_i w^f_i, w^d_i}
    to get a weighed average of ratio scores

    If we normalize the weight part to add to 1 then we can get per-feature
    scores.

    References:
        http://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> fsv = np.array([
        ...     [ 0.82992172,  1.56136119,  0.66465378],
        ...     [ 0.8000412 ,  2.14719748,  1.        ],
        ...     [ 0.80848503,  2.6816361 ,  1.        ],
        ...     [ 0.86761665,  2.70189977,  1.        ],
        ...     [ 0.8004055 ,  1.58753884,  0.92178345],])
        >>> weight_filtxs = np.array([1, 2], dtype=np.int32)
        >>> nonweight_filtxs = np.array([0], dtype=np.int32)
        >>> new_fs = weighted_average_scoring(fsv, weight_filtxs, nonweight_filtxs)
        >>> result = new_fs
        >>> print(result)
        [ 0.08585277  0.17123899  0.21611761  0.23367671  0.11675666]

    """
    weight_fs    = fsv.T.take(weight_filtxs, axis=0).T.prod(axis=1)
    nonweight_fs = fsv.T.take(nonweight_filtxs, axis=0).T.prod(axis=1)
    weight_fs_norm01 = weight_fs / weight_fs.sum()
    #weight_fs_norm01[np.isnan(weight_fs_norm01)] = 0.0
    # If weights are nan, fill them with zeros
    weight_fs_norm01 = np.nan_to_num(weight_fs_norm01)
    new_fs = np.multiply(nonweight_fs, weight_fs_norm01)
    return new_fs


def assert_zipcompress(arr_list, flags_list, axis=None):
    num_flags = [len(flags) for flags in flags_list]
    if axis is None:
        num_arrs = [arr.size for arr in arr_list]
    else:
        num_arrs = [arr.shape[axis] for arr in arr_list]
    assert num_flags == num_arrs, 'not able to zipcompress'


def zipcompress_safe(arr_list, flags_list, axis=None):
    arr_list = list(arr_list)
    flags_list = list(flags_list)
    assert_zipcompress(arr_list, flags_list, axis=axis)
    return zipcompress(arr_list, flags_list, axis)


def zipcompress(arr_list, flags_list, axis=None):
    return [np.compress(flags, arr, axis=axis) for arr, flags in zip(arr_list, flags_list)]


def ziptake(arr_list, indices_list, axis=None):
    return [arr.take(indices, axis=axis) for arr, indices in zip(arr_list, indices_list)]


def zipcat(arr1_list, arr2_list, axis=None):
    r"""
    Args:
        arr1_list (list):
        arr2_list (list):
        axis (None): (default = None)

    Returns:
        list:

    CommandLine:
        python -m vtool.other --exec-zipcat --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arr1_list = [np.array([0, 0, 0]), np.array([0, 0, 0, 0])]
        >>> arr2_list = [np.array([1, 1, 1]), np.array([1, 1, 1, 1])]
        >>> axis = None
        >>> arr3_list = zipcat(arr1_list, arr2_list, axis)
        >>> arr3_list0 = zipcat(arr1_list, arr2_list, axis=0)
        >>> arr3_list1 = zipcat(arr1_list, arr2_list, axis=1)
        >>> arr3_list2 = zipcat(arr1_list, arr2_list, axis=2)
        >>> print('arr3_list = %s' % (ut.repr3(arr3_list),))
        >>> print('arr3_list0 = %s' % (ut.repr3(arr3_list0),))
        >>> print('arr3_list2 = %s' % (ut.repr3(arr3_list2),))
    """
    assert len(arr1_list) == len(arr2_list), 'lists must correspond'
    if axis is None:
        arr1_iter = arr1_list
        arr2_iter = arr2_list
    else:
        arr1_iter = [atleast_nd(arr1, axis + 1) for arr1 in arr1_list]
        arr2_iter = [atleast_nd(arr2, axis + 1) for arr2 in arr2_list]
    arrs_iter = list(zip(arr1_iter, arr2_iter))
    arr3_list = [np.concatenate(arrs, axis=axis) for arrs in arrs_iter]
    return arr3_list


def atleast_nd(arr, n, tofront=False):
    r"""
    View inputs as arrays with at least n dimensions.
    TODO: Commit to numpy

    Args:
        arr (array_like): One array-like object.  Non-array inputs are
                converted to arrays.  Arrays that already have n or more dimensions
                are preserved.
        n (int):
        tofront (bool): if True new dimensions are added to the front of the array

    CommandLine:
        python -m vtool.other --exec-atleast_nd --show

    Returns:
        ndarray :
            An array with ``a.ndim >= n``.  Copies are avoided where possible,
            and views with three or more dimensions are returned.  For example,
            a 1-D array of shape ``(N,)`` becomes a view of shape
            ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a view of shape
            ``(M, N, 1)``.

    See Also:
        atleast_1d, atleast_2d, atleast_3d

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> n = 2
        >>> arr = np.array([1, 1, 1])
        >>> arr_ = atleast_nd(arr, n)
        >>> result = ut.repr2(arr_.tolist())
        >>> print(result)
        [[1], [1], [1]]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> n = 4
        >>> arr1 = [1, 1, 1]
        >>> arr2 = np.array(0)
        >>> arr3 = np.array([[[[[1]]]]])
        >>> arr1_ = atleast_nd(arr1, n)
        >>> arr2_ = atleast_nd(arr2, n)
        >>> arr3_ = atleast_nd(arr3, n)
        >>> result1 = ut.repr2(arr1_.tolist())
        >>> result2 = ut.repr2(arr2_.tolist())
        >>> result3 = ut.repr2(arr3_.tolist())
        >>> result = '\n'.join([result1, result2, result3])
        >>> print(result)
        [[[[1]]], [[[1]]], [[[1]]]]
        [[[[0]]]]
        [[[[[1]]]]]

    Ignore:
        # Hmm, mine is actually faster
        %timeit atleast_nd(arr, 3)
        %timeit np.atleast_3d(arr)
    """
    arr_ = np.asanyarray(arr)
    ndims = len(arr_.shape)
    if n is not None and ndims <  n:
        # append the required number of dimensions to the end
        if tofront:
            expander = (None,) * (n - ndims) + (Ellipsis,)
        else:
            expander = (Ellipsis,) + (None,) * (n - ndims)
        arr_ = arr_[expander]
    return arr_


def iter_reduce_ufunc(ufunc, arr_iter, out=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arr_list = [
        ...     np.array([0, 1, 2, 3, 8, 9]),
        ...     np.array([4, 1, 2, 3, 4, 5]),
        ...     np.array([0, 5, 2, 3, 4, 5]),
        ...     np.array([1, 1, 6, 3, 4, 5]),
        ...     np.array([0, 1, 2, 7, 4, 5])
        ... ]
        >>> memory = np.array([9, 9, 9, 9, 9, 9])
        >>> gen_memory = memory.copy()
        >>> def arr_gen(arr_list, gen_memory):
        ...     for arr in arr_list:
        ...         gen_memory[:] = arr
        ...         yield gen_memory
        >>> print('memory = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> ufunc = np.maximum
        >>> res1 = iter_reduce_ufunc(ufunc, iter(arr_list), out=None)
        >>> res2 = iter_reduce_ufunc(ufunc, iter(arr_list), out=memory)
        >>> res3 = iter_reduce_ufunc(ufunc, arr_gen(arr_list, gen_memory), out=memory)
        >>> print('res1       = %r' % (res1,))
        >>> print('res2       = %r' % (res2,))
        >>> print('res3       = %r' % (res3,))
        >>> print('memory     = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> assert np.all(res1 == res2)
        >>> assert np.all(res2 == res3)
    """
    # Get first item in iterator
    try:
        initial = next(arr_iter)
    except StopIteration:
        return None
    # Populate the outvariable if specified otherwise make a copy of the first
    # item to be the output memory
    if out is not None:
        out[:] = initial
    else:
        out = initial.copy()
    # Iterate and reduce
    for arr in arr_iter:
        ufunc(out, arr, out=out)
    return out


def clipnorm(arr, min_, max_, out=None):
    """
    normalizes arr to the range 0 to 1 using min_ and max_ as clipping bounds
    """
    if max_ == 1 and min_ == 0:
        if out is not None:
            out[:] = arr
        else:
            out = arr.copy()
        return out
    out_args = tuple() if out is None else (out,)
    arr_ = np.subtract(arr, min_, *out_args)
    arr_ = np.divide(arr_, max_ - min_, *out_args)
    arr_ = np.clip(arr_, 0.0, 1.0, *out_args)
    return arr_


def intersect1d_reduce(arr_list, assume_unique=False):
    arr_iter = iter(arr_list)
    out = six.next(arr_iter)
    for arr in arr_iter:
        out = np.intersect1d(out, arr, assume_unique=assume_unique)
    return out


def componentwise_dot(arr1, arr2):
    """
    a dot product is a componentwise multiplication of
    two vector and then a sum.

    Args:
        arr1 (ndarray)
        arr2 (ndarray):

    Returns:
        ndarray: cosangle

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> np.random.seed(0)
        >>> arr1 = np.random.rand(3, 128)
        >>> arr1 = arr1 / np.linalg.norm(arr1, axis=1)[:, None]
        >>> arr2 = arr1
        >>> cosangle = componentwise_dot(arr1, arr2)
        >>> result = str(cosangle)
        >>> print(result)
        [ 1.  1.  1.]
    """
    cosangle = np.multiply(arr1, arr2).sum(axis=-1).T
    return cosangle


def intersect2d_indices(A, B):
    r"""
    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):

    Returns:
        tuple: (ax_list, bx_list)

    CommandLine:
        python -m vtool.other --test-intersect2d_indices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> A = np.array([[ 158,  171], [ 542,  297], [ 955, 1113], [ 255, 1254], [ 976, 1255], [ 170, 1265]])
        >>> B = np.array([[ 117,  211], [ 158,  171], [ 255, 1254], [ 309,  328], [ 447, 1148], [ 750,  357], [ 976, 1255]])
        >>> # execute function
        >>> (ax_list, bx_list) = intersect2d_indices(A, B)
        >>> # verify results
        >>> result = str((ax_list, bx_list))
        >>> print(result)
        (array([0, 3, 4]), array([1, 2, 6]))
    """
    flag_list1, flag_list2 = intersect2d_flags(A, B)
    ax_list = np.flatnonzero(flag_list1)
    bx_list = np.flatnonzero(flag_list2)
    return ax_list, bx_list


def intersect2d_flags(A, B):
    r"""
    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):

    Returns:
        tuple: (flag_list1, flag_list2)

    CommandLine:
        python -m vtool.other --test-intersect2d_flags

    SeeAlso:
        np.in1d - the one dimensional version

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> A = np.array([[609, 307], [ 95, 344], [  1, 690]])
        >>> B = np.array([[ 422, 1148], [ 422,  968], [ 481, 1148], [ 750, 1132], [ 759,  159]])
        >>> # execute function
        >>> (flag_list1, flag_list2) = intersect2d_flags(A, B)
        >>> # verify results
        >>> result = str((flag_list1, flag_list2))
        >>> print(result)
        (array([False, False, False], dtype=bool), array([False, False, False, False, False], dtype=bool))
    """
    A_, B_, C_  = intersect2d_structured_numpy(A, B)
    flag_list1 = flag_intersection(A_, C_)
    flag_list2 = flag_intersection(B_, C_)
    return flag_list1, flag_list2


def flag_intersection(X_, C_):
    if X_.size == 0 or C_.size == 0:
        flags = np.full(X_.shape[0], False, dtype=np.bool)
        #return np.empty((0,), dtype=np.bool)
    else:
        flags = np.logical_or.reduce([X_ == c for c in C_]).T[0]
    return flags


def intersect2d_structured_numpy(A, B, assume_unique=False):
    """
    References:
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    nrows, ncols = A.shape
    assert A.dtype == B.dtype, ('A and B must have the same dtypes.'
                                'A.dtype=%r, B.dtype=%r' % (A.dtype, B.dtype))
    [('f%d' % i, A.dtype) for i in range(ncols)]
    #dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
    #dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
    #         'formats': ncols * [A.dtype]}
    dtype = {'names': ['f%d' % (i,) for i in range(ncols)],
             'formats': ncols * [A.dtype]}
    #try:
    A_ = np.ascontiguousarray(A).view(dtype)
    B_ = np.ascontiguousarray(B).view(dtype)
    C_ = np.intersect1d(A_, B_, assume_unique=assume_unique)
    #C = np.intersect1d(A.view(dtype),
    #                   B.view(dtype),
    #                   assume_unique=assume_unique)
    #except ValueError:
    #    C = np.intersect1d(A.copy().view(dtype),
    #                       B.copy().view(dtype),
    #                       assume_unique=assume_unique)
    return A_, B_, C_


def intersect2d_numpy(A, B, assume_unique=False, return_indices=False):
    """
    References::
        http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays/8317155#8317155

    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):
        assume_unique (bool):

    Returns:
        ndarray[ndims=2]: C

    CommandLine:
        python -m vtool.other --test-intersect2d_numpy

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> A = np.array([[  0,  78,  85, 283, 396, 400, 403, 412, 535, 552],
        ...               [152,  98,  32, 260, 387, 285,  22, 103,  55, 261]]).T
        >>> B = np.array([[403,  85, 412,  85, 815, 463, 613, 552],
        ...                [ 22,  32, 103, 116, 188, 199, 217, 254]]).T
        >>> assume_unique = False
        >>> # execute function
        >>> C, Ax, Bx = intersect2d_numpy(A, B, return_indices=True)
        >>> # verify results
        >>> result = str((C.T, Ax, Bx))
        >>> print(result)
        (array([[ 85, 403, 412],
               [ 32,  22, 103]]), array([2, 6, 7]), array([0, 1, 2]))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> A = np.array([[1, 2, 3], [1, 1, 1]])
        >>> B = np.array([[1, 2, 3], [1, 2, 14]])
        >>> C, Ax, Bx = intersect2d_numpy(A, B, return_indices=True)
        >>> result = str((C, Ax, Bx))
        >>> print(result)
        (array([[1, 2, 3]]), array([0]), array([0]))
    """
    nrows, ncols = A.shape
    A_, B_, C_ = intersect2d_structured_numpy(A, B, assume_unique)
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C_.view(A.dtype).reshape(-1, ncols)
    if return_indices:
        ax_list = np.flatnonzero(flag_intersection(A_, C_))
        bx_list = np.flatnonzero(flag_intersection(B_, C_))
        return C, ax_list, bx_list
    else:
        return C


@profile
def nearest_point(x, y, pts, mode='random'):
    """ finds the nearest point(s) in pts to (x, y) """
    dists = (pts.T[0] - x) ** 2 + (pts.T[1] - y) ** 2
    fx = dists.argmin()
    mindist = dists[fx]
    other_fx = np.where(mindist == dists)[0]
    if len(other_fx) > 0:
        if mode == 'random':
            np.random.shuffle(other_fx)
            fx = other_fx[0]
        if mode == 'all':
            fx = other_fx
        if mode == 'first':
            fx = fx
    return fx, mindist


def get_uncovered_mask(covered_array, covering_array):
    r"""
    Args:
        covered_array (ndarray):
        covering_array (ndarray):

    Returns:
        ndarray: flags

    CommandLine:
        python -m vtool.other --test-get_uncovered_mask

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> covered_array = [1, 2, 3, 4, 5]
        >>> covering_array = [2, 4, 5]
        >>> flags = get_uncovered_mask(covered_array, covering_array)
        >>> result = str(flags)
        >>> print(result)
        [ True False  True False False]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> covered_array = [1, 2, 3, 4, 5]
        >>> covering_array = []
        >>> flags = get_uncovered_mask(covered_array, covering_array)
        >>> result = str(flags)
        >>> print(result)
        [ True  True  True  True  True]

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> covered_array = np.array([
        ...  [1, 2, 3],
        ...  [4, 5, 6],
        ...  [7, 8, 9],
        ... ], dtype=np.int32)
        >>> covering_array = [2, 4, 5]
        >>> flags = get_uncovered_mask(covered_array, covering_array)
        >>> result = ut.numpy_str(flags)
        >>> print(result)
        np.array([[ True, False,  True],
                  [False, False,  True],
                  [ True,  True,  True]], dtype=bool)

    Ignore::
        covering_array = [1, 2, 3, 4, 5, 6, 7]
        %timeit get_uncovered_mask(covered_array, covering_array)
        100000 loops, best of 3: 18.6 µs per loop
        %timeit get_uncovered_mask2(covered_array, covering_array)
        100000 loops, best of 3: 16.9 µs per loop


    """
    if len(covering_array) == 0:
        return np.ones(np.shape(covered_array), dtype=np.bool)
    else:
        flags_iter = (np.not_equal(covered_array, item) for item in covering_array)
        mask_array = iter_reduce_ufunc(np.logical_and, flags_iter)
        return mask_array
    #if len(covering_array) == 0:
    #    return np.ones(np.shape(covered_array), dtype=np.bool)
    #else:
    #    flags_list = (np.not_equal(covered_array, item) for item in covering_array)
    #    mask_array = and_lists(*flags_list)
    #    return mask_array


#def get_uncovered_mask2(covered_array, covering_array):
#    if len(covering_array) == 0:
#        return np.ones(np.shape(covered_array), dtype=np.bool)
#    else:
#        flags_iter = (np.not_equal(covered_array, item) for item in covering_array)
#        mask_array = iter_reduce_ufunc(np.logical_and, flags_iter)
#        return mask_array


def get_covered_mask(covered_array, covering_array):
    return ~get_uncovered_mask(covered_array, covering_array)


def mult_lists(*args):
    return np.multiply.reduce(args)


def or_lists(*args):
    """
    Like np.logical_and, but can take more than 2 arguments

    SeeAlso:
        and_lists
    """
    flags = np.logical_or.reduce(args)
    return flags


def and_lists(*args):
    """
    Like np.logical_and, but can take more than 2 arguments

    CommandLine:
        python -m vtool.other --test-and_lists

    SeeAlso:
       or_lists

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> arg1 = np.array([1, 1, 1, 1,])
        >>> arg2 = np.array([1, 1, 0, 1,])
        >>> arg3 = np.array([0, 1, 0, 1,])
        >>> args = (arg1, arg2, arg3)
        >>> flags = and_lists(*args)
        >>> result = str(flags)
        >>> print(result)
        [False  True False  True]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> size = 10000
        >>> rng = np.random.RandomState(0)
        >>> arg1 = rng.randint(2, size=size)
        >>> arg2 = rng.randint(2, size=size)
        >>> arg3 = rng.randint(2, size=size)
        >>> args = (arg1, arg2, arg3)
        >>> flags = and_lists(*args)
        >>> # ensure equal division
        >>> segments = 5
        >>> validx = np.where(flags)[0]
        >>> endx = int(segments * (validx.size // (segments)))
        >>> parts = np.split(validx[:endx], segments)
        >>> result = str(list(map(np.sum, parts)))
        >>> print(result)
        [243734, 714397, 1204989, 1729375, 2235191]

    %timeit reduce(np.logical_and, args)
    %timeit np.logical_and.reduce(args)  # wins with more data
    """
    return np.logical_and.reduce(args)


@profile
def axiswise_operation2(arr1, arr2, op, axis=0):
    """
    Apply opperation to each row

    >>> arr1 = (255 * np.random.rand(5, 128)).astype(np.uint8)
    >>> arr2 = vecs.mean(axis=0)
    >>> op = np.subtract
    >>> axis = 0

    performs an operation between an
    (N x A x B ... x Z) array with an
    (N x 1) array

    %timeit op(arr1, arr2[np.newaxis, :])
    %timeit op(arr1, arr2[None, :])
    %timeit op(arr1, arr2.reshape(1, arr2.shape[0]))
    arr2.shape = (1, arr2.shape[0])
    %timeit op(arr1, arr2)
    """
    raise NotImplementedError()


@profile
def rowwise_operation(arr1, arr2, op):
    """
    DEPRICATE THIS IS POSSIBLE WITH STRICTLY BROADCASTING AND
    USING np.newaxis

    DEPRICATE, numpy has better ways of doing this.
    Is the rowwise name correct? Should it be colwise?

    performs an operation between an
    (N x A x B ... x Z) array with an
    (N x 1) array
    """
    # FIXME: not sure this is the correct terminology
    assert arr1.shape[0] == arr2.shape[0]
    broadcast_dimensions = arr1.shape[1:]  # need padding for
    tileshape = tuple(list(broadcast_dimensions) + [1])
    arr2_ = np.rollaxis(np.tile(arr2, tileshape), -1)
    rowwise_result = op(arr1, arr2_)
    return rowwise_result


def colwise_operation(arr1, arr2, op):
    arr1T = arr1.T
    arr2T = arr2.T
    rowwise_result = rowwise_operation(arr1T, arr2T, op)
    colwise_result = rowwise_result.T
    return colwise_result


def compare_matrix_columns(matrix, columns, comp_op=np.equal, logic_op=np.logical_or):
    """
    REPLACE WITH:
        qfx2_invalid = logic_op.reduce([comp_op([:, None], qfx2_normnid) for col1 in qfx2_topnid.T])

    """
    # FIXME: Generalize
    #row_matrix = matrix.T
    #row_list   = columns.T
    return compare_matrix_to_rows(matrix.T, columns.T, comp_op=comp_op, logic_op=logic_op).T


@profile
def compare_matrix_to_rows(row_matrix, row_list, comp_op=np.equal, logic_op=np.logical_or):
    """
    Compares each row in row_list to each row in row matrix using comp_op
    Both must have the same number of columns.
    Performs logic_op on the results of each individual row

    SeeAlso:
        ibeis.algo.hots.nn_weights.mark_name_valid_normalizers

    compop   = np.equal
    logic_op = np.logical_or
    """
    row_result_list = [np.array([comp_op(matrow, row) for matrow in row_matrix])
                       for row in row_list]
    output = row_result_list[0]
    for row_result in row_result_list[1:]:
        logic_op(output, row_result, out=output)
        #output = logic_op(output, row_result)
    return output


def norm01(array, dim=None):
    """
    normalizes a numpy array from 0 to 1 based in its extent

    Args:
        array (ndarray):
        dim   (int):

    Returns:
        ndarray:

    CommandLine:
        python -m vtool.other --test-norm01

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> array = np.array([ 22, 1, 3, 2, 10, 42, ])
        >>> dim = None
        >>> array_norm = norm01(array, dim)
        >>> result = np.array_str(array_norm, precision=3)
        >>> print(result)
        [ 0.512  0.     0.049  0.024  0.22   1.   ]
    """
    if not ut.is_float(array):
        array = array.astype(np.float32)
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    array_norm = np.divide(np.subtract(array, array_min), array_exnt)
    return array_norm


def weighted_geometic_mean_unnormalized(data, weights):
    terms = [x ** w for x, w in zip(data, weights)]
    termprod = iter_reduce_ufunc(np.multiply, iter(terms))
    return termprod


def weighted_geometic_mean(data, weights):
    r"""
    Args:
        data (list of ndarrays):
        weights (ndarray):

    Returns:
        ndarray: gmean_

    CommandLine:
        python -m vtool.other --test-weighted_geometic_mean

    References:
        https://en.wikipedia.org/wiki/Weighted_geometric_mean

    SeeAlso:
        scipy.stats.mstats.gmean

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> data = [np.array(.9), np.array(.5)]
        >>> weights = np.array([1.0, .5])
        >>> gmean_ = weighted_geometic_mean(data, weights)
        >>> result = ('gmean_ = %.3f' % (gmean_,))
        >>> print(result)
        gmean_ = 0.740

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> img1 = rng.rand(4, 4)
        >>> img2 = rng.rand(4, 4)
        >>> data = [img1, img2]
        >>> weights = np.array([.5, .5])
        >>> gmean_ = weighted_geometic_mean(data, weights)
        >>> result = ut.hz_str('gmean_ = ', ut.numpy_str(gmean_, precision=2))
        >>> print(result)
        gmean_ = np.array([[ 0.11,  0.77,  0.68,  0.69],
                           [ 0.64,  0.72,  0.45,  0.83],
                           [ 0.34,  0.5 ,  0.34,  0.71],
                           [ 0.54,  0.62,  0.14,  0.26]], dtype=np.float64)

    Ignore:
        res1 = ((img1 ** .5 * img2 ** .5)) ** 1
        res2 = np.sqrt(img1 * img2)
    """
    terms = [x ** w for x, w in zip(data, weights)]
    termprod = iter_reduce_ufunc(np.multiply, iter(terms))
    exponent = 1 / np.sum(weights)
    gmean_ = termprod ** exponent
    return gmean_


def grab_webcam_image():
    """
    References:
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    CommandLine:
        python -m vtool.other --test-grab_webcam_image --show

    Example:
        >>> # SCRIPT
        >>> from vtool.other import *  # NOQA
        >>> import vtool as vt
        >>> img = grab_webcam_image()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img)
        >>> #vt.imwrite('webcap.jpg', img)
        >>> ut.show_if_requested()
    """
    import cv2
    cap = cv2.VideoCapture(0)
    # Capture frame-by-frame
    ret, img = cap.read()
    # When everything done, release the capture
    cap.release()
    return img


#def xor_swap(arr1, arr2, inplace=True):
#    if not inplace:
#        arr1 = arr1.copy()
#        arr2 = arr2.copy()
#    np.bitwise_xor(arr1, arr2, out=arr1)
#    np.bitwise_xor(arr1, arr2, out=arr2)
#    np.bitwise_xor(arr1, arr2, out=arr1)
#    return arr1, arr2


def find_first_true_indices(flags_list):
    """
    TODO: move to vtool

    returns a list of indexes where the index is the first True position
    in the corresponding sublist or None if it does not exist

    in other words: for each row finds the smallest True column number or None

    Args:
        flags_list (list): list of lists of booleans

    CommandLine:
        python -m utool.util_list --test-find_first_true_indices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> flags_list = [[True, False, True],
        ...               [False, False, False],
        ...               [False, True, True],
        ...               [False, False, True]]
        >>> # execute function
        >>> index_list = find_first_true_indices(flags_list)
        >>> # verify results
        >>> result = str(index_list)
        >>> print(result)
        [0, None, 1, 2]
    """
    def tryget_fisrt_true(flags):
        index_list = np.where(flags)[0]
        index = None if len(index_list) == 0 else index_list[0]
        return index
    index_list = [tryget_fisrt_true(flags) for flags in flags_list]
    return index_list


def find_next_true_indices(flags_list, offset_list):
    r"""
    Uses output of either this function or find_first_true_indices
    to find the next index of true flags

    Args:
        flags_list (list): list of lists of booleans

    CommandLine:
        python -m utool.util_list --test-find_next_true_indices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> # build test data
        >>> flags_list = [[True, False, True],
        ...               [False, False, False],
        ...               [False, True, True],
        ...               [False, False, True]]
        >>> offset_list = find_first_true_indices(flags_list)
        >>> # execute function
        >>> index_list = find_next_true_indices(flags_list, offset_list)
        >>> # verify results
        >>> result = str(index_list)
        >>> print(result)
        [2, None, 2, None]
    """
    def tryget_next_true(flags, offset_):
        offset = offset_ + 1
        relative_flags = flags[offset:]
        rel_index_list = np.where(relative_flags)[0]
        index = None if len(rel_index_list) == 0 else rel_index_list[0] + offset
        return index
    index_list = [None if offset is None else tryget_next_true(flags, offset)
                  for flags, offset in zip(flags_list, offset_list)]
    return index_list


def ensure_rng(seed=None):
    """
    Returns a numpy random number generator given a seed.
    """
    if seed is None:
        rng = np.random
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    return rng


def safe_extreme(arr, op=np.nanmax, fill=np.nan, finite=False):
    if finite:
        arr = arr.compress(np.isfinite(arr))
    if len(arr) == 0:
        return fill
    else:
        return op(arr)


def safe_max(arr, fill=np.nan):
    return safe_extreme(arr, np.max, fill)


def safe_min(arr, fill=np.nan):
    return fill if arr is None or len(arr) == 0 else arr.min()


@profile
def multigroup_lookup_naive(lazydict, keys_list, subkeys_list, custom_func):
    r"""
    Slow version of multigroup_lookup. Makes a call to custom_func for each
    item in zip(keys_list, subkeys_list).

    SeeAlso:
        vt.multigroup_lookup
    """
    data_lists = []
    for keys, subkeys in zip(keys_list, subkeys_list):
        subvals_list = [
            custom_func(lazydict, key, [subkey])[0]
            for key, subkey in zip(keys, subkeys)
        ]
        data_lists.append(subvals_list)
    return data_lists


@profile
def multigroup_lookup(lazydict, keys_list, subkeys_list, custom_func):
    r"""
    Efficiently calls custom_func for each item in zip(keys_list, subkeys_list)
    by grouping subkeys to minimize the number of calls to custom_func.

    We are given multiple lists of keys, and subvals.
    The goal is to group the subvals by keys and apply the subval lookups
    (a call to a function) to the key only once and at the same time.

    Args:
        lazydict (dict of utool.LazyDict):
        keys_list (list):
        subkeys_list (list):
        custom_func (func): must have signature custom_func(lazydict, key, subkeys)

    SeeAlso:
        vt.multigroup_lookup_naive - unoptomized version, but simple to read

    Example:
        >>> # SLOW_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> import vtool as vt
        >>> fpath_list = [ut.grab_test_imgpath(key) for key in ut.util_grabdata.get_valid_test_imgkeys()]
        >>> lazydict = {count: vt.testdata_annot_metadata(fpath) for count, fpath in enumerate(fpath_list)}
        >>> aids_list = np.array([(3, 2), (0, 2), (1, 2), (2, 3)])
        >>> fms       = np.array([[2, 5], [2, 3], [2, 1], [3, 4]])
        >>> keys_list = aids_list.T
        >>> subkeys_list = fms.T
        >>> def custom_func(lazydict, key, subkeys):
        >>>     annot = lazydict[key]
        >>>     kpts = annot['kpts']
        >>>     rchip = annot['rchip']
        >>>     kpts_m = kpts.take(subkeys, axis=0)
        >>>     warped_patches = vt.get_warped_patches(rchip, kpts_m)[0]
        >>>     return warped_patches
        >>> data_lists1 = multigroup_lookup(lazydict, keys_list, subkeys_list, custom_func)
        >>> data_lists2 = multigroup_lookup_naive(lazydict, keys_list, subkeys_list, custom_func)
        >>> vt.sver_c_wrapper.asserteq(data_lists1, data_lists2)

    Example:
        >>> keys_list = [np.array([]), np.array([]), np.array([])]
        >>> subkeys_list = [np.array([]), np.array([]), np.array([])]
    """
    import vtool as vt
    # Group the keys in each multi-list individually
    multi_groups = [vt.group_indices(keys) for keys in keys_list]
    # Combine keys across multi-lists usings a dict_stack
    dict_list = [dict(zip(k, v)) for k, v in multi_groups]
    nested_order = ut.dict_stack2(dict_list, default=[])
    # Use keys and values for explicit ordering
    group_key_list = list(nested_order.keys())
    if len(group_key_list) == 0:
        return multigroup_lookup_naive(lazydict, keys_list, subkeys_list, custom_func)
    group_subxs_list = list(nested_order.values())
    # Extract unique and flat subkeys.
    # Maintain an information to invert back into multi-list form
    group_uf_subkeys_list = []
    group_invx_list = []
    group_cumsum_list = []
    for key, subxs in zip(group_key_list, group_subxs_list):
        # Group subkeys for each key
        subkey_group = vt.ziptake(subkeys_list, subxs, axis=0)
        flat_subkeys, group_cumsum = ut.invertible_flatten2(subkey_group)
        unique_subkeys, invx = np.unique(flat_subkeys, return_inverse=True)
        # Append info
        group_uf_subkeys_list.append(unique_subkeys)
        group_invx_list.append(invx)
        group_cumsum_list.append(group_cumsum)
    # Apply custom function (lookup) to unique each key and its flat subkeys
    group_subvals_list = [
        custom_func(lazydict, key, subkeys)
        for key, subkeys in zip(group_key_list, group_uf_subkeys_list)
    ]
    # Efficiently invert values back into input shape
    # First invert the subkey groupings
    multi_subvals_list = [[] for _ in range(len(multi_groups))]
    _iter = zip(group_key_list, group_subvals_list, group_cumsum_list, group_invx_list)
    for key, subvals, group_cumsum, invx in _iter:
        nonunique_subvals = ut.take(subvals, invx)
        unflat_subvals_list = ut.unflatten2(nonunique_subvals, group_cumsum)
        for subvals_list, unflat_subvals in zip(multi_subvals_list, unflat_subvals_list):
            subvals_list.append(unflat_subvals)
    # Then invert the key groupings
    data_lists = []
    multi_groupxs_list = list(zip(*group_subxs_list))
    for subvals_list, groupxs in zip(multi_subvals_list, multi_groupxs_list):
        datas = vt.invert_apply_grouping(subvals_list, groupxs)
        data_lists.append(datas)
    return data_lists


def asserteq(output1, output2, thresh=1E-8, nestpath=None, level=0, lbl1=None,
             lbl2=None, output_lbl=None, verbose=True, iswarning=False):
    """
    recursive equality checks

    asserts that output1 and output2 are close to equal.
    """
    failed = False
    if lbl1 is None:
        lbl1 = ut.get_varname_from_stack(output1, N=1)
    if lbl2 is None:
        lbl2 = ut.get_varname_from_stack(output2, N=1)
    # Setup
    if nestpath is None:
        # record the path through the nested structure as testing goes on
        nestpath = []
    # print out these variables in all error cases
    common_keys = ['lbl1', 'lbl2', 'level', 'nestpath']
    # CHECK: types
    try:
        assert type(output1) == type(output2), 'types are not equal'
    except AssertionError as ex:
        print(type(output1))
        print(type(output2))
        ut.printex(ex, 'FAILED TYPE CHECKS',
                   keys=common_keys + [(type, 'output1'), (type, 'output2')],
                   iswarning=iswarning)
        failed = True
        if not iswarning:
            raise
    # CHECK: length
    if hasattr(output1, '__len__'):
        try:
            assert len(output1) == len(output2), 'lens are not equal'
        except AssertionError as ex:
            keys = common_keys + [(len, 'output1'), (len, 'output2'), ]
            ut.printex(ex, 'FAILED LEN CHECKS. ', keys=keys)
            raise
    # CHECK: ndarrays
    if isinstance(output1, np.ndarray):
        ndarray_keys = ['output1.shape', 'output2.shape']
        # CHECK: ndarray shape
        try:
            assert output1.shape == output2.shape, 'ndarray shapes are unequal'
        except AssertionError as ex:
            keys = common_keys + ndarray_keys
            ut.printex(ex, 'FAILED NUMPY SHAPE CHECKS.', keys=keys,
                       iswarning=iswarning)
            failed = True
            if not iswarning:
                raise
        # CHECK: ndarray equality
        try:
            passed, error = ut.almost_eq(output1, output2, thresh,
                                         ret_error=True)
            assert np.all(passed), 'ndarrays are unequal.'
        except AssertionError as ex:
            # Statistics on value difference and value difference
            # above the thresholds
            diff_stats = ut.get_stats(error)  # NOQA
            error_stats = ut.get_stats(error[error >= thresh])  # NOQA
            keys = common_keys + ndarray_keys + [
                (len, 'output1'), (len, 'output2'), ('diff_stats'),
                ('error_stats'), ('thresh'),
            ]
            PRINT_VAL_SAMPLE = True
            if PRINT_VAL_SAMPLE:
                keys += ['output1', 'output2']
            ut.printex(ex, 'FAILED NUMPY CHECKS.', keys=keys,
                       iswarning=iswarning)
            failed = True
            if not iswarning:
                raise
    # CHECK: list/tuple items
    elif isinstance(output1, (tuple, list)):
        for count, (item1, item2) in enumerate(zip(output1, output2)):
            # recursive call
            try:
                asserteq(
                    item1, item2, lbl1=lbl2, lbl2=lbl1, thresh=thresh,
                    nestpath=nestpath + [count], level=level + 1)
            except AssertionError as ex:
                ut.printex(ex, 'recursive call failed',
                           keys=common_keys + ['item1', 'item2', 'count'],
                           iswarning=iswarning)
                failed = True
                if not iswarning:
                    raise
    # CHECK: scalars
    else:
        try:
            assert output1 == output2, 'output1 != output2'
        except AssertionError as ex:
            print('nestpath= %r' % (nestpath,))
            ut.printex(ex, 'FAILED SCALAR CHECK.',
                       keys=common_keys + ['output1', 'output2'],
                       iswarning=iswarning)
            failed = True
            if not iswarning:
                raise
    if verbose and level == 0:
        if not failed:
            print('PASSED %s == %s' % (lbl1, lbl2))
        else:
            print('WARNING %s != %s' % (lbl1, lbl2))


def compare_implementations(func1, func2, args, show_output=False, lbl1='', lbl2='', output_lbl=None):
    """
    tests two different implementations of the same function
    """
    print('+ --- BEGIN COMPARE IMPLEMENTATIONS ---')
    func1_name = ut.get_funcname(func1)
    func2_name = ut.get_funcname(func2)
    print('func1_name = %r' % (func1_name,))
    print('func2_name = %r' % (func2_name,))
    # test both versions
    with ut.Timer('time func1=' + func1_name) as t1:
        output1 = func1(*args)
    with ut.Timer('time func2=' + func2_name) as t2:
        output2 = func2(*args)
    if t2.ellapsed == 0:
        t2.ellapsed = 1e9
    print('speedup = %r' % (t1.ellapsed / t2.ellapsed))
    try:
        asserteq(output1, output2, lbl1=lbl1, lbl2=lbl2, output_lbl=output_lbl)
        print('implementations are in agreement :) ')
    except AssertionError as ex:
        # prints out a nested list corresponding to nested structure
        ut.printex(ex, 'IMPLEMENTATIONS DO NOT AGREE', keys=[
            ('func1_name'),
            ('func2_name'), ]
        )
        raise
    finally:
        depth_profile1 = ut.depth_profile(output1)
        depth_profile2 = ut.depth_profile(output2)
        type_profile1 = ut.list_type_profile(output1)
        type_profile2 = ut.list_type_profile(output2)
        print('depth_profile1 = ' + ut.list_str(depth_profile1))
        print('depth_profile2 = ' + ut.list_str(depth_profile2))
        print('type_profile1 = ' + (type_profile1))
        print('type_profile2 = ' + (type_profile2))
    print('L ___ END COMPARE IMPLEMENTATIONS ___')
    return output1

    #out_inliers_py, out_errors_py, out_mats_py = py_output
    #out_inliers_c, out_errors_c, out_mats_c = c_output
    #if show_output:
    #    print('python output:')
    #    print(out_inliers_py)
    #    print(out_errors_py)
    #    print(out_mats_py)
    #    print('c output:')
    #    print(out_inliers_c)
    #    print(out_errors_c)
    #    print(out_mats_c)
    #msg =  'c and python disagree'
    #try:
    #    assert ut.lists_eq(out_inliers_c, out_inliers_py), msg
    #except AssertionError as ex:
    #    ut.printex(ex)
    #    raise
    #try:
    #    passed, error = ut.almost_eq(out_errors_c, out_errors_py, 1E-7, ret_error=True)
    #    assert np.all(passed), msg
    #except AssertionError as ex:
    #    passed_flat = passed.ravel()
    #    error_flat = error.ravel()
    #    failed_indexes = np.where(~passed_flat)[0]
    #    failing_errors = error_flat.take(failed_indexes)
    #    print(failing_errors)
    #    ut.printex(ex)
    #    raise
    #try:
    #    assert np.all(ut.almost_eq(out_mats_c, out_mats_py, 1E-9)), msg
    #except AssertionError as ex:
    #    ut.printex(ex)
    #    raise
    #return out_inliers_c


def bow_test():
    x  = np.array([1, 0, 0, 0, 0, 0], dtype=np.float)
    c1 = np.array([1, 0, 1, 0, 0, 1], dtype=np.float)
    c2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.float)
    x /= x.sum()
    c1 /= c1.sum()
    c2 /= c2.sum()

    fred_query = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    sue_query  = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    tom_query  = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # columns that are distinctive per name
    #                      f1  f2  s1  s2  s3  t1  z1  z2  z3  z4, z5, z6
    fred1      = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    fred2      = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    sue1       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    sue2       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    sue3       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    tom1       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)

    names         = ['fred', 'sue', 'tom']
    num_exemplars = [     3,     2,     1]

    ax2_nx = np.array(ut.flatten([[nx] * num for nx, num in enumerate(num_exemplars)]))

    total = sum(num_exemplars)

    num_words = total * 2
    # bow vector for database
    darr = np.zeros((total, num_words))

    for ax in range(len(darr)):
        nx = ax2_nx[ax]
        num = num_exemplars[nx]
        darr[ax, ax] = 1
        darr[ax, ax + total] = 1

    # nx2_axs = dict(zip(*))
    import vtool as vt
    groupxs = vt.group_indices(ax2_nx)[1]
    class_bows = np.vstack([arr.sum(axis=0) for arr in vt.apply_grouping(darr, groupxs)])
    # generate a query for each class
    true_class_bows = class_bows[:]
    # noise words
    true_class_bows[:, -total:] = 1
    true_class_bows = true_class_bows / true_class_bows.sum(axis=1)[:, None]

    class_bows = class_bows / class_bows.sum(axis=1)[:, None]

    confusion = np.zeros((len(names), len(names)))

    for trial in range(1000):
        # bow vector for query
        qarr = np.zeros((len(names), num_words))

        for cx in range(len(class_bows)):
            sample = np.random.choice(np.arange(num_words), size=30, p=true_class_bows[cx])
            hist = np.histogram(sample, bins=np.arange(num_words + 1))[0]
            qarr[cx] = (hist / hist.max()) >= .5
        # normalize histograms
        qarr = qarr / qarr.sum(axis=1)[:, None]

        # Scoring for each class
        similarity = qarr.dot(class_bows.T)
        distance = 1 - similarity
        confusion += distance

    x /= x.sum()
    c1 /= c1.sum()
    c2 /= c2.sum()

    print(x.dot(c1))
    print(x.dot(c2))


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.other
        python -m vtool.other --allexamples
        python -m vtool.other --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
