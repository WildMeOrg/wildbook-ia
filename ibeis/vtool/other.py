# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import functools  # NOQA
from six import next
from six.moves import zip, range  # NOQA
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[other]', DEBUG=False)


def argsort_multiarray(arrays, reverse=False):
    sorting_records = np.rec.fromarrays(arrays)
    sort_stride = (-reverse * 2) + 1
    sortx = sorting_records.argsort()[::sort_stride]
    return sortx


def compute_unique_data_ids(data):
    """
    CommandLine:
        python -m vtool.other --test-compute_unique_data_ids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.other import *  # NOQA
        >>> data = np.array([[0, 0], [0, 1], [1, 1], [0, 0], [.534432, .432432], [.534432, .432432]])
        >>> dataid_list = compute_unique_data_ids(data)
        >>> dataid_list = compute_unique_data_ids(data)
        >>> result = str(dataid_list)
        >>> print(result)
        >>> print(len(np.unique(dataid_list)))
        >>> print(len((dataid_list)))
        [0, 1, 2, 0, 3, 3]

    """
    # construct a unique id for every edge
    hashable_rows = [tuple(row_.tolist()) for row_ in data]
    dataid_list = np.array(compute_unique_data_ids_(hashable_rows), dtype=np.int32)
    return dataid_list


def compute_unique_data_ids_(hashable_rows):
    iddict_ = {}
    for row in hashable_rows:
        if row not in iddict_:
            iddict_[row] = len(iddict_)
    dataid_list = ut.dict_take(iddict_, hashable_rows)
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
    dataid_list = np.array([sum([item * offset for item, offset in zip(row, offsets)]) for row in data])
    return dataid_list


@profile
def trytake(list_, index_list):
    return None if list_ is None else list_take_(list_, index_list)


@profile
def list_take_(list_, index_list):
    if isinstance(list_, np.ndarray):
        return list_.take(index_list, axis=0)
    else:
        return ut.list_take(list_, index_list)


@profile
def list_compress_(list_, flag_list):
    if isinstance(list_, np.ndarray):
        return list_.compress(flag_list, axis=0)
    else:
        return ut.filter_items(list_, flag_list)


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
    part1_indexes = part1_indexes.astype(np.int32)
    part2_indexes = part2_indexes.astype(np.int32)
    return part1_indexes, part2_indexes


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


def zipcompress(arr_list, flags_list, axis=None):
    return [np.compress(flags, arr, axis=axis) for arr, flags in zip(arr_list, flags_list)]


def ziptake(arr_list, indices_list, axis=None):
    return [arr.take(indices, axis=axis) for arr, indices in zip(arr_list, indices_list)]


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
    nrows, ncols = A.shape
    assert A.dtype == B.dtype, ('A and B must have the same dtypes.'
                                'A.dtype=%r, B.dtype=%r' % (A.dtype, B.dtype))
    dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
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
        >>> np.random.seed(0)
        >>> arg1 = np.random.randint(2, size=size)
        >>> arg2 = np.random.randint(2, size=size)
        >>> arg3 = np.random.randint(2, size=size)
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
    flags = np.logical_and.reduce(args)
    return flags
    # TODO: Cython
    # TODO: remove reduce statement (bleh)
    #if six.PY2:
    #    flags =  reduce(np.logical_and, args)
    #elif six.PY3:
    #    flags =  functools.reduce(np.logical_and, args)
    #return flags


def and_3lists(arr1, arr2, arr3):
    """
    >>> from vtool.other import *  # NOQA
    >>> np.random.seed(53)
    >>> arr1 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> arr2 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> arr3 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> output = and_3lists(arr1, arr2, arr3)
    >>> print(utool.hashstr(output))
    prxuyy1w%ht57jaf

    #if CYTH
    #CYTH_INLINE
    cdef:
        np.ndarray arr1
        np.ndarray arr2
        np.ndarray arr3
    #endif
    """
    return np.logical_and(np.logical_and(arr1, arr2), arr3)


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
        ibeis.model.hots.nn_weights.mark_name_valid_normalizers

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
