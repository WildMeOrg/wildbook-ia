# -*- coding: utf-8 -*
"""
These functions might be PR quality for numpy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import ubelt as ub
from six import next
from six.moves import zip, range


def atleast_nd(arr, n, tofront=False):
    r"""
    View inputs as arrays with at least n dimensions.
    TODO: Submit as a PR to numpy

    Args:
        arr (array_like): One array-like object.  Non-array inputs are
                converted to arrays.  Arrays that already have n or more
                dimensions are preserved.
        n (int): number of dimensions to ensure
        tofront (bool): if True new dimensions are added to the front of the
            array.  otherwise they are added to the back.

    CommandLine:
        python -m vtool_ibeis.numpy_utils atleast_nd

    Returns:
        ndarray :
            An array with ``a.ndim >= n``.  Copies are avoided where possible,
            and views with three or more dimensions are returned.  For example,
            a 1-D array of shape ``(N,)`` becomes a view of shape
            ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a view
            of shape ``(M, N, 1)``.

    See Also:
        ensure_shape, np.atleast_1d, np.atleast_2d, np.atleast_3d

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> n = 2
        >>> arr = np.array([1, 1, 1])
        >>> arr_ = atleast_nd(arr, n)
        >>> result = ub.repr2(arr_.tolist())
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> n = 4
        >>> arr1 = [1, 1, 1]
        >>> arr2 = np.array(0)
        >>> arr3 = np.array([[[[[1]]]]])
        >>> arr1_ = atleast_nd(arr1, n)
        >>> arr2_ = atleast_nd(arr2, n)
        >>> arr3_ = atleast_nd(arr3, n)
        >>> result1 = ub.repr2(arr1_.tolist())
        >>> result2 = ub.repr2(arr2_.tolist())
        >>> result3 = ub.repr2(arr3_.tolist())
        >>> result = '\n'.join([result1, result2, result3])
        >>> print(result)
    """
    arr_ = np.asanyarray(arr)
    ndims = len(arr_.shape)
    if n is not None and ndims <  n:
        # append the required number of dimensions to the front or back
        if tofront:
            expander = (None,) * (n - ndims) + (Ellipsis,)
        else:
            expander = (Ellipsis,) + (None,) * (n - ndims)
        arr_ = arr_[expander]
    return arr_


def ensure_shape(arr, dimshape):
    """
    TODO: Submit as a PR to numpy?

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> ensure_shape(np.array([[1, 2]]), (None, 2))
        >>> ensure_shape(np.array([]), (None, 2))
    """
    if isinstance(dimshape, tuple):
        n = len(dimshape)
    else:
        n = dimshape
        dimshape = None
    arr_ = atleast_nd(arr, n)
    if dimshape is not None:
        newshape = tuple([
            d1 if d2 is None else d2
            for d1, d2 in zip(arr_.shape, dimshape)])
        arr_.shape = newshape
    return arr_


def fromiter_nd(iter_, shape, dtype):
    """
    Like np.fromiter but handles iterators that generated
    n-dimensional arrays. Slightly faster than np.array.

    Note:
        np.vstack(list_) is still faster than
        vt.fromiter_nd(ut.iflatten(list_))

    Args:
        iter_ (iter): an iterable that generates homogenous ndarrays
        shape (tuple): the expected output shape
        dtype (dtype): the numpy datatype of the generated ndarrays

    Note:
        The iterable must yeild a numpy array. It cannot yeild a Python list.

    CommandLine:
        python -m vtool_ibeis.numpy_utils fromiter_nd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> dtype = np.float
        >>> total = 11
        >>> rng = np.random.RandomState(0)
        >>> iter_ = (rng.rand(5, 7, 3) for _ in range(total))
        >>> shape = (total, 5, 7, 3)
        >>> result = fromiter_nd(iter_, shape, dtype)
        >>> assert result.shape == shape

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> dtype = np.int
        >>> qfxs = np.array([1, 2, 3])
        >>> dfxs = np.array([4, 5, 6])
        >>> iter_ = (np.array(x) for x in ut.product(qfxs, dfxs))
        >>> total = len(qfxs) * len(dfxs)
        >>> shape = (total, 2)
        >>> result = fromiter_nd(iter_, shape, dtype)
        >>> assert result.shape == shape

    Timeit:
        >>> dtype = np.uint8
        >>> feat_dim = 128
        >>> mu = 1000
        >>> sigma = 500
        >>> n_data = 1000
        >>> rng = np.random.RandomState(42)
        >>> n_feat_list = np.clip(rng.randn(n_data) * sigma + mu, 0, np.inf).astype(np.int)
        >>> # Make a large list of vectors of various sizes
        >>> print('Making random vectors')
        >>> vecs_list = [(rng.rand(num, feat_dim) * 255).astype(dtype) for num in n_feat_list]
        >>> mega_bytes = sum([x.nbytes for x in vecs_list]) / 2 ** 20
        >>> print('mega_bytes = %r' % (mega_bytes,))
        >>> import itertools as it
        >>> import vtool_ibeis as vt
        >>> n_total = n_feat_list.sum()
        >>> target1 = np.vstack(vecs_list)
        >>> iter_ = it.chain.from_iterable(vecs_list)
        >>> shape = (n_total, feat_dim)
        >>> target2 = vt.fromiter_nd(it.chain.from_iterable(vecs_list), shape, dtype=dtype)
        >>> assert np.all(target1 == target2)

        %timeit np.vstack(vecs_list)
        20.4ms
        %timeit vt.fromiter_nd(it.chain.from_iterable(vecs_list), shape, dtype)
        102ms

        iter_ = it.chain.from_iterable(vecs_list)
        %time vt.fromiter_nd(iter_, shape, dtype)
        %time np.vstack(vecs_list)
    """
    num_rows = shape[0]
    chunksize = np.prod(shape[1:])
    itemsize = np.dtype(dtype).itemsize
    # Create dtype that makes an entire ndarray appear as a single item
    chunk_dtype = np.dtype((np.void, itemsize * chunksize))
    arr = np.fromiter(iter_, count=num_rows, dtype=chunk_dtype)
    # Convert back to original dtype and shape
    arr = arr.view(dtype)
    arr.shape = shape
    return arr


def index_to_boolmask(index_list, maxval=None, isflat=True):
    r"""
    transforms a list of indicies into a boolean mask

    Args:
        index_list (ndarray):
        maxval (None): (default = None)

    Kwargs:
        maxval

    Returns:
        ndarray: mask

    CommandLine:
        python -m vtool_ibeis.util_numpy index_to_boolmask

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.util_numpy import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> index_list = np.array([(0, 0), (1, 1), (2, 1)])
        >>> maxval = (3, 3)
        >>> mask = vt.index_to_boolmask(index_list, maxval, isflat=False)
        >>> result = ('mask =\n%s' % (str(mask.astype(np.uint8)),))
        >>> print(result)
        [[1 0 0]
         [0 1 0]
         [0 1 0]]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.util_numpy import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> index_list = np.array([0, 1, 4])
        >>> maxval = 5
        >>> mask = vt.index_to_boolmask(index_list, maxval, isflat=True)
        >>> result = ('mask = %s' % (str(mask.astype(np.uint8)),))
        >>> print(result)
        mask = [1 1 0 0 1]

    """
    #assert index_list.min() >= 0
    if maxval is None:
        maxval = index_list.max()
    mask = np.zeros(maxval, dtype=np.bool)
    if not isflat:
        # assumes non-flat
        mask.__setitem__(tuple(index_list.T), True)
        #mask.__getitem__(tuple(index_list.T))
    else:
        mask[index_list] = True
    return mask


def multiaxis_reduce(ufunc, arr, startaxis=0):
    """
    used to get max/min over all axes after <startaxis>

    CommandLine:
        python -m vtool_ibeis.numpy_utils --test-multiaxis_reduce

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
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


def iter_reduce_ufunc(ufunc, arr_iter, out=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
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


def unique_row_indexes(arr):
    """ np.unique on rows

    Args:
        arr (ndarray): 2d array

    Returns:
        ndarray: unique_rowx

    References:
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

    CommandLine:
        python -m vtool_ibeis.numpy_utils --test-unique_row_indexes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.numpy_utils import *  # NOQA
        >>> arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [.534, .432], [.534, .432], [1, 0], [0, 1]])
        >>> unique_rowx = unique_row_indexes(arr)
        >>> result = ('unique_rowx = %s' % (ub.repr2(unique_rowx),))
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


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool_ibeis.numpy_utils
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
