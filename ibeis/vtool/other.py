from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
from six import next
from six.moves import zip, range  # NOQA
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[other]', DEBUG=False)


def index_partition(item_list, part1_items):
    """
    returns two lists. The first are the indecies of items in item_list that
    are in part1_items. the second is the indicies in item_list that are not
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


def ziptake(arr_list, indicies_list, axis=None):
    return [arr.take(indicies, axis=axis) for arr, indicies in zip(arr_list, indicies_list)]


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
