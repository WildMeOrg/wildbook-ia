# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[math]', DEBUG=False)


tau = 2 * np.pi  # References: tauday.com

eps = 1E-9


def ensure_monotone_strictly_increasing(arr_):
    """
    Breaks up streaks of equal values by interpolating between the next lowest and next highest value

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.math import *   # NOQA
        >>> import numpy as np
        >>> arr_ = np.array([0.4, 0.4, 0.4, 0.5, 0.6, 0.6, 0.6, 0.7, 0.9, 0.9, 0.91, 0.92, 1.0, 1.0])
        >>> #                  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,  12,  13
        >>> arr = ensure_monotone_strictly_increasing(arr_)
        >>> assert strictly_increasing(arr), 'ensure strict monotonic failed'
    """

    arr = ensure_monotone_increasing(arr_)
    #assert non_decreasing(arr), 'ensure monotonic failed'
    size = len(arr)
    index_list = np.nonzero(np.diff(arr) == 0)[0]
    consecutive_groups = group_consecutive(index_list)
    index_groups = [np.array(group.tolist() + [group.max() + 1]) for group in consecutive_groups]

    runlen_list = [len(group) for group in index_groups]
    # Handle ending corner case
    isend_list = [(group[-1] + 1) < size for group in index_groups]
    min_vals = [arr[group[0]]      if isend else (arr[group[0] - 1] + arr[group[0]]) / 2.0
                for group, isend in zip(index_groups, isend_list)]
    max_vals = [arr[group[-1] + 1] if isend else arr[group[-1]]
                for group, isend in zip(index_groups, isend_list)]
    fill_list = [np.linspace(min_, max_, len_, endpoint=not isend) for min_, max_, len_, isend in zip(min_vals, max_vals, runlen_list, isend_list)]

    for group, fill in zip(index_groups, fill_list):
        arr[group[0]:group[-1] + 1] = fill
    #assert strictly_increasing(arr), 'ensure strict monotonic failed'
    return arr


def group_consecutive(arr):
    """
    Returns lists of consecutive values

    References:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    return np.array_split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def strictly_increasing(L):
    """
    References:
        http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
    """
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    """
    References:
        http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
    """
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L):
    """
    References:
        http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
    """
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    """
    References:
        http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
    """
    return all(x <= y for x, y in zip(L, L[1:]))


def ensure_monotone_increasing(arr_):
    arr = arr_.copy()
    size = len(arr)
    # Ensure increasing from right
    for lx in range(1, size):
        rx = (size - lx - 1)
        if arr[rx] > arr[rx + 1]:
            arr[rx] = arr[rx + 1]
    # ensure increasing from left
    for lx in range(0, size - 1):
        if arr[lx] > arr[lx + 1]:
            arr[lx + 1] = arr[lx]
    return arr


def ensure_monotone_decreasing(arr_):
    arr = arr_.copy()
    size = len(arr)
    # Ensure decreasing from right
    for lx in range(1, size):
        rx = (size - lx - 1)
        if arr[rx] < arr[rx + 1]:
            arr[rx] = arr[rx + 1]
    # ensure increasing from left
    for lx in range(0, size - 1):
        if arr[lx] > arr[lx + 1]:
            arr[lx + 1] = arr[lx]
    return arr


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, vtool.math; utool.doctest_funcs(vtool.math, allexamples=True)"
        python -c "import utool, vtool.math; utool.doctest_funcs(vtool.math)"
        python vtool/math.py
        python vtool/math.py --allexamples
        python vtool/math.py --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
