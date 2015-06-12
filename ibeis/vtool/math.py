# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
from six.moves import range, zip
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[math]', DEBUG=False)


TAU = np.pi * 2  # References: tauday.com

eps = 1E-9


def ensure_monotone_strictly_increasing(arr_, left_endpoint=None, right_endpoint=None, zerohack=False, onehack=False):
    """

    Args:
        arr_ (ndarray): sequence to monotonize
        zerohack (bool): default False, if True sets the first element to be zero and linearlly interpolates to the first nonzero item
        onehack (bool):  default False, if True one will not be in the resulting array (replaced with number very close to one)

    CommandLine:
        python -m vtool.math --test-ensure_monotone_strictly_increasing --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.math import *   # NOQA
        >>> import numpy as np
        >>> arr_ = np.array([0.4, 0.4, 0.4, 0.5, 0.6, 0.6, 0.6, 0.7, 0.9, 0.9, 0.91, 0.92, 1.0, 1.0])
        >>> arr = ensure_monotone_strictly_increasing(arr_)
        >>> assert strictly_increasing(arr), 'ensure strict monotonic failed'

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> import vtool as vt
        >>> left_endpoint = 0.0
        >>> right_endpoint = 1.0
        >>> domain = np.arange(100)
        >>> arr_ = np.sin(np.pi * (domain / 100) - 2.3) + (np.random.rand(len(domain)) - .5) * .1 + 1.2
        >>> #arr_ = vt.tests.dummy.testdata_nonmonotonic()
        >>> #domain = np.arange(len(arr_))
        >>> arr = ensure_monotone_strictly_increasing(arr_, left_endpoint, right_endpoint)
        >>> result = str(arr)
        >>> print(result)
        >>> assert strictly_increasing(arr), 'ensure strict monotonic failed'
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(domain, arr_, 'r-', fnum=1, pnum=(3, 1, 1), title='before', equal_aspect=False)
        >>> arr2 = ensure_monotone_increasing(arr_)
        >>> pt.plot2(domain, arr, 'b-', fnum=1, pnum=(3, 1, 2), equal_aspect=False)
        >>> pt.plot2(domain, arr2, 'r-', fnum=1, pnum=(3, 1, 2), title='after monotonization (decreasing)', equal_aspect=False)
        >>> pt.plot2(domain, arr, 'r-', fnum=1, pnum=(3, 1, 3), title='after monotonization (strictly decreasing)', equal_aspect=False)
        >>> ut.show_if_requested()
    """
    #with ut.EmbedOnException():
    arr = ensure_monotone_increasing(arr_)
    #assert strictly_increasing(arr), 'ensure strict monotonic failed'
    #import utool as ut
    #print(ut.get_stats(arr))
    #if arr.max() == 1.0:
    #    ut.embed()
    if zerohack:
        left_endpoint = 0.0
    if onehack:
        right_endpoint = 1.0
    arr = breakup_equal_streak(arr, left_endpoint, right_endpoint)
    return arr


def ensure_monotone_strictly_decreasing(arr_, left_endpoint=None, right_endpoint=None):
    """

    Args:
        arr_ (ndarray):
        left_endpoint (None):
        right_endpoint (None):

    Returns:
        ndarray: arr

    CommandLine:
        python -m vtool.math --test-ensure_monotone_strictly_decreasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> import vtool as vt
        >>> domain = np.arange(100)
        >>> arr_ = np.sin(np.pi * (domain / 75) + 1.3) + (np.random.rand(len(domain)) - .5) * .05 + 1.0
        >>> #arr_ = vt.tests.dummy.testdata_nonmonotonic()
        >>> #domain = np.arange(len(arr_))
        >>> left_endpoint = 2.5
        >>> right_endpoint = 0.25
        >>> arr = ensure_monotone_strictly_decreasing(arr_, left_endpoint, right_endpoint)
        >>> result = str(arr)
        >>> print(result)
        >>> assert strictly_decreasing(arr), 'ensure strict monotonic failed'
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(domain, arr_, 'r-', fnum=1, pnum=(3, 1, 1), title='before', equal_aspect=False)
        >>> arr2 = ensure_monotone_decreasing(arr_)
        >>> pt.plot2(domain, arr, 'b-', fnum=1, pnum=(3, 1, 2), equal_aspect=False)
        >>> pt.plot2(domain, arr2, 'r-', fnum=1, pnum=(3, 1, 2), title='after monotonization (decreasing)', equal_aspect=False)
        >>> pt.plot2(domain, arr, 'r-', fnum=1, pnum=(3, 1, 3), title='after monotonization (strictly decreasing)', equal_aspect=False)
        >>> ut.show_if_requested()
    """
    #raise NotImplementedError('unfinished')
    arr = ensure_monotone_decreasing(arr_)
    # FIXME: doesn't work yet I don't think
    arr = breakup_equal_streak(arr, left_endpoint, right_endpoint)
    return arr


def breakup_equal_streak(arr_in, left_endpoint=None, right_endpoint=None):
    """
    Breaks up streaks of equal values by interpolating between the next lowest and next highest value
    """

    #memtrack = ut.MemoryTracker(disable=False)
    #memtrack.report('[BREAKUP_EQUAL_STREAK]')
    #assert non_decreasing(arr), 'ensure monotonic failed'
    arr = arr_in.copy()
    size = len(arr)
    index_list = np.nonzero(np.diff(arr) == 0)[0]
    if len(index_list) == 0:
        # If there are no consecutive numbers then arr must be strictly
        # increasing
        return arr

    consecutive_groups = group_consecutive(index_list)
    index_groups = [np.array(group.tolist() + [group.max() + 1]) for group in consecutive_groups]

    runlen_list = [len(group) for group in index_groups]
    # Handle ending corner case

    isend_list = [(group[-1] + 1) < size for group in index_groups]

    min_vals = [arr[group[0]]      if isend else (arr[group[0] - 1] + arr[group[0]]) / 2.0
                for group, isend in zip(index_groups, isend_list)]

    #memtrack.report('[MIN]')

    max_vals = [arr[group[-1] + 1] if isend else arr[group[-1]]
                for group, isend in zip(index_groups, isend_list)]

    #memtrack.report('[MAX]')

    fill_list = [np.linspace(min_, max_, len_, endpoint=not isend)
                 for min_, max_, len_, isend in zip(min_vals, max_vals, runlen_list, isend_list)]

    #memtrack.report('[FILL]')

    for group, fill in zip(index_groups, fill_list):
        arr[group[0]:group[-1] + 1] = fill

    #memtrack.report('[GROUP]')

    if left_endpoint is not None and len(index_groups) > 0:
        # Set the leftmost value to be exactly ``left_endpoint``
        group_ = index_groups[0]
        if group_[0] == 0:
            group_0_slice = slice(group_[0], group_[-1] + 1)
            arr[group_0_slice] = np.linspace(left_endpoint, arr[group_[-1]], len(group_))
        else:
            arr[0] = left_endpoint
    if right_endpoint is not None:
        # Dont be so confident
        # Set the rightmost value to be almost ``right_endpoint``
        range_ = np.abs(arr_in[0] - arr_in[-1])
        if arr_in[0] < right_endpoint:
            # increasing arr
            almost_right = right_endpoint - (range_ * .001)
            # The second highest value in arr, or close enough
            maxish = min(almost_right, arr[arr < right_endpoint].max())
            newmax = (right_endpoint + maxish) / 2.0
            arr[arr >= maxish] = np.linspace(maxish, newmax, sum(arr >= maxish))
        else:
            # decreasing arr
            almost_right = right_endpoint + (range_ * .001)
            # The second lowest value in arr, or close enough
            minish = max(almost_right, arr[arr > right_endpoint].min())
            newmin = (right_endpoint + minish) / 2.0
            arr[arr <= minish] = np.linspace(minish, newmin, sum(arr <= minish))
    #memtrack.report('[FIXED]')
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


def ensure_monotone_increasing(arr_, fromright=True, fromleft=True):
    r"""
    Args:
        arr_ (ndarray):

    Returns:
        ndarray: arr

    CommandLine:
        python -m vtool.math --test-ensure_monotone_increasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> np.random.seed(0)
        >>> size_ = 100
        >>> domain = np.arange(size_)
        >>> arr_ = np.sin(np.pi * (domain / 100) - 2.3) + (np.random.rand(len(domain)) - .5) * .1
        >>> arr = ensure_monotone_increasing(arr_, fromleft=False, fromright=True)
        >>> result = str(arr)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(domain, arr_, 'r-', fnum=1, pnum=(2, 1, 1), title='before', equal_aspect=False)
        >>> pt.plot2(domain, arr, 'r-', fnum=1, pnum=(2, 1, 2), title='after monotonization (increasing)', equal_aspect=False)
        >>> ut.show_if_requested()
    """
    arr = arr_.copy()
    size = len(arr)
    # Ensure increasing from right
    if fromright:
        for lx in range(1, size):
            rx = (size - lx - 1)
            if arr[rx] > arr[rx + 1]:
                arr[rx] = arr[rx + 1]
    if fromleft:
        # ensure increasing from left
        for lx in range(0, size - 1):
            if arr[lx] > arr[lx + 1]:
                arr[lx + 1] = arr[lx]
    return arr


def ensure_monotone_decreasing(arr_, fromleft=True, fromright=True):
    r"""
    Args:
        arr_ (ndarray):

    Returns:
        ndarray: arr

    CommandLine:
        python -m vtool.math --test-ensure_monotone_decreasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> np.random.seed(0)
        >>> size_ = 100
        >>> domain = np.arange(size_)
        >>> arr_ = np.sin(np.pi * (domain / 100) ) + (np.random.rand(len(domain)) - .5) * .1
        >>> arr = ensure_monotone_decreasing(arr_, fromright=True, fromleft=True)
        >>> result = str(arr)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(domain, arr_, 'r-', fnum=1, pnum=(2, 1, 1), title='before', equal_aspect=False)
        >>> pt.plot2(domain, arr, 'r-', fnum=1, pnum=(2, 1, 2), title='after monotonization (decreasing)', equal_aspect=False)
        >>> ut.show_if_requested()
    """
    arr = arr_.copy()
    size = len(arr)
    if fromright:
        # Ensure decreasing from right
        for lx in range(1, size):
            rx = (size - lx - 1)
            if arr[rx] < arr[rx + 1]:
                arr[rx] = arr[rx + 1]
    if fromleft:
        # ensure increasing from left
        for lx in range(0, size - 1):
            if arr[lx] < arr[lx + 1]:
                arr[lx + 1] = arr[lx]
    return arr


def test_language_modulus():
    """
    References:
        http://en.wikipedia.org/wiki/Modulo_operation
    """
    import math
    import utool as ut
    TAU = math.pi * 2
    num_list = [-8, -1, 0, 1, 2, 6, 7, 29]
    modop_result_list = []
    fmod_result_list = []
    for num in num_list:
        num = float(num)
        modop_result_list.append(num % TAU)
        fmod_result_list.append(math.fmod(num, TAU))
    table = ut.make_csv_table([num_list, modop_result_list, fmod_result_list],
                              ['num',  'modop', 'fmod'], 'mods', [float, float, float] )
    print(table)


def iceil(num, dtype=np.int32):
    """ Integer ceiling. (because numpy doesn't seem to have it!)

    Args:
        num (ndarray or scalar):

    Returns:
        ndarray or scalar:

    CommandLine:
        python -m vtool.math --test-iceil

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import vtool as vt
        >>> num = 1.5
        >>> result = repr(vt.iceil(num))
        >>> print(result)
        2

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import vtool as vt
        >>> num = [1.5, 2.9]
        >>> result = ut.numpy_str(vt.iceil(num))
        >>> print(result)
        np.array([2, 3], dtype=np.int32)
    """
    return np.ceil(num).astype(dtype)


def iround(num, dtype=np.int32):
    """ Integer round. (because numpy doesn't seem to have it!) """
    return np.round(num).astype(dtype)


def gauss_func1d(x, mu=0.0, sigma=1.0):
    r"""
    Args:
        x (?):
        mu (float):
        sigma (float):

    CommandLine:
        python -m vtool.math --test-gauss_func1d

    CommandLine:
        python -m vtool.math --test-gauss_func1d --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> # build test data
        >>> x = np.array([-2, -1, -.5, 0, .5, 1, 2])
        >>> mu = 0.0
        >>> sigma = 1.0
        >>> # execute function
        >>> gaussval = gauss_func1d(x, mu, sigma)
        >>> # verify results
        >>> result = np.array_repr(gaussval, precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot(x, gaussval)
        >>> ut.show_if_requested()
        array([ 0.05,  0.24,  0.35,  0.4 ,  0.35,  0.24,  0.05])
    """
    coeff = np.reciprocal(sigma * np.sqrt(TAU))
    exponent_expr_numer = np.power(np.subtract(x, mu), 2)
    exponent_expr_denom = (-2 * (sigma ** 2))
    exponent_expr = np.divide(exponent_expr_numer, exponent_expr_denom)
    gaussval = coeff * np.exp(exponent_expr)
    return gaussval


def gauss_func1d_unnormalized(x, sigma=1.0):
    """
    faster version of gauss_func1d with no normalization. So the maximum point
    will have a value of 1.0


    CommandLine:
        python -m vtool.math --test-gauss_func1d_unnormalized --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> # build test data
        >>> x = np.array([-2, -1, -.5, 0, .5, 1, 2])
        >>> sigma = 1.0
        >>> # execute function
        >>> gaussval = gauss_func1d_unnormalized(x, sigma)
        >>> # verify results
        >>> result = np.array_repr(gaussval, precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot(x, gaussval)
        >>> ut.show_if_requested()
        array([ 0.05,  0.24,  0.35,  0.4 ,  0.35,  0.24,  0.05])
    """
    exponent_expr_denom = (-2 * (sigma ** 2))
    tmp = exponent_expr_numer = np.power(x, 2.0)
    exponent_expr = np.divide(exponent_expr_numer, exponent_expr_denom, out=tmp)
    gaussval = np.exp(exponent_expr, out=tmp)
    return gaussval


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, vtool.math; utool.doctest_funcs(vtool.math, allexamples=True)"
        python -c "import utool, vtool.math; utool.doctest_funcs(vtool.math)"
        python -m vtool.math
        python -m vtool.math --allexamples
        python -m vtool.math --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
