# -*- coding: utf-8 -*-
"""
# LICENCE Apache 2 or whatever

FIXME: monotization functions need more hueristics
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
from six.moves import range, zip
(print, rrr, profile) = ut.inject2(__name__, '[math]', DEBUG=False)


TAU = np.pi * 2  # References: tauday.com

eps = 1E-9


def interpolate_nans(arr):
    r"""
    replaces nans with interpolated values or 0

    Args:
        arr (ndarray):

    Returns:
        ndarray: new_arr

    CommandLine:
        python -m vtool.util_math --exec-interpolate_nans

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> arr = np.array([np.nan, np.nan, np.nan, np.nan])
        >>> new_arr = interpolate_nans(arr)
        >>> result = ('new_arr = %s' % (str(new_arr),))
        >>> print(result)
        new_arr = [ 0.  0.  0.  0.]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> arr = np.array([np.nan, 1, np.nan, np.nan, np.nan, np.nan, 10, np.nan, 5])
        >>> new_arr = interpolate_nans(arr)
        >>> result = ('new_arr = %s' % (str(new_arr),))
        >>> print(result)
        new_arr = [  1.    1.    2.8   4.6   6.4   8.2  10.    7.5   5. ]
    """
    import vtool as vt
    new_arr = arr.copy()
    nan_idxs = np.where(np.isnan(arr))[0]
    consecutive_groups = vt.group_consecutive(nan_idxs)
    last_index = len(arr) - 1
    for group in consecutive_groups:
        min_ = group.min()
        max_ = group.max()
        if min_ == 0 and max_ == last_index:
            upper = lower = 0
        else:
            if min_ != 0:
                lower = arr[min_ - 1]
            if max_ != last_index:
                upper = arr[max_ + 1]
            if min_ == 0:
                lower = upper
            if max_ == last_index:
                upper = lower
        new_arr[min_:max_ + 1] = np.linspace(lower, upper, len(group) + 2)[1:-1]
    return new_arr


def ensure_monotone_strictly_increasing(arr_, left_endpoint=None, right_endpoint=None, zerohack=False, onehack=False, newmode=True):
    """

    Args:
        arr_ (ndarray): sequence to monotonize
        zerohack (bool): default False, if True sets the first element to be zero and linearlly interpolates to the first nonzero item
        onehack (bool):  default False, if True one will not be in the resulting array (replaced with number very close to one)

    References:
        http://mathoverflow.net/questions/17464/making-a-non-monotone-function-monotone
        http://stackoverflow.com/questions/28563711/make-a-numpy-array-monotonic-without-a-python-loop
        https://en.wikipedia.org/wiki/Isotonic_regression
        http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html

    CommandLine:
        python -m vtool.util_math --test-ensure_monotone_strictly_increasing --show
        python -m vtool.util_math --test-ensure_monotone_strictly_increasing --show --offset=0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.util_math import *   # NOQA
        >>> import numpy as np
        >>> arr_ = np.array([0.4, 0.4, 0.4, 0.5, 0.6, 0.6, 0.6, 0.7, 0.9, 0.9, 0.91, 0.92, 1.0, 1.0])
        >>> arr = ensure_monotone_strictly_increasing(arr_)
        >>> assert strictly_increasing(arr), 'ensure strict monotonic failed1'

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> import vtool as vt
        >>> left_endpoint = None
        >>> rng = np.random.RandomState(0)
        >>> right_endpoint = None
        >>> domain = np.arange(100)
        >>> offset = ut.get_argval('--offset', type_=float, default=2.3)
        >>> arr_ = np.sin(np.pi * (domain / 100) - offset) + (rng.rand(len(domain)) - .5) * .1 + 1.2
        >>> #arr_ = vt.tests.dummy.testdata_nonmonotonic()
        >>> #domain = np.arange(len(arr_))
        >>> arr = ensure_monotone_strictly_increasing(arr_, left_endpoint, right_endpoint)
        >>> result = str(arr)
        >>> print(result)
        >>> print('arr = %r' % (arr,))
        >>> print('arr = %r' % (np.diff(arr),))
        >>> assert non_decreasing(arr), 'ensure nondecreasing failed2'
        >>> assert strictly_increasing(arr), 'ensure strict monotonic failed2'
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
    arr = ensure_monotone_increasing(arr_, newmode=newmode)
    #assert strictly_increasing(arr), 'ensure strict monotonic failed'
    #import utool as ut
    #print(ut.get_stats(arr))
    #if arr.max() == 1.0:
    #    ut.embed()
    if zerohack:
        left_endpoint = 0.0
    if onehack:
        right_endpoint = 1.0
    #print('arr_in = %r' % (arr,))
    #print('right_endpoint = %r' % (right_endpoint,))
    #print('left_endpoint = %r' % (left_endpoint,))
    arr = breakup_equal_streak(arr, left_endpoint, right_endpoint)
    #print('arr_out = %r' % (arr,))
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
        python -m vtool.util_math --test-ensure_monotone_strictly_decreasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> import vtool as vt
        >>> domain = np.arange(100)
        >>> rng = np.random.RandomState(0)
        >>> arr_ = np.sin(np.pi * (domain / 75) + 1.3) + (rng.rand(len(domain)) - .5) * .05 + 1.0
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

    Args:
        arr_in (?):
        left_endpoint (None): (default = None)
        right_endpoint (None): (default = None)

    Returns:
        ndarray: arr -

    CommandLine:
        python -m vtool.util_math --exec-breakup_equal_streak
        python -m vtool.util_math --test-ensure_monotone_strictly_increasing --show --offset=0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> arr_in = np.array([0, 0, 1, 1, 2, 2], dtype=np.float32)
        >>> arr_in = np.array([ 1.20488135,  1.2529297 ,  1.27306686,  1.29859663,
        >>>    1.31769871, 1.37102388,  1.38114004,  1.45732054,  1.48119571,  1.48119571,
        >>>     1.5381895 ,  1.54162741,  1.57492901,  1.61129523,  1.61129523,
        >>>     1.61270343,  1.63377551,  1.7423034 ,  1.76364247,  1.79908459,
        >>>     1.83564709,  1.83819742,  1.83819742,  1.86786967,  1.86786967,
        >>>     1.90720142,  1.90720142,  1.92293973,  1.92293973, ]) / 2
        >>> left_endpoint = 0
        >>> right_endpoint = 1.0
        >>> arr = breakup_equal_streak(arr_in, left_endpoint, right_endpoint)
        >>> assert strictly_increasing(arr)
        >>> result = ('arr = %s' % (str(arr),))
        >>> print(result)
    """
    #assert non_decreasing(arr), 'ensure monotonic failed'
    arr = arr_in.copy()

    # Find maxish and minish before adjusting
    if right_endpoint is not None:
        # Dont be so confident
        # Set the rightmost value to be almost ``right_endpoint``
        range_ = np.abs(arr_in[0] - arr_in[-1])
        if arr_in[0] < right_endpoint:
            # increasing arr
            almost_right = right_endpoint - (range_ * .001)
            # The second highest value in arr, or close enough
            maxish = min(almost_right, arr[arr < right_endpoint].max())
            newmax = (right_endpoint * .9 + maxish * .1)
        else:
            # decreasing arr
            almost_right = right_endpoint + (range_ * .001)
            # The second lowest value in arr, or close enough
            minish = max(almost_right, arr[arr > right_endpoint].min())
            #newmin = (right_endpoint + minish) / 2.0
            newmin = (right_endpoint * .9 + minish * .1)

    size = len(arr)
    is_same = np.abs(np.diff(arr)) < 1E-8
    #is_same = np.diff(arr) == 0
    #index_list = np.nonzero(np.diff(arr) == 0)[0]
    index_list = np.nonzero(is_same)[0]
    if len(index_list) == 0:
        # If there are no consecutive numbers then arr must be strictly
        # increasing
        return arr

    consecutive_groups = group_consecutive(index_list)
    index_groups = [np.array(group.tolist() + [group.max() + 1]) for group in consecutive_groups]
    # Nope this is right
    # Hack because sometimes things arent't grouped correctly
    # items in index groups are consectuive and breaking things
    # arr[ut.flatten(index_groups)]
    #index_groups2 = []
    #for group in index_groups:
    #    if len(index_groups2) == 0:
    #        index_groups2.append(group)
    #    elif index_groups2[-1][-1] + 1 == group[0]:
    #        print('group = %r' % (group,))
    #        # JOIN CASE
    #    else:
    #        index_groups2.append(group)

    runlen_list = [len(group) for group in index_groups]
    # Handle ending corner case

    #isend_list = [(group[-1] + 1) < size for group in index_groups]
    # Error? Should this be less?
    #isend_list = [(group[-1] + 1) < size for group in index_groups]
    isend_list = [(group[-1] + 1) >= size for group in index_groups]
    isstart_list = [group[0] == 0 for group in index_groups]

    min_vals = [
        arr[group[0]]
        if isstart else
        (.49 * arr[group[0] - 1] + .51 * arr[group[0]])  # value between previous and this one (bumped to right)
        for group, isstart in zip(index_groups, isstart_list)
    ]

    max_vals = [
        arr[group[-1]]  # Max value is the value of the previous group?
        if isend else
        (.49 * arr[group[-1] + 1] + .51 * arr[group[-1]])  # value between next and this one (bumped to right)
        for group, isend in zip(index_groups, isend_list)
    ]

    #import vtool as vt
    #vt.apply_grouping(arr, index_groups)
    #np.vstack((min_vals, max_vals)).T

    fill_list = [np.linspace(min_, max_, len_, endpoint=not isend)
                 for min_, max_, len_, isend in zip(min_vals, max_vals, runlen_list, isend_list)]

    for group, fill in zip(index_groups, fill_list):
        arr[group[0]:group[-1] + 1] = fill

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
        if arr_in[0] < right_endpoint:
            # increasing arr
            if len(isend_list) > 0 and isend_list[-1]:
                # adjust for adjustments
                maxish = min(min_vals[-1], maxish)
            arr[arr >= maxish] = np.linspace(maxish, newmax, sum(arr >= maxish))
        else:
            if len(isstart_list) > 0 and isstart_list[0]:
                minish = max(max_vals[0], minish)
            # decreasing arr
            arr[arr <= minish] = np.linspace(minish, newmin, sum(arr <= minish))
    return arr


def group_consecutive(arr):
    """
    Returns lists of consecutive values

    References:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy

    Args:
        arr (ndarray): must be integral and unique

    Returns:
        ndarray: arr -

    CommandLine:
        python -m vtool.util_math --exec-group_consecutive

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> arr = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 99, 100, 101])
        >>> groups = group_consecutive(arr)
        >>> result = ('groups = %s' % (str(groups),))
        >>> print(result)
        groups = [array([1, 2, 3]), array([ 5,  6,  7,  8,  9, 10]), array([15]), array([ 99, 100, 101])]
    """
    #is_nonconsec = np.abs(np.diff(arr)) < 1E-2
    #split_indicies = np.nonzero(is_nonconsec)[0] + 1
    split_indicies = np.nonzero(np.diff(arr) != 1)[0] + 1
    groups = np.array_split(arr, split_indicies)
    return groups
    #return np.array_split(arr, np.where(np.diff(arr) != 1)[0] + 1)


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


def ensure_monotone_increasing(arr_, fromright=True, fromleft=True, newmode=True):
    r"""
    Args:
        arr_ (ndarray):

    Returns:
        ndarray: arr

    CommandLine:
        python -m vtool.util_math --test-ensure_monotone_increasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> size_ = 100
        >>> domain = np.arange(size_)
        >>> offset = ut.get_argval('--offset', type_=float, default=2.3)
        >>> arr_ = np.sin(np.pi * (domain / 100) - offset) + (rng.rand(len(domain)) - .5) * .1
        >>> arr = ensure_monotone_increasing(arr_, fromleft=False, fromright=True)
        >>> result = str(arr)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot2(domain, arr_, 'r-', fnum=1, pnum=(2, 1, 1), title='before', equal_aspect=False)
        >>> pt.plot2(domain, arr, 'r-', fnum=1, pnum=(2, 1, 2), title='after monotonization (increasing)', equal_aspect=False)
        >>> ut.show_if_requested()
    """
    if newmode:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression()
        arr = ir.fit_transform(np.arange(len(arr_)), arr_)
    else:
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
        python -m vtool.util_math --test-ensure_monotone_decreasing --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> size_ = 100
        >>> domain = np.arange(size_)
        >>> arr_ = np.sin(np.pi * (domain / 100) ) + (rng.rand(len(domain)) - .5) * .1
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
        python -m vtool.util_math --test-iceil

    Setup:
        >>> from vtool.util_math import *  # NOQA
        >>> import vtool as vt

    Example0:
        >>> # ENABLE_DOCTEST
        >>> num = 1.5
        >>> result = repr(vt.iceil(num))
        >>> print(result)
        2

    Example1:
        >>> # ENABLE_DOCTEST
        >>> num = [1.5, 2.9]
        >>> result = ut.repr2(vt.iceil(num), with_dtype=True)
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
        python -m vtool.util_math --test-gauss_func1d

    CommandLine:
        python -m vtool.util_math --test-gauss_func1d --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
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
        python -m vtool.util_math --test-gauss_func1d_unnormalized --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
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


def logistic_01(x):
    r"""
    Args:
        x (?):

    CommandLine:
        python -m vtool.util_math --exec-logistic_01 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.util_math import *  # NOQA
        >>> x = np.linspace(0, 1)
        >>> y = logistic_01(x)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.plot(x, y)
        >>> ut.show_if_requested()
    """
    from scipy.special import expit
    y = expit(((x * 2) - 1.0) * 6)
    return y
    # return L / (1 + np.exp(-k * (x - x0)))


def logit(x):
    from scipy.special import logit
    return logit(x)


# def relu_01(x, p):
#     return np.minimum(0, (x - p) / (1 - p))


def beaton_tukey_loss(u, a=1):
    """
    CommandLine:
        python -m plottool.draw_func2 --exec-plot_func --show --range=-8,8 --func=vt.beaton_tukey_weight,vt.beaton_tukey_loss

    References:
        Steward_Robust%20parameter%20estimation%20in%20computer%20vision.pdf
    """
    result = np.empty(u.shape, dtype=u.dtype)
    is_case1 = np.abs(u) <= a
    u1 = u[is_case1]
    result[is_case1] = ((a ** 2) / 6) * (1 - (1 - (u1 / a) ** 2) ** 3)
    result[~is_case1] = (a ** 2 / 6)
    return result


def beaton_tukey_weight(u, a=1):
    """
    CommandLine:
        python -m plottool.draw_func2 --exec-plot_func --show --range=-8,8 --func=vt.beaton_tukey_weight

    References:
        Steward_Robust%20parameter%20estimation%20in%20computer%20vision.pdf
    """
    result = np.empty(u.shape, dtype=u.dtype)
    is_case1 = np.abs(u) <= a
    u1 = u[is_case1]
    result[is_case1] = u1 * (1 - (u1 / a) ** 2) ** 2
    result[~is_case1] = 0
    return result


def gauss_parzen_est(dist, L=1, sigma=.38):
    """
    python -m plottool.draw_func2 --exec-plot_func --show --range=-.2,.2 --func=vt.gauss_parzen_est
    python -m plottool.draw_func2 --exec-plot_func --show --range=0,1 --func=vt.gauss_parzen_est
    """
    tau = np.pi * 2
    const_term = np.log(L * sigma * np.sqrt(tau))
    return np.exp((-dist / (2 * sigma ** 2)) - const_term)


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, vtool.util_math; utool.doctest_funcs(vtool.util_math, allexamples=True)"
        python -c "import utool, vtool.util_math; utool.doctest_funcs(vtool.util_math)"
        python -m vtool.util_math
        python -m vtool.util_math --allexamples
        python -m vtool.util_math --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
