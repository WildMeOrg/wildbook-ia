# Licence:
#
# TODO: Rename
# util_science
#
from __future__ import absolute_import, division, print_function
import numpy as np
from collections import defaultdict
from itertools import izip
from . import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


def normalize(array, dim=0):
    return norm_zero_one(array, dim)


def norm_zero_one(array, dim=0):
    'normalizes a numpy array from 0 to 1'
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    return np.divide(np.subtract(array, array_min), array_exnt)


def find_std_inliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


def choose(n, k):
    import scipy.misc
    return scipy.misc.comb(n, k, True)


def cartesian(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7],
           [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7],
           [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7]])
    '''
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def almost_eq(a, b, thresh=1E-11, ret_error=False):
    """ checks if floating point number are equal to a threshold """
    error = np.abs(a - b)
    passed = error < thresh
    if ret_error:
        return passed, error
    return passed


def xywh_to_tlbr(bbox, img_wh):
    """ converts xywh format to (tlx, tly, blx, bly) """
    (img_w, img_h) = img_wh
    if img_w == 0 or img_h == 0:
        img_w = 1
        img_h = 1
        msg = '[cc2.1] Your csv tables have an invalid ROI.'
        print(msg)
        #warnings.warn(msg)
        #ht = 1
        #wt = 1
    # Ensure ROI is within bounds
    (x, y, w, h) = bbox
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_w - 1)
    y2 = min(y + h, img_h - 1)
    return (x1, y1, x2, y2)


def build_reverse_mapping(uid_list, cluster_list):
    """
    Given a list of ids (uid_list) and a corresponding cluster index list
    (cluster_list), this builds a mapping from cluster index to uids
    """
    # Sort by clusterid for cache efficiency
    sortx = cluster_list.argsort()
    cluster_list = cluster_list[sortx]
    uid_list = uid_list[sortx]
    # Initialize dict of lists
    cluster2_uids = defaultdict(list)
    for uid, cluster in izip(uid_list, cluster_list):
        cluster2_uids[cluster].append(uid)
    return cluster2_uids


def unpack_items_sorted(dict_, sortfn, reverse=True):
    """ Unpacks and sorts the dictionary by sortfn """
    items = dict_.items()
    sorted_items = sorted(items, key=sortfn, reverse=reverse)
    sorted_keys, sorted_vals = list(izip(*sorted_items))
    return sorted_keys, sorted_vals


def unpack_items_sorted_by_lenvalue(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key """
    def sort_lenvalue(item):
        return len(item[1])
    return unpack_items_sorted(dict_, sort_lenvalue)


def unpack_items_sorted_by_value(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key """
    def sort_value(item):
        return item[1]
    return unpack_items_sorted(dict_, sort_value)


def flatten_membership_mapping(uid_list, members_list):
    num_members = sum(map(len, members_list))
    flat_uids = [None for _ in xrange(num_members)]
    flat_members = [None for _ in xrange(num_members)]
    count = 0
    for uid, members in izip(uid_list, members_list):
        for member in members:
            flat_uids[count]    = uid
            flat_members[count] = member
            count += 1
    return flat_uids, flat_members
