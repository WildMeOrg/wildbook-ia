from __future__ import absolute_import, division, print_function
import utool
import pandas as pd
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[pdh]')


def Int32Index(data, dtype=np.int32, copy=True, name=None, tupleize_cols=True):
    return pd.Index(data, dtype=dtype, copy=copy, name=name, tupleize_cols=tupleize_cols)

INTEGER_TYPE = np.int32
if INTEGER_TYPE is np.int32:
    IntIndex = Int32Index
else:
    IntIndex = pd.Int64Index


def IntSeries(data, *args, **kwargs):
    if 'index' not in kwargs:
        index = IntIndex(np.arange(len(data), dtype=INTEGER_TYPE))
        return pd.Series(data, *args, index=index, **kwargs)
    else:
        return pd.Series(data, *args, **kwargs)


def pandasify_dict1d(dict_, keys, val_name, series_name, dense=True):
    """ Turns dict into heirarchy of series """
    if dense:
        key2_series = pd.Series(
            {key: pd.Series(dict_.get(key, []), name=val_name,)
             for key in keys},
            index=keys, name=series_name)
    else:
        key2_series = pd.Series(
            {key: pd.Series(dict_.get(key), name=val_name,)
             for key in keys},
            index=IntIndex(dict_.keys(), name=keys.name), name=series_name)
    return key2_series


def pandasify_dict2d(dict_, keys, key2_index, columns, series_name):
    """ Turns dict into heirarchy of dataframes """
    key2_df = pd.Series(
        {key: pd.DataFrame(dict_[key], index=key2_index[key], columns=columns,)
         for key in keys},
        index=keys, name=series_name)
    return key2_df


def pandasify_list2d(list_, keys, columns, val_name, series_name):
    """ Turns dict into heirarchy of dataframes """
    key2_df = pd.Series(
        [pd.DataFrame(item,
                      index=IntIndex(np.arange(len(item)), name=val_name),
                      columns=columns,) for item in list_],
        index=keys, name=series_name)
    return key2_df


def ensure_numpy(data):
    return data.values if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) else data


def group_indicies(groupids):
    """
    >>> groupids = np.array(np.random.randint(0, 4, size=100))

    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
    """
    # Sort items and groupids by groupid
    sortx = groupids.argsort()
    groupids_sorted = groupids[sortx]
    num_items = groupids.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, groupids.dtype)
    diff[1:(num_items)] = np.diff(groupids_sorted)
    idxs = np.where(diff > 0)[0]
    num_groups = idxs.size - 1
    # Groups are between bounding indexes
    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T
    group_idxs = [sortx[lx:rx] for lx, rx in lrx_pairs]
    # Unique group keys
    keys = groupids_sorted[idxs[0:num_groups]]
    #items_sorted = items[sortx]
    #vals = [items_sorted[lx:rx] for lx, rx in lrx_pairs]
    return keys, group_idxs


def groupby(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    >>> items = groupids
    """
    keys, group_idxs = group_indicies(groupids)
    vals = [items[idxs] for idxs in group_idxs]
    return keys, vals


def groupby_gen(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    """
    for key, val in zip(*groupby(items, groupids)):
        yield (key, val)


def groupby_dict(items, groupids):
    # Build a dict
    grouped = {key: val for key, val in groupby_gen(items, groupids)}
    return grouped


