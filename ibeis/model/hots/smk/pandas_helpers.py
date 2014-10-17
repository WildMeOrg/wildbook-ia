from __future__ import absolute_import, division, print_function
import utool
import pandas as pd
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[pdh]')
from ibeis.model.hots.hstypes import VEC_DIM, INTEGER_TYPE


class LazyGetter(object):

    def __init__(self, getter_func):
        self.getter_func = getter_func

    def __getitem__(self, index):
        return self.getter_func(index)

    def __call__(self, index):
        return self.getter_func(index)


#def lazy_getter(getter_func):
#    def lazy_closure(*args):
#        return getter_func(*args)
#    return lazy_closure


class DataFrameProxy(object):
    """
    pandas is actually really slow. This class emulates it so
    I don't have to change my function calls, but without all the slowness.
    """

    def __init__(self, ibs):
        self.ibs = ibs

    def __getitem__(self, key):
        if key == 'kpts':
            return LazyGetter(self.ibs.get_annot_kpts)
        elif key == 'vecs':
            return LazyGetter(self.ibs.get_annot_desc)
        elif key == 'labels':
            return LazyGetter(self.ibs.get_annot_class_labels)


@profile
def Int32Index(data, dtype=np.int32, copy=True, name=None):
    return pd.Index(data, dtype=dtype, copy=copy, name=name)

if INTEGER_TYPE is np.int32:
    IntIndex = Int32Index
else:
    IntIndex = pd.Int64Index


@profile
def RangeIndex(size, name=None):
    arr = np.arange(size, dtype=INTEGER_TYPE)
    index = IntIndex(arr, copy=False, name=name)
    return index


VEC_COLUMNS  = RangeIndex(VEC_DIM, name='vec')
KPT_COLUMNS = pd.Index(['xpos', 'ypos', 'a', 'c', 'd', 'theta'], name='kpt')
PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Index)


@profile
def IntSeries(data, *args, **kwargs):
    if 'index' not in kwargs:
        index = IntIndex(np.arange(len(data), dtype=INTEGER_TYPE))
        return pd.Series(data, *args, index=index, **kwargs)
    else:
        return pd.Series(data, *args, **kwargs)


@profile
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


@profile
def pandasify_dict2d(dict_, keys, key2_index, columns, series_name):
    """ Turns dict into heirarchy of dataframes """
    item_list  = [dict_[key] for key in keys]
    index_list = [key2_index[key] for key in keys]
    _data = {
        key: pd.DataFrame(item, index=index, columns=columns,)
        for key, item, index in zip(keys, item_list, index_list)
    }
    key2_df = pd.Series(_data, index=keys, name=series_name)
    return key2_df


@profile
def pandasify_list2d(item_list, keys, columns, val_name, series_name):
    """ Turns dict into heirarchy of dataframes """
    index_list = [RangeIndex(len(item), name=val_name) for item in item_list]
    _data = [pd.DataFrame(item, index=index, columns=columns,)
             for item, index in zip(item_list, index_list)]
    key2_df = pd.Series(_data, index=keys, name=series_name)
    return key2_df


@profile
def ensure_values(data):
    if isinstance(data, (np.ndarray, list)):
        return data
    elif isinstance(data, PANDAS_TYPES):
        return data.values
    elif isinstance(data, dict):
        return np.array(list(data.values()))
    else:
        raise AssertionError(type(data))


@profile
def ensure_index(data):
    if isinstance(data, PANDAS_TYPES):
        return data.index
    elif isinstance(data, dict):
        return np.array(list(data.keys()))
    else:
        return np.arange(len(data))
        #raise AssertionError(type(data))


def ensure_values_subset(data, keys):
    if isinstance(data, dict):
        return [data[key] for key in keys]
    elif isinstance(data, PANDAS_TYPES):
        return [ensure_values(item) for item in data[keys].values]
    else:
        raise AssertionError(type(data))


def ensure_values_scalar_subset(data, keys):
    if isinstance(data, dict):
        return [data[key] for key in keys]
    elif isinstance(data, PANDAS_TYPES):
        return [item for item in data[keys].values]
    else:
        raise AssertionError(type(data))


def ensure_2d_values(data):
    #if not isinstance(data, PANDAS_TYPES):
    #    return data
    data_ = ensure_values(data)
    if len(data_) == 0:
        return data_
    else:
        if isinstance(data_[0], PANDAS_TYPES):
            return [item.values for item in data]
        else:
            raise AssertionError(type(data))


@profile
def pandasify_rvecs_list(wx_sublist, wx2_idxs_values, rvecs_list, aids_list,
                         fxs_list):
    assert len(rvecs_list) == len(wx2_idxs_values)
    assert len(rvecs_list) == len(wx_sublist)
    rvecsdf_list = [
        pd.DataFrame(rvecs, index=idxs, columns=VEC_COLUMNS)
        for rvecs, idxs in zip(rvecs_list, wx2_idxs_values)]  # 413 ms
    _aids_list = [pd.Series(aids) for aids in aids_list]
    wx2_rvecs = IntSeries(rvecsdf_list, index=wx_sublist, name='rvec')
    wx2_aids  = IntSeries(_aids_list, index=wx_sublist, name='wx2_aids')
    wx2_fxs   = IntSeries(fxs_list, index=wx_sublist, name='wx2_aids')
    return wx2_rvecs, wx2_aids, wx2_fxs


@profile
def pandasify_agg_list(wx_sublist, aggvecs_list, aggaids_list, aggfxs_list):
    """
    Example:
        >>> from ibeis.model.hots.smk.pandas_helpers import *
    """
    _aids_list    = [IntSeries(aids, name='aids') for aids in aggaids_list]
    _aggvecs_list = [pd.DataFrame(vecs, index=aids, columns=VEC_COLUMNS)
                     for vecs, aids in zip(aggvecs_list, _aids_list)]
    wx2_aggaids = IntSeries(_aids_list, index=wx_sublist, name='wx2_aggaids')
    wx2_aggvecs = pd.Series(_aggvecs_list, index=wx_sublist, name='wx2_aggvecs')
    wx2_aggfxs = pd.Series(aggfxs_list, index=wx_sublist, name='wx2_aggfxs')
    return wx2_aggvecs, wx2_aggaids, wx2_aggfxs
