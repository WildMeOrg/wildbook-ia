from __future__ import absolute_import, division, print_function
import utool
import pandas as pd
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[pdh]')
from ibeis.model.hots.smk.hstypes import VEC_DIM, INTEGER_TYPE


def Int32Index(data, dtype=np.int32, copy=True, name=None):
    return pd.Index(data, dtype=dtype, copy=copy, name=name)

if INTEGER_TYPE is np.int32:
    IntIndex = Int32Index
else:
    IntIndex = pd.Int64Index


def RangeIndex(size, name=None):
    arr = np.arange(size, dtype=INTEGER_TYPE)
    index = IntIndex(arr, copy=False, name=name)
    return index


VEC_COLUMNS  = RangeIndex(VEC_DIM, name='vec')
KPT_COLUMNS = pd.Index(['xpos', 'ypos', 'a', 'c', 'd', 'theta'], name='kpt')


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
    _data = {
        key: pd.DataFrame(dict_[key], index=key2_index[key], columns=columns,)
        for key in keys
    }
    key2_df = pd.Series(_data, index=keys, name=series_name)
    return key2_df


def pandasify_list2d(list_, keys, columns, val_name, series_name):
    """ Turns dict into heirarchy of dataframes """
    _data = [
        pd.DataFrame(item, index=IntIndex(np.arange(len(item)), name=val_name), columns=columns,)
        for item in list_]
    key2_df = pd.Series(_data, index=keys, name=series_name)
    return key2_df


def ensure_numpy(data):
    return data.values if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) else data


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


def pandasify_agg_list(wx_sublist, aggvecs_list, aggaids_list, aggfxs_list):
    """
    from ibeis.model.hots.smk.pandas_helpers import *
    """
    _aids_list    = [IntSeries(aids, name='aids') for aids in aggaids_list]
    _aggvecs_list = [pd.DataFrame(vecs, index=aids, columns=VEC_COLUMNS)
                     for vecs, aids in zip(aggvecs_list, _aids_list)]
    wx2_aggaids = IntSeries(_aids_list, index=wx_sublist, name='wx2_aggaids')
    wx2_aggvecs = pd.Series(_aggvecs_list, index=wx_sublist, name='wx2_aggvecs')
    wx2_aggfxs = pd.Series(aggfxs_list, index=wx_sublist, name='wx2_aggfxs')
    return wx2_aggvecs, wx2_aggaids, wx2_aggfxs
