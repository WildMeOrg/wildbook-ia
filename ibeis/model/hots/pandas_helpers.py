from __future__ import absolute_import, division, print_function
import utool
import pandas as pd
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[pdh]')


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
            index=pd.Int64Index(dict_.keys(), name=keys.name), name=series_name)
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
                      index=pd.Int64Index(np.arange(len(item)), name=val_name),
                      columns=columns,) for item in list_],
        index=keys, name=series_name)
    return key2_df


def ensure_numpy(data):
    return data.values if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) else data
