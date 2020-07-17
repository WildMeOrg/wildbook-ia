# -*- coding: utf-8 -*-
"""Integrates numpy types into sqlite3"""
from __future__ import absolute_import, division, print_function
import io
import uuid
from sqlite3 import register_adapter, register_converter

import numpy as np
import utool as ut


__all__ = ()


def _read_numpy_from_sqlite3(blob):
    # INVESTIGATE: Is memory freed up correctly here?
    out = io.BytesIO(blob)
    out.seek(0)
    # return np.load(out)
    # Is this better?
    arr = np.load(out)
    out.close()
    return arr


def _read_bool(b):
    return None if b is None else bool(b)


def _write_bool(b):
    return b


def _write_numpy_to_sqlite3(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return memoryview(out.read())


def _read_uuid_from_sqlite3(blob):
    try:
        return uuid.UUID(bytes_le=blob)
    except ValueError as ex:
        ut.printex(ex, keys=['blob'])
        raise


def _read_dict_from_sqlite3(blob):
    return ut.from_json(blob)
    # return uuid.UUID(bytes_le=blob)


def _write_dict_to_sqlite3(dict_):
    return ut.to_json(dict_)


def _write_uuid_to_sqlite3(uuid_):
    return memoryview(uuid_.bytes_le)


def register_numpy_dtypes():
    py_int_type = int
    for dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        register_adapter(dtype, py_int_type)
    register_adapter(np.float32, float)
    register_adapter(np.float64, float)


def register_numpy():
    """
    Tell SQL how to deal with numpy arrays
    Utility function allowing numpy arrays to be stored as raw blob data
    """
    register_converter('NUMPY', _read_numpy_from_sqlite3)
    register_converter('NDARRAY', _read_numpy_from_sqlite3)
    register_adapter(np.ndarray, _write_numpy_to_sqlite3)


def register_uuid():
    """ Utility function allowing uuids to be stored in sqlite """
    register_converter('UUID', _read_uuid_from_sqlite3)
    register_adapter(uuid.UUID, _write_uuid_to_sqlite3)


def register_dict():
    register_converter('DICT', _read_dict_from_sqlite3)
    register_adapter(dict, _write_dict_to_sqlite3)


def register_list():
    register_converter('LIST', ut.from_json)
    register_adapter(list, ut.to_json)


# def register_bool():
#     # FIXME: ensure this works
#     register_converter('BOOL', _read_bool)
#     register_adapter(bool, _write_bool)


register_numpy_dtypes()
register_numpy()
register_uuid()
register_dict()
register_list()
# register_bool()  # TODO
