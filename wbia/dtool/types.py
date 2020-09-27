# -*- coding: utf-8 -*-
"""Mapping of Python types to SQL types"""
import io
import json
import uuid

import numpy as np
from sqlalchemy.types import Integer as SAInteger
from sqlalchemy.types import TypeDecorator, UserDefinedType


__all__ = (
    'Dict',
    'Integer',
    'List',
    'NDArray',
    'Number',
    'SQL_TYPE_TO_SA_TYPE',
    'UUID',
)

# DDD (26-Sept-12020) Deprecated in favor of SQL_TYPE_TO_SA_TYPE
TYPE_TO_SQLTYPE = {
    np.ndarray: 'NDARRAY',
    uuid.UUID: 'UUID',
    np.float32: 'REAL',
    np.float64: 'REAL',
    float: 'REAL',
    int: 'INTEGER',
    str: 'TEXT',
    # bool: 'BOOL',  # TODO
    bool: 'INTEGER',
    dict: 'DICT',
    list: 'LIST',
}


class JSONCodeableType(UserDefinedType):

    # Abstract properties
    base_py_type = None
    col_spec = None

    def get_col_spec(self, **kw):
        return self.col_spec

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            else:
                if isinstance(value, self.base_py_type):
                    return json.dumps(value)
                else:
                    return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            else:
                if not isinstance(value, self.base_py_type):
                    return json.loads(value)
                else:
                    return value

        return process


class NumPyPicklableType(UserDefinedType):

    # Abstract properties
    base_py_types = None
    col_spec = None

    def get_col_spec(self, **kw):
        return self.col_spec

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            else:
                if isinstance(value, self.base_py_types):
                    out = io.BytesIO()
                    np.save(out, value)
                    out.seek(0)
                    return out.read()
                else:
                    return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            else:
                if not isinstance(value, self.base_py_types):
                    out = io.BytesIO(value)
                    out.seek(0)
                    arr = np.load(out, allow_pickle=True)
                    out.close()
                    return arr
                else:
                    return value

        return process


class Dict(JSONCodeableType):
    base_py_type = dict
    col_spec = 'DICT'


class Integer(TypeDecorator):
    impl = SAInteger

    def process_bind_param(self, value, dialect):
        return int(value)


class List(JSONCodeableType):
    base_py_type = list
    col_spec = 'LIST'


class NDArray(NumPyPicklableType):
    base_py_types = (np.ndarray,)
    col_spec = 'NDARRAY'


NP_NUMBER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
)


class Number(NumPyPicklableType):
    base_py_types = NP_NUMBER_TYPES
    col_spec = 'NUMPY'


class UUID(UserDefinedType):
    def get_col_spec(self, **kw):
        return 'UUID'

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            else:
                if not isinstance(value, uuid.UUID):
                    return '%.32x' % uuid.UUID(value).int
                else:
                    # hexstring
                    return '%.32x' % value.int

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            else:
                if not isinstance(value, uuid.UUID):
                    return uuid.UUID(value)
                else:
                    return value

        return process


_USER_DEFINED_TYPES = (Dict, List, NDArray, Number, UUID)
# SQL type (e.g. 'DICT') to SQLAlchemy type:
SQL_TYPE_TO_SA_TYPE = {cls().get_col_spec(): cls for cls in _USER_DEFINED_TYPES}
