# -*- coding: utf-8 -*-
"""Mapping of Python types to SQL types"""
import json
import uuid

import numpy as np
from sqlalchemy.types import Integer as SAInteger
from sqlalchemy.types import TypeDecorator, UserDefinedType


__all__ = (
    'Dict',
    'Integer',
    'List',
    'TYPE_TO_SQLTYPE',
    'UUID',
)


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
