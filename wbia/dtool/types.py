# -*- coding: utf-8 -*-
"""Mapping of Python types to SQL types"""
import json
import uuid

import numpy as np
from sqlalchemy.types import UserDefinedType


__all__ = (
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


class List(UserDefinedType):
    def get_col_spec(self, **kw):
        return 'LIST'

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            else:
                if isinstance(value, list):
                    return json.dumps(value)
                else:
                    return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            else:
                if not isinstance(value, list):
                    return json.loads(value)
                else:
                    return value

        return process


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
