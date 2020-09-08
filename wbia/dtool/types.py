# -*- coding: utf-8 -*-
"""Mapping of Python types to SQL types"""
import uuid

import numpy as np
from sqlalchemy.types import UserDefinedType


__all__ = (
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
