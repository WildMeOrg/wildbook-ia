# -*- coding: utf-8 -*-
"""Mapping of Python types to SQL types"""
import io
import uuid

import numpy as np
from utool.util_cache import from_json, to_json
import sqlalchemy
from sqlalchemy.sql import text
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
    postgresql_base_type = 'json'

    def get_col_spec(self, **kw):
        return self.col_spec

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            else:
                return to_json(value)

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            elif dialect.name == 'postgresql':
                # postgresql doesn't need the value to be json decoded
                return value
            else:
                return from_json(value)

        return process


class NumPyPicklableType(UserDefinedType):

    # Abstract properties
    base_py_types = None
    col_spec = None
    postgresql_base_type = 'bytea'

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
        if value is not None:
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
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)

            if dialect.name == 'sqlite':
                return value.bytes_le
            elif dialect.name == 'postgresql':
                return value
            else:
                if not isinstance(value, uuid.UUID):
                    return uuid.UUID(value).bytes_le
                else:
                    # hexstring
                    return value.bytes_le
                raise RuntimeError(f'Unknown dialect {dialect.name}')

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            if dialect.name == 'sqlite':
                if not isinstance(value, uuid.UUID):
                    return uuid.UUID(bytes_le=value)
                else:
                    return value
            elif dialect.name == 'postgresql':
                return value
            else:
                raise RuntimeError(f'Unknown dialect {dialect.name}')

        return process


_USER_DEFINED_TYPES = (Dict, List, NDArray, Number, UUID)
# SQL type (e.g. 'DICT') to SQLAlchemy type:
SQL_TYPE_TO_SA_TYPE = {cls().get_col_spec(): cls for cls in _USER_DEFINED_TYPES}
# Map postgresql types to SQLAlchemy types (postgresql type names are lowercase)
SQL_TYPE_TO_SA_TYPE.update(
    {cls().get_col_spec().lower(): cls for cls in _USER_DEFINED_TYPES}
)
SQL_TYPE_TO_SA_TYPE['INTEGER'] = Integer
SQL_TYPE_TO_SA_TYPE['integer'] = Integer
SQL_TYPE_TO_SA_TYPE['bigint'] = Integer


def initialize_postgresql_types(conn, schema):
    domain_names = conn.execute(
        """\
        SELECT domain_name FROM information_schema.domains
        WHERE domain_schema = (select current_schema)"""
    ).fetchall()
    for type_name, cls in SQL_TYPE_TO_SA_TYPE.items():
        if type_name not in domain_names and hasattr(cls, 'postgresql_base_type'):
            base_type = cls.postgresql_base_type
            try:
                conn.execute(f'CREATE DOMAIN {type_name} AS {base_type}')
            except sqlalchemy.exc.ProgrammingError:
                conn.execute(text('SET SCHEMA :schema'), schema=schema)
