# -*- coding: utf-8 -*-
"""
custom sqlite3 module that supports numpy types
"""
from __future__ import absolute_import, division, print_function
import sys
import six
import io
import uuid
import numpy as np
import utool as ut
from six.moves import input

ut.noinject(__name__, '[dtool.__SQLITE__]')


VERBOSE_SQL = (
    '--veryverbose' in sys.argv or '--verbose' in sys.argv or '--verbsql' in sys.argv
)
TRY_NEW_SQLITE3 = False

# SQL This should be the only file which imports sqlite3
if not TRY_NEW_SQLITE3:
    from sqlite3 import Binary, register_adapter, register_converter
    from sqlite3 import *  # NOQA

# try:
#    # Try to import the correct version of sqlite3
#    if VERBOSE_SQL:
#        from pysqlite2 import dbapi2
#        import sqlite3
#        print('dbapi2.sqlite_version  = %r' % dbapi2.sqlite_version)
#        print('sqlite3.sqlite_version = %r' % sqlite3.sqlite_version)
#        print('using dbapi2 as lite')
#        # Clean namespace
#        del sqlite3
#        del dbapi2
#    if not TRY_NEW_SQLITE3:
#        raise ImportError('user wants python sqlite3')
#    from pysqlite2.dbapi2 import *  # NOQA
# except ImportError as ex:
#    if VERBOSE_SQL:
#        print(ex)
#    # Fallback
#    from sqlite3 import *  # NOQA
#    if VERBOSE_SQL:
#        print('using sqlite3 as lite')


def REGISTER_SQLITE3_TYPES():
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

    if six.PY2:

        def _write_numpy_to_sqlite3(arr):
            out = io.BytesIO()
            np.save(out, arr)
            out.seek(0)
            # return buffer(out.read())
            return Binary(out.read())

    else:

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
            print('WARNING: COULD NOT PARSE UUID %r, GIVING RANDOM' % (blob,))
            input('continue... [enter]')
            return uuid.uuid4()

    if six.PY2:

        def _write_uuid_to_sqlite3(uuid_):
            # return buffer(uuid_.bytes_le)
            return Binary(uuid_.bytes_le)

    elif six.PY3:

        def _write_uuid_to_sqlite3(uuid_):
            return memoryview(uuid_.bytes_le)

    def register_numpy_dtypes():
        if VERBOSE_SQL:
            print('Register NUMPY dtypes with SQLite3')

        py_int_type = long if six.PY2 else int  # NOQA
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

    def _read_dict_from_sqlite3(blob):
        return ut.from_json(blob)
        # return uuid.UUID(bytes_le=blob)

    def _write_dict_to_sqlite3(dict_):
        return ut.to_json(dict_)

    def register_numpy():
        """
        Tell SQL how to deal with numpy arrays
        Utility function allowing numpy arrays to be stored as raw blob data
        """
        if VERBOSE_SQL:
            print('Register NUMPY with SQLite3')
        register_converter('NUMPY', _read_numpy_from_sqlite3)
        register_converter('NDARRAY', _read_numpy_from_sqlite3)
        register_adapter(np.ndarray, _write_numpy_to_sqlite3)

    def register_uuid():
        """ Utility function allowing uuids to be stored in sqlite """
        if VERBOSE_SQL:
            print('Register UUID with SQLite3')
        register_converter('UUID', _read_uuid_from_sqlite3)
        register_adapter(uuid.UUID, _write_uuid_to_sqlite3)

    def register_dict():
        if VERBOSE_SQL:
            print('Register DICT with SQLite3')
        register_converter('DICT', _read_dict_from_sqlite3)
        register_adapter(dict, _write_dict_to_sqlite3)

    def register_list():
        if VERBOSE_SQL:
            print('Register LIST with SQLite3')
        register_converter('LIST', ut.from_json)
        register_adapter(list, ut.to_json)

    def register_bool():
        # FIXME: ensure this works
        if VERBOSE_SQL:
            print('Register BOOL with SQLite3')
        register_converter('BOOL', _read_bool)
        register_adapter(bool, _write_bool)

    register_numpy_dtypes()
    register_numpy()
    register_uuid()
    register_dict()
    register_list()
    # register_bool()  # TODO


REGISTER_SQLITE3_TYPES()


# def connect2(fpath, text_factory=None):
#    """ wrapper around lite.connect """
#    connection = connect(fpath, detect_types=PARSE_DECLTYPES)
#    return connection
#    #timeout=5,
#    # check_same_thread=False)
#    # isolation_level='DEFERRED',
#    # cached_statements=1000

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

if six.PY2:
    TYPE_TO_SQLTYPE[six.text_type] = 'TEXT'

# Clean namespace
del REGISTER_SQLITE3_TYPES
