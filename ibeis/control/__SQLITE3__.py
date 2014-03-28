from __future__ import division, print_function
import sys
from utool.util_inject import inject_print_functions
print, print_, printDBG = inject_print_functions(__name__, '[SQLITE3]', DEBUG=False)

VERBOSE = '--verbose' in sys.argv


# SQL This should be the only file which imports sqlite3
try:
    # Try to import the correct version of sqlite3
    if VERBOSE:
        from pysqlite2 import dbapi2
        import sqlite3
        print('dbapi2.sqlite_version  = %r' % dbapi2.sqlite_version)
        print('sqlite3.sqlite_version = %r' % sqlite3.sqlite_version)
        print('using dbapi2 as lite')
        # Clean namespace
        del sqlite3
        del dbapi2
    from pysqlite2.dbapi2 import *  # NOQA
except ImportError as ex:
    if VERBOSE:
        print(ex)
    # Fallback
    from sqlite3 import *  # NOQA
    if VERBOSE:
        print('using sqlite3 as lite')


def REGISTER_SQLITE3_TYPES():
    import io
    import uuid
    import numpy as np
    def _read_numpy_from_sqlite3(blob):
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)

    def _write_numpy_to_sqlite3(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return buffer(out.read())

    def _read_uuid_from_sqlite3(blob):
        return uuid.UUID(bytes_le=blob)

    def _write_uuid_to_sqlite3(uuid_):
        return buffer(uuid_.bytes_le)

    # Tell SQL how to deal with numpy arrays
    def register_numpy():
        """ Utility function allowing numpy arrays to be stored as raw blob data """
        print('Register NUMPY with SQLite3')
        register_converter('NUMPY', _read_numpy_from_sqlite3)
        register_adapter(np.ndarray, _write_numpy_to_sqlite3)

    def register_uuid():
        """ Utility function allowing uuids to be stored in sqlite """
        print('Register UUID with SQLite3')
        register_converter('UUID', _read_uuid_from_sqlite3)
        register_adapter(uuid.UUID, _write_uuid_to_sqlite3)

    register_numpy()
    register_uuid()
REGISTER_SQLITE3_TYPES()

# Clean namespace
del REGISTER_SQLITE3_TYPES
