from __future__ import division, print_function
# Python
import io
import textwrap  # NOQA
from os.path import join
# SQL This should be the only file which import sqlite3
try:
    # Try to import the correct version of sqlite3
    from pysqlite2 import dbapi2
    import sqlite3
    print('dbapi2.sqlite_version  = %r' % dbapi2.sqlite_version)
    print('sqlite3.sqlite_version = %r' % sqlite3.sqlite_version)
    print('using dbapi2 as lite')
    # Clean namespace
    del sqlite3
    del dbapi2
    from pysqlite2 import dbapi2 as lite
except ImportError as ex:
    print(ex)
    # Fallback
    import sqlite3 as lite
    print('using sqlite3 as lite')
# Science
import numpy as np


# Tell SQL how to deal with numpy arrays
def _REGISTER_NUMPY_WITH_SQLITE3():
    """ Utility function allowing numpy arrays to be stored as raw blob data """
    def _write_numpy_to_sqlite3(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return buffer(out.read())

    def _read_numpy_from_sqlite3(blob):
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)

    lite.register_adapter(np.ndarray, _write_numpy_to_sqlite3)
    lite.register_converter('NUMPY', _read_numpy_from_sqlite3)

_REGISTER_NUMPY_WITH_SQLITE3()

# Clean up namespace
del(_REGISTER_NUMPY_WITH_SQLITE3)


class SQLDatabaseControl(object):
    def __init__(db, database_path, database_file='database.sqlite3'):
        """
            SQLite3 Documentation: http://www.sqlite.org/docs.html
            -------------------------------------------------------
            SQL INSERT: http://www.w3schools.com/sql/sql_insert.asp
            SQL UPDATE: http://www.w3schools.com/sql/sql_update.asp
            SQL SELECT: http://www.w3schools.com/sql/sql_select.asp
            SQL DELETE: http://www.w3schools.com/sql/sql_delete.asp
            -------------------------------------------------------
            Init the SQLite3 database connection and the query execution object.
            If the database does not exist, it will be automatically created
            upon this object's instantiation.
        """
        print('[sql.__init__]')
        # Get SQL file path
        db.dir_  = database_path
        db.fname = database_file
        fpath    = join(db.dir_, db.fname)
        # Open the SQL database connection with support for custom types
        db.connection = lite.connect(fpath, detect_types=lite.PARSE_DECLTYPES)
        db.querier    = db.connection.cursor()

    def get_sql_version(db):
        db.query('''
                 SELECT sqlite_version()
                 ''')
        sql_version = db.result()

        print('[sql] SELECT sqlite_version = %r' % (sql_version,))
        # The version number sqlite3 module. NOT the version of SQLite library.
        print('[sql] sqlite3.version = %r' % (lite.version,))
        # The version of the SQLite library
        print('[sql] sqlite3.sqlite_version = %r' % (lite.sqlite_version,))
        return sql_version

    def schema(db, table, schema_list):
        """
            schema_dict - list of table columns tuples
                {
                    (column_1_name, column_1_type),
                    (column_2_name, column_2_type),
                    ...
                    (column_N_name, column_N_type),
                }
            ---------------------------------------------
            column_n_name - string name of column heading
            column_n_type - NULL | INTEGER | REAL | TEXT | BLOB | NUMPY
                The column type can be appended with ' PRIMARY KEY' to indicate
                the unique id for the table.  It can also specify a default
                value for the column with ' DEFAULT [VALUE]'.  It can also
                specify ' NOT NULL' to indicate the column cannot be empty.
            ---------------------------------------------
            The table will only be created if it does not exist.  Therefore,
            this can be done on every table without fear of deleting old data.
            ---------------------------------------------
            TODO: Add handling for column addition between software versions.
            Column deletions will not be removed from the database schema.
        """
        print('[sql.schema] ensuring table=%r' % table)
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        body_list = ['%s %s' % (name, type_)
                     for (name, type_) in schema_list]
        op_head = 'CREATE TABLE IF NOT EXISTS %s (' % table
        op_body = ', '.join(body_list)
        op_foot = ')'
        operation = op_head + op_body + op_foot
        db.query(operation, [])

    def query(db, operation, parameters=(), auto_commit=False):
        """
            operation - parameterized SQL query string.
                Parameterized prevents SQL injection attacks by using an ordered
                representation ( ? ) or by using an ordered, text representation
                name ( :value )

            parameters - list of values or a dictionary of representations and
                         corresponding values
                * Ordered Representation -
                    List of values in the order the question marks appear in the
                    sql operation string
                * Unordered Representation -
                    Dictionary of (text representation name -> value) in an
                    arbirtary order that will be filled into the cooresponging
                    slots of the sql operation string
        """
        print('[sql.query]')
        status = False
        try:
            status = db.querier.execute(operation, parameters)
            if auto_commit:
                db.commit()
        except Exception as ex:
            print('[sql] Caught Exception: %r' % (ex,))
            status = True
            raise
        return status

    def querymany(db, operation, parameters_iter, auto_commit=True, errmsg=None):
        """ same as query but takes a iterable of parameters instead of just one
        This function is a bit messy right now. Needs cleaning up
        """
        print('[sql.querymany]')
        #operation = textwrap.dedent(operation).strip()
        try:
            # Format 1
            #qstat_flag_list = db.querier.executemany(operation, parameters_iter)

            # Format 2
            #qstat_flag_list = [db.querier.execute(operation, parameters)
                                 #for parameters in parameters_iter]

            # Format 3
            qstat_flag_list = []
            for parameters in parameters_iter:
                stat_flag = db.querier.execute(operation, parameters)
                qstat_flag_list.append(stat_flag)
        except Exception as ex1:
            print('\n<!!!>')
            print('[!sql] Caught but cannot handle %s: %r' % (type(ex1), ex1,))
            print('[!sql] operation=\n%s' % operation)
            try:
                print('[!sql] failed paramters=%r' % (parameters,))
            except NameError:
                print('[!!sql] failed before parameters populated')
            try:
                print('[!sql] failed qstat_flag_list=%r' % (qstat_flag_list,))
            except NameError:
                print('[!sql] failed before qstat_flag_list populated')
            print('[!sql] parameters_iter=%r' % (parameters_iter,))
            print('</!!!>\n')
            db.dump()
            raise
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex1))

        try:
            if auto_commit:
                db.commit(qstat_flag_list, errmsg)
            else:
                return qstat_flag_list
        except Exception as ex2:
            print('\n<!!!>')
            print('[!sql] Caught %s: %r' % (type(ex2), ex2,))
            print('[!sql] operation=\n%s' % operation)
            print('</!!!>\n')
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex2))

    def result(db):
        print('[sql.result]')
        return db.querier.fetchone()

    def result_iter(db):
        # Jon: I think we should be using the fetchmany command here
        # White iteration is efficient, I believe it still interupts
        # the sql work. If we let sql work uninterupted by python it
        # should go faster
        print('[sql.result_iter]')
        while True:
            result = db.result()
            if not result:
                raise StopIteration()
            yield result[0]

    def commit(db, qstat_flag_list=[], errmsg=None):
        """
            Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        print('[sql.commit]')
        if not all(qstat_flag_list):
            raise lite.DatabaseError(errmsg)
        else:
            db.connection.commit()

    def dump(db, file_=None):
        """
            Same output as shell command below
            > sqlite3 database.sqlite3 .dump > database.dump.txt

            If file_=sys.stdout dumps to standard out

            This saves the current database schema structure and data into a
            text dump. The entire database can be recovered from this dump
            file. The default will store a dump parallel to the current
            database file.
        """
        if file_ is None:
            dump_dir = db.dir_
            dump_fname = db.fname + '.dump.txt'
            dump_fpath = join(dump_dir, dump_fname)
            with open(dump_fpath, 'w') as file_:
                db.dump(file_)
        else:
            print('[sql.dump]')
            db.commit()
            for line in db.connection.iterdump():
                file_.write('%s\n' % line)
