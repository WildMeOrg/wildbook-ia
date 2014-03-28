from __future__ import division, print_function
# Python
from os.path import join, exists
import __SQLITE3__ as lite
import utool
from ibeis.dev import params
(print, print_, printDBG) = utool.inject_print_functions(__name__, '[sql]')


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
            Init the SQLite3 database connection and the execution object.
            If the database does not exist, it will be automatically created
            upon this object's instantiation.
        """
        printDBG('[sql.__init__]')
        # Get SQL file path
        db.dir_  = database_path
        db.fname = database_file
        assert exists(db.dir_), '[sql] db.dir_=%r does not exist!' % db.dir_
        fpath    = join(db.dir_, db.fname)
        # Open the SQL database connection with support for custom types
        db.connection = lite.connect(fpath, detect_types=lite.PARSE_DECLTYPES)
        db.executor   = db.connection.cursor()

    def get_sql_version(db):
        db.execute('''
                 SELECT sqlite_version()
                 ''', verbose=False)
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
        # print('[sql.schema] ensuring table=%r' % table)
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        body_list = ['%s %s' % (name, type_)
                     for (name, type_) in schema_list]
        op_head = 'CREATE TABLE IF NOT EXISTS %s (' % table
        op_body = ', '.join(body_list)
        op_foot = ')'
        operation = op_head + op_body + op_foot
        db.execute(operation, [], verbose=False)

    def execute(db, operation, parameters=(), auto_commit=False, errmsg=None,
                verbose=True):
        """
            operation - parameterized SQL operation string.
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
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.execute] caller_name=%r' % caller_name)
        status = False
        try:
            status = db.executor.execute(operation, parameters)
            if auto_commit:
                db.commit(verbose=False)
        except Exception as ex:
            print('[sql] Caught Exception: %r' % (ex,))
            status = True
            raise
        return status

    def executemany(db, operation, parameters_iter, auto_commit=True,
                    errmsg=None, verbose=True):
        """ same as execute but takes a iterable of parameters instead of just one
        This function is a bit messy right now. Needs cleaning up
        """
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.executemany] caller_name=%r' % caller_name)
        #import textwrap
        #operation = textwrap.dedent(operation).strip()
        try:
            # Format 1
            #qstat_flag_list = db.executor.executemany(operation, parameters_iter)

            # Format 2
            #qstat_flag_list = [db.executor.execute(operation, parameters)
                                 #for parameters in parameters_iter]

            # Format 3
            qstat_flag_list = []
            for parameters in parameters_iter:
                stat_flag = db.executor.execute(operation, parameters)
                qstat_flag_list.append(stat_flag)
        except Exception as ex1:
            print('\n<!!! ERROR>')
            print('[!sql] executemany threw %s: %r' % (type(ex1), ex1,))
            print('[!sql] operation=\n%s' % operation)
            if 'parameters' in vars():
                print('[!sql] failed paramters=%r' % (parameters,))
            else:
                print('[!!sql] failed before parameters populated')
            if 'qstat_flag_list' in vars():
                print('[!sql] failed qstat_flag_list=%r' % (qstat_flag_list,))
            else:
                print('[!sql] failed before qstat_flag_list populated')
            print('[!sql] parameters_iter=%r' % (parameters_iter,))
            print('</!!! ERROR>\n')
            db.dump()
            raise
            #raise lite.DatabaseError('%s --- %s' % (errmsg, ex1))

        try:
            if auto_commit:
                db.commit(qstat_flag_list, errmsg, verbose=False)
            else:
                return qstat_flag_list
        except Exception as ex2:
            print('\n<!!! ERROR>')
            print('[!sql] Caught %s: %r' % (type(ex2), ex2,))
            print('[!sql] operation=\n%s' % operation)
            print('</!!! ERROR>\n')
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex2))

    def result(db, verbose=True):
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.result] caller_name=%r' % caller_name)
        return db.executor.fetchone()

    def result_iter(db):
        # Jon: I think we should be using the fetchmany command here
        # White iteration is efficient, I believe it still interupts
        # the sql work. If we let sql work uninterupted by python it
        # should go faster

        # Jason: That's fine, it will just be a bigger memory footprint
        # Speed vs Footprint
        caller_name = utool.util_dbg.get_caller_name()
        print('[sql.result_iter] caller_name=%r' % caller_name)
        while True:
            result = db.result(verbose=False)
            if not result:
                raise StopIteration()
            yield result[0]

    def commit(db, qstat_flag_list=[], errmsg=None, verbose=True):
        """
            Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.commit] caller_name=%r' % caller_name)
        if not all(qstat_flag_list):
            raise lite.DatabaseError(errmsg)
        else:
            db.connection.commit()

            if params.args.auto_dump:
                db.dump(auto_commit=False)

    def dump(db, file_=None, auto_commit=True):
        """
            Same output as shell command below
            > sqlite3 database.sqlite3 .dump > database.dump.txt

            If file_=sys.stdout dumps to standard out

            This saves the current database schema structure and data into a
            text dump. The entire database can be recovered from this dump
            file. The default will store a dump parallel to the current
            database file.
        """
        if file_ is None or isinstance(file_, str):
            if file_ is None:
                dump_dir = db.dir_
                dump_fname = db.fname + '.dump.txt'
                dump_fpath = join(dump_dir, dump_fname)
            else:
                dump_fpath = file_
            with open(dump_fpath, 'w') as file_:
                db.dump(file_, auto_commit)
        else:
            print('[sql.dump]')
            if auto_commit:
                db.commit(verbose=False)
            for line in db.connection.iterdump():
                file_.write('%s\n' % line)
