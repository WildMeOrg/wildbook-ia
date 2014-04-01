from __future__ import division, print_function
# Python
from os.path import join, exists
import __SQLITE3__ as lite
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[sql]', DEBUG=False)


VERBOSE = utool.VERBOSE
AUTODUMP = utool.get_flag('--auto-dump')


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
        if not exists(fpath):
            print('[sql] Initializing new database')
        # Open the SQL database connection with support for custom types
        db.connection = lite.connect(fpath, detect_types=lite.PARSE_DECLTYPES)
        db.executor   = db.connection.cursor()
        db.table_columns = {}

    def get_column_names(db, table):
        """ Returns the sql table columns """
        column_names = [name for name, type_ in  db.table_columns[table]]
        return column_names

    def get_tables(db):
        return db.table_columns.keys()

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
            schema_list - list of table columns tuples
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
        if not utool.QUIET:
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
        db.execute(operation, [], verbose=False)
        # Append to internal storage
        db.table_columns[table] = schema_list

    def execute(db, operation, parameters=(), auto_commit=False, errmsg=None,
                verbose=VERBOSE):
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
                    errmsg=None, verbose=VERBOSE, unpack_scalars=True):
        """
        Input:
            operation - an sql command to be executed
                e.g.
                operation = '''
                SELECT column
                FROM table
                WHERE
                (
                    column_1=?,
                    ...,
                    column_N=?
                )
                '''
            parameter_list - an iterable of parameters
                e.g.
                parameter_list = [(col1, colN),
                                  ...,
                                   (col1, ... colN),
                                    ]


        same as execute but takes a iterable of parameters instead of just one
        This function is a bit messy right now. Needs cleaning up
        """
        caller_name = utool.util_dbg.get_caller_name()
        if errmsg is None:
            errmsg = '%s ERROR' % caller_name
        if verbose:
            print('[sql.executemany] caller_name=%r' % caller_name)
        # Do any preprocesing on the SQL command / query
        #import textwrap
        #operation = textwrap.dedent(operation).strip()
        operation_type = operation.split()[0].strip()
        result_list = []
        # Compute everything in Python before sending queries to SQL
        parameters_list = list(parameters_iter)
        num_params = len(parameters_list)
        # Define progress printing / logging / ... functions
        mark_prog, end_prog = utool.progress_func(
            max_val=num_params,
            lbl='[sql] execute %s: ' % operation_type)
        try:
            # For each parameter in an input list
            for count, parameters in enumerate(parameters_list):
                mark_prog(count)  # mark progress
                if verbose:
                    print('\n[sql] operation=\n%s' % operation)
                    print('[sql] paramters=%r' % (parameters,))
                # Send command to SQL
                # (all other results will be invalided)
                stat_flag  = db.executor.execute(operation,  # NOQA
                                                 parameters)
                def result_gen():
                    while True:
                        result = db.executor.fetchone()
                        if not result:
                            raise StopIteration()
                        yield result[0]
                # Read results
                resulttup = [result for result in result_gen()]
                # Append to the list of queries
                if len(resulttup) > 0 and unpack_scalars:
                    result_list.append(resulttup[0])
                else:
                    result_list.append(resulttup)
            end_prog()
            num_results = len(result_list)
            if num_results != 0 and num_results != num_params:
                raise lite.Error('num_params=%r <> num_results=%r' % (num_params, num_results))
        except lite.Error as ex1:
            print('\n<!!! ERROR>')
            print('[!sql] executemany threw %s: %r' % (type(ex1), ex1,))
            print('[!sql] %s' % (errmsg,))
            print('[!sql] operation=\n%s' % operation)
            if 'parameters' in vars():
                if len(parameters) > 4:
                    paraminfostr = utool.indentjoin(map(repr,
                                                        enumerate(
                                                            [(type(_), _) for _ in parameters])), '\n  ')
                    print('[!sql] failed paramters=' + paraminfostr)
                else:
                    print('[!sql] failed paramters=%r' % (parameters,))
            else:
                print('[!!sql] failed before parameters populated')
            print('[!sql] parameters_iter=%r' % (parameters_iter,))
            print('</!!! ERROR>\n')
            db.dump()
            raise
            #raise lite.DatabaseError('%s --- %s' % (errmsg, ex1))
        if auto_commit:
            db.commit(errmsg=errmsg, verbose=False)
        return result_list

    def result(db, verbose=VERBOSE):
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.result] caller_name=%r' % caller_name)
        return db.executor.fetchone()

    def result_list(db, verbose=VERBOSE):
        if verbose:
            caller_name = utool.util_dbg.get_caller_name()
            print('[sql.result_list] caller_name=%r' % caller_name)
        return list(db.result_iter())

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

    def commit(db, qstat_flag_list=[], errmsg=None, verbose=VERBOSE):
        """
            Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        try:
            if verbose:
                caller_name = utool.util_dbg.get_caller_name()
                print('[sql.commit] caller_name=%r' % caller_name)
            if not all(qstat_flag_list):
                raise lite.DatabaseError(errmsg)
            else:
                db.connection.commit()
                if AUTODUMP:
                    db.dump(auto_commit=False)
        except lite.Error as ex2:
            print('\n<!!! ERROR>')
            print('[!sql] Caught %s: %r' % (type(ex2), ex2,))
            caller_name = utool.util_dbg.get_caller_name()
            print('[!sql] caller_name=%r' % caller_name)
            print('</!!! ERROR>\n')
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex2))

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
