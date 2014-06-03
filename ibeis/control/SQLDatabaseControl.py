from __future__ import absolute_import, division, print_function
# Python
from itertools import imap
import re
from os.path import join, exists
# Tools
import utool
from . import __SQLITE3__ as lite
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[sql]', DEBUG=False)


VERBOSE = utool.VERBOSE
AUTODUMP = utool.get_flag('--auto-dump')

QUIET = utool.QUIET or utool.get_flag('--quiet-sql')


# =======================
# Helper Functions
# =======================


def _executor(executor, opeartion, params):
    """ HELPER: Send command to SQL (all other results are invalided)
    Execute an SQL Command
    """
    executor.execute(opeartion, params)


def _results_gen(executor, verbose=VERBOSE, get_last_id=False):
    """ HELPER - Returns as many results as there are.
    Careful. Overwrites the results once you call it.
    Basically: Dont call this twice.
    """
    if get_last_id:
        # The sqlite3_last_insert_rowid(D) interface returns the
        # <b> rowid of the most recent successful INSERT </b>
        # into a rowid table in D
        _executor(executor, 'SELECT last_insert_rowid()', ())
    # Wraping fetchone in a generator for some pretty tight calls.
    while True:
        result = executor.fetchone()
        if not result:
            raise StopIteration()
        else:
            # Results are always returned wraped in a tuple
            yield result[0] if len(result) == 1 else result


def _unpacker(results_):
    """ HELPER: Unpacks results if unpack_scalars is True """
    results = None if len(results_) == 0 else results_[0]
    assert len(results_) < 2, 'throwing away results!'
    return results


# =======================
# SQL Context Class
# =======================


class SQLExecutionContext(object):
    """ A good with context to use around direct sql calls
    """
    def __init__(context, db, operation, num_params=None, auto_commit=True,
                 start_transaction=False):
        context.auto_commit = auto_commit
        context.db = db  # Reference to sqldb
        context.operation = operation
        context.num_params = num_params
        context.start_transaction = start_transaction
        #context.__dict__.update(locals())  # Too mystic?
        context.operation_type = get_operation_type(operation)  # Parse the optype

    def __enter__(context):
        """ Checks to see if the operating will change the database """
        utool.printif(lambda:
                      '[sql] Callers: ' + utool.get_caller_name(range(3, 6)),
                      VERBOSE)
        # Mark if the database will change
        if any([context.operation_type.startswith(op) for op in
                ['INSERT', 'UPDATE', 'DELETE']]):
            context.db.about_to_change = False or context.db.about_to_change
        else:
            context.db.changed = True
            context.db.changed = False
        if context.num_params is None:
            context.operation_label = ('[sql] execute num_params=%d optype=%s: '
                                       % (context.num_params, context.operation_type))
        else:
            context.operation_label = '[sql] executeone optype=%s: ' % (context.operation_type)
        # Start SQL Transaction
        if context.start_transaction:
            context.db.executor.execute('BEGIN', ())
        # Comment out timeing code
        #if not QUIET:
        #    context.tt = utool.tic(context.operation_label)

    # --- with SQLExecutionContext: statment code happens here ---

    def execute_and_generate_results(context, params):
        """ HELPER FOR CONTEXT STATMENT """
        executor = context.db.executor
        operation = context.db.operation
        executor.execute(operation, params)
        is_insert = context.operation_type.upper().startswith('INSERT')
        return _results_gen(executor, get_last_id=is_insert)

    def __exit__(context, type_, value, trace):
        #if not QUIET:
        #    utool.tic(context.tt)
        if trace is None:
            # Commit the transaction
            if context.auto_commit:
                context.db.commit(verbose=False)
            # NO MORE CALLBACKS ????
            # Emit callback if changing
            #if context.changed:
            #    print('[sql] changed database')
            #    if context.db.dbchanged_callback is not None:
            #        context.db.dbchanged_callback()
        else:
            # An SQLError is a serious offence.
            # Dump on error
            print('[sql] FATAL ERROR IN QUERY')
            context.db.dump()
            utool.sys.exit(1)


def get_operation_type(operation):
    """
    Parses the operation_type from an SQL operation
    """
    operation = ' '.join(operation.split('\n').strip())
    operation_type = operation.split(' ')[0].strip()
    if operation_type.startswith('SELECT'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('INSERT'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    elif operation_type.startswith('UPDATE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('DELETE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    else:
        operation_args = None
    operation_type += ' ' + operation_args.replace('\n', ' ')
    return operation_type


class SQLDatabaseController(object):
    """ SQLDatabaseController an efficientish interface into SQL """

    def __init__(db, sqldb_dpath='.', sqldb_fname='database.sqlite3'):
        """ Creates db and opens connection

            SQLite3 Documentation: http://www.sqlite.org/docs.html
            -------------------------------------------------------
            SQL INSERT: http://www.w3schools.com/sql/sql_insert.asp
            SQL UPDATE: http://www.w3schools.com/sql/sql_update.asp
            SQL DELETE: http://www.w3schools.com/sql/sql_delete.asp
            SQL SELECT: http://www.w3schools.com/sql/sql_select.asp
            -------------------------------------------------------
            Init the SQLite3 database connection and the execution object.
            If the database does not exist, it will be automatically created
            upon this object's instantiation.
        """
        printDBG('[sql.__init__]')
        # Get SQL file path
        db.dir_  = sqldb_dpath
        db.fname = sqldb_fname
        assert exists(db.dir_), '[sql] db.dir_=%r does not exist!' % db.dir_
        fpath    = join(db.dir_, db.fname)
        if not exists(fpath):
            print('[sql] Initializing new database')
        # Open the SQL database connection with support for custom types
        db.connection = lite.connect(fpath, detect_types=lite.PARSE_DECLTYPES)
        db.executor   = db.connection.cursor()
        db.table_columns = {}
        db.about_to_change = False  # used by apitablemodel for cache invalidation
        db.dbchanged_callback = None

    def execute(db, operation, params=(), verbose=VERBOSE):
        """ DEPRICATE """
        db.executor.execute(operation, params)

    def result(db, verbose=VERBOSE):
        """ DEPRICATE """
        return db.executor.fetchone()

    def result_iter(db):
        """ DEPRICATE """
        return _results_gen(db.executor)

    def get_isdirty(db):
        """ DEPRICATE """
        return db.about_to_change

    def set_isdirty(db, flag):
        """ DEPRICATE """
        db.isdirty = flag

    def connect_dbchanged_callback(db, callback):
        """ DEPRICATE """
        db.dbchanged_callback = callback

    def disconnect_dbchanged_callback(db):
        """ DEPRICATE """
        db.dbchanged_callback = None

    def dump_tables_to_csv(db):
        """ Convenience: Dumps all csv database files to disk """
        dump_dir = join(db.dir_, 'CSV_DUMP')
        utool.ensuredir(dump_dir)
        for tablename in db.table_columns.iterkeys():
            table_fname = tablename + '.csv'
            table_csv = db.get_table_csv(tablename)
            with open(join(dump_dir, table_fname), 'w') as file_:
                file_.write(table_csv)

    def get_column_names(db, tablename):
        """ Conveinience: Returns the sql tablename columns """
        column_names = [name for name, type_ in  db.table_columns[tablename]]
        return column_names

    def get_tables(db):
        """ Conveinience: """
        return db.table_columns.keys()

    @profile
    def get_column(db, tablename, name):
        """ Conveinience: """
        _table, (_column,) = db.sanatize_sql(tablename, (name,))
        column_vals = db.executeone(
            operation='''
            SELECT %s
            FROM %s
            ''' % (_column, _table))
        return column_vals

    def get_table_csv(db, tablename, exclude_columns=[]):
        """ Conveinience: Converts a tablename to csv format """
        header_name  = '# TABLENAME: %r' % tablename
        column_nametypes = db.table_columns[tablename]
        column_names = [name for (name, type_) in column_nametypes]
        header_types = utool.indentjoin(column_nametypes, '\n# ')
        column_list = []
        column_labels = []
        for name in column_names:
            if name in exclude_columns:
                continue
            column_vals = db.get_column(tablename, name)
            column_list.append(column_vals)
            column_labels.append(name.replace(tablename[:-1] + '_', ''))
        # remove column prefix for more compact csvs

        #=None, column_list=[], header='', column_type=None
        header = header_name + header_types
        csv_table = utool.make_csv_table(column_list, column_labels, header)
        return csv_table

    def get_sql_version(db):
        """ Conveinience """
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

    #==============
    # API INTERFACE
    #==============

    def _executeone_operation_fmt(db, operation_fmt, fmtdict):
        operation = operation_fmt.format(**fmtdict)
        return db.executeone(operation, auto_commit=True, errmsg=None, verbose=VERBOSE)

    def _executemany_operation_fmt(db, operation_fmt, fmtdict):
        operation = operation_fmt.format(**fmtdict)
        return db.executemany(operation, auto_commit=True, errmsg=None, verbose=VERBOSE)

    #@ider
    def get_valid_ids(db, tblname, **kwargs):
        """ valid ider """
        fmtdict = {
            'tblname': tblname,
        }
        operation_fmt = '''
        SELECT rowid FROM {tblname}
        '''
        return db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)

    #@adder
    def add(db, tblname, colname_list, vals_iter, params_list, **kwargs):
        """ adder """
        fmtdict = {
            'tblname_str'  : tblname,
            'erotemes_str' : ','.join(['?'] * len(colname_list)),
            'adders_str'   : ',\n'.join(
                ['%s_%s = ?' % (tblname[:-1], name) for name in colname_list]),
        }
        operation_fmt = '''
            INSERT {tblname_str}
            (
            rowid=?,
            {adders_str}
            )
            VALUES (NULL, {erotemes_str})
            '''
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=vals_iter, **kwargs)

    #@getter
    def get(db, tblname, colnames, id_iter, unpack_scalars=True, **kwargs):
        """ getter """
        fmtdict = dict(tblname=tblname, colnames=colnames)
        operation_fmt = '''
            SELECT {colnames}
            FROM {tblname}
            WHERE rowid=?
            '''
        params_iter = ((_uid,) for _uid in id_iter)
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter, **kwargs)

    #@setter
    def set(db, tblname, colnames, id_list, val_iter, **kwargs):
        """ setter """
        fmtdict = {
            'tblname_str': tblname,
        }
        operation_fmt = '''
            UPDATE {tblname_str}
            SET {setter_str}
            WHERE rowid=?
            '''
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=val_iter, **kwargs)

    #@deleter
    def delete(db, tblname, colname, id_list, **kwargs):
        """ deleter """
        fmtdict = {}
        operation_fmt = '''
            DELETE
            FROM {tblname_str}
            WHERE {deleter_str}
            '''
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=id_list,
                                             **kwargs)

    #=========
    # API CORE
    #=========

    @profile
    def sanatize_sql(db, tablename, columns=None):
        """ Sanatizes an sql tablename and column. Use sparingly """
        tablename = re.sub('[^a-z_]', '', tablename)
        valid_tables = db.get_tables()
        if tablename not in valid_tables:
            raise Exception('UNSAFE TABLE: tablename=%r' % tablename)
        if columns is None:
            return tablename
        else:
            def _sanitize_sql_helper(column):
                column = re.sub('[^a-z_]', '', column)
                valid_columns = db.get_column_names(tablename)
                if column not in valid_columns:
                    raise Exception('UNSAFE COLUMN: tablename=%r column=%r' %
                                    (tablename, column))
                    return None
                else:
                    return column

            columns = [_sanitize_sql_helper(column) for column in columns]
            columns = [column for column in columns if columns is not None]

            return tablename, columns

    @profile
    def schema(db, tablename, schema_list, table_constraints=[]):
        """ Creates a table in the database with some schema and constraints

            schema_list - list of tablename columns tuples
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
                the unique id for the tablename.  It can also specify a default
                value for the column with ' DEFAULT [VALUE]'.  It can also
                specify ' NOT NULL' to indicate the column cannot be empty.
            ---------------------------------------------
            The tablename will only be created if it does not exist.  Therefore,
            this can be done on every tablename without fear of deleting old data.
            ---------------------------------------------
            TODO: Add handling for column addition between software versions.
            Column deletions will not be removed from the database schema.
        """
        printDBG('[sql] schema ensuring tablename=%r' % tablename)
        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        body_list = ['%s %s' % (name, type_)
                     for (name, type_) in schema_list]
        op_head = 'CREATE TABLE IF NOT EXISTS %s (' % tablename
        op_body = ', '.join(body_list + table_constraints)
        op_foot = ')'
        operation = op_head + op_body + op_foot
        db.executeone(operation, [], verbose=False)
        # Append to internal storage
        db.table_columns[tablename] = schema_list

    @profile
    def executeone(db, operation, params=(), auto_commit=True, errmsg=None, verbose=VERBOSE):
        """
            operation - parameterized SQL operation string.
                Parameterized prevents SQL injection attacks by using an ordered
                representation ( ? ) or by using an ordered, text representation
                name ( :value )

            params - list of values or a dictionary of representations and
                         corresponding values
                * Ordered Representation -
                    List of values in the order the question marks appear in the
                    sql operation string
                * Unordered Representation -
                    Dictionary of (text representation name -> value) in an
                    arbirtary order that will be filled into the cooresponging
                    slots of the sql operation string
        """
        with SQLExecutionContext(db, operation, num_params=1) as context:
            try:
                result_iter = context.execute_and_generate_results(params)
                result_list = list(result_iter)
            except Exception as ex:
                utool.printex(ex, key_list=[(str, 'operation'), 'params'])
                utool.sys.exit(1)
                raise
        return result_list

    @profile
    def executemany(db, operation, params_iter, auto_commit=True, errmsg=None,
                    verbose=VERBOSE, unpack_scalars=True, num_params=None):
        """
        Input:
            operation - an sql command to be executed e.g.
                operation = '''
                SELECT colname
                FROM tblname
                WHERE
                (
                    colname_1=?,
                    ...,
                    colname_N=?
                )
                '''
            params_iter - a sequence of params e.g.
                params_iter = [
                    (col1, ..., colN),
                           ...,
                    (col1, ..., colN),
                ]
        Output:
            results_iter - a sequence of data results

            FOR AN INSERT STATEMENT
                [

            FOR AN UPDATE STATEMENT

            FOR A  SELECT STATEMENT

            FOR A  DELETE STATEMENT


        if unpack_scalars is True results are returned as:


        same as execute but takes a iterable of params instead of just one
        This function is a bit messy right now. Needs cleaning up
        """
        # Aggresively compute iterator if the num_params is not given
        def _prepare(num_params, params_iter):
            if num_params is None:
                params_iter = list(params_iter)
                num_params  = len(params_iter)
            return num_params, params_iter
        num_params, params_iter = _prepare(num_params, params_iter)
        # Do not compute executemany without params
        if num_params == 0:
            utool.printif(lambda: utool.unindent(
                '''
                [sql!] WARNING: dont use executemany with no params use executeone instead.
                '''), VERBOSE)
            return []
        with SQLExecutionContext(db, operation, num_params,
                                 start_transaction=True) as context:
            try:
                # Python list-comprehension magic.
                results_iter = map(list, (
                    context.execute_and_generate_results(
                        db.executor, operation, params)
                    for params in params_iter))
                if unpack_scalars:
                    results_iter = list(imap(_unpacker, results_iter))
                # Eager evaluation of results (for development)
                results_list = list(results_iter)
            except Exception as ex:
                # Error reporting
                utool.printex(ex, key_list=[(str, 'operation'), 'params', 'params_iter'])
                print(utool.get_caller_name(range(1, 10)))
                results_list = None
                raise
        return results_list

    @profile
    def commit(db, qstat_flag_list=[], errmsg=None, verbose=VERBOSE):
        """ Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        try:
            if not all(qstat_flag_list):
                raise lite.DatabaseError(errmsg)
            else:
                db.connection.commit()
                if AUTODUMP:
                    db.dump(auto_commit=False)
        except lite.Error as ex2:
            print('\n<!!! ERROR>')
            utool.printex(ex2, '[!sql] Caught ex2=')
            caller_name = utool.util_dbg.get_caller_name()
            print('[!sql] caller_name=%r' % caller_name)
            print('</!!! ERROR>\n')
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex2))

    @profile
    def dump(db, file_=None, auto_commit=True):
        """ Same output as shell command below
            > sqlite3 database.sqlite3 .dump > database.dump.txt

            If file_=sys.stdout dumps to standard out

            This saves the current database schema structure and data into a
            text dump. The entire database can be recovered from this dump
            file. The default will store a dump parallel to the current
            database file.
        """
        if file_ is None or isinstance(file_, (str, unicode)):
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
