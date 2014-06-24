from __future__ import absolute_import, division, print_function
# Python
from itertools import imap, izip
import re
from os.path import join, exists
# Tools
import utool
from . import __SQLITE3__ as lite
DEBUG = False
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[sql]', DEBUG=DEBUG)


VERBOSE = utool.VERBOSE
PRINT_SQL = utool.get_flag('--print-sql')
AUTODUMP = utool.get_flag('--auto-dump')

QUIET = utool.QUIET or utool.get_flag('--quiet-sql')


def default_decorator(func):
    return func
    #return utool.indent_func('[sql.' + func.func_name + ']')(func)

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
            result = result[0] if len(result) == 1 else result
            #if get_last_id and result == 0:
            #    result = None
            yield result


def _unpacker(results_):
    """ HELPER: Unpacks results if unpack_scalars is True """
    results = None if len(results_) == 0 else results_[0]
    assert len(results_) < 2, 'throwing away results! { %r }' % (results_)
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
        #utool.printif(lambda: '[sql] Callers: ' + utool.get_caller_name(range(3, 6)), DEBUG)
        if context.num_params is None:
            context.operation_label = ('[sql] execute num_params=%d optype=%s: '
                                       % (context.num_params, context.operation_type))
        else:
            context.operation_label = '[sql] executeone optype=%s: ' % (context.operation_type)
        # Start SQL Transaction
        if context.start_transaction:
            _executor(context.db.executor, 'BEGIN', ())
        if PRINT_SQL:
            print(context.operation_label)
        # Comment out timeing code
        #if not QUIET:
        #    context.tt = utool.tic(context.operation_label)
        return context

    # --- with SQLExecutionContext: statment code happens here ---

    def execute_and_generate_results(context, params):
        """ HELPER FOR CONTEXT STATMENT """
        executor = context.db.executor
        operation = context.operation
        try:
            _executor(context.db.executor, operation, params)
            is_insert = context.operation_type.upper().startswith('INSERT')
            return _results_gen(executor, get_last_id=is_insert)
        except lite.IntegrityError:
            raise

    def __exit__(context, type_, value, trace):
        #if not QUIET:
        #    utool.tic(context.tt)
        if trace is None:
            # Commit the transaction
            if context.auto_commit:
                context.db.commit(verbose=False)
        else:
            # An SQLError is a serious offence.
            # Dump on error
            print('[sql] FATAL ERROR IN QUERY')
            context.db.dump()
            # utool.sys.exit(1)


def get_operation_type(operation):
    """
    Parses the operation_type from an SQL operation
    """
    operation = ' '.join(operation.split('\n')).strip()
    operation_type = operation.split(' ')[0].strip()
    if operation_type.startswith('SELECT'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('INSERT'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    elif operation_type.startswith('UPDATE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('DELETE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('CREATE'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
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
        db.table_columns = utool.odict()
        db.cache = {}
        db.stack = []
        db.table_constraints = utool.odict()

    #==============
    # CONVINENCE
    #==============

    @default_decorator
    def dump_tables_to_csv(db):
        """ Convenience: Dumps all csv database files to disk """
        dump_dir = join(db.dir_, 'CSV_DUMP')
        utool.ensuredir(dump_dir)
        for tablename in db.table_columns.iterkeys():
            table_fname = tablename + '.csv'
            table_csv = db.get_table_csv(tablename)
            with open(join(dump_dir, table_fname), 'w') as file_:
                file_.write(table_csv)

    @default_decorator
    def get_column_names(db, tablename):
        """ Conveinience: Returns the sql tablename columns """
        column_names = [name for name, type_ in  db.table_columns[tablename]]
        return column_names

    @default_decorator
    def get_table_names(db):
        """ Conveinience: """
        return db.table_columns.keys()

    def get_tables(db):
        #DEPRICATED
        return db.get_table_names()

    @default_decorator
    def get_column(db, tablename, name):
        """ Conveinience: """
        _table, (_column,) = db.sanatize_sql(tablename, (name,))
        column_vals = db.executeone(
            operation='''
            SELECT %s
            FROM %s
            ''' % (_column, _table))
        return column_vals

    @default_decorator
    def get_table_csv(db, tablename, exclude_columns=[]):
        """ Conveinience: Converts a tablename to csv format """
        column_nametypes = db.table_columns[tablename]
        column_names = [name for (name, type_) in column_nametypes]
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
        header = db.get_table_csv_header(tablename)
        csv_table = utool.make_csv_table(column_list, column_labels, header)
        return csv_table

    @default_decorator
    def get_table_csv_header(db, tablename):
        column_nametypes = db.table_columns[tablename]
        header_constraints = '# CONSTRAINTS: %r' % db.table_constraints[tablename]
        header_name  = '# TABLENAME: %r' % tablename
        header_types = utool.indentjoin(column_nametypes, '\n# ')
        header = header_name + header_types + '\n' + header_constraints
        return header

    def print_schema(db):
        for tablename in db.table_columns.iterkeys():
            print(db.get_table_csv_header(tablename) + '\n')

    @default_decorator
    def get_sql_version(db):
        """ Conveinience """
        _executor(db.executor, '''
                   SELECT sqlite_version()
                   ''', verbose=False)
        sql_version = db.executor.fetchone()

        print('[sql] SELECT sqlite_version = %r' % (sql_version,))
        # The version number sqlite3 module. NOT the version of SQLite library.
        print('[sql] sqlite3.version = %r' % (lite.version,))
        # The version of the SQLite library
        print('[sql] sqlite3.sqlite_version = %r' % (lite.sqlite_version,))
        return sql_version

    #==============
    # API INTERFACE
    #==============

    #@ider
    @default_decorator
    def _get_all_ids(db, tblname, **kwargs):
        """ valid ider """
        fmtdict = {
            'tblname': tblname,
        }
        operation_fmt = '''
        SELECT rowid FROM {tblname}
        ORDER BY rowid ASC
        '''
        return db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)

    #@ider
    @default_decorator
    def get_valid_ids(db, tblname, **kwargs):
        return db._get_all_ids(tblname, **kwargs)

    #@adder
    @default_decorator
    def add(db, tblname, colnames, params_iter, **kwargs):
        """ adder """
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        fmtdict = {
            'tblname'  : tblname,
            'questionmarks' : ', '.join(['?'] * len(colnames)),
            'params'   : ',\n'.join(colnames),
        }
        operation_fmt = utool.unindent('''
            INSERT INTO {tblname}(
            rowid,
            {params}
            ) VALUES (NULL, {questionmarks})
            ''')
        rowid_list = db._executemany_operation_fmt(operation_fmt, fmtdict,
                                                   params_iter=params_iter, **kwargs)
        return rowid_list

    #@adder
    @default_decorator
    def add_cleanly(db, tblname, colnames, params_iter, get_rowid_from_uuid):
        """
        Extra input:
            the first item of params_iter must be a uuid,
        uuid_list - a non-rowid column which identifies a row
            get_rowid_from_uuid - function which does what it says
        e.g:
            get_rowid_from_uuid = ibs.get_image_gids_from_uuid

        """
        # ADD_CLEANLY_1: PREPROCESS INPUT
        # eagerly evaluate for uuids
        params_list = list(params_iter)
        # Extract uuids from the params list (requires eager eval)
        # FIXME: the uuids being at index 0 is a hack
        uuid_list = [None if params is None else params[0] for params in params_list]
        # ADD_CLEANLY_2: PREFORM INPUT CHECKS
        # check which parameters are valid
        isvalid_list = [params is not None for params in params_list]
        # Check for duplicate inputs
        isunique_list = utool.flag_unique_items(uuid_list)
        # Check to see if this already exists in the database
        rowid_list_   = get_rowid_from_uuid(uuid_list)
        isnew_list    = [rowid is None for rowid in rowid_list_]
        if not all(isunique_list):
            print('[WARNING]: duplicate inputs to db.add_cleanly')
        # Flag each item that needs to added to the database
        isdirty_list = map(all, izip(isvalid_list, isunique_list, isnew_list))
        # ADD_CLEANLY_3.1: EXIT IF CLEAN
        if not any(isdirty_list):
            # There is nothing to add. Return the rowids
            return rowid_list_
        # ADD_CLEANLY_3.2: PERFORM DIRTY ADDITIONS
        # Add any unadded parameters to the database
        dirty_params = utool.filter_items(params_list, isdirty_list)
        print('[sql] adding %r/%r new %s' % (len(dirty_params), len(params_list), tblname))
        try:
            db.add(tblname, colnames, dirty_params)
        except Exception as ex:
            #unique_uuids = utool.unique_ordered(uuid_list)
            #assert len(unique_uuids) == len(uuid_list), 'duplicate inputs'
            #assert unique_uuids == uuid_list, 'duplicate inputs'
            #ibs = utool.search_stack_for_var('ibs')
            #print(ibs.get_valid_gids())
            utool.printex(ex, key_list=['isdirty_list', 'uuid_list', 'rowid_list_'])
            #utool.embed()
            raise
        #results =
        # If the result was already in the database (and ignored), it will return None.
        # Thus, go and get the row_id if the index is None
        #results = [get_rowid_from_uuid([uuid_list[index]])[0]
        #           if results[index] is None
        #           else results[index]
        #           for index in range(len(results))]
        rowid_list = get_rowid_from_uuid(uuid_list)
        # ADD_CLEANLY_4: SANITY CHECK AND RETURN
        assert len(rowid_list) == len(params_list)
        return rowid_list

    #@getter
    @default_decorator
    def get_where(db, tblname, colnames, params_iter, where_clause,
                  unpack_scalars=True, **kwargs):
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        fmtdict = {
            'tblname'     : tblname,
            'colnames'    : ', '.join(colnames),
            'where_clauses' : 'WHERE ' + where_clause,
        }
        operation_fmt = '''
            SELECT {colnames}
            FROM {tblname}
            {where_clauses}
            '''
        val_list = db._executemany_operation_fmt(operation_fmt, fmtdict,
                                                 params_iter=params_iter,
                                                 unpack_scalars=unpack_scalars,
                                                 **kwargs)
        return val_list

    #@getter
    @default_decorator
    def get(db, tblname, colnames, id_iter=None, id_colname='rowid',
                unpack_scalars=True, **kwargs):
        """ getter """
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        if unpack_scalars is None:
            unpack_scalars = id_colname is None
        assert unpack_scalars is not None, 'unpack_scalars is None'
        where_clause = (id_colname + '=?')
        params_iter = ((_rowid,) for _rowid in id_iter)

        return db.get_where(tblname, colnames, params_iter, where_clause, unpack_scalars=unpack_scalars)

    #@getter
    @default_decorator
    def get_executeone(db, tblname, colnames, **kwargs):
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        fmtdict = {
            'tblname'         : tblname,
            'colnames_str'    : ', '.join(colnames),
        }
        operation_fmt = '''
            SELECT {colnames_str}
            FROM {tblname}
            '''
        val_list = db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)
        return val_list

    #@getter
    @default_decorator
    def get_executeone_where(db, tblname, colnames, where_clause, params, **kwargs):
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        fmtdict = {
            'tblname'         : tblname,
            'colnames_str'    : ', '.join(colnames),
            'where_clause'    : where_clause
        }
        operation_fmt = '''
            SELECT {colnames_str}
            FROM {tblname}
            WHERE {where_clause}
            '''
        val_list = db._executeone_operation_fmt(operation_fmt, fmtdict, params=params, **kwargs)
        return val_list

    #@setter
    @default_decorator
    def set(db, tblname, colnames, val_list, id_list, id_colname='rowid', **kwargs):
        """ setter """
        #OFF printDBG('------------------------')
        #OFF printDBG('set_(table=%r, prop_key=%r)' % (table, prop_key))
        #OFF printDBG('set_(rowid_list=%r, val_list=%r)' % (rowid_list, val_list))
        #from operator import xor
        #assert not xor(utool.isiterable(rowid_list),
        #               utool.isiterable(val_list)), 'invalid mixing of iterable and scalar inputs'

        #if not utool.isiterable(rowid_list) and not utool.isiterable(val_list):
        #    rowid_list = (rowid_list,)
        #    val_list = (val_list,)
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        val_list = list(val_list)
        id_list = list(id_list)
        if not QUIET:
            print('[sql] SETTER: ' + utool.get_caller_name())
            print('[sql] * tblname=%r' % (tblname,))
            print('[sql] * val_list=%r' % (val_list,))
            print('[sql] * id_list=%r' % (id_list,))
            print('[sql] * id_colname=%r' % (id_colname,))
        assert  len(val_list) == len(id_list), 'list inputs have different lengths'
        fmtdict = {
            'tblname_str': tblname,
            'assign_str': ',\n'.join(['%s=?' % name for name in colnames]),
            'where_clause'   : (id_colname + '=?'),
        }
        operation_fmt = '''
            UPDATE {tblname_str}
            SET {assign_str}
            WHERE {where_clause}
            '''
        params_iter = utool.flattenize(izip(val_list, id_list))
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter, **kwargs)

    #@deleter
    @default_decorator
    def delete(db, tblname, id_list, id_colname=None, **kwargs):
        """ deleter """
        fmtdict = {
            'tblname' : tblname,
            'tblname': tblname,
            'rowid_str'   : ('rowid=?' if id_colname is None else id_colname + '=?'),
        }
        operation_fmt = '''
            DELETE
            FROM {tblname}
            WHERE {rowid_str}
            '''
        params_iter = ((_rowid,) for _rowid in id_list)
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter,
                                             **kwargs)

    #==============
    # CORE WRAPPERS
    #==============

    @default_decorator
    def _executeone_operation_fmt(db, operation_fmt, fmtdict, params=None):
        if params is None:
            params = []
        operation = operation_fmt.format(**fmtdict)
        return db.executeone(operation, params, auto_commit=True,  verbose=VERBOSE)

    @default_decorator
    def _executemany_operation_fmt(db, operation_fmt, fmtdict, params_iter, unpack_scalars=True):
        operation = operation_fmt.format(**fmtdict)
        return db.executemany(operation, params_iter, unpack_scalars=unpack_scalars,
                              auto_commit=True, verbose=VERBOSE)

    #=========
    # SQLDB CORE
    #=========

    @profile
    @default_decorator
    def sanatize_sql(db, tablename, columns=None):
        """ Sanatizes an sql tablename and column. Use sparingly """
        tablename = re.sub('[^a-z_0-9]', '', tablename)
        valid_tables = db.get_tables()
        if tablename not in valid_tables:
            raise Exception('UNSAFE TABLE: tablename=%r' % tablename)
        if columns is None:
            return tablename
        else:
            def _sanitize_sql_helper(column):
                column = re.sub('[^a-z_0-9]', '', column)
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

    @default_decorator
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
        db.table_constraints[tablename] = table_constraints

    @default_decorator
    def executeone(db, operation, params=(), auto_commit=True, verbose=VERBOSE):
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
                # utool.sys.exit(1)
                raise
        return result_list

    @default_decorator
    def executemany(db, operation, params_iter, auto_commit=True,
                    verbose=VERBOSE, unpack_scalars=True, num_params=None):
        """
        Input:
            operation - an sql command to be executed e.g.
                operation = '''
                SELECT colname
                FROM tblname
                WHERE
                (colname_1=?, ..., colname_N=?)
                '''
            params_iter - a sequence of params e.g.
                params_iter = [(col1, ..., colN), ..., (col1, ..., colN),]
        Output:
            results_iter - a sequence of data results
        """
        # Aggresively compute iterator if the num_params is not given
        if num_params is None:
            params_iter = list(params_iter)
            num_params  = len(params_iter)
        # Do not compute executemany without params
        if num_params == 0:
            utool.printif(lambda: '[sql!] WARNING: dont use executemany with no params use executeone instead.', VERBOSE)
            return []
        # Execute with context
        with SQLExecutionContext(db, operation, num_params,
                                 start_transaction=True) as context:
            try:
                # Execute each query and get results with a list comprehension
                # using the SQLExecutionContext (with eager evalutaion for now)
                results_iter = map(list, (context.execute_and_generate_results(params) for params in params_iter))
                if unpack_scalars:
                    results_iter = list(imap(_unpacker, results_iter))
                results_list = list(results_iter)  # Eager evaluation of results (for development)
            except Exception as ex:
                # Error reporting
                utool.printex(ex, key_list=[(str, 'operation'), 'num_params', 'params_iter'])
                raise
        return results_list

    @default_decorator
    def commit(db, qstat_flag_list=[],  verbose=VERBOSE, errmsg=None):
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

    @default_decorator
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
