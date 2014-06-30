from __future__ import absolute_import, division, print_function
# Python
from itertools import imap, izip
from os.path import join, exists
import utool
# Tools
#from ibeis.control._sql_database_control_helpers import *  # NOQA
#from ibeis.control import _sql_database_control_helpers as sqlhelpers
from ibeis.control._sql_helpers import (_unpacker, _executor, lite,
                                        sanatize_sql, SQLExecutionContext,
                                        default_decorator, DEBUG)


(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[sql]', DEBUG=DEBUG)

VERBOSE = utool.VERBOSE
QUIET = utool.QUIET or utool.get_flag('--quiet-sql')
AUTODUMP = utool.get_flag('--auto-dump')

""" If would be really great if we could get a certain set of setters, getters,
and deleters to be indexed into only with rowids. If we could do this than for
that subset of data, we could hook up a least recently used cache which is
populated whenever you get data from some table/colname using the rowid as the
key. The cache would then only have to be invalidated if we were going to set /
get data from that same rowid.  This would offer big speadups for both the
recognition algorithm and the GUI. """

from functools import wraps


def common_decor(func):
    @wraps(func)
    def closure_common(db, *args, **kwargs):
        return func(db, *args, **kwargs)
    return default_decorator(closure_common)


def getter_sql(func):
    @wraps(func)
    def closure_getter(db, *args, **kwargs):
        return func(db, *args, **kwargs)
    return common_decor(func)


def adder_sql(func):
    return common_decor(func)


def setter_sql(func):
    return common_decor(func)


def deleter_sql(func):
    return common_decor(func)


def ider_sql(func):
    return common_decor(func)


class SQLDatabaseController(object):
    """ SQLDatabaseController an efficientish interface into SQL """

    def __init__(db, sqldb_dpath='.', sqldb_fname='database.sqlite3',
                 text_factory=unicode):
        """ Creates db and opens connection """
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
        db.connection.text_factory = text_factory
        # Get a cursor which will preform sql commands / queries / executions
        db.cur = db.connection.cursor()
        # Optimize the database (if anything is set)
        db.optimize()
        # Table info
        db.table_columns     = utool.odict()
        db.table_constraints = utool.odict()
        db.table_docstr      = utool.odict()
        # TODO:
        db.stack = []
        db.cache = {}  # key \in [tblname][colnames][rowid]

    #==============
    # API INTERFACE
    #==============

    @ider_sql
    def get_all_rowids(db, tblname):
        """ returns a list of all rowids from a table in ascending order """
        fmtdict = {'tblname': tblname, }
        operation_fmt = '''
        SELECT rowid
        FROM {tblname}
        ORDER BY rowid ASC
        '''
        return db._executeone_operation_fmt(operation_fmt, fmtdict)

    @ider_sql
    def get_all_rowids_where(db, tblname, where_clause, params, **kwargs):
        """ returns a list of rowids from a table in ascending order satisfying
        a condition """
        fmtdict = {'tblname': tblname, 'where_clause': where_clause, }
        operation_fmt = '''
        SELECT rowid
        FROM {tblname}
        WHERE {where_clause}
        ORDER BY rowid ASC
        '''
        return db._executeone_operation_fmt(operation_fmt, fmtdict, params, **kwargs)

    def _add(db, tblname, colnames, params_iter, **kwargs):
        """ ADDER NOTE: use add_cleanly """
        fmtdict = {'tblname'  : tblname,
                    'erotemes' : ', '.join(['?'] * len(colnames)),
                    'params'   : ',\n'.join(colnames), }
        operation_fmt = '''
        INSERT INTO {tblname}(
        rowid,
        {params}
        ) VALUES (NULL, {erotemes})
        '''
        rowid_list = db._executemany_operation_fmt(operation_fmt, fmtdict,
                                                   params_iter=params_iter, **kwargs)
        return rowid_list

    @adder_sql
    def add_cleanly(db, tblname, colnames, params_iter, get_rowid_from_uuid, unique_paramx=[0]):
        """ ADDER Extra input: the first item of params_iter must be a uuid, """
        # ADD_CLEANLY_1: PREPROCESS INPUT
        params_list = list(params_iter)  # eagerly evaluate for uuids
        # Extract uuids from the params list (requires eager eval)
        uuid_lists = [[None if params is None else params[x]
                       for params in params_list]
                      for x in unique_paramx]
        # ADD_CLEANLY_2: PREFORM INPUT CHECKS
        # check which parameters are valid
        isvalid_list = [params is not None for params in params_list]
        # Check for duplicate inputs
        isunique_list = utool.flag_unique_items(list(izip(*uuid_lists)))
        # Check to see if this already exists in the database
        rowid_list_   = get_rowid_from_uuid(*uuid_lists)
        isnew_list    = [rowid is None for rowid in rowid_list_]
        if VERBOSE and not all(isunique_list):
            print('[WARNING]: duplicate inputs to db.add_cleanly')
        # Flag each item that needs to added to the database
        isdirty_list = map(all, izip(isvalid_list, isunique_list, isnew_list))
        # ADD_CLEANLY_3.1: EXIT IF CLEAN
        if not any(isdirty_list):
            return rowid_list_  # There is nothing to add. Return the rowids
        # ADD_CLEANLY_3.2: PERFORM DIRTY ADDITIONS
        # Add any unadded parameters to the database
        dirty_params = utool.filter_items(params_list, isdirty_list)
        print('[sql] adding %r/%r new %s' % (len(dirty_params), len(params_list), tblname))
        try:
            db._add(tblname, colnames, dirty_params)
        except Exception as ex:
            utool.printex(ex, key_list=['isdirty_list', 'uuid_list', 'rowid_list_'])
            raise
        # TODO: We should only have to preform a subset of adds here
        # (at the positions where rowid_list was None in the getter check)
        rowid_list = get_rowid_from_uuid(*uuid_lists)
        # ADD_CLEANLY_4: SANITY CHECK AND RETURN
        assert len(rowid_list) == len(params_list), 'failed sanity check'
        return rowid_list

    @getter_sql
    def get_where(db, tblname, colnames, params_iter, where_clause, unpack_scalars=True, **kwargs):
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        fmtdict = { 'tblname'     : tblname,
                    'colnames'    : ', '.join(colnames),
                    'where_clauses' :  where_clause, }
        operation_fmt = '''
        SELECT {colnames}
        FROM {tblname}
        WHERE {where_clauses}
        '''
        val_list = db._executemany_operation_fmt(operation_fmt, fmtdict,
                                                 params_iter=params_iter,
                                                 unpack_scalars=unpack_scalars,
                                                 **kwargs)
        return val_list

    @getter_sql
    def get_rowid_from_superkey(db, tblname, params_iter=None, superkey_colnames=None, **kwargs):
        """ getter which uses the constrained superkeys instead of rowids """
        where_clause = 'AND'.join([colname + '=?' for colname in superkey_colnames])
        return db.get_where(tblname, ('rowid',), params_iter, where_clause, **kwargs)

    @getter_sql
    def get(db, tblname, colnames, id_iter=None, id_colname='rowid', **kwargs):
        """ getter """
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        where_clause = (id_colname + '=?')
        params_iter = ((_rowid,) for _rowid in id_iter)

        return db.get_where(tblname, colnames, params_iter, where_clause, **kwargs)

    @setter_sql
    def set(db, tblname, colnames, val_list, id_list, id_colname='rowid', **kwargs):
        """ setter """
        if isinstance(colnames, (str, unicode)):
            colnames = (colnames,)
        val_list = list(val_list)  # eager evaluation
        id_list = list(id_list)  # eager evaluation
        if not QUIET and VERBOSE:
            print('[sql] SETTER: ' + utool.get_caller_name())
            print('[sql] * tblname=%r' % (tblname,))
            print('[sql] * val_list=%r' % (val_list,))
            print('[sql] * id_list=%r' % (id_list,))
            print('[sql] * id_colname=%r' % (id_colname,))
        try:
            num_val = len(val_list)
            num_id = len(id_list)
            assert num_val == num_id, 'list inputs have different lengths'
        except AssertionError as ex:
            utool.printex(ex, key_list=['num_val', 'num_id'])
            raise
        fmtdict = {
            'tblname_str'  : tblname,
            'assign_str'   : ',\n'.join(['%s=?' % name for name in colnames]),
            'where_clause' : (id_colname + '=?'),
        }
        operation_fmt = '''
            UPDATE {tblname_str}
            SET {assign_str}
            WHERE {where_clause}
            '''
        #params_iter = utool.flattenize(izip(val_list, id_list))
        params_iter = utool.flattenize(izip(val_list, id_list))
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter, **kwargs)

    @deleter_sql
    def delete(db, tblname, id_list, id_colname='rowid', **kwargs):
        """ deleter. USE delete_rowids instead """
        fmtdict = {
            'tblname'   : tblname,
            'rowid_str' : (id_colname + '=?'),
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

    @deleter_sql
    def delete_rowids(db, tblname, rowid_list, **kwargs):
        """ deletes the the rows in rowid_list """
        fmtdict = {
            'tblname'   : tblname,
            'rowid_str' : ('rowid=?'),
        }
        operation_fmt = '''
            DELETE
            FROM {tblname}
            WHERE {rowid_str}
            '''
        params_iter = ((_rowid,) for _rowid in rowid_list)
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter,
                                             **kwargs)

    #==============
    # CORE WRAPPERS
    #==============

    @default_decorator
    def _executeone_operation_fmt(db, operation_fmt, fmtdict, params=None, **kwargs):
        if params is None:
            params = []
        operation = operation_fmt.format(**fmtdict)
        return db.executeone(operation, params, auto_commit=True,
                             verbose=VERBOSE, **kwargs)

    @default_decorator
    def _executemany_operation_fmt(db, operation_fmt, fmtdict, params_iter,
                                   unpack_scalars=True, **kwargs):
        operation = operation_fmt.format(**fmtdict)
        return db.executemany(operation, params_iter, unpack_scalars=unpack_scalars,
                              auto_commit=True, verbose=VERBOSE, **kwargs)

    #=========
    # SQLDB CORE
    #=========

    @default_decorator
    def optimize(db):
        # http://web.utk.edu/~jplyon/sqlite/SQLite_optimization_FAQ.html#pragma-cache_size
        print('[sql] executing sql optimizions')
        #db.cur.execute('PRAGMA cache_size = 1024;')
        #db.cur.execute('PRAGMA page_size = 1024;')
        #db.cur.execute('PRAGMA synchronous = OFF;')

    @default_decorator
    def schema(db, tablename, schema_list, table_constraints=[], docstr=''):
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
        db.table_docstr[tablename] = docstr

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
        TODO: SEPARATE
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
        # --- ARGS PREPROC ---
        # Aggresively compute iterator if the num_params is not given
        if num_params is None:
            params_iter = list(params_iter)
            num_params  = len(params_iter)
        # Do not compute executemany without params
        if num_params == 0:
            utool.printif(
                lambda: ('[sql!] WARNING: dont use executemany'
                         'with no params use executeone instead.'), VERBOSE)
            return []
        # --- SQL EXECUTION CODE ---
        def closure_execute_many(context):
            _sql_exec_gen = context.execute_and_generate_results
            results_iter = map(list, (_sql_exec_gen(params) for params in params_iter))  # list of iterators
            if unpack_scalars:
                results_iter = list(imap(_unpacker, results_iter))  # list of iterators
            results_list = list(results_iter)  # Eager evaluation
            return results_list
        # --- SQL EXECUTION ---
        contextkw = {'num_params': num_params, 'start_transaction': True}
        with SQLExecutionContext(db, operation, **contextkw) as context:
            try:
                locals_ = locals()
                results_list = closure_execute_many(context)
            except Exception as ex:
                utool.printex(ex, utool.dict_str(locals_))
                raise
        return results_list

    @default_decorator
    def commit(db,  verbose=VERBOSE):
        """ Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        try:
            db.connection.commit()
            if AUTODUMP:
                db.dump(auto_commit=False)
        except lite.Error as ex2:
            utool.printex(ex2, '[!sql] Error during commit')
            raise

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
                dump_fpath = join(db.dir_, db.fname + '.dump.txt')
            else:
                dump_fpath = file_
            with open(dump_fpath, 'w') as file_:
                db.dump(file_, auto_commit)
        else:
            if utool.VERYVERBOSE:
                print('[sql.dump]')
            if auto_commit:
                db.commit(verbose=False)
            for line in db.connection.iterdump():
                file_.write('%s\n' % line)

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

    @default_decorator
    def get_column(db, tablename, name):
        """ Conveinience: """
        _table, (_column,) = sanatize_sql(db, tablename, (name,))
        column_vals = db.executeone(
            operation='''
            SELECT %s
            FROM %s
            ORDER BY rowid ASC
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
        header_doc = utool.indentjoin(utool.unindent(db.table_docstr[tablename]).split('\n'), '\n# ')
        header = header_doc + '\n' + header_name + header_types + '\n' + header_constraints
        return header

    def print_schema(db):
        for tablename in db.table_columns.iterkeys():
            print(db.get_table_csv_header(tablename) + '\n')

    @default_decorator
    def get_sql_version(db):
        """ Conveinience """
        _executor(db.cur, '''
                   SELECT sqlite_version()
                   ''', verbose=False)
        sql_version = db.cur.fetchone()

        print('[sql] SELECT sqlite_version = %r' % (sql_version,))
        # The version number sqlite3 module. NOT the version of SQLite library.
        print('[sql] sqlite3.version = %r' % (lite.version,))
        # The version of the SQLite library
        print('[sql] sqlite3.sqlite_version = %r' % (lite.sqlite_version,))
        return sql_version


# LONG DOCSTRS
#SQLDatabaseController.add_cleanly.__docstr__ = """
#uuid_list - a non-rowid column which identifies a row
#get_rowid_from_uuid - function which does what it says
#e.g:
#    get_rowid_from_uuid = ibs.get_image_gids_from_uuid
#    params_list = [(uuid.uuid4(),) for _ in xrange(7)]
#    unique_paramx = [0]

#            params_list = [(uuid.uuid4(), 42) for _ in xrange(7)]
#            unique_paramx = [0, 1]
#"""

#SQLDatabaseController.__init__.__docstr__ = """
#            SQLite3 Documentation: http://www.sqlite.org/docs.html
#            -------------------------------------------------------
#            SQL INSERT: http://www.w3schools.com/sql/sql_insert.asp
#            SQL UPDATE: http://www.w3schools.com/sql/sql_update.asp
#            SQL DELETE: http://www.w3schools.com/sql/sql_delete.asp
#            SQL SELECT: http://www.w3schools.com/sql/sql_select.asp
#            -------------------------------------------------------
#            Init the SQLite3 database connection and the execution object.
#            If the database does not exist, it will be automatically created
#            upon this object's instantiation.
#            """
