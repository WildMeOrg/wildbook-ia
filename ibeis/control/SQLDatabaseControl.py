from __future__ import absolute_import, division, print_function
# Python
import six
from six.moves import map, zip
from os.path import join, exists
import utool
# Tools
from ibeis.control._sql_helpers import (_unpacker, sanatize_sql,
                                        SQLExecutionContext)
from ibeis.control import __SQLITE3__ as lite
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[sql]')


def default_decorator(func):
    return func
    #return profile(func)
    #return utool.indent_func('[sql.' + func.__name__ + ']')(func)

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
    #return common_decor(func)
    return func


def setter_sql(func):
    #return common_decor(func)
    return func


def deleter_sql(func):
    #return common_decor(func)
    return func


def ider_sql(func):
    #return common_decor(func)
    return func


__STR__ = str if six.PY3 else unicode


class SQLDatabaseController(object):
    """ SQLDatabaseController an efficientish interface into SQL
    </CYTH>
    """

    def __init__(db, sqldb_dpath='.', sqldb_fname='database.sqlite3',
                 text_factory=__STR__):
        """ Creates db and opens connection """
        #with utool.Timer('New SQLDatabaseController'):
        #printDBG('[sql.__init__]')
        # Table info
        db.table_columns     = utool.odict()
        db.table_constraints = utool.odict()
        db.table_docstr      = utool.odict()
        # TODO:
        db.stack = []
        db.cache = {}  # key \in [tblname][colnames][rowid]
        # Get SQL file path
        db.dir_  = sqldb_dpath
        db.fname = sqldb_fname
        assert exists(db.dir_), '[sql] db.dir_=%r does not exist!' % db.dir_
        fpath    = join(db.dir_, db.fname)
        if not exists(fpath):
            print('[sql] Initializing new database')
        # Open the SQL database connection with support for custom types
        #lite.enable_callback_tracebacks(True)
        #fpath = ':memory:'
        db.connection = lite.connect2(fpath)
        db.connection.text_factory = text_factory
        #db.connection.isolation_level = None  # turns sqlite3 autocommit off
        COPY_TO_MEMORY = utool.get_flag('--copy-db-to-memory')
        if COPY_TO_MEMORY:
            db._copy_to_memory()
            db.connection.text_factory = text_factory
        # Get a cursor which will preform sql commands / queries / executions
        db.cur = db.connection.cursor()
        # Optimize the database (if anything is set)
        #db.optimize()

    def _copy_to_memory(db):
        # http://stackoverflow.com/questions/3850022/python-sqlite3-load-existing-db-file-to-memory
        from six.moves import cStringIO
        print('[sql] Copying database into RAM')
        tempfile = cStringIO()
        for line in db.connection.iterdump():
            tempfile.write('%s\n' % line)
        db.connection.close()
        tempfile.seek(0)
        # Create a database in memory and import from tempfile
        db.connection = lite.connect2(":memory:")
        db.connection.cursor().executescript(tempfile.read())
        db.connection.commit()
        db.connection.row_factory = lite.Row

    #==============
    # API INTERFACE
    #==============

    #@ider_sql
    def get_all_rowids(db, tblname):
        """ returns a list of all rowids from a table in ascending order """
        fmtdict = {'tblname': tblname, }
        operation_fmt = '''
        SELECT rowid
        FROM {tblname}
        ORDER BY rowid ASC
        '''
        return db._executeone_operation_fmt(operation_fmt, fmtdict)

    #@ider_sql
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

    #@adder_sql
    def add_cleanly(db, tblname, colnames, params_iter, get_rowid_from_superkey, superkey_paramx=(0,)):
        """ ADDER Extra input:
            the first item of params_iter must be a superkey (like a uuid), """
        # ADD_CLEANLY_1: PREPROCESS INPUT
        params_list = list(params_iter)  # eagerly evaluate for superkeys
        # Extract superkeys from the params list (requires eager eval)
        superkey_lists = [[None if params is None else params[x]
                           for params in params_list]
                          for x in superkey_paramx]
        # ADD_CLEANLY_2: PREFORM INPUT CHECKS
        # check which parameters are valid
        isvalid_list = [params is not None for params in params_list]
        # Check for duplicate inputs
        isunique_list = utool.flag_unique_items(list(zip(*superkey_lists)))
        # Check to see if this already exists in the database
        rowid_list_ = get_rowid_from_superkey(*superkey_lists)
        isnew_list  = [rowid is None for rowid in rowid_list_]
        if VERBOSE and not all(isunique_list):
            print('[WARNING]: duplicate inputs to db.add_cleanly')
        # Flag each item that needs to added to the database
        isdirty_list = list(map(all, zip(isvalid_list, isunique_list, isnew_list)))
        # ADD_CLEANLY_3.1: EXIT IF CLEAN
        if not any(isdirty_list):
            return rowid_list_  # There is nothing to add. Return the rowids
        # ADD_CLEANLY_3.2: PERFORM DIRTY ADDITIONS
        dirty_params = utool.filter_items(params_list, isdirty_list)
        if utool.VERBOSE:
            print('[sql] adding %r/%r new %s' % (len(dirty_params), len(params_list), tblname))
        # Add any unadded parameters to the database
        try:
            db._add(tblname, colnames, dirty_params)
        except Exception as ex:
            utool.printex(ex, key_list=['isdirty_list', 'superkey_lists', 'rowid_list_'])
            raise
        # TODO: We should only have to preform a subset of adds here
        # (at the positions where rowid_list was None in the getter check)
        rowid_list = get_rowid_from_superkey(*superkey_lists)
        # ADD_CLEANLY_4: SANITY CHECK AND RETURN
        assert len(rowid_list) == len(params_list), 'failed sanity check'
        return rowid_list

    #@getter_sql
    def get_where(db, tblname, colnames, params_iter, where_clause, unpack_scalars=True, **kwargs):
        assert isinstance(colnames, tuple)
        #if isinstance(colnames, six.string_types):
        #    colnames = (colnames,)
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

    #@getter_sql
    def get_rowid_from_superkey(db, tblname, params_iter=None, superkey_colnames=None, **kwargs):
        """ getter which uses the constrained superkeys instead of rowids """
        where_clause = 'AND'.join([colname + '=?' for colname in superkey_colnames])
        return db.get_where(tblname, ('rowid',), params_iter, where_clause, **kwargs)

    #@getter_sql
    def get(db, tblname, colnames, id_iter=None, id_colname='rowid', **kwargs):
        """ getter """
        assert isinstance(colnames, tuple)
        #if isinstance(colnames, six.string_types):
        #    colnames = (colnames,)
        where_clause = (id_colname + '=?')
        params_iter = ((_rowid,) for _rowid in id_iter)
        return db.get_where(tblname, colnames, params_iter, where_clause, **kwargs)

    #@setter_sql
    def set(db, tblname, colnames, val_iter, id_iter, id_colname='rowid', **kwargs):
        """ setter """
        assert isinstance(colnames, tuple)
        #if isinstance(colnames, six.string_types):
        #    colnames = (colnames,)
        val_list = list(val_iter)  # eager evaluation
        id_list = list(id_iter)  # eager evaluation
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

        # TODO: The flattenize can be removed if we pass in val_lists instead
        params_iter = utool.flattenize(list(zip(val_list, id_list)))
        #params_iter = list(zip(val_list, id_list))
        return db._executemany_operation_fmt(operation_fmt, fmtdict,
                                             params_iter=params_iter, **kwargs)

    #@deleter_sql
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

    #@deleter_sql
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
        # --- ARGS PREPROC ---
        # Aggresively compute iterator if the num_params is not given
        if num_params is None:
            params_iter = list(params_iter)
            num_params  = len(params_iter)
        # Do not compute executemany without params
        if num_params == 0:
            if VERBOSE:
                print('[sql!] WARNING: dont use executemany'
                      'with no params use executeone instead.')
            return []
        # --- SQL EXECUTION ---
        contextkw = {'num_params': num_params, 'start_transaction': True}
        with SQLExecutionContext(db, operation, **contextkw) as context:
            #try:
            results_iter = list(map(list, (context.execute_and_generate_results(params) for params in params_iter)))  # list of iterators
            if unpack_scalars:
                results_iter = list(map(_unpacker, results_iter))  # list of iterators
            results_list = list(results_iter)  # Eager evaluation
            #except Exception as ex:
            #    utool.printex(ex)
            #    raise
        return results_list

    #@default_decorator
    #def commit(db,  verbose=VERBOSE):
    #    try:
    #        db.connection.commit()
    #        if AUTODUMP:
    #            db.dump(auto_commit=False)
    #    except lite.Error as ex2:
    #        utool.printex(ex2, '[!sql] Error during commit')
    #        raise

    @default_decorator
    def dump_to_file(db, file_, auto_commit=True):
        if utool.VERYVERBOSE:
            print('[sql.dump]')
        if auto_commit:
            db.connection.commit()
            #db.commit(verbose=False)
        for line in db.connection.iterdump():
            file_.write('%s\n' % line)

    #==============
    # CONVINENCE
    #==============

    @default_decorator
    def dump(db, file_=None, auto_commit=True):
        if file_ is None or isinstance(file_, six.string_types):
            dump_fpath = file_
            if dump_fpath is None:
                dump_fpath = join(db.dir_, db.fname + '.dump.txt')
            with open(dump_fpath, 'w') as file_:
                db.dump_to_file(file_, auto_commit)
        else:
            db.dump_to_file(file_)

    @default_decorator
    def dump_tables_to_csv(db):
        """ Convenience: Dumps all csv database files to disk """
        dump_dir = join(db.dir_, 'CSV_DUMP')
        utool.ensuredir(dump_dir)
        for tablename in six.iterkeys(db.table_columns):
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
        column_lbls = []
        for name in column_names:
            if name in exclude_columns:
                continue
            column_vals = db.get_column(tablename, name)
            column_list.append(column_vals)
            column_lbls.append(name.replace(tablename[:-1] + '_', ''))
        # remove column prefix for more compact csvs

        #=None, column_list=[], header='', column_type=None
        header = db.get_table_csv_header(tablename)
        csv_table = utool.make_csv_table(column_list, column_lbls, header)
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
        for tablename in six.iterkeys(db.table_columns):
            print(db.get_table_csv_header(tablename) + '\n')

    @default_decorator
    def get_sql_version(db):
        """ Conveinience """
        db.cur.execute('SELECT sqlite_version()')
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
#get_rowid_from_superkey - function which does what it says
#e.g:
#    get_rowid_from_superkey = ibs.get_image_gids_from_uuid
#    params_list = [(uuid.uuid4(),) for _ in range(7)]
#    superkey_paramx = [0]

#            params_list = [(uuid.uuid4(), 42) for _ in range(7)]
#            superkey_paramx = [0, 1]
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
#""" Same output as shell command below
#    > sqlite3 database.sqlite3 .dump > database.dump.txt

#    If file_=sys.stdout dumps to standard out

#    This saves the current database schema structure and data into a
#    text dump. The entire database can be recovered from this dump
#    file. The default will store a dump parallel to the current
#    database file.
#"""
#""" Commits staged changes to the database and saves the binary
#    representation of the database to disk.  All staged changes can be
#    commited one at a time or after a batch - which allows for batch
#    error handling without comprimising the integrity of the database.
#"""
#"""
#TODO: SEPARATE
#Input:
#    operation - an sql command to be executed e.g.
#        operation = '''
#        SELECT colname
#        FROM tblname
#        WHERE
#        (colname_1=?, ..., colname_N=?)
#        '''
#    params_iter - a sequence of params e.g.
#        params_iter = [(col1, ..., colN), ..., (col1, ..., colN),]
#Output:
#    results_iter - a sequence of data results
#"""
#"""
#operation - parameterized SQL operation string.
#    Parameterized prevents SQL injection attacks by using an ordered
#    representation ( ? ) or by using an ordered, text representation
#    name ( :value )

#params - list of values or a dictionary of representations and
#                corresponding values
#    * Ordered Representation -
#        List of values in the order the question marks appear in the
#        sql operation string
#    * Unordered Representation -
#        Dictionary of (text representation name -> value) in an
#        arbirtary order that will be filled into the cooresponging
#        slots of the sql operation string
#"""
#""" Creates a table in the database with some schema and constraints
#    schema_list - list of tablename columns tuples
#        {
#            (column_1_name, column_1_type),
#            (column_2_name, column_2_type),
#            ...
#            (column_N_name, column_N_type),
#        }
#    ---------------------------------------------
#    column_n_name - string name of column heading
#    column_n_type - NULL | INTEGER | REAL | TEXT | BLOB | NUMPY
#        The column type can be appended with ' PRIMARY KEY' to indicate
#        the unique id for the tablename.  It can also specify a default
#        value for the column with ' DEFAULT [VALUE]'.  It can also
#        specify ' NOT NULL' to indicate the column cannot be empty.
#    ---------------------------------------------
#    The tablename will only be created if it does not exist.  Therefore,
#    this can be done on every tablename without fear of deleting old data.
#    ---------------------------------------------
#    TODO: Add handling for column addition between software versions.
#    Column deletions will not be removed from the database schema.
#"""
