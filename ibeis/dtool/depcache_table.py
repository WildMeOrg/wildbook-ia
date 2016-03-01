# -*- coding: utf-8 -*-
"""
Module contining DependencyCacheTable

python -m dtool.depcache_control --exec-make_graph --show
python -m dtool.depcache_control --exec-make_graph --show --reduce
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from six.moves import zip, range
from os.path import join, exists
from dtool import __SQLITE__ as lite
(print, rrr, profile) = ut.inject2(__name__, '[depcache_table]')


EXTERN_SUFFIX = '_extern_uri'

CONFIG_TABLE     = 'config'
CONFIG_ROWID     = 'config_rowid'
CONFIG_HASHID    = 'config_hashid'
CONFIG_TABLENAME = 'config_tablename'  # tablename associated with config
CONFIG_STRID     = 'config_strid'


if ut.is_developer():
    GRACE_PERIOD = 0
else:
    GRACE_PERIOD = 0
#ALLOW_NONE_YIELD = False
ALLOW_NONE_YIELD = True


def predrop_grace_period(tablename, seconds=None):
    """ Hack that gives the user some time to abort deleting everything """
    global GRACE_PERIOD
    warnmsg_fmt = ut.codeblock(
        '''
        WARNING TABLE={tablename} IS MODIFIED

        About to reset (DROP) entire cache={tablename}.

        Generally this is OK and you shouldnt worry because depcache
        information should be recomputable.

        If you really dont want this to happen you have {seconds} seconds to
        kill this process before deletion occurs.
        ''')
    if seconds is None:
        seconds = GRACE_PERIOD
        GRACE_PERIOD = max(0, GRACE_PERIOD // 2)
    warnmsg = warnmsg_fmt.format(tablename=tablename, seconds=seconds)
    #return ut.are_you_sure(warnmsg)
    return ut.grace_period(warnmsg, seconds)


def ensure_config_table(db):
    """ SQL definition of configuration table. """
    config_addtable_kw = ut.odict(
        [
            ('tablename', CONFIG_TABLE,),
            ('coldef_list', [
                (CONFIG_ROWID, 'INTEGER PRIMARY KEY'),
                (CONFIG_HASHID, 'TEXT'),
                (CONFIG_TABLENAME, 'TEXT'),
                (CONFIG_STRID, 'TEXT'),
            ],),
            ('docstr', 'table for algo configurations'),
            ('superkeys', [(CONFIG_HASHID,)]),
            ('dependson', [])
        ]
    )

    if not db.has_table(CONFIG_TABLE):
        db.add_table(**config_addtable_kw)
    else:
        current_state = db.get_table_autogen_dict(CONFIG_TABLE)
        new_state = config_addtable_kw
        if current_state['coldef_list'] != new_state['coldef_list']:
            if predrop_grace_period(CONFIG_TABLE):
                db.drop_all_tables()
                db.add_table(**new_state)
            else:
                raise NotImplementedError('Need to be able to modify tables')


@ut.reloadable_class
class DependencyCacheTable(ut.NiceRepr):
    r"""
    An individual node in the dependency graph.

    Attributes:
        db (dtool.SQLDatabaseController): pointer to underlying database
        depc (dtool.DependencyCache): pointer to parent cache
        tablename (str): name of the table
        docstr (str): documentation for table
        parent_tablenames (str): parent tables in depcache
        data_colnames (List[str]): columns produced by preproc_func
        data_coltypes (List[str]): column SQL types produced by preproc_func
        preproc_func (func): worker function

    CommandLine:
        python -m dtool.depcache_table --exec-DependencyCacheTable

    Example:
        >>> # ENABLE_DOCTEST
        >>> from dtool.depcache_table import *  # NOQA
        >>> from dtool.example_depcache import testdata_depc
        >>> depc = testdata_depc()
        >>> print(depc['vsmany'])
        >>> print(depc['spam'])
        >>> print(depc['vsone'])
        >>> print(depc['nnindexer'])
    """

    def __init__(table, depc=None, parent_tablenames=None, tablename=None,
                 data_colnames=None, data_coltypes=None, preproc_func=None,
                 docstr='no docstr', fname=None, asobject=False,
                 chunksize=None, isinteractive=False, default_to_unpack=False):
        """ recieves kwargs from depc._register_prop """
        try:
            table.db = None
        except Exception:
            # HACK: jedi type hinting. Need to have non-obvious condition
            from dtool.sql_control import SQLDatabaseController
            table.db = SQLDatabaseController()
        table.fpath_to_db = {}
        import re
        assert re.search('[0-9]', tablename) is None, 'tablename=%r cannot contain numbers' % (tablename,)

        table.parent_tablenames = parent_tablenames
        table.tablename = tablename
        table.data_colnames = tuple(data_colnames)
        table.data_coltypes = data_coltypes
        table.preproc_func = preproc_func
        table.on_delete = None
        table.isinteractive = isinteractive
        table.default_to_unpack = default_to_unpack
        table.docstr = docstr
        table.fname = fname
        table.depc = depc
        table.subproperties = {}
        table.chunksize = chunksize
        table._asobject = asobject
        # SQL Internals
        table.sqldb_fpath = None
        table._internal_parent_id_colnames = ()
        table._internal_parent_id_coltypes = ()
        table._internal_parent_extra_colnames = ()
        table._internal_parent_extra_coltypes = ()
        table._parent_id_prefixes = ()
        table._internal_data_colnames = ()
        table._nested_idxs = []
        table.extern_read_funcs = {}
        table.extern_write_funcs = {}
        table.extern_ext_config_keys = {}
        table.extern_io_classes = {}
        table.multi_parent_colxs = {}
        # Update internals
        table._update_internal_parentcol()
        table._update_internal_datacol()
        table._update_internal_allcol()
        # Check for errors
        table._assert_self()

    @profile
    def initialize(table, _debug=None):
        """
        Ensures the SQL schema for this cache table
        """
        table.db = table.depc.fname_to_db[table.fname]
        #print('Checking sql for table=%r' % (table.tablename,))
        if not table.db.has_table(table.tablename):
            if _debug or ut.VERBOSE:
                print('Initializing table=%r' % (table.tablename,))
            new_state = table.get_addtable_kw()
            table.db.add_table(**new_state)
        else:
            # TODO: Check for table modifications
            new_state = table.get_addtable_kw()
            try:
                current_state = table.db.get_table_autogen_dict(table.tablename)
            except Exception as ex:
                strict = True
                ut.printex(ex, 'TABLE %s IS CORRUPTED' % (table.tablename,), iswarning=not strict)
                if strict:
                    raise
                table.clear_table()
                current_state = table.db.get_table_autogen_dict(table.tablename)

            if current_state['coldef_list'] != new_state['coldef_list']:
                print('WARNING TABLE IS MODIFIED')
                if predrop_grace_period(table.tablename):
                    table.clear_table()
                else:
                    raise NotImplementedError('Need to be able to modify tables')

    def __nice__(table):
        num_parents = len(table.parent_tablenames)
        num_cols = len(table.data_colnames)
        return '(%s) nP=%d%s nC=%d' % (table.tablename, num_parents, '*' if
                                       False and table.ismulti else '', num_cols)

    def _assert_self(table):
        assert len(table.data_colnames) == len(table.data_coltypes), (
            'specify same number of colnames and coltypes')
        if table.preproc_func is not None:
            # Check that preproc_func has a valid signature
            # ie (depc, parent_ids, config)
            argspec = ut.get_func_argspec(table.preproc_func)
            args = argspec.args
            if argspec.varargs and argspec.keywords:
                assert len(args) == 1, (
                    'varargs and kwargs must have one arg for depcache')
            else:
                if len(args) < 3:
                    print('args = %r' % (args,))
                    msg = (
                        'preproc_func=%r for table=%s must have a '
                        'depcache arg, at least one parent rowid arg, '
                        'and a config arg') % (
                            table.preproc_func, table.tablename,)
                    raise AssertionError(msg)
                rowid_args = args[1:-1]
                if len(rowid_args) != len(table.parents()):
                    print('table.preproc_func = %r' %
                          (table.preproc_func,))
                    print('args = %r' % (args,))
                    print('rowid_args = %r' % (rowid_args,))
                    msg = (
                        ('preproc function for table=%s must have as many '
                         'rowids %d args as parent %d') % (
                            table.tablename, len(rowid_args),
                             len(table.parents()))
                    )
                    raise AssertionError(msg)
        extern_class_colattrs = [colattr for colattr in table.data_col_attrs if colattr.get('is_external_class')]
        for colattr in extern_class_colattrs:
            cls = colattr['coltype']
            # Check external column class funcs
            argspec = ut.get_func_argspec(cls.__init__)
            if argspec.defaults is not None:
                num_nondefault = len(argspec.args) - len(argspec.defaults)
            else:
                num_nondefault = len(argspec.args)
            if num_nondefault > 1:
                msg = ut.codeblock(
                    '''
                    External args must be able to be constructed without any
                    args. IE: You need a default __init__(self) method
                    ''')
                raise AssertionError(msg)

    def get_addtable_kw(table):
        coldef_list = [(colattr['intern_colname'], colattr['sqltype'])
                       for colattr in table.internal_col_attrs]
        add_table_kw = ut.odict([
            ('tablename', table.tablename,),
            ('coldef_list', coldef_list,),
            ('docstr', table.docstr,),
            ('superkeys', [table.superkey_colnames],),
            ('dependson', table.parents()),
        ])
        return add_table_kw

    def _update_internal_allcol(table):
        r"""
        Build column definitions convinient for sql

        CommandLine:
            python -m dtool.depcache_table --exec-_update_internal_allcol --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> for table in ut.take(depc, ['vsone', 'spam', 'notch', 'vsmany', 'chip', 'multitest']): #depc.tables:
            >>>     print('----')
            >>>     table._update_internal_allcol()
            >>>     print(table)
            >>>     ut.colorprint('table.internal_col_attrs = %s' % (ut.repr3(table.internal_col_attrs, nl=1, sorted_=False)), 'python')
            >>>     print('table.parent_col_attrs = %s' % (ut.repr3(table.parent_col_attrs, nl=2),))
            >>>     print('table.data_col_attrs = %s' % (ut.repr3(table.data_col_attrs, nl=2),))
            >>>     for a in ut.get_instance_attrnames(table, with_properties=True, default=False):
            >>>         print('  table.%s = %r' % (a, getattr(table, a)))
        """
        internal_col_attrs = []

        # Append primary column
        colattr = ut.odict([
            ('intern_colname', table.rowid_colname),
            ('sqltype', 'INTEGER PRIMARY KEY'),
            ('isprimary', True),
        ])
        colattr['intern_colx'] = len(internal_col_attrs)
        internal_col_attrs.append(colattr)

        # Append parent columns
        for parent_colattr in table.parent_col_attrs:
            colattr = ut.odict()
            colattr['intern_colname'] = parent_colattr['intern_colname']
            colattr['parent_table'] = parent_colattr['parent_table']
            colattr['ismulti'] = parent_colattr['ismulti']
            colattr['sqltype'] = parent_colattr['sqltype']
            colattr['parent_colx'] = parent_colattr['parent_colx']
            colattr['intern_colx'] = len(internal_col_attrs)
            colattr['isparent'] = True
            colattr['issuper'] = True
            internal_col_attrs.append(colattr)

        # Append config columns
        colattr = ut.odict([
            ('intern_colname', CONFIG_ROWID),
            ('sqltype', 'INTEGER DEFAULT 0'),
            ('issuper', True),
        ])
        colattr['intern_colx'] = len(internal_col_attrs)

        # Append data columns
        internal_col_attrs.append(colattr)
        for data_colattr in table.data_col_attrs:
            colname = data_colattr['colname']
            if data_colattr.get('isnested', False):
                for nestcol in data_colattr['nestattrs']:
                    colattr = ut.odict()
                    colattr['intern_colname'] = nestcol['flat_colname']
                    colattr['sqltype'] = nestcol['sqltype']
                    colattr['intern_colx'] = len(internal_col_attrs)
                    colattr['data_colx'] = data_colattr['data_colx']
                    colattr['colname'] = colname
                    colattr['isdata'] = True
                    internal_col_attrs.append(colattr)
            else:
                colattr = ut.odict()
                colattr['intern_colname'] = data_colattr['intern_colname']
                colattr['sqltype'] = data_colattr['sqltype']
                colattr['intern_colx'] = len(internal_col_attrs)
                colattr['data_colx'] = data_colattr['data_colx']
                colattr['isdata'] = True
                colattr['colname'] = colname
                if data_colattr.get('is_external', False):
                    colattr['is_external_pointer'] = True
                    colattr['write_func'] = data_colattr['write_func']
                    colattr['read_func'] = data_colattr['read_func']
                internal_col_attrs.append(colattr)

        # Append extra columns
        for parent_colattr in table.parent_col_attrs:
            for extra_colattr in parent_colattr.get('extra_cols', []):
                colattr = ut.odict()
                colattr['intern_colname'] = extra_colattr['intern_colname']
                colattr['sqltype'] = extra_colattr['sqltype']
                colattr['intern_colx'] = len(internal_col_attrs)
                colattr['isextra'] = True
                internal_col_attrs.append(colattr)

        table.internal_col_attrs = internal_col_attrs

    def _update_internal_datacol(table):
        """
        Constructs the columns needed to represent relationship to data

        Infers interal properties about this table given the colnames and
        datatypes

        CommandLine:
            python -m dtool.depcache_table --exec-_update_internal_datacol --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> for table in depc.tables:
            >>>     print('----')
            >>>     table._update_internal_datacol()
            >>>     print(table)
            >>>     print('table.data_col_attrs = %s' % (ut.repr3(table.data_col_attrs, nl=8)))
            >>> table = depc['probchip']
            >>> table = depc['spam']
            >>> table = depc['vsone']
        """
        data_col_attrs = []

        # TODO: can rewrite much of this
        extern_read_funcs = {}
        extern_write_funcs = {}
        extern_ext_config_keys = {}
        extern_io_classes = {}
        internal_data_colnames = []
        internal_data_coltypes = []
        nested_to_flat = {}
        external_to_internal = {}

        def make_extern_io_funcs(cls):
            # Hack in read/write defaults for pickleable classes
            def _read_func(fpath, verbose=ut.VERBOSE):
                state_dict = ut.load_data(fpath, verbose=verbose)
                self = cls()
                self.__setstate__(state_dict)
                if hasattr(self, 'on_load'):
                    self.on_load(table.depc)
                return self

            def _write_func(fpath, self, verbose=ut.VERBOSE):
                if hasattr(self, 'on_save'):
                    self.on_save(table.depc, fpath)
                ut.save_data(fpath, self.__getstate__(), verbose=verbose, n=4)
            return _read_func, _write_func

        # Parse column datatypes
        _iter = enumerate(zip(table.data_colnames, table.data_coltypes))
        for data_colx, (colname, coltype) in _iter:
            colattr = ut.odict()
            # Check column input subtypes
            is_tuple     = isinstance(coltype, tuple)
            is_func      = ut.is_func_or_method(coltype)
            is_externtup = is_tuple and coltype[0] == 'extern'
            is_functup   = is_tuple and ut.is_func_or_method(coltype[0])
            # Check column input main types
            is_normal   = coltype in lite.TYPE_TO_SQLTYPE
            #is_normal   = not (is_tuple or is_func)
            isnested   = is_tuple and not (is_func or is_externtup)
            is_external = (is_func or is_functup or is_externtup)
            # Switch on input types
            colattr['colname'] = colname
            colattr['coltype'] = coltype
            colattr['data_colx'] = data_colx
            if is_normal:
                # Normal non-nested column
                sqltype = lite.TYPE_TO_SQLTYPE[coltype]
                internal_data_colnames.append(colname)
                internal_data_coltypes.append(sqltype)
                colattr['intern_colname'] = colname
                colattr['sqltype'] = sqltype
                colattr['is_normal'] = is_normal
            elif isnested:
                # Nested non-function normal columns
                table._nested_idxs.append(data_colx)
                nested_to_flat[colname] = []
                colattr['isnested'] = isnested
                nestattrs = colattr['nestattrs'] = []
                for count, subtype in enumerate(coltype):
                    nestattr = ut.odict()
                    nestattrs.append(nestattr)
                    flat_colname = '%s_%d' % (colname, count)
                    sqltype = lite.TYPE_TO_SQLTYPE[subtype]
                    nested_to_flat[colname].append(flat_colname)
                    internal_data_colnames.append(flat_colname)
                    internal_data_coltypes.append(sqltype)
                    nestattr['flat_colname'] = flat_colname
                    nestattr['sqltype'] = sqltype
            elif is_external:
                # Nested external funcs
                write_func = None
                if is_externtup:
                    read_func = coltype[1]
                    if len(coltype) > 2:
                        write_func = coltype[2]
                    #if len(coltype) > 3:
                    #    # specify key in config that determines extension
                    #    extern_ext_config_keys[colname] = coltype[3]
                elif is_functup:
                    read_func = coltype[0]
                else:
                    read_func = coltype
                extern_read_funcs[colname] = read_func
                if write_func:
                    extern_write_funcs[colname] = write_func
                intern_colname = colname + EXTERN_SUFFIX
                external_to_internal[colname] = intern_colname
                sqltype = lite.TYPE_TO_SQLTYPE[str]
                internal_data_colnames.append(intern_colname)
                internal_data_coltypes.append(sqltype)
                colattr['is_external'] = True
                colattr['intern_colname'] = intern_colname
                colattr['write_func'] = write_func
                colattr['read_func'] = read_func
                colattr['sqltype'] = sqltype
            else:
                # External class column
                assert (hasattr(coltype, '__getstate__') and
                        hasattr(coltype, '__setstate__')), (
                        'External classes must have __getstate__ and '
                        '__setstate__ methods')
                read_func, write_func = make_extern_io_funcs(coltype)
                sqltype = lite.TYPE_TO_SQLTYPE[str]
                extern_io_classes[colname]  = coltype
                extern_read_funcs[colname]  = read_func
                extern_write_funcs[colname] = write_func
                intern_colname = colname + EXTERN_SUFFIX
                external_to_internal[colname] = intern_colname
                internal_data_colnames.append(intern_colname)
                internal_data_coltypes.append(sqltype)
                #raise AssertionError('external class columns')
                colattr['is_external'] = True
                colattr['is_external_class'] = True
                colattr['coltype'] = coltype
                colattr['intern_colname'] = intern_colname
                colattr['write_func'] = write_func
                colattr['read_func'] = read_func
                colattr['sqltype'] = sqltype
            data_col_attrs.append(colattr)
        # Set new internal data properties of the table
        assert len(set(internal_data_colnames)) == len(internal_data_colnames)
        assert len(internal_data_coltypes) == len(internal_data_colnames)
        table.extern_read_funcs = extern_read_funcs
        table.extern_write_funcs = extern_write_funcs
        table.extern_ext_config_keys = extern_ext_config_keys
        table.extern_io_classes = extern_io_classes
        table.external_to_internal = external_to_internal
        table.nested_to_flat = nested_to_flat
        table.data_col_attrs = data_col_attrs
        table._internal_data_colnames = tuple(internal_data_colnames)
        table._assert_self()

    def _update_internal_parentcol(table):
        """
        constructs the columns needed to represent relationship to parent

        CommandLine:
            python -m dtool.depcache_table --exec-_update_internal_parentcol --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> for table in ['vsone', 'multitest', 'spam', 'notch', 'vsmany', 'chip']: #depc.tables:
            >>>     print('----')
            >>>     table._update_internal_parentcol()
            >>>     print(table)
            >>>     print('table.parent_col_attrs = %s' % (ut.repr3(table.parent_col_attrs),))
        """
        parent_col_attrs = []

        # Handle dependencies when a parent are pairwise between tables
        ismulti_cols = {}
        multi_parent_colxs = []
        parent_id_prefixs1 = []
        parent_id_prefixs2 = []
        seen_ = ut.ddict(lambda: 1)

        for parent_colx, col in enumerate(table.parent_tablenames):
            colattr = ut.odict()
            # Detect multicolumns
            if col.endswith('*'):
                ismulti = True
                parent_table = col[:-1]
                ismulti_cols[parent_table] = True
                multi_parent_colxs.append(parent_colx)
            else:
                ismulti = False
                parent_table = col
            colattr['col'] = col
            colattr['ismulti'] = ismulti
            colattr['parent_table'] = parent_table
            colattr['parent_colx'] = parent_colx
            parent_id_prefixs1.append(parent_table)
            parent_col_attrs.append(colattr)

        colhist = ut.dict_hist(parent_id_prefixs1)
        for parent_colx, col in enumerate(parent_id_prefixs1):
            colattr = parent_col_attrs[parent_colx]
            if colhist[col] > 1:
                # Duplicate column names recieve indicies
                prefix = col + str(seen_[col])
                seen_[col] += 1
                colattr['is_nwise'] = True
                colattr['n'] = colhist[col]
            else:
                prefix = col
            colattr['prefix'] = prefix
            parent_id_prefixs2.append(prefix)

        _internal_parent_id_colnames = []
        _internal_parent_id_coltypes = []

        _internal_parent_extra_colnames = []
        _internal_parent_extra_coltypes = []

        # Handle case when parent are a set of ids
        for colattr, prefix in zip(parent_col_attrs, parent_id_prefixs2):
            # HACK. ALL COLUMNS ARE MULTI
            # FIXME: will bug one duplicate multi-indexes
            column_ismulti = ismulti_cols.get(prefix, False)  # or (prefix.endswith('*'))
            if column_ismulti:
                # Case when dependencies are many to one
                # hash of set items
                colname = prefix + '_setuuid'
                sqltype = 'TEXT NOT NULL'
                extra_cols = [
                    {'intern_colname': prefix + '_setsize', 'sqltype': 'INTEGER NOT NULL'},
                    {'intern_colname': prefix + '_setfpath', 'sqltype': 'TEXT'},
                ]
                _internal_parent_id_colnames.append(colname)
                _internal_parent_id_coltypes.append(sqltype)
                # number of items in set
                _internal_parent_extra_colnames.append(prefix + '_setsize')
                _internal_parent_extra_colnames.append(prefix + '_setfpath')
                # path to indiviual parts
                _internal_parent_extra_coltypes.append('INTEGER NOT NULL')
                _internal_parent_extra_coltypes.append('TEXT')
                colattr['extra_cols'] = extra_cols
            else:
                # Normal case when dependencies are one to one
                colname = prefix + '_rowid'
                sqltype = 'INTEGER NOT NULL'
                _internal_parent_id_colnames.append(colname)
                _internal_parent_id_coltypes.append(sqltype)
            colattr['intern_colname'] = colname
            colattr['sqltype'] = sqltype

        parent_col_attrs = [ut.order_dict_by(colattr, ['intern_colname', 'sqltype'])
                            for colattr in parent_col_attrs]
        table.parent_col_attrs = parent_col_attrs

        table._internal_parent_id_colnames = tuple(
            _internal_parent_id_colnames)
        table._internal_parent_id_coltypes = tuple(
            _internal_parent_id_coltypes)
        table._internal_parent_extra_colnames = tuple(
            _internal_parent_extra_colnames)
        table._internal_parent_extra_coltypes = tuple(
            _internal_parent_extra_coltypes)
        table._parent_id_prefixes = tuple(parent_id_prefixs2)
        table.multi_parent_colxs = multi_parent_colxs

    def clear_table(table):
        """
        Deletes all data in this table
        """
        print('Clearing data in %r' % (table,))
        table.db.drop_table(table.tablename)
        table.db.add_table(**table.get_addtable_kw())

    @property
    @ut.memoize
    def ismulti(table):
        # TODO: or has multi parent
        return any(table.get_parent_col_attr('ismulti'))

    @property
    @ut.memoize
    def parent(table):
        return ut.odict([(parent_colattr['parent_table'], parent_colattr)
                         for parent_colattr in table.parent_col_attrs])
        #return tuple([parent_colattr['parent_table']
        #              for parent_colattr in table.parent_col_attrs])

    def parents(table, data=None):
        if data:
            return [(parent_colattr['parent_table'], parent_colattr)
                    for parent_colattr in table.parent_col_attrs]
        else:
            return [parent_colattr['parent_table']
                    for parent_colattr in table.parent_col_attrs]

    def get_parent_col_attr(table, key):
        return ut.dict_take_column(table.parent_col_attrs, key)

    def get_intern_data_col_attr(table, key):
        return ut.dict_take_column(table.internal_data_col_attrs, key)

    def get_intern_col_attr(table, key):
        return ut.dict_take_column(table.internal_col_attrs, key)

    def get_data_col_attr(table, key):
        return ut.dict_take_column(table.data_col_attrs, key)

    # --- Standard Properties

    @property
    def internal_data_col_attrs(table):
        flags = table.get_intern_col_attr('isdata')
        return ut.compress(table.internal_col_attrs, flags)

    @property
    def internal_parent_col_attrs(table):
        flags = table.get_intern_col_attr('isparent')
        return ut.compress(table.internal_col_attrs, flags)

    @ut.memoize
    def requestable_col_attrs(table):
        # Maps names of requestable columns to indicies of internal columns
        requestable_col_attrs = {}
        for colattr in table.internal_data_col_attrs:
            rattr = {}
            colname = colattr['intern_colname']
            rattr['intern_colx'] = colattr['intern_colx']
            rattr['intern_colname'] = colattr['intern_colname']
            requestable_col_attrs[colname] = rattr

        for colattr in table.data_col_attrs:
            rattr = {}
            if colattr.get('isnested'):
                nest_internal_names = ut.take_column(colattr['nestattrs'], 'flat_colname')
                nest_attrs = ut.dict_take(requestable_col_attrs, nest_internal_names)
                rattr['intern_colname'] = nest_internal_names
                rattr['intern_colx'] = ut.take_column(nest_attrs, 'intern_colx')
                rattr['isnested'] = True
            elif colattr.get('is_external'):
                intern_attr = requestable_col_attrs[colattr['intern_colname']]
                rattr['intern_colname'] = intern_attr['intern_colname']
                rattr['intern_colx'] = intern_attr['intern_colx']
                rattr['read_func'] = colattr['read_func']
                rattr['write_func'] = colattr['write_func']
                rattr['is_extern'] = True
            else:
                continue
            colname = colattr['colname']
            rattr['colname'] = colname
            requestable_col_attrs[colname] = rattr
        return requestable_col_attrs

    # --- / Standard Properties

    @property
    @ut.memoize
    def parent_id_tablenames(table):
        tablenames = tuple([parent_colattr['parent_table']
                            for parent_colattr in table.parent_col_attrs])
        return tablenames

    @property
    @ut.memoize
    def parent_id_prefix(table):
        prefixes = tuple([parent_colattr['prefix']
                          for parent_colattr in table.parent_col_attrs])
        return prefixes

    @property
    def children(table):
        import networkx as nx
        graph = table.depc.make_graph(with_implicit=False)
        children_tablenames = nx.neighbors(graph, table.tablename)
        return children_tablenames

    @property
    def extern_columns(table):
        colnames = table.get_data_col_attr('colname')
        flags = table.get_data_col_attr('is_extern')
        return ut.compress(colnames, flags)

    @property
    def rowid_colname(table):
        """ rowid of this table used by other dependant tables """
        return table.tablename + '_rowid'

    @property
    def superkey_colnames(table):
        return table.parent_id_colnames + (CONFIG_ROWID,)

    @property
    def parent_id_colnames(table):
        return tuple([colattr['intern_colname'] for colattr in table.parent_col_attrs])

    #@property
    #def _table_colnames(table):
    #    return

    def print_schemadef(table):
        print('\n'.join(table.db.get_table_autogen_str(table.tablename)))

    def print_configs(table):
        """
        CommandLine:
            python -m dtool.depcache_table --exec-print_configs

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> table = depc['chip']
            >>> rowids = depc.get_rowids('spam', [1, 2])
            >>> table.print_configs()

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> table = depc['keypoint']
            >>> config = table.configclass()
            >>> rowids = table.get_rowids_from_root([1, 2], config=config)
            >>> config = table.configclass(adapt_shape=False)
            >>> rowids = table.get_rowids_from_root([1, 2], config=config)
            >>> table.print_configs()
        """
        text = table.db.get_table_csv(CONFIG_TABLE,
                                      params_iter=[(table.tablename,)],
                                      andwhere_colnames=(CONFIG_TABLENAME,))
        print(text)

    def print_csv(table):
        print(table.db.get_table_csv(table.tablename))

    def new_request(table, qaids, daids, cfgdict=None):
        request = table.depc.new_request(table.tablename, qaids, daids,
                                         cfgdict=cfgdict)
        return request

    def get_rowids_from_root(table, root_rowids, config=None):
        return table.depc.get_rowids(table.tablename, root_rowids,
                                     config=config)

    @property
    def configclass(table):
        return table.depc.configclass_dict[table.tablename]

    @property
    def requestclass(table):
        return table.depc.requestclass_dict.get(table.tablename, None)

    # ---------------------------
    # --- CONFIGURATION TABLE ---
    # ---------------------------

    def get_config_rowid(table, config=None, _debug=None):
        if isinstance(config, int):
            config_rowid = config
        else:
            config_rowid = table.add_config(config, _debug)
        return config_rowid

    def get_config_hashid(table, config_rowid_list):
        hashid_list = table.db.get(
            CONFIG_TABLE, colnames=(CONFIG_HASHID,), id_iter=config_rowid_list,
            id_colname=CONFIG_ROWID)
        return hashid_list

    def get_config_rowid_from_hashid(table, config_hashid_list):
        config_rowid_list = table.db.get(
            CONFIG_TABLE, colnames=(CONFIG_ROWID,),
            id_iter=config_hashid_list,
            id_colname=CONFIG_HASHID)
        return config_rowid_list

    def add_config(table, config, _debug=None):
        try:
            # assume config is AlgoRequest or TableConfig
            config_strid = config.get_cfgstr()
        except AttributeError:
            config_strid = ut.to_json(config)
        config_hashid = ut.hashstr27(config_strid)
        if table.depc._debug or _debug:
            print('config_strid = %r' % (config_strid,))
            print('config_hashid = %r' % (config_hashid,))
        get_rowid_from_superkey = table.get_config_rowid_from_hashid
        colnames = (CONFIG_HASHID, CONFIG_TABLENAME, CONFIG_STRID,)
        param_list = [(config_hashid, table.tablename, config_strid,)]
        config_rowid_list = table.db.add_cleanly(
            CONFIG_TABLE, colnames, param_list,
            get_rowid_from_superkey)
        config_rowid = config_rowid_list[0]
        if table.depc._debug:
            print('config_rowid_list = %r' % (config_rowid_list,))
            print('config_rowid = %r' % (config_rowid,))
        return config_rowid

    # ----------------------
    # --- GETTERS NATIVE ---
    # ----------------------

    def _get_all_rowids(table):
        pass

    #def add_rows_from_multiparent(table, parent_rowids)

    def add_rows_from_parent(table, parent_ids_, preproc_args,
                             config=None, verbose=True, _debug=None):
        """
        Lazy addition
        """
        _debug = table.depc._debug if _debug is None else _debug
        # Get requested configuration id
        config_rowid = table.get_config_rowid(config)

        initial_rowid_list = table._get_rowid(parent_ids_, config=config)

        if table.depc._debug:
            print('[deptbl.add] initial_rowid_list = %s' %
                  (ut.trunc_repr(initial_rowid_list),))
            print('[deptbl.add] config_rowid = %r' % (config_rowid,))
        # Get corresponding "dirty" parent rowids
        isdirty_list = ut.flag_None_items(initial_rowid_list)
        num_dirty = sum(isdirty_list)
        num_total = len(parent_ids_)

        fmtstr = '[deptbl.add] Add %d / %d new props to %r for config_rowid=%r'
        if num_dirty > 0:
            with ut.Indenter('[ADD]', enabled=_debug):
                if verbose or _debug:
                    tup = (num_dirty, num_total, table.tablename,
                           config_rowid)
                    print(fmtstr % tup)
                table._compute_dirty_rows(parent_ids_, preproc_args,
                                          config_rowid, isdirty_list, config)
                if verbose or _debug:
                    print('[deptbl.add] finished add')
                # Dverything is clean in the database, now get correct order.
                rowid_list = table._get_rowid(parent_ids_, config=config)
        else:
            rowid_list = initial_rowid_list
        if _debug:
            print('[deptbl.add] rowid_list = %s' % ut.trunc_repr(rowid_list))
        return rowid_list

    def _compute_dirty_rows(table, parent_ids_, preproc_args, config_rowid,
                            isdirty_list, config, verbose=True):
        """
        Does work of computing and caching dirty rowids
        >>> from dtool.depcache_table import *  # NOQA
        """
        dirty_parent_ids  = ut.compress(parent_ids_, isdirty_list)
        dirty_preproc_args = ut.compress(preproc_args, isdirty_list)
        # CALL EXTERNAL PREPROCESSING / GENERATION FUNCTION
        try:
            # Pack arguments into column-wise order to send to the func
            argsT = zip(*dirty_preproc_args)
            argsT = list(argsT)  # TODO: remove
            if table._asobject:
                # Convinience
                argsT = [table.depc.get_obj(parent, rowids)
                         for parent, rowids in zip(table.parents(),
                                                   dirty_parent_ids)]
            # hack out config if given a request
            config_ = config.config if hasattr(config, 'config') else config
            # call registered worker function
            proptup_gen = table.preproc_func(table.depc, *argsT,
                                             config=config_)
            #proptup_gen = list(proptup_gen)
            # Append rowids and rectify nested and external columns
            dirty_params_iter = table.prepare_storage(
                dirty_parent_ids, proptup_gen, dirty_preproc_args,
                config_rowid, config_)
            #dirty_params_iter = list(dirty_params_iter)
            # Break iterator into chunks
            chunksize = ut.ifnone(len(dirty_parent_ids), table.chunksize)
            nInput = len(dirty_parent_ids)
            # Report computation progress
            prog_iter = ut.ProgChunks(dirty_params_iter, chunksize, nInput,
                                      lbl='add %s chunk' % (table.tablename))
            # TODO: Separate into func which can be specified as a callback.
            #colnames =
            intern_colnames = ut.take_column(table.internal_col_attrs, 'intern_colname')
            insertable_flags = [not colattr.get('isprimary') for colattr in table.internal_col_attrs]
            colnames = tuple(ut.compress(intern_colnames, insertable_flags))
            for dirty_params_chunk in prog_iter:
                # None data means that there was an error for a specific row
                if ALLOW_NONE_YIELD:
                    dirty_params_chunk = ut.filter_Nones(dirty_params_chunk)
                nInput = len(dirty_params_chunk)
                table.db._add(table.tablename, colnames, dirty_params_chunk,
                              nInput=nInput)
        except Exception as ex:
            ut.printex(ex, 'error in add_rowids', keys=[
                'table',
                'table.parents()',
                'parent_ids_',
                'config',
                'argsT',
                'config_rowid',
                'dirty_parent_ids',
                'table.preproc_func'
                'preproc_args',
            ])
            raise

    def prepare_storage(table, dirty_parent_ids, proptup_gen,
                        dirty_preproc_args, config_rowid, config):
        """
        Converts output from ``preproc_func`` to data that can be stored in SQL
        """
        if table.default_to_unpack:
            # Hack for tables explicilty specified with a single column
            proptup_gen = (None if data is None else (data,)
                           for data in proptup_gen)
        # Flatten nested columns
        #if len(table._nested_idxs) > 0:
        if any(table.get_data_col_attr('isnested')):
            proptup_gen = table._prepare_storage_nested(proptup_gen)
        # Write external columns
        if len(table.extern_write_funcs) > 0:
            proptup_gen = table._prepare_storage_extern(dirty_parent_ids,
                                                        config_rowid, config,
                                                        proptup_gen)
        # Concatenate data with internal rowids / config-id
        for parent_id_, data_cols, args_ in zip(dirty_parent_ids, proptup_gen,
                                                dirty_preproc_args):
            try:
                if ALLOW_NONE_YIELD and data_cols is None:
                    yield None
                    continue

                multi_parent_flags = table.get_parent_col_attr('ismulti')
                multi_args = ut.compress(args_, multi_parent_flags)
                parent_extra = tuple(ut.flatten([(len(arg), 'not stored')
                                                 for arg in multi_args]))
                yield parent_id_ + (config_rowid,) + data_cols + parent_extra
            except Exception as ex:
                ut.printex(ex, 'cat error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise

    def _prepare_storage_nested(table, proptup_gen):
        """
        Hack for when a sql schema has tuples defined in it.
        Accepts nested tuples and flattens them to fit into the sql tables
        """
        nCols = len(table.data_colnames)
        idxs1 = ut.where(table.get_data_col_attr('isnested'))
        idxs2 = ut.index_complement(idxs1, nCols)
        for data in proptup_gen:
            if ALLOW_NONE_YIELD and data is None:
                yield None
                continue
            # Split data into nested and unnested columns
            unnested_cols = list(zip(ut.take(data, idxs2)))
            nested_cols = ut.take(data, idxs1)
            grouped_items = [nested_cols, unnested_cols]
            groupxs = [idxs1, idxs2]
            # Flatten nested columns
            unflat = ut.ungroup(grouped_items, groupxs, nCols - 1)
            # Recombine the data
            data_new = tuple(ut.flatten(unflat))
            yield data_new

    def _prepare_storage_extern(table, dirty_parent_ids, config_rowid,
                                config, proptup_gen):
        """
        Writes external data to disk if write function is specified.
        """
        internal_data_col_attrs = table.internal_data_col_attrs
        writable_flags = ut.dict_take_column(internal_data_col_attrs,
                                             'write_func', False)
        extern_colattrs = ut.compress(internal_data_col_attrs, writable_flags)
        extern_colnames = ut.dict_take_column(extern_colattrs, 'colname')
        extern_writers = ut.dict_take_column(extern_colattrs, 'write_func')

        nCols = len(internal_data_col_attrs)
        idxs1 = ut.where(writable_flags)
        #extern_colnames = list(table.extern_write_funcs.keys())
        #nCols = len(table._internal_data_colnames)
        #idxs1 = [
        #    ut.listfind(table._internal_data_colnames, col + EXTERN_SUFFIX)
        #    for col in extern_colnames
        #]
        idxs2 = ut.index_complement(idxs1, nCols)
        extern_fnames_list = list(zip(*[
            table._get_extern_fnames(dirty_parent_ids, config_rowid, config,
                                     col)
            for col in extern_colnames
        ]))
        # get extern cache directory and fpaths
        cache_dpath = table.depc.cache_dpath
        extern_dname = 'extern_' + table.tablename
        extern_dpath = join(cache_dpath, extern_dname)
        ut.ensuredir(extern_dpath, verbose=False or table.depc._debug)
        extern_fpaths_list = [
            [join(extern_dpath, fname) for fname in fnames]
            for fnames in extern_fnames_list
        ]

        for data, extern_fpaths in zip(proptup_gen, extern_fpaths_list):
            pass
            if ALLOW_NONE_YIELD and data is None:
                yield None
                continue
            normal_data = ut.take(data, idxs2)
            extern_data = ut.take(data, idxs1)
            # Write external data to disk
            try:
                for obj, fpath, col, write_func in zip(extern_data,
                                                       extern_fpaths,
                                                       extern_colnames,
                                                       extern_writers):
                    print('WRITING %r' % (col,))
                    print('fpath = %r' % (fpath,))
                    #table.data_col_attrs
                    #write_func = table.requestable_col_attrs()[col]['write_func']
                    #write_func = table.extern_write_funcs[col]
                    #write_func(obj, fpath, True)
                    #print('fpath = %r' % (fpath,))
                    write_func(fpath, obj)
                    ut.assert_exists(fpath, verbose=True)
                    #verbose=True)
            except Exception as ex:
                ut.printex(ex, 'write extern col error', keys=[
                    'config_rowid', 'data'])
                raise
            # Return path instead of data
            grouped_items = [extern_fpaths, normal_data]
            groupxs = [idxs1, idxs2]
            data_new = tuple(ut.ungroup(grouped_items, groupxs, nCols - 1))
            yield data_new

    def _get_extern_fnames(table, parent_rowids, config_rowid, config,
                           colname=None):
        config_hashid = table.get_config_hashid([config_rowid])[0]
        prefix = table.tablename
        if colname is not None:
            prefix += '_' + colname
        fmtstr = '{prefix}_id={rowids}_{config_hashid}{ext}'
        # HACK: check if the config specifies the extension type
        extkey = table.extern_ext_config_keys.get(colname, 'ext')
        ext = config[extkey] if extkey in config else '.cPkl'
        fname_list = [
            fmtstr.format(prefix=prefix,
                          rowids='_'.join(list(map(str, rowids))),
                          config_hashid=config_hashid, ext=ext)
            for rowids in parent_rowids
        ]
        return fname_list

    def get_rowid(table, parent_rowids, config=None, ensure=True, eager=True,
                  nInput=None, recompute=False, _debug=None):
        r"""
        get feat rowids of chip under the current state configuration
        if ensure is True, this function is equivalent to add_rows_from_parent

        Args:
            parent_rowids (list): list of tuples with the parent rowids as the
                    value of each tuple
            config (None): (default = None)
            ensure (bool): eager evaluation if True (default = True)
            eager (bool): (default = True)
            nInput (int): (default = None)
            recompute (bool): (default = False)
            _debug (None): (default = None)

        Returns:
            list: rowid_list

        CommandLine:
            python -m dtool.depcache_table --exec-get_rowid

        Example5:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> # Test get behavior for multi (model) tables
            >>> depc = testdata_depc()
            >>> table = depc['multitest']
            >>> config = table.configclass()
            >>> exec(ut.execstr_funckw(table.get_rowid), globals())
            >>> _debug = True
            >>> depc.get_rowids('chip', [1, 2, 3, 4, 5])
            >>> depc.get_rowids('spam', [2, 3])
            >>> parent_rowids = [((1, 2, 3, 4), 3, (1, 2,), 1), (None, None, None, 1),
            >>>                  ((1, 2, 3, 4, 5), None, (1, 2,), 1), ((1, 2,), 1, (2, 3,), 1)]
            >>> rowids = table.get_rowid(parent_rowids, config=config, _debug=_debug)
            >>> result = ('rowids = %r' % (rowids,))
            >>> indexer = table.get_row_data(rowids)
            >>> print('indexer = %r' % (indexer,))
            >>> print(result)
            rowids = [1, None, None, 2]
        """
        _debug = table.depc._debug if _debug is None else _debug
        if _debug:
            print('[deptbl.get_rowid] Lookup %s rowids from superkey with %d parent' % (
                table.tablename, len(parent_rowids)))
            print('[deptbl.get_rowid] config = %r' % (config,))
            print('[deptbl.get_rowid] ensure = %r' % (ensure,))

        if ALLOW_NONE_YIELD:
            # Force entire row to be none if any are none
            anyNone_flags = [x is None or any(ut.flag_None_items(x))
                             for x in parent_rowids]
            idxs2 = ut.where(anyNone_flags)
            idxs1 = ut.index_complement(idxs2, len_=len(parent_rowids))
            valid_parent_ids_ = ut.take(parent_rowids, idxs1)
        else:
            valid_parent_ids_ = parent_rowids

        preproc_args = valid_parent_ids_

        if table.ismulti:
            # Convert any parent-id containing multiple values into a hash of uuids
            multi_parent_flags = table.get_parent_col_attr('ismulti')
            num_parents = len(multi_parent_flags)
            multi_parent_colxs = ut.where(multi_parent_flags)
            multi_parent_colxs1 = table.multi_parent_colxs
            assert multi_parent_colxs1 == multi_parent_colxs

            normal_colxs = ut.index_complement(multi_parent_colxs, num_parents)
            multi_parents = [ut.apply_grouping(ids_, multi_parent_colxs) for ids_ in valid_parent_ids_]
            normal_parents = [ut.apply_grouping(ids_, normal_colxs) for ids_ in valid_parent_ids_]
            # TODO: give each table a uuid getter function that derives from get_root_uuids
            multicol_tables = ut.take(table.parents(), multi_parent_colxs)
            parent_uuid_getters = [table.depc.get_root_uuid if col == table.depc.root else ut.identity
                                   for col in multicol_tables]
            #parent_uuid_getters = [table.depc.get_root_uuid for idx in table.multi_parent_colxs]
            #[table.depc[col].get_internal_columns([2, 3], (CONFIG_ROWID,)) for col in multicol_tables]
            parent_uuids_list = [[uuid_getter(ids_) for uuid_getter, ids_ in
                                  zip(parent_uuid_getters, ids_tup)]
                                 for ids_tup in multi_parents]
            multiset_uuid_list = [[ut.hashable_to_uuid(uuids) for uuids in parent_uuids_tup]
                                  for parent_uuids_tup in parent_uuids_list]
            # preproc args are usually the same as parent ids.  Model tables
            # are the exception.
            #parent_num = len(parent_rowids)
            #parent_ids_ = [(multiset_uuid, parent_num)]
            #parent_ids_ = [(multiset_uuid,) for multiset_uuid in multiset_uuid_list]
            parent_ids_ = [tuple(
                ut.ungroup([uuids, normalids], [multi_parent_colxs, normal_colxs], num_parents - 1))
                for uuids, normalids in zip(multiset_uuid_list, normal_parents)
            ]
        else:
            parent_ids_ = valid_parent_ids_

        if recompute:
            # get existing rowids, delete them, recompute the request
            rowid_list_ = table._get_rowid(parent_ids_, config=config,
                                           eager=True, nInput=None)
            table.delete_rows(rowid_list_)

        if ensure or recompute:
            rowid_list_ = table.add_rows_from_parent(
                parent_ids_, preproc_args, config=config)
        else:
            rowid_list_ = table._get_rowid(
                parent_ids_, config=config, eager=eager, nInput=nInput)

        if ALLOW_NONE_YIELD:
            rowid_list = ut.ungroup([rowid_list_], [idxs1], len(parent_rowids) - 1)
        else:
            rowid_list = rowid_list_
        return rowid_list

    def _get_rowid(table, parent_ids_, config=None, eager=True, nInput=None,
                   _debug=None):
        colnames = (table.rowid_colname,)
        config_rowid = table.get_config_rowid(config=config)
        _debug = table.depc._debug if _debug is None else _debug
        if _debug:
            print('_get_rowid')
            print('_get_rowid table.tablename = %r ' % (table.tablename,))
            print('_get_rowid parent_ids_ = %s' % (ut.trunc_repr(parent_ids_)))
            print('_get_rowid config = %s' % (config))
            print('_get_rowid table.rowid_colname = %s' % (table.rowid_colname))
            print('_get_rowid config_rowid = %s' % (config_rowid))
        andwhere_colnames = table.superkey_colnames
        params_iter = (ids_ + (config_rowid,) for ids_ in parent_ids_)
        params_iter = list(params_iter)
        #print('**params_iter = %r' % (params_iter,))
        rowid_list = table.db.get_where2(table.tablename, colnames,
                                         params_iter, andwhere_colnames,
                                         eager=eager, nInput=nInput)
        if _debug:
            print('_get_rowid rowid_list = %s' % (ut.trunc_repr(rowid_list)))
        return rowid_list

    def delete_rows(table, rowid_list, delete_extern=False, verbose=None):
        """
        CommandLine:
            python -m dtool.depcache_table --exec-delete_rows

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> #table = depc['keypoint']
            >>> table = depc['chip']
            >>> exec(ut.execstr_funckw(table.delete_rows), globals())
            >>> tablename = table.tablename
            >>> graph = depc.make_graph(with_implicit=False)
            >>> config1 = None
            >>> config2 = table.configclass(version=-1)
            >>> config3 = table.configclass(version=-1, ext='.jpg')
            >>> config4 = table.configclass(ext='.jpg')
            >>> # Create several configs of rowid
            >>> aids = [1, 2, 3]
            >>> depc.get_rowids('spam', aids, config=config1)
            >>> depc.get_rowids('spam', aids, config=config2)
            >>> depc.get_rowids('spam', aids, config=config3)
            >>> depc.get_rowids('spam', aids, config=config4)
            >>> # Delete the png configs
            >>> rowid_list1 = depc.get_rowids(table.tablename, aids,
            >>>                               config=config2)
            >>> rowid_list2 = depc.get_rowids(table.tablename, aids,
            >>>                               config=config1)
            >>> rowid_list = rowid_list1 + rowid_list2
            >>> assert len(ut.setintersect_ordered(rowid_list1, rowid_list2)) == 0
            >>> table.delete_rows(rowid_list)
        """
        #import networkx as nx
        #from dtool.algo.preproc import preproc_feat
        if table.on_delete is not None:
            table.on_delete()
        if ut.NOT_QUIET:
            print('Requested delete of %d rows from %s' % (
                len(rowid_list), table.tablename))
            print('delete_extern = %r' % (delete_extern,))

        # TODO:
        # REMOVE EXTERNAL FILES
        internal_colnames = table.get_intern_data_col_attr('intern_colname')
        is_extern = table.get_intern_data_col_attr('is_external_pointer')
        extern_colnames = tuple(ut.compress(internal_colnames, is_extern))
        if len(extern_colnames) > 0:
            uri_list = table.get_internal_columns(rowid_list,
                                                  extern_colnames,
                                                  unpack_scalars=False,
                                                  keepwrap=False)
            fpath_list = ut.flatten(uri_list)
            if delete_extern:
                print('fpath_list = %r' % (fpath_list,))
                print('deleting internal files')
                ut.remove_file_list(fpath_list)
            else:
                print('would delete fpath_list = %r' % (fpath_list,))

        # DELETE EXPLICITLY DEFINED CHILDREN
        # (TODO: handle implicit definitions)
        if True:
            def get_child_partial_rowids(child_table, rowid_list,
                                         parent_colnames):
                colnames = (child_table.rowid_colname,)
                andwhere_colnames = parent_colnames
                params_iter = ((rowid,) for rowid in rowid_list)
                params_iter = list(params_iter)
                child_rowids = table.db.get_where2(child_table.tablename,
                                                   colnames, params_iter,
                                                   andwhere_colnames)
                return child_rowids

            for child in table.children:
                child_table = table.depc[child]
                if not child_table.ismulti:
                    # Hack, wont work for vsone / multisets
                    parent_colnames = (child_table.parent[table.tablename]['intern_colname'],)
                    child_rowids = get_child_partial_rowids(child_table,
                                                            rowid_list,
                                                            parent_colnames)
                    child_table.delete_rows(child_rowids)

        if ut.NOT_QUIET:
            print('Deleting %d non-None rows from %s' % (
                len(ut.filter_Nones(rowid_list)), table.tablename))

        # Finalize: Delete rows from this table
        table.db.delete_rowids(table.tablename, rowid_list)
        num_deleted = len(ut.filter_Nones(rowid_list))
        return num_deleted

    def get_row_data(table, tbl_rowids, colnames=None, _debug=None,
                     read_extern=True, extra_tries=1, eager=True,
                     nInput=None):
        r"""
        colnames = ('mask', 'size')
        FIXME: unpacking is confusing with sql controller
        TODO: Clean up and allow for eager=False

        CommandLine:
            python -m dtool.depcache_table --exec-get_row_data

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> table = depc['chip']
            >>> tbl_rowids = depc.get_rowids('chip', [1, 2, 3])
            >>> #tbl_rowids += [None]
            >>> #colnames = ('size_1', 'size', 'chip', 'chip' + EXTERN_SUFFIX)
            >>> colnames = ('size_1', 'size', 'chip' + EXTERN_SUFFIX, 'chip')
            >>> _debug = True
            >>> read_extern = True
            >>> extra_tries = 1
            >>> kwargs = dict(read_extern=read_extern,
            >>>               extra_tries=extra_tries, _debug=_debug)
            >>> prop_list = table.get_row_data(tbl_rowids, colnames, **kwargs)
            >>> prop_list0 = ut.take_column(prop_list, [0, 1, 2]) # take small data
            >>> print(ut.repr2(prop_list0, nl=1))

            [
                [372, (545, 372), '/home/joncrall/code/dtool/DEPCACHE/extern_chip/chip_chip_id=1_pyrappzicqoskdjq.png'],
                [2453, (1707, 2453), '/home/joncrall/code/dtool/DEPCACHE/extern_chip/chip_chip_id=2_pyrappzicqoskdjq.png'],
                [390, (520, 390), '/home/joncrall/code/dtool/DEPCACHE/extern_chip/chip_chip_id=3_pyrappzicqoskdjq.png'],
            ]
        """
        _debug = table.depc._debug if _debug is None else _debug
        if _debug:
            print(('Get col of tablename=%r, colnames=%r with '
                   'tbl_rowids=%s') % (table.tablename, colnames,
                                       ut.trunc_repr(tbl_rowids)))
        ####
        # Resolve requested column names
        unpack_columns = table.default_to_unpack
        if colnames is None:
            requested_colnames = table.data_colnames
        elif isinstance(colnames, six.string_types):
            requested_colnames = (colnames,)
            unpack_columns = True
        else:
            requested_colnames = colnames

        if _debug:
            print('requested_colnames = %r' % (requested_colnames,))
        ####
        # Map requested colnames flat to internal colnames

        if True:
            total = 0
            intern_colnames1 = []
            extern_resolve_colxs1 = []
            nesting_xs1 = []  # how to resolve unnesting

            for c in requested_colnames:
                if c in table.external_to_internal:
                    intern_colnames1.append([table.external_to_internal[c]])
                    read_func = table.extern_read_funcs[c]
                    extern_resolve_colxs1.append((total, read_func))
                    nesting_xs1.append(total)
                    total += 1
                elif c in table.nested_to_flat:
                    nest = table.nested_to_flat[c]
                    nesting_xs1.append(list(range(total, total + len(nest))))
                    intern_colnames1.append(nest)
                    total += len(nest)
                else:
                    nesting_xs1.append(total)
                    intern_colnames1.append([c])
                    total += 1

            flat_intern_colnames1 = tuple(ut.flatten(intern_colnames1))

        requestable_col_attrs = table.requestable_col_attrs()
        requested_colattrs = ut.take(requestable_col_attrs, requested_colnames)
        intern_colxs = [xs if ut.isiterable(xs) else [xs] for xs in ut.take_column(requested_colattrs, 'intern_colx')]
        nested_offsets_end = ut.cumsum(ut.lmap(len, intern_colxs))
        nested_offsets_start = [0] + nested_offsets_end[:-1]
        extern_flags = [colattr.get('is_extern', False) for colattr in requested_colattrs]
        extern_resolve_colxs_ = ut.compress(nested_offsets_start, extern_flags)
        extern_read_funcs = ut.take_column(ut.compress(requested_colattrs, extern_flags), 'read_func')
        # TODO: this can be cleaned up
        nesting_xs = [x1 if x2 - x1 == 1 else list(range(x1, x2)) for x1, x2 in zip(nested_offsets_start, nested_offsets_end)]
        intern_colnames = ut.unflat_take(ut.take_column(table.internal_col_attrs, 'intern_colname'), intern_colxs)
        extern_resolve_colxs = list(zip(extern_resolve_colxs_, extern_read_funcs))
        flat_intern_colnames = tuple(ut.flatten(intern_colnames))

        if True:
            assert intern_colnames == intern_colnames1
            assert nesting_xs == nesting_xs1
            assert flat_intern_colnames == flat_intern_colnames1
            assert extern_resolve_colxs == extern_resolve_colxs1

        if _debug:
            print('[deptbl.get_row_data] flat_intern_colnames = %r' %
                  (flat_intern_colnames,))

        if ALLOW_NONE_YIELD:
            nonNone_flags = ut.flag_not_None_items(tbl_rowids)
            nonNone_tbl_rowids = ut.compress(tbl_rowids, nonNone_flags)
            idxs1 = ut.where(nonNone_flags)
        else:
            nonNone_tbl_rowids = tbl_rowids
            idxs1 = []
        idxs2 = ut.index_complement(idxs1, len(tbl_rowids))

        ####
        # Read data stored in SQL
        # FIXME: understand unpack_scalars and keepwrap
        eager = True
        nInput = None
        raw_prop_list = table.get_internal_columns(
            nonNone_tbl_rowids, flat_intern_colnames, eager, nInput,
            unpack_scalars=True, keepwrap=True)
        ####
        # Read data specified by any external columns
        prop_listT = list(zip(*raw_prop_list))
        for extern_colx, read_func in extern_resolve_colxs:
            if _debug:
                print('[deptbl.get_row_data] read_func = %r' % (read_func,))
            data_list = []
            failed_list = []
            for uri in prop_listT[extern_colx]:
                # FIXME: only do this for a localpath
                uri_full = join(table.depc.cache_dpath, uri)
                try:
                    if read_extern:
                        data = read_func(uri_full)
                    else:
                        ut.assertpath(uri_full)
                        data = uri_full
                except Exception as ex:
                    ut.printex(ex, 'failed to load external data',
                               iswarning=(extra_tries > 0),
                               keys=['extra_tries', 'uri', 'uri_full',
                                     (exists, 'uri_full'), 'read_func'])

                    if extra_tries == 0:
                        raise
                    failed_list.append(True)
                    data = None
                else:
                    failed_list.append(False)
                data_list.append(data)
            if any(failed_list):
                # FIXME: should directly recompute the data in the rows
                # rather than deleting the rowids.  Need the parent ids and
                # config to do that.
                failed_rowids = ut.compress(nonNone_tbl_rowids, failed_list)
                table.delete_rows(failed_rowids, delete_extern=False)
                raise Exception('Non existant data on disk. Need to recompute rows')
            prop_listT[extern_colx] = data_list
        ####
        # Unflatten data into any given nested structure
        nested_proplistT = ut.list_unflat_take(prop_listT, nesting_xs)
        for tx in ut.where([isinstance(xs, list) for xs in nesting_xs]):
            nested_proplistT[tx] = list(zip(*nested_proplistT[tx]))
        prop_list = list(zip(*nested_proplistT))
        ####
        # Unpack single column datas if requested
        if unpack_columns:
            prop_list = [None if p is None else p[0] for p in prop_list]

        if ALLOW_NONE_YIELD:
            prop_list = ut.ungroup(
                [prop_list, [None] * len(idxs2)],
                [idxs1, idxs2], len(tbl_rowids) - 1)
        return prop_list

    def get_internal_columns(table, tbl_rowids, colnames=None, eager=True,
                             nInput=None, unpack_scalars=True, keepwrap=False):
        prop_list = table.db.get(
            table.tablename, colnames, tbl_rowids,
            id_colname=table.rowid_colname, eager=eager, nInput=nInput,
            unpack_scalars=unpack_scalars, keepwrap=keepwrap)
        return prop_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.depcache_table
        python -m dtool.depcache_table --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
