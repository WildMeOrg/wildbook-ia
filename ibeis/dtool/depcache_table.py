# -*- coding: utf-8 -*-
"""
Module contining DependencyCacheTable

python -m dtool.depcache_control --exec-make_graph --show
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
    GRACE_PERIOD = 5
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
        >>> table = depc['vsmany']
        >>> print(table)
        >>> table = depc['spam']
        >>> table = depc['vsone']
        >>> print(table)
    """

    def __init__(table, depc=None, parent_tablenames=None, tablename=None,
                 data_colnames=None, data_coltypes=None, preproc_func=None,
                 docstr='no docstr', fname=None, asobject=False,
                 chunksize=None, ismulti=False,
                 isinteractive=False, default_to_unpack=False):
        """ recieves kwargs from depc._register_prop """
        try:
            table.db = None
        except Exception:
            # HACK: jedi type hinting. Need to have non-obvious condition
            from dtool.sql_control import SQLDatabaseController
            table.db = SQLDatabaseController()
        table.fpath_to_db = {}

        table.parent_tablenames = parent_tablenames
        table.tablename = tablename
        table.data_colnames = tuple(data_colnames)
        table.data_coltypes = data_coltypes
        table.preproc_func = preproc_func
        table.on_delete = None
        table.ismulti = ismulti
        table.isinteractive = isinteractive
        table.default_to_unpack = default_to_unpack
        table.docstr = docstr
        table.fname = fname
        table.depc = depc
        table.subproperties = {}
        table.chunksize = chunksize
        table._asobject = asobject
        # SQL Internals
        table._internal_parent_id_colnames = ()
        table._internal_parent_id_coltypes = ()
        table._internal_parent_extra_colnames = ()
        table._internal_parent_extra_coltypes = ()
        table._parent_id_prefixes = ()
        table._internal_data_colnames = ()
        table._internal_data_coltypes = ()
        table._nested_idxs = []
        table.sqldb_fpath = None
        table.extern_read_funcs = {}
        table.extern_write_funcs = {}
        table.extern_ext_config_keys = {}
        table.extern_io_classes = {}
        # Update internals
        table._update_parentcol_internal()
        table._update_datacol_internal()
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
                                       table.ismulti else '', num_cols)

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
                if len(rowid_args) != len(table.parents):
                    print('table.preproc_func = %r' %
                          (table.preproc_func,))
                    print('args = %r' % (args,))
                    print('rowid_args = %r' % (rowid_args,))
                    msg = (
                        ('preproc function for table=%s must have as many '
                         'rowids %d args as parents %d') % (
                            table.tablename, len(rowid_args),
                             len(table.parents))
                    )
                    raise AssertionError(msg)
        for colname, cls in table.extern_io_classes.items():
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
        primary_coldef = [(table.rowid_colname, 'INTEGER PRIMARY KEY')]
        internal_parent_id_coldef = list(zip(
            table._internal_parent_id_colnames,
            table._internal_parent_id_coltypes,
        ))
        config_coldef = [
            (CONFIG_ROWID, 'INTEGER DEFAULT 0')
        ]
        internal_parent_extras_coldef = list(zip(
            table._internal_parent_extra_colnames,
            table._internal_parent_extra_coltypes,
        ))
        internal_data_coldef = list(zip(
            table._internal_data_colnames,
            table._internal_data_coltypes
        ))
        coldef_list = (
            primary_coldef +
            internal_parent_id_coldef +
            config_coldef +
            internal_data_coldef +
            internal_parent_extras_coldef
        )
        add_table_kw = ut.odict([
            ('tablename', table.tablename,),
            ('coldef_list', coldef_list,),
            ('docstr', table.docstr,),
            ('superkeys', [table.superkey_colnames],),
            ('dependson', table.parents),
        ])
        return add_table_kw

    def _update_datacol_internal(table):
        """
        Constructs the columns needed to represent relationship to data

        Infers interal properties about this table given the colnames and
        datatypes

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> for table in depc.tables:
            >>>     print('----')
            >>>     table._update_datacol_internal()
            >>>     print(table)
            >>>     print('data_colnames = %r' % (table.data_colnames,))
            >>>     print('data_coltypes = %r' % (table.data_coltypes,))
            >>>     print('_internal_data_colnames = %r' % (table._internal_data_colnames,))
            >>>     print('_internal_data_coltypes = %r' % (table._internal_data_coltypes,))
            >>>     print('nested_to_flat = %r' % (table.nested_to_flat,))
            >>>     print('external_to_internal = %r' % (table.external_to_internal,))
            >>>     print('extern_read_funcs = %r' % (table.extern_read_funcs,))
            >>> table = depc['probchip']
            >>> table = depc['spam']
            >>> table = depc['vsone']
        """
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
        for colx, (colname, coltype) in _iter:
            # Check column input subtypes
            is_tuple     = isinstance(coltype, tuple)
            is_func      = ut.is_func_or_method(coltype)
            is_externtup = is_tuple and coltype[0] == 'extern'
            is_functup   = is_tuple and ut.is_func_or_method(coltype[0])
            # Check column input main types
            is_normal   = coltype in lite.TYPE_TO_SQLTYPE
            #is_normal   = not (is_tuple or is_func)
            is_nested   = is_tuple and not (is_func or is_externtup)
            is_external = (is_func or is_functup or is_externtup)
            # Switch on input types
            if is_normal:
                # Normal non-nested column
                internal_data_colnames.append(colname)
                internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[coltype])
            elif is_nested:
                # Nested non-function normal columns
                table._nested_idxs.append(colx)
                nested_to_flat[colname] = []
                for count, subtype in enumerate(coltype):
                    flat_colname = '%s_%d' % (colname, count)
                    nested_to_flat[colname].append(flat_colname)
                    internal_data_colnames.append(flat_colname)
                    internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[subtype])
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
                internal_data_colnames.append(intern_colname)
                internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[str])
            else:
                # External class column
                assert (hasattr(coltype, '__getstate__') and
                        hasattr(coltype, '__setstate__')), (
                        'External classes must have __getstate__ and '
                        '__setstate__ methods')
                read_func, write_func = make_extern_io_funcs(coltype)
                extern_io_classes[colname]  = coltype
                extern_read_funcs[colname]  = read_func
                extern_write_funcs[colname] = write_func
                intern_colname = colname + EXTERN_SUFFIX
                external_to_internal[colname] = intern_colname
                internal_data_colnames.append(intern_colname)
                internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[str])
                #raise AssertionError('external class columns')
        # Set new internal data properties of the table
        assert len(set(internal_data_colnames)) == len(internal_data_colnames)
        assert len(internal_data_coltypes) == len(internal_data_colnames)
        table.extern_read_funcs = extern_read_funcs
        table.extern_write_funcs = extern_write_funcs
        table.extern_ext_config_keys = extern_ext_config_keys
        table.extern_io_classes = extern_io_classes
        table.external_to_internal = external_to_internal
        table.nested_to_flat = nested_to_flat
        table._internal_data_colnames = tuple(internal_data_colnames)
        table._internal_data_coltypes = tuple(internal_data_coltypes)
        table._assert_self()

    def _update_parentcol_internal(table):
        """
        constructs the columns needed to represent relationship to parents
        """
        # Handle dependencies when a parents are pairwise between tables
        parent_id_prefixs = []
        seen_ = ut.ddict(lambda: 1)
        colhist = ut.dict_hist(table.parents)
        for col in table.parents:
            if colhist[col] > 1:
                # Hack such that duplicate column names receive a count index
                prefix = col + str(seen_[col])
                seen_[col] += 1
            else:
                prefix = col
            # maybe add set to multicol?
            parent_id_prefixs.append(prefix)

        _internal_parent_id_colnames = []
        _internal_parent_id_coltypes = []

        _internal_parent_extra_colnames = []
        _internal_parent_extra_coltypes = []

        # Handle case when parents are a set of ids
        for prefix in parent_id_prefixs:
            # HACK. ALL COLUMNS ARE MULTI
            column_ismulti = table.ismulti  # or (prefix.endswith('*'))
            if not column_ismulti:
                # Normal case when dependencies are one to one
                _internal_parent_id_colnames.append(prefix + '_rowid')
                _internal_parent_id_coltypes.append('INTEGER NOT NULL')
            else:
                # Case when dependencies are many to one
                # hash of set items
                _internal_parent_id_colnames.append(prefix + '_setuuid')
                _internal_parent_id_coltypes.append('INTEGER NOT NULL')
                # number of items in set
                _internal_parent_extra_colnames.append(prefix + '_setsize')
                _internal_parent_extra_colnames.append(prefix + '_setfpath')
                # path to indiviual parts
                _internal_parent_extra_coltypes.append('INTEGER NOT NULL')
                _internal_parent_extra_coltypes.append('TEXT')
        table._internal_parent_id_colnames = tuple(
            _internal_parent_id_colnames)
        table._internal_parent_id_coltypes = tuple(
            _internal_parent_id_coltypes)
        table._internal_parent_extra_colnames = tuple(
            _internal_parent_extra_colnames)
        table._internal_parent_extra_coltypes = tuple(
            _internal_parent_extra_coltypes)
        table._parent_id_prefixes = tuple(parent_id_prefixs)

    def clear_table(table):
        """
        Deletes all data in this table
        """
        print('Clearing data in %r' % (table,))
        table.db.drop_table(table.tablename)
        table.db.add_table(**table.get_addtable_kw())

    @property
    def parents(table):
        return table.parent_tablenames

    @property
    def parent_id_colnames(table):
        return table._internal_parent_id_colnames

    @property
    def parent_id_prefixes(table):
        return table._parent_id_prefixes

    @property
    def children(table):
        import networkx as nx
        graph = table.depc.make_graph(with_implicit=False)
        children_tablenames = nx.neighbors(graph, table.tablename)
        return children_tablenames

    @property
    def columns(table):
        return table.data_colnames

    @property
    def extern_columns(table):
        return list(table.external_to_internal.keys())

    @property
    def rowid_colname(table):
        """ rowid of this table used by other dependant tables """
        return table.tablename + '_rowid'

    @property
    def superkey_colnames(table):
        return table.parent_id_colnames + (CONFIG_ROWID,)

    @property
    def _table_colnames(table):
        return

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
        return table.depc.requestclass_dict[table.tablename]

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

    def add_rows_from_parent(table, parent_ids_, preproc_args, idxs1, idxs2,
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
        if ALLOW_NONE_YIELD:
            rowid_list = ut.ungroup([rowid_list], [idxs1], len(parent_ids_) - 1)
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
                         for parent, rowids in zip(table.parents,
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
            colnames = (
                table.superkey_colnames +
                table._internal_data_colnames +
                table._internal_parent_extra_colnames
            )
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
                'table.parents',
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
        if len(table._nested_idxs) > 0:
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
                col_is_multi = table.ismulti  # hack
                multi_args = [arg for arg in args_ if col_is_multi]
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
        idxs1 = table._nested_idxs
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
        #extern_colnames = list(table.extern_io_classes.keys())
        extern_colnames = list(table.extern_write_funcs.keys())
        nCols = len(table._internal_data_colnames)
        idxs1 = [
            ut.listfind(table._internal_data_colnames, col + EXTERN_SUFFIX)
            for col in extern_colnames
        ]
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
            if ALLOW_NONE_YIELD and data is None:
                yield None
                continue
            normal_data = ut.take(data, idxs2)
            extern_data = ut.take(data, idxs1)
            # Write external data to disk
            try:
                for obj, fpath, col in zip(extern_data, extern_fpaths,
                                           extern_colnames):
                    print('WRITING %r' % (col,))
                    print('fpath = %r' % (fpath,))
                    write_func = table.extern_write_funcs[col]
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

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> table = depc['vsmany']
            >>> request = table.new_request([1, 2], [3, 4])
            >>> results = request.execute()
            >>> print(results)

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> config = {}
            >>> table = depc['spam']
            >>> rowid_dict = depc.get_all_descendant_rowids('spam', [1, 2], levels_up=1)
            >>> parent_rowids = list(zip(*ut.dict_take(rowid_dict, table.parents)))
            >>> rowids = table.get_rowid(parent_rowids)
            >>> print(rowids)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> # Test normal case
            >>> depc = testdata_depc()
            >>> table = depc['notch']
            >>> config = table.configclass()
            >>> _debug = True
            >>> parent_rowids = list(zip([1, 2, 3, 4]))
            >>> rowids = table.get_rowid(parent_rowids, config=config, _debug=_debug)
            >>> print(rowids)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> table = depc['vsone']
            >>> _debug = True
            >>> # Test pairwise parent case
            >>> config = request = depc.new_request('vsone', [1, 2], [2, 3, 4])
            >>> parent_rowids = request.parent_rowids
            >>> ut.colorprint('Testing add_rows via getters', 'blue')
            >>> rowids = table.get_rowid(parent_rowids, config=request, _debug=_debug)
            >>> match_list = request.execute()
            >>> print(match_list)
            >>> print(rowids)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> # Test get behavior for ismulti=True (model) tables
            >>> depc = testdata_depc()
            >>> table = depc['nnindexer']
            >>> config = table.configclass()
            >>> _debug = True
            >>> parent_rowids = [((1, 2, 3, 4),)]
            >>> #exec(ut.execstr_funckw(depc.add_rows_from_parent), globals())
            >>> #ut.exec_func(table.add_rows_from_parent, globals(), sentinal='fmtstr = ')
            >>> rowids = table.get_rowid(parent_rowids, config=config, _debug=_debug)
            >>> table.get_row_data(rowids)
            >>> print(rowids)
        """
        _debug = table.depc._debug if _debug is None else _debug
        if _debug:
            print('[deptbl.get_rowid] Lookup %s rowids from superkey with %d parents' % (
                table.tablename, len(parent_rowids)))
            print('[deptbl.get_rowid] config = %r' % (config,))
            print('[deptbl.get_rowid] ensure = %r' % (ensure,))

        # Find leaf rowids that need to be computed
        if table.ismulti:
            assert len(parent_rowids) == 1, (
                'can only take one set at a time %r' % (parent_rowids,))
            # Hack, implementation only will work for nnindexer
            assert all(ut.flatten(parent_rowids)), 'cant have None input to models'
            #parent_num = len(parent_rowids)
            # TODO: partition into multi cols and normal cols.
            # TODO: Allow for more than one multicol to be specified.
            parent_uuids = table.depc.get_root_uuid(parent_rowids[0][0])
            multiset_uuid = ut.hashable_to_uuid(parent_uuids)
            #parent_ids_ = [(multiset_uuid, parent_num)]
            parent_ids_ = [(multiset_uuid,)]
            preproc_args = parent_rowids[0:1]
            idxs2 = []
            idxs1 = [0]
        else:
            if ALLOW_NONE_YIELD:
                # Force entire row to be none if any are none
                anyNone_flags = [any(ut.flag_None_items(x))
                                 for x in parent_rowids]
                idxs2 = ut.where(anyNone_flags)
                idxs1 = ut.index_complement(idxs2, len_=len(parent_rowids))
                #error_parent_rowids =  ut.take(parent_rowids, idxs2)
                parent_ids_ = ut.take(parent_rowids, idxs1)
            else:
                parent_ids_ = parent_rowids
            # preproc args are usually the same as parent rowids.
            # Model tables are the exception.
            preproc_args = parent_ids_

        if recompute:
            # FIXME: make recompute work with table.ismulti
            # get existing rowids, delete them, recompute the request
            rowid_list = table._get_rowid(parent_ids_, config=config,
                                          eager=True, nInput=None)
            table.delete_rows(rowid_list)
        if ensure or recompute:
            rowid_list = table.add_rows_from_parent(
                parent_ids_, preproc_args, idxs1, idxs2, config=config)
        else:
            rowid_list = table._get_rowid(
                parent_ids_, config=config, eager=eager, nInput=nInput)
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
            >>> from dtool.depcache_table import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> #table = depc['keypoint']
            >>> table = depc['chip']
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

        Ignore:
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
        # if len(table.extern_write_funcs):
        if delete_extern:
            if len(table.extern_columns) > 0:
                extern_colnames = tuple(table.external_to_internal.values())
                uri_list = table.get_internal_columns(rowid_list,
                                                      extern_colnames,
                                                      unpack_scalars=False,
                                                      keepwrap=False)
                fpath_list = ut.flatten(uri_list)
                print('fpath_list = %r' % (fpath_list,))
                print('deleting internal files')
                ut.remove_file_list(fpath_list)

        # DELETE EXPLICITLY DEFINED CHILDREN
        # (TODO: handle implicit definitions)
        if True:
            parent_colnames = (table.tablename,)

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
                     read_extern=True, extra_tries=1):
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
            >>> colnames = ('size_1', 'size', 'chip' + EXTERN_SUFFIX)
            >>> _debug = True
            >>> read_extern = True
            >>> extra_tries = 1
            >>> kwargs = dict(read_extern=read_extern,
            >>>               extra_tries=extra_tries, _debug=_debug)
            >>> prop_list = table.get_row_data(tbl_rowids, colnames, **kwargs)
            >>> print(ut.repr2(prop_list, nl=1))
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
        total = 0
        intern_colnames = []
        extern_resolve_colxs = []
        nesting_xs = []  # how to resolve unnesting
        for c in requested_colnames:
            if c in table.external_to_internal:
                intern_colnames.append([table.external_to_internal[c]])
                read_func = table.extern_read_funcs[c]
                extern_resolve_colxs.append((total, read_func))
                nesting_xs.append(total)
                total += 1
            elif c in table.nested_to_flat:
                nest = table.nested_to_flat[c]
                nesting_xs.append(list(range(total, total + len(nest))))
                intern_colnames.append(nest)
                total += len(nest)
            else:
                nesting_xs.append(total)
                intern_colnames.append([c])
                total += 1
        flat_intern_colnames = tuple(ut.flatten(intern_colnames))
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
                        #colname = flat_intern_colnames[extern_colx]
                        #cls = table.extern_io_classes[colname]
                        #if hasattr(cls, 'on_load'):
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
