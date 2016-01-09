# -*- coding: utf-8 -*-
"""
implicit version of dependency cache from ibeis/templates/template_generator
#python -m ibeis.templates.template_generator --key feat --modfname={autogen_modname}

SeeAlso:
    https://pypi.python.org/pypi/luigi
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import uuid
import six
from six.moves import zip, range
from dtool import sql_control
(print, rrr, profile) = ut.inject2(__name__, '[depcache]')


CONFIG_TABLE = 'config'
CONFIG_ROWID = 'config_rowid'
CONFIG_HASHID = 'config_hashid'
EXTERN_SUFFIX = '_extern_uri'


TYPE_TO_SQLTYPE = {
    np.ndarray: 'NDARRAY',
    uuid.UUID: 'UUID',
    float: 'REAL',
    int: 'INTEGER',
    str: 'TEXT',
}


if six.PY2:
    TYPE_TO_SQLTYPE[six.text_type] = 'TEXT'


# List of globally registered functions
__PREPROC_REGISTER__ = []
__ALGO_REGISTER__ = []


class AlgoRequest(object):
    pass

class AlgoParams(object):
    pass


def check_register(args, kwargs):
    assert len(args) < 6, 'too many args'
    assert 'preproc_func' not in kwargs, 'cannot specify func in wrapper'


def register_preproc(*args, **kwargs):
    """
    Global regsiter proproc function that will define a table for all
    dependency caches containing the parents. See
    DependencyCache._register_prop for additional information.

    Args:
        tablename (str):
        parents (list): (default = None)
        colnames (list): (default = None)
        coltypes (list): (default = None)
        docstr (str): (default = None)
        fname (str):  file name(default = None)
        asobject (bool): (default = False)
        chunksize (int): (default = None)

    SeeAlso
        DependencyCache._register_prop - class specific version.
    """
    def register_preproc_wrapper(func):
        check_register(args, kwargs)
        kwargs['preproc_func'] = func
        __PREPROC_REGISTER__.append((args, kwargs))
        return func
    return register_preproc_wrapper


def register_algo(*args, **kwargs):
    """

    Args:
        algoname (str):
        algo_request_class (class):
        docstr (None): (default = None)
        fname (str):  file name (default = None)
        chunksize (None): (default = None)

    SeeAlso
        DependencyCache._register_algo
    """
    def _wrapper(func):
        kwargs['algo_func'] = func
        __ALGO_REGISTER__.append((args, kwargs))
        return func
    return _wrapper


class DependencyCache(object):
    """

    To use this class a user must:

        * on root modification, call depc.on_root_modified

        * use decorators to register relevant functions

        * write an algorithm that accepts an AlgoRequest
        object, containing root ids and a configuration object.


    """
    def __init__(depc, root_tablename=None, cache_dpath='./DEPCACHE',
                 controller=None, default_fname=None, root_asobject=None,
                 use_globals=True, get_root_hashid=None):
        # Root of all dependencies
        depc.root_tablename = root_tablename
        # Directory all cachefiles are stored in
        depc.cache_dpath = cache_dpath
        # Parent (ibs) controller
        depc.controller = controller
        # Internal dictionary of dependant tables
        depc.cachetable_dict = {}
        # Mapping of different files properties are stored in
        depc.fname_to_db = {}
        # Function to map a root rowid to an object
        depc._root_asobject = root_asobject
        depc._use_globals = use_globals
        depc.get_root_hashid = get_root_hashid
        if default_fname is None:
            default_fname = ':memory:'
        depc.default_fname = default_fname

    def register_preproc(depc, *args, **kwargs):
        """ Decorator for registration of cachables """
        def register_preproc_wrapper(func):
            check_register(args, kwargs)
            kwargs['preproc_func'] = func
            depc._register_prop(*args, **kwargs)
            return func
        return register_preproc_wrapper

    def _register_algo(depc, algoname, algo_func=None, docstr=None,
                       fname=None, chunksize=None):
        pass

    def _register_prop(depc, tablename, parents=None, colnames=None,
                       coltypes=None, preproc_func=None, docstr=None,
                       fname=None, asobject=False, chunksize=None):
        """
        Registers a table with this dependency cache.
        """
        # print('Registering tablename=%r' % (tablename,))
        # print('Registering preproc_func=%r' % (preproc_func,))
        if parents is None:
            parents = [depc.root]
        if colnames is None:
            colnames = ['data']
        if coltypes is None:
            coltypes = [np.ndarray] * len(colnames)
        if fname is None:
            fname = depc.default_fname
        if docstr is None:
            docstr = 'no docstr'

        depc.fname_to_db[fname] = None
        table = DependencyCacheTable(
            depc=depc,
            parent_tablenames=parents,
            tablename=tablename,
            docstr=docstr,
            data_colnames=colnames,
            data_coltypes=coltypes,
            preproc_func=preproc_func,
            asobject=asobject,
            fname=fname,
            chunksize=chunksize,
        )
        depc.cachetable_dict[tablename] = table
        return table

    @profile
    def initialize(depc):
        print('[depc] INITIALIZE DEPCACHE')

        if depc._use_globals:
            print(' * regsitering %d global preproc funcs' % (len(__PREPROC_REGISTER__),))
            for args_, kwargs_ in __PREPROC_REGISTER__:
                depc._register_prop(*args_, **kwargs_)

        ut.ensuredir(depc.cache_dpath)
        #print('depc.cache_dpath = %r' % (depc.cache_dpath,))
        config_addtable_kw = ut.odict(
            [
                ('tablename', CONFIG_TABLE,),
                ('coldef_list', [
                    (CONFIG_ROWID, 'INTEGER PRIMARY KEY'),
                    (CONFIG_HASHID, 'TEXT'),
                ],),
                ('docstr', 'table for algo configurations'),
                ('superkeys', [(CONFIG_HASHID,)]),
                ('dependson', [])
            ]
        )
        #print(ut.repr3(config_addtable_kw))

        #print('depc.fname_to_db.keys = %r' % (depc.fname_to_db,))
        for fname in depc.fname_to_db.keys():
            #print('fname = %r' % (fname,))
            if fname == ':memory:':
                fpath = fname
            else:
                fname_ = ut.ensure_ext(fname, '.sqlite')
                fpath = ut.unixjoin(depc.cache_dpath, fname_)
            #print('fpath = %r' % (fpath,))
            db = sql_control.SQLDatabaseController(fpath=fpath, simple=True)
            if not db.has_table(CONFIG_TABLE):
                db.add_table(**config_addtable_kw)
            depc.fname_to_db[fname] = db
        print('[depc] Finished initialization')

        for table in depc.cachetable_dict.values():
            table.initialize()

    def get_edges(depc):
        edges = [(parent, key) for key, table in  depc.cachetable_dict.items()
                 for parent in table.parents]
        return edges

    def print_schemas(depc):
        for fname, db in depc.fname_to_db.items():
            print('fname = %r' % (fname,))
            db.print_schema()

    def print_table_csv(depc, tablename):
        depc[tablename]

    def print_all_tables(depc):
        for tablename, table in depc.cachetable_dict.items():
            db = table.db
            db.print_table_csv(tablename)

    def make_digraph(depc):
        """
        CommandLine:
            python -m dtool.depends_cache --exec-make_digraph --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> import networkx as netx
            >>> depc = testdata_depc()
            >>> graph = depc.make_digraph()
            >>> ut.quit_if_noshow()
            >>> pos = netx.pydot_layout(graph, prog='dot')
            >>> import plottool as pt
            >>> pt.ensure_pylab_qt4()
            >>> pt.figure()
            >>> ax = pt.gca()
            >>> netx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=1100)
            >>> ut.show_if_requested()
        """
        import networkx as nx
        graph = nx.DiGraph()
        nodes = list(depc.cachetable_dict.keys())
        edges = depc.get_edges()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _custom_str(depc):
        typestr = depc.__class__.__name__
        infostr_ = 'nTables=%d' % len(depc.cachetable_dict)
        custom_str = '<%s(%s) %s at %s>' % (typestr, depc.root_tablename,
                                            infostr_, hex(id(depc)))
        return custom_str

    @property
    def root(depc):
        return depc.root_tablename

    def __repr__(depc):
        return depc._custom_str()

    def __str__(depc):
        return depc._custom_str()

    def __getitem__(depc, key):
        return depc.cachetable_dict[key]

    @ut.memoize
    def get_dependencies(depc, tablename):
        """
        gets level dependences from root to tablename

        CommandLine:
            python -m dtool.depends_cache --exec-get_dependencies --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'fgweight'
            >>> result = ut.repr3(depc.get_dependencies(tablename), nl=1)
            >>> print(result)
            [
                ['dummy_annot'],
                ['chip', 'probchip'],
                ['keypoint'],
                ['fgweight'],
            ]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'spam'
            >>> result = ut.repr3(depc.get_dependencies(tablename), nl=1)
            >>> print(result)
            [
                ['dummy_annot'],
                ['chip', 'probchip'],
                ['keypoint'],
                ['fgweight'],
                ['spam'],
            ]
        """
        root = depc.root_tablename
        children_, parents_ = list(zip(*depc.get_edges()))
        child_to_parents = ut.group_items(children_, parents_)
        to_root = {tablename: ut.paths_to_root(tablename, root, child_to_parents)}
        from_root = ut.reverse_path(to_root, root, child_to_parents)
        dependency_levels_ = ut.get_levels(from_root)
        dependency_levels = ut.longest_levels(dependency_levels_)
        #print('child_to_parents = %s' % (ut.repr3(child_to_parents),))
        #print('to_root = %r' % (to_root,))
        #print('from_root = %r' % (from_root,))
        return dependency_levels

    @ut.memoize
    def get_dependants(depc, tablename):
        """
        gets level dependences table to the leaves

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> result = ut.repr3(depc.get_dependants(tablename), nl=1)
            >>> print(result)
            [
                ['chip'],
                ['keypoint'],
                ['fgweight', 'descriptor'],
                ['spam'],
            ]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'spam'
            >>> result = ut.repr3(depc.get_dependants(tablename), nl=1)
            >>> print(result)
            [
                ['spam'],
            ]

        """
        children_, parents_ = list(zip(*depc.get_edges()))
        parent_to_children = ut.group_items(parents_, children_)
        to_leafs = {tablename: ut.path_to_leafs(tablename, parent_to_children)}
        dependency_levels_ = ut.get_levels(to_leafs)
        dependency_levels = ut.longest_levels(dependency_levels_)
        return dependency_levels

    def get_descendant_rowids(depc, tablename, root_rowids, config=None):
        r"""
        Args:
            tablename (?):
            root_rowids (?):
            config (None): (default = None)

        Returns:
            dict: rowid_dict

        CommandLine:
            python -m dtool.depends_cache --exec-get_descendant_rowids --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> root_rowids = [2, 3]
            >>> config, ensure, eager, nInput = None, True, True, None
            >>> result = ut.repr3(depc.get_descendant_rowids(tablename, root_rowids, config), nl=1)
            >>> print(result)
        """
        print('GET DESCENDANT ROWIDS %s ' % (tablename,))
        dependency_levels = depc.get_dependants(tablename)
        rowid_list = depc.get_rowids(tablename, root_rowids, config)
        rowid_dict = {tablename: rowid_list}
        for level_keys in dependency_levels[1:]:
            # TODO
            # FIXME; there will be multiple rowids for children.
            pass
        #     #print('* level_keys %s ' % (level_keys,))
        #     for key in level_keys:
        #         #print('  * key = %r' % (key,))
        #         table = depc[key]
        #         # due to different configs
        #         child_rowids = list(zip(*ut.dict_take(rowid_dict, table.children)))
        #         # print('parent_rowids = %r' % (parent_rowids,))
        #         # child_rowids = table.get_rowid_from_superkey(
        #         #     parent_rowids, config=config, eager=eager, nInput=nInput,
        #         #     ensure=ensure)
        #         # print('child_rowids = %r' % (child_rowids,))
        #         rowid_dict[key] = child_rowids
        return rowid_dict

    def get_ancestor_rowids(depc, tablename, root_rowids, config=None,
                            ensure=True, eager=True, nInput=None):
        r"""
        Args:
            tablename (str):
            root_rowids (list):
            config (None): (default = None)
            ensure (bool):  eager evaluation if True(default = True)
            eager (bool): (default = True)
            nInput (None): (default = None)

        CommandLine:
            python -m dtool.depends_cache --exec-get_ancestor_rowids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> depc = testdata_depc()
            >>> tablename = 'spam'
            >>> root_rowids = [1, 2, 3]
            >>> config, ensure, eager, nInput = None, True, True, None
            >>> result = ut.repr3(depc.get_ancestor_rowids(tablename, root_rowids, config, ensure, eager, nInput), nl=1)
            >>> print(result)
            {
                'chip': [1, 2, 3],
                'dummy_annot': [1, 2, 3],
                'fgweight': [1, 2, 3],
                'keypoint': [1, 2, 3],
                'probchip': [1, 2, 3],
                'spam': [1, 2, 3],
            }
        """
        # print('GET ANCESTOR ROWIDS %s ' % (tablename,))
        dependency_levels = depc.get_dependencies(tablename)
        # print('root_rowids = %r' % (root_rowids,))
        #print('dependency_levels = %s' % (ut.repr3(dependency_levels, nl=2),))
        rowid_dict = {depc.root: root_rowids}
        for level_keys in dependency_levels[1:]:
            #print('* level_keys %s ' % (level_keys,))
            for key in level_keys:
                #print('  * key = %r' % (key,))
                table = depc[key]
                parent_rowids = list(zip(*ut.dict_take(rowid_dict, table.parents)))
                # print('parent_rowids = %r' % (parent_rowids,))
                child_rowids = table.get_rowid_from_superkey(
                    parent_rowids, config=config, eager=eager, nInput=nInput,
                    ensure=ensure)
                # print('child_rowids = %r' % (child_rowids,))
                rowid_dict[key] = child_rowids
        return rowid_dict

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
                   eager=True, nInput=None):
        """
        Returns the rowids of `tablename` that correspond to `root_rowids` using `config`.
        """
        rowid_dict = depc.get_ancestor_rowids(tablename, root_rowids,
                                              config=config, ensure=ensure,
                                              eager=eager, nInput=nInput)
        rowid_list = rowid_dict[tablename]
        return rowid_list

    def delete_property(depc, tablename, root_rowids, config=None):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids` using `config`.
        """
        rowid_list = depc.get_rowids(root_rowids, config=config, ensure=False)
        table = depc[tablename]
        num_deleted = table.delete_rows(rowid_list)
        return num_deleted

    @ut.accepts_scalar_input2(argx_list=[1])
    def get_property(depc, tablename, root_rowids, colnames=None, config=None,
                     ensure=True):
        """
        Gets the data in `colnames` of `tablename` that correspond to
        `root_rowids` using `config`.  if colnames is None, all columns are
        returned.
        """
        # Allow scalar input
        # is_scalar = not ut.isiterable(root_rowids)
        # if is_scalar:
        #     root_rowids = [root_rowids]
        # Vectorized get of properties
        tbl_rowids = depc.get_rowids(tablename, root_rowids, config,
                                     ensure=ensure)
        table = depc[tablename]
        prop_list = table.get_col(tbl_rowids, colnames)
        # Scalar output if scalar input
        # if is_scalar:
        #     prop_list = prop_list[0]
        return prop_list

    @ut.accepts_scalar_input2(argx_list=[1])
    def get_obj(depc, tablename, root_rowids, config=None, ensure=True):
        """ Convinience function. Gets data in `tablename` as a list of objects. """
        try:
            if tablename == depc.root:
                obj_list = [depc._root_asobject(rowid) for rowid in root_rowids]
            else:
                def make_property_getter(rowid, colname):
                    def wrapper():
                        return depc.get_property(
                            tablename, rowid, colnames=colname, config=config,
                            ensure=ensure)
                    return wrapper
                colnames = depc[tablename].data_colnames
                obj_list = [
                    ut.LazyDict({colname: make_property_getter(rowid, colname)
                                 for colname in colnames})
                    for rowid in root_rowids
                ]
            return obj_list
            # data_list = depc.get_property(tablename, root_rowids, config)
            # # TODO: lazy dict
            # return [dict(zip(colnames, data)) for data in data_list]
        except Exception as ex:
            ut.printex(ex, 'error in getobj', keys=['tablename', 'root_rowids', 'colnames'])
            raise

    def request_algorithm(depc, algoname):
        pass


class DependencyCacheTable(object):

    def __init__(table, depc=None, parent_tablenames=None, tablename=None,
                 data_colnames=None, data_coltypes=None, preproc_func=None,
                 docstr='no docstr', fname=None, asobject=False, chunksize=None):

        table.fpath_to_db = {}

        table.parent_tablenames = parent_tablenames
        table.tablename = tablename
        table.data_colnames = tuple(data_colnames)
        table.data_coltypes = data_coltypes
        table.preproc_func = preproc_func

        table._internal_data_colnames = []
        table._internal_data_coltypes = []
        table._nested_idxs = []
        table.sqldb_fpath = None
        table.extern_read_funcs = {}
        table._nested_idxs2 = []

        table.docstr = docstr
        table.fname = fname
        table.depc = depc
        table.db = None
        table.chunksize = None
        table._asobject = asobject
        table._update_internals()
        table._assert_self()

    def _assert_self(table):
        if table.preproc_func is not None:
            argspec = ut.get_func_argspec(table.preproc_func)
            args = argspec.args
            if argspec.varargs and argspec.keywords:
                assert len(args) == 1, 'varargs and kwargs must have one arg for depcache'
            else:
                if len(args) < 3:
                    print('args = %r' % (args,))
                    assert False, 'preproc func must have a depcache arg, at least one parent rowid arg, and a config arg'
                rowid_args = args[1:-1]
                if len(rowid_args) != len(table.parents):
                    print('table.preproc_func = %r' % (table.preproc_func,))
                    print('args = %r' % (args,))
                    print('rowid_args = %r' % (rowid_args,))
                    msg = (
                        'preproc function for table=%s must have as many rowids %d args as parents %d' % (
                            table.tablename, len(rowid_args), len(table.parents))
                    )
                    assert False, msg

    def _update_internals(table):
        extern_read_funcs = {}
        internal_data_colnames = []
        internal_data_coltypes = []
        _nested_idxs2 = []

        nested_to_flat = {}

        external_to_internal = {}

        for colx, (colname, coltype) in enumerate(zip(table.data_colnames, table.data_coltypes)):
            if isinstance(coltype, tuple) or ut.is_func_or_method(coltype):
                if ut.is_func_or_method(coltype) or ut.is_func_or_method(coltype[0]) or coltype[0] == 'extern':
                    if isinstance(coltype, tuple):
                        if coltype[0] == 'extern':
                            read_func = coltype[1]
                        else:
                            read_func = coltype[0]
                    else:
                        read_func = coltype
                    extern_read_funcs[colname] = read_func
                    _nested_idxs2.append(len(internal_data_colnames))
                    intern_colname = colname + EXTERN_SUFFIX
                    internal_data_colnames.append(intern_colname)
                    internal_data_coltypes.append(TYPE_TO_SQLTYPE[str])
                    external_to_internal[colname] = intern_colname
                else:
                    nest = []
                    table._nested_idxs.append(colx)
                    nested_to_flat[colname] = []
                    for count, dimtype in enumerate(coltype):
                        nest.append(len(internal_data_colnames))
                        flat_colname = '%s_%d' % (colname, count)
                        nested_to_flat[colname].append(flat_colname)
                        internal_data_colnames.append(flat_colname)
                        internal_data_coltypes.append(TYPE_TO_SQLTYPE[dimtype])
                    _nested_idxs2.append(nest)
            else:
                _nested_idxs2.append(len(internal_data_colnames))
                internal_data_colnames.append(colname)
                internal_data_coltypes.append(TYPE_TO_SQLTYPE[coltype])

        assert len(set(internal_data_colnames)) == len(internal_data_colnames)
        assert len(internal_data_coltypes) == len(internal_data_colnames)
        table.extern_read_funcs = extern_read_funcs
        table.external_to_internal = external_to_internal
        table.nested_to_flat = nested_to_flat
        table._nested_idxs2 = _nested_idxs2
        table._internal_data_colnames = tuple(internal_data_colnames)
        table._internal_data_coltypes = tuple(internal_data_coltypes)
        table._assert_self()

    def get_addtable_kw(table):
        primary_coldef = [(table.rowid_colname, 'INTEGER PRIMARY KEY')]
        parent_coldef = [(key, 'INTEGER NOT NULL') for key in table.parent_rowid_colnames]
        config_coldef = [(CONFIG_ROWID, 'INTEGER DEFAULT 0')]
        internal_data_coldef = list(zip(table._internal_data_colnames,
                                        table._internal_data_coltypes))

        coldef_list = primary_coldef + parent_coldef + config_coldef + internal_data_coldef
        add_table_kw = ut.odict([
            ('tablename', table.tablename,),
            ('coldef_list', coldef_list,),
            ('docstr', table.docstr,),
            ('superkeys', [table.superkey_colnames],),
            ('dependson', table.parents),
        ])
        return add_table_kw

    def initialize(table):
        table.db = table.depc.fname_to_db[table.fname]
        if not table.db.has_table(table.tablename):
            table.db.add_table(**table.get_addtable_kw())

    def print_schemadef(table):
        print('\n'.join(table.db.get_table_autogen_str(table.tablename)))

    def _get_all_rowids(table):
        pass

    @property
    def parents(table):
        return table.parent_tablenames

    @property
    def rowid_colname(table):
        return table.tablename + '_rowid'

    @property
    def parent_rowid_colnames(table):
        #return tuple([table.depc[parent].rowid_colname for parent in table.parents])
        return tuple([parent + '_rowid' for parent in table.parents])

    @property
    def superkey_colnames(table):
        return table.parent_rowid_colnames + (CONFIG_ROWID,)

    @property
    def _table_colnames(table):
        return table.superkey_colnames + table._internal_data_colnames

    def _custom_str(table):
        typestr = table.__class__.__name__
        custom_str = '<%s(%s) at %s>' % (typestr, table.tablename, hex(id(table)))
        return custom_str

    def __repr__(table):
        return table._custom_str()

    def __str__(table):
        return table._custom_str()

    # ---------------------------
    # --- CONFIGURATION TABLE ---
    # ---------------------------

    def get_config_rowid(table, config=None):
        #config_hashid = config.get('feat_cfgstr')
        #assert config_hashid is not None
        # TODO store config_rowid in qparams
        #else:
        #    config_hashid = db.cfg.feat_cfg.get_cfgstr()
        if config is not None:
            if isinstance(config, AlgoRequest):
                pass
            elif isinstance(config, AlgoParams):
                pass
            else:
                try:
                    #config_hashid = 'none'
                    config_hashid = config.get(table.tablename + '_hashid')
                except KeyError:
                    try:
                        subconfig = config.get(table.tablename + '_config')
                        config_hashid = ut.hashstr27(ut.to_json(subconfig))
                    except KeyError:
                        print('Warning: Config must either contain a string <tablename>_hashid or a dict <tablename>_config')
                        raise
        else:
            config_hashid = 'none'
        config_rowid = table.add_config(config_hashid)
        return config_rowid

    def get_config_rowid_from_hashid(table, config_hashid_list):
        config_rowid_list = table.db.get(
            CONFIG_TABLE, (CONFIG_ROWID,), config_hashid_list,
            id_colname=CONFIG_HASHID)
        return config_rowid_list

    def add_config(table, config_hashid):
        get_rowid_from_superkey = table.get_config_rowid_from_hashid
        config_rowid_list = table.db.add_cleanly(
            CONFIG_TABLE, (CONFIG_HASHID,), [(config_hashid,)],
            get_rowid_from_superkey)
        config_rowid = config_rowid_list[0]
        return config_rowid

    # ----------------------
    # --- GETTERS NATIVE ---
    # ----------------------

    def add_rows_from_parent(table, parent_rowids, config=None, verbose=True,
                             return_num_dirty=False):
        """
        Lazy addition
        """
        try:
            # Get requested configuration id
            config_rowid = table.get_config_rowid(config)
            # Find leaf rowids that need to be computed
            initial_rowid_list = table._get_rowid_from_superkey(parent_rowids,
                                                                config=config)
            # Get corresponding "dirty" parent rowids
            isdirty_list = ut.flag_None_items(initial_rowid_list)
            dirty_parent_rowids = ut.compress(parent_rowids, isdirty_list)
            num_dirty = len(dirty_parent_rowids)
            num_total = len(parent_rowids)
            if num_dirty > 0:
                if verbose:
                    fmtstr = 'adding %d / %d new props to %r for config_rowid=%r'
                    print(fmtstr % (num_dirty, num_total, table.tablename,
                                    config_rowid))
                args = zip(*dirty_parent_rowids)
                if table._asobject:
                    # Convinience
                    args = [table.depc.get_obj(parent, rowids)
                            for parent, rowids in zip(table.parents, args)]
                # CALL EXTERNAL PREPROCESSING / GENERATION FUNCTION
                proptup_gen = table.preproc_func(table.depc, *args, config=config)

                #proptup_gen = list(proptup_gen)

                if len(table._nested_idxs) > 0:
                    # TODO: rewrite
                    nested_nCols = len(table.data_colnames)
                    idxs1 = table._nested_idxs
                    mask1 = ut.index_to_boolmask(idxs1, nested_nCols)
                    mask2 = ut.not_list(mask1)
                    idxs2 = ut.where(mask2)
                    def unnest_data(data):
                        unnested_cols = list(zip(ut.take(data, idxs2)))
                        nested_cols = ut.take(data, idxs1)
                        grouped_items = [nested_cols, unnested_cols]
                        groupxs = [idxs1, idxs2]
                        unflat = ut.ungroup(grouped_items, groupxs, nested_nCols - 1)
                        return tuple(ut.flatten(unflat))
                    # Hack when a sql schema has tuples defined in it
                    proptup_gen = (unnest_data(data) for data in proptup_gen)

                #proptup_gen = list(proptup_gen)

                dirty_params_iter = (
                    parent_rowids + (config_rowid,) + data_cols
                    for parent_rowids, data_cols in zip(dirty_parent_rowids, proptup_gen))
                #dirty_params_iter = list(dirty_params_iter)
                #print('dirty_params_iter = %s' % (ut.repr2(dirty_params_iter, nl=1),))
                CHUNKED_ADD = table.chunksize is not None
                if CHUNKED_ADD:
                    for dirty_params_chunk in ut.ichunks(dirty_params_iter,
                                                         chunksize=table.chunksize):
                        table.db._add(table.tablename, table._table_colnames,
                                      dirty_params_chunk,
                                      nInput=len(dirty_params_chunk))
                else:
                    nInput = num_dirty
                    table.db._add(table.tablename, table._table_colnames,
                                  dirty_params_iter, nInput=nInput)
                # Now that the dirty params are added get the correct order of rowids
                rowid_list = table._get_rowid_from_superkey(parent_rowids,
                                                            config=config)
            else:
                rowid_list = initial_rowid_list
            if return_num_dirty:
                return rowid_list, num_dirty
            else:
                return rowid_list
        except Exception as ex:
            ut.printex(ex, 'error in add_rowids', keys=[
                'table', 'parent_rowids', 'config', 'args',
                'dirty_parent_rowids', 'table.preproc_func'])
            raise

    def get_rowid_from_superkey(table, parent_rowids, config=None, ensure=True,
                                eager=True, nInput=None, recompute=False):
        r"""
        get feat rowids of chip under the current state configuration
        if ensure is True, this function is equivalent to add_rows_from_parent

        Args:
            parent_rowids (list): list of tuples with the parent rowids as the
                value of each tuple
            config (None): (default = None)
            ensure (bool):  eager evaluation if True (default = True)
            eager (bool): (default = True)
            nInput (int): (default = None)
            recompute (bool): (default = False)

        Returns:
            list: rowid_list
        """
        # print('Lookup %s rowids from superkey with %d parents' % (
        #     table.tablename, len(parent_rowids)))
        #rowid_list = parent_rowids
        #return rowid_list

        if recompute:
            # get existing rowids, delete them, recompute the request
            rowid_list = table._get_rowid_from_superkey(
                parent_rowids, config=config, eager=eager, nInput=nInput)
            table.delete_rows(rowid_list)
            rowid_list = table.add_rows_from_parent(parent_rowids, config=config)
        elif ensure:
            rowid_list = table.add_rows_from_parent(parent_rowids, config=config)
        else:
            rowid_list = table._get_rowid_from_superkey(
                parent_rowids, config=config, eager=eager, nInput=nInput)
        return rowid_list

    def _get_rowid_from_superkey(table, parent_rowids, config=None, eager=True,
                                 nInput=None):
        """
        equivalent to get_rowid_from_superkey except ensure is constrained to be False.
        """
        colnames = (table.rowid_colname,)
        config_rowid = table.get_config_rowid(config=config)
        and_where_colnames = table.superkey_colnames
        params_iter = (rowids + (config_rowid,) for rowids in parent_rowids)
        params_iter = list(params_iter)
        #print('**params_iter = %r' % (params_iter,))
        rowid_list = table.db.get_where2(table.tablename, colnames, params_iter,
                                         and_where_colnames, eager=eager,
                                         nInput=nInput)
        return rowid_list

    def delete_rows(table, rowid_list):
        #from dtool.algo.preproc import preproc_feat
        if table.on_delete is not None:
            table.on_delete()
        if ut.VERBOSE:
            print('deleting %d rows' % len(rowid_list))
        # Finalize: Delete table
        table.db.delete_rowids(table.tablename, rowid_list)
        num_deleted = len(ut.filter_Nones(rowid_list))
        return num_deleted

    def get_col(table, tbl_rowids, colnames=None):
        """
        colnames = ('mask', 'size')

        FIXME; unpacking is confusing with sql controller
        """
        # print('Get prop of %r, colnames=%r' % (table, colnames))
        try:
            request_unpack = False
            if colnames is None:
                colnames = table.data_colnames
                #table._internal_data_colnames
            else:
                if isinstance(colnames, six.text_type):
                    request_unpack = True
                    colnames = (colnames,)
            # print('* colnames = %r' % (colnames,))

            eager = True
            nInput = None

            total = 0
            intern_colnames = []
            extern_resolve_colxs = []
            nesting_xs = []

            for c in colnames:
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

            # do sql read
            # FIXME: understand unpack_scalars and keepwrap
            raw_prop_list = table.get_internal_columns(
                tbl_rowids, flat_intern_colnames, eager, nInput,
                unpack_scalars=True, keepwrap=True)
            # unpack_scalars=not
            # request_unpack)
            # print('depth(raw_prop_list) = %r' % (ut.depth_profile(raw_prop_list),))

            prop_listT = list(zip(*raw_prop_list))
            for extern_colx, read_func in extern_resolve_colxs:
                data_list = []
                for uri in prop_listT[extern_colx]:
                    try:
                        # FIXME: only do this for a localpath
                        uri1 = ut.unixjoin(table.depc.cache_dpath, uri)
                        data = read_func(uri1)
                    except Exception as ex:
                        ut.printex(ex, 'failed to load external data', iswarning=False)
                        raise
                        # FIXME
                        #data = None
                    data_list.append(data)
                prop_listT[extern_colx] = data_list

            nested_proplistT = ut.list_unflat_take(prop_listT, nesting_xs)

            for tx in ut.where([isinstance(xs, list) for xs in nesting_xs]):
                nested_proplistT[tx] = list(zip(*nested_proplistT[tx]))

            prop_list = list(zip(*nested_proplistT))

            if request_unpack:
                prop_list = [None if p is None else p[0] for p in prop_list]
        except Exception as ex:
            ut.printex(ex, 'failed in get col', keys=[
                'table.tablename',
                'request_unpack',
                'tbl_rowids',
                'colnames',
                'raw_prop_list',
                (ut.depth_profile, 'raw_prop_list'),
                'prop_listT',
                (ut.depth_profile, 'prop_listT'),
                'nesting_xs',
                'nested_proplistT',
                'prop_list'])
            raise
        return prop_list

    def get_internal_columns(table, tbl_rowids, colnames=None, eager=True,
                             nInput=None, unpack_scalars=True, keepwrap=False):
        prop_list = table.db.get(
            table.tablename, colnames, tbl_rowids,
            id_colname=table.rowid_colname, eager=eager, nInput=nInput,
            unpack_scalars=unpack_scalars, keepwrap=keepwrap)
        return prop_list


if False:
    # Example of global preproc function
    dummy_root_tablename = 'dummy_annot'
    @register_preproc(tablename='dummy', parents=[dummy_root_tablename], colnames=['data'], coltypes=[str])
    def dummy_global_preproc_func(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Requesting global dummy ')
        for rowid in parent_rowids:
            yield 'dummy'


def testdata_depc(fname=None):
    import vtool as vt
    gpath_list = ut.lmap(ut.grab_test_imgpath, ut.get_valid_test_imgkeys(), verbose=False)

    dummy_root = 'dummy_annot'

    def root_asobject(aid):
        """ Convinience for writing preproc funcs """
        gpath = gpath_list[aid]
        root_obj = ut.LazyDict({
            'aid': aid,
            'gpath': gpath,
            'image': lambda: vt.imread(gpath)
        })
        return root_obj

    depc = DependencyCache(root_tablename=dummy_root, default_fname=fname,
                           root_asobject=root_asobject, use_globals=False)
    _register_preproc = depc.register_preproc

    @_register_preproc(
        tablename='chipmask', parents=[dummy_root], colnames=['size', 'mask'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)])
    def dummy_manual_chipmask(depc, parent_rowids, config=None):
        import vtool as vt
        from plottool import interact_impaint
        mask_dpath = ut.unixjoin(depc.cache_dpath, 'ManualChipMask')
        ut.ensuredir(mask_dpath)
        if config is None:
            config = {}
        print('Requesting user defined chip mask')
        for rowid in parent_rowids:
            img = vt.imread(gpath_list[rowid])
            mask = interact_impaint.impaint_mask2(img)
            mask_fpath = ut.unixjoin(mask_dpath, 'mask%d.png' % (rowid,))
            vt.imwrite(mask_fpath, mask)
            w, h = vt.get_size(mask)
            yield (w, h), mask_fpath

    @_register_preproc(
        tablename='chip', parents=[dummy_root], colnames=['size', 'chip'],
        coltypes=[(int, int), vt.imread], asobject=True)
    def dummy_preproc_chip(depc, annot_list, config=None):
        """
        TODO: Infer properties from docstr

        Args:
            annot_list (list): list of annot objects
            config (dict): config dictionary

        Returns:
            tuple : ((int, int), ('extern', vt.imread))
        """
        if config is None:
            config = {}
        # Demonstates using asobject to get input to function as a dictionary
        # of properties
        for annot in annot_list:
            print('Computing chips of annot=%r' % (annot,))
            chip_fpath = annot['gpath']
            w, h = vt.image.open_image_size(chip_fpath)
            size = (w, h)
            print('* chip_fpath = %r' % (chip_fpath,))
            print('* size = %r' % (size,))
            yield size, chip_fpath

    @_register_preproc(
        'probchip', [dummy_root], ['size', 'probchip'],
        coltypes=[(int, int), ('extern', vt.imread)])
    def dummy_preproc_probchip(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing probchip')
        for rowid in parent_rowids:
            yield (rowid, rowid), 'probchip.jpg'

    @_register_preproc(
        'keypoint', ['chip'], ['kpts', 'num'], [np.ndarray, int],
        docstr='Used to store individual chip features (ellipses)',)
    def dummy_preproc_kpts(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing kpts')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 6)) + rowid, 7 + rowid

    @_register_preproc('descriptor', ['keypoint'], ['vecs'], [np.ndarray],)
    def dummy_preproc_vecs(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing vecs')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 8), dtype=np.uint8) + rowid,

    @_register_preproc('fgweight', ['keypoint', 'probchip'], ['fgweight'], [np.ndarray],)
    def dummy_preproc_fgweight(depc, kpts_rowid, probchip_rowid, config=None):
        if config is None:
            config = {}
        print('Computing fgweight')
        for rowid1, rowid2 in zip(kpts_rowid, probchip_rowid):
            yield np.ones(7 + rowid1),

    @_register_preproc('notch', [dummy_root], ['notchdata'],)
    def dummy_preproc_notch(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing notch')
        for rowid in parent_rowids:
            yield np.empty(5 + rowid),

    @_register_preproc('spam', ['fgweight', 'chip', 'keypoint'],
                       ['spam', 'eggs', 'size', 'uuid', 'vector', 'textdata'],
                       [str, int, (int, int), uuid.UUID, np.ndarray, ('extern', ut.readfrom)],
                       docstr='I dont like spam',)
    def dummy_preproc_spam(depc, *args, **kwargs):
        config = kwargs.get('config', None)
        if config is None:
            config = {}
        print('Computing notch')
        ut.writeto('tmp.txt', ut.lorium_ipsum())
        for x in zip(*args):
            size = (42, 21)
            uuid = ut.get_zero_uuid()
            vector = np.ones(3)
            yield ('spam', 3665, size, uuid, vector, 'tmp.txt')

    # table = depc['spam']
    # print(ut.repr2(table.get_addtable_kw(), nl=2))

    depc.initialize()

    # table.print_schemadef()
    # print(table.db.get_schema_current_autogeneration_str())
    return depc


def dummy_example_depcacahe():
    r"""
    CommandLine:
        python -m dtool.depends_cache --exec-dummy_example_depcacahe --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from dtool.depends_cache import *  # NOQA
        >>> depc = dummy_example_depcacahe()
    """

    fname = None
    # fname = 'dummy_default_depcache'
    fname = ':memory:'

    depc = testdata_depc(fname)

    tablename = 'fgweight'
    # print('[test] fgweight_path =\n%s' % (ut.repr3(depc.get_dependencies(tablename), nl=1),))
    # print('[test] keypoint =\n%s' % (ut.repr3(depc.get_dependencies('keypoint'), nl=1),))
    # print('[test] descriptor =\n%s' % (ut.repr3(depc.get_dependencies('descriptor'), nl=1),))
    # print('[test] spam =\n%s' % (ut.repr3(depc.get_dependencies('spam'), nl=1),))

    desc_rowids = depc.get_rowids('descriptor', [1, 2, 3])  # NOQA

    root_rowids = [1]
    if 1:
        tablename = 'chip'
        col1 = 'chip'
    else:
        # Manual interaction given
        tablename = 'chipmask'
        col1 = 'mask'
    col2 = 'size'
    table = depc[tablename]  # NOQA

    # You can get a reference to data rows using the "root" (dummy_annot) rowids
    # By default, if the data has not been computed, then it will be computed
    # for you. But if you specify ensure=False, None will be returned if the data
    # has not been computed yet.

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    #assert tbl_rowids[0] is None

    # The default is for the data to be computed though. Manaual interactions will
    # launch as necessary.

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=True)  # NOQA
    assert tbl_rowids[0] is not None

    # Now the data is cached and will not need to be computed again

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    assert tbl_rowids[0] is not None

    # The rowids can be used to lookup data values directly. By default all data
    # in a row is returned.

    datas = depc[tablename].get_col(tbl_rowids)  # NOQA

    # But you can also ask for a specific column

    col1_data = depc[tablename].get_col(tbl_rowids, col1)  # NOQA

    # In the case of external columns, you can lookup the hidden id as follows

    col1_paths = depc[tablename].get_col(tbl_rowids, (col1 + EXTERN_SUFFIX,))  # NOQA

    # But you can also just the root rowids directly. This is the simplest way to
    # access data and really "all you need to know"
    datas = depc.get_property(tablename, root_rowids, (col1, col2))  # NOQA

    # One input = one output
    chip = depc.get_property('chip', 1, 'chip')  # NOQA
    print('[test] chip.sum() = %r' % (chip.sum(),))

    col_tup_list = depc.get_property('chip', [1], ('size',))
    print('[test] col_tup_list = %r' % (col_tup_list,))

    col_list = depc.get_property('chip', [1], 'size')
    print('[test] col_list = %r' % (col_list,))

    col = depc.get_property('chip', 1, 'size')
    print('[test] col = %r' % (col,))

    cols = depc.get_property('chip', 1, 'size')
    print('[test] cols = %r' % (cols,))

    chip_dict = depc.get_obj('chip', 1)

    print('chip_dict = %r' % (chip_dict,))
    print('chip_dict["size"] = %r' % (chip_dict['size'],))
    print('chip_dict["chip"] = %r' % (chip_dict['chip'],))
    print('chip_dict = %r' % (chip_dict,))

    return depc


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.depends_cache
        python -m dtool.depends_cache --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
