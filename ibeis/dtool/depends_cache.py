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
import six
from six.moves import zip, range
from dtool import sql_control
from dtool import __SQLITE__ as lite
(print, rrr, profile) = ut.inject2(__name__, '[depcache]')


CONFIG_TABLE = 'config'
CONFIG_ROWID = 'config_rowid'
CONFIG_HASHID = 'config_hashid'
EXTERN_SUFFIX = '_extern_uri'


# global function registry
__PREPROC_REGISTER__ = ut.ddict(list)
__ALGO_REGISTER__ = ut.ddict(list)


def make_depcache_decors(root_tablename):
    """
    Makes global decorators to register functions for a tablename.

    A preproc function is meant to belong only to a single parent

    An algo function belongs to the root node, and may depend on a set of root
    nodes rather than just a single one.
    """

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
            __PREPROC_REGISTER__[root_tablename].append((args, kwargs))
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
            __ALGO_REGISTER__[root_tablename].append((args, kwargs))
            return func
        return _wrapper

    return register_preproc, register_algo


class AlgoRequest(object):
    """ Base class for algo request objects """
    pass


class AlgoParams(object):
    """ Base class for heirarchiacl params """
    pass


def check_register(args, kwargs):
    assert len(args) < 6, 'too many args'
    assert 'preproc_func' not in kwargs, 'cannot specify func in wrapper'


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

    def register_algo(depc, *args, **kwargs):
        """ Decorator for registration of cachables """
        def reg_algo_wrapper(func):
            check_register(args, kwargs)
            kwargs['algo_func'] = func
            depc._register_algo(*args, **kwargs)
            return func
        return reg_algo_wrapper

    def _register_algo(depc, algoname, algo_func=None, docstr=None,
                       fname=None, chunksize=None):
        depc._register_prop(algoname, preproc_func=algo_func)
        pass

    def _register_prop(depc, tablename, parents=None, colnames=None,
                       coltypes=None, preproc_func=None, docstr=None,
                       fname=None, asobject=False, chunksize=None):
        """
        Registers a table with this dependency cache.
        """
        # print('Registering tablename=%r' % (tablename,))
        # print('Registering preproc_func=%r' % (preproc_func,))
        if isinstance(tablename, six.string_types):
            tablename = six.text_type(tablename)
        if parents is None:
            parents = [depc.root]
        if colnames is None:
            colnames = ['data']
        else:
            colnames = ut.lmap(six.text_type, colnames)
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
        print('[depc] INITIALIZE %s DEPCACHE' % (depc.root.upper(),))

        if depc._use_globals:
            print(' * regsitering %d global preproc funcs' % (len(__PREPROC_REGISTER__[depc.root]),))
            for args_, kwargs_ in __PREPROC_REGISTER__[depc.root]:
                depc._register_prop(*args_, **kwargs_)

            print(' * regsitering %d global algos ' % (len(__ALGO_REGISTER__[depc.root]),))
            for args_, kwargs_ in __ALGO_REGISTER__[depc.root]:
                pass
                # depc._register_algo(*args_, **kwargs_)

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
            if ut.get_argflag('--clear-all-depcache'):
                ut.delete(fpath)
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
            >>> from dtool.examples.dummy_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> graph = depc.make_digraph()
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.ensure_pylab_qt4()
            >>> pt.show_netx(graph)
            >>> ut.show_if_requested()
        """
        import networkx as nx
        graph = nx.DiGraph()
        nodes = list(depc.cachetable_dict.keys())
        edges = depc.get_edges()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def show_digraph(depc, **kwargs):
        import plottool as pt
        graph = depc.make_digraph()
        pt.show_netx(graph, **kwargs)

    def _custom_str(depc):
        typestr = depc.__class__.__name__
        infostr_ = 'nTables=%d' % len(depc.cachetable_dict)
        custom_str = '<%s(%s) %s at %s>' % (typestr, depc.root_tablename,
                                            infostr_, hex(id(depc)))
        return custom_str

    def __repr__(depc):
        return depc._custom_str()

    def __str__(depc):
        return depc._custom_str()

    @property
    def root(depc):
        return depc.root_tablename

    def __getitem__(depc, key):
        return depc.cachetable_dict[key]

    # @ut.memoize
    def get_dependencies(depc, tablename):
        """
        gets level dependences from root to tablename

        CommandLine:
            python -m dtool.depends_cache --exec-get_dependencies
            python -m dtool.depends_cache --exec-get_dependencies:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> from dtool.examples.dummy_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> result = ut.repr3(depc.get_dependencies(tablename), nl=1)
            >>> print(result)
            [
                ['dummy_annot'],
                ['chip'],
            ]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> from dtool.examples.dummy_depcache import testdata_depc
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
            >>> from dtool.examples.dummy_depcache import testdata_depc
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
        try:
            assert tablename in depc.cachetable_dict, 'tablename=%r does not exist' % (tablename,)
            root = depc.root_tablename
            children_, parents_ = list(zip(*depc.get_edges()))
            child_to_parents = ut.group_items(children_, parents_)
            if ut.VERBOSE:
                print('root = %r' % (root,))
                print('tablename = %r' % (tablename,))
                print('child_to_parents = %s' % (ut.repr3(child_to_parents),))
            to_root = {tablename: ut.paths_to_root(tablename, root, child_to_parents)}
            if ut.VERBOSE:
                print('to_root = %r' % (to_root,))
            from_root = ut.reverse_path(to_root, root, child_to_parents)
            dependency_levels_ = ut.get_levels(from_root)
            dependency_levels = ut.longest_levels(dependency_levels_)
        except Exception as ex:
            ut.printex(ex, 'error getting dependencies',
                       keys=[
                           'root',
                           'children_to_parents',
                           'to_root',
                           'from_root',
                           'dependency_levels_',
                           'dependency_levels',
                       ])
            raise

        #print('from_root = %r' % (from_root,))
        return dependency_levels

    # @ut.memoize
    def get_dependants(depc, tablename):
        """
        gets level dependences table to the leaves

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> from dtool.examples.dummy_depcache import testdata_depc
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
            >>> from dtool.examples.dummy_depcache import testdata_depc
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
            >>> from dtool.examples.dummy_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> root_rowids = [2, 3]
            >>> config, ensure, eager, nInput = None, True, True, None
            >>> result = ut.repr3(depc.get_descendant_rowids(tablename, root_rowids, config), nl=1)
            >>> print(result)
        """
        print('[depc.descendant] GET DESCENDANT ROWIDS %s ' % (tablename,))
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
        Connects `root_rowids` to rowids in `tablename`, and computes all
        values needed along the way.

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
            >>> from dtool.examples.dummy_depcache import testdata_depc
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
        # if True:
        with ut.Indenter('[ANCE %s]' % (tablename,)):
            print(' * GET ANCESTOR ROWIDS %s ' % (tablename,))
            print(' * config = %r' % (config,))
            dependency_levels = depc.get_dependencies(tablename)
            # print('root_rowids = %r' % (root_rowids,))
            print('[depc.ancestor] dependency_levels = %s' % (ut.repr3(dependency_levels, nl=2),))
            rowid_dict = {depc.root: root_rowids}
            for level_keys in dependency_levels[1:]:
                print(' * level_keys %s ' % (level_keys,))
                for key in level_keys:
                    print('   * key = %r' % (key,))
                    table = depc[key]
                    parent_rowids = list(zip(*ut.dict_take(rowid_dict, table.parents)))
                    print('   * parent_rowids = %r' % (ut.truncate_str(repr(parent_rowids), 50),))
                    child_rowids = table.get_rowid_from_superkey(
                        parent_rowids, config=config, eager=eager, nInput=nInput,
                        ensure=ensure)
                    print('   * child_rowids = %r' % (ut.truncate_str(repr(child_rowids), 50),))
                    rowid_dict[key] = child_rowids
            print(' GOT ANCESTOR ROWIDS')
        return rowid_dict

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
                   eager=True, nInput=None):
        """
        Returns the rowids of `tablename` that correspond to `root_rowids` using `config`.
        """
        with ut.Indenter('[DEPC.GET_ROWIDS %s]' % (tablename,)):
            print(' * root_rowids=%s' % (ut.truncate_str(repr(root_rowids), 50),))
            print(' * config = %r' % (config,))
            rowid_dict = depc.get_ancestor_rowids(tablename, root_rowids,
                                                  config=config, ensure=ensure,
                                                  eager=eager, nInput=nInput)
            rowid_list = rowid_dict[tablename]
            print(' * return rowid_list = %r' % (ut.truncate_str(repr(rowid_list), 50),))
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
        with ut.Indenter('[GETPROP %s]' % (tablename,)):
            print('* root_rowids=%s' % (ut.truncate_str(repr(root_rowids), 50)))
            print(' * config = %r' % (config,))
            # Vectorized get of properties
            tbl_rowids = depc.get_rowids(tablename, root_rowids, config,
                                         ensure=ensure)
            print('[depc.get] tbl_rowids = %s' % (ut.truncate_str(repr(tbl_rowids), 50),))
            table = depc[tablename]
            prop_list = table.get_col(tbl_rowids, colnames)
            print('* return prop_list=%s' % (ut.truncate_str(repr(prop_list), 50),))
        return prop_list

    def get_native_property(depc, tablename, tbl_rowids, colnames=None):
        with ut.Indenter('[GETNATIVE %s]' % (tablename,)):
            table = depc[tablename]
            prop_list = table.get_col(tbl_rowids, colnames)
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
    """
    An individual node in the dependency graph.
    """

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
                    internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[str])
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
                        internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[dimtype])
                    _nested_idxs2.append(nest)
            else:
                _nested_idxs2.append(len(internal_data_colnames))
                internal_data_colnames.append(colname)
                internal_data_coltypes.append(lite.TYPE_TO_SQLTYPE[coltype])

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
        if False:
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
                            print('[deptbl.config] Warning: Config must either contain a string <tablename>_hashid or a dict <tablename>_config')
                            raise
            else:
                config_hashid = 'none'
        config_hashid = ut.hashstr27(ut.to_json(config))
        config_rowid = table.add_config(config_hashid)
        return config_rowid

    def get_config_rowid_from_hashid(table, config_hashid_list):
        config_rowid_list = table.db.get(
            CONFIG_TABLE, (CONFIG_ROWID,), config_hashid_list,
            id_colname=CONFIG_HASHID)
        return config_rowid_list

    def add_config(table, config_hashid):
        print('config_hashid = %r' % (config_hashid,))
        get_rowid_from_superkey = table.get_config_rowid_from_hashid
        config_rowid_list = table.db.add_cleanly(
            CONFIG_TABLE, (CONFIG_HASHID,), [(config_hashid,)],
            get_rowid_from_superkey)
        print('config_rowid_list = %r' % (config_rowid_list,))
        config_rowid = config_rowid_list[0]
        print('config_rowid = %r' % (config_rowid,))
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
            print('[deptbl.add] initial_rowid_list = %s' % (ut.truncate_str(repr(initial_rowid_list), 50),))
            print('[deptbl.add] config_rowid = %r' % (config_rowid,))
            # Get corresponding "dirty" parent rowids
            isdirty_list = ut.flag_None_items(initial_rowid_list)
            dirty_parent_rowids = ut.compress(parent_rowids, isdirty_list)
            num_dirty = len(dirty_parent_rowids)
            num_total = len(parent_rowids)
            verbose = True
            if num_dirty > 0:
                with ut.Indenter('[ADD]'):
                    if verbose:
                        fmtstr = '[deptbl.add] adding %d / %d new props to %r for config_rowid=%r'
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
                    def concat_rowids_data(dirty_parent_rowids, proptup_gen):
                        for parent_rowids, data_cols in zip(dirty_parent_rowids, proptup_gen):
                            try:
                                yield parent_rowids + (config_rowid,) + data_cols
                            except Exception as ex:
                                ut.printex(ex, 'cat error', keys=['config_rowid', 'data_cols', 'parent_rowids'])
                                raise

                    dirty_params_iter = concat_rowids_data(dirty_parent_rowids, proptup_gen)

                    # dirty_params_iter = (
                    #     parent_rowids + (config_rowid,) + data_cols
                    #     for parent_rowids, data_cols in zip(dirty_parent_rowids, proptup_gen))
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
                    print('[deptbl.add] rowid_list = %s' % (ut.truncate_str(repr(rowid_list), 50),))
            else:
                rowid_list = initial_rowid_list
            if return_num_dirty:
                return rowid_list, num_dirty
            else:
                print('[deptbl.add] rowid_list = %s' % (ut.truncate_str(repr(rowid_list), 50),))
                return rowid_list
        except Exception as ex:
            ut.printex(ex, 'error in add_rowids', keys=[
                'table', 'parent_rowids', 'config', 'args', 'config_rowid',
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
        print('[deptbl.get_rowid] Lookup %s rowids from superkey with %d parents' % (
            table.tablename, len(parent_rowids)))
        print('[deptbl.get_rowid] config = %r' % (config,))
        print('[deptbl.get_rowid] ensure = %r' % (ensure,))
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
        print('_get_rowid_from_superkey')
        print('_get_rowid_from_superkey table.tablename = %r ' % (table.tablename,))
        print('_get_rowid_from_superkey parent_rowids = %s' % (ut.truncate_str(repr(parent_rowids), 50)))
        print('_get_rowid_from_superkey config = %s' % (config))
        print('_get_rowid_from_superkey table.rowid_colname = %s' % (table.rowid_colname))
        print('_get_rowid_from_superkey config_rowid = %s' % (config_rowid))
        and_where_colnames = table.superkey_colnames
        params_iter = (rowids + (config_rowid,) for rowids in parent_rowids)
        params_iter = list(params_iter)
        #print('**params_iter = %r' % (params_iter,))
        rowid_list = table.db.get_where2(table.tablename, colnames, params_iter,
                                         and_where_colnames, eager=eager,
                                         nInput=nInput)
        print('_get_rowid_from_superkey rowid_list = %s' % (ut.truncate_str(repr(rowid_list), 50)))
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
        print('[deptbl.get_col] Get col of tablename=%r, colnames=%r with tbl_rowids=%s' %
              (table, colnames, ut.truncate_str(repr(tbl_rowids), 50)))
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
