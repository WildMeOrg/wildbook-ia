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
from six.moves import zip
from dtool import sql_control
from dtool import depcache_table
(print, rrr, profile) = ut.inject2(__name__, '[depcache]')


# global function registry
__PREPROC_REGISTER__ = ut.ddict(list)
__ALGO_REGISTER__ = ut.ddict(list)


REG_PREPROC_DOC = """
Args:
    tablename (str):
    parents (list): (default = None)
    colnames (list): (default = None)
    coltypes (list): (default = None)
    chunksize (int): (default = None)
    config_class (dtool.TableConfig): derivative of dtool.TableConfig (default = None)
    docstr (str): (default = None)
    fname (str):  file name(default = None)
    asobject (bool): hacky dont use (default = False)

SeeAlso
    dtool.DependencyCache._register_prop
"""

REG_ALGO_DOC = """
Args:
    algoname (str):
    algo_result_class (class):
    config_class (dtool.AlgoConfig): derivative of dtool.AlgoConfig (default = None)
    docstr (None): (default = None)
    fname (str):  file name (default = None)
    chunksize (None): (default = None)

SeeAlso
    dtool.DependencyCache._register_algo
"""


def make_depcache_decors(root_tablename):
    """
    Makes global decorators to register functions for a tablename.

    A preproc function is meant to belong only to a single parent

    An algo function belongs to the root node, and may depend on a set of root
    nodes rather than just a single one.
    """

    @ut.apply_docstr(REG_PREPROC_DOC)
    def register_preproc(*args, **kwargs):
        """
        Global regsiter proproc function that will define a table for all
        dependency caches containing the parents. See
        DependencyCache._register_prop for additional information.

        """
        def register_preproc_wrapper(func):
            check_register(args, kwargs)
            kwargs['preproc_func'] = func
            __PREPROC_REGISTER__[root_tablename].append((args, kwargs))
            return func
        return register_preproc_wrapper

    @ut.apply_docstr(REG_ALGO_DOC)
    def register_algo(*args, **kwargs):
        def _wrapper(func):
            kwargs['algo_func'] = func
            __ALGO_REGISTER__[root_tablename].append((args, kwargs))
            return func
        return _wrapper

    return register_preproc, register_algo


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
        if default_fname is None:
            default_fname = ':memory:'
        # Root of all dependencies
        depc.root_tablename = root_tablename
        # Directory all cachefiles are stored in
        depc.cache_dpath = ut.truepath(cache_dpath)
        # Parent (ibs) controller
        depc.controller = controller
        # Internal dictionary of dependant tables
        depc.cachetable_dict = {}
        depc.configclass_dict = {}
        # Mapping of different files properties are stored in
        depc.fname_to_db = {}
        # Function to map a root rowid to an object
        depc._root_asobject = root_asobject
        depc._use_globals = use_globals
        depc.get_root_hashid = get_root_hashid
        depc.default_fname = default_fname
        depc._debug = False
        # depc._debug = True

    @ut.apply_docstr(REG_PREPROC_DOC)
    def register_preproc(depc, *args, **kwargs):
        """
        Decorator for registration of cachables
        """
        def register_preproc_wrapper(func):
            check_register(args, kwargs)
            kwargs['preproc_func'] = func
            depc._register_prop(*args, **kwargs)
            return func
        return register_preproc_wrapper

    @ut.apply_docstr(REG_ALGO_DOC)
    def register_algo(depc, *args, **kwargs):
        def reg_algo_wrapper(func):
            check_register(args, kwargs)
            kwargs['algo_func'] = func
            depc._register_algo(*args, **kwargs)
            return func
        return reg_algo_wrapper

    @ut.apply_docstr(REG_ALGO_DOC)
    def _register_algo(depc, algoname, algo_result_class=None,
                       config_class=None, algo_func=None,
                       docstr=None, fname=None, chunksize=None):
        """
        Registers an algorithm for the root of this dependency cache
        """
        if algo_result_class is None:
            import dtool
            algo_result_class = dtool.AlgoResult

        unbound_args = ut.get_unbound_args(algo_result_class.__init__)
        if len(unbound_args) > 1:
            msg = ut.codeblock(
                '''
                {classname} __init__ should not have any (non-self) unbound args.
                Detected len({unbound_args}) > 1 unbound args
                ''').format(classname=ut.get_classname(algo_result_class),
                            unbound_args=unbound_args)
            raise ValueError(msg)

        depc._register_prop(algoname,
                            coltypes=[algo_result_class.load_from_fpath],
                            config_class=config_class,
                            preproc_func=algo_func,
                            isalgo=True)

    # @ut.ut.apply_docstr()
    @ut.apply_docstr(REG_PREPROC_DOC)
    def _register_prop(depc, tablename, parents=None, colnames=None,
                       coltypes=None, preproc_func=None, docstr=None,
                       fname=None, chunksize=None,
                       config_class=None, isalgo=False, asobject=False):
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
        table = depcache_table.DependencyCacheTable(
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
            isalgo=isalgo,
        )
        depc.cachetable_dict[tablename] = table
        depc.configclass_dict[tablename] = config_class
        return table

    @profile
    def initialize(depc):
        print('[depc] INITIALIZE %s DEPCACHE' % (depc.root.upper(),))

        if depc._use_globals:
            reg_preproc = __PREPROC_REGISTER__[depc.root]
            reg_algos = __ALGO_REGISTER__[depc.root]
            print(' * regsitering %d global preproc funcs' % len(reg_preproc))
            for args_, kwargs_ in reg_preproc:
                depc._register_prop(*args_, **kwargs_)
            print(' * regsitering %d global algos ' % len(reg_algos))
            for args_, kwargs_ in reg_preproc:
                depc._register_algo(*args_, **kwargs_)

        ut.ensuredir(depc.cache_dpath)
        #print('depc.cache_dpath = %r' % (depc.cache_dpath,))
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
            depcache_table.ensure_config_table(db)
            depc.fname_to_db[fname] = db
        print('[depc] Finished initialization')

        for table in depc.cachetable_dict.values():
            table.initialize()

    def get_edges(depc):
        edges = [(parent, key)
                 for key, table in depc.cachetable_dict.items()
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
            python -m dtool --tf DependencyCache.make_digraph --show

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
        import networkx as netx
        graph = netx.DiGraph()
        nodes = list(depc.cachetable_dict.keys())
        edges = depc.get_edges()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        shape_dict = {
            # 'algo': 'star',
            'algo': 'circle',
            'node': 'circle',
            # 'root': 'rhombus',
            'root': 'circle',
        }
        color_dict = {
            'algo': 'g',
            'node': None,
            'root': 'r',
        }
        def _node_attrs(dict_):
            props = {k: dict_[v.tabletype] for k, v in
                     depc.cachetable_dict.items()}
            props[depc.root] = dict_['root']
            return props
        netx.set_node_attributes(graph, 'color', _node_attrs(color_dict))
        netx.set_node_attributes(graph, 'shape', _node_attrs(shape_dict))
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
            >>> depc._debug = True
            >>> tablename = 'spam'
            >>> root_rowids = [1, 2, 3]
            >>> config, ensure, eager, nInput = None, True, True, None
            >>> result = ut.repr3(depc.get_ancestor_rowids(tablename, root_rowids,
            >>>                                            config, ensure, eager,
            >>>                                            nInput), nl=1)
            >>> print(result)
            {
                'chip': [1, 2, 3],
                'dummy_annot': [1, 2, 3],
                'fgweight': [1, 2, 3],
                'keypoint': [1, 2, 3],
                'probchip': [1, 2, 3],
                'spam': [1, 2, 3],
            }

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depends_cache import *  # NOQA
            >>> from dtool.examples.dummy_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> depc._debug = True
            >>> tablename = 'dumbalgo'
            >>> root_rowids = [1, 2, 3]
            >>> config, ensure, eager, nInput = None, True, True, None
            >>> result = ut.repr3(depc.get_ancestor_rowids(tablename, root_rowids,
            >>>                                            config, ensure, eager,
            >>>                                            nInput), nl=1)
            >>> print(result)
        """
        # if config is None:
        configclass = depc.configclass_dict[tablename]
        if configclass is not None:
            # TODO: configclass should belong here
            pass
        # if True:
        with ut.Indenter('[ANCE %s]' % (tablename,), enabled=depc._debug):
            if depc._debug:
                print(' * GET ANCESTOR ROWIDS %s ' % (tablename,))
                print(' * config = %r' % (config,))
            dependency_levels = depc.get_dependencies(tablename)
            configclass_levels = [[depc.configclass_dict.get(key, None)
                                   for key in keys] for keys in dependency_levels]
            if depc._debug:
                print('[depc.ancestor] dependency_levels = %s' %
                      (ut.repr3(dependency_levels, nl=2),))
                print('[depc.ancestor] dependency_levels = %s' %
                      (ut.repr3(configclass_levels, nl=2),))
            rowid_dict = {depc.root: root_rowids}
            for level_keys in dependency_levels[1:]:
                if depc._debug:
                    print(' * level_keys %s ' % (level_keys,))
                #[depc.configclass_dict.get(key, None) for key in level_keys]
                for key in level_keys:
                    configclass = depc.configclass_dict.get(key, None)
                    if depc._debug:
                        print('   * key = %r' % (key,))
                        print('   * configclass = %r' % (configclass,))
                    if configclass is None:
                        config_ = config
                    else:
                        if config is None:
                            config_ = configclass()
                        else:
                            config_ = configclass(**config)
                    table = depc[key]
                    parent_rowids = list(zip(*ut.dict_take(rowid_dict,
                                                           table.parents)))
                    if depc._debug:
                        print('   * parent_rowids = %r' %
                              (ut.trunc_repr(parent_rowids),))
                    child_rowids = table.get_rowid_from_superkey(
                        parent_rowids, config=config_, eager=eager, nInput=nInput,
                        ensure=ensure)
                    if depc._debug:
                        print('   * child_rowids = %r' %
                              (ut.trunc_repr(child_rowids),))
                    rowid_dict[key] = child_rowids
            if depc._debug:
                print(' GOT ANCESTOR ROWIDS')
        return rowid_dict

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
                   eager=True, nInput=None):
        """
        Returns the rowids of `tablename` that correspond to `root_rowids`
        using `config`.
        """
        with ut.Indenter('[DEPC.GET_ROWIDS %s]' % (tablename,),
                         enabled=depc._debug):
            if depc._debug:
                print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids),))
                print(' * config = %r' % (config,))
            rowid_dict = depc.get_ancestor_rowids(tablename, root_rowids,
                                                  config=config, ensure=ensure,
                                                  eager=eager, nInput=nInput)
            rowid_list = rowid_dict[tablename]
            if depc._debug:
                print(' * return rowid_list = %r' % (ut.trunc_repr(rowid_list),))
        return rowid_list

    def delete_property(depc, tablename, root_rowids, config=None):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids`
        using `config`.
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
        with ut.Indenter('[GETPROP %s]' % (tablename,), enabled=depc._debug):
            if depc._debug:
                print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids)))
                print(' * config = %r' % (config,))
            # Vectorized get of properties
            tbl_rowids = depc.get_rowids(tablename, root_rowids, config,
                                         ensure=ensure)
            if depc._debug:
                print('[depc.get] tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
            table = depc[tablename]
            prop_list = table.get_row_data(tbl_rowids, colnames)
            if depc._debug:
                print('* return prop_list=%s' % (ut.trunc_repr(prop_list),))
        return prop_list

    def get_native_property(depc, tablename, tbl_rowids, colnames=None):
        with ut.Indenter('[GETNATIVE %s]' % (tablename,), enabled=depc._debug):
            if depc._debug:
                print(' * tablename = %r' % (tablename,))
                print(' * colnames = %r' % (colnames,))
                print(' * tbl_rowids=%s' % (ut.trunc_repr(tbl_rowids)))
            table = depc[tablename]
            prop_list = table.get_row_data(tbl_rowids, colnames)
        return prop_list

    @ut.accepts_scalar_input2(argx_list=[1])
    def get_obj(depc, tablename, root_rowids, config=None, ensure=True):
        """ Convinience function. Gets data in `tablename` as a list of
        objects. """
        print('WARNING EXPERIMENTAL')
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
