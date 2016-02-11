# -*- coding: utf-8 -*-
"""
implicit version of dependency cache from ibeis/templates/template_generator
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import six
from six.moves import zip
from dtool import sql_control
from dtool import depcache_table
from dtool import base
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
    configclass (dtool.TableConfig): derivative of dtool.TableConfig.
        if None, a default class will be constructed for you. (default = None)
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
    configclass (dtool.AlgoConfig): derivative of dtool.AlgoConfig (default = None)
    docstr (None): (default = None)
    fname (str):  file name (default = None)
    chunksize (None): (default = None)

SeeAlso
    dtool.DependencyCache._register_algo
"""


def make_depcache_decors(root_tablename):
    """
    Makes global decorators to register functions for a tablename.

    A preproc function is meant to belong only to a single parent An algo
    function belongs to the root node, and may depend on a set of root nodes
    rather than just a single one.
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


class _CoreDependencyCache(object):
    """ Core worker functions for the depcache """

    #@ut.apply_docstr(REG_PREPROC_DOC)
    def _register_prop(depc, tablename, parents=None, colnames=None,
                       coltypes=None, preproc_func=None, docstr=None,
                       fname=None, chunksize=None, configclass=None,
                       #version=None,
                       isalgo=False, isinteractive=False,
                       asobject=False):
        """
        Registers a table with this dependency cache.
        """

        if depc._debug:
            print('[depc] Registering tablename=%r' % (tablename,))
            print('[depc]  * preproc_func=%r' % (preproc_func,))
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
        if configclass is None:
            # Make a default config with no parameters
            default_cfgdict = configclass
            configclass = base.dict_as_config({}, tablename)
        if isinstance(configclass, dict):
            # Dynamically make config class
            default_cfgdict = configclass
            configclass = base.dict_as_config(default_cfgdict, tablename)

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
            #version=version,
            isinteractive=isinteractive,
        )
        depc.cachetable_dict[tablename] = table
        depc.configclass_dict[tablename] = configclass
        return table

    #@ut.apply_docstr(REG_ALGO_DOC)
    def _register_algo(depc, algoname,
                       algo_result_class=None,
                       algo_request_class=None,
                       configclass=None,
                       algo_func=None,
                       #version=None,
                       docstr=None, fname=None, chunksize=None):
        """
        Registers an algorithm for the root of this dependency cache
        """
        import dtool
        if algo_result_class is None:
            algo_result_class = dtool.AlgoResult
        if algo_request_class is None:
            algo_request_class = dtool.AlgoRequest

        depc.requestclass_dict[algoname] = algo_request_class
        depc.resultclass_dict[algoname] = algo_result_class

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
                            configclass=configclass,
                            preproc_func=algo_func,
                            isalgo=True,
                            #version=version,
                            chunksize=chunksize)

    @profile
    def initialize(depc, _debug=None):
        """
        Creates all registered tables
        """
        print('[depc] INITIALIZE %s DEPCACHE' % (depc.root.upper(),))
        _debug = depc._debug if _debug is None else _debug
        if depc._use_globals:
            reg_preproc = __PREPROC_REGISTER__[depc.root]
            reg_algos = __ALGO_REGISTER__[depc.root]
            print('[depc.init] Regsitering %d global preproc funcs' % len(reg_preproc))
            for args_, kwargs_ in reg_preproc:
                depc._register_prop(*args_, **kwargs_)
            print('[depc.init] Regsitering %d global algos ' % len(reg_algos))
            for args_, kwargs_ in reg_algos:
                depc._register_algo(*args_, **kwargs_)

        ut.ensuredir(depc.cache_dpath)

        for fname in depc.fname_to_db.keys():
            if fname == ':memory:':
                fpath = fname
            else:
                fname_ = ut.ensure_ext(fname, '.sqlite')
                fpath = ut.unixjoin(depc.cache_dpath, fname_)
            if ut.get_argflag('--clear-all-depcache'):
                ut.delete(fpath)
            db = sql_control.SQLDatabaseController(fpath=fpath, simple=True)
            depcache_table.ensure_config_table(db)
            depc.fname_to_db[fname] = db
        print('[depc] Finished initialization')

        for table in depc.cachetable_dict.values():
            table.initialize(_debug=_debug)

        # HACKS:
        # Define injected functions for autocomplete convinience
        class InjectedDepc(object):
            pass
        depc.d = InjectedDepc()
        depc.w = InjectedDepc()
        d = depc.d
        w = depc.w
        for table in depc.cachetable_dict.values():
            attrname = 'get_{tablename}_rowids'.format(tablename=table.tablename)
            get_rowids = ut.partial(depc.get_rowids, table.tablename)
            wobj = InjectedDepc()
            # Set flat version
            setattr(d, attrname, get_rowids)
            # Set nested version
            setattr(w, table.tablename, wobj)
            setattr(wobj, 'get_rowids', get_rowids)

    def clear_all(depc):
        print('Clearning all cached data in %r' % (depc,))
        for table in depc.cachetable_dict.values():
            table.clear_table()

    # @ut.memoize
    def get_dependencies(depc, tablename):
        """
        gets level dependences from root to tablename

        CommandLine:
            python -m dtool.depcache_control --exec-get_dependencies
            python -m dtool.depcache_control --exec-get_dependencies:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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
            assert tablename in depc.cachetable_dict, (
                'tablename=%r does not exist' % (tablename,))
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
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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

    def get_all_ancestor_rowids(depc, tablename, native_rowids):
        r"""
        Gets the root_rowids of the root table associated with the
        `native_rowids` of `tablename.

        Args:
            tablename (str):
            root_rowids (list):
            config (None): (default = None)

        Returns:
            dict: rowid_dict

        CommandLine:
            python -m dtool.depcache_control --exec-get_all_ancestor_rowids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'spam'
            >>> target_root_rowids = [4, 9, 7]
            >>> native_rowids = depc.get_rowids(tablename, target_root_rowids)
            >>> rowid_dict = depc.get_all_ancestor_rowids(tablename, native_rowids)
            >>> root_rowids = list(rowid_dict[depc.root])
            >>> print(ut.repr3(rowid_dict, nl=1))
            >>> assert root_rowids == target_root_rowids
        """
        if depc._debug:
            print('[depc.descendant] GET ANSCESTOR ROWIDS %s ' % (tablename,))
        #get_native_property
        dependency_levels = depc.get_dependencies(tablename)
        rowid_dict = {}
        rowid_dict[tablename] = native_rowids

        # FIXME: not implemented very efficiently
        # Can do shortest existing path instead
        for level_keys in dependency_levels[::-1]:
            for tablekey in level_keys:
                if tablekey == depc.root:
                    break
                table = depc[tablekey]
                child_rowids = rowid_dict[tablekey]
                colnames = table.parent_rowid_colnames
                parent_rowids_listT = table.get_internal_columns(
                    child_rowids, colnames, keepwrap=True)
                parent_rowids_list = list(zip(*parent_rowids_listT))
                for parent_key, parent_rowids in zip(table.parents, parent_rowids_list):
                    rowid_dict[parent_key] = parent_rowids
        return rowid_dict

    def get_all_descendant_rowids(depc, tablename, root_rowids, config=None,
                                  ensure=True, eager=True, nInput=None,
                                  recompute=False, recompute_all=False,
                                  levels_up=None, _debug=False):
        r"""
        Connects `root_rowids` to rowids in `tablename`, and computes all
        values needed along the way. This is the main workhorse function for
        dependency computations.

        Args:
            tablename (str): table to compute dependencies to
            root_rowids (list): rowids for ``tablename``
            config (dict): config applicable for all tables (default = None)
            ensure (bool): eager evaluation if True(default = True)
            eager (bool): (default = True)
            nInput (None): (default = None)
            recompute (bool): (default = False)
            recompute_all (bool): (default = False)
            levels_up (int): only partially compute dependencies (default = 0)
            _debug (bool): (default = False)

        CommandLine:
            python -m dtool.depcache_control --exec-get_all_descendant_rowids:0
            python -m dtool.depcache_control --exec-get_all_descendant_rowids:1

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> tablename = 'spam'
            >>> root_rowids = [1, 2]
            >>> config1 = {'size': 500}
            >>> config2 = {'size': 100}
            >>> config3 = {'size': 500, 'adapt_shape': False}
            >>> ensure, eager, nInput = True, True, None
            >>> _debug = True
            >>> rowid_dict1 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config1, ensure, eager, nInput, _debug=_debug)
            >>> rowid_dict2 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config2, ensure, eager, nInput, _debug=_debug)
            >>> rowid_dict3 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config3, ensure, eager, nInput, _debug=_debug)
            >>> result1 = ut.repr3(rowid_dict1, nl=1)
            >>> result2 = ut.repr3(rowid_dict2, nl=1)
            >>> result3 = ut.repr3(rowid_dict3, nl=1)
            >>> result = '\n'.join([result1, result2, result3])
            >>> print(result)
            {
                'chip': [1, 2],
                'dummy_annot': [1, 2],
                'fgweight': [1, 2],
                'keypoint': [1, 2],
                'probchip': [1, 2],
                'spam': [1, 2],
            }
            {
                'chip': [3, 4],
                'dummy_annot': [1, 2],
                'fgweight': [3, 4],
                'keypoint': [3, 4],
                'probchip': [1, 2],
                'spam': [3, 4],
            }
            {
                'chip': [1, 2],
                'dummy_annot': [1, 2],
                'fgweight': [5, 6],
                'keypoint': [5, 6],
                'probchip': [1, 2],
                'spam': [5, 6],
            }


        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> _debug = True
            >>> tablename = 'dumbalgo'
            >>> config = depc.configclass_dict['dumbalgo']()
            >>> root_rowids = [1, 2, 3]
            >>> ensure, eager, nInput = False, True, None
            >>> # Get rowids of algo ( should be None )
            >>> rowid_dict = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config, ensure, eager, nInput,
            >>>     _debug=_debug)
            >>> result = ut.repr3(rowid_dict, nl=1)
            >>> print(result)
            {
                'dumbalgo': [None, None, None],
                'dummy_annot': [1, 2, 3],
            }

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> # Make sure algo config can correctly get properites
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> recompute = False
            >>> recompute_all = False
            >>> _debug = True
            >>> root_rowids = [1, 2]
            >>> configclass = depc.configclass_dict['chip']
            >>> config_ = configclass()
            >>> config1 = depc.configclass_dict['dumbalgo'](size=500)
            >>> config2 = depc.configclass_dict['dumbalgo'](size=100)
            >>> config = config2
            >>> prop_dicts1 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config=config1, _debug=_debug)
            >>> prop_dicts2 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config=config2, _debug=_debug)
            >>> print(prop_dicts2)
            >>> print(prop_dicts1)
            >>> assert prop_dicts1 != prop_dicts2
        """
        # TODO: Need to have a nice way of ensuring configs dont overlap
        # via namespaces.
        _debug = depc._debug if _debug is None else _debug
        indenter = ut.Indenter('[Descend-%s]' % (tablename,), enabled=_debug)
        if _debug:
            indenter.start()
            print(' * GET DESCENDANT ROWIDS %s ' % (tablename,))
            print(' * config = %r' % (config,))
        dependency_levels = depc.get_dependencies(tablename)
        if levels_up is not None:
            dependency_levels = dependency_levels[:-levels_up]

        configclass_levels = [[depc.configclass_dict.get(tablekey, None)
                               for tablekey in keys] for keys in dependency_levels]
        if _debug:
            print('[depc] dependency_levels = %s' %
                  (ut.repr3(dependency_levels, nl=1),))
            print('[depc] dependency_levels = %s' %
                  (ut.repr3(configclass_levels, nl=1),))
        rowid_dict = {depc.root: root_rowids}
        # Ensure that each level ``tablename``'s dependencies have been computed
        for level_keys in dependency_levels[1:]:
            if _debug:
                print(' * level_keys %s ' % (level_keys,))
            # For each table in the level
            for tablekey in level_keys:
                configclass = depc.configclass_dict.get(tablekey, None)
                requestclass = depc.requestclass_dict.get(tablekey, None)
                if configclass is None:
                    config_ = config
                else:
                    # Grab the correct configclass for the current table
                    config_ = None
                    if config is None:
                        config_ = configclass()
                    elif len(getattr(config, '_subconfig_attrs', [])) > 0:
                        # Get correct config for implicit dependencies
                        target_name = configclass().get_config_name()
                        if target_name in config._subconfig_names:
                            _index = config._subconfig_names.index(target_name)
                            subcfg_attr = config._subconfig_attrs[_index]
                            config_ = config[subcfg_attr]
                    if config_ is None:
                        # Preferable way to get configs with explicit
                        # configs
                        config_ = configclass(**config)
                table = depc[tablekey]
                parent_rowids = list(zip(*ut.dict_take(rowid_dict,
                                                       table.parents)))
                if _debug:
                    print('   * tablekey = %r' % (tablekey,))
                    print('   * configclass = %r' % (configclass,))
                    print('   * requestclass = %r' % (requestclass,))
                    print('   * config_ = %r' % (config_,))
                    print('   * parent_rowids = %s' %
                          (ut.trunc_repr(parent_rowids),))
                _recompute = recompute_all or (tablekey == tablename and recompute)
                child_rowids = table.get_rowid(
                    parent_rowids, config=config_, eager=eager, nInput=nInput,
                    ensure=ensure, recompute=_recompute)
                if _debug:
                    print('   * child_rowids = %s' %
                          (ut.trunc_repr(child_rowids),))
                rowid_dict[tablekey] = child_rowids
        if _debug:
            print(' GOT DESCENDANT ROWIDS')
            indenter.stop()
        return rowid_dict

    def new_algo_request(depc, algoname, qaids, daids, cfgdict=None):
        requestclass = depc.requestclass_dict[algoname]
        request = requestclass.new_algo_request(depc, algoname, qaids, daids, cfgdict)
        return request

    def get_ancestor_rowids(depc, tablename, native_rowids, anscestor_tablename):
        """
        anscestor_tablename = depc.root
        native_rowids = cid_list
        tablename = const.CHIP_TABLE
        """
        rowid_dict = depc.get_all_ancestor_rowids(tablename, native_rowids)
        anscestor_rowids = list(rowid_dict[anscestor_tablename])
        return anscestor_rowids

    def get_root_rowids(depc, tablename, native_rowids):
        return depc.get_ancestor_rowids(tablename, native_rowids, depc.root)

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
                   eager=True, nInput=None, _debug=None, recompute=False,
                   recompute_all=False):
        """
        Returns the rowids of `tablename` that correspond to `root_rowids`
        using `config`.
        """
        _debug = depc._debug if _debug is None else _debug
        with ut.Indenter('[GetRowID-%s]' % (tablename,),
                         enabled=_debug):
            if _debug:
                print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids),))
                print(' * config = %r' % (config,))
            # Compute everything from the root to the requested table
            rowid_dict = depc.get_all_descendant_rowids(
                tablename, root_rowids, config=config, ensure=ensure,
                eager=eager, nInput=nInput, recompute=recompute,
                recompute_all=recompute_all, _debug=ut.countdown_flag(_debug))
            rowid_list = rowid_dict[tablename]
            if _debug:
                print(' * return rowid_list = %s' % (ut.trunc_repr(rowid_list),))
        return rowid_list

    @ut.accepts_scalar_input2(argx_list=[1])
    def get_property(depc, tablename, root_rowids, colnames=None, config=None,
                     ensure=True, _debug=None, recompute=False,
                     recompute_all=False, read_extern=True):
        """
        Primary function to load or compute values in the dependency cache.

        Gets the data in `colnames` of `tablename` that correspond to
        `root_rowids` using `config`.  if colnames is None, all columns are
        returned.
        """
        _debug = depc._debug if _debug is None else _debug
        with ut.Indenter('[GetProp-%s]' % (tablename,), enabled=_debug):
            if _debug:
                print(' * tablename=%s' % (tablename))
                print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids)))
                print(' * colnames = %r' % (colnames,))
                print(' * config = %r' % (config,))
            # Vectorized get of properties
            tbl_rowids = depc.get_rowids(tablename, root_rowids, config=config,
                                         ensure=ensure, _debug=_debug,
                                         recompute=recompute,
                                         recompute_all=recompute_all)
            if _debug:
                print('[depc.get] tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
            table = depc[tablename]
            prop_list = table.get_row_data(tbl_rowids, colnames,
                                           read_extern=read_extern,
                                           _debug=_debug)
            if _debug:
                print('* return prop_list=%s' % (ut.trunc_repr(prop_list),))
        return prop_list

    get = get_property

    def get_native_property(depc, tablename, tbl_rowids, colnames=None,
                            _debug=None, read_extern=True):
        _debug = depc._debug if _debug is None else _debug
        with ut.Indenter('[GetNative %s]' % (tablename,), enabled=_debug):
            if _debug:
                print(' * tablename = %r' % (tablename,))
                print(' * colnames = %r' % (colnames,))
                print(' * tbl_rowids=%s' % (ut.trunc_repr(tbl_rowids)))
            table = depc[tablename]
            prop_list = table.get_row_data(tbl_rowids, colnames, _debug=_debug,
                                           read_extern=read_extern)
        return prop_list

    get_native = get_native_property

    def delete_property(depc, tablename, root_rowids, config=None):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids`
        using `config`.
        """
        rowid_list = depc.get_rowids(root_rowids, config=config, ensure=False)
        table = depc[tablename]
        num_deleted = table.delete_rows(rowid_list)
        return num_deleted


# Define the class with some "nice extras """
@six.add_metaclass(ut.ReloadingMetaclass)
class DependencyCache(_CoreDependencyCache, ut.NiceRepr):
    """
    To use this class a user must:
        * on root modification, call depc.on_root_modified
        * use decorators to register relevant functions
        * write an algorithm that accepts an AlgoRequest
          object, containing root ids and a configuration object.
    """
    def __init__(depc, root_tablename=None, cache_dpath='./DEPCACHE',
                 controller=None, default_fname=None,
                 #root_asobject=None,
                 get_root_uuid=None,
                 use_globals=True):
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
        depc.requestclass_dict = {}
        depc.resultclass_dict = {}
        # Mapping of different files properties are stored in
        depc.fname_to_db = {}
        # Function to map a root rowid to an object
        #depc._root_asobject = root_asobject
        depc._use_globals = use_globals
        depc.default_fname = default_fname
        depc._debug = ut.get_argflag(('--debug-depcache', '--debug-depc'))
        depc.get_root_uuid = get_root_uuid
        # depc._debug = True

    def get_tablenames(depc):
        return list(depc.cachetable_dict.keys())

    @property
    def tablenames(depc):
        return depc.get_tablenames()

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

    def print_config_tables(depc):
        for fname in depc.fname_to_db:
            print('---')
            print('db_fname = %r' % (fname,))
            depc.fname_to_db[fname].print_table_csv('config')

    def get_edges(depc):
        edges = [(parent, tablekey)
                 for tablekey, table in depc.cachetable_dict.items()
                 for parent in table.parents]
        return edges

    def get_implicit_edges(depc):
        # add implicit edges
        implicit_edges = []
        _inverted_ccdict = ut.invert_dict(depc.configclass_dict)
        for tablename2, configclass in depc.configclass_dict.items():
            if hasattr(configclass, 'get_sub_config_list'):
                cfg = configclass()
                subconfigs = cfg.get_sub_config_list()
                tablename1_list = ut.dict_take(_inverted_ccdict, subconfigs, None)
                for tablename1 in ut.filter_Nones(tablename1_list):
                    implicit_edges.append((tablename1, tablename2, {'implicit': True}))
        return implicit_edges

    def make_digraph(depc):
        """
        Helper "fluff" function

        CommandLine:
            python -m dtool --tf DependencyCache.make_digraph --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
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

        implicit_edges = depc.get_implicit_edges()
        graph.add_edges_from(implicit_edges)

        shape_dict = {
            # 'algo': 'star',
            'algo': 'circle',
            'node': 'circle',
            # 'root': 'rhombus',
            'root': 'circle',
        }
        import plottool as pt
        color_dict = {
            'algo': pt.DARK_GREEN,  # 'g',
            'node': None,
            'root': pt.RED,  # 'r',
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
        """ Helper "fluff" function """
        import plottool as pt
        graph = depc.make_digraph()
        pt.show_netx(graph, **kwargs)

    def __nice__(depc):
        infostr_ = 'nTables=%d' % len(depc.cachetable_dict)
        return '(%s) %s' % (depc.root_tablename, infostr_)

    def __getitem__(depc, tablekey):
        return depc.cachetable_dict[tablekey]

    @property
    def root(depc):
        return depc.root_tablename


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.depcache_control
        python -m dtool.depcache_control --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
