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
from collections import defaultdict
(print, rrr, profile) = ut.inject2(__name__, '[depcache]')


# global function registry
PREPROC_REGISTER = defaultdict(list)
SUBPROP_REGISTER = defaultdict(list)


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

SeeAlso:
    depcache_table.DependencyCacheTable
"""


def check_register(args, kwargs):
    assert len(args) < 6, 'too many args'
    assert 'preproc_func' not in kwargs, 'cannot specify func in wrapper'


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

        See dtool.depcache_control.REG_PREPROC_DOC if docstr is not autogened
        """
        def register_preproc_wrapper(func):
            check_register(args, kwargs)
            kwargs['preproc_func'] = func
            PREPROC_REGISTER[root_tablename].append((args, kwargs))
            return func
        return register_preproc_wrapper

    def register_subprop(*args, **kwargs):
        def _wrapper(func):
            kwargs['preproc_func'] = func
            SUBPROP_REGISTER[root_tablename].append((args, kwargs))
            return func
        return _wrapper

    _depcdecors = ut.odict({
        'preproc': register_preproc,
        'subprop': register_subprop,
    })
    return _depcdecors


class _CoreDependencyCache(object):
    """
    Core worker functions for the depcache
    Inherited by a calss with some "nice extras
    """

    def _register_prop(depc, tablename, parents=None, colnames=None,
                       coltypes=None, preproc_func=None, fname=None,
                       configclass=None, requestclass=None,
                       **kwargs):
        """
        Registers a table with this dependency cache.
        Essentially passes args down to make a DependencyTable.

        SEE: dtool.REG_PREPROC_DOC
        """
        if depc._debug:
            print('[depc] Registering tablename=%r' % (tablename,))
            print('[depc]  * preproc_func=%r' % (preproc_func,))
        if isinstance(tablename, six.string_types):
            tablename = six.text_type(tablename)
        if parents is None:
            parents = [depc.root]
        if colnames is None:
            colnames = 'data'
            if coltypes is None:
                coltypes = np.ndarray

        # Check if just a single column is given
        if not ut.isiterable(colnames):
            colnames = [colnames]
            coltypes = [coltypes]
            default_to_unpack = True
        else:
            default_to_unpack = False

            colnames = ut.lmap(six.text_type, colnames)
        if coltypes is None:
            raise ValueError('must specify coltypes of %s' % (tablename,))
            coltypes = [np.ndarray] * len(colnames)
        if fname is None:
            fname = depc.default_fname
        if configclass is None:
            # Make a default config with no parameters
            default_cfgdict = configclass
            configclass = base.dict_as_config({}, tablename)
        if isinstance(configclass, dict):
            # Dynamically make config class
            default_cfgdict = configclass
            configclass = base.dict_as_config(default_cfgdict, tablename)
        if requestclass is not None:
            depc.requestclass_dict[tablename] = requestclass

        depc.fname_to_db[fname] = None
        table = depcache_table.DependencyCacheTable(
            depc=depc,
            parent_tablenames=parents,
            tablename=tablename,
            data_colnames=colnames,
            data_coltypes=coltypes,
            preproc_func=preproc_func,
            fname=fname,
            default_to_unpack=default_to_unpack,
            **kwargs
        )
        depc.cachetable_dict[tablename] = table
        depc.configclass_dict[tablename] = configclass
        return table

    def _register_subprop(depc, tablename, propname=None, preproc_func=None):
        """ subproperties are always recomputeed on the fly """
        table = depc.cachetable_dict[tablename]
        table.subproperties[propname] = preproc_func

    def close(depc):
        for fname, db in depc.fname_to_db.items():
            db.close()

    @profile
    def initialize(depc, _debug=None):
        """
        Creates all registered tables
        """
        print('[depc] Initialize %s depcache' % (depc.root.upper(),))
        _debug = depc._debug if _debug is None else _debug
        if depc._use_globals:
            reg_preproc = PREPROC_REGISTER[depc.root]
            reg_subprop = SUBPROP_REGISTER[depc.root]
            if ut.VERBOSE:
                print('[depc.init] Registering %d global preproc funcs' % len(reg_preproc))
            for args_, kwargs_ in reg_preproc:
                depc._register_prop(*args_, **kwargs_)
            if ut.VERBOSE:
                print('[depc.init] Registering %d global subprops ' % len(reg_subprop))
            for args_, kwargs_ in reg_subprop:
                depc._register_subprop(*args_, **kwargs_)

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
        if ut.VERBOSE:
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
        inject_patterns = [
            ('get_{tablename}_rowids', depc.get_rowids),
            ('get_{tablename}_config_history', depc.get_config_history),
        ]
        for table in depc.cachetable_dict.values():
            wobj = InjectedDepc()
            # Set nested version
            setattr(w, table.tablename, wobj)
            for dfmtstr, func in inject_patterns:
                funcname = ut.get_funcname(func)
                attrname = dfmtstr.format(tablename=table.tablename)
                get_rowids = ut.partial(func, table.tablename)
                # Set flat version
                setattr(d, attrname, get_rowids)
                setattr(wobj, funcname, func)
            dfmtstr = 'get_{tablename}_{colname}'
            for colname in table.data_colnames:
                get_prop = ut.partial(depc.get_property, table.tablename, colnames=colname)
                attrname = dfmtstr.format(tablename=table.tablename, colname=colname)
                # Set flat version
                setattr(d, attrname, get_prop)
                setattr(wobj, 'get_' + colname, get_prop)

    # -----------------------------
    # GRAPH INSPECTION

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
            >>> tablename = 'fgweight'
            >>> result = ut.repr3(depc.get_dependencies(tablename), nl=1)
            >>> print(result)
            [
                ['dummy_annot'],
                ['chip', 'probchip'],
                ['keypoint'],
                ['fgweight'],
            ]
        """
        try:
            # get_ancestor_levels
            assert tablename in depc.cachetable_dict, (
                'tablename=%r does not exist' % (tablename,))
            root = depc.root_tablename
            children_, parents_ = list(zip(*depc.get_edges()))
            child_to_parents = ut.group_items(children_, parents_)
            if ut.VERYVERBOSE:
                print('root = %r' % (root,))
                print('tablename = %r' % (tablename,))
                print('child_to_parents = %s' % (ut.repr3(child_to_parents),))
            to_root = {tablename: ut.paths_to_root(tablename, root, child_to_parents)}
            if ut.VERYVERBOSE:
                print('to_root = %r' % (to_root,))
            from_root = ut.reverse_path(to_root, root, child_to_parents)
            dependency_levels_ = ut.get_levels(from_root)
            dependency_levels = ut.longest_levels(dependency_levels_)
        except Exception as ex:
            ut.printex(ex, 'error getting dependencies',
                       keys=['tablename', 'root', 'children_to_parents',
                             'to_root', 'from_root', 'dependency_levels_',
                             'dependency_levels', ])
            raise

        return dependency_levels

    def get_dependants(depc, tablename):
        """
        gets level dependences table to the leaves. ie ancestors

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
                ['fgweight', 'nnindexer', 'descriptor'],
                ['spam'],
                ['multitest'],
                ['multitest_score'],
            ]
        """
        # get_descendant_levels
        edges = depc.get_edges()
        children_, parents_ = list(zip(*edges))
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
                colnames = table.parent_id_colnames
                parent_rowids_listT = table.get_internal_columns(
                    child_rowids, colnames, keepwrap=True)
                parent_rowids_list = list(zip(*parent_rowids_listT))
                for parent_key, parent_rowids in zip(table.parent, parent_rowids_list):
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
            >>> config1 = {'dim_size': 500}
            >>> config2 = {'dim_size': 100}
            >>> config3 = {'dim_size': 500, 'adapt_shape': False}
            >>> ensure, eager, nInput = True, True, None
            >>> _debug = True
            >>> rowid_dict1 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config1, ensure, eager, nInput, _debug=_debug)
            >>> rowid_dict2 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config2, ensure, eager, nInput, _debug=_debug)
            >>> rowid_dict3 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config3, ensure, eager, nInput, _debug=_debug)
            >>> result1 = 'rowid_dict1 = ' + ut.repr3(rowid_dict1, nl=1)
            >>> result2 = 'rowid_dict2 = ' + ut.repr3(rowid_dict2, nl=1)
            >>> result3 = 'rowid_dict3 = ' + ut.repr3(rowid_dict3, nl=1)
            >>> result = '\n'.join([result1, result2, result3])
            >>> print(result)
            rowid_dict1 = {
                'chip': [1, 2],
                'dummy_annot': [1, 2],
                'fgweight': [1, 2],
                'keypoint': [1, 2],
                'probchip': [1, 2],
                'spam': [1, 2],
            }
            rowid_dict2 = {
                'chip': [3, 4],
                'dummy_annot': [1, 2],
                'fgweight': [3, 4],
                'keypoint': [3, 4],
                'probchip': [1, 2],
                'spam': [3, 4],
            }
            rowid_dict3 = {
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
            >>> tablename = 'vsmany'
            >>> config = depc.configclass_dict['vsmany']()
            >>> root_rowids = [1, 2, 3]
            >>> ensure, eager, nInput = False, True, None
            >>> # Get rowids of algo ( should be None )
            >>> rowid_dict = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config, ensure, eager, nInput,
            >>>     _debug=_debug)
            >>> result = ut.repr3(rowid_dict, nl=1)
            >>> print(result)
            {
                'dummy_annot': [1, 2, 3],
                'vsmany': [None, None, None],
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
            >>> config1 = depc.configclass_dict['vsmany'](dim_size=500)
            >>> config2 = depc.configclass_dict['vsmany'](dim_size=100)
            >>> config = config2
            >>> prop_dicts1 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config=config1, _debug=_debug)
            >>> prop_dicts2 = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config=config2, _debug=_debug)
            >>> print(prop_dicts2)
            >>> print(prop_dicts1)
            >>> assert prop_dicts1 != prop_dicts2

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.get_all_descendant_rowids), globals())
            >>> _debug = True
            >>> qaids, daids = [1, 2, 4], [2, 3, 4]
            >>> root_rowids = list(zip(*ut.product(qaids, daids)))
            >>> request = depc.new_request('vsone', qaids, daids)
            >>> results = request.execute()
            >>> tablename = 'vsone'
            >>> rowid_dict = depc.get_all_descendant_rowids(
            >>>     tablename, root_rowids, config=None, _debug=_debug)
        """
        # TODO: Need to have a nice way of ensuring configs dont overlap
        # via namespaces.
        _debug = depc._debug if _debug is None else _debug
        indenter = ut.Indenter('[Descend-to-%s]' % (tablename,), enabled=_debug)
        if _debug:
            indenter.start()
            print(' * GET DESCENDANT ROWIDS %s ' % (tablename,))
            print(' * config = %r' % (config,))
        dependency_levels = depc.get_dependencies(tablename)
        if levels_up is not None:
            dependency_levels = dependency_levels[:-levels_up]

        configclass_levels = [
            [depc.configclass_dict.get(tablekey, None)
             for tablekey in keys]
            for keys in dependency_levels
        ]
        if _debug:
            print('[depc] dependency_levels = %s' %
                  ut.repr3(dependency_levels, nl=1))
            print('[depc] config_levels = %s' %
                  ut.repr3(configclass_levels, nl=1))

        # TODO: better support for multi-edges
        if (len(root_rowids) > 0 and ut.isiterable(root_rowids[0]) and
             not depc[tablename].ismulti):
            rowid_dict = {}
            for colx, col in enumerate(root_rowids):
                rowid_dict[depc.root + '%d' % (colx + 1,)] = col
            rowid_dict[depc.root] = ut.unique_ordered(ut.flatten(root_rowids))
        else:
            rowid_dict = {depc.root: root_rowids}

        # Ensure that each level ``tablename``'s dependencies have been computed
        for level_keys in dependency_levels[1:]:
            if _debug:
                print(' * level_keys %s ' % (level_keys,))
            # For each table in the level
            for tablekey in level_keys:
                try:
                    child_rowids = depc._expand_level_rowids(
                        tablename, tablekey, rowid_dict, ensure, eager, nInput,
                        config, recompute, recompute_all, _debug)
                except Exception as ex:
                    table = depc[tablekey]  # NOQA
                    ut.printex(ex, 'error expanding rowids',
                               keys=['tablename', 'tablekey', 'rowid_dict',
                                     'config', 'table', 'dependency_levels'])
                    raise
                rowid_dict[tablekey] = child_rowids
        if _debug:
            print(' GOT DESCENDANT ROWIDS')
            indenter.stop()
        return rowid_dict

    def _ensure_config(depc, tablekey, config):
        """
        Creates a full table configuration with all defaults using config

        Args:
            tablekey (str): name of the table to grab config from
            config (dict): may be overspecified or underspecfied
        """
        configclass = depc.configclass_dict.get(tablekey, None)
        #requestclass = depc.requestclass_dict.get(tablekey, None)
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
        return config_

    #def get_relevant_subconfigs(depc, tablename, config):
    #    depc._ensure_config(tablename, config)
    #    pass

    def _get_parent_rowids(depc, table, rowid_dict):
        # FIXME to handle multiedges correctly
        parent_rowidsT = ut.dict_take(rowid_dict,
                                      table.parent_id_tablenames)
        if table.ismulti:
            parent_rowids = parent_rowidsT
        else:
            parent_rowids = ut.list_transpose(parent_rowidsT)
        return parent_rowids

    def _expand_level_rowids(depc, tablename, tablekey, rowid_dict, ensure,
                             eager, nInput, config, recompute, recompute_all,
                             _debug):
        table = depc[tablekey]
        config_ = depc._ensure_config(tablekey, config)
        parent_rowids = depc._get_parent_rowids(table, rowid_dict)
        if _debug:
            print('   * tablekey = %r' % (tablekey,))
            print('   * config_ = %r' % (config_,))
            print('   * config_rowid = %r' % (table.get_config_rowid(config_),))
            print('   * parent_rowids = %s' % (ut.trunc_repr(parent_rowids),))
        _recompute = recompute_all or (tablekey == tablename and recompute)
        level_rowids = table.get_rowid(
            parent_rowids, config=config_, eager=eager, nInput=nInput,
            ensure=ensure, recompute=_recompute)
        if _debug:
            print('   * level_rowids = %s' % (ut.trunc_repr(level_rowids),))
        return level_rowids

    # -----------------------------
    # STATE GETTERS

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
                   eager=True, nInput=None, _debug=None, recompute=False,
                   recompute_all=False):
        """
        Returns the rowids of `tablename` that correspond to `root_rowids`
        using `config`.

        Ignore:
            tablename = 'nnindexer'
            multi_rowids = (1, 2, 3, 4, 5)
            root_rowids = [[multi_rowids]]
            import plottool as pt
            pt.ensure_pylab_qt4()

            from dtool.depcache_control import *  # NOQA
            from dtool.example_depcache import testdata_depc
            depc = testdata_depc()
            exec(ut.execstr_funckw(depc.get_rowids), globals())
            print(ut.depth_profile(root_rowids))
            tablename = 'neighbs'
            table = depc[tablename]  # NOQA
            import plottool as pt
            pt.ensure_pylab_qt4()
            _debug = depc._debug = True
            depc.get_rowids(tablename, root_rowids, config, _debug=_debug)

            pt.show_nx(depc.graph)
            for key, val in table.type_to_subgraph.items():
                pt.show_nx(val)
                pt.set_title(key)

        CommandLine:
            python -m dtool.depcache_control --exec-get_rowids
            python -m dtool.depcache_control --dump-get_rowids
            python -m dtool.depcache_control --exec-get_rowids:0

        GridParams:
            >>> param_grid = dict(
            >>>     tablename=[ 'spam', 'neighbs'] # 'spam', 'multitest_score','keypoint'],
            >>>   #tablename=['neighbs', 'keypoint', 'spam', 'multitest_score','keypoint'],
            >>> )
            >>> flat_root_ids = [1, 2, 3]
            >>> combos = ut.all_dict_combinations(param_grid)
            >>> index = 0
            >>> keys = 'tablename'.split(', ')
            >>> tablename, = ut.dict_take(combos[index], keys)

        Setup:
            >>> # DISABLE_GRID_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
            >>> import plottool as pt
            >>> pt.ensure_pylab_qt4()
            >>> #pt.show_nx(depc.graph)

        GridExample0:
            >>> table = depc[tablename]  # NOQA
            >>> flat_root_ids = [1, 2, 3]
            >>> root_rowids = [flat_root_ids for _ in table.input_order]
            >>> print('root_rowids = %r' % (root_rowids,))
            >>> #root_rowids = [[flat_root_ids], [(flat_root_ids,)]]
            >>> #root_rowids = [list(zip(flat_root_ids)), (flat_root_ids,)]
            >>> _debug = True
            >>> depc.get_rowids(tablename, root_rowids, config, _debug=_debug)
            >>> for key, val in table.type_to_subgraph.items():
            >>>     pt.show_nx(val)
            >>>     pt.set_title(key)

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
            >>> root_rowids = [1, 2, 3]
            >>> tablename = 'spam'
            >>> table = depc[tablename]
            >>> kp_rowids = depc.get_rowids(tablename, root_rowids)
            >>> #result = ('prop_list = %s' % (ut.repr2(prop_list),))
            >>> #print(result)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
            >>> flat_root_ids = [1, 2, 3]
            >>> kp_rowids = depc.get_rowids('keypoint', flat_root_ids)
            >>> root_rowids = [flat_root_ids] * 8
            >>> _debug = True
            >>> tablename = 'nnindexer'
            >>> tablename = 'multitest_score'
            >>> table = depc[tablename]  # NOQA
            >>> #result = ('prop_list = %s' % (ut.repr2(prop_list),))
            >>> # print(result)
        """
        _debug = depc._debug if _debug is None else _debug
        if _debug:
            print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids),))
            print(' * config = %r' % (config,))
        table = depc[tablename]  # NOQA
        INDEXER_VERSION = False

        if tablename == 'neighbor_index':
            """
            python -m ibeis.core_annots --exec-compute_neighbor_index --show
            """

            import utool
            utool.embed()

        if INDEXER_VERSION or tablename == 'neighbs':
            compute_order = table.compute_order
            depend_order = compute_order['depend_compute_ids']
            input_order = compute_order['input_compute_ids']

            if _debug:
                print(' * input_order = %s' % (ut.repr3(input_order, nl=1),))
                print(' * depend_order = %s' % (ut.repr3(depend_order, nl=1),))
            if len(input_order) > 1:
                assert ut.depth_atleast(root_rowids, 2), (
                    'input_order = %r' % (input_order,))

            with ut.Indenter('[GetRowID-%s]' % (tablename,),
                             enabled=_debug):
                # New way to get rowids
                input_level = depend_order[0]
                mid_levels = depend_order[1:-1]
                output_level = depend_order[-1]

                # List that holds a mapping from input order to input "name"
                input_order_lookup = ut.make_index_lookup(input_order)
                # Dictionary that holds the rowids computed for each table
                # while tracing the dependencies.
                rowid_lookup = ut.odict([(key, ut.odict()) for key in input_order])

                # Need to split each path into parts.
                # Each part represents another level of unflattening
                # (because root indicies are all flat)

                # Handle input level
                assert input_level[0] == depc.root
                for compute_id in input_order:
                    # for name in input_names:
                    argx = input_order_lookup[compute_id]
                    rowid_lookup[compute_id] = root_rowids[argx]
                    # HACK: Flatten to scalars
                    # The inputs should just be given in the "correct" nesting.
                    # TODO: determine what correct nesting is.
                    for i in range(5):
                        try:
                            current = rowid_lookup[compute_id]
                            rowid_lookup[compute_id] = ut.flatten(current)
                        except Exception:
                            pass

                level = 0
                if _debug:
                    print('input_order_lookup = %r' % (input_order_lookup,))
                    ut.printdict(rowid_lookup, 'rowid_lookup')

                def handle_level(compute_id, rowid_lookup, _recompute, level):
                    print('+--- HANDLE LEVEL %d -------' % (level,))
                    tablekey = compute_id[0]
                    input_suff = compute_id[1]
                    config_ = depc._ensure_config(tablekey, config)
                    table = depc[tablekey]
                    lookupkeys = [(n, input_suff) for n in table.parent_id_tablenames]
                    # ordering = ut.dict_take(input_order_lookup, input_names)
                    # sortx = ut.argsort(ordering)
                    # FIXME: get inputs for each table.
                    # input_names = ut.take(input_names, sortx)
                    # lookupkeys = list(ut.iprod(table.parent_id_tablenames, input_names))
                    # lookupkeys = list(zip(table.parent_id_tablenames, input_types))
                    if _debug:
                        print('---- LOCALS ------')
                        ut.print_locals(compute_id, tablekey, lookupkeys, table)
                        print('L----------')
                    # FIXME generalize
                    _parent_ids = [rowid_lookup[tblkey] for tblkey in lookupkeys]
                    if table.ismulti:
                        parent_rowidsT = [[tuple(x)] for x in _parent_ids]
                    else:
                        parent_rowidsT = _parent_ids
                    parent_rowidsT = np.broadcast_arrays(*parent_rowidsT)
                    parent_rowids = list(zip(*parent_rowidsT))
                    # Probably not right for general multi-input
                    import utool
                    with utool.embed_on_exception_context:
                        next_rowids = table.get_rowid(
                            parent_rowids, config=config_, eager=eager, nInput=nInput,
                            ensure=ensure, recompute=_recompute)
                    rowid_lookup[compute_id] = next_rowids
                    if _debug:
                        ut.printdict(rowid_lookup, 'rowid_lookup')
                    if _debug:
                        print('L___ HANDLE LEVEL %d -------' % (level,))
                    return next_rowids

                # Handle mid levels
                _recompute = recompute_all
                for level, compute_id in enumerate(mid_levels, start=1):
                    handle_level(compute_id, rowid_lookup, _recompute, level)
                level += 1

                # Handel final (requested) level
                compute_id = output_level
                _recompute = recompute
                rowid_list =  handle_level(compute_id, rowid_lookup,
                                           _recompute, level)
        else:
            with ut.Indenter('[GetRowID-%s]' % (tablename,),
                             enabled=_debug):
                # TODO: Get nonself rowids first
                # THen get self rowids for debugging ease
                try:
                    if False:
                        recompute_ = recompute or recompute_all
                        parent_rowids = depc._get_parent_input(
                            tablename, root_rowids, config, ensure=True, _debug=None,
                            recompute=False, recompute_all=False, eager=True,
                            nInput=None)
                        config_ = depc._ensure_config(tablename, config)
                        #if onthefly:
                        #    pass
                        table = depc[tablename]
                        rowid_list = table.get_rowid(
                            parent_rowids, config=config_, eager=eager, nInput=nInput,
                            ensure=ensure, recompute=recompute_)
                    else:
                        # Compute everything from the root to the requested table
                        rowid_dict = depc.get_all_descendant_rowids(
                            tablename, root_rowids, config=config, ensure=ensure,
                            eager=eager, nInput=nInput, recompute=recompute,
                            recompute_all=recompute_all, _debug=ut.countdown_flag(_debug))
                        rowid_list = rowid_dict[tablename]
                except depcache_table.ExternalStorageException:
                    print('EXTERNAL EXCEPTION One retry in get_rowids')
                    rowid_dict = depc.get_all_descendant_rowids(
                        tablename, root_rowids, config=config, ensure=ensure,
                        eager=eager, nInput=nInput, recompute=recompute,
                        recompute_all=recompute_all, _debug=ut.countdown_flag(_debug))
                    rowid_list = rowid_dict[tablename]
        if _debug:
            print(' * return rowid_list = %s' % (ut.trunc_repr(rowid_list),))
        return rowid_list

    def get_ancestor_rowids(depc, tablename, native_rowids, ancestor_tablename=None):
        """
        ancestor_tablename = depc.root
        native_rowids = cid_list
        tablename = const.CHIP_TABLE
        """
        if ancestor_tablename is None:
            ancestor_tablename = depc.root
        # rowid_dict = depc.get_all_ancestor_rowids(tablename, native_rowids)
        # ancestor_rowids = list(rowid_dict[ancestor_tablename])
        table = depc[tablename]
        ancestor_rowids = table.get_ancestor_rowids(native_rowids, ancestor_tablename)
        return ancestor_rowids

    def _get_parent_input(depc, tablename, root_rowids, config, ensure=True,
                          _debug=None, recompute=False, recompute_all=False,
                          eager=True, nInput=None):
        # Get ancestor rowids that are descendants of root
        table = depc[tablename]
        rowid_dict = depc.get_all_descendant_rowids(
            tablename, root_rowids, config=config, ensure=ensure,
            eager=eager, nInput=nInput, recompute=recompute,
            recompute_all=recompute_all, _debug=ut.countdown_flag(_debug),
            levels_up=1)
        parent_rowids = depc._get_parent_rowids(table, rowid_dict)
        return parent_rowids

    def _parse_sqlkw(kwargs):
        default_sqlkw = dict(
            _debug=None, ensure=True, recompute=False, recompute_all=False,
            eager=True, nInput=None, read_extern=True, onthefly=False,
        )
        otherkw = kwargs.copy()
        sqlkw = {key: otherkw.pop(key, val) for key, val in default_sqlkw.items()}
        return sqlkw, otherkw

    @ut.accepts_scalar_input2(argx_list=[1])
    def get_property(depc, tablename, root_rowids, colnames=None, config=None,
                     ensure=True, _debug=None, recompute=False,
                     recompute_all=False, eager=True, nInput=None,
                     read_extern=True, onthefly=False, num_retries=1,
                     hack_paths=False):
        """
        Primary function to load or compute values in the dependency cache.

        Gets the data in `colnames` of `tablename` that correspond to
        `root_rowids` using `config`.  if colnames is None, all columns are
        returned.

        Args:
            tablename (str): table name containing desired property
            root_rowids (List[int]): ids of the root object
            colnames (None): desired property (default = None)
            config (None): (default = None)
            read_extern: if False then only returns extern URI
            hack_paths: if False then does not compute extern info just returns
                path that it will be located at

        Returns:
            list: prop_list

        CommandLine:
            python -m dtool.depcache_control --exec-get_property

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.get_property), globals())
            >>> _debug = True
            >>> tablename = 'keypoint'
            >>> root_rowids = [1, 2, 3]
            >>> prop_list = depc.get_property(
            >>>     tablename, root_rowids, colnames, config, ensure, _debug,
            >>>     recompute, recompute_all, read_extern, eager, nInput)
            >>> result = ('prop_list = %s' % (ut.repr2(prop_list),))
            >>> print(result)
        """
        if tablename == depc.root_tablename:
            return depc.root_getters[colnames](root_rowids)
            #pass
        _debug = depc._debug if _debug is None else _debug
        with ut.Indenter('[GetProp-%s]' % (tablename,), enabled=_debug):
            if _debug:
                print(' * tablename=%s' % (tablename))
                print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids)))
                print(' * colnames = %r' % (colnames,))
                print(' * config = %r' % (config,))

            if hack_paths and not ensure and not read_extern:
                # HACK: should be able to not compute rows to get certain properties
                from os.path import join
                #recompute_ = recompute or recompute_all
                parent_rowids = depc._get_parent_input(
                    tablename, root_rowids, config, ensure=True, _debug=None,
                    recompute=False, recompute_all=False, eager=True,
                    nInput=None)
                config_ = depc._ensure_config(tablename, config)
                table = depc[tablename]
                extern_dpath = table.extern_dpath
                ut.ensuredir(extern_dpath, verbose=False or table.depc._debug)
                fname_list = table.get_extern_fnames(parent_rowids,
                                                     config=config_,
                                                     extern_col_index=0)
                fpath_list = [join(extern_dpath, fname) for fname in fname_list]
                return fpath_list

            for trynum in range(num_retries):
                try:
                    # Vectorized get of properties
                    tbl_rowids = depc.get_rowids(tablename, root_rowids,
                                                 config=config, ensure=ensure,
                                                 _debug=_debug, eager=eager,
                                                 recompute=recompute,
                                                 recompute_all=recompute_all,
                                                 nInput=nInput)
                    if _debug:
                        print('[depc.get] tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
                    table = depc[tablename]
                    prop_list = table.get_row_data(tbl_rowids, colnames,
                                                   read_extern=read_extern,
                                                   _debug=_debug, eager=eager,
                                                   ensure=ensure, nInput=nInput)
                except depcache_table.ExternalStorageException:
                    print('!!* Hit ExternalStorageException')
                    if trynum == num_retries - 1:
                        raise
                else:
                    break
            if _debug:
                print('* return prop_list=%s' % (ut.trunc_repr(prop_list),))
        return prop_list

    def get_config_history(depc, tablename, root_rowids, config=None):
        # Vectorized get of properties
        tbl_rowids = depc.get_rowids(tablename, root_rowids, config=config)
        return depc[tablename].get_config_history(tbl_rowids)

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

    def new_request(depc, tablename, qaids, daids, cfgdict=None):
        """ creates a request for data that can be executed later """
        print('[depc] NEW %s request' % (tablename,))
        requestclass = depc.requestclass_dict[tablename]
        request = requestclass.new(depc, qaids, daids, cfgdict,
                                   tablename=tablename)
        return request

    def get_root_rowids(depc, tablename, native_rowids):
        return depc.get_ancestor_rowids(tablename, native_rowids, depc.root)

    # -----------------------------
    # STATE MODIFIERS

    def notify_root_changed(depc, root_rowids, prop):
        """
        this is where we are notified that a "registered" root property has
        changed.
        """
        print('[depc] notified that columns (%s) for (%d) row(s) were modified' %
              (prop, len(root_rowids),))
        # for key in tables_depending_on(prop)
        #depc.delete_property(key, root_rowids)
        # TODO: check which properties were invalidated by this prop
        # TODO; remove invalidated properties
        #depc.delete_root(root_rowids)
        pass

    def clear_all(depc):
        print('Clearning all cached data in %r' % (depc,))
        for table in depc.cachetable_dict.values():
            table.clear_table()

    def delete_property(depc, tablename, root_rowids, config=None):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids`
        using `config`.
        """
        rowid_list = depc.get_rowids(tablename, root_rowids, config=config,
                                     ensure=False)
        table = depc[tablename]
        num_deleted = table.delete_rows(rowid_list)
        return num_deleted

    def make_root_info_uuid(depc, root_rowids, info_props):
        """
        Creates a uuid that depends on certain properties of the root object.
        This is used for implicit cache invalidation because, if those
        properties change then this uuid also changes.

        The depcache needs to know about stateful properties of dynamic root
        objects in order to correctly compute their hashes.

        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> root_rowids = ibs._get_all_aids()
        >>> depc = ibs.depc_annot
        >>> info_props = ['image_uuid', 'verts', 'theta']
        >>> info_props = ['image_uuid', 'verts', 'theta', 'name', 'species', 'yaw']
        """
        getters = ut.dict_take(depc.root_getters, info_props)
        infotup_list = zip(*[getter(root_rowids) for getter in getters])
        info_uuid_list = [ut.augment_uuid(*tup) for tup in infotup_list]
        return info_uuid_list

    def get_uuids(depc, tablename, root_rowids, config=None):
        """
        # TODO: Make uuids for dependant object based on root uuid and path of
        # construction.
        """
        if tablename == depc.root:
            uuid_list = depc.get_root_uuid(root_rowids)
        return uuid_list

    def delete_root(depc, root_rowids):
        r"""
        Args:
            root_rowids (list):

        CommandLine:
            python -m dtool.depcache_control delete_root --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.delete_root), globals())
            >>> root_rowids = [1]
            >>> depc.delete_root(root_rowids)
            >>> depc.get('fgweight', [1])
            >>> depc.delete_root(root_rowids)
        """
        graph = depc.make_graph(implicit=False)
        # check to make sure child does not have another parent
        children = [child for child in graph.succ[depc.root_tablename]
                    if len(graph.pred[child]) == 1]
        for tablename in children:
            depc.delete_property(tablename, root_rowids)


@six.add_metaclass(ut.ReloadingMetaclass)
class DependencyCache(_CoreDependencyCache, ut.NiceRepr):
    """
    To use this class a user must:
        * on root modification, call depc.on_root_modified
        * use decorators to register relevant functions
    """
    def __init__(depc, root_tablename=None, cache_dpath='./DEPCACHE',
                 controller=None, default_fname=None,
                 #root_asobject=None,
                 get_root_uuid=None,
                 root_getters=None,
                 use_globals=True):
        if default_fname is None:
            default_fname = ':memory:'
        depc.root_getters = root_getters
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
    def tables(depc):
        return list(depc.cachetable_dict.values())

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

    def get_edges(depc, data=False):
        if data:
            def get_edgedata(tablekey, parentkey, parent_data):
                if parent_data['ismulti'] or parent_data['isnwise']:
                    edge_type_parts = []
                    local_input_id = ''
                    if parent_data['ismulti']:
                        # TODO: give different ids to multi edges
                        # edge_type_parts.append('multi_%s_%s' % (parentkey, tablekey))
                        edge_type_parts.append('multi')
                        local_input_id += '*'
                    if parent_data['isnwise']:
                        edge_type_parts.append('nwise_%s' % (
                             parent_data['nwise_idx'],))
                        local_input_id += six.text_type(parent_data['nwise_idx'])
                        # edge_type_parts.append('nwise_%s_%s_%s' % (
                        #     parentkey, tablekey, parent_data['nwise_idx'],))
                    edge_type_id = '_'.join(edge_type_parts)
                else:
                    edge_type_id = 'normal'
                    local_input_id = '1'
                edge_data = {
                    'ismulti': parent_data['ismulti'],
                    'isnwise': parent_data.get('isnwise'),
                    'nwise_idx': parent_data.get('nwise_idx'),
                    'parent_colx': parent_data.get('parent_colx'),
                    'edge_type': edge_type_id,
                    'local_input_id': local_input_id,
                    'taillabel': local_input_id,  # proper graphviz attribute
                }
                return edge_data
            edges = [
                (parentkey, tablekey,
                 get_edgedata(tablekey, parentkey, parent_data))
                for tablekey, table in depc.cachetable_dict.items()
                for parentkey, parent_data in table.parents(data=True)
            ]
        else:
            edges = [
                (parentkey, tablekey)
                for tablekey, table in depc.cachetable_dict.items()
                for parentkey in table.parents(data=False)
            ]
        return edges

    def get_implicit_edges(depc, data=False):
        """
        Edges defined by subconfigurations
        """
        # add implicit edges
        implicit_edges = []
        # Map config classes to tablenames
        _inverted_ccdict = ut.invert_dict(depc.configclass_dict)
        for tablename2, configclass in depc.configclass_dict.items():
            cfg = configclass()
            subconfigs = cfg.get_sub_config_list()
            if subconfigs is not None and len(subconfigs) > 0:
                tablename1_list = ut.dict_take(_inverted_ccdict, subconfigs, None)
                for tablename1 in ut.filter_Nones(tablename1_list):
                    implicit_edges.append((tablename1, tablename2))
        if data:
            implicit_edges = [(e1, e2, {'implicit': True})
                              for e1, e2 in implicit_edges]
        return implicit_edges

    @ut.memoize
    def make_graph(depc, **kwargs):
        """
        Helper "fluff" function

        CommandLine:
            python -m dtool --tf DependencyCache.make_graph --show --reduced

            python -m ibeis.control.IBEISControl show_depc_annot_graph --show --reduced

            python -m ibeis.control.IBEISControl show_depc_annot_graph --show --reduced --testmode
            python -m ibeis.control.IBEISControl show_depc_annot_graph --show --testmode

            python -m ibeis.control.IBEISControl --test-show_depc_image_graph --show --reduced
            python -m ibeis.control.IBEISControl --test-show_depc_image_graph --show

            python -m ibeis.scripts.specialdraw double_depcache_graph --show --testmode

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> import utool as ut
            >>> depc = testdata_depc()
            >>> graph = depc.make_graph(reduced=ut.get_argflag('--reduced'))
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.ensure_pylab_qt4()
            >>> import networkx as nx
            >>> #pt.show_nx(nx.dag.transitive_closure(graph))
            >>> #pt.show_nx(ut.nx_transitive_reduction(graph))
            >>> pt.show_nx(graph)
            >>> pt.show_nx(graph, layout='agraph')
            >>> ut.show_if_requested()
        """
        import networkx as nx
        # graph = nx.DiGraph()
        graph = nx.MultiDiGraph()
        nodes = list(depc.cachetable_dict.keys())
        edges = depc.get_edges(data=True)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        if kwargs.get('implicit', True):
            implicit_edges = depc.get_implicit_edges(data=True)
            graph.add_edges_from(implicit_edges)

        shape_dict = {
            'node': 'circle',
            #'node': 'rect',
            'node': 'ellipse',
            #'root': 'rhombus',
            #'root': 'circle',
            #'root': 'circle',
            'root': 'ellipse',
            #'root': 'rect',
        }
        import plottool as pt
        color_dict = {
            #'algo': pt.DARK_GREEN,  # 'g',
            'node': pt.NEUTRAL_BLUE,
            'root': pt.RED,  # 'r',
        }
        def _node_attrs(dict_):
            props = {k: dict_['node'] for k, v in
                     depc.cachetable_dict.items()}
            props[depc.root] = dict_['root']
            return props
        nx.set_node_attributes(graph, 'color', _node_attrs(color_dict))
        nx.set_node_attributes(graph, 'shape', _node_attrs(shape_dict))
        if kwargs.get('reduced', False):
            # FIXME; There is a bug in the reduction of the image depc graph
            # Reduce only the non-multi part of the graph
            nonmulti_graph = graph.copy()
            multi_data_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                                if d.get('ismulti')]
            multi_edges = [(u, v) for u, v, d in multi_data_edges]
            nonmulti_graph.remove_edges_from(multi_edges)

            # Hack to recognize that implicit edges on multi-input edges
            # PROBABLY means that the implicit edge is also a multi edge This
            # needs to be fixed by indicating which parent edge the implicit
            # edge actually corresponds to.
            multi_data_edges_ = []
            for edge in multi_data_edges:
                node = edge[1]
                in_edges = list(nonmulti_graph.in_edges(node, data=True))
                if len(in_edges) == 1:
                    u, v, d = in_edges[0]
                    # If there is only one implicit edge on a multi-edge, it very
                    # likely means that the implicit edge should have the same
                    # input-id as the multi-edge
                    if d.get('implicit'):
                        nonmulti_graph.remove_edge(u, v)
                    # hack the multi-edge to reduce to the implicit version
                    new_edge = (u, edge[1], edge[2])
                    multi_data_edges_.append(new_edge)
                else:
                    multi_data_edges_.append(edge)
            multi_data_edges = multi_data_edges_

            # <ATTEMPT>
            # The transitive reduction should respect impolicit edges,
            # but the end result should not contain any implicit edges

            # Transitive reduction wrt implicit edges
            # For every node...
            implicit_aware = 1

            if implicit_aware:
                removed_in_edges = {}
                for node in graph.nodes():
                    in_edges = list(graph.in_edges(node, data=True))
                    # if there is an implicit incoming edge
                    implicit_flags = [edge[2].get('implicit') for edge in in_edges]
                    explicit_flags = ut.not_list(implicit_flags)
                    implicit_edges = ut.compress(in_edges, implicit_flags)
                    flag = True
                    for edge in implicit_edges:
                        # Ignore this edge if there is a common descendant
                        if ut.nx_common_descendants(graph, node, edge[0]):
                            pass
                            #removed_in_edges[node] = implicit_edges
                            #flag = False

                    if flag and any(implicit_edges):
                        # then remove all non-implicit incoming edges
                        remove_non_implicit = ut.compress(in_edges, explicit_flags)
                        # remember this nodes removed in edges
                        removed_in_edges[node] = remove_non_implicit
                to_remove2 = ut.take_column(ut.flatten(removed_in_edges.values()), [0, 1])
                nonmulti_graph.remove_edges_from(to_remove2)

            graph_tr = ut.nx_transitive_reduction(nonmulti_graph)

            if implicit_aware:
                # Place old non-implicit structure on top of reduced implicit edges
                for node in removed_in_edges.keys():
                    old_edges = removed_in_edges[node]
                    for edge in graph_tr.in_edges(node, data=True):
                        data = edge[2]
                        for old_edge in old_edges:
                            # TODO: handle multiple old edges
                            data.update(old_edge[2])
                            data['implicit'] = False

            # HACK IN STRUCTURE
            # Multi Edges
            # (doesn't quite work)
            if False:
                for u, v, data in graph.edges(data=True):
                    if data.get('ismulti'):
                        new_parent = nx.shortest_path(graph_tr, u, v)[-2]
                        #graph_tr[new_parent][v][0]['is_multi'] = True
                        print("NEW MULTI")
                        print((new_parent, v))
                        nx.set_edge_attributes(graph_tr, 'ismulti',
                                               {(new_parent, v, 0): True})
                        #print(v)
            else:
                pass
                graph_tr.add_edges_from(multi_data_edges)

            parents = ut.ddict(list)
            for u, v, data in graph.edges(data=True):
                parents[v].append(u)

            for node, ps in parents.items():
                num_connect = ut.dict_hist(ps)
                nwise_parents = [(k, v) for k, v in num_connect.items() if v > 1]

                for p, n in nwise_parents:
                    new_parent = nx.shortest_path(graph_tr, p, node)[-2]
                    for x in range(n - 1):
                        #import utool
                        #utool.embed()
                        graph_tr.add_edge(new_parent, node)
                    #G_tr[new_parent][v][0]['is_multi'] = True
            nx.set_node_attributes(graph_tr, 'color', _node_attrs(color_dict))
            nx.set_node_attributes(graph_tr, 'shape', _node_attrs(shape_dict))
            graph = graph_tr

        if kwargs.get('remove_local_input_id', False):
            ut.nx_delete_edge_attr(graph, 'local_input_id')

        return graph

    @property
    def graph(depc):
        return depc.make_graph()

    @property
    def explicit_graph(depc):
        return depc.make_graph(implicit=False)

    @property
    def reduced_graph(depc):
        return depc.make_graph(reduced=True)

    def show_graph(depc, reduced=False, **kwargs):
        """ Helper "fluff" function """
        import plottool as pt
        graph = depc.make_graph(reduced=reduced)
        if ut.is_developer():
            ut.ensure_pylab_qt4()
        kwargs['layout'] = 'agraph'
        pt.show_nx(graph, **kwargs)

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
