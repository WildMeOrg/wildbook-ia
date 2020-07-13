# -*- coding: utf-8 -*-
"""
implicit version of dependency cache from wbia/templates/template_generator
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import six
from six.moves import zip
from wbia.dtool import sql_control
from wbia.dtool import depcache_table
from wbia.dtool import base
from collections import defaultdict

(print, rrr, profile) = ut.inject2(__name__)


# global function registry
PREPROC_REGISTER = defaultdict(list)
SUBPROP_REGISTER = defaultdict(list)


REG_PREPROC_DOC = """
Args:
    tablename (str): name of the node (corrsponds to SQL table)
    parents (list): tables this node depends on
    colnames (list): data returned by this table
    coltypes (list): types of data returned by this table
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

    _depcdecors = ut.odict({'preproc': register_preproc, 'subprop': register_subprop})
    return _depcdecors


class _CoreDependencyCache(object):
    """
    Core worker functions for the depcache
    Inherited by a calss with some "nice extras
    """

    @profile
    def _register_prop(
        depc,
        tablename,
        parents=None,
        colnames=None,
        coltypes=None,
        preproc_func=None,
        fname=None,
        configclass=None,
        requestclass=None,
        default_to_unpack=None,
        **kwargs,
    ):
        """
        Registers a table with this dependency cache.
        Essentially passes args down to make a DependencyTable.

        SEE: dtool.REG_PREPROC_DOC
        """
        if depc._debug:
            print('[depc] Registering tablename=%r' % (tablename,))
            print('[depc]  * preproc_func=%r' % (preproc_func,))
        # ----------
        # Sanitize inputs
        if isinstance(tablename, six.string_types):
            tablename = six.text_type(tablename)
        if parents is None:
            parents = [depc.root]
        if colnames is None:
            colnames = 'data'
            if coltypes is None:
                coltypes = np.ndarray
        # Check if just a single column is given
        if default_to_unpack is None:
            if ut.isiterable(colnames):
                default_to_unpack = False
                colnames = ut.lmap(six.text_type, colnames)
            else:
                colnames = [colnames]
                coltypes = [coltypes]
                default_to_unpack = True
        if coltypes is None:
            raise ValueError('must specify coltypes of %s' % (tablename,))
            coltypes = [np.ndarray] * len(colnames)
        if fname is None:
            fname = depc.default_fname
        if configclass is None:
            # Make a default config with no parameters
            configclass = {}
        if isinstance(configclass, dict):
            # Dynamically make config class
            default_cfgdict = configclass
            configclass = base.make_configclass(default_cfgdict, tablename)
        # ----------
        # Register a new table and configuration
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
            **kwargs,
        )
        depc.cachetable_dict[tablename] = table
        depc.configclass_dict[tablename] = configclass
        return table

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

    def _register_subprop(depc, tablename, propname=None, preproc_func=None):
        """ subproperties are always recomputeed on the fly """
        table = depc.cachetable_dict[tablename]
        table.subproperties[propname] = preproc_func

    def close(depc):
        """
        Close all managed SQL databases
        """
        for fname, db in depc.fname_to_db.items():
            db.close()

    @profile
    def initialize(depc, _debug=None):
        """
        Creates all registered tables
        """
        print(
            '[depc] Initialize %s depcache in %r' % (depc.root.upper(), depc.cache_dpath)
        )
        _debug = depc._debug if _debug is None else _debug
        if depc._use_globals:
            reg_preproc = PREPROC_REGISTER[depc.root]
            reg_subprop = SUBPROP_REGISTER[depc.root]
            if ut.VERBOSE:
                print(
                    '[depc.init] Registering %d global preproc funcs' % len(reg_preproc)
                )
            for args_, _kwargs in reg_preproc:
                depc._register_prop(*args_, **_kwargs)
            if ut.VERBOSE:
                print('[depc.init] Registering %d global subprops ' % len(reg_subprop))
            for args_, _kwargs in reg_subprop:
                depc._register_subprop(*args_, **_kwargs)

        ut.ensuredir(depc.cache_dpath)

        # Memory filestore
        # if False:
        #    # http://docs.pyfilesystem.org/en/latest/getting_started.html
        #    pip install fs

        for fname in depc.fname_to_db.keys():
            if fname == ':memory:':
                fpath = fname
            else:
                fname_ = ut.ensure_ext(fname, '.sqlite')
                from os.path import dirname

                prefix_dpath = dirname(fname_)
                if prefix_dpath:
                    ut.ensuredir(ut.unixjoin(depc.cache_dpath, prefix_dpath))
                fpath = ut.unixjoin(depc.cache_dpath, fname_)
            # if ut.get_argflag('--clear-all-depcache'):
            #     ut.delete(fpath)
            db = sql_control.SQLDatabaseController(
                fpath=fpath, always_check_metadata=False
            )
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
                get_prop = ut.partial(depc.get, table.tablename, colnames=colname)
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

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
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
            assert tablename in depc.cachetable_dict, 'tablename=%r does not exist' % (
                tablename,
            )
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
            dependency_levels = list(map(sorted, dependency_levels))
        except Exception as ex:
            ut.printex(
                ex,
                'error getting dependencies',
                keys=[
                    'tablename',
                    'root',
                    'children_to_parents',
                    'to_root',
                    'from_root',
                    'dependency_levels_',
                    'dependency_levels',
                ],
            )
            raise

        return dependency_levels

    def _ensure_config(depc, tablekey, config, _debug=False):
        """
        Creates a full table configuration with all defaults using config

        Args:
            tablekey (str): name of the table to grab config from
            config (dict): may be overspecified or underspecfied
        """
        configclass = depc.configclass_dict.get(tablekey, None)
        # requestclass = depc.requestclass_dict.get(tablekey, None)
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
                if _debug:
                    print(' **config = %r' % (config,))
                config_ = configclass(**config)
                if _debug:
                    print(' config_ = %r' % (config_,))
        return config_

    def get_config_trail(depc, tablename, config):
        graph = depc.make_graph(implicit=True)
        tablename_list = ut.nx_all_nodes_between(graph, depc.root, tablename)
        tablename_list = ut.nx_topsort_nodes(graph, tablename_list)
        config_trail = []
        for tablekey in tablename_list:
            if tablekey in depc.configclass_dict:
                config_ = depc._ensure_config(tablekey, config)
                config_trail.append(config_)
        return config_trail

    def get_config_trail_str(depc, tablename, config):
        config_trail = depc.get_config_trail(tablename, config)
        trail_cfgstr = '_'.join([x.get_cfgstr() for x in config_trail])
        return trail_cfgstr

    def _get_parent_input(
        depc,
        tablename,
        root_rowids,
        config,
        ensure=True,
        _debug=None,
        recompute=False,
        recompute_all=False,
        eager=True,
        nInput=None,
    ):
        # Get ancestor rowids that are descendants of root
        table = depc[tablename]
        rowid_dict = depc.get_all_descendant_rowids(
            tablename,
            root_rowids,
            config=config,
            ensure=ensure,
            eager=eager,
            nInput=nInput,
            recompute=recompute,
            recompute_all=recompute_all,
            _debug=ut.countdown_flag(_debug),
            levels_up=1,
        )
        parent_rowids = depc._get_parent_rowids(table, rowid_dict)
        return parent_rowids

    # -----------------------------
    # STATE GETTERS

    def rectify_input_tuple(depc, exi_inputs, input_tuple):
        """
        Standardizes inputs allowed for convinience into the expected input for
        get_parent_rowids.
        """
        input_tuple_ = input_tuple

        if isinstance(input_tuple_, (list, np.ndarray)):
            if len(exi_inputs) != 1:
                msg = '#expected=%d, #got=1' % (len(exi_inputs),)
                msg += '. Did you forget to cast multi-inputs to a tuple?'
                raise ValueError(msg)
            # Inputs should always be a tuple of lists.
            input_tuple_ = (input_tuple_,)
        if len(exi_inputs) == 1:
            # HACK: for simple case where we only need one parent
            if isinstance(input_tuple_, (tuple,)):
                if len(input_tuple_) == 0:
                    input_tuple_ = []
                elif len(input_tuple_) > 1:
                    if not ut.isiterable(input_tuple_[0]):
                        input_tuple_ = (input_tuple_,)
        if len(exi_inputs) != len(input_tuple_):
            msg = '#expected=%d, #got=%d' % (len(exi_inputs), len(input_tuple_))
            raise ValueError(msg)

        # rectify input depth
        rectified_input = []
        for x, d in zip(input_tuple_, exi_inputs.expected_input_depth()):
            if d == 0:
                if not ut.isiterable(x):
                    rectified_input.append([x])
                else:
                    rectified_input.append(x)
            else:
                if ut.list_depth(x) == 0:
                    rectified_input.append([x])
                else:
                    rectified_input.append(x)
        return rectified_input

    def get_parent_rowids(depc, target_tablename, input_tuple, config=None, **kwargs):
        """
        Returns the parent rowids needed to get / compute a property of
        tablename

        Args:
            input_tuple :
                to be explicit send in as a tuple of lists.  Each list
                corresponds to parent information needed by expanded rmis (root
                most input).

                Each item in the tuple correponds a root most node, and should
                be specified as a list of inputs. For single items this is a
                scalar, for multi-items it is a list.

                For example if you have a property like a chip that depends on
                only one parent, then to get the chips for the first N
                annotations your list input tuple is:
                    input_tuple = ([1, 2, 3, ..., N],)

                For a single multi inputs:
                If you want to get two vocabs for even and odd annots then you
                have:
                    ([[0, 2, 4, ...], [1, 3, 5, ...]],)

                For a single comparasion version multi inputs:
                If you want to query the first N annotats against two
                vocabs then you have:
                    ([1, 2, 3, ..., N], [[0, 2, 4, ...], [1, 3, 5, ...]],)
                (Note this only works if broadcasting is on)
        """
        _kwargs = kwargs.copy()
        _recompute = _kwargs.pop('recompute_all', False)
        _debug = _kwargs.get('_debug', False)
        _hack_rootmost = _kwargs.pop('_hack_rootmost', False)
        _debug = depc._debug if _debug is None else _debug
        if config is None:
            config = {}

        with ut.Indenter('[GetParentID-%s]' % (target_tablename,), enabled=_debug):
            if _debug:
                print(ut.color_text('Enter get_parent_rowids', 'blue'))
                print(' * target_tablename = %r' % (target_tablename,))
                print(' * input_tuple=%s' % (ut.trunc_repr(input_tuple),))
                print(' * config = %r' % (config,))
            target_table = depc[target_tablename]

            # TODO: Expand to the appropriate given inputs
            if _hack_rootmost:
                # Hack: if true, we are given inputs in rootmost form
                exi_inputs = target_table.rootmost_inputs
            else:
                # otherwise we are given inputs in totalroot form
                exi_inputs = target_table.rootmost_inputs.total_expand()
            if _debug:
                print(' * exi_inputs=%s' % (exi_inputs,))

            rectified_input = depc.rectify_input_tuple(exi_inputs, input_tuple)

            rowid_dict = {}
            for rmi, rowids in zip(exi_inputs.rmi_list, rectified_input):
                rowid_dict[rmi] = rowids

            compute_edges = exi_inputs.flat_compute_rmi_edges()
            if _debug:
                print(' * rectified_input=%s' % ut.trunc_repr(rectified_input))
                print(' * compute_edges=%s' % ut.repr2(compute_edges, nl=2))

            for count, (input_nodes, output_node) in enumerate(compute_edges, start=1):
                if _debug:
                    ut.cprint(
                        ' * COMPUTING %d/%d EDGE %r -- %r'
                        % (count, len(compute_edges), input_nodes, output_node),
                        'blue',
                    )
                tablekey = output_node.tablename
                table = depc[tablekey]
                input_nodes_ = input_nodes
                if _debug:
                    print(
                        'table.parent_id_tablenames = %r' % (table.parent_id_tablenames,)
                    )
                    print('input_nodes_ = %r' % (input_nodes_,))
                input_multi_flags = [
                    node.ismulti and node in exi_inputs.rmi_list for node in input_nodes_
                ]

                # Args currently go in like this:
                # args  = [..., (pid_{i,1}, pid_{i,2}, ..., pid_{i,M}), ...]
                # They get converted into
                # argsT = [... (pid_{1,j}, ... pid_{N,j}) ...]
                # i = row, j = col
                sig_multi_flags = table.get_parent_col_attr('ismulti')
                parent_rowidsT = ut.take(rowid_dict, input_nodes_)
                parent_rowids_ = []
                # TODO: will need to figure out which columns to zip and which
                # columns to product (ie take product over ones that have 1
                # item, and zip ones that have equal amount of items)
                for flag1, flag2, rowidsT in zip(
                    sig_multi_flags, input_multi_flags, parent_rowidsT
                ):
                    if flag1 and flag2:
                        parent_rowids_.append(rowidsT)
                    elif flag1 and not flag2:
                        parent_rowids_.append([rowidsT])
                    elif not flag1 and flag2:
                        assert len(rowidsT) == 1
                        parent_rowids_.append(rowidsT[0])
                    else:
                        parent_rowids_.append(rowidsT)
                # Assume that we are either given corresponding lists or single values
                # that must be broadcast.
                rowlens = list(map(len, parent_rowids_))
                maxlen = max(rowlens)
                parent_rowids2_ = [
                    r * maxlen if len(r) == 1 else r for r in parent_rowids_
                ]
                _parent_rowids = list(zip(*parent_rowids2_))
                # _parent_rowids = list(ut.product(*parent_rowids_))

                if _debug:
                    print(
                        'parent_rowids_ = %s'
                        % (
                            ut.repr4(
                                [ut.trunc_repr(ids_) for ids_ in parent_rowids_],
                                strvals=True,
                            )
                        )
                    )
                    print(
                        'parent_rowids2_ = %s'
                        % (
                            ut.repr4(
                                [ut.trunc_repr(ids_) for ids_ in parent_rowids2_],
                                strvals=True,
                            )
                        )
                    )
                    print(
                        '_parent_rowids = %s'
                        % (
                            ut.truncate_str(
                                ut.repr4(
                                    [ut.trunc_repr(ids_) for ids_ in _parent_rowids],
                                    strvals=True,
                                )
                            )
                        )
                    )

                if _debug:
                    ut.cprint('-------------', 'blue')
                if output_node.tablename != target_tablename:
                    # Get table configuration
                    config_ = depc._ensure_config(tablekey, config, _debug)

                    output_rowids = table.get_rowid(
                        _parent_rowids, config=config_, recompute=_recompute, **_kwargs
                    )
                    rowid_dict[output_node] = output_rowids
                    # table.get_model_inputs(table.get_model_uuid(output_rowids)[0])
                else:
                    # We are only computing up to the parents of the table here.
                    parent_rowids = _parent_rowids
                    break
            # rowids = rowid_dict[output_node]
            return parent_rowids

    def check_rowids(depc, tablename, input_tuple, config={}):
        """
        Returns a list of flags where True means the row has been computed and
        False means that it needs to be computed.
        """
        existing_rowids = depc.get_rowids(
            tablename, input_tuple, config=config, ensure=False
        )
        flags = ut.flag_not_None_items(existing_rowids)
        return flags

    def get_rowids(depc, tablename, input_tuple, **rowid_kw):
        """
        Used to get tablename rowids. Ensures rows exist unless ensure=False.
        rowids uniquely specify parent inputs and a configuration.

        CommandLine:
            python -m dtool.depcache_control get_rowids --show
            python -m dtool.depcache_control get_rowids:1

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc3(True)
            >>> exec(ut.execstr_funckw(depc.get), globals())
            >>> kwargs = {}
            >>> root_rowids = [1, 2, 3]
            >>> root_rowids2 = [(4, 5, 6, 7)]
            >>> root_rowids3 = root_rowids2
            >>> _debug = True
            >>> tablename = 'smk_match'
            >>> input_tuple = (root_rowids, root_rowids2, root_rowids3)
            >>> target_table = depc[tablename]
            >>> inputs = target_table.rootmost_inputs.total_expand()
            >>> depc.get_rowids(tablename, input_tuple, _debug=_debug)
            >>> depc.print_all_tables()

        Example:
            >>> # ENABLE_DOCTEST
            >>> # Test external / ensure getters
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> config = {}
            >>> depc = testdata_depc()
            >>> aids = [1,]
            >>> depc.delete_property('keypoint', aids, config=config)
            >>> chip_fpaths = depc.get('chip', aids, 'chip', config=config, read_extern=False)
            >>> ut.remove_file_list(chip_fpaths)
            >>> rowids = depc.get_rowids('keypoint', aids, ensure=True, config=config)
            >>> print('rowids = %r' % (rowids,))

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc()
            >>> depc.clear_all()
            >>> root_rowids = [1, 2]
            >>> config = {}
            >>> # Recompute the first few, make sure the rowids do not change
            >>> _ = depc.get_rowids('chip', root_rowids + [3], config=config)
            >>> assert _ == [1, 2, 3]
            >>> initial_rowids = depc.get_rowids('chip', root_rowids, config=config)
            >>> recomp_rowids = depc.get_rowids('chip', root_rowids, config=config, recompute=True)
            >>> assert recomp_rowids == initial_rowids, 'rowids should not change due to recompute'
        """
        target_tablename = tablename
        _debug = rowid_kw.get('_debug', False)
        _debug = depc._debug if _debug is None else _debug
        _kwargs = rowid_kw.copy()
        config = _kwargs.pop('config', {})
        _hack_rootmost = _kwargs.pop('_hack_rootmost', False)
        _recompute_all = _kwargs.pop('recompute_all', False)
        recompute = _kwargs.pop('recompute', _recompute_all)
        table = depc[target_tablename]

        parent_rowids = depc.get_parent_rowids(
            target_tablename,
            input_tuple,
            config=config,
            _hack_rootmost=_hack_rootmost,
            **_kwargs,
        )

        with ut.Indenter('[GetRowId-%s]' % (target_tablename,), enabled=_debug):
            config_ = depc._ensure_config(target_tablename, config, _debug)
            rowids = table.get_rowid(
                parent_rowids, config=config_, recompute=recompute, **_kwargs
            )
        return rowids

    @ut.accepts_scalar_input2(argx_list=[1])
    def get(
        depc,
        tablename,
        root_rowids,
        colnames=None,
        config=None,
        ensure=True,
        _debug=None,
        recompute=False,
        recompute_all=False,
        eager=True,
        nInput=None,
        read_extern=True,
        onthefly=False,
        num_retries=1,
        hack_paths=False,
    ):
        r"""
        Access dependant properties the primary objects using primary ids.

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
            python -m dtool.depcache_control --exec-get

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc3(True)
            >>> exec(ut.execstr_funckw(depc.get), globals())
            >>> aids = [1, 2, 3]
            >>> _debug = True
            >>> tablename = 'labeler'
            >>> root_rowids = aids
            >>> prop_list = depc.get(
            >>>     tablename, root_rowids, colnames)
            >>> result = ('prop_list = %s' % (ut.repr2(prop_list),))
            >>> print(result)
            prop_list = [('labeler([root(1)]:42)',), ('labeler([root(2)]:42)',), ('labeler([root(3)]:42)',)]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc3(True)
            >>> exec(ut.execstr_funckw(depc.get), globals())
            >>> aids = [1, 2, 3]
            >>> _debug = True
            >>> tablename = 'smk_match'
            >>> tablename = 'vocab'
            >>> table = depc[tablename]
            >>> root_rowids = [aids]
            >>> prop_list = depc.get(
            >>>     tablename, root_rowids, colnames, config)
            >>> result = ('prop_list = %s' % (ut.repr2(prop_list),))
            >>> print(result)
            prop_list = [('vocab([root(1;2;3)]:42)',)]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc3(True)
            >>> exec(ut.execstr_funckw(depc.get), globals())
            >>> aids = [1, 2, 3]
            >>> _debug = True
            >>> depc = testdata_depc()
            >>> tablename = 'chip'
            >>> table = depc[tablename]
            >>> root_rowids = aids
            >>> # Ensure chips are computed
            >>> prop_list1 = depc.get(tablename, root_rowids)
            >>> # Get file paths and delete them
            >>> prop_list2 = depc.get(tablename, root_rowids, read_extern=False)
            >>> n = ut.remove_file_list(ut.take_column(prop_list2, 1))
            >>> assert n == len(prop_list2), 'files were not computed'
            >>> prop_list3 = depc.get(tablename, root_rowids)
            >>> assert np.all(prop_list1[0][1] == prop_list3[0][1]), 'computed same info'
        """
        if tablename == depc.root_tablename:
            return depc.root_getters[colnames](root_rowids)
            # pass
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

                # recompute_ = recompute or recompute_all
                parent_rowids = depc.get_parent_rowids(
                    tablename,
                    root_rowids,
                    config=config,
                    ensure=True,
                    _debug=None,
                    recompute_all=False,
                    eager=True,
                    nInput=None,
                )
                config_ = depc._ensure_config(tablename, config)
                if _debug:
                    print(' * (ensured) config_ = %r' % (config_,))
                table = depc[tablename]
                extern_dpath = table.extern_dpath
                ut.ensuredir(extern_dpath, verbose=False or table.depc._debug)
                fname_list = table.get_extern_fnames(
                    parent_rowids, config=config_, extern_col_index=0
                )
                fpath_list = [join(extern_dpath, fname) for fname in fname_list]
                return fpath_list

            if nInput is None and ut.is_listlike(root_rowids):
                nInput = len(root_rowids)

            rowid_kw = dict(
                config=config,
                nInput=nInput,
                eager=eager,
                ensure=ensure,
                recompute=recompute,
                recompute_all=recompute_all,
                _debug=_debug,
            )

            rowdata_kw = dict(
                read_extern=read_extern,
                _debug=_debug,
                num_retries=num_retries,
                eager=eager,
                ensure=ensure,
                nInput=nInput,
            )

            input_tuple = root_rowids

            for trynum in range(num_retries + 1):
                try:
                    table = depc[tablename]
                    # Vectorized get of properties
                    tbl_rowids = depc.get_rowids(tablename, input_tuple, **rowid_kw)
                    if _debug:
                        print('[depc.get] tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
                    prop_list = table.get_row_data(tbl_rowids, colnames, **rowdata_kw)
                except depcache_table.ExternalStorageException:
                    print('!!* Hit ExternalStorageException')
                    if trynum == num_retries:
                        raise
                else:
                    break
            if _debug:
                print('* return prop_list=%s' % (ut.trunc_repr(prop_list),))
        return prop_list

    def get_native(
        depc, tablename, tbl_rowids, colnames=None, _debug=None, read_extern=True
    ):
        """
        Gets data using internal ids, which is faster if you have them.

        CommandLine:
            python -m dtool.depcache_control get_native:0
            python -m dtool.depcache_control get_native:1

        Example:
            >>> # ENABLE_DOCTEST
            >>> # Simple test of get native
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> config = {}
            >>> depc = testdata_depc()
            >>> tablename = 'keypoint'
            >>> aids = [1,]
            >>> tbl_rowids = depc.get_rowids(tablename, aids, config=config)
            >>> data = depc.get_native(tablename, tbl_rowids)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc()
            >>> config = {}
            >>> tablename = 'chip'
            >>> colnames = extern_colname = 'chip'
            >>> aids = [1, 2]
            >>> depc.delete_property(tablename, aids, config=config)
            >>> # Ensure chip rowids exist then delete external data without
            >>> # notifying the depcache. This forces the depcache to recover
            >>> tbl_rowids = chip_rowids = depc.get_rowids(tablename, aids, config=config)
            >>> data_fpaths = depc.get(tablename, aids, extern_colname, config=config, read_extern=False)
            >>> ut.remove_file_list(data_fpaths)
            >>> chips = depc.get_native(tablename, tbl_rowids, extern_colname)
            >>> print('chips = %r' % (chips,))
        """
        tbl_rowids = list(tbl_rowids)
        _debug = depc._debug if _debug is None else _debug
        with ut.Indenter('[GetNative %s]' % (tablename,), enabled=_debug):
            if _debug:
                print(' * tablename = %r' % (tablename,))
                print(' * colnames = %r' % (colnames,))
                print(' * tbl_rowids=%s' % (ut.trunc_repr(tbl_rowids)))
            table = depc[tablename]
            # import utool
            # with utool.embed_on_exception_context:
            # try:
            prop_list = table.get_row_data(
                tbl_rowids, colnames, _debug=_debug, read_extern=read_extern
            )
            # except depcache_table.ExternalStorageException:
            #    # This code is a bit rendant and would probably live better elsewhere
            #    # Also need to fix issues if more than one column specified
            #    extern_uris = table.get_row_data(
            #        tbl_rowids, colnames, _debug=_debug, read_extern=False,
            #        delete_on_fail=True, ensure=False)
            #    from os.path import exists
            #    error_flags = [exists(uri) for uri in extern_uris]
            #    redo_rowids = ut.compress(tbl_rowids, ut.not_list(error_flags))
            #    parent_rowids = table.get_parent_rowids(redo_rowids)
            #    # config_rowids = table.get_row_cfgid(redo_rowids)
            #    configs = table.get_row_configs(redo_rowids)
            #    assert ut.allsame(list(map(id, configs))), 'more than one config not yet supported'
            #    config = configs[0]
            #    table.get_rowid(parent_rowids, recompute=True, config=config)

            #    # TRY ONE MORE TIME
            #    prop_list = table.get_row_data(tbl_rowids, colnames, _debug=_debug,
            #                                   read_extern=read_extern,
            #                                   delete_on_fail=False)
        return prop_list

    def get_config_history(depc, tablename, root_rowids, config=None):
        # Vectorized get of properties
        tbl_rowids = depc.get_rowids(tablename, root_rowids, config=config)
        return depc[tablename].get_config_history(tbl_rowids)

    def get_root_rowids(depc, tablename, native_rowids):
        r"""
        Args:
            tablename (str):
            native_rowids (list):

        Returns:
            list:

        CommandLine:
            python -m dtool.depcache_control get_root_rowids --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache import *  # NOQA
            >>> depc = testdata_depc()
            >>> config1 = {'adapt_shape': False}
            >>> config2 = {'adapt_shape': True}
            >>> root_rowids = [2, 3, 5, 7]
            >>> native_rowids1 = depc.get_rowids('keypoint', root_rowids, config=config1)
            >>> native_rowids2 = depc.get_rowids('keypoint', root_rowids, config=config2)
            >>> ancestor_rowids1 = list(depc.get_root_rowids('keypoint', native_rowids1))
            >>> ancestor_rowids2 = list(depc.get_root_rowids('keypoint', native_rowids2))
            >>> assert native_rowids1 != native_rowids2, 'should have different native rowids'
            >>> assert ancestor_rowids1 == root_rowids, 'should have same root'
            >>> assert ancestor_rowids2 == root_rowids, 'should have same root'
        """
        return depc.get_ancestor_rowids(tablename, native_rowids, depc.root)

    def get_ancestor_rowids(depc, tablename, native_rowids, ancestor_tablename=None):
        """
        ancestor_tablename = depc.root; native_rowids = cid_list; tablename = const.CHIP_TABLE
        """
        if ancestor_tablename is None:
            ancestor_tablename = depc.root
        table = depc[tablename]
        ancestor_rowids = table.get_ancestor_rowids(native_rowids, ancestor_tablename)
        return ancestor_rowids

    def new_request(depc, tablename, qaids, daids, cfgdict=None):
        """ creates a request for data that can be executed later """
        print('[depc] NEW %s request' % (tablename,))
        requestclass = depc.requestclass_dict[tablename]
        request = requestclass.new(depc, qaids, daids, cfgdict, tablename=tablename)
        return request

    # -----------------------------
    # STATE MODIFIERS

    def delete_property(depc, tablename, root_rowids, config=None, _debug=False):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids`
        using `config`.

        FIXME: make this work for all configs
        """
        rowid_list = depc.get_rowids(
            tablename, root_rowids, config=config, ensure=False, _debug=_debug
        )
        table = depc[tablename]
        num_deleted = table.delete_rows(rowid_list)
        return num_deleted

    def delete_property_all(depc, tablename, root_rowids, _debug=False):
        """
        Deletes the rowids of `tablename` that correspond to `root_rowids`
        using `config`.

        FIXME: make this work for all configs
        """
        table = depc[tablename]
        all_rowid_list = table._get_all_rowids()
        if len(all_rowid_list) == 0:
            return 0

        ancestor_rowid_list = depc.get_ancestor_rowids(tablename, all_rowid_list)

        rowid_list = []
        root_rowids_set = set(root_rowids)
        for rowid, ancestor_rowid in zip(all_rowid_list, ancestor_rowid_list):
            if ancestor_rowid in root_rowids_set:
                rowid_list.append(rowid)

        num_deleted = table.delete_rows(rowid_list)
        return num_deleted


@six.add_metaclass(ut.ReloadingMetaclass)
class DependencyCache(_CoreDependencyCache, ut.NiceRepr):
    """
    Currently, to use this class a user must:
        * on root modification, call depc.on_root_modified
        * use decorators to register relevant functions
    """

    def __init__(
        depc,
        root_tablename=None,
        cache_dpath='./DEPCACHE',
        controller=None,
        default_fname=None,
        # root_asobject=None,
        get_root_uuid=None,
        root_getters=None,
        use_globals=True,
    ):
        if default_fname is None:
            default_fname = root_tablename + '_primary_cache'
            # default_fname = ':memory:'
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
        # depc._root_asobject = root_asobject
        depc._use_globals = use_globals
        depc.default_fname = default_fname
        if get_root_uuid is None:
            print('WARNING NEED UUID FUNCTION')
            # HACK
            get_root_uuid = ut.identity
        depc.get_root_uuid = get_root_uuid
        depc.delete_exclude_tables = {}
        depc._debug = ut.get_argflag(('--debug-depcache', '--debug-depc'))

    def get_tablenames(depc):
        return list(depc.cachetable_dict.keys())

    @property
    def tables(depc):
        return list(depc.cachetable_dict.values())

    @property
    def tablenames(depc):
        return depc.get_tablenames()

    def print_schemas(depc):
        for fname, db in depc.fname_to_db.items():
            print('fname = %r' % (fname,))
            db.print_schema()

    # def print_table_csv(depc, tablename):
    #    depc[tablename]

    def print_table(depc, tablename):
        depc[tablename].print_table()

    def print_all_tables(depc):
        for tablename, table in depc.cachetable_dict.items():
            table.print_table()
            # db = table.db
            # db.print_table_csv(tablename)

    def print_config_tables(depc):
        for fname in depc.fname_to_db:
            print('---')
            print('db_fname = %r' % (fname,))
            depc.fname_to_db[fname].print_table_csv('config')

    def get_edges(depc, data=False):
        """
        edges for networkx structure
        """
        if data:

            def get_edgedata(tablekey, parentkey, parent_data):
                if parent_data['ismulti'] or parent_data['isnwise']:
                    edge_type_parts = []
                    # local_input_id = ''
                    if parent_data['ismulti']:
                        # TODO: give different ids to multi edges
                        # edge_type_parts.append('multi_%s_%s' % (parentkey, tablekey))
                        edge_type_parts.append('multi')
                        # local_input_id += '*'
                    if parent_data['isnwise']:
                        args = (parent_data['nwise_idx'],)
                        edge_type_parts.append('nwise_%s' % args)
                        # local_input_id += six.text_type(parent_data['nwise_idx'])
                        # edge_type_parts.append('nwise_%s_%s_%s' % (
                        #     parentkey, tablekey, parent_data['nwise_idx'],))
                    edge_type_id = '_'.join(edge_type_parts)
                else:
                    edge_type_id = 'normal'
                    # local_input_id = '1'
                edge_data = {
                    'ismulti': parent_data['ismulti'],
                    'isnwise': parent_data.get('isnwise'),
                    'nwise_idx': parent_data.get('nwise_idx'),
                    'parent_colx': parent_data.get('parent_colx'),
                    'edge_type': edge_type_id,
                    # 'local_input_id': local_input_id,
                    'local_input_id': parent_data.get('local_input_id'),
                    'taillabel': parent_data.get('local_input_id'),
                    # 'taillabel': local_input_id,  # proper graphviz attribute
                }
                return edge_data

            edges = [
                (parentkey, tablekey, get_edgedata(tablekey, parentkey, parent_data))
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
            implicit_edges = [(e1, e2, {'implicit': True}) for e1, e2 in implicit_edges]
        return implicit_edges

    @ut.memoize
    def make_graph(depc, **kwargs):
        """
        Constructs a networkx representation of the dependency graph

        CommandLine:
            python -m dtool --tf DependencyCache.make_graph --show --reduced

            python -m wbia.control.IBEISControl show_depc_annot_graph --show --reduced

            python -m wbia.control.IBEISControl show_depc_annot_graph --show --reduced --testmode
            python -m wbia.control.IBEISControl show_depc_annot_graph --show --testmode

            python -m wbia.control.IBEISControl --test-show_depc_image_graph --show --reduced
            python -m wbia.control.IBEISControl --test-show_depc_image_graph --show

            python -m wbia.scripts.specialdraw double_depcache_graph --show --testmode

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> import utool as ut
            >>> depc = testdata_depc()
            >>> graph = depc.make_graph(reduced=ut.get_argflag('--reduced'))
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.ensureqt()
            >>> import networkx as nx
            >>> #pt.show_nx(nx.dag.transitive_closure(graph))
            >>> #pt.show_nx(ut.nx_transitive_reduction(graph))
            >>> pt.show_nx(graph)
            >>> pt.show_nx(graph, layout='agraph')
            >>> ut.show_if_requested()

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> import utool as ut
            >>> depc = testdata_depc()
            >>> graph = depc.make_graph(reduced=True)
            >>> # xdoctest: +REQUIRES(--show)
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.ensureqt()
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
            # 'node': 'circle',
            # 'node': 'rect',
            'node': 'ellipse',
            # 'root': 'rhombus',
            # 'root': 'circle',
            # 'root': 'circle',
            'root': 'ellipse',
            # 'root': 'rect',
        }
        # import wbia.plottool as pt
        NEUTRAL_BLUE = np.array((159, 159, 241, 255)) / 255.0
        RED = np.array((255, 0, 0, 255)) / 255.0
        color_dict = {
            # 'algo': pt.DARK_GREEN,  # 'g',
            'node': NEUTRAL_BLUE,
            'root': RED,  # 'r',
        }

        def _node_attrs(dict_):
            props = {k: dict_['node'] for k, v in depc.cachetable_dict.items()}
            props[depc.root] = dict_['root']
            return props

        nx.set_node_attributes(graph, name='color', values=_node_attrs(color_dict))
        nx.set_node_attributes(graph, name='shape', values=_node_attrs(shape_dict))
        if kwargs.get('reduced', False):
            # FIXME; There is a bug in the reduction of the image depc graph
            # Reduce only the non-multi part of the graph
            nonmulti_graph = graph.copy()
            multi_data_edges = [
                (u, v, d) for u, v, d in graph.edges(data=True) if d.get('ismulti')
            ]
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
                            # removed_in_edges[node] = implicit_edges
                            # flag = False

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
                        # graph_tr[new_parent][v][0]['is_multi'] = True
                        print('NEW MULTI')
                        print((new_parent, v))
                        nx.set_edge_attributes(
                            graph_tr, name='ismulti', values={(new_parent, v, 0): True}
                        )
                        # print(v)
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
                        # import utool
                        # utool.embed()
                        graph_tr.add_edge(new_parent, node)
                    # G_tr[new_parent][v][0]['is_multi'] = True
            nx.set_node_attributes(graph_tr, name='color', values=_node_attrs(color_dict))
            nx.set_node_attributes(graph_tr, name='shape', values=_node_attrs(shape_dict))
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
        import wbia.plottool as pt

        graph = depc.make_graph(reduced=reduced)
        if ut.is_developer():
            ut.ensureqt()
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

    def delete_root(
        depc,
        root_rowids,
        delete_extern=None,
        _debug=False,
        table_config_filter=None,
        prop=None,
    ):
        r"""
        Deletes all properties of a root object regardless of config

        Args:
            root_rowids (list):

        CommandLine:
            python -m dtool.depcache_control delete_root --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> exec(ut.execstr_funckw(depc.delete_root), globals())
            >>> root_rowids = [1]
            >>> depc.delete_root(root_rowids, _debug=0)
            >>> depc.get('fgweight', [1])
            >>> depc.delete_root(root_rowids, _debug=0)
        """
        # graph = depc.make_graph(implicit=False)
        # hack
        # check to make sure child does not have another parent
        rowid_dict = depc.get_allconfig_descendant_rowids(
            root_rowids, table_config_filter
        )
        # children = [child for child in graph.succ[depc.root_tablename]
        #            if sum([len(e) for e in graph.pred[child].values()]) == 1]
        # depc.delete_property(tablename, root_rowids, _debug=_debug)
        num_deleted = 0
        for tablename, table_rowids in rowid_dict.items():
            if tablename == depc.root:
                continue
            # Specific prop exclusion
            delete_exclude_table_set_prop = depc.delete_exclude_tables.get(prop, [])
            delete_exclude_table_set_all = depc.delete_exclude_tables.get(None, [])
            if (
                tablename in delete_exclude_table_set_prop
                or tablename in delete_exclude_table_set_all
            ):
                continue
            table = depc[tablename]
            num_deleted += table.delete_rows(table_rowids, delete_extern=delete_extern)
        return num_deleted

    def register_delete_table_exclusion(depc, tablename, prop):
        if prop not in depc.delete_exclude_tables:
            depc.delete_exclude_tables[prop] = set([])
        depc.delete_exclude_tables[prop].add(tablename)
        args = (ut.repr3(depc.delete_exclude_tables),)
        print('[depc] Updated delete tables: %s' % args)

    def get_allconfig_descendant_rowids(depc, root_rowids, table_config_filter=None):
        import networkx as nx

        # list(nx.topological_sort(nx.bfs_tree(graph, depc.root)))
        # decendants = nx.descendants(graph, depc.root)
        # raise NotImplementedError()

        graph = depc.make_graph(implicit=True)
        root = depc.root
        rowid_dict = {}
        rowid_dict[root] = root_rowids

        # Find all rowids that inherit from the specific root rowids
        sinks = list(ut.nx_sink_nodes(nx.bfs_tree(graph, depc.root)))
        for target_tablename in sinks:
            path = nx.shortest_path(graph, root, target_tablename)
            for parent, child in ut.itertwo(path):
                child_table = depc[child]
                relevant_col_attrs = [
                    attrs
                    for attrs in child_table.parent_col_attrs
                    if attrs['parent_table'] == parent
                ]
                parent_colnames = [
                    attrs['intern_colname'] for attrs in relevant_col_attrs
                ]

                params_iter = list(zip(rowid_dict[parent]))

                for parent_colname in parent_colnames:
                    child_rowids = child_table.db.get_where_eq_set(
                        child_table.tablename,
                        (child_table.rowid_colname,),
                        params_iter,
                        unpack_scalars=False,
                        where_colnames=[parent_colname],
                        op='AND',
                    )
                    # child_rowids = ut.flatten(child_rowids)
                    if table_config_filter is not None:
                        config_filter = table_config_filter.get(
                            child_table.tablename, None
                        )
                        if config_filter is not None:
                            # If a config filter is specified only grab rows that
                            # meed the filter
                            tbl_cfgids = child_table.get_row_cfgid(child_rowids)
                            cfgid2_rowids = ut.group_items(child_rowids, tbl_cfgids)
                            unique_cfgids = cfgid2_rowids.keys()
                            unique_cfgids = ut.filter_Nones(unique_cfgids)
                            unique_configs = child_table.get_config_from_rowid(
                                unique_cfgids
                            )
                            passed_rowids = []
                            for config, cfgid in zip(unique_configs, tbl_cfgids):
                                if all(
                                    [
                                        config[key] == val
                                        for key, val in config_filter.items()
                                    ]
                                ):
                                    passed_rowids.extend(cfgid2_rowids[cfgid])
                            child_rowids = passed_rowids
                    rowid_dict[child] = ut.unique(
                        child_rowids + rowid_dict.get(child, [])
                    )
        return rowid_dict

    def notify_root_changed(depc, root_rowids, prop, force_delete=False):
        """
        this is where we are notified that a "registered" root property has
        changed.
        """
        print(
            '[depc] notified that columns (%s) for (%d) row(s) were modified'
            % (prop, len(root_rowids),)
        )
        # for key in tables_depending_on(prop)
        # depc.delete_property(key, root_rowids)
        # TODO: check which properties were invalidated by this prop
        # TODO; remove invalidated properties
        if force_delete:
            depc.delete_root(root_rowids, prop=prop)

    def clear_all(depc):
        print('Clearning all cached data in %r' % (depc,))
        for table in depc.cachetable_dict.values():
            table.clear_table()

    def make_root_info_uuid(depc, root_rowids, info_props):
        """
        Creates a uuid that depends on certain properties of the root object.
        This is used for implicit cache invalidation because, if those
        properties change then this uuid also changes.

        The depcache needs to know about stateful properties of dynamic root
        objects in order to correctly compute their hashes.

        >>> #ibs = wbia.opendb(defaultdb='testdb1')
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

    get_native_property = _CoreDependencyCache.get_native
    get_property = _CoreDependencyCache.get

    def stacked_config(depc, source, dest, config):
        r"""
        CommandLine:
            python -m dtool.depcache_control stacked_config --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.depcache_control import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> source = depc.root
            >>> dest = 'fgweight'
            >>> config = {}
            >>> stacked_config = depc.stacked_config(source, dest, config)
            >>> cfgstr = stacked_config.get_cfgstr()
            >>> result = ('cfgstr = %s' % (ut.repr2(cfgstr),))
            >>> print(result)
        """
        if config is None:
            config = {}
        if source is None:
            source = depc.root
        graph = depc.make_graph(implicit=True)
        requires_tables = ut.setdiff(
            ut.nx_all_nodes_between(graph, source, dest), [source]
        )
        # requires_tables = ut.setdiff(ut.nx_all_nodes_between(depc.graph, 'annotations', 'featweight'), ['annotations'])
        requires_tables = ut.nx_topsort_nodes(depc.graph, requires_tables)
        requires_configs = [
            depc.configclass_dict[tblname](**config) for tblname in requires_tables
        ]
        # cfgstr_list = [cfg.get_cfgstr() for cfg in requires_configs]
        stacked_config = base.StackedConfig(requires_configs)
        return stacked_config
        # cfgstr = stacked_config.get_cfgstr()
        # cfgstr = '_'.join(cfgstr_list)
        # return cfgstr


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
