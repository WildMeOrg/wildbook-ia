from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
from six.moves import zip
from dtool import depcache_table


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
        >>> # DISABLE_DOCTEST
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
                keys = ['tablename', 'tablekey', 'rowid_dict', 'config',
                        'table', 'dependency_levels']
                ut.printex(ex, 'error expanding rowids', keys=keys)
                raise
            rowid_dict[tablekey] = child_rowids
    if _debug:
        print(' GOT DESCENDANT ROWIDS')
        indenter.stop()
    return rowid_dict


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
