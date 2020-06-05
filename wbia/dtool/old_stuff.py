# -*- coding: utf-8 -*-
# import utool as ut
# import numpy as np
# from six.moves import zip
# from wbia.dtool import depcache_table


# def get_all_descendant_rowids(depc, tablename, root_rowids, config=None,
#                               ensure=True, eager=True, nInput=None,
#                               recompute=False, recompute_all=False,
#                               levels_up=None, _debug=False):
#     r"""
#     Connects `root_rowids` to rowids in `tablename`, and computes all
#     values needed along the way. This is the main workhorse function for
#     dependency computations.

#     Args:
#         tablename (str): table to compute dependencies to
#         root_rowids (list): rowids for ``tablename``
#         config (dict): config applicable for all tables (default = None)
#         ensure (bool): eager evaluation if True(default = True)
#         eager (bool): (default = True)
#         nInput (None): (default = None)
#         recompute (bool): (default = False)
#         recompute_all (bool): (default = False)
#         levels_up (int): only partially compute dependencies (default = 0)
#         _debug (bool): (default = False)

#     CommandLine:
#         python -m dtool.depcache_control --exec-get_all_descendant_rowids:0
#         python -m dtool.depcache_control --exec-get_all_descendant_rowids:1

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> tablename = 'spam'
#         >>> root_rowids = [1, 2]
#         >>> config1 = {'dim_size': 500}
#         >>> config2 = {'dim_size': 100}
#         >>> config3 = {'dim_size': 500, 'adapt_shape': False}
#         >>> ensure, eager, nInput = True, True, None
#         >>> _debug = True
#         >>> rowid_dict1 = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config1, ensure, eager, nInput, _debug=_debug)
#         >>> rowid_dict2 = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config2, ensure, eager, nInput, _debug=_debug)
#         >>> rowid_dict3 = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config3, ensure, eager, nInput, _debug=_debug)
#         >>> result1 = 'rowid_dict1 = ' + ut.repr3(rowid_dict1, nl=1)
#         >>> result2 = 'rowid_dict2 = ' + ut.repr3(rowid_dict2, nl=1)
#         >>> result3 = 'rowid_dict3 = ' + ut.repr3(rowid_dict3, nl=1)
#         >>> result = '\n'.join([result1, result2, result3])
#         >>> print(result)
#         rowid_dict1 = {
#             'chip': [1, 2],
#             'dummy_annot': [1, 2],
#             'fgweight': [1, 2],
#             'keypoint': [1, 2],
#             'probchip': [1, 2],
#             'spam': [1, 2],
#         }
#         rowid_dict2 = {
#             'chip': [3, 4],
#             'dummy_annot': [1, 2],
#             'fgweight': [3, 4],
#             'keypoint': [3, 4],
#             'probchip': [1, 2],
#             'spam': [3, 4],
#         }
#         rowid_dict3 = {
#             'chip': [1, 2],
#             'dummy_annot': [1, 2],
#             'fgweight': [5, 6],
#             'keypoint': [5, 6],
#             'probchip': [1, 2],
#             'spam': [5, 6],
#         }


#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> _debug = True
#         >>> tablename = 'vsmany'
#         >>> config = depc.configclass_dict['vsmany']()
#         >>> root_rowids = [1, 2, 3]
#         >>> ensure, eager, nInput = False, True, None
#         >>> # Get rowids of algo ( should be None )
#         >>> rowid_dict = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config, ensure, eager, nInput,
#         >>>     _debug=_debug)
#         >>> result = ut.repr3(rowid_dict, nl=1)
#         >>> print(result)
#         {
#             'dummy_annot': [1, 2, 3],
#             'vsmany': [None, None, None],
#         }

#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> # Make sure algo config can correctly get properites
#         >>> depc = testdata_depc()
#         >>> tablename = 'chip'
#         >>> recompute = False
#         >>> recompute_all = False
#         >>> _debug = True
#         >>> root_rowids = [1, 2]
#         >>> configclass = depc.configclass_dict['chip']
#         >>> config_ = configclass()
#         >>> config1 = depc.configclass_dict['vsmany'](dim_size=500)
#         >>> config2 = depc.configclass_dict['vsmany'](dim_size=100)
#         >>> config = config2
#         >>> prop_dicts1 = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config=config1, _debug=_debug)
#         >>> prop_dicts2 = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config=config2, _debug=_debug)
#         >>> print(prop_dicts2)
#         >>> print(prop_dicts1)
#         >>> assert prop_dicts1 != prop_dicts2

#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> exec(ut.execstr_funckw(depc.get_all_descendant_rowids), globals())
#         >>> _debug = True
#         >>> qaids, daids = [1, 2, 4], [2, 3, 4]
#         >>> root_rowids = list(zip(*ut.product(qaids, daids)))
#         >>> request = depc.new_request('vsone', qaids, daids)
#         >>> results = request.execute()
#         >>> tablename = 'vsone'
#         >>> rowid_dict = depc.get_all_descendant_rowids(
#         >>>     tablename, root_rowids, config=None, _debug=_debug)
#     """
#     # TODO: Need to have a nice way of ensuring configs dont overlap
#     # via namespaces.
#     _debug = depc._debug if _debug is None else _debug
#     indenter = ut.Indenter('[Descend-to-%s]' % (tablename,), enabled=_debug)
#     if _debug:
#         indenter.start()
#         print(' * GET DESCENDANT ROWIDS %s ' % (tablename,))
#         print(' * config = %r' % (config,))
#     dependency_levels = depc.get_dependencies(tablename)
#     if levels_up is not None:
#         dependency_levels = dependency_levels[:-levels_up]

#     configclass_levels = [
#         [depc.configclass_dict.get(tablekey, None)
#          for tablekey in keys]
#         for keys in dependency_levels
#     ]
#     if _debug:
#         print('[depc] dependency_levels = %s' %
#               ut.repr3(dependency_levels, nl=1))
#         print('[depc] config_levels = %s' %
#               ut.repr3(configclass_levels, nl=1))

#     # TODO: better support for multi-edges
#     if (len(root_rowids) > 0 and ut.isiterable(root_rowids[0]) and
#          not depc[tablename].ismulti):
#         rowid_dict = {}
#         for colx, col in enumerate(root_rowids):
#             rowid_dict[depc.root + '%d' % (colx + 1,)] = col
#         rowid_dict[depc.root] = ut.unique_ordered(ut.flatten(root_rowids))
#     else:
#         rowid_dict = {depc.root: root_rowids}

#     # Ensure that each level ``tablename``'s dependencies have been computed
#     for level_keys in dependency_levels[1:]:
#         if _debug:
#             print(' * level_keys %s ' % (level_keys,))
#         # For each table in the level
#         for tablekey in level_keys:
#             try:
#                 child_rowids = depc._expand_level_rowids(
#                     tablename, tablekey, rowid_dict, ensure, eager, nInput,
#                     config, recompute, recompute_all, _debug)
#             except Exception as ex:
#                 table = depc[tablekey]  # NOQA
#                 keys = ['tablename', 'tablekey', 'rowid_dict', 'config',
#                         'table', 'dependency_levels']
#                 ut.printex(ex, 'error expanding rowids', keys=keys)
#                 raise
#             rowid_dict[tablekey] = child_rowids
#     if _debug:
#         print(' GOT DESCENDANT ROWIDS')
#         indenter.stop()
#     return rowid_dict


# def get_rowids(depc, tablename, root_rowids, config=None, ensure=True,
#                eager=True, nInput=None, _debug=None, recompute=False,
#                recompute_all=False):
#     """
#     Returns the rowids of `tablename` that correspond to `root_rowids`
#     using `config`.

#     Ignore:
#         tablename = 'nnindexer'
#         multi_rowids = (1, 2, 3, 4, 5)
#         root_rowids = [[multi_rowids]]
#         import wbia.plottool as pt
#         pt.ensureqt()

#         from wbia.dtool.depcache_control import *  # NOQA
#         from wbia.dtool.example_depcache import testdata_depc
#         depc = testdata_depc()
#         exec(ut.execstr_funckw(depc.get_rowids), globals())
#         print(ut.depth_profile(root_rowids))
#         tablename = 'neighbs'
#         table = depc[tablename]  # NOQA
#         import wbia.plottool as pt
#         pt.ensureqt()
#         _debug = depc._debug = True
#         depc.get_rowids(tablename, root_rowids, config, _debug=_debug)

#         pt.show_nx(depc.graph)
#         for key, val in table.type_to_subgraph.items():
#             pt.show_nx(val)
#             pt.set_title(key)

#     CommandLine:
#         python -m dtool.depcache_control --exec-get_rowids
#         python -m dtool.depcache_control --dump-get_rowids
#         python -m dtool.depcache_control --exec-get_rowids:0

#     GridParams:
#         >>> param_grid = dict(
#         >>>     tablename=[ 'spam', 'neighbs'] # 'spam', 'multitest_score','keypoint'],
#         >>>   #tablename=['neighbs', 'keypoint', 'spam', 'multitest_score','keypoint'],
#         >>> )
#         >>> flat_root_ids = [1, 2, 3]
#         >>> combos = ut.all_dict_combinations(param_grid)
#         >>> index = 0
#         >>> keys = 'tablename'.split(', ')
#         >>> tablename, = ut.dict_take(combos[index], keys)

#     Setup:
#         >>> # DISABLE_GRID_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
#         >>> import wbia.plottool as pt
#         >>> pt.ensureqt()
#         >>> #pt.show_nx(depc.graph)

#     GridExample0:
#         >>> table = depc[tablename]  # NOQA
#         >>> flat_root_ids = [1, 2, 3]
#         >>> root_rowids = [flat_root_ids for _ in table.input_order]
#         >>> print('root_rowids = %r' % (root_rowids,))
#         >>> #root_rowids = [[flat_root_ids], [(flat_root_ids,)]]
#         >>> #root_rowids = [list(zip(flat_root_ids)), (flat_root_ids,)]
#         >>> _debug = True
#         >>> depc.get_rowids(tablename, root_rowids, config, _debug=_debug)
#         >>> for key, val in table.type_to_subgraph.items():
#         >>>     pt.show_nx(val)
#         >>>     pt.set_title(key)

#     Example1:
#         >>> # ENABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
#         >>> root_rowids = [1, 2, 3]
#         >>> tablename = 'spam'
#         >>> table = depc[tablename]
#         >>> kp_rowids = depc.get_rowids(tablename, root_rowids)
#         >>> #result = ('prop_list = %s' % (ut.repr2(prop_list),))
#         >>> #print(result)

#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from wbia.dtool.depcache_control import *  # NOQA
#         >>> from wbia.dtool.example_depcache import testdata_depc
#         >>> depc = testdata_depc()
#         >>> exec(ut.execstr_funckw(depc.get_rowids), globals())
#         >>> flat_root_ids = [1, 2, 3]
#         >>> kp_rowids = depc.get_rowids('keypoint', flat_root_ids)
#         >>> root_rowids = [flat_root_ids] * 8
#         >>> _debug = True
#         >>> tablename = 'nnindexer'
#         >>> tablename = 'multitest_score'
#         >>> table = depc[tablename]  # NOQA
#         >>> #result = ('prop_list = %s' % (ut.repr2(prop_list),))
#         >>> # print(result)
#     """
#     _debug = depc._debug if _debug is None else _debug
#     if _debug:
#         print(' * root_rowids=%s' % (ut.trunc_repr(root_rowids),))
#         print(' * config = %r' % (config,))
#     table = depc[tablename]  # NOQA
#     INDEXER_VERSION = False

#     if tablename == 'neighbor_index':
#         """
#         python -m wbia.core_annots --exec-compute_neighbor_index --show
#         """

#         import utool
#         utool.embed()

#     if INDEXER_VERSION or tablename == 'neighbs':
#         compute_order = table.compute_order
#         depend_order = compute_order['depend_compute_ids']
#         input_order = compute_order['input_compute_ids']

#         if _debug:
#             print(' * input_order = %s' % (ut.repr3(input_order, nl=1),))
#             print(' * depend_order = %s' % (ut.repr3(depend_order, nl=1),))
#         if len(input_order) > 1:
#             assert ut.depth_atleast(root_rowids, 2), (
#                 'input_order = %r' % (input_order,))

#         with ut.Indenter('[GetRowID-%s]' % (tablename,),
#                          enabled=_debug):
#             # New way to get rowids
#             input_level = depend_order[0]
#             mid_levels = depend_order[1:-1]
#             output_level = depend_order[-1]

#             # List that holds a mapping from input order to input "name"
#             input_order_lookup = ut.make_index_lookup(input_order)
#             # Dictionary that holds the rowids computed for each table
#             # while tracing the dependencies.
#             rowid_lookup = ut.odict([(key, ut.odict()) for key in input_order])

#             # Need to split each path into parts.
#             # Each part represents another level of unflattening
#             # (because root indicies are all flat)

#             # Handle input level
#             assert input_level[0] == depc.root
#             for compute_id in input_order:
#                 # for name in input_names:
#                 argx = input_order_lookup[compute_id]
#                 rowid_lookup[compute_id] = root_rowids[argx]
#                 # HACK: Flatten to scalars
#                 # The inputs should just be given in the "correct" nesting.
#                 # TODO: determine what correct nesting is.
#                 for i in range(5):
#                     try:
#                         current = rowid_lookup[compute_id]
#                         rowid_lookup[compute_id] = ut.flatten(current)
#                     except Exception:
#                         pass

#             level = 0
#             if _debug:
#                 print('input_order_lookup = %r' % (input_order_lookup,))
#                 ut.printdict(rowid_lookup, 'rowid_lookup')

#             def handle_level(compute_id, rowid_lookup, _recompute, level):
#                 print('+--- HANDLE LEVEL %d -------' % (level,))
#                 tablekey = compute_id[0]
#                 input_suff = compute_id[1]
#                 config_ = depc._ensure_config(tablekey, config)
#                 table = depc[tablekey]
#                 lookupkeys = [(n, input_suff) for n in table.parent_id_tablenames]
#                 # ordering = ut.dict_take(input_order_lookup, input_names)
#                 # sortx = ut.argsort(ordering)
#                 # FIXME: get inputs for each table.
#                 # input_names = ut.take(input_names, sortx)
#                 # lookupkeys = list(ut.iprod(table.parent_id_tablenames, input_names))
#                 # lookupkeys = list(zip(table.parent_id_tablenames, input_types))
#                 if _debug:
#                     print('---- LOCALS ------')
#                     ut.print_locals(compute_id, tablekey, lookupkeys, table)
#                     print('L----------')
#                 # FIXME generalize
#                 _parent_ids = [rowid_lookup[tblkey] for tblkey in lookupkeys]
#                 if table.ismulti:
#                     parent_rowidsT = [[tuple(x)] for x in _parent_ids]
#                 else:
#                     parent_rowidsT = _parent_ids
#                 parent_rowidsT = np.broadcast_arrays(*parent_rowidsT)
#                 parent_rowids = list(zip(*parent_rowidsT))
#                 # Probably not right for general multi-input
#                 import utool
#                 with utool.embed_on_exception_context:
#                     next_rowids = table.get_rowid(
#                         parent_rowids, config=config_, eager=eager, nInput=nInput,
#                         ensure=ensure, recompute=_recompute)
#                 rowid_lookup[compute_id] = next_rowids
#                 if _debug:
#                     ut.printdict(rowid_lookup, 'rowid_lookup')
#                 if _debug:
#                     print('L___ HANDLE LEVEL %d -------' % (level,))
#                 return next_rowids

#             # Handle mid levels
#             _recompute = recompute_all
#             for level, compute_id in enumerate(mid_levels, start=1):
#                 handle_level(compute_id, rowid_lookup, _recompute, level)
#             level += 1

#             # Handel final (requested) level
#             compute_id = output_level
#             _recompute = recompute
#             rowid_list =  handle_level(compute_id, rowid_lookup,
#                                        _recompute, level)
#     else:
#         with ut.Indenter('[GetRowID-%s]' % (tablename,),
#                          enabled=_debug):
#             # TODO: Get nonself rowids first
#             # THen get self rowids for debugging ease
#             try:
#                 if False:
#                     recompute_ = recompute or recompute_all
#                     parent_rowids = depc._get_parent_input(
#                         tablename, root_rowids, config, ensure=True, _debug=None,
#                         recompute=False, recompute_all=False, eager=True,
#                         nInput=None)
#                     config_ = depc._ensure_config(tablename, config)
#                     #if onthefly:
#                     #    pass
#                     table = depc[tablename]
#                     rowid_list = table.get_rowid(
#                         parent_rowids, config=config_, eager=eager, nInput=nInput,
#                         ensure=ensure, recompute=recompute_)
#                 else:
#                     # Compute everything from the root to the requested table
#                     rowid_dict = depc.get_all_descendant_rowids(
#                         tablename, root_rowids, config=config, ensure=ensure,
#                         eager=eager, nInput=nInput, recompute=recompute,
#                         recompute_all=recompute_all, _debug=ut.countdown_flag(_debug))
#                     rowid_list = rowid_dict[tablename]
#             except depcache_table.ExternalStorageException:
#                 print('EXTERNAL EXCEPTION One retry in get_rowids')
#                 rowid_dict = depc.get_all_descendant_rowids(
#                     tablename, root_rowids, config=config, ensure=ensure,
#                     eager=eager, nInput=nInput, recompute=recompute,
#                     recompute_all=recompute_all, _debug=ut.countdown_flag(_debug))
#                 rowid_list = rowid_dict[tablename]
#     if _debug:
#         print(' * return rowid_list = %s' % (ut.trunc_repr(rowid_list),))
#     return rowid_list


# def _get_parent_input(depc, tablename, root_rowids, config, ensure=True,
#                       _debug=None, recompute=False, recompute_all=False,
#                       eager=True, nInput=None):
#     # Get ancestor rowids that are descendants of root
#     table = depc[tablename]
#     rowid_dict = depc.get_all_descendant_rowids(
#         tablename, root_rowids, config=config, ensure=ensure,
#         eager=eager, nInput=nInput, recompute=recompute,
#         recompute_all=recompute_all, _debug=ut.countdown_flag(_debug),
#         levels_up=1)
#     parent_rowids = depc._get_parent_rowids(table, rowid_dict)
#     return parent_rowids

# #def get_relevant_subconfigs(depc, tablename, config):
# #    depc._ensure_config(tablename, config)
# #    pass


# def _get_parent_rowids(depc, table, rowid_dict):
#     # FIXME to handle multiedges correctly
#     parent_rowidsT = ut.dict_take(rowid_dict,
#                                   table.parent_id_tablenames)
#     if table.ismulti:
#         parent_rowids = parent_rowidsT
#     else:
#         parent_rowids = ut.list_transpose(parent_rowidsT)
#     return parent_rowids


# def _expand_level_rowids(depc, tablename, tablekey, rowid_dict, ensure,
#                          eager, nInput, config, recompute, recompute_all,
#                          _debug):
#     table = depc[tablekey]
#     config_ = depc._ensure_config(tablekey, config)
#     parent_rowids = depc._get_parent_rowids(table, rowid_dict)
#     if _debug:
#         print('   * tablekey = %r' % (tablekey,))
#         print('   * (ensured) config_ = %r' % (config_,))
#         print('   * config_rowid = %r' % (table.get_config_rowid(config_),))
#         print('   * parent_rowids = %s' % (ut.trunc_repr(parent_rowids),))
#     _recompute = recompute_all or (tablekey == tablename and recompute)
#     level_rowids = table.get_rowid(
#         parent_rowids, config=config_, eager=eager, nInput=nInput,
#         ensure=ensure, recompute=_recompute)
#     if _debug:
#         print('   * level_rowids = %s' % (ut.trunc_repr(level_rowids),))
#     r


# def test_getprop_with_configs():
#     r"""
#     CommandLine:
#         python -m dtool.example_depcache2 test_getprop_with_configs --show

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from wbia.dtool.example_depcache2 import *  # NOQA
#         >>> test_getprop_with_configs()
#     """
#     config1 = {'manual_extract': True}
#     config2 = {'manual_extract': False}
#     depc = testdata_depc2()

#     aid = 2
#     _debug = False

#     cropchip1 = depc.get('cropchip', aid, 'img', config=config1)
#     cropchip2 = depc.get('cropchip', aid, 'img', config=config2)
#     print('cropchip1.shape = %r' % (cropchip1.shape,))
#     print('cropchip2.shape = %r' % (cropchip2.shape,))
#     cropchip1 = depc.get('cropchip', aid, 'img', config=config1)
#     cropchip2 = depc.get('cropchip', aid, 'img', config=config2)

#     print('cropchip1.shape = %r' % (cropchip1.shape,))
#     print('cropchip2.shape = %r' % (cropchip2.shape,))

#     chip = depc.get('chip', aid, 'img')
#     print('chip.shape = %r' % (chip.shape,))

#     tip1 = depc.get('tip', aid, config=config1, _debug=_debug)
#     tip2 = depc.get('tip', aid, config=config2, _debug=_debug)

#     print('tip1 = %r' % (tip1,))
#     print('tip2 = %r' % (tip2,))

#     depc.print_all_tables()
#     depc.print_config_tables()
#     #import utool
#     #utool.embed()


# def testdata_depc2():
#     """
#     Example of local registration
#     sudo pip install freetype-py

#     CommandLine:
#         python -m dtool.example_depcache2 testdata_depc2 --show

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from wbia.dtool.example_depcache2 import *  # NOQA
#         >>> depc = testdata_depc2()
#         >>> ut.quit_if_noshow()
#         >>> import wbia.plottool as pt
#         >>> depc.show_graph()
#         >>> ut.show_if_requested()
#     """
#     from wbia import dtool
#     import vtool as vt
#     from vtool import fontdemo

#     # put the test cache in the dtool repo
#     dtool_repo = dirname(ut.get_module_dir(dtool))
#     cache_dpath = join(dtool_repo, 'DEPCACHE2')

#     root = 'annot'

#     depc = dtool.DependencyCache(
#         root_tablename=root, cache_dpath=cache_dpath, use_globals=False)

#     # ----------

#     class ChipConfig(dtool.Config):
#         _param_info_list = [
#             ut.ParamInfo('dim_size', 500),
#             ut.ParamInfo('ext', '.png'),
#         ]

#     @depc.register_preproc(
#         tablename='chip', parents=[root], colnames=['size', 'img'],
#         coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
#         configclass=ChipConfig)
#     def compute_chip(depc, aids, config=None):
#         for aid in aids:
#             chip = fontdemo.get_text_test_img(str(aid))
#             size = vt.get_size(chip)
#             yield size, chip

#     # ----------

#     class TipConfig(dtool.Config):
#         _param_info_list = [
#             ut.ParamInfo('manual_extract', False, hideif=False),
#         ]

#     @depc.register_preproc(
#         tablename='tip', parents=['chip'],
#         colnames=['notch', 'left', 'right'],
#         coltypes=[np.ndarray, np.ndarray, np.ndarray],
#         configclass=TipConfig,
#     )
#     def compute_tips(depc, chip_rowids, config=None):
#         manual_extract = config['manual_extract']
#         chips = depc.get_native('chip', chip_rowids, 'img')
#         for chip in chips:
#             seed = (chip).sum()
#             perb = ((seed % 1000) / 1000) * .25
#             w, h = vt.get_size(chip)
#             if manual_extract:
#                 # Make noticable difference between config outputs
#                 lpb =  np.ceil(w * perb)
#                 npb =  np.ceil(h * perb)
#                 rpb = -np.ceil(w * perb)
#             else:
#                 lpb =  np.ceil(w * perb / 2)
#                 npb = -np.ceil(h * perb)
#                 rpb = -np.ceil(w * perb)
#             wh = np.array([w, h], dtype=np.int32)[None, :]
#             rel_base = np.array([[.0, .5], [.5, .5], [1., .5]])
#             offset   = np.array([[lpb, 0], [0, npb], [rpb, 0]])
#             tip = np.round((wh * rel_base)) + offset
#             left, notch, right = tip
#             yield left, notch, right

#     # ----------

#     class CropChipConfig(dtool.Config):
#         _param_info_list = [
#             ut.ParamInfo('dim_size', 500),
#         ]

#     @depc.register_preproc(
#         tablename='cropchip', parents=['chip', 'tip'],
#         colnames=['img'],
#         coltypes=[np.ndarray],
#         configclass=CropChipConfig,
#     )
#     def compute_cropchip(depc, cids, tids, config=None):
#         print("COMPUTE CROPCHIP")
#         print('config = %r' % (config,))
#         chips = depc.get_native('chip', cids, 'img')
#         tips = depc.get_native('tip', tids)
#         print('tips = %r' % (tips,))
#         for chip, tip in zip(chips, tips):
#             notch, left, right = tip
#             lx = left[0]
#             rx = right[0]
#             cropped_chip = chip[lx:(rx - 1), ...]
#             yield (cropped_chip,)

#     # ----------

#     class TrailingEdgeConfig(dtool.Config):
#         _param_info_list = []

#     @depc.register_preproc(
#         tablename='trailingedge', parents=['cropchip'],
#         colnames=['te'],
#         coltypes=[np.ndarray],
#         configclass=TrailingEdgeConfig,
#     )
#     def compute_trailing_edge(depc, cropped_chips, config=None):
#         for cc in cropped_chips:
#             #depc.get_native('chip', cids)
#             size = 1
#             te = np.arange(size)
#             yield (te,)

#     depc.initialize()
#     return depc


# def testdata_depc_image():
#     """
#     Example of local registration
#     sudo pip install freetype-py

#     CommandLine:
#         python -m dtool.example_depcache2 testdata_depc_image --show

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from wbia.dtool.example_depcache2 import *  # NOQA
#         >>> depc = testdata_depc_image()
#         >>> ut.quit_if_noshow()
#         >>> import wbia.plottool as pt
#         >>> depc.show_graph()
#         >>> depc['detection'].show_input_graph()
#         >>> print(depc['detection'].compute_order)
#         >>> ut.show_if_requested()
#     """
#     from wbia import dtool

#     # put the test cache in the dtool repo
#     dtool_repo = dirname(ut.get_module_dir(dtool))
#     cache_dpath = join(dtool_repo, 'DEPCACHE2')

#     root = 'image'

#     depc = dtool.DependencyCache(
#         root_tablename=root, cache_dpath=cache_dpath, use_globals=False)

#     # ----------
#     dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
#     def dummy_func(depc, *args, **kwargs):
#         for row_arg in zip(*args):
#             yield (np.array([42]),)

#     depc.register_preproc(tablename='detector', parents=['image*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='detection', parents=['image', 'detector'], **dummy_cols)(dummy_func)

#     depc.initialize()
#     return depc


# def testdata_depc_annot():
#     """
#     CommandLine:
#         python -m dtool.example_depcache2 testdata_depc_annot --show

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from wbia.dtool.example_depcache2 import *  # NOQA
#         >>> depc = testdata_depc_annot()
#         >>> ut.quit_if_noshow()
#         >>> import wbia.plottool as pt
#         >>> depc.show_graph()
#         >>> tablename = 'featweight'
#         >>> table = depc[tablename]
#         >>> table.show_input_graph()
#         >>> print(table.compute_order)
#         >>> ut.show_if_requested()
#     """
#     from wbia import dtool
#     # put the test cache in the dtool repo
#     dtool_repo = dirname(ut.get_module_dir(dtool))
#     cache_dpath = join(dtool_repo, 'DEPCACHE2')
#     dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
#     def dummy_func(depc, *args, **kwargs):
#         for row_arg in zip(*args):
#             yield (np.array([42]),)

#     # NOTE: Consider the smk_match.
#     # It would be really cool if we could say that the vocab
#     # for the input to the parent smk_vec must be the same vocab
#     # that was used to compute the inverted index. How do we encode that?

#     root = 'annot'
#     #vocab_parent = 'annot'
#     #vocab_parent = 'chip'
#     #vocab_parent = 'feat'
#     vocab_parent = 'featweight'
#     depc = dtool.DependencyCache(
#         root_tablename=root, cache_dpath=cache_dpath, use_globals=False)
#     depc.register_preproc(tablename='chip', parents=['annot'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='fgmodel', parents=['chip*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='probchip', parents=['annot', 'fgmodel'], **dummy_cols)(dummy_func)
#     #depc.register_preproc(tablename='probchip', parents=['annot'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='feat', parents=['chip'], **dummy_cols)(dummy_func)
#     #depc.register_preproc(tablename='feat', parents=['annot'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='featweight', parents=['feat', 'probchip'], **dummy_cols)(dummy_func)

#     depc.register_preproc(tablename='indexer', parents=[vocab_parent + '*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='neighbs', parents=[vocab_parent, 'indexer'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='vocab', parents=[vocab_parent + '*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='smk_vec', parents=[vocab_parent, 'vocab'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='inv_index', parents=['smk_vec*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='smk_match', parents=['smk_vec', 'inv_index'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='vsone', parents=[vocab_parent, vocab_parent], **dummy_cols)(dummy_func)

#     depc.register_preproc(tablename='viewpoint_model', parents=['annot*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='viewpoint', parents=['annot', 'viewpoint_model'], **dummy_cols)(dummy_func)

#     depc.register_preproc(tablename='quality_model', parents=['annot*'], **dummy_cols)(dummy_func)
#     depc.register_preproc(tablename='quality', parents=['annot', 'quality_model'], **dummy_cols)(dummy_func)

#     depc.initialize()
#     return depc

# return level_rowids
