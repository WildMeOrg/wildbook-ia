    def _add_dirty_rows(table, parent_rowids, config_rowid, isdirty_list,
                        config, verbose=True):
        """ Does work of adding dirty rowids """
        dirty_parent_rowids = ut.compress(parent_rowids, isdirty_list)
        try:
            # CALL EXTERNAL PREPROCESSING / GENERATION FUNCTION
            if table.isalgo:
                # TODO: DEPRICATE OLD ALGO REQUST STRUCTURE
                # HACK: config here is a request
                request = config
                #subreq = request.shallow_copy # TODO
                # FIXME: Need to vsone querys and name-vs-name queries work
                # here.
                if table.productinput:
                    # Roundabout way of forcing algo requests into the depcache
                    # structure Very ugly
                    subreq_list = list(request.shallowcopy_vsonehack(
                        qmask=isdirty_list))
                    proptup_gen_list = [table.preproc_func(table.depc, subreq)
                                        for subreq in subreq_list]
                    from itertools import chain
                    proptup_gen = chain(*proptup_gen_list)
                    dirty_params_iter = table._yeild_algo_result(
                        dirty_parent_rowids, proptup_gen, config_rowid)
                    #proptup_gen = list(proptup_gen)
                else:
                    subreq = request.shallowcopy(qmask=isdirty_list)
                    # CALL REGISTRED ALGO WORKER FUNCTION
                    proptup_gen = table.preproc_func(table.depc, subreq)
                    dirty_params_iter = table._yeild_algo_result(
                        dirty_parent_rowids, proptup_gen, config_rowid)
            else:
                args = zip(*dirty_parent_rowids)
                if table._asobject:
                    # Convinience
                    args = [table.depc.get_obj(parent, rowids)
                            for parent, rowids in zip(table.parents, args)]
                # hack config out of request
                config_ = config.config if hasattr(config, 'config') else config
                # CALL REGISTRED TABLE WORKER FUNCTION
                proptup_gen = table.preproc_func(table.depc, *args,
                                                 config=config_)
                if len(table._nested_idxs) > 0:
                    assert not table.isalgo
                    unnest_data = table._make_unnester()
                    proptup_gen = (unnest_data(data) for data in proptup_gen)
                dirty_params_iter = table._concat_rowids_data(
                    dirty_parent_rowids, proptup_gen, config_rowid)

            chunksize = (len(dirty_parent_rowids)
                         if table.chunksize is None else table.chunksize)

            # TODO: Separate this as a function which can be specified as a
            # callback.
            num_chunks = int(ceil(len(dirty_parent_rowids) / chunksize))
            chunk_iter = ut.ichunks(dirty_params_iter, chunksize=chunksize)
            lbl = 'adding %s chunk' % (table.tablename)
            prog_iter = ut.ProgIter(chunk_iter, nTotal=num_chunks, lbl=lbl)
            for dirty_params_chunk in prog_iter:
                nInput = len(dirty_params_chunk)
                if table.isalgo:
                    # HACKS, really this should be for anything that has a
                    # extern write function
                    sql_chunks = table._save_algo_result(dirty_params_chunk)
                    table.db._add(table.tablename, table._table_colnames,
                                  sql_chunks, nInput=nInput)
                else:
                    table.db._add(table.tablename, table._table_colnames,
                                  dirty_params_chunk, nInput=nInput)
        except Exception as ex:
            ut.printex(ex, 'error in add_rowids', keys=[
                'table', 'table.parents', 'parent_rowids', 'config', 'args',
                'config_rowid', 'dirty_parent_rowids', 'table.preproc_func'])
            raise


    def _concat_rowids_data(table, dirty_parent_rowids, proptup_gen,
                            config_rowid):
        for parent_rowids, data_cols in zip(dirty_parent_rowids, proptup_gen):
            try:
                yield parent_rowids + (config_rowid,) + data_cols
            except Exception as ex:
                ut.printex(ex, 'cat error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise

    def _yeild_algo_result(table, dirty_parent_rowids, proptup_gen, config_rowid):
        # TODO: generalize to all external data that needs to be written
        # explicitly
        extern_fname_list = table._get_extern_fnames(dirty_parent_rowids,
                                                     config_rowid)
        extern_dpath = table._get_extern_dpath()
        ut.ensuredir(extern_dpath, verbose=True or table.depc._debug)
        fpath_list = [join(extern_dpath, fname) for fname in extern_fname_list]
        _iter = zip(dirty_parent_rowids, proptup_gen, fpath_list)
        for parent_rowids, algo_result, extern_fpath in _iter:
            yield parent_rowids, config_rowid, algo_result, extern_fpath

    def _save_algo_result(table, dirty_params_chunk):
        for tup in dirty_params_chunk:
            parent_rowids, config_rowid, algo_result, extern_fpath = tup
            try:
                algo_result.save_to_fpath(extern_fpath, True)
                yield parent_rowids + (config_rowid,) + (extern_fpath,)
            except Exception as ex:
                ut.printex(ex, 'cat2 error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise


class AlgoRequest(BaseRequest, ut.NiceRepr):
    """
    Base class for algo request objects
    Need this for TestResult Integration

    This class might not be need, and is being added for
    compatibility support.
    The problem it solve is having daids as part of a config.  A config should
    be used to specify algorithm parameters, but a referense set of matchable
    annotations seems to go beyond that.  Therefore, AlgoRequest.

    Ignore:
        cls = dtool.AlgoRequest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from dtool.base import *  # NOQA
        >>> from dtool.example_depcache import testdata_depc
        >>> depc = testdata_depc()
        >>> #request1 = depc.new_algo_request('vsone', [1, 2], [1, 2])
        >>> request2 = depc.new_request('vsmany', [1, 2], [1, 2])
    """
    _isnewreq = True
    _qaids_independent = True
    _daids_independent = False

    @classmethod
    def new_algo_request(cls, depc, algoname, qaids, daids, cfgdict=None):
        request = cls()
        request._qaids = None
        request._daids = None

        request.depc = depc
        request.qaids = qaids
        request.daids = daids
        if cfgdict is None:
            cfgdict = {}
        configclass = depc.configclass_dict[algoname]
        config = configclass(**cfgdict)

        request.config = config
        request.algoname = algoname

        # hack
        request.params = dict(config.parse_items())
        return request

    @property
    def ibs(request):
        """ HACK specific to ibeis """
        if request.depc is None:
            return None
        return request.depc.controller

    def get_external_data_config2(request):
        # HACK
        #return None
        #print('[d] request.params = %r' % (request.params,))
        return request.params

    def get_external_query_config2(request):
        # HACK
        #return None
        #print('[q] request.params = %r' % (request.params,))
        return request.params

    @property
    def qaids(request):
        return request._qaids

    @qaids.setter
    def qaids(request, qaids):
        request._qaids = safeop(np.array, qaids)

    @property
    def daids(request):
        return request._daids

    @property
    def cfgstr(request):
        return request.get_cfgstr()

    @daids.setter
    def daids(request, daids):
        request._daids = safeop(np.array, daids)

    def get_parent_rowids(request):
        if request._daids_independent:
            parent_rowids = list(product(request.qaids, request.daids))
        else:
            parent_rowids = list(zip(request.qaids))
        return parent_rowids

    def execute(request, qaids=None, use_cache=None):
        if qaids is not None:
            qaids = [qaids] if not ut.isiterable(qaids) else qaids
            subreq = request.shallowcopy(qaids=qaids)
            return subreq.execute(use_cache=True)
        else:
            tablename = request.algoname
            table = request.depc[tablename]
            if use_cache is None:
                use_cache = not ut.get_argflag('--nocache')

            parent_rowids = request.get_parent_rowids()
            rowids = table.get_rowid(parent_rowids, config=request,
                                     recompute=not use_cache)
            result_list = table.get_row_data(rowids)
            return ut.get_list_column(result_list, 0)

    def shallowcopy_vsonehack(request, qmask=None, qaids=None):
        # Roundabout way of forcing algo requests into the depcache structure
        # Very ugly
        parent_rowids = request.get_parent_rowids()
        dirty_parents = ut.compress(parent_rowids, qmask)
        dirty_qaids = ut.take_column(dirty_parents, 0)
        dirty_daids = ut.take_column(dirty_parents, 1)
        groupxs = ut.group_indices(dirty_qaids)[1]
        daids_list = ut.apply_grouping(dirty_daids, groupxs)
        qaids_list = ut.apply_grouping(dirty_qaids, groupxs)
        for qaids, daids in zip(qaids_list, daids_list):
            #subreq = copy.copy(request)  # copy calls setstate and getstate
            subreq = request.__class__()
            subreq.__dict__.update(request.__dict__)
            subreq.qaids = qaids
            subreq.qaids = daids
            yield subreq

    def shallowcopy(request, qmask=None, qaids=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of the
        qx and dx objects.  used to generate chunks of vsone and vsmany queries
        """
        #subreq = copy.copy(request)  # copy calls setstate and getstate
        subreq = request.__class__()
        subreq.__dict__.update(request.__dict__)
        if qmask is not None:
            assert qaids is None, 'cannot specify both'
            qaid_list  = subreq.qaids
            subreq.qaids = ut.compress(qaid_list, qmask)
        elif qaids is not None:
            subreq.qaids = qaids
        return subreq

    def get_query_hashid(request):
        return request._get_rootset_hashid(request.qaids, 'Q')

    def get_data_hashid(request):
        return request._get_rootset_hashid(request.daids, 'D')

    def get_pipe_cfgstr(request):
        return request.config.get_cfgstr()

    def get_pipe_hashid(request):
        return ut.hashstr27(request.get_pipe_cfgstr())

    def get_cfgstr(request, with_input=None, with_data=None, with_pipe=True,
                   hash_pipe=False):
        r"""
        main cfgstring used to identify the 'querytype'
        """
        if with_input is None:
            #with_input = False
            with_input = not request._qaids_independent

        if with_data is None:
            #with_data = True
            # non-independent aids must be in config string
            with_data = not request._daids_independent

        cfgstr_list = []
        if with_input:
            cfgstr_list.append(request.get_query_hashid())
        if with_data:
            cfgstr_list.append(request.get_data_hashid())
        if with_pipe:
            if hash_pipe:
                cfgstr_list.append(request.get_pipe_hashid())
            else:
                cfgstr_list.append(request.get_pipe_cfgstr())
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def get_full_cfgstr(request):
        """ main cfgstring used to identify the algo hash id """
        full_cfgstr = request.get_cfgstr(with_input=True)
        return full_cfgstr

    def __nice__(request):
        dbname = (None if request.depc is None or request.depc.controller is None
                  else request.depc.controller.get_dbname())
        infostr_ = 'nQ=%s, nD=%s %s' % (len(request.qaids), len(request.daids),
                                        request.get_pipe_hashid())
        return '(%s) %s' % (dbname, infostr_)

    #def _get_rootset_hashid(request, root_rowids, prefix):
    #    uuid_type = 'V'
    #    label = ''.join((prefix, uuid_type, 'UUIDS'))
    #    uuid_list = request.depc.get_root_uuid(root_rowids)
    #    #uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=True)
    #    uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=False)
    #    return uuid_hashid

    #def __getstate__(request):
    #    state_dict = request.__dict__.copy()
    #    # SUPER HACK
    #    state_dict['dbdir'] = request.depc.controller.get_dbdir()
    #    del state_dict['depc']
    #    del state_dict['config']
    #    return state_dict

    def __setstate__(request, state_dict):
        import ibeis
        dbdir = state_dict['dbdir']
        del state_dict['dbdir']
        params = state_dict['params']
        depc = ibeis.opendb(dbdir=dbdir, web=False).depc
        configclass = depc.configclass_dict[state_dict['algoname'] ]
        config = configclass(**params)
        state_dict['depc'] = depc
        state_dict['config'] = config
        request.__dict__.update(state_dict)



    #def _row_exists(table, parent_rowids, config=None, eager=True, nInput=None,
    #                _debug=None):
    #    config_rowid = table.get_config_rowid(config=config)
    #    andwhere_colnames = table.superkey_colnames
    #    params_iter = (rowids + (config_rowid,) for rowids in parent_rowids)
    #    params_iter = list(params_iter)
    #    tblname = table.tablename
    #    flag_list = table.db.exists_where2(tblname, params_iter,
    #                                       andwhere_colnames, eager=eager,
    #                                       nInput=nInput)
    #    return flag_list


        path2_pidx = ut.make_index_lookup(path_edges_nodata, dict_factory=ut.odict)
        assert isinstance(path2_pidx, ut.odict)
        # Build mapping from each edge to the paths that is a part of
        pidx_list = ut.flatten([[idx] * len(path) for path, idx in path2_pidx.items()])
        edge_list = ut.flatten(accum_path_edges)
        edge_nodata_list = ut.flatten(path_edges_nodata)
        edge_nodata_to_datas = ut.group_items(edge_list, edge_nodata_list)
        # edge_hashable_list = [repr(edge) for edge in edge_list]
        # unique_edge_flags1 = ut.flag_unique_items(edge_nodata_list)
        # unique_edge_flags2 = ut.flag_unique_items(edge_hashable_list)
        # assert unique_edge_flags2 == unique_edge_flags1
        edge2_pidx = dict(ut.group_items(pidx_list, edge_nodata_list))
        # unique_edges = list(set(edge_nodata_list))

        type_to_paths = ut.ddict(list)
        for edge_nodata, edges in edge_nodata_to_datas.items():
            print('edge = %r' % (edge_nodata,))
            u, v, k = edge_nodata
            rinput_path_ids = ut.take_column(ut.take_column(edges, 3), 'rinput_path_id')
            rinput_path_id = rinput_path_ids[0]
            print('rinput_path_ids = %r' % (rinput_path_ids,))
            # edge_data = graph.edge[u][v][k]
            # local_input_id = edge_data['local_input_id']
            # print('local_input_id = %r' % (local_input_id,))
            pidxs = edge2_pidx[edge_nodata]
            paths = ut.take(accum_path_edges, pidxs)
            # Only take the path up to the current edge?
            # paths = [path[:path.index(edge) + 1] for path in paths]
            # print('paths = %s' % ut.repr3(paths, nl=2))
            # print('--------')
            type_to_paths[rinput_path_id].extend(paths)
            # type_to_paths[local_input_id].extend(paths)
            # type_to_pidxs[local_input_id] = pidxs
            # edge_type = edge_data['edge_type']
            # if edge_type != 'normal':
            #     pidxs = edge2_pidx[(u, v, k)]
            #     type_to_pidxs[edge_type] = pidxs
            #     type_to_paths[edge_type].extend(ut.take(path_edges, pidxs))
        # type_to_pidxs = {}

        expanded_input_graph = {}
        for type_, paths in type_to_paths.items():
            sub_edges = ut.flatten(paths)
            subgraph = ut.subgraph_from_edges(graph, sub_edges)
            expanded_input_graph[type_] = subgraph

        # use_normal = False
        # if use_normal:
        #     normal_pidxs = ut.index_complement(
        #         ut.flatten(type_to_pidxs.values()), len(path_edges))
        #     paths = ut.take(path_edges, normal_pidxs)
        #     sub_edges = ut.flatten(paths)
        #     subgraph = ut.subgraph_from_edges(graph, sub_edges)
        #     if len(subgraph.node) > 0:
        #         expanded_input_graph['normal'] = subgraph

    @ut.memoize
    def nonfinal_compute_order(table):
        """
        Returns which nodes to compute first, and what inputs are needed

            >>> from dtool.depcache_control import *  # NOQA
            >>> from dtool.example_depcache import testdata_depc
            >>> import plottool as pt
            >>> pt.ensureqt()
            >>> depc = testdata_depc()
            >>> tablename = 'neighbs'
            >>> tablename = 'multitest_score'
            >>> table = depc[tablename]
            >>> nonfinal_compute_order = table.nonfinal_compute_order()
            >>> print(ut.repr3(nonfinal_compute_order))

        """
        import networkx as nx
        expanded_input_graph = table.expanded_input_graph
        composed_graph = nx.compose_all(expanded_input_graph.values())
        #pt.show_nx(composed_graph)
        topsort = nx.topological_sort(composed_graph)
        type_to_dependlevels = ut.map_dict_vals(ut.level_order,
                                                expanded_input_graph)
        level_orders = type_to_dependlevels
        # Find computation order for all dependencies
        nonfinal_compute_order = ut.merge_level_order(level_orders, topsort)
        return nonfinal_compute_order

    @property
    @ut.memoize
    def expected_input_order(table):
        """
        Returns what input (to depc.get_rowids) ordering should be be in
        parent_rowids
        """
        from six.moves import zip_longest
        nonfinal_compute_order = table.nonfinal_compute_order()
        expanded_input_graph = table.expanded_input_graph
        hgroupids = ut.ddict(list)
        for _tablename, order in reversed(nonfinal_compute_order):
            if _tablename == table.depc.root:
                continue
            for t in order:
                s = expanded_input_graph[t]
                colxs = [y['parent_colx'] for x in s.pred[_tablename].values()
                         for y in x.values()]
                assert len(colxs) > 0
                colx = min(colxs)
                #order_colxs.append(colx)
                hgroupids[t].append(colx)

        hgroupids = dict(hgroupids)

        keys = hgroupids.keys()
        vals = hgroupids.values()
        groupids = list(zip_longest(*vals, fillvalue=0))
        hgroups = ut.hierarchical_group_items(keys, groupids)
        fgroups = ut.flatten_dict_items(hgroups)
        fkey_list = [int(''.join(map(str, key))) for key in fgroups.keys()]
        fval_list = fgroups.values()

        dupkeys = ut.find_duplicate_items(fkey_list)
        assert len(dupkeys) == 0, 'cannot have duplicate orderings'

        expected_input_order = ut.flatten(ut.sortedby(fval_list, fkey_list))
        return expected_input_order

        if False:
            nodes = ut.all_nodes_between(graph, source, target)
            tablegraph = graph.subgraph(nodes)
            import plottool as pt
            # pt.show_nx(tablegraph.reverse())
            # sink = ut.nx_sink_nodes(tablegraph)[0]
            # bfs_edges = list(ut.bfs_multi_edges(G, sink, data=True, reverse=True))
            G = tablegraph
            source = ut.nx_source_nodes(tablegraph)[0]
            bfs_edges = list(ut.bfs_multi_edges(G, source, data=0, reverse=False))
            print('bfs_edges = %r' % (bfs_edges,))
            T = nx.MultiDiGraph()
            T.add_node(source)
            T.add_edges_from(bfs_edges)
            pt.show_nx(T)

            def find_suffix(k, d):
                suffix = ''
                if d['ismulti']:
                    suffix += '_SET'
                if k != 0:
                    suffix += '_X' + str(k)
                return suffix

            G2 = nx.MultiDiGraph()
            # for u, v, k, d in G.edges(keys=True, data=True):
            edge_iter = ((u, v, k, d) for u in nx.topological_sort(G)[::-1] for v, kd in G[u].items() for k, d in kd.items())
            edges = list(edge_iter)
            for u, v, k, d in edges:
                s0 = ''
                # s0 = '_X0'
                suffix = find_suffix(k, d)
                if len(suffix) == 0:
                    G2.add_edge(u + s0, v + s0, attr_dict=d)
                else:
                    G2.add_edge(u + suffix, v + s0, attr_dict=d)
                    path_list = list(ut.all_multi_paths(G, source, u, data=True))
                    for path in path_list:
                        rpath = ut.reverse_path_edges(path)
                        parent_suffix = suffix
                        for redge in rpath:
                            v2, u2, k2, d2 = redge
                            u2 += parent_suffix
                            parent_suffix += find_suffix(k2, d2)
                            v2 += parent_suffix
                            if not G2.has_edge(u2, v2):
                                # if p2 not in G2.node:
                                G2.add_edge(u2, v2)

            pt.show_nx(G2)
            pt.show_nx(G)

        def compress_rinput_pathid(rinput_path_id):
            prev = None  # rinput_path_id[0]
            compressed = []
            for item in rinput_path_id:
                #if item != prev and not (item == '1' and prev == '2'):
                #if item != prev:
                compressed.append(item)
                # else:
                #     compressed.append(prev)
                #prev = item
            if len(compressed) > 1:
                compressed = compressed[1:]
            compressed = tuple(compressed)
            return compressed
        [[edge[3]['rinput_path_id'] for edge in path] for path in accum_path_edges]
        x = [[compress_rinput_pathid(edge[3]['rinput_path_id']) for edge in path] for path in accum_path_edges]

            >>> for type_, subgraph in expanded_input_graph.items():
            >>>     inter.append_plot(ut.partial(pt.show_nx, subgraph,
            >>>                                  title=type_))
            >>> composed_graph = nx.compose_all(expanded_input_graph.values())
            >>> inter.append_plot(ut.partial(pt.show_nx, composed_graph,
            >>>                              title='composed'))
        path_edges_nodata = ut.lmap(tuple, ut.lmap(ut.take_column, accum_path_edges, colx=slice(0, 3)))


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


#def _parse_sqlkw(kwargs):
#    default_sqlkw = dict(
#        _debug=None, ensure=True, recompute=False, recompute_all=False,
#        eager=True, nInput=None, read_extern=True, onthefly=False,
#    )
#    otherkw = kwargs.copy()
#    sqlkw = {key: otherkw.pop(key, val) for key, val in default_sqlkw.items()}
#    return sqlkw, otherkw


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
