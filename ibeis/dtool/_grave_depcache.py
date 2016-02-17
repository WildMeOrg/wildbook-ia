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
