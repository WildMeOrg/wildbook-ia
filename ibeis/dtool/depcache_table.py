# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from six.moves import zip, range
from os.path import join, exists
from math import ceil
from dtool import base
from dtool import __SQLITE__ as lite
(print, rrr, profile) = ut.inject2(__name__, '[depcache_table]')


EXTERN_SUFFIX = '_extern_uri'

CONFIG_TABLE  = 'config'
CONFIG_ROWID  = 'config_rowid'
CONFIG_HASHID = 'config_hashid'
CONFIG_STRID  = 'config_strid'


def predrop_grace_period(tablename, seconds=10):
    warnmsg_fmt = ut.codeblock(
        '''
        WARNING TABLE={tablename} IS MODIFIED

        About to reset (DROP) entire cache={tablename}.

        Generally this is OK and you shouldnt worry because depcache
        information should be recomputable.

        If you really dont want this to happen you have {seconds} seconds to
        kill this process before deletion occurs.
        ''')
    warnmsg = warnmsg_fmt.format(tablename=tablename, seconds=seconds)
    return ut.grace_period(warnmsg, seconds)
    #return ut.are_you_sure(warnmsg)


def ensure_config_table(db):
    config_addtable_kw = ut.odict(
        [
            ('tablename', CONFIG_TABLE,),
            ('coldef_list', [
                (CONFIG_ROWID, 'INTEGER PRIMARY KEY'),
                (CONFIG_HASHID, 'TEXT'),
                (CONFIG_STRID, 'TEXT'),
            ],),
            ('docstr', 'table for algo configurations'),
            ('superkeys', [(CONFIG_HASHID,)]),
            ('dependson', [])
        ]
    )

    if not db.has_table(CONFIG_TABLE):
        db.add_table(**config_addtable_kw)
    else:
        current_state = db.get_table_autogen_dict(CONFIG_TABLE)
        new_state = config_addtable_kw
        if current_state['coldef_list'] != new_state['coldef_list']:
            if predrop_grace_period(CONFIG_TABLE):
                db.drop_all_tables()
                db.add_table(**new_state)
            else:
                raise NotImplementedError('Need to be able to modify tables')


class DependencyCacheTable(object):
    """
    An individual node in the dependency graph.
    """

    def __init__(table, depc=None, parent_tablenames=None, tablename=None,
                 data_colnames=None, data_coltypes=None, preproc_func=None,
                 docstr='no docstr', fname=None, asobject=False,
                 version=None,
                 chunksize=None, isalgo=False, isinteractive=False):

        table.db = None
        table.fpath_to_db = {}

        table.parent_tablenames = parent_tablenames
        table.tablename = tablename
        table.data_colnames = tuple(data_colnames)
        table.data_coltypes = data_coltypes
        table.preproc_func = preproc_func
        table.on_delete = None

        table._internal_data_colnames = []
        table._internal_data_coltypes = []
        table._nested_idxs = []
        table.sqldb_fpath = None
        table.extern_read_funcs = {}
        table._nested_idxs2 = []
        table.isalgo = isalgo
        table.isinteractive = isinteractive
        table.version = version

        table.docstr = docstr
        table.fname = fname
        table.depc = depc
        table.chunksize = chunksize
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
                if not table.isalgo:
                    if len(args) < 3:
                        print('args = %r' % (args,))
                        msg = ('preproc_func=%r for table=%s must have a depcache arg, at'
                               ' least one parent rowid arg, and a config'
                               ' arg') % (table.preproc_func, table.tablename,)
                        raise AssertionError(msg)
                    rowid_args = args[1:-1]
                    if len(rowid_args) != len(table.parents):
                        print('table.preproc_func = %r' % (table.preproc_func,))
                        print('args = %r' % (args,))
                        print('rowid_args = %r' % (rowid_args,))
                        msg = (
                            ('preproc function for table=%s must have as many '
                             'rowids %d args as parents %d') % (
                                table.tablename, len(rowid_args), len(table.parents))
                        )
                        raise AssertionError(msg)

    def _update_internals(table):
        extern_read_funcs = {}
        # TODO: can rewrite much of this
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
        parent_coldef = [(key, 'INTEGER NOT NULL')
                         for key in table.parent_rowid_colnames]
        config_coldef = [(CONFIG_ROWID, 'INTEGER DEFAULT 0')]
        internal_data_coldef = list(zip(table._internal_data_colnames,
                                        table._internal_data_coltypes))

        coldef_list = (primary_coldef + parent_coldef +
                       config_coldef + internal_data_coldef)
        add_table_kw = ut.odict([
            ('tablename', table.tablename,),
            ('coldef_list', coldef_list,),
            ('docstr', table.docstr,),
            ('superkeys', [table.superkey_colnames],),
            ('dependson', table.parents),
        ])
        return add_table_kw

    @profile
    def initialize(table):
        table.db = table.depc.fname_to_db[table.fname]
        #print('Checking sql for table=%r' % (table.tablename,))

        if not table.db.has_table(table.tablename):
            print('Initializing table=%r' % (table.tablename,))
            new_state = table.get_addtable_kw()
            table.db.add_table(**new_state)
        else:
            # TODO: Check for table modifications
            new_state = table.get_addtable_kw()
            current_state = table.db.get_table_autogen_dict(table.tablename)
            if current_state['coldef_list'] != new_state['coldef_list']:
                print('WARNING TABLE IS MODIFIED')
                if predrop_grace_period(table.tablename):
                    table.clear_table()
                else:
                    raise NotImplementedError('Need to be able to modify tables')

    def clear_table(table):
        print('Clearing data in %r' % (table,))
        table.db.drop_table(table.tablename)
        table.db.add_table(**table.get_addtable_kw())

    def print_schemadef(table):
        print('\n'.join(table.db.get_table_autogen_str(table.tablename)))

    def print_csv(table):
        print(table.db.get_table_csv(table.tablename))

    def _get_all_rowids(table):
        pass

    @property
    def tabletype(table):
        return 'algo' if table.isalgo else 'node'

    @property
    def parents(table):
        return table.parent_tablenames

    @property
    def children(table):
        # TODO
        pass

    @property
    def columns(table):
        return table.data_colnames

    @property
    def extern_columns(table):
        return list(table.external_to_internal.keys())

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

    def get_config_rowid(table, config=None, _debug=None):
        if isinstance(config, int):
            config_rowid = config
        else:
            config_rowid = table.add_config(config, _debug)
        return config_rowid

    def get_config_hashid(table, config_rowid_list):
        hashid_list = table.db.get(
            CONFIG_TABLE, (CONFIG_HASHID,), config_rowid_list,
            id_colname=CONFIG_ROWID)
        return hashid_list

    def get_config_rowid_from_hashid(table, config_hashid_list):
        config_rowid_list = table.db.get(
            CONFIG_TABLE, (CONFIG_ROWID,), config_hashid_list,
            id_colname=CONFIG_HASHID)
        return config_rowid_list

    def add_config(table, config, _debug=None):
        if isinstance(config, base.TableConfig):
            config_strid = config.get_cfgstr()
        elif isinstance(config, base.AlgoRequest):
            config_strid = config.get_cfgstr()
        else:
            config_strid = ut.to_json(config)
        if table.version is not None:
            config_strid += '_version(%s)' % (table.version,)
        config_hashid = ut.hashstr27(config_strid)
        if table.depc._debug or _debug:
            print('config_strid = %r' % (config_strid,))
            print('config_hashid = %r' % (config_hashid,))
        get_rowid_from_superkey = table.get_config_rowid_from_hashid
        config_rowid_list = table.db.add_cleanly(
            CONFIG_TABLE, (CONFIG_HASHID, CONFIG_STRID,), [(config_hashid, config_strid,)],
            get_rowid_from_superkey)
        config_rowid = config_rowid_list[0]
        if table.depc._debug:
            print('config_rowid_list = %r' % (config_rowid_list,))
            print('config_rowid = %r' % (config_rowid,))
        return config_rowid

    # ----------------------
    # --- GETTERS NATIVE ---
    # ----------------------

    def _make_unnester(table):
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
            unflat = ut.ungroup(grouped_items, groupxs,
                                nested_nCols - 1)
            return tuple(ut.flatten(unflat))
        # Hack when a sql schema has tuples defined in it
        return unnest_data

    def _concat_rowids_data(table, dirty_parent_rowids, proptup_gen, config_rowid):
        for parent_rowids, data_cols in zip(dirty_parent_rowids, proptup_gen):
            try:
                yield parent_rowids + (config_rowid,) + data_cols
            except Exception as ex:
                ut.printex(ex, 'cat error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise

    def _concat_rowids_algo_result(table, dirty_parent_rowids, proptup_gen, config_rowid):
        # TODO: generalize to all external data that needs to be written
        # explicitly
        extern_fname_list = table._get_extern_fnames(dirty_parent_rowids, config_rowid)
        extern_dpath = table._get_extern_dpath()
        ut.ensuredir(extern_dpath, verbose=True or table.depc._debug)
        fpath_list = [join(extern_dpath, fname) for fname in extern_fname_list]
        for parent_rowids, algo_result, extern_fpath in zip(dirty_parent_rowids, proptup_gen, fpath_list):
            try:
                algo_result.save_to_fpath(extern_fpath)
                yield parent_rowids + (config_rowid,) + (extern_fpath,)
            except Exception as ex:
                ut.printex(ex, 'cat2 error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise

    def _yeild_algo_result(table, dirty_parent_rowids, proptup_gen, config_rowid):
        # TODO: generalize to all external data that needs to be written
        # explicitly
        extern_fname_list = table._get_extern_fnames(dirty_parent_rowids, config_rowid)
        extern_dpath = table._get_extern_dpath()
        ut.ensuredir(extern_dpath, verbose=True or table.depc._debug)
        fpath_list = [join(extern_dpath, fname) for fname in extern_fname_list]
        for parent_rowids, algo_result, extern_fpath in zip(dirty_parent_rowids, proptup_gen, fpath_list):
            yield parent_rowids, config_rowid, algo_result, extern_fpath

    def _save_algo_result(table, dirty_params_chunk):
        for parent_rowids, config_rowid, algo_result, extern_fpath in dirty_params_chunk:
            try:
                algo_result.save_to_fpath(extern_fpath, True)
                yield parent_rowids + (config_rowid,) + (extern_fpath,)
            except Exception as ex:
                ut.printex(ex, 'cat2 error', keys=[
                    'config_rowid', 'data_cols', 'parent_rowids'])
                raise

    def _get_extern_dpath(table):
        cache_dpath = table.depc.cache_dpath
        extern_dname = 'extern_' + table.tablename
        extern_dpath = join(cache_dpath, extern_dname)
        return extern_dpath

    def _get_extern_fnames(table, parent_rowids, config_rowid):
        # TODO: respect request objects
        # Only applies to algorithm tables
        config_hashid = table.get_config_hashid([config_rowid])[0]
        fmtstr = table.tablename + '_id={rowids}_{config_hashid}{ext}'
        fname_list = [fmtstr.format(rowids='_'.join(list(map(str, rowids))),
                                    config_hashid=config_hashid, ext='.cPkl')
                      for rowids in parent_rowids]
        return fname_list

    def _add_dirty_rows(table, dirty_parent_rowids, config_rowid, isdirty_list, config,
                        verbose=True):
        """ Does work of adding dirty rowids """
        try:
            # CALL EXTERNAL PREPROCESSING / GENERATION FUNCTION
            if table.isalgo:
                # HACK: config here is a request
                request = config
                #subreq = request.shallow_copy # TODO
                subreq = request.shallowcopy(qmask=isdirty_list)
                proptup_gen = table.preproc_func(table.depc, subreq)
                #dirty_params_iter = table._concat_rowids_algo_result(
                #    dirty_parent_rowids, proptup_gen, config_rowid)
                dirty_params_iter = table._yeild_algo_result(
                    dirty_parent_rowids, proptup_gen, config_rowid)
            else:
                args = zip(*dirty_parent_rowids)
                if table._asobject:
                    # Convinience
                    args = [table.depc.get_obj(parent, rowids)
                            for parent, rowids in zip(table.parents, args)]
                proptup_gen = table.preproc_func(table.depc, *args,
                                                 config=config)
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
                'table',
                'table.parents',
                'parent_rowids',
                'config',
                'args',
                'config_rowid',
                'dirty_parent_rowids',
                'table.preproc_func'
            ])
            raise

    def add_rows_from_parent(table, parent_rowids, config=None, verbose=True,
                             _debug=None):
        """
        Lazy addition
        """
        if _debug is None:
            _debug = table.depc._debug
        # Get requested configuration id
        config_rowid = table.get_config_rowid(config)
        # Find leaf rowids that need to be computed
        initial_rowid_list = table._get_rowid(parent_rowids, config=config)
        if table.depc._debug:
            print('[deptbl.add] initial_rowid_list = %s' %
                  (ut.trunc_repr(initial_rowid_list),))
            print('[deptbl.add] config_rowid = %r' % (config_rowid,))
        # Get corresponding "dirty" parent rowids
        isdirty_list = ut.flag_None_items(initial_rowid_list)
        dirty_parent_rowids = ut.compress(parent_rowids, isdirty_list)
        num_dirty = len(dirty_parent_rowids)
        num_total = len(parent_rowids)

        if num_dirty > 0:
            with ut.Indenter('[ADD]', enabled=_debug):
                if verbose or _debug:
                    fmtstr = ('[deptbl.add] adding %d / %d new props to %r '
                              'for config_rowid=%r')
                    print(fmtstr % (num_dirty, num_total, table.tablename,
                                    config_rowid))
                print("ADD DIRTY")
                table._add_dirty_rows(dirty_parent_rowids, config_rowid,
                                      isdirty_list, config)
                # Get correct order, now that everything is clean in the database
                print("GET ROWID")
                rowid_list = table._get_rowid(parent_rowids,
                                                            config=config)
        else:
            rowid_list = initial_rowid_list
        if _debug:
            print('[deptbl.add] rowid_list = %s' %
                  (ut.trunc_repr(rowid_list),))
        return rowid_list

    def get_rowid(table, parent_rowids, config=None, ensure=True, eager=True,
                  nInput=None, recompute=False):
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
        if table.depc._debug:
            print('[deptbl.get_rowid] Lookup %s rowids from superkey with %d parents' % (
                table.tablename, len(parent_rowids)))
            print('[deptbl.get_rowid] config = %r' % (config,))
            print('[deptbl.get_rowid] ensure = %r' % (ensure,))
        #rowid_list = parent_rowids
        #return rowid_list

        if recompute:
            # get existing rowids, delete them, recompute the request
            rowid_list = table._get_rowid(parent_rowids, config=config,
                                          eager=eager, nInput=nInput)
            table.delete_rows(rowid_list)
            rowid_list = table.add_rows_from_parent(parent_rowids, config=config)
        elif ensure:
            rowid_list = table.add_rows_from_parent(parent_rowids, config=config)
        else:
            rowid_list = table._get_rowid(
                parent_rowids, config=config, eager=eager, nInput=nInput)
        return rowid_list

    def _get_rowid(table, parent_rowids, config=None, eager=True, nInput=None):
        """
        equivalent to get_rowid except ensure is constrained to be False.
        """
        colnames = (table.rowid_colname,)
        config_rowid = table.get_config_rowid(config=config)
        if table.depc._debug:
            print('_get_rowid')
            print('_get_rowid table.tablename = %r ' % (table.tablename,))
            print('_get_rowid parent_rowids = %s' % (ut.trunc_repr(parent_rowids)))
            print('_get_rowid config = %s' % (config))
            print('_get_rowid table.rowid_colname = %s' % (table.rowid_colname))
            print('_get_rowid config_rowid = %s' % (config_rowid))
        and_where_colnames = table.superkey_colnames
        params_iter = (rowids + (config_rowid,) for rowids in parent_rowids)
        params_iter = list(params_iter)
        #print('**params_iter = %r' % (params_iter,))
        rowid_list = table.db.get_where2(table.tablename, colnames, params_iter,
                                         and_where_colnames, eager=eager,
                                         nInput=nInput)
        if table.depc._debug:
            print('_get_rowid rowid_list = %s' % (ut.trunc_repr(rowid_list)))
        return rowid_list

    def delete_rows(table, rowid_list, verbose=None):
        #from dtool.algo.preproc import preproc_feat
        if table.on_delete is not None:
            table.on_delete()
        if ut.NOT_QUIET:
            print('deleting %d rows from %s' % (len(rowid_list), table.tablename))
        # Finalize: Delete table
        table.db.delete_rowids(table.tablename, rowid_list)
        num_deleted = len(ut.filter_Nones(rowid_list))
        return num_deleted

    def get_row_data(table, tbl_rowids, colnames=None, _debug=None,
                     read_extern=True, extra_tries=1):
        """
        colnames = ('mask', 'size')

        FIXME; unpacking is confusing with sql controller

        TODO: Clean up and allow for eager=False
        """
        _debug = table.depc._debug if _debug is None else _debug
        if _debug:
            print(('[deptbl.get_row_data] Get col of tablename=%r, colnames=%r '
                   'with tbl_rowids=%s') %
                  (table.tablename, colnames, ut.trunc_repr(tbl_rowids)))
        try:
            request_unpack = False
            if colnames is None:
                resolved_colnames = table.data_colnames
                #table._internal_data_colnames
            else:
                if isinstance(colnames, six.string_types):
                    request_unpack = True
                    resolved_colnames = (colnames,)
                else:
                    resolved_colnames = colnames
            if _debug:
                print('[deptbl.get_row_data] resolved_colnames = %r' %
                      (resolved_colnames,))

            eager = True
            nInput = None

            total = 0
            intern_colnames = []
            extern_resolve_colxs = []
            nesting_xs = []

            for c in resolved_colnames:
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
            if _debug:
                print('[deptbl.get_row_data] flat_intern_colnames = %r' %
                      (flat_intern_colnames,))

            # do sql read
            # FIXME: understand unpack_scalars and keepwrap
            raw_prop_list = table.get_internal_columns(
                tbl_rowids, flat_intern_colnames, eager, nInput,
                unpack_scalars=True, keepwrap=True)
            # unpack_scalars=not
            # request_unpack)
            # print('depth(raw_prop_list) = %r' % (ut.depth_profile(raw_prop_list),))
            #import utool
            #utool.embed()

            prop_listT = list(zip(*raw_prop_list))
            for extern_colx, read_func in extern_resolve_colxs:
                data_list = []
                failed_list = []
                if _debug:
                    print('[deptbl.get_row_data] read_func = %r' % (read_func,))
                for uri in prop_listT[extern_colx]:
                    try:
                        # FIXME: only do this for a localpath
                        uri_full = join(table.depc.cache_dpath, uri)
                        if read_extern:
                            data = read_func(uri_full)
                        else:
                            ut.assertpath(uri_full)
                            data = uri_full
                    except Exception as ex:
                        ut.printex(ex, 'failed to load external data',
                                   iswarning=False,
                                   keys=[
                                       'extra_tries',
                                       'uri',
                                       'uri_full',
                                       (exists, 'uri_full'),
                                       'read_func'])
                        #raise
                        if extra_tries == 0:
                            raise
                        failed_list.append(True)
                        data = None
                    else:
                        failed_list.append(False)
                    data_list.append(data)
                if any(failed_list):
                    # FIXME: should directly recompute the data in the rows
                    # rather than deleting the rowids.  Need the parent ids and
                    # config to do that.
                    failed_rowids = ut.compress(tbl_rowids, failed_list)
                    table.delete_rows(failed_rowids)
                    raise Exception('Non existant data on disk. Need to recompute rows')
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
                'resolved_colnames',
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
