# -*- coding: utf-8 -*-
"""
implicit version of dependency cache from templates/template_generator
"""
from __future__ import absolute_import, division, print_function, unicode_literals
#from ibeis import constants as const
import utool as ut
import numpy as np
import uuid
import six
import functools
from ibeis.control.SQLDatabaseControl import SQLDatabaseController


CONFIG_TABLE = '_CONFIG_'
CONFIG_ROWID = 'config_rowid'


TYPE_TO_SQLTYPE = {
    np.ndarray: 'NDARRAY',
    uuid.UUID: 'UUID',
    int: 'INTEGER',
    str: 'TEXT',
}


if six.PY2:
    TYPE_TO_SQLTYPE[six.text_type] = 'TEXT'


#python -m ibeis.templates.template_generator --key feat --modfname={autogen_modname}

def testdata_depcache():
    r"""
    CommandLine:
        python -m ibeis.depends_cache --exec-testdata_depcache --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.depends_cache import *  # NOQA
        >>> depc = testdata_depcache()
        >>> import networkx as netx
        >>> import plottool as pt
        >>> graph = depc.make_digraph()
        >>> pos = netx.pydot_layout(graph, prog='dot')
        >>> pt.ensure_pylab_qt4()
        >>> pt.figure()
        >>> ax = pt.gca()
        >>> netx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=1100)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    preproc = ut.DynStruct()
    preproc.preproc_chip = None
    preproc.preproc_probchip = None
    preproc.preproc_keypoint = None
    preproc.preproc_descriptor = None
    preproc.preproc_fgweight = None
    preproc.preproc_notch = None

    depc = DependencyCache(root_tablename='annot')

    depc.register_property(
        tablename='chip',
        parents=['annot'],
        colnames=['size', 'uri'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
        preproc_func=preproc.preproc_chip,
    )
    depc.register_property(
        'probchip',
        ['annot'],
        ['size', 'uri'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
        preproc_func=preproc.preproc_probchip,
    )
    depc.register_property(
        'keypoint',
        ['chip'],
        ['kpts', 'num'],
        [np.ndarray, int],
        preproc_func=preproc.preproc_keypoint,
        docstr='''
        Used to store individual chip features (ellipses)
        ''',
    )
    depc.register_property(
        'descriptor',
        ['keypoint'],
        ['vecs'],
        [np.ndarray],
        preproc_func=preproc.preproc_descriptor,
    )
    depc.register_property(
        'fgweight',
        ['keypoint', 'probchip'],
        ['fgweight'],
        [np.ndarray],
        preproc_func=preproc.preproc_fgweight,
    )
    depc.register_property(
        'notch',
        ['annot'],
        ['notchdata'],
        preproc_func=preproc.preproc_notch,
    )

    table = depc.register_property(
        'spam',
        ['fgweight', 'chip', 'keypoint'],
        ['spam', 'eggs', 'size', 'uuid', 'vector', 'text_fpath'],
        [str, int, (int, int), uuid.UUID, np.ndarray, ('extern', ut.readfrom)],
        preproc_func=None,
        docstr='I dont like spam',
    )

    #tablename = 'fgweight'
    #print('fgweight_path =\n%s' % (ut.repr3(depc.get_dependencies(tablename)),))
    #print('keypoint =\n%s' % (ut.repr3(depc.get_dependencies('keypoint')),))
    #print('descriptor =\n%s' % (ut.repr3(depc.get_dependencies('descriptor')),))
    #print('spam =\n%s' % (ut.repr3(depc.get_dependencies('spam')),))

    #depc.get_rowids('descriptor', [1, 2, 3])
    #depc.get_rowids('spam', [1, 2, 3])

    print(ut.repr2(table.get_addtable_kw(), nl=2))

    depc.initialize()

    table.print_schemadef()
    print(table.db.get_schema_current_autogeneration_str())

    return depc


class DependencyCache(object):
    def __init__(depc, root_tablename=None, cache_path='.'):
        depc.root_tablename = root_tablename
        depc.cachetable_dict = {}
        depc.fname_to_db = {}

    def initialize(depc):
        print('INITIALIZE DEPCACHE')
        for fname in depc.fname_to_db.keys():
            print('fname = %r' % (fname,))
            depc.fname_to_db[fname] = SQLDatabaseController(fpath=fname, simple=True)
        for table in depc.cachetable_dict.values():
            table.initialize()

    def make_digraph(depc):
        nodes = list(depc.cachetable_dict.keys())
        edges = [(parent, key) for key, table in  depc.cachetable_dict.items() for parent in table.parents]
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def register_property(depc, tablename, parents=None, colnames=None,
                          coltypes=None, preproc_func=None, docstr='no docstr',
                          fname=None):
        if coltypes is None:
            coltypes = [np.ndarray] * len(colnames)
        if fname is None:
            fname = ':memory:'
        depc.fname_to_db[fname] = None
        table = DependencyCacheTable(
            depc=depc,
            parent_tablenames=parents,
            tablename=tablename,
            docstr=docstr,
            data_colnames=colnames,
            data_coltypes=coltypes,
            preproc_func=preproc_func,
            fname=fname,
        )
        depc.cachetable_dict[tablename] = table
        return table

    def _custom_str(depc):
        typestr = depc.__class__.__name__
        infostr_ = 'nTables=%d' % len(depc.cachetable_dict)
        custom_str = '<%s(%s) %s at %s>' % (typestr, depc.root_tablename, infostr_, hex(id(depc)))
        return custom_str

    @property
    def root(depc):
        return depc.root_tablename

    def __repr__(depc):
        return depc._custom_str()

    def __str__(depc):
        return depc._custom_str()

    def __getitem__(depc, key):
        return depc.cachetable_dict[key]

    def delete_property(depc, tablename, root_rowids, config=None):
        rowid_list = depc.get_rowids(root_rowids, config=config, ensure=False)
        table = depc[tablename]
        return table.delete_rows(rowid_list)

    def get_property(depc, tablename, root_rowids, colnames=None, config=None):
        native_rowids = depc.get_rowids(depc, tablename, root_rowids, config)
        table = depc[tablename]
        if colnames is None:
            colnames = table.colnames
        prop_list = table.get_native_prop(native_rowids, colnames)
        return prop_list

    def get_rowids(depc, tablename, root_rowids, config=None, ensure=True, eager=True, nInput=None):
        print('GET ROWIDS %s ' % (tablename,))

        dependency_levels = depc.get_dependencies(tablename)
        rowid_dict = {depc.root: root_rowids}
        print('dependency_levels = %s' % (ut.repr3(dependency_levels),))

        for level_keys in dependency_levels[1:]:
            print('* level_keys %s ' % (level_keys,))
            for key in level_keys:
                print('  * key = %r' % (key,))
                table = depc[key]
                parent_rowids = ut.dict_take(rowid_dict, table.parents)
                child_rowids = table.get_rowid_from_superkey(parent_rowids, config, eager, nInput)
                rowid_dict[key] = child_rowids

        rowid_list = rowid_dict[tablename]
        return rowid_list

    def get_dependencies(depc, tablename):
        """
        gets level dependences from root to tablename
        """
        print('GET DEPENDS %s ' % (tablename,))

        def dict_depth(dict_, accum=0):
            if not isinstance(dict_, dict):
                return accum
            return max([dict_depth(val, accum + 1) for key, val in dict_.items()])

        def path_to_root(tablename, root):
            if tablename == root:
                return None
            table = depc.cachetable_dict[tablename]
            return {parent: path_to_root(parent, root) for parent in table.parent_tablenames}

        def reverse_path(dict_, root):
            def get_allkeys(dict_):
                if not isinstance(dict_, dict):
                    return []
                subkeys = [[key] + get_allkeys(val) for key, val in dict_.items()]
                return ut.unique_ordered(ut.flatten(subkeys))
            # Hacky but illustrative
            allkeys = get_allkeys(dict_)
            mat = np.zeros((len(allkeys), len(allkeys)))
            for key in allkeys:
                if key != root:
                    for parent in depc.cachetable_dict[key].parent_tablenames:
                        mat[allkeys.index(parent)][allkeys.index(key)] = 1
            def traverse_path(start, end, seen_=None):
                if seen_ is None:
                    seen_ = set([])
                index = allkeys.index(start)
                sub_indexes = np.where(mat[index])[0]
                if len(sub_indexes) > 0:
                    subkeys_ = ut.take(allkeys, sub_indexes)
                    subkeys = [subkey for subkey in subkeys_
                               if subkey not in seen_]
                    for sk in subkeys:
                        seen_.add(sk)
                    if len(subkeys) > 0:
                        return {subkey: traverse_path(subkey, end, seen_)
                                for subkey in subkeys}
                return None
            end = None
            return {root: traverse_path(root, end, seen_=None)}
        to_root = {tablename: path_to_root(tablename, depc.root_tablename)}
        print('to_root = %r' % (to_root,))
        from_root = reverse_path(to_root, depc.root_tablename)
        print('from_root = %r' % (from_root,))

        def get_levels(dict_, n=0, levels=None):
            if levels is None:
                levels = [[] for _ in range(dict_depth(dict_))]
            if dict_ is None:
                return []
            for key in dict_.keys():
                levels[n].append(key)
            for val in dict_.values():
                get_levels(val, n + 1, levels)
            return levels

        dependency_levels = get_levels(from_root)

        return dependency_levels


class DependencyCacheTable(object):

    def __init__(self, depc=None, parent_tablenames=None, tablename=None,
                 data_colnames=None, data_coltypes=None, preproc_func=None,
                 docstr='no docstr', fname=None):

        self.fpath_to_db = {}

        self.parent_tablenames = parent_tablenames
        self.tablename = tablename
        self.data_colnames = tuple(data_colnames)
        self.data_coltypes = data_coltypes
        self.preproc_func = preproc_func

        self._internal_data_colnames = []
        self._internal_data_coltypes = []
        self.sqldb_fpath = None
        self.extern_read_funcs = []
        self.docstr = docstr
        self.fname = fname
        self.depc = depc
        self.db = None
        self._update_internals()

    def _update_internals(self):
        extern_read_funcs = {}
        internal_data_colnames = []
        internal_data_coltypes = []

        for colname, coltype in zip(self.data_colnames, self.data_coltypes):
            if isinstance(coltype, tuple):
                if coltype[0] == 'extern':
                    read_func = coltype[1]
                    extern_read_funcs[colname] = read_func
                    internal_data_colnames.append(colname + '_extern_uri')
                    internal_data_coltypes.append(TYPE_TO_SQLTYPE[str])
                else:
                    for count, dimtype in enumerate(coltype):
                        internal_data_colnames.append('%s_%d' % (colname, count))
                        internal_data_coltypes.append(TYPE_TO_SQLTYPE[dimtype])
            else:
                internal_data_colnames.append(colname)
                internal_data_coltypes.append(TYPE_TO_SQLTYPE[coltype])

        assert len(set(internal_data_colnames)) == len(internal_data_colnames)
        assert len(internal_data_coltypes) == len(internal_data_colnames)
        self._internal_data_colnames = internal_data_colnames
        self._internal_data_coltypes = internal_data_coltypes

    def get_addtable_kw(self):
        primary_coldef = [(self.rowid_colname, 'INTEGER PRIMARY KEY')]
        parent_coldef = [(key, 'INTEGER NOT NULL') for key in self.parent_rowid_colnames]
        config_coldef = [(CONFIG_ROWID, 'INTEGER DEFAULT 0')]
        internal_data_coldef = list(zip(self._internal_data_colnames,
                                        self._internal_data_coltypes))

        coldef_list = primary_coldef + parent_coldef + config_coldef + internal_data_coldef
        add_table_kw = ut.odict([
            ('tablename', self.tablename,),
            ('coldef_list', coldef_list,),
            ('docstr', self.docstr,),
            ('superkeys', [self.superkey_colnames],),
            ('dependson', self.parents),
        ])
        return add_table_kw

    def initialize(self):
        self.db = self.depc.fname_to_db[self.fname]
        self.db.add_table(**self.get_addtable_kw())

    def print_schemadef(self):
        print('\n'.join(self.db.get_table_autogen_str(self.tablename)))

    def _get_all_rowids(self):
        pass

    @property
    def parents(self):
        return self.parent_tablenames

    @property
    def rowid_colname(self):
        return self.tablename + '_rowid'

    @property
    def parent_rowid_colnames(self):
        #return tuple([self.depc[parent].rowid_colname for parent in self.parents])
        return tuple([parent + '_rowid' for parent in self.parents])

    @property
    def superkey_colnames(self):
        return self.parent_rowid_colnames + (CONFIG_ROWID,)

    @property
    def _table_colnames(self):
        return self.superkey_colnames + self.data_colnames

    def _custom_str(self):
        typestr = self.__class__.__name__
        custom_str = '<%s(%s) at %s>' % (typestr, self.tablename, hex(id(self)))
        return custom_str

    def __repr__(self):
        return self._custom_str()

    def __str__(self):
        return self._custom_str()

    def get_config_rowids(self, db, config=None):
        if config is not None:
            config_hashid = config.get('feat_cfgstr')
            assert config_hashid is not None
            # TODO store config_rowid in qparams
        else:
            config_hashid = db.cfg.feat_cfg.get_cfgstr()
        config_rowid = self.add_config(db, config_hashid)
        return config_rowid

    # ---------------------------
    # --- CONFIGURATION TABLE ---
    # ---------------------------

    def get_config_rowid_from_hashid(db, config_hashid_list):
        config_rowid_list = db.get(CONFIG_TABLE, ('config_rowid',), config_hashid_list,
                                   id_colname='config_hashid')
        return config_rowid_list

    def add_config(self, db, config_hashid_list):
        get_rowid_from_superkey = self.get_config_rowid_from_hashid
        config_rowid_list = db.add_cleanly(CONFIG_TABLE, ('config_hashid',),
                                           config_hashid_list,
                                           get_rowid_from_superkey)
        return config_rowid_list

    # ----------------------
    # --- GETTERS NATIVE ---
    # ----------------------

    def add_rows_from_parent(self, parent_rowids, config=None, verbose=True, return_num_dirty=False):
        """
        add_chip_feat
        """
        # Get requested configuration id
        config_rowid = self.get_config_rowid(config)
        # Find leaf rowids that need to be computed
        initial_rowid_list = self._get_rowid_from_superkey(
            self, parent_rowids, config=config)
        # Get corresponding "dirty" parent rowids
        isdirty_list = ut.flag_None_items(initial_rowid_list)
        dirty_parent_rowids_list = ut.compress(parent_rowids, isdirty_list)
        num_dirty = len(dirty_parent_rowids_list)
        num_total = len(parent_rowids)
        if num_dirty > 0:
            if verbose:
                fmtstr = 'adding %d / %d new props for config_rowid=%r'
                print(fmtstr % (num_dirty, num_total, config_rowid))
            get_rowid_from_superkey = functools.partial(
                self._get_rowid_from_superkey, config=config)
            # CALL EXTERNAL PREPROCESSING / GENERATION FUNCTION
            proptup_gen = self.preproc_func(dirty_parent_rowids_list,
                                            config=config)
            dirty_params_iter = (
                parent_rowids + (config_rowid,) + data_cols
                for parent_rowids, data_cols in
                zip(dirty_parent_rowids_list, proptup_gen)
            )
            CHUNKED_ADD = False
            if CHUNKED_ADD:
                for dirty_params_chunk in ut.ichunks(dirty_params_iter,
                                                     chunksize=128):
                    self.db._add(self.tablename, self._table_colnames,
                                 dirty_params_chunk,
                                 nInput=len(dirty_params_chunk))
            else:
                nInput = num_dirty
                self.db._add(self.tablename, self._table_colnames,
                             dirty_params_iter, nInput=nInput)
            # Now that the dirty params are added get the correct order of rowids
            rowid_list = get_rowid_from_superkey(parent_rowids)
        else:
            rowid_list = initial_rowid_list
        if return_num_dirty:
            return rowid_list, num_dirty

    def get_rowid_from_superkey(self, parent_rowids, config=None, ensure=True,
                                eager=True, nInput=None, recompute=False):
        r"""
        Args:
            parent_rowids (list): list of tuples with the parent rowids as the
                value of each tuple
            config (None): (default = None)
            ensure (bool):  eager evaluation if True (default = True)
            eager (bool): (default = True)
            nInput (None): (default = None)
            recompute (bool): (default = False)

        Returns:
            list: rowid_list

        get_chip_feat_rowid

        get feat rowids of chip under the current state configuration
        if ensure is True, this function is equivalent to add_chip_feats
        """
        assert len(parent_rowids) == len(self.parents)
        print('Lookup %s rowids from superkey with %d parents' % (
            self.tablename, len(parent_rowids)))
        rowid_list = parent_rowids
        return rowid_list

        if recompute:
            # get existing rowids, delete them, recompute the request
            rowid_list = self._get_rowid_from_superkey(
                parent_rowids, config=config, eager=eager, nInput=nInput)
            self.delete_rows(self, rowid_list)
            rowid_list = self.add_chip_feat(self, parent_rowids, config=config)
        elif ensure:
            rowid_list = self.add_chip_feat(self, parent_rowids, config=config)
        else:
            rowid_list = self._get_rowid_from_superkey(
                parent_rowids, config=config, eager=eager, nInput=nInput)
        return rowid_list

    def _get_rowid_from_superkey(self, parent_rowids, config=None, eager=True, nInput=None):
        """
        equivalent to get_rowid_from_superkey except ensure is constrained to
        be False.  Also you save a stack frame because get_chip_feat_rowid just
        calls this function if ensure is False
        """
        colnames = (self.rowid_colname,)
        config_rowid = self.get_feat_config_rowid(config=config)
        and_where_colnames = self.superkey_colnames
        params_iter = (rowids + (config_rowid,) for rowids in parent_rowids)
        rowid_list = self.db.get_where2(self.tablename, colnames, params_iter,
                                        and_where_colnames, eager=eager,
                                        nInput=nInput)
        return rowid_list

    def delete_rows(self, rowid_list):
        #from ibeis.algo.preproc import preproc_feat
        if self.on_delete is not None:
            self.on_delete()
        if ut.VERBOSE:
            print('deleting %d rows' % len(rowid_list))
        # Finalize: Delete self
        self.db.delete_rowids(self.tablename, rowid_list)
        num_deleted = len(ut.filter_Nones(rowid_list))
        return num_deleted

    def get_external_columns(self, rowid_list, colnames, eager=True, nInput=None):
        internal_uris = self.get_internal_columns(colnames)
        internal_uris_T = zip(*internal_uris)
        return [[self.read_func(uri) for uri in uris] for uris in internal_uris_T]

    def get_internal_columns(self, rowid_list, colnames, eager=True, nInput=None):
        feat_nFeat_list = self.db.get(
            self.tablename, colnames, rowid_list,
            id_colname=self.rowid_colname, eager=eager, nInput=nInput)
        return feat_nFeat_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.depends_cache
        python -m ibeis.depends_cache --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
