# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m dtool.example_depcache --exec-dummy_example_depcacahe --show
    python -m dtool.depcache_control --exec-make_graph --show
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import uuid
from os.path import join, dirname
from six.moves import zip
from wbia.dtool import depcache_control
from wbia import dtool


if False:
    # Example of global registration
    DUMMY_ROOT_TABLENAME = 'dummy_annot'
    _depcdecors = depcache_control.make_depcache_decors(DUMMY_ROOT_TABLENAME)
    register_preproc = _depcdecors['preproc']
    register_subprop = _depcdecors['subprop']

    @register_preproc(
        tablename='dummy',
        parents=[DUMMY_ROOT_TABLENAME],
        colnames=['data'],
        coltypes=[str],
    )
    def dummy_global_preproc_func(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Requesting global dummy ')
        for rowid in parent_rowids:
            yield 'dummy'


class DummyKptsConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('adapt_shape', True),
            ut.ParamInfo('adapt_angle', False),
        ]


class DummyIndexerConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('index_method', 'single'),
        ut.ParamInfo('trees', 8),
        ut.ParamInfo('algorithm', 'kdtree'),
    ]
    # FIXME: triggers duplicate error
    # _sub_config_list = [
    #    DummyKptsConfig
    # ]


class DummyNNConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('K', 4),
            ut.ParamInfo('Knorm', 1),
            ut.ParamInfo('checks', 800),
            ut.ParamInfo('version', 1),
        ]


class DummySVERConfig(dtool.Config):
    _param_info_list = [ut.ParamInfo('sver_on', True), ut.ParamInfo('xy_thresh', 0.01)]


class DummyChipConfig(dtool.Config):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.example_depcache import *  # NOQA
        >>> cfg = DummyChipConfig()
        >>> cfg.dim_size = 700
        >>> cfg.histeq = True
        >>> print(cfg)
        >>> cfg.histeq = False
        >>> print(cfg)
    """

    _param_info_list = [
        ut.ParamInfo(
            'resize_dim', 'width', valid_values=['area', 'width', 'heigh', 'diag']
        ),
        ut.ParamInfo('dim_size', 500, 'sz'),
        ut.ParamInfo('preserve_aspect', True),
        ut.ParamInfo('histeq', False, hideif=False),
        ut.ParamInfo('ext', '.png'),
        ut.ParamInfo('version', 0),
    ]


class ProbchipConfig(dtool.Config):
    """
    CommandLine:
        python -m dtool.example_depcache --exec-ProbchipConfig --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dtool.depcache_control import *  # NOQA
        >>> from wbia.dtool.example_depcache import testdata_depc
        >>> depc = testdata_depc()
        >>> table = depc['probchip']
        >>> exec(ut.execstr_funckw(table.get_rowid), globals())
        >>> config = table.configclass(testerror=True)
        >>> root_rowids = [1, 2, 3]
        >>> parent_rowids = list(zip(root_rowids))
        >>> proptup_gen = list(table.preproc_func(depc, root_rowids, config))
        >>> pc_rowids = depc.get_rowids('probchip', root_rowids, config)
        >>> prop_list2 = depc.get('probchip', root_rowids, config=config, read_extern=False)
        >>> print(prop_list2)
        >>> #depc.new_request('probchip', [1, 2, 3])
        >>> fg_rowids = depc.get_rowids('fgweight', root_rowids, config)
        >>> fg = depc.get('fgweight', root_rowids, config=config)
        >>> #############
        >>> config = table.configclass(testerror=False)
        >>> root_rowids = [1, 2, 3]
        >>> parent_rowids = list(zip(root_rowids))
        >>> proptup_gen = list(table.preproc_func(depc, root_rowids, config))
        >>> pc_rowids2 = depc.get_rowids('probchip', root_rowids, config)
        >>> prop_list2 = depc.get('probchip', root_rowids, config=config, read_extern=False)
        >>> print(prop_list2)
        >>> #depc.new_request('probchip', [1, 2, 3])
        >>> fg_rowids2 = depc.get_rowids('fgweight', root_rowids, config)
    """

    _param_info_list = [
        ut.ParamInfo('testerror', False, hideif=False),
        ut.ParamInfo('ext', '.png', hideif='.png'),
    ]


class DummyVsManyConfig(dtool.Config):
    # Different pipeline components can go here as well as dependencies
    # that were not explicitly enumerated in the tree structure
    _param_info_list = [
        # ut.ParamInfo('score_method', 'csum'),
        # should this be the only thing here?
        # ut.ParamInfo('daids', None),
        ut.ParamInfo('distinctiveness_model', None),
        ut.ParamInfo('version', 2),
    ]
    _sub_config_list = [
        # I guess different annots might want different configs ...
        DummyChipConfig,
        DummyKptsConfig,
        DummyIndexerConfig,
        DummyNNConfig,
        DummySVERConfig,
    ]


class DummyVsOneConfig(dtool.Config):
    def get_sub_config_list(self):
        # Different pipeline components can go here
        # as well as dependencies that were not
        # explicitly enumerated in the tree structure
        return [
            # I guess different annots might want different configs ...
            DummyChipConfig,
            DummyKptsConfig,
            DummyIndexerConfig,
            DummyNNConfig,
            DummySVERConfig,
        ]

    def get_param_info_list(self):
        return [
            ut.ParamInfo('distinctiveness_model', None),
            ut.ParamInfo('ratio_thresh', None),
        ]


class DummyVsOneRequest(dtool.VsOneSimilarityRequest):
    pass


class DummyVsManyRequest(dtool.VsManySimilarityRequest):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.example_depcache import *  # NOQA
        >>> algo_config = DummyVsManyConfig()
        >>> print(algo_config)
    """

    pass


class DummyAnnotMatch(dtool.MatchResult):
    pass


class DummyVsOneMatch(dtool.AlgoResult, ut.NiceRepr):
    def __init__(self):
        self.score = None
        self.qaid = None
        self.daid = None
        self.fm = None

    def __nice__(self):
        return '(%d-vs-%d) %.2f' % (self.qaid, self.daid, self.score)


def testdata_depc(fname=None):
    """
    Example of local registration
    """

    from wbia import dtool
    import vtool as vt

    gpath_list = ut.lmap(ut.grab_test_imgpath, ut.get_valid_test_imgkeys(), verbose=False)

    dummy_root = 'dummy_annot'

    def get_root_uuid(aid_list):
        return ut.lmap(ut.hashable_to_uuid, aid_list)

    # put the test cache in the dtool repo
    dtool_repo = dirname(ut.get_module_dir(dtool))
    cache_dpath = join(dtool_repo, 'DEPCACHE')

    depc = dtool.DependencyCache(
        root_tablename=dummy_root,
        default_fname=fname,
        cache_dpath=cache_dpath,
        get_root_uuid=get_root_uuid,
        # root_asobject=root_asobject,
        use_globals=False,
    )

    @depc.register_preproc(
        tablename='chip',
        parents=[dummy_root],
        colnames=['size', 'chip'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
        configclass=DummyChipConfig,
    )
    def dummy_preproc_chip(depc, annot_rowid_list, config=None):
        """
        TODO: Infer properties from docstr?

        Args:
            depc (dtool.DependencyCache):
            annot_rowid_list (list): list of annot rowids
            config (dict): config dictionary

        Returns:
            tuple : ((int, int), ('extern', vt.imread))
        """
        if config is None:
            config = {}
        # Demonstates using asobject to get input to function as a dictionary
        # of properties
        # for annot in annot_list:
        # print('[preproc] Computing chips of aid=%r' % (aid,))
        print('[preproc] Computing chips')
        for aid in annot_rowid_list:
            # aid = annot['aid']
            # chip_fpath = annot['gpath']
            chip_fpath = gpath_list[aid]
            # w, h = vt.image.open_image_size(chip_fpath)
            chip = vt.imread(chip_fpath)
            size = vt.get_size(chip)
            # size = (w, h)
            print('Dummpy preproc chip yeilds')
            print('* chip_fpath = %r' % (chip_fpath,))
            print('* size = %r' % (size,))
            # yield size, chip_fpath
            yield size, chip

    @depc.register_preproc(
        'probchip',
        [dummy_root],
        ['size', 'probchip'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite, '.png')],
        configclass=ProbchipConfig,
    )
    def dummy_preproc_probchip(depc, root_rowids, config):
        print('[preproc] Computing probchip')
        for rowid in root_rowids:
            if config['testerror']:
                if rowid % 2 == 0:
                    # Test error yeilds None on even rowids
                    yield None
                    continue
            rng = np.random.RandomState(rowid)
            probchip = rng.randint(0, 255, size=(64, 64))
            # probchip = np.zeros((64, 64))
            size = (rowid, rowid)
            yield size, probchip

    @depc.register_preproc(
        'keypoint',
        ['chip'],
        ['kpts', 'num'],
        [np.ndarray, int],
        # default_onthefly=True,
        configclass=DummyKptsConfig,
        docstr='Used to store individual chip features (ellipses)',
    )
    def dummy_preproc_kpts(depc, chip_rowids, config=None):
        if config is None:
            config = {}
        print('config = %r' % (config,))
        adapt_shape = config['adapt_shape']
        print('[preproc] Computing kpts')

        ut.assert_all_not_None(chip_rowids, 'chip_rowids')
        # This is in here to attempt to trigger a failure of the chips dont
        # exist and the feature cache is called.
        chip_fpath_list = depc.get_native('chip', chip_rowids, 'chip', read_extern=False)
        print('computing featurse from chip_fpath_list = %r' % (chip_fpath_list,))

        for rowid in chip_rowids:
            if adapt_shape:
                kpts = np.zeros((7 + rowid, 6)) + rowid
            else:
                kpts = np.ones((7 + rowid, 6)) + rowid
            num = len(kpts)
            yield kpts, num

    @depc.register_preproc(
        'descriptor', ['keypoint'], ['vecs'], [np.ndarray],
    )
    def dummy_preproc_vecs(depc, kp_rowid, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing vecs')
        for rowid in kp_rowid:
            yield np.ones((7 + rowid, 8), dtype=np.uint8) + rowid,

    @depc.register_preproc(
        'fgweight', ['keypoint', 'probchip'], ['fgweight'], [np.ndarray],
    )
    def dummy_preproc_fgweight(depc, kpts_rowid, probchip_rowid, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing fgweight')
        for rowid1, rowid2 in zip(kpts_rowid, probchip_rowid):
            yield np.ones(7 + rowid1),

    @depc.register_preproc(
        tablename='vsmany',
        colnames='annotmatch',
        coltypes=DummyAnnotMatch,
        requestclass=DummyVsManyRequest,
        configclass=DummyVsManyConfig,
    )
    def vsmany_matching(depc, qaids, config=None):
        """
        CommandLine:
            python -m dtool.base --exec-VsManySimilarityRequest
        """
        print('RUNNING DUMMY VSMANY ALGO')
        daids = config.daids
        qaids = qaids

        sver_on = config.dummy_sver_cfg['sver_on']
        kpts_list = depc.get_property('keypoint', list(qaids))  # NOQA
        # dummy_preproc_kpts
        for qaid in qaids:
            dnid_list = [1, 1, 2, 2]
            unique_nids = [1, 2]
            if sver_on:
                annot_score_list = [0.2, 0.2, 0.4, 0.5]
                name_score_list = [0.2, 0.5]
            else:
                annot_score_list = [0.3, 0.3, 0.6, 0.9]
                name_score_list = [0.1, 0.7]
            annot_match = DummyAnnotMatch(
                qaid, daids, dnid_list, annot_score_list, unique_nids, name_score_list
            )
            yield annot_match

    SIMPLE = 0
    if not SIMPLE:

        @depc.register_preproc(
            tablename='chipmask',
            parents=[dummy_root],
            colnames=['size', 'mask'],
            coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
        )
        def dummy_manual_chipmask(depc, parent_rowids, config=None):
            import vtool as vt
            from wbia.plottool import interact_impaint

            mask_dpath = join(depc.cache_dpath, 'ManualChipMask')
            ut.ensuredir(mask_dpath)
            if config is None:
                config = {}
            print('Requesting user defined chip mask')
            for rowid in parent_rowids:
                img = vt.imread(gpath_list[rowid])
                mask = interact_impaint.impaint_mask2(img)
                mask_fpath = join(mask_dpath, 'mask%d.png' % (rowid,))
                vt.imwrite(mask_fpath, mask)
                w, h = vt.get_size(mask)
                yield (w, h), mask_fpath

        @depc.register_preproc(
            'notch', [dummy_root], ['notchdata'], [np.ndarray],
        )
        def dummy_preproc_notch(depc, parent_rowids, config=None):
            if config is None:
                config = {}
            print('[preproc] Computing notch')
            for rowid in parent_rowids:
                yield np.empty(5 + rowid),

        @depc.register_preproc(
            'spam',
            ['fgweight', 'chip', 'keypoint'],
            ['spam', 'eggs', 'size', 'uuid', 'vector', 'textdata'],
            [str, int, (int, int), uuid.UUID, np.ndarray, ('extern', ut.readfrom)],
            docstr='I dont like spam',
        )
        def dummy_preproc_spam(depc, *args, **kwargs):
            config = kwargs.get('config', None)
            if config is None:
                config = {}
            print('[preproc] Computing spam')
            ut.writeto('tmp.txt', ut.lorium_ipsum())
            for x in zip(*args):
                size = (42, 21)
                uuid = ut.get_zero_uuid()
                vector = np.ones(3)
                yield ('spam', 3665, size, uuid, vector, 'tmp.txt')

        @depc.register_preproc(
            'nnindexer',
            ['keypoint*'],
            ['flann'],
            [str],  # [('extern', ut.load_data)],
            configclass=DummyIndexerConfig,
        )
        def dummy_preproc_indexer(depc, parent_rowids_list, config=None):
            print('COMPUTING DUMMY INDEXER')
            # assert len(parent_rowids_list) == 1, 'handles only one indexer'
            for parent_rowids in parent_rowids_list:
                yield (
                    'really cool flann object'
                    + str(config.get_cfgstr())
                    + ' '
                    + str(parent_rowids),
                )

        @depc.register_preproc(
            'notchpair',
            ['notch', 'notch'],
            ['pairscore'],
            [int],  # [('extern', ut.load_data)],
            # configclass=DummyIndexerConfig,
        )
        def dummy_notchpair(depc, n1, n2, config=None):
            print('COMPUTING MULTITEST 1 ')
            # assert len(parent_rowids_list) == 1, 'handles only one indexer'
            for nn1, nn2 in zip(n1, n2):
                yield (nn1 + nn2,)

        @depc.register_preproc(
            'multitest',
            [
                'keypoint',
                'notch',
                'notch',
                'fgweight*',
                'notchpair*',
                'notchpair*',
                'notchpair',
                'nnindexer',
            ],
            ['foo'],
            [str],  # [('extern', ut.load_data)],
            # configclass=DummyIndexerConfig,
        )
        def dummy_multitest(depc, *args, **kwargs):
            print('COMPUTING MULTITEST 1 ')
            # assert len(parent_rowids_list) == 1, 'handles only one indexer'
            for x in zip(args):
                yield ('cool multi object' + str(kwargs) + ' ' + str(x),)

        # TEST MULTISET DEPENDENCIES
        @depc.register_preproc(
            'multitest_score',
            ['multitest'],
            ['score'],
            [int],  # [('extern', ut.load_data)],
            # configclass=DummyIndexerConfig,
        )
        def dummy_multitest_score(depc, parent_rowids, config=None):
            print('COMPUTING DEPENDENCY OF MULTITEST 1 ')
            # assert len(parent_rowids_list) == 1, 'handles only one indexer'
            for parent_rowids in zip(parent_rowids):
                yield (parent_rowids,)

        # TEST MULTISET DEPENDENCIES
        @depc.register_preproc(
            'multitest_score_x',
            ['multitest_score', 'multitest_score'],
            ['score'],
            [int],  # [('extern', ut.load_data)],
            # configclass=DummyIndexerConfig,
        )
        def multitest_score_x(depc, *args, **kwargs):
            raise NotImplementedError('hack')

        # REGISTER MATCHING ALGORITHMS

        @depc.register_preproc(
            tablename='neighbs',
            colnames=['qx2_idx', 'qx2_dist'],
            coltypes=[np.ndarray, np.ndarray],
            parents=['keypoint', 'fgweight', 'nnindexer', 'nnindexer'],
        )
        def neighbs(depc, *args, **kwargs):
            """
            CommandLine:
                python -m dtool.base --exec-VsManySimilarityRequest
            """
            # dummy_preproc_kpts
            for qaid in zip(args):
                yield np.array([qaid]), np.array([qaid])

        @depc.register_preproc(
            tablename='neighbs_score',
            colnames=['qx2_dist'],
            coltypes=[np.ndarray],
            parents=['neighbs'],
        )
        def neighbs_score(depc, *args, **kwargs):
            """
            CommandLine:
                python -m dtool.base --exec-VsManySimilarityRequest
            """
            raise NotImplementedError('hack')

        @depc.register_preproc(
            'vsone',
            [dummy_root, dummy_root],
            ['score', 'match_obj', 'fm'],
            [float, DummyVsOneMatch, np.ndarray],
            requestclass=DummyVsOneRequest,
            configclass=DummyVsOneConfig,
            chunksize=2,
        )
        def compute_vsone_matching(depc, qaids, daids, config):
            """
            CommandLine:
                python -m dtool.base --exec-VsOneSimilarityRequest
            """
            print('RUNNING DUMMY VSONE ALGO')
            for qaid, daid in zip(qaids, daids):
                match = DummyVsOneMatch()
                match.qaid = qaid
                match.daid = daid
                match.fm = np.array([[1, 2], [3, 4]])
                score = match.score = qaid + daid
                yield (score, match, match.fm)

    # table = depc['spam']
    # print(ut.repr2(table.get_addtable_kw(), nl=2))
    depc.initialize()
    # table.print_schemadef()
    # print(table.db.get_schema_current_autogeneration_str())
    return depc


def example_getter_methods(depc, tablename, root_rowids):
    """
    example of different ways to get data
    """
    from wbia import dtool

    print('\n+---')
    print('Running getter example')
    print(' * tablename=%r' % (tablename))
    print(' * root_rowids=%r' % (ut.trunc_repr(tablename)))

    # You can get a reference to data rows using the "root" (dummy_annot) rowids
    # By default, if the data has not been computed, then it will be computed
    # for you. But if you specify ensure=False, None will be returned if the data
    # has not been computed yet.
    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    print('tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)')
    print('tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
    # assert tbl_rowids[0] is None

    # The default is for the data to be computed though. Manaual interactions will
    # launch as necessary.
    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=True)  # NOQA
    print('tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=True)')
    print('tbl_rowids = %s' % (ut.trunc_repr(tbl_rowids),))
    assert tbl_rowids[0] is not None

    # Now the data is cached and will not need to be computed again
    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    assert tbl_rowids[0] is not None

    # Can lookup a table, which can access data directly.  The rowids can be
    # used to lookup data values directly. By default all data in a row is
    # returned.
    table = depc[tablename]
    datas = table.get_row_data(tbl_rowids)  # NOQA

    # But you can also ask for a specific column
    col1 = table.columns[0]
    col1_data = table.get_row_data(tbl_rowids, col1)  # NOQA

    # In the case of external columns:
    if len(table.extern_columns) > 0:
        excol = table.extern_columns[0]
        # you can lookup the value of the external data very simply
        extern_data = table.get_row_data(tbl_rowids, (excol,))  # NOQA
        print('extern_data = table.get_row_data(tbl_rowids, (excol,))')
        print(ut.varinfo_str(extern_data, 'extern_data'))
        # you can lookup the hidden paths as follows
        extern_paths = table.get_row_data(
            tbl_rowids, (excol + dtool.depcache_table.EXTERN_SUFFIX,)
        )  # NOQA
        print(
            'extern_paths = table.get_row_data(tbl_rowids, (excol + dtool.depcache_table.EXTERN_SUFFIX,))'
        )
        print(ut.varinfo_str(extern_paths, 'extern_paths'))

    # But you can also just the root rowids directly. This is the simplest way
    # to access data and really "all you need to know"
    if len(table.columns) > 1:
        col1, col2 = table.columns[0:2]
        datas = depc.get_property(tablename, root_rowids, (col1, col2))  # NOQA

    print('L__')


def test_getters(depc):
    # One input = one output
    chip = depc.get_property('chip', 1, 'chip')  # NOQA
    print('[test] chip.sum() = %r' % (chip.sum(),))

    col_tup_list = depc.get_property('chip', [1], ('size',))
    print('[test] col_tup_list = %r' % (col_tup_list,))

    col_list = depc.get_property('chip', [1], 'size')
    print('[test] col_list = %r' % (col_list,))

    col = depc.get_property('chip', 1, 'size')
    print('[test] col = %r' % (col,))

    cols = depc.get_property('chip', 1, 'size')
    print('[test] cols = %r' % (cols,))

    if False:
        chip_dict = depc.get_obj('chip', 1)

        print('chip_dict = %r' % (chip_dict,))
        for key in chip_dict.keys():
            print(ut.varinfo_str(chip_dict[key], 'chip_dict["%s"]' % (key,)))
        # print('chip_dict["chip"] = %s' % (ut.trunc_repr(chip_dict['chip']),))
        print('chip_dict = %r' % (chip_dict,))


def dummy_example_depcacahe():
    r"""
    CommandLine:
        python -m dtool.example_depcache --exec-dummy_example_depcacahe

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.example_depcache import *  # NOQA
        >>> depc = dummy_example_depcacahe()
    """
    fname = None
    # fname = 'dummy_default_depcache'
    fname = ':memory:'

    depc = testdata_depc(fname)

    tablename = 'fgweight'
    # print('[test] fgweight_path =\n%s' % (ut.repr3(depc.get_dependencies(tablename), nl=1),))
    # print('[test] keypoint =\n%s' % (ut.repr3(depc.get_dependencies('keypoint'), nl=1),))
    # print('[test] descriptor =\n%s' % (ut.repr3(depc.get_dependencies('descriptor'), nl=1),))
    # print('[test] spam =\n%s' % (ut.repr3(depc.get_dependencies('spam'), nl=1),))

    root_rowids = [5, 3]
    desc_rowids = depc.get_rowids('descriptor', root_rowids)  # NOQA

    table = depc[tablename]  # NOQA

    # example_getter_methods(depc, 'vsmany', root_rowids)
    # example_getter_methods(depc, 'chipmask', root_rowids)
    # example_getter_methods(depc, 'keypoint', root_rowids)
    # example_getter_methods(depc, 'chip', root_rowids)

    test_getters(depc)

    # import wbia.plottool as pt
    # pt.ensureqt()

    graph = depc.make_graph()  # NOQA
    # pt.show_nx(graph)

    print('---------- 111 -----------')

    # Try testing the algorithm
    req = depc.new_request('vsmany', root_rowids, root_rowids, {})
    print('req = %r' % (req,))
    req.execute()

    print('---------- 222 -----------')

    cfgdict = {'sver_on': False}
    req = depc.new_request('vsmany', root_rowids, [root_rowids], cfgdict)
    req.execute()

    print('---------- 333 -----------')

    cfgdict = {'sver_on': False, 'adapt_shape': False}
    req = depc.new_request('vsmany', root_rowids, root_rowids, cfgdict)
    req.execute()

    print('---------- 444 -----------')

    req = depc.new_request('vsmany', root_rowids, root_rowids, {})
    req.execute()

    # ut.InstanceList(
    db = list(depc.fname_to_db.values())[0]
    # db_list = ut.InstanceList(depc.fname_to_db.values())
    # db_list.print_table_csv('config', exclude_columns='config_strid')

    print('config table')
    tablename = 'config'
    column_list, column_names = db.get_table_column_data(
        tablename, exclude_columns=['config_strid']
    )
    print(
        '\n'.join(
            [
                ut.hz_str(*list(ut.interleave((r, [', '] * (len(r) - 1)))))
                for r in list(
                    zip(*[[ut.repr3(r, nl=2) for r in col] for col in column_list])
                )
            ]
        )
    )

    return depc


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.example_depcache
        python -m dtool.example_depcache --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
