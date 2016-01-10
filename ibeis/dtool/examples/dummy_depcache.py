# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m dtool.examples.dummy_depcache --exec-dummy_example_depcacahe --show
    python -m dtool.depends_cache --exec-make_digraph --show
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import uuid
from os.path import join
from six.moves import zip
from dtool import depends_cache


if False:
    DUMMY_ROOT_TABLENAME = 'dummy_annot'
    register_preproc, register_algo = depends_cache.make_depcache_decors(DUMMY_ROOT_TABLENAME)

    # Example of global preproc function
    @register_preproc(tablename='dummy', parents=[DUMMY_ROOT_TABLENAME], colnames=['data'], coltypes=[str])
    def dummy_global_preproc_func(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Requesting global dummy ')
        for rowid in parent_rowids:
            yield 'dummy'


def testdata_depc(fname=None):
    import dtool
    import vtool as vt
    gpath_list = ut.lmap(ut.grab_test_imgpath, ut.get_valid_test_imgkeys(), verbose=False)

    dummy_root = 'dummy_annot'

    def root_asobject(aid):
        """ Convinience for writing preproc funcs """
        gpath = gpath_list[aid]
        root_obj = ut.LazyDict({
            'aid': aid,
            'gpath': gpath,
            'image': lambda: vt.imread(gpath)
        })
        return root_obj

    depc = dtool.DependencyCache(
        root_tablename=dummy_root, default_fname=fname,
        root_asobject=root_asobject, use_globals=False)

    @depc.register_preproc(
        tablename='chipmask', parents=[dummy_root], colnames=['size', 'mask'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)])
    def dummy_manual_chipmask(depc, parent_rowids, config=None):
        import vtool as vt
        from plottool import interact_impaint
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

    class DummyChipConfig(dtool.TableConfig):
        def get_param_info_list(self):
            return [
                ut.ParamInfo('resize_dim', 'width',
                             valid_values=['area', 'width', 'heigh', 'diag']),
                ut.ParamInfo('size', 500, 'sz'),
                ut.ParamInfo('preserve_aspect', True),
                ut.ParamInfo('histeq', False, hideif=False),
                ut.ParamInfo('fmt', '.png'),
            ]

    cfg = DummyChipConfig()
    cfg.size = 700
    cfg.histeq = True
    print(cfg)
    cfg.histeq = False
    print(cfg)

    @depc.register_preproc(
        tablename='chip', parents=[dummy_root], colnames=['size', 'chip'],
        coltypes=[(int, int), vt.imread], config_class=DummyChipConfig,
        asobject=True)
    def dummy_preproc_chip(depc, annot_list, config=None):
        """
        TODO: Infer properties from docstr

        Args:
            annot_list (list): list of annot objects
            config (dict): config dictionary

        Returns:
            tuple : ((int, int), ('extern', vt.imread))
        """
        if config is None:
            config = {}
        # Demonstates using asobject to get input to function as a dictionary
        # of properties
        for annot in annot_list:
            print('[preproc] Computing chips of annot=%r' % (annot,))
            chip_fpath = annot['gpath']
            w, h = vt.image.open_image_size(chip_fpath)
            size = (w, h)
            print('* chip_fpath = %r' % (chip_fpath,))
            print('* size = %r' % (size,))
            yield size, chip_fpath

    @depc.register_preproc(
        'probchip', [dummy_root], ['size', 'probchip'],
        coltypes=[(int, int), ('extern', vt.imread)])
    def dummy_preproc_probchip(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing probchip')
        for rowid in parent_rowids:
            yield (rowid, rowid), 'probchip.jpg'

    class DummyKptsConfig(dtool.TableConfig):
        def get_param_info_list(self):
            return [
                ut.ParamInfo('adapt_shape', True),
                ut.ParamInfo('adapt_angle', False),
            ]

    @depc.register_preproc(
        'keypoint', ['chip'], ['kpts', 'num'], [np.ndarray, int],
        config_class=DummyKptsConfig,
        docstr='Used to store individual chip features (ellipses)',)
    def dummy_preproc_kpts(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing kpts')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 6)) + rowid, 7 + rowid

    @depc.register_preproc('descriptor', ['keypoint'], ['vecs'], [np.ndarray],)
    def dummy_preproc_vecs(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing vecs')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 8), dtype=np.uint8) + rowid,

    @depc.register_preproc('fgweight', ['keypoint', 'probchip'], ['fgweight'], [np.ndarray],)
    def dummy_preproc_fgweight(depc, kpts_rowid, probchip_rowid, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing fgweight')
        for rowid1, rowid2 in zip(kpts_rowid, probchip_rowid):
            yield np.ones(7 + rowid1),

    @depc.register_preproc('notch', [dummy_root], ['notchdata'],)
    def dummy_preproc_notch(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('[preproc] Computing notch')
        for rowid in parent_rowids:
            yield np.empty(5 + rowid),

    @depc.register_preproc(
        'spam', ['fgweight', 'chip', 'keypoint'],
        ['spam', 'eggs', 'size', 'uuid', 'vector', 'textdata'],
        [str, int, (int, int), uuid.UUID, np.ndarray, ('extern', ut.readfrom)],
        docstr='I dont like spam',)
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

    class DummyIndexerConfig(dtool.AlgoConfig):
        def get_param_info_list(self):
            return [
                ut.ParamInfo('index_method', 'single'),
                ut.ParamInfo('trees', 8),
                ut.ParamInfo('algorithm', 'kdtree'),
            ]

    class DummyNNConfig(dtool.AlgoConfig):
        def get_param_info_list(self):
            return [
                ut.ParamInfo('K', 4),
                ut.ParamInfo('Knorm', 1),
                ut.ParamInfo('checks', 800),
            ]

    class DummySVERConfig(dtool.AlgoConfig):
        def get_param_info_list(self):
            return [
                ut.ParamInfo('sv_on', True),
                ut.ParamInfo('xy_thresh', .01)
            ]

    class DummyAlgoConfig(dtool.AlgoConfig):
        def get_sub_config_list(self):
            # Different pipeline compoments can go here
            # as well as dependencies that were not
            # explicitly enumerated in the tree structure
            return [
                DummyChipConfig,  # I guess different annots might want different configs ...
                DummyKptsConfig,
                DummyIndexerConfig,
                DummyNNConfig,
                DummySVERConfig
            ]

        def get_param_info_list(self):
            return [
                ut.ParamInfo('score_method', 'csum'),
                ut.ParamInfo('daid_list', None),
            ]

    algo_config = DummyAlgoConfig()
    print(algo_config)

    class DummyAnnotMatch(dtool.AlgoResult):
        def __init__(self, qaid=None, daid_list=None, dnid_list=None,
                     annot_score_list=None, unique_nids=None,
                     name_score_list=None):
            self.qaid = qaid
            self.daid_list = daid_list
            self.dnid_list = dnid_list
            self.annot_score_list = annot_score_list
            self.name_score_list = name_score_list

    @depc.register_algo('dumbalgo', DummyAnnotMatch, config_class=DummyAlgoConfig)
    def dummy_matching_algo(depc, aids, config=None):
        daid_list = config['daid_list']
        for qaid in aids:
            dnid_list = [1, 1, 2, 2]
            unique_nids = [1, 2]
            annot_score_list = [.2, .2, .4, .5]
            name_score_list = [.2, .5]
            annot_match = DummyAnnotMatch(qaid, daid_list, dnid_list,
                                          annot_score_list, unique_nids,
                                          name_score_list)
            yield annot_match

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
    import dtool
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
    #assert tbl_rowids[0] is None

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
        extern_paths = table.get_row_data(tbl_rowids, (excol + dtool.depcache_table.EXTERN_SUFFIX,))  # NOQA
        print('extern_paths = table.get_row_data(tbl_rowids, (excol + dtool.depcache_table.EXTERN_SUFFIX,))')
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

    chip_dict = depc.get_obj('chip', 1)

    print('chip_dict = %r' % (chip_dict,))
    for key in chip_dict.keys():
        print(ut.varinfo_str(chip_dict[key], 'chip_dict["%s"]' % (key,)))
    # print('chip_dict["chip"] = %s' % (ut.trunc_repr(chip_dict['chip']),))
    print('chip_dict = %r' % (chip_dict,))


def dummy_example_depcacahe():
    r"""
    CommandLine:
        python -m dtool.examples.dummy_depcache --exec-dummy_example_depcacahe --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from dtool.examples.dummy_depcache import *  # NOQA
        >>> depc = dummy_example_depcacahe()
        >>> ut.show_if_requested()
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

    # Try testing the algorithm
    example_getter_methods(depc, 'dumbalgo', root_rowids)

    # example_getter_methods(depc, 'chipmask', root_rowids)
    # example_getter_methods(depc, 'keypoint', root_rowids)
    # example_getter_methods(depc, 'chip', root_rowids)

    test_getters(depc)

    import plottool as pt
    # pt.ensure_pylab_qt4()

    graph = depc.make_digraph()
    pt.show_netx(graph)

    return depc

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.examples.dummy_depcache
        python -m dtool.examples.dummy_depcache --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
