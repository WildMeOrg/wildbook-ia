# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import uuid
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

    depc = depends_cache.DependencyCache(
        root_tablename=dummy_root, default_fname=fname,
        root_asobject=root_asobject, use_globals=False)

    @depc.register_preproc(
        tablename='chipmask', parents=[dummy_root], colnames=['size', 'mask'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)])
    def dummy_manual_chipmask(depc, parent_rowids, config=None):
        import vtool as vt
        from plottool import interact_impaint
        mask_dpath = ut.unixjoin(depc.cache_dpath, 'ManualChipMask')
        ut.ensuredir(mask_dpath)
        if config is None:
            config = {}
        print('Requesting user defined chip mask')
        for rowid in parent_rowids:
            img = vt.imread(gpath_list[rowid])
            mask = interact_impaint.impaint_mask2(img)
            mask_fpath = ut.unixjoin(mask_dpath, 'mask%d.png' % (rowid,))
            vt.imwrite(mask_fpath, mask)
            w, h = vt.get_size(mask)
            yield (w, h), mask_fpath

    @depc.register_preproc(
        tablename='chip', parents=[dummy_root], colnames=['size', 'chip'],
        coltypes=[(int, int), vt.imread], asobject=True)
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
            print('Computing chips of annot=%r' % (annot,))
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
        print('Computing probchip')
        for rowid in parent_rowids:
            yield (rowid, rowid), 'probchip.jpg'

    @depc.register_preproc(
        'keypoint', ['chip'], ['kpts', 'num'], [np.ndarray, int],
        docstr='Used to store individual chip features (ellipses)',)
    def dummy_preproc_kpts(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing kpts')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 6)) + rowid, 7 + rowid

    @depc.register_preproc('descriptor', ['keypoint'], ['vecs'], [np.ndarray],)
    def dummy_preproc_vecs(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing vecs')
        for rowid in parent_rowids:
            yield np.ones((7 + rowid, 8), dtype=np.uint8) + rowid,

    @depc.register_preproc('fgweight', ['keypoint', 'probchip'], ['fgweight'], [np.ndarray],)
    def dummy_preproc_fgweight(depc, kpts_rowid, probchip_rowid, config=None):
        if config is None:
            config = {}
        print('Computing fgweight')
        for rowid1, rowid2 in zip(kpts_rowid, probchip_rowid):
            yield np.ones(7 + rowid1),

    @depc.register_preproc('notch', [dummy_root], ['notchdata'],)
    def dummy_preproc_notch(depc, parent_rowids, config=None):
        if config is None:
            config = {}
        print('Computing notch')
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
        print('Computing notch')
        ut.writeto('tmp.txt', ut.lorium_ipsum())
        for x in zip(*args):
            size = (42, 21)
            uuid = ut.get_zero_uuid()
            vector = np.ones(3)
            yield ('spam', 3665, size, uuid, vector, 'tmp.txt')

    @depc.register_algo('dumbalgo')
    def dummy_matching_algo(depc, aids, config=None):
        for aid in aids:
            return

    # table = depc['spam']
    # print(ut.repr2(table.get_addtable_kw(), nl=2))

    depc.initialize()

    # table.print_schemadef()
    # print(table.db.get_schema_current_autogeneration_str())
    return depc


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

    root_rowids = [5, 3, 2]

    desc_rowids = depc.get_rowids('descriptor', root_rowids)  # NOQA

    root_rowids = [1]
    if 1:
        tablename = 'chip'
        col1 = 'chip'
    else:
        # Manual interaction given
        tablename = 'chipmask'
        col1 = 'mask'
    col2 = 'size'
    table = depc[tablename]  # NOQA

    # You can get a reference to data rows using the "root" (dummy_annot) rowids
    # By default, if the data has not been computed, then it will be computed
    # for you. But if you specify ensure=False, None will be returned if the data
    # has not been computed yet.

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    #assert tbl_rowids[0] is None

    # The default is for the data to be computed though. Manaual interactions will
    # launch as necessary.

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=True)  # NOQA
    assert tbl_rowids[0] is not None

    # Now the data is cached and will not need to be computed again

    tbl_rowids = depc.get_rowids(tablename, root_rowids, ensure=False)  # NOQA
    assert tbl_rowids[0] is not None

    # The rowids can be used to lookup data values directly. By default all data
    # in a row is returned.

    datas = depc[tablename].get_row_data(tbl_rowids)  # NOQA

    # But you can also ask for a specific column

    col1_data = depc[tablename].get_row_data(tbl_rowids, col1)  # NOQA

    # In the case of external columns, you can lookup the hidden id as follows

    col1_paths = depc[tablename].get_row_data(tbl_rowids, (col1 + depends_cache.EXTERN_SUFFIX,))  # NOQA

    # But you can also just the root rowids directly. This is the simplest way to
    # access data and really "all you need to know"
    datas = depc.get_property(tablename, root_rowids, (col1, col2))  # NOQA

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
    print('chip_dict["size"] = %r' % (chip_dict['size'],))
    print('chip_dict["chip"] = %r' % (chip_dict['chip'],))
    print('chip_dict = %r' % (chip_dict,))

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
