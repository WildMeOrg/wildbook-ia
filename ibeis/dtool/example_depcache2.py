from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
from os.path import join, dirname
from six.moves import zip


def test_getprop_with_configs():
    r"""
    CommandLine:
        python -m dtool.example_depcache2 test_getprop_with_configs --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> test_getprop_with_configs()
    """
    config1 = {'manual_extract': True}
    config2 = {'manual_extract': False}
    depc = testdata_depc2()

    aid = 2
    _debug = False

    cropchip1 = depc.get('cropchip', aid, 'img', config=config1)
    cropchip2 = depc.get('cropchip', aid, 'img', config=config2)
    print('cropchip1.shape = %r' % (cropchip1.shape,))
    print('cropchip2.shape = %r' % (cropchip2.shape,))
    cropchip1 = depc.get('cropchip', aid, 'img', config=config1)
    cropchip2 = depc.get('cropchip', aid, 'img', config=config2)

    print('cropchip1.shape = %r' % (cropchip1.shape,))
    print('cropchip2.shape = %r' % (cropchip2.shape,))

    chip = depc.get('chip', aid, 'img')
    print('chip.shape = %r' % (chip.shape,))

    tip1 = depc.get('tip', aid, config=config1, _debug=_debug)
    tip2 = depc.get('tip', aid, config=config2, _debug=_debug)

    print('tip1 = %r' % (tip1,))
    print('tip2 = %r' % (tip2,))

    depc.print_all_tables()
    depc.print_config_tables()
    #import utool
    #utool.embed()


def testdata_depc2():
    """
    Example of local registration
    sudo pip install freetype-py

    CommandLine:
        python -m dtool.example_depcache2 testdata_depc2 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc2()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> depc.show_graph()
        >>> ut.show_if_requested()
    """
    import dtool
    import vtool as vt
    from vtool import fontdemo

    # put the test cache in the dtool repo
    dtool_repo = dirname(ut.get_module_dir(dtool))
    cache_dpath = join(dtool_repo, 'DEPCACHE2')

    root = 'annot'

    depc = dtool.DependencyCache(
        root_tablename=root, cache_dpath=cache_dpath, use_globals=False)

    # ----------

    class ChipConfig(dtool.Config):
        _param_info_list = [
            ut.ParamInfo('dim_size', 500),
            ut.ParamInfo('ext', '.png'),
        ]

    @depc.register_preproc(
        tablename='chip', parents=[root], colnames=['size', 'img'],
        coltypes=[(int, int), ('extern', vt.imread, vt.imwrite)],
        configclass=ChipConfig)
    def compute_chip(depc, aids, config=None):
        for aid in aids:
            chip = fontdemo.get_text_test_img(str(aid))
            size = vt.get_size(chip)
            yield size, chip

    # ----------

    class TipConfig(dtool.Config):
        _param_info_list = [
            ut.ParamInfo('manual_extract', False, hideif=False),
        ]

    @depc.register_preproc(
        tablename='tip', parents=['chip'],
        colnames=['notch', 'left', 'right'],
        coltypes=[np.ndarray, np.ndarray, np.ndarray],
        configclass=TipConfig,
    )
    def compute_tips(depc, chip_rowids, config=None):
        manual_extract = config['manual_extract']
        chips = depc.get_native('chip', chip_rowids, 'img')
        for chip in chips:
            seed = (chip).sum()
            perb = ((seed % 1000) / 1000) * .25
            w, h = vt.get_size(chip)
            if manual_extract:
                # Make noticable difference between config outputs
                lpb =  np.ceil(w * perb)
                npb =  np.ceil(h * perb)
                rpb = -np.ceil(w * perb)
            else:
                lpb =  np.ceil(w * perb / 2)
                npb = -np.ceil(h * perb)
                rpb = -np.ceil(w * perb)
            wh = np.array([w, h], dtype=np.int32)[None, :]
            rel_base = np.array([[.0, .5], [.5, .5], [1., .5]])
            offset   = np.array([[lpb, 0], [0, npb], [rpb, 0]])
            tip = np.round((wh * rel_base)) + offset
            left, notch, right = tip
            yield left, notch, right

    # ----------

    class CropChipConfig(dtool.Config):
        _param_info_list = [
            ut.ParamInfo('dim_size', 500),
        ]

    @depc.register_preproc(
        tablename='cropchip', parents=['chip', 'tip'],
        colnames=['img'],
        coltypes=[np.ndarray],
        configclass=CropChipConfig,
    )
    def compute_cropchip(depc, cids, tids, config=None):
        print("COMPUTE CROPCHIP")
        print('config = %r' % (config,))
        chips = depc.get_native('chip', cids, 'img')
        tips = depc.get_native('tip', tids)
        print('tips = %r' % (tips,))
        for chip, tip in zip(chips, tips):
            notch, left, right = tip
            lx = left[0]
            rx = right[0]
            cropped_chip = chip[lx:(rx - 1), ...]
            yield (cropped_chip,)

    # ----------

    class TrailingEdgeConfig(dtool.Config):
        _param_info_list = []

    @depc.register_preproc(
        tablename='trailingedge', parents=['cropchip'],
        colnames=['te'],
        coltypes=[np.ndarray],
        configclass=TrailingEdgeConfig,
    )
    def compute_trailing_edge(depc, cropped_chips, config=None):
        for cc in cropped_chips:
            #depc.get_native('chip', cids)
            size = 1
            te = np.arange(size)
            yield (te,)

    depc.initialize()
    return depc


def testdata_depc3():
    """
    Example of local registration

    CommandLine:
        python -m dtool.example_depcache2 testdata_depc3 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc3()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> depc.show_graph()
        >>> depc['smk_match'].show_input_graph()
        >>> depc['vsone'].show_input_graph()
        >>> depc['neighbs'].show_input_graph()
        >>> #depc['viewpoint_classification'].show_input_graph()
        >>> print(depc['smk_match'].compute_order)
        >>> ut.show_if_requested()
    """
    import dtool

    # put the test cache in the dtool repo
    dtool_repo = dirname(ut.get_module_dir(dtool))
    cache_dpath = join(dtool_repo, 'DEPCACHE2')

    root = 'annot'

    depc = dtool.DependencyCache(
        root_tablename=root, cache_dpath=cache_dpath, use_globals=False)

    # ----------
    dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
    def dummy_func(depc, *args, **kwargs):
        return None

    depc.register_preproc(tablename='indexer', parents=['annot*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='neighbs', parents=['annot', 'indexer'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='vocab', parents=['annot*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='smk_vec', parents=['annot', 'vocab'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='inv_index', parents=['smk_vec*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='smk_match', parents=['smk_vec', 'inv_index'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='vsone', parents=['annot', 'annot'], **dummy_cols)(dummy_func)
    #depc.register_preproc(tablename='viewpoint_classifier', parents=['annot*'], **dummy_cols)(dummy_func)
    #depc.register_preproc(tablename='viewpoint_classification', parents=['annot', 'viewpoint_classifier'], **dummy_cols)(dummy_func)

    depc.initialize()
    return depc


def testdata_depc_image():
    """
    Example of local registration
    sudo pip install freetype-py

    CommandLine:
        python -m dtool.example_depcache2 testdata_depc_image --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc_image()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> depc.show_graph()
        >>> depc['detection'].show_input_graph()
        >>> print(depc['detection'].compute_order)
        >>> ut.show_if_requested()
    """
    import dtool

    # put the test cache in the dtool repo
    dtool_repo = dirname(ut.get_module_dir(dtool))
    cache_dpath = join(dtool_repo, 'DEPCACHE2')

    root = 'image'

    depc = dtool.DependencyCache(
        root_tablename=root, cache_dpath=cache_dpath, use_globals=False)

    # ----------
    dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
    def dummy_func(depc, *args, **kwargs):
        return None

    depc.register_preproc(tablename='detector', parents=['image*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='detection', parents=['image', 'detector'], **dummy_cols)(dummy_func)

    depc.initialize()
    return depc


def testdata_depc_annot():
    """
    CommandLine:
        python -m dtool.example_depcache2 testdata_depc_annot --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc_annot()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> depc.show_graph()
        >>> tablename = 'featweight'
        >>> table = depc[tablename]
        >>> table.show_input_graph()
        >>> print(table.compute_order)
        >>> ut.show_if_requested()
    """
    import dtool
    # put the test cache in the dtool repo
    dtool_repo = dirname(ut.get_module_dir(dtool))
    cache_dpath = join(dtool_repo, 'DEPCACHE2')
    dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
    def dummy_func(depc, *args, **kwargs):
        return None

    # NOTE: Consider the smk_match.
    # It would be really cool if we could say that the vocab
    # for the input to the parent smk_vec must be the same vocab
    # that was used to compute the inverted index. How do we encode that?

    root = 'annot'
    #vocab_parent = 'annot'
    #vocab_parent = 'chip'
    #vocab_parent = 'feat'
    vocab_parent = 'featweight'
    depc = dtool.DependencyCache(
        root_tablename=root, cache_dpath=cache_dpath, use_globals=False)
    depc.register_preproc(tablename='chip', parents=['annot'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='fgmodel', parents=['chip*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='probchip', parents=['annot', 'fgmodel'], **dummy_cols)(dummy_func)
    #depc.register_preproc(tablename='probchip', parents=['annot'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='feat', parents=['chip'], **dummy_cols)(dummy_func)
    #depc.register_preproc(tablename='feat', parents=['annot'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='featweight', parents=['feat', 'probchip'], **dummy_cols)(dummy_func)

    depc.register_preproc(tablename='indexer', parents=[vocab_parent + '*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='neighbs', parents=[vocab_parent, 'indexer'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='vocab', parents=[vocab_parent + '*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='smk_vec', parents=[vocab_parent, 'vocab'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='inv_index', parents=['smk_vec*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='smk_match', parents=['smk_vec', 'inv_index'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='vsone', parents=[vocab_parent, vocab_parent], **dummy_cols)(dummy_func)

    depc.register_preproc(tablename='viewpoint_model', parents=['annot*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='viewpoint', parents=['annot', 'viewpoint_model'], **dummy_cols)(dummy_func)

    depc.register_preproc(tablename='quality_model', parents=['annot*'], **dummy_cols)(dummy_func)
    depc.register_preproc(tablename='quality', parents=['annot', 'quality_model'], **dummy_cols)(dummy_func)

    depc.initialize()
    return depc


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.example_depcache2
        python -m dtool.example_depcache2 --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
