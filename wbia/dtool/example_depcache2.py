# -*- coding: utf-8 -*-
from pathlib import Path

import utool as ut
from six.moves import zip

from wbia.dtool.depcache_control import DependencyCache
from wbia.dtool.example_depcache import DummyController


HERE = Path(__file__).parent.resolve()


def _depc_factory(name, cache_dir):
    """DependencyCache factory for the examples

    Args:
        name (str): name of the cache (e.g. 'annot')
        cache_dir (str): name of the cache directory

    """
    cache_dpath = HERE / cache_dir
    controller = DummyController(cache_dpath)
    depc = DependencyCache(
        controller,
        name,
        ut.identity,
        table_name=None,
        root_getters=None,
        use_globals=False,
    )
    return depc


def depc_34_helper(depc):
    def register_dummy_config(tablename, parents, **kwargs):
        config_param = tablename + '_param'

        def dummy_single_func(depc, *row_arg, **kw):
            config = kw.get('config')
            param_val = config[config_param]
            data = []
            for row, p in zip(row_arg, parents):
                p = p.replace('*', '')
                # print('p = %r' % (p,))
                # print('row = %r' % (row,))
                if not p.startswith(depc.root):
                    native_cols = depc.get_native(p, ut.ensure_iterable(row))
                    parent_data = '+'.join(['#'.join(col) for col in native_cols])
                else:
                    parent_data = (
                        'root(' + ';'.join(list(map(str, ut.ensure_iterable(row)))) + ')'
                    )
                data += [parent_data]
            d = '[' + '&'.join(data) + ']'
            retstr = tablename + '(' + d + ':' + str(param_val) + ')'
            return (retstr,)
            # return (data + tablename + repr(row_arg) + repr(param_val)),

        if kwargs.get('vectorized'):
            dummy_func = dummy_single_func
        else:

            def dummy_gen_func(depc, *argsT, **kw):
                # config = kw.get('config')
                # param_val = config[config_param]
                for row_arg in zip(*argsT):
                    yield dummy_single_func(depc, *row_arg, **kw)
                    # (tablename + repr(row_arg) + repr(param_val)),
                # yield (np.array([row_arg]),)

            dummy_func = dummy_gen_func
        from wbia.dtool import base

        configclass = base.make_configclass({config_param: 42}, tablename)
        dummy_cols = dict(
            colnames=['data'], coltypes=[str], configclass=configclass, **kwargs
        )
        depc.register_preproc(tablename=tablename, parents=parents, **dummy_cols)(
            dummy_func
        )

    return register_dummy_config


def testdata_depc3(in_memory=True):
    """
    Example of local registration

    CommandLine:
        python -m dtool.example_depcache2 testdata_depc3 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc3()
        >>> data = depc.get('labeler', [1, 2, 3], 'data', _debug=True)
        >>> data = depc.get('indexer', [[1, 2, 3]], 'data', _debug=True)
        >>> depc.print_all_tables()
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> depc.show_graph()
        >>> from wbia.plottool.interactions import ExpandableInteraction
        >>> inter = ExpandableInteraction(nCols=2)
        >>> depc['smk_match'].show_input_graph(inter)
        >>> depc['vsone'].show_input_graph(inter)
        >>> #depc['vocab'].show_input_graph(inter)
        >>> depc['neighbs'].show_input_graph(inter)
        >>> inter.start()
        >>> #depc['viewpoint_classification'].show_input_graph()
        >>> ut.show_if_requested()
    """
    depc = _depc_factory('annot', 'DEPCACHE3')

    # ----------
    # dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])
    register_dummy_config = depc_34_helper(depc)

    register_dummy_config(tablename='labeler', parents=['annot'])
    register_dummy_config(tablename='meta_labeler', parents=['labeler'])
    register_dummy_config(tablename='indexer', parents=['annot*'])
    # register_dummy_config(tablename='neighbs', parents=['annot', 'indexer'])
    register_dummy_config(tablename='neighbs', parents=['meta_labeler', 'indexer'])
    register_dummy_config(tablename='vocab', parents=['annot*'])
    # register_dummy_config(tablename='smk_vec', parents=['annot', 'vocab'], vectorized=True)
    register_dummy_config(tablename='smk_vec', parents=['annot', 'vocab'])
    # vectorized=True)
    # register_dummy_config(tablename='inv_index', parents=['smk_vec*'])
    register_dummy_config(tablename='inv_index', parents=['smk_vec*', 'vocab'])
    register_dummy_config(tablename='smk_match', parents=['smk_vec', 'inv_index'])
    register_dummy_config(tablename='vsone', parents=['annot', 'annot'])
    # register_dummy_config(tablename='viewpoint_classifier', parents=['annot*'])
    # register_dummy_config(tablename='viewpoint_classification', parents=['annot', 'viewpoint_classifier'])

    depc.initialize()
    return depc


def testdata_depc4(in_memory=True):
    """
    Example of local registration

    CommandLine:
        python -m dtool.example_depcache2 testdata_depc4 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc4()
        >>> #data = depc.get('labeler', [1, 2, 3], 'data', _debug=True)
        >>> #data = depc.get('indexer', [[1, 2, 3]], 'data', _debug=True)
        >>> depc.print_all_tables()
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> depc.show_graph()
        >>> from wbia.plottool.interactions import ExpandableInteraction
        >>> inter = ExpandableInteraction(nCols=2)
        >>> depc['smk_match'].show_input_graph(inter)
        >>> depc['vsone'].show_input_graph(inter)
        >>> depc['vocab'].show_input_graph(inter)
        >>> depc['neighbs'].show_input_graph(inter)
        >>> inter.start()
        >>> #depc['viewpoint_classification'].show_input_graph()
        >>> ut.show_if_requested()
    """
    depc = _depc_factory('annot', 'DEPCACHE4')

    # ----------
    # dummy_cols = dict(colnames=['data'], coltypes=[np.ndarray])

    register_dummy_config = depc_34_helper(depc)

    register_dummy_config(tablename='chip', parents=['annot'])
    register_dummy_config(tablename='probchip', parents=['annot'])
    register_dummy_config(tablename='feat', parents=['chip', 'probchip'])
    register_dummy_config(tablename='labeler', parents=['feat'])

    register_dummy_config(tablename='indexer', parents=['feat*'])
    register_dummy_config(tablename='neighbs', parents=['feat', 'indexer'])
    register_dummy_config(tablename='vocab', parents=['feat*'])
    register_dummy_config(
        tablename='smk_vec', parents=['feat', 'vocab'], vectorized=False
    )
    # register_dummy_config(tablename='inv_index', parents=['smk_vec*'])
    register_dummy_config(tablename='inv_index', parents=['smk_vec*', 'vocab'])
    register_dummy_config(tablename='smk_match', parents=['smk_vec', 'inv_index'])
    register_dummy_config(tablename='vsone', parents=['feat', 'feat'])
    # register_dummy_config(tablename='viewpoint_classifier', parents=['annot*'])
    # register_dummy_config(tablename='viewpoint_classification', parents=['annot', 'viewpoint_classifier'])

    depc.initialize()
    return depc


def testdata_custom_annot_depc(dummy_dependencies, in_memory=True):
    depc = _depc_factory('annot', 'DEPCACHE5')

    # ----------
    register_dummy_config = depc_34_helper(depc)

    for dummy in dummy_dependencies:
        register_dummy_config(**dummy)

    depc.initialize()
    return depc
