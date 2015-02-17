#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
from functools import partial
print, print_, printDBG, rrr, profile = utool.inject(__name__,
                                                     '[TIME_GEN_PREPROC]')


def timeit2(stmt, setup='', number=1000):
    import timeit
    stmt_ = utool.unindent(stmt)
    setup_ = utool.unindent(setup)
    print('----------')
    print('TIMEIT: \n' + stmt_)
    try:
        total_time = timeit.timeit(stmt_, setup_, number=number)
    except Exception as ex:
        utool.printex(ex, iswarning=False)
        raise
    print(' * timed: %r seconds' % (total_time))


@profile
def TIME_GEN_PREPROC_IMG(ibs):
    from ibeis.model.preproc.preproc_image import add_images_params_gen
    print('[TIME_GEN_PREPROC_IMG]')
    gid_list = ibs.get_valid_gids()
    gpath_list = ibs.get_image_paths(gid_list)

    # STABILITY

    if not utool.get_argflag('--nostable'):
        # TEST 1
        with utool.Timer('parallel chunksize=1'):
            output1 = list(add_images_params_gen(gpath_list, chunksize=1))
        print(utool.truncate_str(str(output1), 80))
        assert len(output1) == len(gpath_list), 'chuncksize changes output'

        # TEST 2
        with utool.Timer('parallel chunksize=2'):
            output2 = list(add_images_params_gen(gpath_list, chunksize=2))
        print(utool.truncate_str(str(output2), 80))
        assert output1 == output2, 'chuncksize changes output'

        # TEST N
        with utool.Timer('parallel chunksize=None'):
            outputN = list(add_images_params_gen(gpath_list, chunksize=None))
        print(utool.truncate_str(str(output2), 80))
        assert outputN == output2, 'chuncksize changes output'

    # BENCHMARK

    setup = utool.unindent(
        '''
        from ibeis.model.preproc.preproc_image import add_images_params_gen
        genkw = dict(prog=False, verbose=True)
        gpath_list = %r
        ''' % (gpath_list,))

    print(utool.truncate_str(str(gpath_list), 80))
    print('Processing %d images' % (len(gpath_list),))
    timeit3 = partial(timeit2, setup=setup, number=3)
    timeit3('list(add_images_params_gen(gpath_list, chunksize=None, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=None, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=1, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=2, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=4, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=8, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=16, **genkw))')
    timeit3('list(add_images_params_gen(gpath_list, chunksize=32, **genkw))')

    print('[/TIME_GEN_PREPROC_IMG]')
    return locals()


@profile
def TIME_GEN_PREPROC_FEAT(ibs):
    print('[TIME_GEN_PREPROC_FEAT]')
    from ibeis.model.preproc.preproc_feat import generate_feats
    from six.moves import zip
    import numpy as np

    def _listeq(x1, x2):
        if isinstance(x1, np.ndarray):
            return np.all(x2 == x2)
        return x1 == x2

    aid_list = ibs.get_valid_aids()
    cid_list = ibs.get_annot_chip_rowids(aid_list)
    cfpath_list = ibs.get_chip_uris(cid_list)

    # STABILITY

    if not utool.get_argflag('--nostable'):
        # TEST 1
        with utool.Timer('parallel chunksize=1'):
            output1 = list(generate_feats(cfpath_list, chunksize=1))
        print(utool.truncate_str(str(output1), 80))

        # TEST 2
        with utool.Timer('parallel chunksize=2'):
            output2 = list(generate_feats(cfpath_list, chunksize=2))
        print(utool.truncate_str(str(output2), 80))

        assert all([_listeq(*xtup) for tup in zip(output1, output2)
                    for xtup in zip(*tup)]), 'chuncksize changes output'

        # TEST N
        with utool.Timer('parallel chunksize=None'):
            outputN = list(generate_feats(cfpath_list, chunksize=None))
        print(utool.truncate_str(str(output2), 80))

        assert all([_listeq(*xtup) for tup in zip(outputN, output2)
                    for xtup in zip(*tup)]), 'chuncksize changes output'

    # BENCHMARK

    setup = utool.unindent(
        '''
        from ibeis.model.preproc.preproc_feat import generate_feats
        genkw = dict(prog=False, verbose=True)
        cfpath_list = %r
        ''' % (cfpath_list,))

    print(utool.truncate_str(str(cid_list), 80))
    print('Processing %d chips' % (len(cid_list),))
    timeit3 = partial(timeit2, setup=setup, number=1)
    timeit3('list(generate_feats(cfpath_list, chunksize=None, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=None, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=1, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=2, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=4, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=8, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=16, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=32, **genkw))')
    timeit3('list(generate_feats(cfpath_list, chunksize=64, **genkw))')

    #list(generate_feats(cfpath_list, chunksize=None, **genkw))
    #[parallel] initializing pool with 7 processes
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=21
    # * timed: 125.17100650510471 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=None, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=21
    # * timed: 97.37531812573734 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=1, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=1
    # * timed: 89.11060989484363 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=2, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=2
    # * timed: 89.3294122591355 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=4, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=4
    # * timed: 114.7752637914524 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=8, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=8
    # * timed: 123.35112345890252 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=16, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=16
    # * timed: 124.47361485097099 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=32, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=32
    # * timed: 126.47238857719219 seconds
    #----------
    #list(generate_feats(cfpath_list, chunksize=64, **genkw))
    #[parallel] executing 1049 gen_feat_worker tasks using 7 processes with chunksize=64
    # * timed: 137.3404114996564 seconds

    print('[/TIME_GEN_PREPROC_FEAT]')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='GZ_ALL', gui=False)
    ibs = main_locals['ibs']
    time_locals = {}

    # Varying chunksize seems not to do much on windows :(

    #time_locals.update(TIME_GEN_PREPROC_IMG(ibs))
    time_locals.update(TIME_GEN_PREPROC_FEAT(ibs))
    execstr = utool.execstr_dict(time_locals, 'time_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
