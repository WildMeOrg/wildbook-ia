# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import vtool as vt
import numpy as np
import utool as ut
from vtool._pyflann_backend import pyflann as pyflann
from os.path import basename, exists  # NOQA
from six.moves import range
from wbia.algo.hots import neighbor_index_cache

# import mem_top

(print, rrr, profile) = ut.inject2(__name__)


def augment_nnindexer_experiment():
    """

    References:
        http://answers.opencv.org/question/44592/flann-index-training-fails-with-segfault/

    CommandLine:
        utprof.py -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment
        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment

        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_MTEST --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6
        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --show
        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --show


        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --no-api-cache --nocache-uuids

        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_MTEST --show
        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show

        # RUNS THE SEGFAULTING CASE
        python -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show
        # Debug it
        gdb python
        run -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show
        gdb python
        run -m wbia.algo.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6


    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> # execute function
        >>> augment_nnindexer_experiment()
        >>> # verify results
        >>> ut.show_if_requested()

    """
    import wbia

    # build test data
    # ibs = wbia.opendb('PZ_MTEST')
    ibs = wbia.opendb(defaultdb='PZ_Master0')
    if ibs.get_dbname() == 'PZ_MTEST':
        initial = 1
        addition_stride = 4
        max_ceiling = 100
    elif ibs.get_dbname() == 'PZ_Master0':
        initial = 128
        # addition_stride = 64
        # addition_stride = 128
        addition_stride = 256
        max_ceiling = 10000
        # max_ceiling = 4000
        # max_ceiling = 2000
        # max_ceiling = 600
    else:
        assert False
    all_daids = ibs.get_valid_aids(species='zebra_plains')
    qreq_ = ibs.new_query_request(all_daids, all_daids)
    max_num = min(max_ceiling, len(all_daids))

    # Clear Caches
    ibs.delete_flann_cachedir()
    neighbor_index_cache.clear_memcache()
    neighbor_index_cache.clear_uuid_cache(qreq_)

    # Setup
    all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    # ensure all features are computed

    nnindexer_list = []
    addition_lbl = 'Addition'
    _addition_iter = list(range(initial + 1, max_num, addition_stride))
    addition_iter = iter(
        ut.ProgressIter(_addition_iter, lbl=addition_lbl, freq=1, autoadjust=False)
    )
    time_list_addition = []
    # time_list_reindex = []
    addition_count_list = []
    tmp_cfgstr_list = []

    # for _ in range(80):
    #    next(addition_iter)
    try:
        memtrack = ut.MemoryTracker(disable=False)
        for count in addition_iter:
            aid_list_ = all_randomize_daids_[0:count]
            # Request an indexer which could be an augmented version of an existing indexer.
            with ut.Timer(verbose=False) as t:
                memtrack.report('BEFORE AUGMENT')
                nnindexer_ = neighbor_index_cache.request_augmented_wbia_nnindexer(
                    qreq_, aid_list_
                )
                memtrack.report('AFTER AUGMENT')
            nnindexer_list.append(nnindexer_)
            addition_count_list.append(count)
            time_list_addition.append(t.ellapsed)
            tmp_cfgstr_list.append(nnindexer_.cfgstr)
            print('===============\n\n')
        print(ut.repr2(time_list_addition))
        print(ut.repr2(list(map(id, nnindexer_list))))
        print(ut.repr2(tmp_cfgstr_list))
        print(ut.repr2(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))

        IS_SMALL = False

        if IS_SMALL:
            nnindexer_list = []
        reindex_label = 'Reindex'
        # go backwards for reindex
        _reindex_iter = list(range(initial + 1, max_num, addition_stride))[::-1]
        reindex_iter = ut.ProgressIter(_reindex_iter, lbl=reindex_label)
        time_list_reindex = []
        # time_list_reindex = []
        reindex_count_list = []

        for count in reindex_iter:
            print('\n+===PREDONE====================\n')
            # check only a single size for memory leaks
            # count = max_num // 16 + ((x % 6) * 1)
            # x += 1

            aid_list_ = all_randomize_daids_[0:count]
            # Call the same code, but force rebuilds
            memtrack.report('BEFORE REINDEX')
            with ut.Timer(verbose=False) as t:
                nnindexer_ = neighbor_index_cache.request_augmented_wbia_nnindexer(
                    qreq_, aid_list_, force_rebuild=True, memtrack=memtrack
                )
            memtrack.report('AFTER REINDEX')
            ibs.print_cachestats_str()
            print(
                '[nnindex.MEMCACHE] size(NEIGHBOR_CACHE) = %s'
                % (ut.get_object_size_str(neighbor_index_cache.NEIGHBOR_CACHE.items()),)
            )
            print(
                '[nnindex.MEMCACHE] len(NEIGHBOR_CACHE) = %s'
                % (len(neighbor_index_cache.NEIGHBOR_CACHE.items()),)
            )
            print(
                '[nnindex.MEMCACHE] size(UUID_MAP_CACHE) = %s'
                % (ut.get_object_size_str(neighbor_index_cache.UUID_MAP_CACHE),)
            )
            print('totalsize(nnindexer) = ' + ut.get_object_size_str(nnindexer_))
            memtrack.report_type(neighbor_index_cache.NeighborIndex)
            ut.print_object_size_tree(nnindexer_, lbl='nnindexer_')
            if IS_SMALL:
                nnindexer_list.append(nnindexer_)
            reindex_count_list.append(count)
            time_list_reindex.append(t.ellapsed)
            # import cv2
            # import matplotlib as mpl
            # print(mem_top.mem_top(limit=30, width=120,
            #                      #exclude_refs=[cv2.__dict__, mpl.__dict__]
            #     ))
            print('L___________________\n\n\n')
        print(ut.repr2(time_list_reindex))
        if IS_SMALL:
            print(ut.repr2(list(map(id, nnindexer_list))))
            print(ut.repr2(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))
    except KeyboardInterrupt:
        print('\n[train] Caught CRTL+C')
        resolution = ''
        from six.moves import input

        while not (resolution.isdigit()):
            print('\n[train] What do you want to do?')
            print('[train]     0 - Continue')
            print('[train]     1 - Embed')
            print('[train]  ELSE - Stop network training')
            resolution = input('[train] Resolution: ')
        resolution = int(resolution)
        # We have a resolution
        if resolution == 0:
            print('resuming training...')
        elif resolution == 1:
            ut.embed()

    import wbia.plottool as pt

    next_fnum = iter(range(0, 1)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    if len(addition_count_list) > 0:
        pt.plot2(
            addition_count_list,
            time_list_addition,
            marker='-o',
            equal_aspect=False,
            x_label='num_annotations',
            label=addition_lbl + ' Time',
        )

    if len(reindex_count_list) > 0:
        pt.plot2(
            reindex_count_list,
            time_list_reindex,
            marker='-o',
            equal_aspect=False,
            x_label='num_annotations',
            label=reindex_label + ' Time',
        )

    pt.set_figtitle('Augmented indexer experiment')

    pt.legend()


def flann_add_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment

    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_MTEST --show
        python -m wbia.algo.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_Master0 --show
        utprof.py -m wbia.algo.hots._neighbor_experiment --test-flann_add_time_experiment --show

        valgrind --tool=memcheck --suppressions=valgrind-python.supp python -m wbia.algo.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_MTEST --no-with-reindex

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> import wbia
        >>> #ibs = wbia.opendb('PZ_MTEST')
        >>> result = flann_add_time_experiment()
        >>> # verify results
        >>> print(result)
        >>> ut.show_if_requested()

    """
    import wbia
    import utool as ut
    import numpy as np
    import wbia.plottool as pt

    def make_flann_index(vecs, flann_params):
        flann = pyflann.FLANN()
        flann.build_index(vecs, **flann_params)
        return flann

    db = ut.get_argval('--db')
    ibs = wbia.opendb(db=db)

    # Input
    if ibs.get_dbname() == 'PZ_MTEST':
        initial = 1
        reindex_stride = 16
        addition_stride = 4
        max_ceiling = 120
    elif ibs.get_dbname() == 'PZ_Master0':
        # ibs = wbia.opendb(db='GZ_ALL')
        initial = 32
        reindex_stride = 32
        addition_stride = 16
        max_ceiling = 300001
    else:
        assert False
    # max_ceiling = 32
    all_daids = ibs.get_valid_aids()
    max_num = min(max_ceiling, len(all_daids))
    flann_params = vt.get_flann_params()

    # Output
    count_list, time_list_reindex = [], []
    count_list2, time_list_addition = [], []

    # Setup
    # all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    all_randomize_daids_ = all_daids
    # ensure all features are computed
    ibs.get_annot_vecs(all_randomize_daids_)

    def reindex_step(count, count_list, time_list_reindex):
        daids = all_randomize_daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = make_flann_index(vecs, flann_params)  # NOQA
        count_list.append(count)
        time_list_reindex.append(t.ellapsed)

    def addition_step(count, flann, count_list2, time_list_addition):
        daids = all_randomize_daids_[count : count + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann.add_points(vecs)
        count_list2.append(count)
        time_list_addition.append(t.ellapsed)

    def make_initial_index(initial):
        daids = all_randomize_daids_[0 : initial + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        flann = make_flann_index(vecs, flann_params)
        return flann

    WITH_REINDEX = not ut.get_argflag('--no-with-reindex')
    if WITH_REINDEX:
        # Reindex Part
        reindex_lbl = 'Reindexing'
        _reindex_iter = range(1, max_num, reindex_stride)
        reindex_iter = ut.ProgressIter(_reindex_iter, lbl=reindex_lbl, freq=1)
        for count in reindex_iter:
            reindex_step(count, count_list, time_list_reindex)

    # Add Part
    flann = make_initial_index(initial)
    addition_lbl = 'Addition'
    _addition_iter = range(initial + 1, max_num, addition_stride)
    addition_iter = ut.ProgressIter(_addition_iter, lbl=addition_lbl)
    for count in addition_iter:
        addition_step(count, flann, count_list2, time_list_addition)

    print('---')
    print('Reindex took time_list_reindex %.2s seconds' % sum(time_list_reindex))
    print('Addition took time_list_reindex  %.2s seconds' % sum(time_list_addition))
    print('---')
    statskw = dict(precision=2, newlines=True)
    print('Reindex stats ' + ut.get_stats_str(time_list_reindex, **statskw))
    print('Addition stats ' + ut.get_stats_str(time_list_addition, **statskw))

    print('Plotting')

    # with pt.FigureContext:

    next_fnum = iter(range(0, 2)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    if WITH_REINDEX:
        pt.plot2(
            count_list,
            time_list_reindex,
            marker='-o',
            equal_aspect=False,
            x_label='num_annotations',
            label=reindex_lbl + ' Time',
            dark=False,
        )

    # pt.figure(fnum=next_fnum())
    pt.plot2(
        count_list2,
        time_list_addition,
        marker='-o',
        equal_aspect=False,
        x_label='num_annotations',
        label=addition_lbl + ' Time',
    )

    pt
    pt.legend()


def subindexer_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment
    """
    import wbia
    import utool as ut
    from vtool._pyflann_backend import pyflann as pyflann
    import wbia.plottool as pt

    ibs = wbia.opendb(db='PZ_Master0')
    daid_list = ibs.get_valid_aids()
    count_list = []
    time_list = []
    flann_params = vt.get_flann_params()
    for count in ut.ProgressIter(range(1, 301)):
        daids_ = daid_list[:]
        np.random.shuffle(daids_)
        daids = daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = pyflann.FLANN()
            flann.build_index(vecs, **flann_params)
        count_list.append(count)
        time_list.append(t.ellapsed)
    count_arr = np.array(count_list)
    time_arr = np.array(time_list)
    pt.plot2(
        count_arr,
        time_arr,
        marker='-',
        equal_aspect=False,
        x_label='num_annotations',
        y_label='FLANN build time',
    )
    # pt.update()


def trytest_incremental_add(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --test-test_incremental_add

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> result = test_incremental_add(ibs)
        >>> print(result)
    """
    import wbia

    sample_aids = wbia.testdata_aids(a='default:pername=1,mingt=2')
    aids1 = sample_aids[::2]
    aids2 = sample_aids[0:5]
    aids3 = sample_aids[:-1]  # NOQA
    daid_list = aids1  # NOQA
    qreq_ = ibs.new_query_request(aids1, aids1)
    nnindexer1 = neighbor_index_cache.request_wbia_nnindexer(  # NOQA
        ibs.new_query_request(aids1, aids1)
    )
    nnindexer2 = neighbor_index_cache.request_wbia_nnindexer(  # NOQA
        ibs.new_query_request(aids2, aids2)
    )

    # TODO: SYSTEM use visual uuids
    items = ibs.get_annot_visual_uuids(aids3)
    uuid_map_fpath = neighbor_index_cache.get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = neighbor_index_cache.read_uuid_map(uuid_map_fpath, 0)
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    nnindexer3 = neighbor_index_cache.request_wbia_nnindexer(  # NOQA
        ibs.new_query_request(uncovered_aids, uncovered_aids)
    )

    # TODO: SYSTEM use visual uuids
    items = ibs.get_annot_visual_uuids(sample_aids)
    uuid_map_fpath = neighbor_index_cache.get_nnindexer_uuid_map_fpath(qreq_)
    # contextlib.closing(shelve.open(uuid_map_fpath)) as uuid_map:
    candidate_uuids = neighbor_index_cache.read_uuid_map(uuid_map_fpath, 0)
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))  # NOQA
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    # uuid_map_fpath = join(flann_cachedir, 'uuid_map.shelf')
    # uuid_map = shelve.open(uuid_map_fpath)
    # uuid_map[daids_hashid] = visual_uuid_list
    # visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
    # visual_uuid_list
    # %timeit neighbor_index_cache.request_wbia_nnindexer(qreq_, use_memcache=False)
    # %timeit neighbor_index_cache.request_wbia_nnindexer(qreq_, use_memcache=True)

    # for uuids in uuid_set
    #    if


def trytest_multiple_add_removes():
    r"""
    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --exec-test_multiple_add_removes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> result = test_multiple_add_removes()
        >>> print(result)
    """
    from wbia.algo.hots.neighbor_index_cache import test_nnindexer

    K = 4
    nnindexer, qreq_, ibs = test_nnindexer('PZ_MTEST', use_memcache=False)

    assert len(nnindexer.get_removed_idxs()) == 0
    print('\n\n --- got nnindex testdata --- ')
    print('')

    @ut.tracefunc_xml
    def print_nnindexer(nnindexer):
        print('nnindexer.get_indexed_aids() = %r' % (nnindexer.get_indexed_aids(),))
        print('nnindexer.num_indexed_vecs() = %r' % (nnindexer.num_indexed_vecs(),))
        print(
            'nnindexer.get_removed_idxs().shape = %r'
            % (nnindexer.get_removed_idxs().shape,)
        )

    print('INITIALIZE TEST')
    print_nnindexer(nnindexer)

    config2_ = qreq_.get_internal_query_config2()
    qaid = 1
    qfx2_vec = ibs.get_annot_vecs(qaid, config2_=config2_)
    (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K)
    aids1 = set(nnindexer.get_nn_aids(qfx2_idx1).ravel())
    print('aids1 = %r' % (aids1,))

    print('')
    print('TESTING ADD')
    add_first_daids = [17, 22]
    nnindexer.add_wbia_support(qreq_, add_first_daids)
    print_nnindexer(nnindexer)
    (qfx2_idx0, qfx2_dist0) = nnindexer.knn(qfx2_vec, K)
    assert np.any(qfx2_idx0 != qfx2_idx1), 'some should change'
    aids0 = set(nnindexer.get_nn_aids(qfx2_idx0).ravel())
    print('aids0 = %r' % (aids0,))

    # execute test function
    print('')
    print('TESTING REMOVE')
    remove_daid_list = [8, 10, 11]
    nnindexer.remove_wbia_support(qreq_, remove_daid_list)
    print_nnindexer(nnindexer)
    # test after modification
    (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K)
    aids2 = set(nnindexer.get_nn_aids(qfx2_idx2).ravel())
    print('aids2 = %r' % (aids2,))
    assert len(aids2.intersection(remove_daid_list)) == 0

    __removed_ids = nnindexer.flann._FLANN__removed_ids
    invalid_idxs = nnindexer.get_removed_idxs()
    assert len(np.intersect1d(invalid_idxs, __removed_ids)) == len(__removed_ids)

    print('')
    print('TESTING DUPLICATE REMOVE')
    nnindexer.remove_wbia_support(qreq_, remove_daid_list)
    print_nnindexer(nnindexer)
    # test after modification
    (qfx2_idx2_, qfx2_dist2_) = nnindexer.knn(qfx2_vec, K)
    assert np.all(qfx2_idx2_ == qfx2_idx2)
    assert np.all(qfx2_dist2_ == qfx2_dist2)

    print('')
    print('TESTING ADD AFTER REMOVE')
    # Is the error here happening because added points seem to
    # get the ids of the removed points?
    new_daid_list = [8, 10]
    nnindexer.add_wbia_support(qreq_, new_daid_list)
    print_nnindexer(nnindexer)
    # test after modification
    (qfx2_idx3, qfx2_dist3) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid3 = nnindexer.get_nn_aids(qfx2_idx3)
    found_removed_idxs = np.intersect1d(qfx2_idx3, nnindexer.get_removed_idxs())
    if len(found_removed_idxs) != 0:
        print('found_removed_idxs.max() = %r' % (found_removed_idxs.max(),))
        print('found_removed_idxs.min() = %r' % (found_removed_idxs.min(),))
        raise AssertionError(
            'found_removed_idxs.shape = %r' % (found_removed_idxs.shape,)
        )
    aids3 = set(qfx2_aid3.ravel())
    assert aids3.intersection(remove_daid_list) == set(new_daid_list).intersection(
        remove_daid_list
    )

    print('TESTING DUPLICATE ADD')
    new_daid_list = [8, 10]
    nnindexer.add_wbia_support(qreq_, new_daid_list)
    # test after modification
    print_nnindexer(nnindexer)
    (qfx2_idx3_, qfx2_dist3_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid3_ = nnindexer.get_nn_aids(qfx2_idx3_)
    assert np.all(qfx2_aid3 == qfx2_aid3_)

    print('TESTING ADD QUERY TO DATABASE')
    add_daid_list1 = [qaid]
    nnindexer.add_wbia_support(qreq_, add_daid_list1)
    print_nnindexer(nnindexer)
    (qfx2_idx4_, qfx2_dist4_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid4_ = nnindexer.get_nn_aids(qfx2_idx4_)
    qfx2_fx4_ = nnindexer.get_nn_featxs(qfx2_idx4_)
    assert np.all(qfx2_aid4_.T[0] == qaid), 'should find self'
    assert ut.issorted(qfx2_fx4_.T[0]), 'should be in order'

    print('TESTING REMOVE QUERY POINTS')
    add_daid_list1 = [qaid]
    nnindexer.remove_wbia_support(qreq_, add_daid_list1)
    print_nnindexer(nnindexer)
    (qfx2_idx5_, qfx2_dist5_) = nnindexer.knn(qfx2_vec, K)
    issame = qfx2_idx5_ == qfx2_idx3_
    percentsame = issame.sum() / issame.size
    print('percentsame = %r' % (percentsame,))
    assert (
        percentsame > 0.85
    ), 'a large majority of the feature idxs should remain the same'

    print_nnindexer(nnindexer)

    # Do this multiple times
    for _ in range(10):
        add_daid_list1 = [qaid]
        nnindexer.add_wbia_support(qreq_, add_daid_list1, verbose=False)
        nnindexer.remove_wbia_support(qreq_, add_daid_list1, verbose=False)
        (qfx2_idxX_, qfx2_distX_) = nnindexer.knn(qfx2_vec, K)
        issame = qfx2_idxX_ == qfx2_idx3_
        percentsame = issame.sum() / issame.size
        print('percentsame = %r' % (percentsame,))
        assert (
            percentsame > 0.85
        ), 'a large majority of the feature idxs should remain the same'

    # Test again with more data
    print('testing remove query points with more data')
    nnindexer.add_wbia_support(qreq_, ibs.get_valid_aids())
    (qfx2_idx6_, qfx2_dist6_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid6_ = nnindexer.get_nn_aids(qfx2_idx6_)
    assert np.all(qfx2_aid6_.T[0] == qaid), 'should be same'

    nnindexer.remove_wbia_support(qreq_, add_daid_list1)
    print_nnindexer(nnindexer)
    (qfx2_idx7_, qfx2_dist6_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid7_ = nnindexer.get_nn_aids(qfx2_idx7_)
    assert np.all(qfx2_aid7_.T[0] != qaid), 'should not be same'

    # Do this multiple times
    for _ in range(10):
        add_daid_list1 = [qaid]
        nnindexer.add_wbia_support(qreq_, add_daid_list1, verbose=True)
        nnindexer.remove_wbia_support(qreq_, add_daid_list1, verbose=True)
        # weird that all seem to work here
        (qfx2_idxX_, qfx2_distX_) = nnindexer.knn(qfx2_vec, K)
        issame = qfx2_idxX_ == qfx2_idx7_
        percentsame = issame.sum() / issame.size
        print('percentsame = %r' % (percentsame,))
        print_nnindexer(nnindexer)
        assert (
            percentsame > 0.85
        ), 'a large majority of the feature idxs should remain the same'

    nnindexer, qreq_, ibs = test_nnindexer('PZ_MTEST', use_memcache=False)
    big_set = ibs.get_valid_aids()[5:]
    remove_later = big_set[10:14]
    nnindexer.add_wbia_support(qreq_, big_set)

    # Try again where remove is not the last operation
    print('testing remove query points with more op')
    extra_data = np.setdiff1d(ibs.get_valid_aids()[0:5], add_daid_list1)
    nnindexer.remove_wbia_support(qreq_, extra_data)

    nnindexer.add_wbia_support(qreq_, add_daid_list1)
    nnindexer.add_wbia_support(qreq_, extra_data)

    (qfx2_idx8_, qfx2_dist8_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid8_ = nnindexer.get_nn_aids(qfx2_idx8_)
    assert np.all(qfx2_aid8_.T[0] == qaid), 'should be same'

    nnindexer.remove_wbia_support(qreq_, extra_data)
    (qfx2_idx9_, qfx2_dist9_) = nnindexer.knn(qfx2_vec, K)
    qfx2_aid9_ = nnindexer.get_nn_aids(qfx2_idx9_)
    assert np.all(qfx2_aid9_.T[0] == qaid), 'should be same'
    nnindexer.remove_wbia_support(qreq_, add_daid_list1)

    nnindexer.add_wbia_support(qreq_, add_daid_list1)
    nnindexer.add_wbia_support(qreq_, extra_data)
    nnindexer.remove_wbia_support(qreq_, remove_later)
    print(nnindexer.ax2_aid)

    aid_list = nnindexer.get_indexed_aids()  # NOQA
    nnindexer.flann.save_index('test.flann')

    idx2_vec_masked = nnindexer.idx2_vec
    idx2_vec_compressed = nnindexer.get_indexed_vecs()

    from vtool._pyflann_backend import pyflann as pyflann

    flann1 = pyflann.FLANN()
    flann1.load_index('test.flann', idx2_vec_masked)

    from vtool._pyflann_backend import pyflann as pyflann

    flann2 = pyflann.FLANN()
    flann2.load_index('test.flann', idx2_vec_compressed)

    # NOW WE NEED TO TEST THAT WE CAN SAVE AND LOAD THIS DATA

    #
    # ax2_nvecs = ut.dict_take(ut.dict_hist(nnindexer.idx2_ax), range(len(nnindexer.ax2_aid)))
    pass


def pyflann_test_remove_add():
    r"""
    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --exec-pyflann_test_remove_add

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> pyflann_test_remove_add()
    """
    from vtool._pyflann_backend import pyflann as pyflann
    import numpy as np

    rng = np.random.RandomState(0)

    print('Test initial save load')
    flann_params = {
        'random_seed': 42,
        # 'log_level': 'debug', 'info',
    }

    # pyflann.flann_ctypes.flannlib.flann_log_verbosity(4)

    print('Test remove and then add disjoint points')
    flann = pyflann.FLANN()
    vecs = (rng.rand(400, 128) * 255).astype(np.uint8)
    flann.build_index(vecs, **flann_params)  # NOQA

    remove_idxs = np.arange(0, len(vecs), 2)
    flann.remove_points(remove_idxs)

    vecs2 = (rng.rand(100, 128) * 255).astype(np.uint8)
    flann.add_points(vecs2)

    all_vecs = flann._get_stacked_data()
    idx_all, dist_all = flann.nn_index(all_vecs, 3)

    nonzero_idxs = np.nonzero(dist_all.T[0] != 0)[0]
    removed_idxs = flann.get_removed_ids()
    assert np.all(nonzero_idxs == removed_idxs)
    print('removed correctly indexes has nonzero dists')
    nonself_idxs = np.nonzero(np.arange(len(idx_all)) != idx_all.T[0])[0]
    assert np.all(nonself_idxs == removed_idxs)
    print('removed indexexes were only ones whos nearest neighbor was not self')


def pyflann_test_remove_add2():
    r"""
    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --exec-pyflann_test_remove_add2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> pyflann_test_remove_add2()
    """
    from vtool._pyflann_backend import pyflann as pyflann
    import numpy as np

    rng = np.random.RandomState(0)
    vecs = (rng.rand(400, 128) * 255).astype(np.uint8)

    print('Test initial save load')
    flann_params = {
        'random_seed': 42,
        'log_level': 'debug',
    }

    # pyflann.flann_ctypes.flannlib.flann_log_verbosity(4)

    print('Test remove and then add THE SAME points')
    flann = pyflann.FLANN()
    flann.build_index(vecs, **flann_params)  # NOQA

    remove_idxs = np.arange(0, len(vecs), 2)
    flann.remove_points(remove_idxs)

    vecs2 = vecs[remove_idxs[0:100]]
    flann.add_points(vecs2)

    all_vecs = flann._get_stacked_data()
    idx_all, dist_all = flann.nn_index(all_vecs, 3)

    removed_idxs = flann.get_removed_ids()
    nonself_idxs = np.nonzero(np.arange(len(idx_all)) != idx_all.T[0])[0]
    assert np.all(nonself_idxs == removed_idxs)
    print('removed indexexes were only ones whos nearest neighbor was not self')
    assert np.all(
        idx_all.T[0][-len(vecs2) :] == np.arange(len(vecs), len(vecs) + len(vecs2))
    )
    print('added vecs correctly got their padded index')
    assert idx_all.T[0].max() == 499


def pyflann_remove_and_save():
    """
    References:
        # Logic goes here
        ~/code/flann/src/cpp/flann/algorithms/kdtree_index.h

        ~/code/flann/src/cpp/flann/util/serialization.h
        ~/code/flann/src/cpp/flann/util/dynamic_bitset.h

        # Bindings go here
        ~/code/flann/src/cpp/flann/flann.cpp
        ~/code/flann/src/cpp/flann/flann.h

        # Contains stuff for the flann namespace like flann::log_level
        # Also has Index with
        # Matrix<ElementType> features; SEEMS USEFUL
        ~/code/flann/src/cpp/flann/flann.hpp


        # Wrappers go here
        ~/code/flann/src/python/pyflann/flann_ctypes.py
        ~/code/flann/src/python/pyflann/index.py

        ~/local/build_scripts/flannscripts/autogen_bindings.py

    Greping:
        cd ~/code/flann/src
        grep -ER cleanRemovedPoints *
        grep -ER removed_points_ *

    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment --exec-pyflann_remove_and_save

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots._neighbor_experiment import *  # NOQA
        >>> pyflann_remove_and_save()
    """
    from vtool._pyflann_backend import pyflann as pyflann
    import numpy as np

    rng = np.random.RandomState(0)
    vecs = (rng.rand(400, 128) * 255).astype(np.uint8)
    vecs2 = (rng.rand(100, 128) * 255).astype(np.uint8)
    qvecs = (rng.rand(10, 128) * 255).astype(np.uint8)

    ut.delete('test1.flann')
    ut.delete('test2.flann')
    ut.delete('test3.flann')
    ut.delete('test4.flann')

    print('\nTest initial save load')
    flann_params = {
        'random_seed': 42,
        # 'log_level': 'debug', 'info',
        # 'log_level': 4,
        'cores': 1,
        'log_level': 'debug',
    }

    # pyflann.flann_ctypes.flannlib.flann_log_verbosity(4)

    flann1 = pyflann.FLANN(**flann_params)
    params1 = flann1.build_index(vecs, **flann_params)  # NOQA
    idx1, dist = flann1.nn_index(qvecs, 3)
    flann1.save_index('test1.flann')

    flann1_ = pyflann.FLANN()
    flann1_.load_index('test1.flann', vecs)
    idx1_, dist = flann1.nn_index(qvecs, 3)
    assert np.all(idx1 == idx1_), 'initial save load fail'

    print('\nTEST ADD SAVE LOAD')
    flann2 = flann1
    flann2.add_points(vecs2)
    idx2, dist = flann2.nn_index(qvecs, 3)
    assert np.any(idx2 != idx1), 'something should change'
    flann2.save_index('test2.flann')

    # Load saved data with added vecs
    tmp = flann2.get_indexed_data()
    vecs_combined = np.vstack([tmp[0]] + tmp[1])

    flann2_ = pyflann.FLANN()
    flann2_.load_index('test2.flann', vecs_combined)
    idx2_, dist = flann2_.nn_index(qvecs, 3)
    assert np.all(idx2_ == idx2), 'loading saved added data fails'

    # Load saved data with remoed vecs
    print('\n\n---TEST REMOVE SAVE LOAD')
    flann1 = pyflann.FLANN()  # rebuild flann1
    _params1 = flann1.build_index(vecs, **flann_params)  # NOQA
    print('\n * CHECK NN')
    _idx1, dist = flann1.nn_index(qvecs, 3)
    idx1 = _idx1

    print('\n * REMOVE POINTS')
    remove_idx_list = np.unique(idx1.T[0][0:10])
    flann1.remove_points(remove_idx_list)
    flann3 = flann1
    print('\n * CHECK NN')
    idx3, dist = flann3.nn_index(qvecs, 3)
    assert (
        len(np.intersect1d(idx3.ravel(), remove_idx_list)) == 0
    ), 'points were not removed'
    print('\n * SAVE')
    flann3.save_index('test3.flann')

    print('\n\n---TEST LOAD SAVED INDEX 0 (with removed points)')
    clean_vecs = np.delete(vecs, remove_idx_list, axis=0)
    flann3.clean_removed_points()
    flann3.save_index('test4.flann')
    flann4 = pyflann.FLANN(**flann_params)
    # THIS CAUSES A SEGFAULT
    flann4.load_index('test4.flann', clean_vecs)
    idx4, dist = flann4.nn_index(qvecs, 3)
    assert np.all(idx4 == idx3), 'load failed'
    print('\nloaded succesfully (WITHOUT THE BAD DATA)')

    print('\n\n---TEST LOAD SAVED INDEX 1 (with removed points)')
    flann4 = pyflann.FLANN(**flann_params)
    flann4.load_index('test3.flann', vecs)
    idx4, dist = flann4.nn_index(qvecs, 3)
    assert np.all(idx4 == idx3), 'load failed'
    print('\nloaded succesfully (BUT NEED TO MAINTAIN BAD DATA)')

    if False:
        print('\n\n---TEST LOAD SAVED INDEX 2 (with removed points)')
        clean_vecs = np.delete(vecs, remove_idx_list, axis=0)
        flann4 = pyflann.FLANN(**flann_params)
        print('\n * CALL LOAD')
        flann4.load_index('test3.flann', clean_vecs)

    # assert np.all(idx1 == _idx1), 'rebuild is not determenistic!'


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots._neighbor_experiment
        python -m wbia.algo.hots._neighbor_experiment --allexamples
        python -m wbia.algo.hots._neighbor_experiment --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    if ut.get_argflag('--test-augment_nnindexer_experiment'):
        # See if exec has something to do with memory leaks
        augment_nnindexer_experiment()
        ut.show_if_requested()
        pass
    else:
        ut.doctest_funcs()
