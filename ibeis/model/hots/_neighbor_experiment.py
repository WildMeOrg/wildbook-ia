from __future__ import absolute_import, division, print_function
import six  # NOQA
import numpy as np
import utool as ut
import pyflann
from os.path import basename, exists  # NOQA
from six.moves import range
from ibeis.model.hots import neighbor_index
#import mem_top

#import vtool.nearest_neighbors as nntool
#from ibeis.model.hots import hstypes
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[neighbor_experiment]', DEBUG=False)


def augment_nnindexer_experiment():
    """

    python -c "import utool; print(utool.auto_docstr('ibeis.model.hots._neighbor_experiment', 'augment_nnindexer_experiment'))"

    References:
        http://answers.opencv.org/question/44592/flann-index-in-python-training-fails-with-segfault/

    CommandLine:
        utprof.py -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment
        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment

        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_MTEST --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6
        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --show
        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --show


        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6 --nosave-flann --no-api-cache --nocache-uuids

        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_MTEST --show
        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show

        # RUNS THE SEGFAULTING CASE
        python -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show
        # Debug it
        gdb python
        run -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --show
        gdb python
        run -m ibeis.model.hots._neighbor_experiment --test-augment_nnindexer_experiment --db PZ_Master0 --diskshow --adjust=.1 --save "augment_experiment_{db}.png" --dpath='.' --dpi=180 --figsize=9,6


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots._neighbor_experiment import *  # NOQA
        >>> # execute function
        >>> augment_nnindexer_experiment()
        >>> # verify results
        >>> ut.show_if_requested()

    """
    import ibeis
    # build test data
    ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
    #ibs = ibeis.opendb('PZ_MTEST')
    ibs = ibeis.opendb(defaultdb='PZ_Master0')
    if ibs.get_dbname() == 'PZ_MTEST':
        initial = 1
        addition_stride = 4
        max_ceiling = 100
    elif ibs.get_dbname() == 'PZ_Master0':
        initial = 128
        #addition_stride = 64
        #addition_stride = 128
        addition_stride = 256
        max_ceiling = 10000
        #max_ceiling = 4000
        #max_ceiling = 2000
        #max_ceiling = 600
    else:
        assert False
    all_daids = ibs.get_valid_aids(species=ZEB_PLAIN)
    qreq_ = ibs.new_query_request(all_daids, all_daids)
    max_num = min(max_ceiling, len(all_daids))

    # Clear Caches
    ibs.delete_flann_cachedir()
    neighbor_index.clear_memcache()
    neighbor_index.clear_uuid_cache(qreq_)

    # Setup
    all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    # ensure all features are computed
    #ibs.get_annot_vecs(all_randomize_daids_, ensure=True)
    #ibs.get_annot_fgweights(all_randomize_daids_, ensure=True)

    nnindexer_list = []
    addition_lbl = 'Addition'
    _addition_iter = list(range(initial + 1, max_num, addition_stride))
    addition_iter = iter(ut.ProgressIter(_addition_iter, lbl=addition_lbl, freq=1, autoadjust=False))
    time_list_addition = []
    #time_list_reindex = []
    addition_count_list = []
    tmp_cfgstr_list = []

    #for _ in range(80):
    #    next(addition_iter)
    try:
        memtrack = ut.MemoryTracker(disable=False)
        for count in addition_iter:
            aid_list_ = all_randomize_daids_[0:count]
            # Request an indexer which could be an augmented version of an existing indexer.
            with ut.Timer(verbose=False) as t:
                memtrack.report('BEFORE AUGMENT')
                nnindexer_ = neighbor_index.request_augmented_ibeis_nnindexer(qreq_, aid_list_)
                memtrack.report('AFTER AUGMENT')
            nnindexer_list.append(nnindexer_)
            addition_count_list.append(count)
            time_list_addition.append(t.ellapsed)
            tmp_cfgstr_list.append(nnindexer_.cfgstr)
            print('===============\n\n')
        print(ut.list_str(time_list_addition))
        print(ut.list_str(list(map(id, nnindexer_list))))
        print(ut.list_str(tmp_cfgstr_list))
        print(ut.list_str(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))

        IS_SMALL = False

        if IS_SMALL:
            nnindexer_list = []
        reindex_label = 'Reindex'
        _reindex_iter = list(range(initial + 1, max_num, addition_stride))[::-1]  # go backwards for reindex
        reindex_iter = ut.ProgressIter(_reindex_iter, lbl=reindex_label)
        time_list_reindex = []
        #time_list_reindex = []
        reindex_count_list = []

        for count in reindex_iter:
            print('\n+===PREDONE====================\n')
            # check only a single size for memory leaks
            #count = max_num // 16 + ((x % 6) * 1)
            #x += 1

            aid_list_ = all_randomize_daids_[0:count]
            # Call the same code, but force rebuilds
            memtrack.report('BEFORE REINDEX')
            with ut.Timer(verbose=False) as t:
                nnindexer_ = neighbor_index.request_augmented_ibeis_nnindexer(qreq_, aid_list_, force_rebuild=True, memtrack=memtrack)
            memtrack.report('AFTER REINDEX')
            ibs.print_cachestats_str()
            print('[nnindex.MEMCACHE] size(NEIGHBOR_CACHE) = %s' % (ut.get_object_size_str(neighbor_index.NEIGHBOR_CACHE.items()),))
            print('[nnindex.MEMCACHE] len(NEIGHBOR_CACHE) = %s' % (len(neighbor_index.NEIGHBOR_CACHE.items()),))
            print('[nnindex.MEMCACHE] size(UUID_MAP_CACHE) = %s' % (ut.get_object_size_str(neighbor_index.UUID_MAP_CACHE),))
            print('totalsize(nnindexer) = ' + ut.get_object_size_str(nnindexer_))
            memtrack.report_type(neighbor_index.NeighborIndex)
            ut.print_object_size_tree(nnindexer_, lbl='nnindexer_')
            if IS_SMALL:
                nnindexer_list.append(nnindexer_)
            reindex_count_list.append(count)
            time_list_reindex.append(t.ellapsed)
            #import cv2
            #import matplotlib as mpl
            #print(mem_top.mem_top(limit=30, width=120,
            #                      #exclude_refs=[cv2.__dict__, mpl.__dict__]
            #     ))
            print('L___________________\n\n\n')
        print(ut.list_str(time_list_reindex))
        if IS_SMALL:
            print(ut.list_str(list(map(id, nnindexer_list))))
            print(ut.list_str(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))
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

    import plottool as pt

    next_fnum = iter(range(0, 1)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    if len(addition_count_list) > 0:
        pt.plot2(addition_count_list, time_list_addition, marker='-o', equal_aspect=False,
                 x_label='num_annotations', label=addition_lbl + ' Time')

    if len(reindex_count_list) > 0:
        pt.plot2(reindex_count_list, time_list_reindex, marker='-o', equal_aspect=False,
                 x_label='num_annotations', label=reindex_label + ' Time')

    pt.set_figtitle('Augmented indexer experiment')

    pt.legend()


def flann_add_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment

    CommandLine:
        python -m ibeis.model.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_MTEST --show
        python -m ibeis.model.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_Master0 --show
        utprof.py -m ibeis.model.hots._neighbor_experiment --test-flann_add_time_experiment --show

        valgrind --tool=memcheck --suppressions=valgrind-python.supp python -m ibeis.model.hots._neighbor_experiment --test-flann_add_time_experiment --db PZ_MTEST --no-with-reindex

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots._neighbor_experiment import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> result = flann_add_time_experiment()
        >>> # verify results
        >>> print(result)
        >>> ut.show_if_requested()

    """
    import ibeis
    import utool as ut
    import numpy as np
    import plottool as pt

    def make_flann_index(vecs, flann_params):
        flann = pyflann.FLANN()
        flann.build_index(vecs, **flann_params)
        return flann

    db = ut.get_argval('--db')
    ibs = ibeis.opendb(db=db)

    # Input
    if ibs.get_dbname() == 'PZ_MTEST':
        initial = 1
        reindex_stride = 16
        addition_stride = 4
        max_ceiling = 120
    elif ibs.get_dbname() == 'PZ_Master0':
        #ibs = ibeis.opendb(db='GZ_ALL')
        initial = 32
        reindex_stride = 32
        addition_stride = 16
        max_ceiling = 300001
    else:
        assert False
    #max_ceiling = 32
    all_daids = ibs.get_valid_aids()
    max_num = min(max_ceiling, len(all_daids))
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()

    # Output
    count_list,  time_list_reindex  = [], []
    count_list2, time_list_addition = [], []

    # Setup
    #all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    all_randomize_daids_ = all_daids
    # ensure all features are computed
    ibs.get_annot_vecs(all_randomize_daids_)

    def reindex_step(count, count_list, time_list_reindex):
        daids    = all_randomize_daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = make_flann_index(vecs, flann_params)  # NOQA
        count_list.append(count)
        time_list_reindex.append(t.ellapsed)

    def addition_step(count, flann, count_list2, time_list_addition):
        daids = all_randomize_daids_[count:count + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann.add_points(vecs)
        count_list2.append(count)
        time_list_addition.append(t.ellapsed)

    def make_initial_index(initial):
        daids = all_randomize_daids_[0:initial + 1]
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

    #with pt.FigureContext:

    next_fnum = iter(range(0, 2)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    if WITH_REINDEX:
        pt.plot2(count_list, time_list_reindex, marker='-o', equal_aspect=False,
                 x_label='num_annotations', label=reindex_lbl + ' Time', dark=False)

    #pt.figure(fnum=next_fnum())
    pt.plot2(count_list2, time_list_addition, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=addition_lbl + ' Time')

    pt
    pt.legend()


def subindexer_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment
    """
    import ibeis
    import utool as ut
    import pyflann
    import plottool as pt
    ibs = ibeis.opendb(db='PZ_Master0')
    daid_list = ibs.get_valid_aids()
    count_list = []
    time_list = []
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()
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
    pt.plot2(count_arr, time_arr, marker='-', equal_aspect=False,
             x_label='num_annotations', y_label='FLANN build time')
    #pt.update()


def test_incremental_add(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.model.hots._neighbor_experiment --test-test_incremental_add

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = test_incremental_add(ibs)
        >>> print(result)
    """
    sample_aids = ibs.get_annot_rowid_sample()
    aids1 = sample_aids[::2]
    aids2 = sample_aids[0:5]
    aids3 = sample_aids[:-1]  # NOQA
    daid_list = aids1  # NOQA
    qreq_ = ibs.new_query_request(aids1, aids1)
    nnindexer1 = neighbor_index.request_ibeis_nnindexer(ibs.new_query_request(aids1, aids1))  # NOQA
    nnindexer2 = neighbor_index.request_ibeis_nnindexer(ibs.new_query_request(aids2, aids2))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(aids3)
    uuid_map_fpath = neighbor_index.get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = neighbor_index.read_uuid_map(uuid_map_fpath, 0)
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    nnindexer3 = neighbor_index.request_ibeis_nnindexer(ibs.new_query_request(uncovered_aids, uncovered_aids))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(sample_aids)
    uuid_map_fpath = neighbor_index.get_nnindexer_uuid_map_fpath(qreq_)
    #contextlib.closing(shelve.open(uuid_map_fpath)) as uuid_map:
    candidate_uuids = neighbor_index.read_uuid_map(uuid_map_fpath, 0)
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))  # NOQA
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    #uuid_map_fpath = join(flann_cachedir, 'uuid_map.shelf')
    #uuid_map = shelve.open(uuid_map_fpath)
    #uuid_map[daids_hashid] = visual_uuid_list
    #visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
    #visual_uuid_list
    #%timeit neighbor_index.request_ibeis_nnindexer(qreq_, use_memcache=False)
    #%timeit neighbor_index.request_ibeis_nnindexer(qreq_, use_memcache=True)

    #for uuids in uuid_set
    #    if


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots._neighbor_experiment
        python -m ibeis.model.hots._neighbor_experiment --allexamples
        python -m ibeis.model.hots._neighbor_experiment --allexamples --noface --nosrc
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
