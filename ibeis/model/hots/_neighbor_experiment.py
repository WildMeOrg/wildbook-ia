from __future__ import absolute_import, division, print_function
import six  # NOQA
import numpy as np
import utool as ut
import pyflann
from os.path import basename, exists  # NOQA
from six.moves import range
from ibeis.model.hots import neighbor_index
#import vtool.nearest_neighbors as nntool
#from ibeis.model.hots import hstypes
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[neighbor_experiment]', DEBUG=False)


def flann_add_time_experiment(update=False):
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-flann_add_time_experiment
        utprof.py -m ibeis.model.hots.neighbor_index --test-flann_add_time_experiment

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> update = True
        >>> result = flann_add_time_experiment(update)
        >>> # verify results
        >>> print(result)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
        #>>> ibeis.main_loop({'ibs': ibs, 'back': None})

    """
    import ibeis
    import utool as ut
    import numpy as np
    import plottool as pt

    def make_flann_index(vecs, flann_params):
        flann = pyflann.FLANN()
        flann.build_index(vecs, **flann_params)
        return flann

    def get_reindex_time(ibs, daids, flann_params):
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = make_flann_index(vecs, flann_params)  # NOQA
        return t.ellapsed

    def get_addition_time(ibs, daids, flann, flann_params):
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann.add_points(vecs)
        return t.ellapsed

    # Input
    #ibs = ibeis.opendb(db='PZ_MTEST')
    #ibs = ibeis.opendb(db='GZ_ALL')
    ibs = ibeis.opendb(db='PZ_Master0')
    #max_ceiling = 32
    initial = 32
    reindex_stride = 32
    addition_stride = 16
    max_ceiling = 300001
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
        ellapsed = get_reindex_time(ibs, daids, flann_params)
        count_list.append(count)
        time_list_reindex.append(ellapsed)

    def addition_step(count, flann, count_list2, time_list_addition):
        daids = all_randomize_daids_[count:count + 1]
        ellapsed = get_addition_time(ibs, daids, flann, flann_params)
        count_list2.append(count)
        time_list_addition.append(ellapsed)

    def make_initial_index(initial):
        daids = all_randomize_daids_[0:initial + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        flann = make_flann_index(vecs, flann_params)
        return flann

    # Reindex Part
    reindex_lbl = 'Reindexing'
    _reindex_iter = range(1, max_num, reindex_stride)
    reindex_iter = ut.ProgressIter(_reindex_iter, lbl=reindex_lbl)
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
    pt.plot2(count_list, time_list_reindex, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=reindex_lbl + ' Time')

    #pt.figure(fnum=next_fnum())
    pt.plot2(count_list2, time_list_addition, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=addition_lbl + ' Time')

    pt
    pt.legend()
    if update:
        pt.update()


def augment_nnindexer_experiment(update=True):
    """

    python -c "import utool; print(utool.auto_docstr('ibeis.model.hots.neighbor_index', 'augment_nnindexer_experiment'))"

    CommandLine:
        utprof.py -m ibeis.model.hots.neighbor_index --test-augment_nnindexer_experiment
        python -m ibeis.model.hots.neighbor_index --test-augment_nnindexer_experiment

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> # build test data
        >>> show = ut.get_argflag('--show')
        >>> update = show
        >>> # execute function
        >>> augment_nnindexer_experiment(update)
        >>> # verify results
        >>> if show:
        ...     from matplotlib import pyplot as plt
        ...     plt.show()

    """
    import ibeis
    import plottool as pt
    # build test data
    ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
    #ibs = ibeis.opendb('PZ_MTEST')
    ibs = ibeis.opendb('PZ_Master0')
    all_daids = ibs.get_valid_aids(species=ZEB_PLAIN)
    qreq_ = ibs.new_query_request(all_daids, all_daids)
    initial = 128
    addition_stride = 64
    max_ceiling = 10000
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
    _addition_iter = range(initial + 1, max_num, addition_stride)
    addition_iter = ut.ProgressIter(_addition_iter, lbl=addition_lbl)
    time_list_addition = []
    #time_list_reindex = []
    count_list = []
    for count in addition_iter:
        aid_list_ = all_randomize_daids_[0:count]
        with ut.Timer(verbose=False) as t:
            nnindexer_ = neighbor_index.request_augmented_ibeis_nnindexer(qreq_, aid_list_)
        nnindexer_list.append(nnindexer_)
        count_list.append(count)
        time_list_addition.append(t.ellapsed)
        print('===============\n\n')
    print(ut.list_str(time_list_addition))
    print(ut.list_str(list(map(id, nnindexer_list))))
    print(ut.list_str(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))

    next_fnum = iter(range(0, 1)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    pt.plot2(count_list, time_list_addition, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=addition_lbl + ' Time')
    pt.legend()
    if update:
        pt.update()


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
    pt.update()


def test_incremental_add(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-test_incremental_add

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
    nnindexer1 = request_ibeis_nnindexer(ibs.new_query_request(aids1, aids1))  # NOQA
    nnindexer2 = request_ibeis_nnindexer(ibs.new_query_request(aids2, aids2))  # NOQA

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

    nnindexer3 = request_ibeis_nnindexer(ibs.new_query_request(uncovered_aids, uncovered_aids))  # NOQA

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
    #%timeit request_ibeis_nnindexer(qreq_, use_memcache=False)
    #%timeit request_ibeis_nnindexer(qreq_, use_memcache=True)

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
    ut.doctest_funcs()
