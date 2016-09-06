# -*- coding: utf-8 -*-
# LICENCE
"""
TODO:
    Does HDBSCAN work on 128 dim vectors?
    http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, zip, map
import six
import utool
import utool as ut
import sys
import numpy as np
import scipy.sparse as spsparse
import vtool.nearest_neighbors as nntool

(print, rrr, profile) = utool.inject2(__name__, '[clustering2]')


CLUSTERS_FNAME = 'akmeans_centroids'


def get_akmeans_cfgstr(data, nCentroids, max_iters=5, initmethod='akmeans++', flann_params={},
                       use_data_hash=True, cfgstr='', akmeans_cfgstr=None):
    if akmeans_cfgstr is None:
        # compute a hashstr based on the data
        cfgstr += '_nC=%d,nIter=%d,init=%s' % (nCentroids, max_iters, initmethod)
        akmeans_cfgstr = nntool.get_flann_cfgstr(data, flann_params,
                                                 cfgstr, use_data_hash)
    return akmeans_cfgstr


def assert_centroids(centroids, data, nCentroids, clip_centroids):
    dbgkeys = ['centroids.shape', 'nCentroids', 'data.shape', ]
    try:
        assert centroids.shape[0] == nCentroids, 'bad number of centroids'
    except Exception as ex:
        utool.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
        if not clip_centroids:
            raise
    try:
        assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
    except Exception as ex:
        utool.printex(ex, keys=dbgkeys)
        raise


def cached_akmeans(data, nCentroids, max_iters=5, flann_params={},
                   cache_dir='default', force_recomp=False, use_data_hash=True,
                   cfgstr='', refine=False, akmeans_cfgstr=None, use_cache=True,
                   appname='vtool',  initmethod='akmeans++', clip_centroids=True):
    """ precompute aproximate kmeans with builtin caching

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nump = 100000
        >>> dims = 128
        >>> nCentroids = 800
        >>> max_iters = 300
        >>> dtype = np.uint8
        >>> data = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)

    Timeit:
        import vtool.clustering2 as clustertool
        max_iters = 300
        flann = p yflann.FLANN()
        centroids1 = flann.kmeans(data, nCentroids, max_iterations=max_iters, dtype=np.uint8)
        centroids2 = clustertool.akmeans(data, nCentroids, max_iters, {})
        %timeit clustertool.akmeans(data, nCentroids, max_iters, {})
        %timeit flann.kmeans(data, nCentroids, max_iterations=max_iters)
    """
    if data.shape[0] < nCentroids:
        dbgkeys = ['centroids.shape', 'nCentroids', 'data.shape', ]
        ex = AssertionError('less data than centroids')
        utool.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
        if not clip_centroids:
            raise ex
        else:
            nCentroids = data.shape[0]
    print('+--- START CACHED AKMEANS')
    # filename prefix constants
    if cache_dir == 'default':
        print('[akmeans] using default cache dir')
        cache_dir = utool.get_app_resource_dir(appname)
        utool.ensuredir(cache_dir)
    # Build a cfgstr if the full one is not specified
    akmeans_cfgstr = get_akmeans_cfgstr(data, nCentroids, max_iters,
                                        initmethod, flann_params,
                                        use_data_hash, cfgstr, akmeans_cfgstr) + initmethod
    try:
        # Try and load a previous centroiding
        if not use_cache or force_recomp:
            raise UserWarning('forceing recommpute')
        centroids = utool.load_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr)
        print('[akmeans.precompute] load successful')
        if refine:
            # Refines the centroid centers if specified
            centroids = refine_akmeans(data, centroids, max_iters=max_iters,
                                       flann_params=flann_params,
                                       cache_dir=cache_dir,
                                       akmeans_cfgstr=akmeans_cfgstr)
        try:
            assert centroids.shape[0] == nCentroids, 'bad number of centroids'
        except Exception as ex:
            utool.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
            if not clip_centroids:
                raise
        try:
            assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
        except Exception as ex:
            utool.printex(ex, keys=dbgkeys)
            raise
        print('L___ END CACHED AKMEANS')
        return centroids
    except IOError as ex:
        utool.printex(ex, 'cache miss', iswarning=True)
    except UserWarning:
        pass
    # First time computation
    print('[akmeans.precompute] pre_akmeans(): calling akmeans')
    # FLANN.AKMEANS IS NOT APPROXIMATE KMEANS
    #if use_external_kmeans:
    #    import p yflann
    #    #import utool
    #    print('[akmeans.precompute] using flann.kmeans... (hope this is approximate)')
    #    flann = p yflann.FLANN()
    #    with utool.Timer('testing time of 1 kmeans iteration') as timer:
    #        centroids = flann.kmeans(data, nCentroids, max_iterations=1)
    #    estimated_time = max_iters * timer.ellapsed
    #    print('Current time:            ' + utool.get_timestamp('printable'))
    #    print('Estimated Total Time:    ' + utool.get_unix_timedelta_str(estimated_time))
    #    print('Estimated finish time:   ' + utool.get_timestamp('printable', delta_seconds=estimated_time))
    #    print('Begining computation...')
    #    centroids = flann.kmeans(data, nCentroids, max_iterations=max_iters)
    #    print('The true finish time is: ' + utool.get_timestamp('printable'))
    #else:
    centroids = akmeans(data, nCentroids, max_iters, initmethod, flann_params)
    assert_centroids(centroids, data, nCentroids, clip_centroids)
    print('[akmeans.precompute] save and return')
    utool.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    print('L___ END CACHED AKMEANS')
    return centroids


def tune_flann2(data):
    import pyflann
    flann = pyflann.FLANN()
    flann_atkwargs = dict(algorithm='autotuned',
                          target_precision=.6,
                          build_weight=0.01,
                          memory_weight=0.0,
                          sample_fraction=0.001)
    print(flann_atkwargs)
    print('Autotuning flann')
    tuned_params = flann.build_index(data, **flann_atkwargs)
    return tuned_params


@profile
def akmeans_plusplus_init(data, K, samples_per_iter=None, flann_params=None):
    """
    Referencs:
        http://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

    Example:
        >>> # SLOW_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> K = 8000  # 64000
        >>> nump = K * 2
        >>> dims = 128
        >>> max_iters = 300
        >>> samples_per_iter = None
        >>> dtype = np.uint8
        >>> flann_params = None
        >>> data = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)
        >>> initial_centers = akmeans_plusplus_init(data, K, samples_per_iter, flann_params)
        >>> #result = str(initial_centers.sum())

    130240088

    Example2:
        >>> # SLOW_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> data = np.vstack(ibs.get_annot_vecs(ibs.get_valid_aids()))
        >>> flann_params = None
        >>> samples_per_iter = 1000
        >>> K = 8000  # 64000
        >>> initial_centers = akmeans_plusplus_init(data, K, samples_per_iter,  flann_params)

    CommandLine:
        utprof.sh ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
        python ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
    """
    import pyflann
    if samples_per_iter is None:
        #sample_fraction = 32.0 / K
        sample_fraction = 64.0 / K
        #sample_fraction = 128.0 / K
        samples_per_iter = int(len(data) * sample_fraction)
    print('akmeans++ on %r points. samples_per_iter=%r. K=%r' % (len(data), samples_per_iter, K))
    #import random
    eps = np.sqrt(data.shape[1])
    flann = pyflann.FLANN()
    # Choose an index and "use" it
    unusedx2_datax = np.arange(len(data), dtype=np.int32)
    chosen_unusedx = np.random.randint(0, len(unusedx2_datax))
    center_indices = [unusedx2_datax[chosen_unusedx]]
    unusedx2_datax = np.delete(unusedx2_datax, chosen_unusedx)

    if flann_params is None:
        flann_params = {}
        flann_params['target_precision'] = .6
        flann_params['trees'] = 1
        flann_params['checks'] = 8
        #flann_params['algorithm'] = 'linear'
        flann_params['algorithm'] = 'kdtree'
        flann_params['iterations'] = 3

    # initalize flann index for approximate nn calculation
    centers = data.take(center_indices, axis=0)
    build_params = flann.build_index(np.array(centers), **flann_params)  # NOQA
    num_sample = min(samples_per_iter, len(data))
    progiter = utool.progiter(range(0, K), lbl='akmeans++ init', freq=200)
    _iter = progiter.iter_rate()
    six.next(_iter)

    #for count in range(1, K):
    for count in _iter:
        # Randomly choose a set of unused potential seed points
        sx2_unusedx = np.random.randint(len(unusedx2_datax), size=num_sample)
        sx2_datax = unusedx2_datax.take(sx2_unusedx)
        # Distance from a random sample of data to current centers
        # (this call takes 98% of the time. optimize here only)
        sample_data = data.take(sx2_datax, axis=0)
        sx2_dist = flann.nn_index(sample_data, 1, checks=flann_params['checks'])[1] + eps
        # Choose data sample index that has a high probability of being a new cluster
        sx2_prob = sx2_dist / sx2_dist.sum()
        chosen_sx = np.where(sx2_prob.cumsum() >= np.random.random() * .98)[0][0]
        chosen_unusedx = sx2_unusedx[chosen_sx]
        chosen_datax = unusedx2_datax[chosen_unusedx]
        # Remove the chosen index from unused indices
        unusedx2_datax = np.delete(unusedx2_datax, chosen_unusedx)
        center_indices.append(chosen_datax)
        chosen_data = data.take(chosen_datax, axis=0)
        # Append new center to data and flann index
        flann.add_points(chosen_data)
    center_indices = np.array(center_indices)
    centers = data.take(center_indices, axis=0)
    print('len(center_indices) = %r' % len(center_indices))
    print('len(set(center_indices)) = %r' % len(set(center_indices)))
    return centers


def akmeans(data, nCentroids, max_iters=5, initmethod='akmeans++',
            flann_params={}, ave_unchanged_thresh=0, ave_unchanged_iterwin=10, monitor=False):
    """
    Approximiate K-Means (using FLANN)

    Quickly partitions data into K=nCentroids centroids.  Cluster centers are
    randomly assigned to datapoints.  Each datapoint is assigned to its
    approximate nearest centroid center.  The centroid centers are recomputed.
    Repeat until approximate convergence.

    Args:
        data - np.array with rows of data.
    """
    # Setup iterations
    centroids = initialize_centroids(nCentroids, data, initmethod)
    return akmeans_iterations(data, centroids, max_iters, flann_params,
                              ave_unchanged_thresh, ave_unchanged_iterwin,
                              monitor=monitor)


def initialize_centroids(nCentroids, data, initmethod='akmeans++'):
    """ Initializes centroids to random datapoints """
    if initmethod == 'akmeans++':
        centroids = np.copy(akmeans_plusplus_init(data, nCentroids))
    elif initmethod == 'random':
        nData = data.shape[0]
        datax_rand = np.arange(0, nData, dtype=np.int32)
        np.random.shuffle(datax_rand)
        centroidx2_datax = datax_rand[0:nCentroids]
        centroids = np.copy(data[centroidx2_datax])
    else:
        raise AssertionError('Unknown initmethod=%r' % (initmethod,))
    return centroids


def refine_akmeans(data, centroids, max_iters=5,
                   flann_params={}, cache_dir='default', cfgstr='',
                   use_data_hash=True, akmeans_cfgstr=None):
    """
    Cached refinement of approximates centroids
    """
    print('[akmeans.precompute] refining:')
    if cache_dir == 'default':
        cache_dir = utool.get_app_resource_dir('vtool')
        utool.ensuredir(cache_dir)
    if akmeans_cfgstr is None:
        akmeans_cfgstr = nntool.get_flann_cfgstr(
            data, flann_params, cfgstr, use_data_hash)
    centroids = akmeans_iterations(data, centroids, max_iters, flann_params, 0, 10)
    utool.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    return centroids


#def test_hdbscan():
#    r"""
#    CommandLine:
#        python -m vtool.clustering2 --exec-test_hdbscan

#    Example:
#        >>> # SCRIPT
#        >>> from vtool.clustering2 import *  # NOQA
#        >>> from vtool.clustering2 import *  # NOQA
#        >>> import numpy as np
#        >>> rng = np.random.RandomState(42)
#        >>> data = rng.randn(1000000, 128)
#        >>> import hdbscan
#        >>> with ut.Timer() as t:
#        >>>     labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(data)

#    """
#    pass


def akmeans_iterations(data, centroids, max_iters, flann_params,
                       ave_unchanged_thresh=0, ave_unchanged_iterwin=10,
                       monitor=False):
    """
    Helper function which continues the iterations of akmeans

    Objective:
        argmin_{S} sum(sum(L2(x, u[i]) for x in S_i) for i in range(k))

    CommandLine:
        python -m vtool.clustering2 akmeans_iterations --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> data = rng.randn(100, 2)
        >>> nCentroids = 5
        >>> flann_params = {}
        >>> max_iters = 1000
        >>> ave_unchanged_thresh = 1
        >>> ave_unchanged_iterwin = 100
        >>> centroids = initialize_centroids(nCentroids, data)
        >>> centroids, hist = akmeans_iterations(data, centroids, max_iters,
        >>>                                      flann_params, ave_unchanged_thresh,
        >>>                                      ave_unchanged_iterwin, monitor=True)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.qt4ensure()
        >>> pt.multi_plot('epoch_num', hist, fnum=2)
        >>> plot_centroids(data, centroids)
        >>> ut.show_if_requested()
    """
    nData = data.shape[0]
    nCentroids = centroids.shape[0]
    # Initialize assignments
    datax2_centroidx_old = -np.ones(nData, dtype=np.int32)
    # Keep track of how many points have changed over an iteration window
    win2_unchanged = np.zeros(ave_unchanged_iterwin, dtype=centroids.dtype) + len(data)

    if monitor:
        history = ut.ddict(list)
        # loss =
        # loss = np.sqrt(dists).mean()
        # history['epoch_num'].append(count + 1)
        # history['loss'].append(loss)
        # history['ave_unchanged'].append(ave_unchanged)

    print((
        '[akmeans] akmeans: data.shape=%r ; nCentroids=%r\n'
        '[akmeans] * max_iters=%r\n'
        '[akmeans] * ave_unchanged_iterwin=%r ; ave_unchanged_thresh=%r\n'
    ) % (data.shape, nCentroids, max_iters,
         ave_unchanged_thresh, ave_unchanged_iterwin))
    sys.stdout.flush()
    for count in ut.ProgIter(range(0, max_iters), nTotal=max_iters, lbl='Akmeans: '):
        # 1) Assign each datapoint to the nearest centroid
        datax2_centroidx, dists = approximate_assignments(centroids, data, 1, flann_params)
        # 2) Compute new centroids (inplace) based on assignments
        centroids = compute_centroids(data, centroids, datax2_centroidx)
        # 3) Convergence Check: which datapoints changed membership?
        num_changed = (datax2_centroidx_old != datax2_centroidx).sum()
        win2_unchanged[count % ave_unchanged_iterwin] = num_changed
        ave_unchanged = win2_unchanged.mean()
        if monitor:
            loss = np.sqrt(dists).mean()
            history['epoch_num'].append(count + 1)
            history['loss'].append(loss)
            history['ave_unchanged'].append(ave_unchanged)
            # import plottool as pt
            # pt.multi_plot('epoch_num', history, fnum=1)
            # pt.update()
        if ave_unchanged < ave_unchanged_thresh:
            break
        else:
            datax2_centroidx_old = datax2_centroidx
    print('Finished akmeans')
    if monitor:
        return centroids, history
    else:
        return centroids


class AnnoyWraper(object):
    """
    flann-like interface to annnoy
    """

    def __init__(self):
        pass

    def build_annoy(self, centroids, trees=3):
        import annoy
        self.a = annoy.AnnoyIndex(centroids.shape[1], metric='euclidean')
        for i, v in enumerate(centroids):
            self.a.add_item(i, v)
        self.a.build(trees)

    def query_annoy(self, query_vecs, num, checks=-1):
        a = self.a
        index_list = []
        dist_list = []
        for v in query_vecs:
            idx, dist = a.get_nns_by_vector(v, num, search_k=checks, include_distances=True)
            index_list.append(idx)
            dist_list.append(dist)
        return np.array(index_list), np.array(dist_list) ** 2

    def nn(self, data_vecs, query_vecs, num, trees=3, checks=-1):
        self.build_annoy(data_vecs, trees)
        return self.query_annoy(query_vecs, num, checks)


ANNOY = 0


def approximate_distances(centroids, data, K, flann_params):
    if ANNOY:
        (_, qdist2_sdist) = AnnoyWraper().nn(centroids, data, K, **flann_params)
    else:
        import pyflann
        (_, qdist2_sdist) = pyflann.FLANN().nn(centroids, data, K, **flann_params)
    return qdist2_sdist


def approximate_assignments(seachedvecs, queryvecs, K, flann_params):
    if ANNOY:
        (qx2_sx, qdist2_sdist) = AnnoyWraper().nn(seachedvecs, queryvecs, K, **flann_params)
        qx2_sx = qx2_sx.ravel()
    else:
        import pyflann
        (qx2_sx, qdist2_sdist) = pyflann.FLANN().nn(seachedvecs, queryvecs, K, **flann_params)
    return qx2_sx, qdist2_sdist


def compute_centroids(data, centroids, datax2_centroidx):
    """
    Computes centroids given datax assignments
    TODO: maybe use the grouping code instad of the LR algorithm

    >>> from vtool.clustering2 import *  # NOQA
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> data = rng.randn(100, 2)
    >>> nCentroids = 5
    >>> flann_params = {}
    >>> centroids = initialize_centroids(nCentroids, data)
    >>> centroids_ = centroids.copy()
    >>> (datax2_centroidx, _) = approximate_assignments(centroids, data, 1, flann_params)
    >>> out = compute_centroids(data, centroids, datax2_centroidx)
    """
    # if True:
    unique_groups, groupxs = group_indices(datax2_centroidx)
    for centroidx, xs in zip(unique_groups, groupxs):
        # Inplace modification of centroid
        centroids[centroidx] = data.take(xs, axis=0).mean()
    # else:
    #     nData = data.shape[0]
    #     nCentroids = centroids.shape[0]
    #     # sort data by centroid
    #     datax_sortx = datax2_centroidx.argsort()
    #     datax_sort  = datax2_centroidx[datax_sortx]
    #     # group datapoints by centroid using a sliding grouping algorithm
    #     centroidx2_dataLRx = [None] * nCentroids
    #     _L = 0
    #     for _R in range(nData + 1):  # Slide R
    #         if _R == nData or datax_sort[_L] != datax_sort[_R]:
    #             centroidx2_dataLRx[datax_sort[_L]] = (_L, _R)
    #             _L = _R
    #     # Compute the centers of each group (centroid) of datapoints
    #     for centroidx, dataLRx in enumerate(centroidx2_dataLRx):
    #         if dataLRx is None:
    #             continue  # ON EMPTY CLUSTER
    #         (_L, _R) = dataLRx
    #         # The centroid center is the mean of its datapoints
    #         centroids[centroidx] = np.mean(data[datax_sortx[_L:_R]], axis=0)
    #         #centroids[centroidx] = np.array(np.round(centroids[centroidx]), dtype=np.uint8)
    return centroids


def jagged_group(groupids_list):
    """ flattens and returns group indexes into the flattened list """
    #flatx2_itemx = np.array(utool.flatten(itemxs_iter))
    flatids = np.array(utool.flatten(groupids_list))
    keys, groupxs = group_indices(flatids)
    return keys, groupxs


def apply_jagged_grouping(unflat_items, groupxs):
    """ takes unflat_list and flat group indices. Returns the unflat grouping """
    flat_items = np.array(utool.flatten(unflat_items))
    item_groups = apply_grouping(flat_items, groupxs)
    return item_groups
    #itemxs_iter = [[count] * len(idx2_groupid) for count, idx2_groupid in enumerate(groupids_list)]


def groupedzip(id_list, datas_list):
    r"""
    Function for grouping multiple lists of data (stored in ``datas_list``)
    using ``id_list``.

    Args:
        id_list (list):
        datas_list (list):

    Returns:
        iterator: _iter

    CommandLine:
        python -m vtool.clustering2 --test-groupedzip

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> # build test data
        >>> id_list = np.array([1, 2, 1, 2, 1, 2, 3])
        >>> datas_list = [
        ...     ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        ...     ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        ... ]
        >>> # execute function
        >>> groupxs, grouped_iter = groupedzip(id_list, datas_list)
        >>> grouped_tuples = list(grouped_iter)
        >>> # verify results
        >>> result = str(groupxs) + '\n'
        >>> result += ut.list_str(grouped_tuples, nl=1)
        >>> print(result)
        [1 2 3]
        [
            (['a', 'c', 'e'], ['A', 'C', 'E']),
            (['b', 'd', 'f'], ['B', 'D', 'F']),
            (['g'], ['G']),
        ]
    """
    unique_ids, groupxs = group_indices(id_list)
    grouped_datas_list = [apply_grouping_(data,  groupxs) for data in datas_list]
    grouped_iter = zip(*grouped_datas_list)
    return unique_ids, grouped_iter


@profile
def group_indices(idx2_groupid):
    r"""
    group_indices

    Args:
        idx2_groupid (ndarray): numpy array of group ids (must be numeric)

    Returns:
        tuple (ndarray, list of ndarrays): (keys, groupxs)

    CommandLine:
        python -m vtool.clustering2 --test-group_indices
        python -m vtool.clustering2 --exec-group_indices:1
        utprof.py -m vtool.clustering2 --test-group_indices:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> result = ut.repr3((keys, groupxs), nobraces=1)
        >>> print(result)
        np.array([1, 2, 3], dtype=np.int64),
        [
            np.array([1, 3, 5], dtype=np.int64),
            np.array([0, 2, 4, 6], dtype=np.int64),
            np.array([ 7,  8,  9, 10], dtype=np.int64),
        ],

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> idx2_groupid = np.array([[  24], [ 129], [ 659], [ 659], [ 24],
        ...       [659], [ 659], [ 822], [ 659], [ 659], [24]])
        >>> # 2d arrays must be flattened before coming into this function so
        >>> # information is on the last axis
        >>> (keys, groupxs) = group_indices(idx2_groupid.T[0])
        >>> result = ut.repr3((keys, groupxs), nobraces=1)
        >>> print(result)
        np.array([ 24, 129, 659, 822], dtype=np.int64),
        [
            np.array([ 0,  4, 10], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2, 3, 5, 6, 8, 9], dtype=np.int64),
            np.array([7], dtype=np.int64),
        ],


    Example2:
        >>> # TIMING_TEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> idx2_groupid = rng.randint(100, 200, 1000)
        >>> group_indices(idx2_groupid)
        >>> [group_indices(rng.randint(100, 200, 1000)) for _ in range(1000)]

    Time:
        >>> import vtool as vt
        >>> import utool as ut
        >>> setup = ut.extract_timeit_setup(vt.group_indices, 2, 'groupxs =')
        >>> stmt_list = ut.codeblock(
                '''
                [sortx[lx:rx] for lx, rx in ut.itertwo(idxs)]
                #[sortx[lx:rx] for lx, rx in zip(idxs[0:-1], idxs[1:])]
                #[sortx[lx:rx] for lx, rx in ut.iter_window(idxs)]
                #[sortx[slice(*_)] for _ in ut.itertwo(idxs)]
                #[sortx[slice(lr, lx)] for lr, lx in ut.itertwo(idxs)]
                #np.split(sortx, idxs[1:-1])
                #np.hsplit(sortx, idxs[1:-1])
                np.array_split(sortx, idxs[1:-1])
                ''').split('\n')
        >>> stmt_list = [x for x in stmt_list if not x.startswith('#')]
        >>> passed, times, outputs = ut.timeit_compare(stmt_list, setup, iterations=10000)

        >>> stmt_list = ut.codeblock(
                '''
                np.diff(groupids_sorted)
                np.ediff1d(groupids_sorted)
                np.subtract(groupids_sorted[1:], groupids_sorted[:-1])
                ''').split('\n')
        >>> stmt_list = [x for x in stmt_list if not x.startswith('#')]
        >>> passed, times, outputs = ut.timeit_compare(stmt_list, setup, iterations=10000)

    Timeit:
        import numba
        group_indices_numba = numba.jit(group_indices)
        group_indices_numba(idx2_groupid)

    SeeAlso:
        apply_grouping

    References:
        http://stackoverflow.com/questions/4651683/
        numpy-grouping-using-itertools-groupby-performance

    TODO:
        Look into np.split
        http://stackoverflow.com/questions/21888406/
        getting-the-indexes-to-the-duplicate-columns-of-a-numpy-array
    """
    # Sort items and idx2_groupid by groupid
    # <len(data) bottlneck>
    sortx = idx2_groupid.argsort()
    groupids_sorted = idx2_groupid.take(sortx)
    num_items = idx2_groupid.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, idx2_groupid.dtype)
    np.subtract(groupids_sorted[1:], groupids_sorted[:-1], out=diff[1:num_items])
    #diff[1:num_items] = np.subtract(groupids_sorted[1:], groupids_sorted[:-1])
    idxs = np.flatnonzero(diff)
    # Groups are between bounding indexes
    # <len(keys) bottlneck>
    groupxs = [sortx[lx:rx] for lx, rx in ut.itertwo(idxs)]  # 34.5%
    # Unique group keys
    keys = groupids_sorted[idxs[:-1]]
    return keys, groupxs


def find_duplicate_items(item_arr):
    """
    Args:
        item_arr (?):

    Returns:
        ?: duplicate_items

    CommandLine:
        python -m vtool.clustering2 --test-find_duplicate_items

    References:
        http://stackoverflow.com/questions/21888406/getting-the-indexes-to-the-duplicate-columns-of-a-numpy-array

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> # build test data
        >>> np.random.seed(0)
        >>> item_arr = np.random.randint(100, size=30)
        >>> # execute function
        >>> duplicate_items = find_duplicate_items(item_arr)
        >>> # verify results
        >>> assert duplicate_items == list(six.iterkeys(ut.find_duplicate_items(item_arr)))
        >>> result = str(duplicate_items)
        >>> print(result)
        [9, 67, 87, 88]
    """
    sortx = item_arr.argsort()
    groupids_sorted = item_arr.take(sortx)

    #duplicate_idxs = np.flatnonzero(~np.diff(groupids_sorted).astype(np.bool))
    diff = np.diff(groupids_sorted)
    #notdiff = np.bitwise_not(diff.astype(np.bool))
    edges = np.flatnonzero(diff.astype(np.bool)) + 1
    duplicate_items = [group[0] for group in np.split(groupids_sorted, edges)
                       if group.shape[0] > 1]
    #duplicate_items = groupids_sorted.take(duplicate_idxs)
    return duplicate_items


def apply_grouping(items, groupxs, axis=0):
    """
    applies grouping from group_indicies
    apply_grouping

    Args:
        items (ndarray):
        groupxs (list of ndarrays):

    Returns:
        list of ndarrays: grouped items

    SeeAlso:
        group_indices
        invert_apply_grouping

    CommandLine:
        python -m vtool.clustering2 --test-apply_grouping

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> #np.random.seed(42)
        >>> #size = 10
        >>> #idx2_groupid = np.array(np.random.randint(0, 4, size=size))
        >>> #items = np.random.randint(5, 10, size=size)
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> items        = np.array([1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> grouped_items = apply_grouping(items, groupxs)
        >>> result = str(grouped_items)
        >>> print(result)
        [array([8, 5, 6]), array([1, 5, 8, 7]), array([5, 3, 0, 9])]
    """
    # SHOULD DO A CONTIGUOUS CHECK HERE
    #items_ = np.ascontiguousarray(items)
    return [items.take(xs, axis=axis) for xs in groupxs]
    #return [items[idxs] for idxs in groupxs]


def apply_grouping_(items, groupxs):
    """ non-optimized version """
    return ut.apply_grouping(items, groupxs)


def invert_apply_grouping(grouped_items, groupxs):
    r"""
    Args:
        grouped_items (list): of lists
        groupxs (list): of lists

    Returns:
        list: items

    CommandLine:
        python -m vtool.clustering2 --test-invert_apply_grouping

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> # build test data
        >>> grouped_items = [[8, 5, 6], [1, 5, 8, 7], [5, 3, 0, 9]]
        >>> groupxs = [np.array([1, 3, 5]), np.array([0, 2, 4, 6]), np.array([ 7,  8,  9, 10])]
        >>> # execute function
        >>> items = invert_apply_grouping(grouped_items, groupxs)
        >>> result = items
        >>> # verify results
        >>> print(result)
        [1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> grouped_items, groupxs = [], []
        >>> result = invert_apply_grouping(grouped_items, groupxs)
        >>> print(result)
        []
    """
    if len(grouped_items) == 0:
        assert len(groupxs) == 0, 'inconsistant. len(grouped_items)=%d, len(groupxs)=%d' % (len(grouped_items), len(groupxs))
        return []
    # maxval = max(map(max, groupxs))
    maxval = _max(list(map(_max, groupxs)))
    ungrouped_items = [None] * (maxval + 1)  # np.full((maxval + 1,), None)
    for itemgroup, xs in zip(grouped_items, groupxs):
        for item, x in zip(itemgroup, xs):
            ungrouped_items[x] = item
    return ungrouped_items


def invert_apply_grouping3(grouped_items, groupxs, maxval):
    ungrouped_items = [None] * (maxval + 1)  # np.full((maxval + 1,), None)
    for itemgroup, xs in zip(grouped_items, groupxs):
        for item, x in zip(itemgroup, xs):
            ungrouped_items[x] = item
    return ungrouped_items


def _max(x):
    return np.max(x) if len(x) > 0 else 0


def invert_apply_grouping2(grouped_items, groupxs, dtype=None):
    """ use only when ungrouping will be complete """
    maxval = _max(list(map(_max, groupxs)))
    ungrouped_items = np.zeros((maxval + 1,), dtype=dtype)
    for itemgroup, ix_list in zip(grouped_items, groupxs):
        ungrouped_items[ix_list] = itemgroup
    return ungrouped_items


def apply_grouping_iter(items, groupxs):
    return (items.take(xs, axis=0) for xs in groupxs)


def apply_grouping_iter2(items, groupxs):
    return (np.array(list(items)).take(xs, axis=0) for xs in groupxs)


def groupby(items, idx2_groupid):
    """
    >>> items    = np.array(np.arange(100))
    >>> idx2_groupid = np.array(np.random.randint(0, 4, size=100))
    >>> items = idx2_groupid
    """
    keys, groupxs = group_indices(idx2_groupid)
    vals = [items[idxs] for idxs in groupxs]
    return keys, vals


def groupby_gen(items, idx2_groupid):
    """
    >>> items    = np.array(np.arange(100))
    >>> idx2_groupid = np.array(np.random.randint(0, 4, size=100))
    """
    for key, val in zip(*groupby(items, idx2_groupid)):
        yield (key, val)


def groupby_dict(items, idx2_groupid):
    # Build a dict
    grouped = {key: val for key, val in groupby_gen(items, idx2_groupid)}
    return grouped


def double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy=False):
    """
    Takes corresponding lists as input and builds a double mapping.

    DEPRICATE

    Args:
        inner_key_list (list): each value_i is a scalar key.
        outer_keys_list (list): each value_i list of scalar keys
        items_list (list): value_i is a list that corresponds to outer_keys_i

    Returns:
        utool.ddict of dicts: outerkey2_innerkey2_items

    Examples:
        >>> from vtool.clustering2 import *  # NOQA
        >>> inner_key_list = [100, 200, 300, 400]
        >>> outer_keys_list = [[10, 20, 20], [30], [30, 10], [20]]
        >>> items_list = [[1, 2, 3], [4], [5, 6], [7]]
        >>> ensure_numpy = True
        >>> outerkey2_innerkey2_items = double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
        >>> print(utool.dict_str(outerkey2_innerkey2_items))
        {
            10: {300: array([6]), 100: array([1])},
            20: {400: array([7]), 100: array([2, 3])},
            30: {200: array([4]), 300: array([5])},
        }

        >>> from vtool.clustering2 import *  # NOQA
        >>> len_ = 3000
        >>> incrementer = utool.make_incrementer()
        >>> nOuterList = [np.random.randint(300) for _ in range(len_)]
        >>> # Define big double_group input
        >>> inner_key_list = np.random.randint(100, size=len_) * 1000 + 1000
        >>> outer_keys_list = [np.random.randint(100, size=nOuter_) for nOuter_ in nOuterList]
        >>> items_list = [np.array([incrementer() for _ in range(nOuter_)]) for nOuter_ in nOuterList]
        >>> ensure_numpy = False
        >>> outerkey2_innerkey2_items = double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
        >>> print(utool.dict_str(outerkey2_innerkey2_items))
        >>> print(utool.dict_str(outerkey2_innerkey2_items[0]))

    Timeit:
        %timeit double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
    """
    if ensure_numpy:
        inner_key_list = np.array(inner_key_list)
        outer_keys_list = np.array(map(np.array, outer_keys_list))
        items_list = np.array(map(np.array, items_list))
    outerkey2_innerkey2_items = utool.ddict(dict)
    _iter =  zip(inner_key_list, outer_keys_list, items_list)
    for inner_key, outer_keys, items in _iter:
        group_outerkeys, groupxs = group_indices(outer_keys)
        subitem_iter = (items.take(xs, axis=0) for xs in groupxs)
        for outer_key, subitems in zip(group_outerkeys, subitem_iter):
            outerkey2_innerkey2_items[outer_key][inner_key] = subitems
    return outerkey2_innerkey2_items
    #daid2_wx2_drvecs = utool.ddict(lambda: utool.ddict(list))
    #for wx, aids, rvecs in zip(wx_sublist, aids_list, rvecs_list1):
    #    group_aids, groupxs = clustertool.group_indices(aids)
    #    rvecs_group = clustertool.apply_grouping(rvecs, groupxs)
    #    for aid, subrvecs in zip(group_aids, rvecs_group):
    #        daid2_wx2_drvecs[aid][wx] = subrvecs


def sparse_normalize_rows(csr_mat):
    pass
    #return sklearn.preprocessing.normalize(csr_mat, norm='l2', axis=1, copy=False)


def sparse_multiply_rows(csr_mat, vec):
    """ Row-wise multiplication of a sparse matrix by a sparse vector """
    csr_vec = spsparse.csr_matrix(vec, copy=False)
    #csr_vec.shape = (1, csr_vec.size)
    sparse_stack = [row.multiply(csr_vec) for row in csr_mat]
    return spsparse.vstack(sparse_stack, format='csr')


# ---------------
# Plotting Code
# ---------------

def plot_centroids(data, centroids, num_pca_dims=3, whiten=False,
                   labels='centroids', fnum=1, prefix=''):
    """ Plots centroids and datapoints. Plots accurately up to 3 dimensions.
    If there are more than 3 dimensions, PCA is used to recude the dimenionality
    to the <num_pca_dims> principal components
    """
    # http://www.janeriksolem.net/2012/03/isomap-with-scikit-learn.html
    if __debug__ and False:
        utool.printex(Exception('INFO'), keys=[
            (type, 'data'),
            'data',
            'data.shape',
        ])
    from plottool import draw_func2 as df2
    data_dims = data.shape[1]
    show_dims = min(num_pca_dims, data_dims)
    if data_dims != show_dims:
        # we can't physiologically see the data, so look at a projection
        print('[akmeans] Doing PCA')
        from sklearn import decomposition
        pcakw = dict(copy=True, n_components=show_dims, whiten=whiten)
        pca = decomposition.PCA(**pcakw).fit(data)
        pca_data = pca.transform(data)
        pca_centroids = pca.transform(centroids)
        print('[akmeans] ...Finished PCA')
    else:
        # pca is not necessary
        print('[akmeans] No need for PCA')
        pca_data = data
        pca_centroids = centroids
    print(pca_data.shape)
    # Make a color for each centroid
    data_x = pca_data[:, 0]
    data_y = pca_data[:, 1]
    clus_x = pca_centroids[:, 0]
    clus_y = pca_centroids[:, 1]
    nCentroids = K = len(centroids)
    if labels == 'centroids':
        datax2_label, dists = approximate_assignments(centroids, data, 1, {})
    else:
        datax2_label = labels
    datax2_label = np.array(datax2_label, dtype=np.int32)
    print(datax2_label)
    assert len(datax2_label.shape) == 1, repr(datax2_label.shape)
    #if datax2_centroids is None:
    #    (datax2_centroidx, _) = p yflann.FLANN().nn(centroids, data, 1)
    #data_colors = colors[np.array(datax2_centroidx, dtype=np.int32)]
    nColors = datax2_label.max() - datax2_label.min() + 1
    print('nColors=%r' % (nColors,))
    print('K=%r' % (K,))
    colors = np.array(df2.distinct_colors(nColors, brightness=.95))
    clus_colors = np.array(df2.distinct_colors(nCentroids, brightness=.95))
    assert labels != 'centroids' or nColors == K
    if __debug__ and False:
        utool.printex(Exception('INFO'), keys=[
            'colors',
            (utool.get_stats, 'colors'),
            'colors.shape',
            'datax2_label',
            (utool.get_stats, 'datax2_label'),
            'datax2_label.shape',
        ])
    assert len(datax2_label.shape) == 1, repr(datax2_label.shape)
    data_colors = colors[datax2_label]
    # Create a figure
    fig = df2.figure(fnum, doclf=True, docla=True)
    if show_dims == 2:
        ax = df2.plt.gca()
        df2.plt.scatter(data_x, data_y, s=20,  c=data_colors, marker='o', alpha=1)
        #df2.plt.scatter(data_x, data_y, s=20,  c=data_colors, marker='o', alpha=.2)
        df2.plt.scatter(clus_x, clus_y, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        df2.dark_background(ax)
    if show_dims == 3:
        from mpl_toolkits.mplot3d import Axes3D  # NOQA
        ax = fig.add_subplot(111, projection='3d')
        data_z = pca_data[:, 2]
        clus_z = pca_centroids[:, 2]
        #ax.scatter(data_x, data_y, data_z, s=20,  c=data_colors, marker='o', alpha=.2)
        ax.scatter(data_x, data_y, data_z, s=20,  c=data_colors, marker='o', alpha=1)
        ax.scatter(clus_x, clus_y, clus_z, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        df2.dark_background(ax)
        #ax.set_alpha(.1)
        #utool.embed()
        #ax.set_frame_on(False)
    ax = df2.plt.gca()
    waswhitestr = ' +whitening' * whiten
    titlestr = ('{prefix}AKmeans: K={K}.'
                'PCA projection {data_dims}D -> {show_dims}D'
                '{waswhitestr}').format(**locals())
    ax.set_title(titlestr)
    return fig


def uniform_sample_hypersphere(num, ndim=2, only_quadrent_1=False):
    r"""
    Not quite done yet

    References:
        https://en.wikipedia.org/wiki/Regular_polytope
        https://en.wikipedia.org/wiki/Platonic_solid#Higher_dimensions
        https://en.wikipedia.org/wiki/Cross-polytope

    Args:
        num (?):
        ndim (int): (default = 2)

    CommandLine:
        python -m vtool.clustering2 --test-uniform_sampe_hypersphere

    Ignore:
        #pip install polytope
        sudo pip install cvxopt  --no-deps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> num = 100
        >>> ndim = 3
        >>> pts = uniform_sampe_hypersphere(num, ndim)
        >>> print(pts)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> if ndim == 2:
        >>>     pt.plot(pts.T[0], pts.T[1], 'gx')
        >>> elif ndim == 3:
        >>>     #pt.plot_surface3d(pts.T[0], pts.T[1], pts.T[2])
        >>>     from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>>     fig = pt.figure(1, doclf=True, docla=True)
        >>>     ax = fig.add_subplot(111, projection='3d')
        >>>     ax.scatter(pts.T[0], pts.T[1], pts.T[2], s=20, marker='o', alpha=1)
        >>>     ax.autoscale(enable=False)
        >>>     ax.set_aspect('equal')
        >>>     df2.dark_background(ax)
        >>> pt.dark_background()
        >>> ut.show_if_requested()
    """
    import vtool as vt
    pts = np.random.rand(num, ndim)
    if not only_quadrent_1:
        pts =  pts * 2 - 1
        pass
    pts = vt.normalize_rows(pts)
    return pts


def unsupervised_multicut_labeling(cost_matrix, thresh=0):
    """
    Notes:
        requires CPLEX

    CommandLine:
        python -m vtool.clustering2 unsupervised_multicut_labeling --show

    Ignore:

        >>> # synthetic data
        >>> import vtool as vt
        >>> size = 100
        >>> thresh = 50
        >>> np.random.randint(0, 1)
        >>> np.zeros((size, size))
        >>> #np.random.rand(size, size)
        >>> size = 45
        >>> #size = 10
        >>> size = 5
        >>> aids = np.arange(size)
        >>> rng = np.random.RandomState(443284320)
        >>> encounter_lbls = rng.randint(0, size, size)
        >>> separation = 5.0
        >>> separation = 1.10
        >>> grid1 = np.tile(encounter_lbls, (size, 1))
        >>> is_match = grid1.T == grid1
        >>> good_pos = np.where(is_match)
        >>> bad_pos = np.where(~s_match)
        >>> cost_matrix_ = np.zeros((size, size))
        >>> cost_matrix_[good_pos] = rng.randn(len(good_pos[0])) + separation
        >>> cost_matrix_[bad_pos] = rng.randn(len(bad_pos[0])) - separation
        >>> false_val = min(cost_matrix_.min(), np.min(rng.randn(1000) - separation))
        >>> true_val = max(cost_matrix_.max(), np.max(rng.randn(500) + separation))
        >>> cost_matrix_[np.diag_indices_from(cost_matrix_)] = true_val
        >>> #cost_matrix_[np.diag_indices_from(cost_matrix_)] = np.inf
        >>> cost_matrix = (cost_matrix_ - false_val) / (true_val - false_val)
        >>> cost_matrix = 2 * (cost_matrix - .5)
        >>> thresh = 0
        >>> labels = vt.unsupervised_multicut_labeling(cost_matrix, thresh)
        >>> diff = ut.compare_groupings(
        >>>     list(ut.group_items(aids, encounter_lbls).values()),
        >>>     list(ut.group_items(aids, labels).values()))
        >>> print('diff = %r' % (diff,))

        #gm, = ut.exec_func_src(vt.unsupervised_multicut_labeling,
        #key_list=['gm'], sentinal='inf = opengm')
        #parameter = opengm.InfParam()
        #%timeit opengm.inference.Multicut(gm, parameter=parameter).infer()

    Example:
        >>> # SCRIPT
        >>> from vtool.clustering2 import *  # NOQA
        >>> import networkx as nx
        >>> import plottool as pt
        >>> rng = np.random.RandomState(443284320)
        >>> pt.ensure_pylab_qt4()
        >>> #
        >>> def make_test_costmatrix(name_labels, view_labels, separation=2):
        >>>     is_same = name_labels == name_labels[:, None]
        >>>     is_comp = np.abs(view_labels - view_labels[:, None]) <= 1
        >>>     good_pos = np.where(is_same)
        >>>     bad_pos = np.where(~is_same)
        >>>     cost_matrix_ = np.zeros((len(name_labels), len(name_labels)))
        >>>     cost_matrix_[good_pos] = rng.randn(len(good_pos[0])) + separation
        >>>     cost_matrix_[bad_pos] = rng.randn(len(bad_pos[0])) - separation
        >>>     cost_matrix_ = (cost_matrix_.T + cost_matrix_) / 2
        >>>     false_val = min(cost_matrix_.min(), np.min(rng.randn(1000) - separation))
        >>>     true_val = max(cost_matrix_.max(), np.max(rng.randn(500) + separation))
        >>>     cost_matrix_[np.diag_indices_from(cost_matrix_)] = true_val
        >>>     cost_matrix = (cost_matrix_ - false_val) / (true_val - false_val)
        >>>     cost_matrix = 2 * (cost_matrix - .5)
        >>>     cost_matrix[np.where(~is_comp)] = 0
        >>>     return cost_matrix
        >>> #
        >>> view_labels = np.array([0, 0, 2, 2, 1, 0, 0, 0])
        >>> name_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        >>> #cost_matrix = make_test_costmatrix(name_labels, view_labels, 2)
        >>> cost_matrix = make_test_costmatrix(name_labels, view_labels, .9)
        >>> #
        >>> def multicut_value(cost_matrix, name_labels):
        >>>     grid1 = np.tile(name_labels, (len(name_labels), 1))
        >>>     isdiff = grid1.T != grid1
        >>>     cut_value = cost_matrix[isdiff].sum()
        >>>     return cut_value
        >>> #
        >>> aids = np.arange(len(name_labels))
        >>> #
        >>> graph = ut.nx_from_matrix(cost_matrix)
        >>> weights = nx.get_edge_attributes(graph, 'weight')
        >>> #
        >>> floatfmt1 = ut.partial(ut.map_dict_vals, lambda x: 'w=%.2f' % x)
        >>> floatfmt2 = ut.partial(ut.map_dict_vals, lambda x: 'l=%.2f' % x)
        >>> #
        >>> lens = ut.map_dict_vals(lambda x: (1 - ((x + 1) / 2)) / 2, weights)
        >>> labels = floatfmt1(weights)
        >>> #labels = floatfmt2(lens)
        >>> nx.set_edge_attributes(graph, 'label', labels)
        >>> #nx.set_edge_attributes(graph, 'len', lens)
        >>> nx.set_node_attributes(graph, 'shape', 'ellipse')
        >>> encounter_lbls_str = [str(x) for x in name_labels]
        >>> node_name_lbls = dict(zip(aids, encounter_lbls_str))
        >>> import vtool as vt
        >>> #
        >>> mcut_labels = vt.unsupervised_multicut_labeling(cost_matrix, thresh=vt.eps)
        >>> diff = ut.compare_groupings(
        >>>     list(ut.group_items(aids, name_labels).values()),
        >>>     list(ut.group_items(aids, mcut_labels).values()))
        >>> print('diff = %r' % (diff,))
        >>> #
        >>> nx.set_node_attributes(graph, 'label', node_name_lbls)
        >>> node_mcut_lbls = dict(zip(aids, mcut_labels))
        >>> nx.set_node_attributes(graph, 'mcut_label', node_mcut_lbls)
        >>> #
        >>> print('mc_val(name) ' + str(multicut_value(cost_matrix, name_labels)))
        >>> print('mc_val(mcut) ' + str(multicut_value(cost_matrix, mcut_labels)))
        >>> #
        >>> ut.color_nodes(graph, 'mcut_label')
        >>> #
        >>> # remove noncomparable edges
        >>> is_comp = np.abs(view_labels - view_labels[:, None]) <= 1
        >>> #
        >>> noncomp_edges = list(zip(*np.where(~is_comp)))
        >>> graph.remove_edges_from(noncomp_edges)
        >>> #
        >>> layoutkw = {
        >>>     'sep' : 5,
        >>>     'prog': 'neato',
        >>>     'overlap': 'false',
        >>>     'splines': 'spline',
        >>> }
        >>> pt.show_nx(graph, layoutkw=layoutkw)
        >>> ut.show_if_requested()

    """
    import opengm
    import numpy as np
    #import plottool as pt
    from itertools import product
    cost_matrix_ = cost_matrix - thresh
    num_vars = len(cost_matrix_)

    # Enumerate undirected edges (node index pairs)
    var_indices = np.arange(num_vars)
    varindex_pairs = np.array(
        [(a1, a2) for a1, a2 in product(var_indices, var_indices)
         if a1 != a2 and a1 > a2], dtype=np.uint32)
    varindex_pairs.sort(axis=1)

    # Create nodes in the graphical model.  In this case there are <num_vars>
    # nodes and each node can be assigned to one of <num_vars> possible labels
    num_nodes = num_vars
    space = np.full((num_nodes,), fill_value=num_vars, dtype=np.int)
    gm = opengm.gm(space)

    # Use one potts function for each edge
    for varx1, varx2 in varindex_pairs:
        cost = cost_matrix_[varx1, varx2]
        potts_func = opengm.PottsFunction((num_vars, num_vars), valueEqual=0, valueNotEqual=cost)
        potts_func_id = gm.addFunction(potts_func)
        var_indicies = np.array([varx1, varx2])
        gm.addFactor(potts_func_id, var_indicies)

    #pt.ensure_pylab_qt4()
    #opengm.visualizeGm(gm=gm)

    # Not sure what parameters are allowed to be passed here.
    parameter = opengm.InfParam()
    inf = opengm.inference.Multicut(gm, parameter=parameter)
    inf.infer()
    labels = inf.arg()
    #print(labels)
    return labels

#def alpha_expansion_cut(graph):
    # https://github.com/amueller/gco_python/blob/master/example.py
    #    import pygco
    #prob_annots2 = prob_annots.copy()
    #finite_probs = (prob_annots2[np.isfinite(prob_annots2)])
    #mean = finite_probs.mean()
    ## make symmetric
    #prob_annots2[~np.isfinite(prob_annots2)] = finite_probs.max() * 2
    #prob_annots2 = (prob_annots2.T + prob_annots2) / 2
    #int_factor = 100 / mean
    #pairwise_cost = (prob_annots2 * int_factor).astype(np.int32)
    #n_labels = 2
    #unary_cost = np.ones((prob_annots.shape[0], n_labels)).astype(np.int32)
    #u, v = np.meshgrid(np.arange(prob_annots.shape[0]).astype(np.int32), np.arange(len(prob_annots)).astype(np.int32))
    #edges = np.vstack((u.flatten(), v.flatten(), pairwise_cost.flatten())).T.astype(np.int32)

    #import pygco
    #n_iter = 5
    #algorithm = 'expansion'
    #unary_cost = np.ascontiguousarray(unary_cost)
    ##pairwise_cost = np.ascontiguousarray(pairwise_cost)
    #pairwise_cost = np.eye(n_labels).astype(np.int32)
    #edges = np.ascontiguousarray(edges)
    #pygco.cut_from_graph(edges, unary_cost, pairwise_cost, n_iter, algorithm)
    #pairwise_cost = prob_annots
    pass


def example_binary():
    import matplotlib.pyplot as plt
    import numpy as np
    from pygco import cut_simple, cut_from_graph
    # generate trivial data
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

    # create unaries
    unaries = x_noisy
    # as we convert to int, we need to multipy to get sensible values
    unaries = (10 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    unaries[:] = 0
    # create potts pairwise
    pairwise = -10 * np.eye(2, dtype=np.int32)

    # do simple cut
    result = cut_simple(unaries, pairwise)

    # use the gerneral graph algorithm
    # first, we construct the grid graph
    inds = np.arange(x.size).reshape(x.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert]).astype(np.int32)

    # we flatten the unaries
    result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)

    # plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(233, title="rounded to integers")
    plt.imshow(unaries[:, :, 0], interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="cut_simple")
    plt.imshow(result, interpolation='nearest')
    plt.subplot(236, title="cut_from_graph")
    plt.imshow(result_graph.reshape(x.shape), interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.clustering2
        python -m vtool.clustering2 --allexamples
        python -m vtool.clustering2 --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
