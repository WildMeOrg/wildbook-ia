# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import range
import six
import utool
import utool as ut
import sys
import numpy as np
import scipy.sparse as spsparse
import pyflann
import vtool.nearest_neighbors as nntool

(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[clustering2]', DEBUG=False)


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

    Example2:
        >>> from vtool.clustering2 import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> data = np.vstack(ibs.get_annot_vecs(ibs.get_valid_aids()))
        >>> flann_params = None
        >>> samples_per_iter = 1000
        >>> K = 8000  # 64000
        >>> initial_centers = akmeans_plusplus_init(data, K, samples_per_iter,  flann_params)

    CommandLine:
        profiler.sh ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
        python ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
    """
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
    center_indicies = [unusedx2_datax[chosen_unusedx]]
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
    centers = data.take(center_indicies, axis=0)
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
        # Remove the chosen index from unused indicies
        unusedx2_datax = np.delete(unusedx2_datax, chosen_unusedx)
        center_indicies.append(chosen_datax)
        chosen_data = data.take(chosen_datax, axis=0)
        # Append new center to data and flann index
        flann.add_points(chosen_data)
    center_indicies = np.array(center_indicies)
    centers = data.take(center_indicies, axis=0)
    print('len(center_indicies) = %r' % len(center_indicies))
    print('len(set(center_indicies)) = %r' % len(set(center_indicies)))
    return centers


def akmeans(data, nCentroids, max_iters=5, initmethod='akmeans++',
            flann_params={}, ave_unchanged_thresh=0, ave_unchanged_iterwin=10):
    """
    Approximiate K-Means (using FLANN)
    Input: data - np.array with rows of data.
    Description: Quickly partitions data into K=nCentroids centroids.  Cluster
    centers are randomly assigned to datapoints.  Each datapoint is assigned to
    its approximate nearest centroid center.  The centroid centers are recomputed.
    Repeat until approximate convergence."""
    # Setup iterations
    centroids = initialize_centroids(nCentroids, data, initmethod)
    centroids = akmeans_iterations(data, centroids, max_iters, flann_params,
                                    ave_unchanged_thresh, ave_unchanged_iterwin)
    return centroids


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
    """ Refines the approximates centroids """
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


def akmeans_iterations(data, centroids, max_iters,
                        flann_params, ave_unchanged_thresh, ave_unchanged_iterwin):
    """ Helper function which continues the iterations of akmeans

    >>> from vtool.clustering2 import *  # NOQA
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(100, 2)
    >>> nCentroids = 5
    >>> flann_params = {}
    >>> max_iters = 100
    >>> ave_unchanged_thresh = 100
    >>> ave_unchanged_iterwin = 100
    >>> centroids = initialize_centroids(nCentroids, data)
    >>> centroids = akmeans_iterations(data, centroids, max_iters, flann_params, ave_unchanged_thresh, ave_unchanged_iterwin)
    >>> plot_centroids(data, centroids)
    """
    nData = data.shape[0]
    nCentroids = centroids.shape[0]
    # Initialize assignments
    datax2_centroidx_old = -np.ones(nData, dtype=np.int32)
    # Keep track of how many points have changed over an iteration window
    win2_unchanged = np.zeros(ave_unchanged_iterwin, dtype=centroids.dtype) + len(data)
    print((
        '[akmeans] akmeans: data.shape=%r ; nCentroids=%r\n'
        '[akmeans] * max_iters=%r\n'
        '[akmeans] * ave_unchanged_iterwin=%r ; ave_unchanged_thresh=%r\n'
    ) % (data.shape, nCentroids, max_iters,
         ave_unchanged_thresh, ave_unchanged_iterwin))
    sys.stdout.flush()
    _mark, _end = utool.log_progress('Akmeans: ', max_iters)
    for count in range(0, max_iters):
        _mark(count)
        # 1) Assign each datapoint to the nearest centroid
        datax2_centroidx = approximate_assignments(centroids, data, 1, flann_params)
        # 2) Compute new centroids based on assignments
        centroids = compute_centroids(data, centroids, datax2_centroidx)
        # 3) Convergence Check: which datapoints changed membership?
        num_changed = (datax2_centroidx_old != datax2_centroidx).sum()
        win2_unchanged[count % ave_unchanged_iterwin] = num_changed
        ave_unchanged = win2_unchanged.mean()
        if ave_unchanged < ave_unchanged_thresh:
            break
        else:
            datax2_centroidx_old = datax2_centroidx
    _end()
    return centroids


def approximate_distances(centroids, data, K, flann_params):
    (_, qdist2_sdist) = pyflann.FLANN().nn(centroids, data, K, **flann_params)
    return qdist2_sdist


def approximate_assignments(seachedvecs, queryvecs, K, flann_params):
    (qx2_sx, _) = pyflann.FLANN().nn(seachedvecs, queryvecs, K, **flann_params)
    return qx2_sx


def compute_centroids(data, centroids, datax2_centroidx):
    """
    Computes centroids given datax assignments
    TODO: maybe use the grouping code instad of the LR algorithm

    >>> from vtool.clustering2 import *  # NOQA
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(100, 2)
    >>> nCentroids = 5
    >>> flann_params = {}
    >>> centroids = initialize_centroids(nCentroids, data)
    >>> centroids_ = centroids.copy()
    >>> (datax2_centroidx, _) = p yflann.FLANN().nn(centroids, data, 1, **flann_params)
    >>> out = compute_centroids(data, centroids, datax2_centroidx)
    """
    nData = data.shape[0]
    nCentroids = centroids.shape[0]
    # sort data by centroid
    datax_sortx = datax2_centroidx.argsort()
    datax_sort  = datax2_centroidx[datax_sortx]
    # group datapoints by centroid using a sliding grouping algorithm
    centroidx2_dataLRx = [None] * nCentroids
    _L = 0
    for _R in range(nData + 1):  # Slide R
        if _R == nData or datax_sort[_L] != datax_sort[_R]:
            centroidx2_dataLRx[datax_sort[_L]] = (_L, _R)
            _L = _R
    # Compute the centers of each group (centroid) of datapoints
    for centroidx, dataLRx in enumerate(centroidx2_dataLRx):
        if dataLRx is None:
            continue  # ON EMPTY CLUSTER
        (_L, _R) = dataLRx
        # The centroid center is the mean of its datapoints
        centroids[centroidx] = np.mean(data[datax_sortx[_L:_R]], axis=0)
        #centroids[centroidx] = np.array(np.round(centroids[centroidx]), dtype=np.uint8)
    return centroids


#def group_indicies2(idx2_groupid):
#    """
#    >>> idx2_groupid = np.array(np.random.randint(0, 4, size=100))

#    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
#    """
#    # Sort items and idx2_groupid by groupid
#    sortx = idx2_groupid.argsort()
#    groupids_sorted = idx2_groupid[sortx]
#    num_items = idx2_groupid.size
#    # Find the boundaries between groups
#    diff = np.ones(num_items + 1, idx2_groupid.dtype)
#    diff[1:(num_items)] = np.diff(groupids_sorted)
#    idxs = np.where(diff > 0)[0]
#    num_groups = idxs.size - 1
#    # Groups are between bounding indexes
#    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T
#    groupxs = [sortx[lx:rx] for lx, rx in lrx_pairs]
#    return groupxs


def group_indicies_pandas(idx2_groupid):
    """
    >>> from vtool.clustering2 import *
    >>> idx2_groupid = np.array(np.random.randint(0, 8000, size=1000000))

    keys1, groupxs2 = group_indicies_pandas(idx2_groupid)
    keys2, groupxs2 = group_indicies(idx2_groupid)

    %timeit group_indicies_pandas(idx2_groupid)
    %timeit group_indicies(idx2_groupid)
    """
    import pandas as pd
    # Pandas is actually unreasonably fast here
    #%timeit dataframe = pd.DataFrame(idx2_groupid, columns=['groupid'])  # 135 us
    #%timeit dfgroup = dataframe.groupby('groupid')  # 33.9 us
    #%timeit groupid2_idxs = dfgroup.indices  # 197 ns
    series = pd.Series(idx2_groupid)  # 66 us
    group = series.groupby(series)    # 32.9 us
    groupid2_idxs = group.indices     # 194 ns
    # Compute inverted index
    groupxs = list(groupid2_idxs.values())  # 412 ns
    keys    = list(groupid2_idxs.keys())    # 488 ns
    return keys, groupxs
#    # Consistency check
#    #for wx in _wx2_idxs.keys():
#    #    assert set(_wx2_idxs[wx]) == set(_wx2_idxs2[wx])


#@profile

def jagged_group(groupids_list):
    """ flattens and returns group indexes into the flattened list """
    #flatx2_itemx = np.array(utool.flatten(itemxs_iter))
    flatids = np.array(utool.flatten(groupids_list))
    keys, groupxs = group_indicies(flatids)
    return keys, groupxs


def apply_jagged_grouping(unflat_items, groupxs):
    """ takes unflat_list and flat group indicies. Returns the unflat grouping """
    flat_items = np.array(utool.flatten(unflat_items))
    item_groups = apply_grouping(flat_items, groupxs)
    return item_groups
    #itemxs_iter = [[count] * len(idx2_groupid) for count, idx2_groupid in enumerate(groupids_list)]


def group_indicies(idx2_groupid):
    """
    group_indicies

    Args:
        idx2_groupid (ndarray): numpy array of group ids (must be numeric)

    Returns:
        tuple (ndarray, list of ndarrays): (keys, groupxs)

    Example:
        >>> from vtool.clustering2 import *  # NOQA
        >>> #np.random.seed(42)
        >>> #size = 10
        >>> #idx2_groupid = np.array(np.random.randint(0, 4, size=size))
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> (keys, groupxs) = group_indicies(idx2_groupid)
        >>> print((keys, groupxs))
        (array([1, 2, 3]), [array([1, 3, 5]), array([0, 2, 4, 6]), array([ 7,  8,  9, 10])])

    SeeAlso:
        apply_grouping

    References:
        http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
    """
    # Sort items and idx2_groupid by groupid
    sortx = idx2_groupid.argsort()  # 2.9%
    groupids_sorted = idx2_groupid[sortx]  # 3.1%
    num_items = idx2_groupid.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, idx2_groupid.dtype)  # 8.6%
    diff[1:(num_items)] = np.diff(groupids_sorted)  # 22.5%
    idxs = np.where(diff > 0)[0]  # 8.8%
    num_groups = idxs.size - 1  # 1.3%
    # Groups are between bounding indexes
    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T  # 28.8%
    groupxs = [sortx[lx:rx] for lx, rx in lrx_pairs]  # 17.5%
    # Unique group keys
    keys = groupids_sorted[idxs[0:num_groups]]  # 4.7%
    #items_sorted = items[sortx]
    #vals = [items_sorted[lx:rx] for lx, rx in lrx_pairs]
    return keys, groupxs


def apply_grouping(items, groupxs):
    """
    apply_grouping

    Args:
        items (ndarray):
        groupxs (list of ndarrays):

    Returns:
        list of ndarrays: grouped items

    SeeAlso:
        group_indicies

    Example:
        >>> from vtool.clustering2 import *  # NOQA
        >>> #np.random.seed(42)
        >>> #size = 10
        >>> #idx2_groupid = np.array(np.random.randint(0, 4, size=size))
        >>> #items = np.random.randint(5, 10, size=size)
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> items        = np.array([1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9])
        >>> (keys, groupxs) = group_indicies(idx2_groupid)
        >>> grouped_items = apply_grouping(items, groupxs)
        >>> print(grouped_items)
        [array([8, 5, 6]), array([1, 5, 8, 7]), array([5, 3, 0, 9])]
    """
    return [items.take(xs, axis=0) for xs in groupxs]
    #return [items[idxs] for idxs in groupxs]


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
    keys, groupxs = group_indicies(idx2_groupid)
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
        group_outerkeys, groupxs = group_indicies(outer_keys)
        subitem_iter = (items.take(xs, axis=0) for xs in groupxs)
        for outer_key, subitems in zip(group_outerkeys, subitem_iter):
            outerkey2_innerkey2_items[outer_key][inner_key] = subitems
    return outerkey2_innerkey2_items
    #daid2_wx2_drvecs = utool.ddict(lambda: utool.ddict(list))
    #for wx, aids, rvecs in zip(wx_sublist, aids_list, rvecs_list1):
    #    group_aids, groupxs = clustertool.group_indicies(aids)
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
        datax2_label = approximate_assignments(centroids, data, 1, {})
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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/vtool/vtool/clustering2.py
        profiler.sh ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
        python ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init
    """
    # Run any doctests
    testable_list = [
        akmeans_plusplus_init
    ]
    ut.doctest_funcs(testable_list)
