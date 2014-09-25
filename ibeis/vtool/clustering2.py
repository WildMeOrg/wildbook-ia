# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import range
import utool
import sys
import numpy as np
import scipy.sparse as spsparse
import pyflann
import vtool.nearest_neighbors as nntool

(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[clustering2]', DEBUG=False)


CLUSTERS_FNAME = 'akmeans_centroids'


def get_akmeans_cfgstr(data, nCentroids, max_iters=5, flann_params={},
                       use_data_hash=True, cfgstr='', akmeans_cfgstr=None):
    if akmeans_cfgstr is None:
        # compute a hashstr based on the data
        cfgstr += '_nC=%d,nIter=%d' % (nCentroids, max_iters)
        akmeans_cfgstr = nntool.get_flann_cfgstr(data, flann_params,
                                                 cfgstr, use_data_hash)
    return akmeans_cfgstr


def cached_akmeans(data, nCentroids, max_iters=5, flann_params={},
                   cache_dir='default', force_recomp=False, use_data_hash=True,
                   cfgstr='', refine=False, akmeans_cfgstr=None, use_cache=True,
                   appname='vtool'):
    """ precompute aproximate kmeans with builtin caching """
    print('[akmeans] pre_akmeans()')
    # filename prefix constants
    if cache_dir == 'default':
        print('[akmeans] using default cache dir')
        cache_dir = utool.get_app_resource_dir(appname)
        utool.ensuredir(cache_dir)
    # Build a cfgstr if the full one is not specified
    akmeans_cfgstr = get_akmeans_cfgstr(data, nCentroids, max_iters, flann_params,
                                        use_data_hash, cfgstr, akmeans_cfgstr)
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
        assert centroids.shape[0] == nCentroids, 'bad number of centroids'
        assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
        return centroids
    except IOError as ex:
        utool.printex(ex, 'cache miss', iswarning=True)
    except UserWarning:
        pass
    # First time computation
    print('[akmeans.precompute] pre_akmeans(): calling akmeans')
    centroids = akmeans(data, nCentroids, max_iters, flann_params)
    print('[akmeans.precompute] save and return')
    utool.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    assert centroids.shape[0] == nCentroids, 'bad number of centroids'
    assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
    return centroids


def akmeans(data, nCentroids, max_iters=5, flann_params={},
            ave_unchanged_thresh=0,
            ave_unchanged_iterwin=10):
    """
    Approximiate K-Means (using FLANN)
    Input: data - np.array with rows of data.
    Description: Quickly partitions data into K=nCentroids centroids.  Cluster
    centers are randomly assigned to datapoints.  Each datapoint is assigned to
    its approximate nearest centroid center.  The centroid centers are recomputed.
    Repeat until approximate convergence."""
    # Setup iterations
    centroids = initialize_centroids(nCentroids, data)
    centroids = akmeans_iterations(data, centroids, max_iters, flann_params,
                                    ave_unchanged_thresh, ave_unchanged_iterwin)
    return centroids


def initialize_centroids(nCentroids, data):
    """ Initializes centroids to random datapoints """
    nData = data.shape[0]
    datax_rand = np.arange(0, nData, dtype=np.int32)
    np.random.shuffle(datax_rand)
    centroidx2_datax = datax_rand[0:nCentroids]
    centroids = np.copy(data[centroidx2_datax])
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
        (datax2_centroidx, _) = pyflann.FLANN().nn(centroids, data, 1, **flann_params)
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
    >>> (datax2_centroidx, _) = pyflann.FLANN().nn(centroids, data, 1, **flann_params)
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


#def group_indicies2(groupids):
#    """
#    >>> groupids = np.array(np.random.randint(0, 4, size=100))

#    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
#    """
#    # Sort items and groupids by groupid
#    sortx = groupids.argsort()
#    groupids_sorted = groupids[sortx]
#    num_items = groupids.size
#    # Find the boundaries between groups
#    diff = np.ones(num_items + 1, groupids.dtype)
#    diff[1:(num_items)] = np.diff(groupids_sorted)
#    idxs = np.where(diff > 0)[0]
#    num_groups = idxs.size - 1
#    # Groups are between bounding indexes
#    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T
#    groupxs = [sortx[lx:rx] for lx, rx in lrx_pairs]
#    return groupxs


#def group_indicies_pandas(groupids):
#    # Pandas is actually unreasonably fast here
#    word_assignments = pd.DataFrame(_idx2_wx, columns=['wx'])  # 141 us
#    # Compute inverted index
#    word_group = word_assignments.groupby('wx')  # 34.5 us
#    _wx2_idxs = word_group['wx'].indices  # 8.6 us
#    # Consistency check
#    #for wx in _wx2_idxs.keys():
#    #    assert set(_wx2_idxs[wx]) == set(_wx2_idxs2[wx])


#@profile
def group_indicies(groupids):
    """
    Total time: 1.29423 s
    >>> groupids = np.array(np.random.randint(0, 4, size=100))

    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
    """
    # Sort items and groupids by groupid
    sortx = groupids.argsort()  # 2.9%
    groupids_sorted = groupids[sortx]  # 3.1%
    num_items = groupids.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, groupids.dtype)  # 8.6%
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
    return [items.take(idxs, axis=0) for idxs in groupxs]
    #return [items[idxs] for idxs in groupxs]


def groupby(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    >>> items = groupids
    """
    keys, groupxs = group_indicies(groupids)
    vals = [items[idxs] for idxs in groupxs]
    return keys, vals


def groupby_gen(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    """
    for key, val in zip(*groupby(items, groupids)):
        yield (key, val)


def groupby_dict(items, groupids):
    # Build a dict
    grouped = {key: val for key, val in groupby_gen(items, groupids)}
    return grouped


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
        datax2_label = pyflann.FLANN().nn(centroids, data, 1)[0]
    else:
        datax2_label = labels
    datax2_label = np.array(datax2_label, dtype=np.int32)
    print(datax2_label)
    assert len(datax2_label.shape) == 1, repr(datax2_label.shape)
    #if datax2_centroids is None:
    #    (datax2_centroidx, _) = pyflann.FLANN().nn(centroids, data, 1)
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
            (utool.mystats, 'colors'),
            'colors.shape',
            'datax2_label',
            (utool.mystats, 'datax2_label'),
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
