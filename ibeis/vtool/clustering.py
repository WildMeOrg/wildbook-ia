# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import range
import utool
import sys
import numpy as np
import scipy.sparse as spsparse
import vtool.nearest_neighbors as nn

(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[akmeans]', DEBUG=False)


CLUSTERS_FNAME = 'akmeans_clusters'
DATAX2CL_FNAME = 'akmeans_datax2cl'


#@profile
def akmeans(data, num_clusters, max_iters=5, flann_params={},
            ave_unchanged_thresh=0,
            ave_unchanged_iterwin=10):
    """Approximiate K-Means (using FLANN)
    Input: data - np.array with rows of data.
    Description: Quickly partitions data into K=num_clusters centroids.  Cluster
    centers are randomly assigned to datapoints.  Each datapoint is assigned to
    its approximate nearest cluster center.  The cluster centers are recomputed.
    Repeat until approximate convergence."""

    # Setup iterations
    #data   = np.array(data, __BOW_DTYPE__)
    num_data = data.shape[0]
    index_dtype = np.uint32  # specify cluster index datatype
    # Initialize to random cluster centroids
    datax_rand = np.arange(0, num_data, dtype=index_dtype)
    np.random.shuffle(datax_rand)
    clusterx2_datax     = datax_rand[0:num_clusters]
    centroids            = np.copy(data[clusterx2_datax])
    datax2_clusterx_old = -np.ones(len(data), dtype=datax_rand.dtype)
    # This function does the work
    (datax2_clusterx, centroids) = _akmeans_iterate(data, centroids,
                                                   datax2_clusterx_old,
                                                   max_iters, flann_params,
                                                   ave_unchanged_thresh,
                                                   ave_unchanged_iterwin)
    return (datax2_clusterx, centroids)


def precompute_akmeans(data, num_clusters, max_iters=5, flann_params={},
                       cache_dir=None, force_recomp=False, use_data_hash=True,
                       cfgstr='', refine=False, akmeans_cfgstr=None):
    """ precompute aproximate kmeans with builtin caching """
    print('[akmeans] pre_akmeans()')
    # filename prefix constants
    assert cache_dir is not None, 'choose a cache directory'
    # Build a cfgstr if the full one is not specified
    if akmeans_cfgstr is None:
        # compute a hashstr based on the data
        akmeans_cfgstr = nn.get_flann_cfgstr(data, flann_params, cfgstr, use_data_hash)
    try:
        # Try and load a previous clustering
        if force_recomp:
            raise UserWarning('forceing recommpute')
        centroids        = utool.load_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr)
        datax2_clusterx = utool.load_cache(cache_dir, DATAX2CL_FNAME, akmeans_cfgstr)
        print('[akmeans.precompute] load successful')
        if refine:
            # Refines the cluster centers if specified
            (datax2_clusterx, centroids) =\
                refine_akmeans(data, datax2_clusterx, centroids,
                               max_iters=max_iters, flann_params=flann_params,
                               cache_dir=cache_dir, akmeans_cfgstr=akmeans_cfgstr)
        return (datax2_clusterx, centroids)
    except IOError as ex:
        utool.printex(ex, 'cache miss', iswarning=True)
    except UserWarning:
        pass
    # First time computation
    print('[akmeans.precompute] pre_akmeans(): calling akmeans')
    (datax2_clusterx, centroids) = akmeans(data, num_clusters, max_iters, flann_params)
    print('[akmeans.precompute] save and return')
    utool.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    utool.save_cache(cache_dir, DATAX2CL_FNAME, akmeans_cfgstr, datax2_clusterx)
    return (datax2_clusterx, centroids)


def refine_akmeans(data, datax2_clusterx, centroids, max_iters=5,
                   flann_params={}, cache_dir=None, cfgstr='',
                   use_data_hash=True, akmeans_cfgstr=None):
    """ Refines the approximates centroids """
    print('[akmeans.precompute] refining:')
    if akmeans_cfgstr is None:
        akmeans_cfgstr = nn.get_flann_cfgstr(data, flann_params, cfgstr, use_data_hash)
    datax2_clusterx_old = datax2_clusterx
    (datax2_clusterx, centroids) = _akmeans_iterate(data, centroids, datax2_clusterx_old, max_iters, flann_params, 0, 10)
    utool.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    utool.save_cache(cache_dir, DATAX2CL_FNAME, akmeans_cfgstr, datax2_clusterx)
    return (datax2_clusterx, centroids)


def sparse_normalize_rows(csr_mat):
    pass
    #return sklearn.preprocessing.normalize(csr_mat, norm='l2', axis=1, copy=False)


def sparse_multiply_rows(csr_mat, vec):
    """ Row-wise multiplication of a sparse matrix by a sparse vector """
    csr_vec = spsparse.csr_matrix(vec, copy=False)
    #csr_vec.shape = (1, csr_vec.size)
    sparse_stack = [row.multiply(csr_vec) for row in csr_mat]
    return spsparse.vstack(sparse_stack, format='csr')


def force_quit_akmeans(signal, frame):
    # FIXME OR DEPRICATE
    try:
        print(utool.unindedent('''
                              --- algos ---
                              Caught Ctrl+C in:
                              function: %r
                              stacksize: %r
                              line_no: %r
                              ''') %
             (frame.f_code.co_name, frame.f_code.co_stacksize, frame.f_lineno))
        #exec(df2.present())
        target_frame = frame
        target_frame_coname = '_akmeans_iterate'
        while True:
            if target_frame.f_code.co_name == target_frame_coname:
                break
            if target_frame.f_code.co_name == '<module>':
                print('Traced back to module level. Missed frame: %r ' %
                      target_frame_coname)
                break
            target_frame = target_frame.f_back
            print('Is target frame?: ' + target_frame.f_code.co_name)

        fpath = target_frame.f_back.f_back.f_locals['fpath']

        #data            = target_frame.f_locals['data']
        centroids        = target_frame.f_locals['centroids']
        datax2_clusterx = target_frame.f_locals['datax2_clusterx']
        utool.save_npz(fpath + '.earlystop', datax2_clusterx, centroids)
    except Exception as ex:
        print(repr(ex))
        exec(utool.IPYTHON_EMBED_STR)


def _compute_cluster_centers(num_data, num_clusters, data, centroids, datax2_clusterx):
    """ Computes the cluster centers and stores output in the outvar: centroids.
    This outvar is also returned """
    # sort data by cluster
    datax_sort    = datax2_clusterx.argsort()
    clusterx_sort = datax2_clusterx[datax_sort]
    # group datapoints by cluster using a sliding grouping algorithm
    _L = 0
    clusterx2_dataLRx = utool.alloc_nones(num_clusters)
    for _R in range(len(datax_sort) + 1):  # Slide R
        if _R == num_data or clusterx_sort[_L] != clusterx_sort[_R]:
            clusterx2_dataLRx[clusterx_sort[_L]] = (_L, _R)
            _L = _R
    # Compute the centers of each group (cluster) of datapoints
    utool.print_('+')
    for clusterx, dataLRx in enumerate(clusterx2_dataLRx):
        if dataLRx is None:
            continue  # ON EMPTY CLUSTER
        (_L, _R) = dataLRx
        # The cluster center is the mean of its datapoints
        centroids[clusterx] = np.mean(data[datax_sort[_L:_R]], axis=0)
        #centroids[clusterx] = np.array(np.round(centroids[clusterx]), dtype=np.uint8)
    return centroids


#@profile
def _akmeans_iterate(data, centroids, datax2_clusterx_old, max_iters,
                     flann_params, ave_unchanged_thresh, ave_unchanged_iterwin):
    """ Helper function which continues the iterations of akmeans """
    num_data = data.shape[0]
    num_clusters = centroids.shape[0]
    # Keep track of how many points have changed in each iteration
    xx2_unchanged = np.zeros(ave_unchanged_iterwin, dtype=centroids.dtype) + len(data)
    print('[akmeans] Running akmeans: data.shape=%r ; num_clusters=%r' %
          (data.shape, num_clusters))
    print('[akmeans] * max_iters = %r ' % max_iters)
    print('[akmeans] * ave_unchanged_iterwin=%r ; ave_unchanged_thresh=%r' %
          (ave_unchanged_thresh, ave_unchanged_iterwin))
    #print('[akmeans] Printing akmeans info in format: time (iterx, ave(#changed), #unchanged)')
    xx = 0
    for xx in range(0, max_iters):
        tt = utool.tic()
        utool.print_('...tic')
        # 1) Find each datapoints nearest cluster center
        (datax2_clusterx, _dist) = nn.ann_flann_once(centroids, data, 1, flann_params)
        ellapsed = utool.toc(tt)
        utool.print_('...toc(%.2fs)' % ellapsed)
        # 2) Compute new cluster centers
        centroids = _compute_cluster_centers(num_data, num_clusters, data, centroids, datax2_clusterx)
        # 3) Check for convergence (no change of cluster index)
        #utool.print_('+')
        num_changed = (datax2_clusterx_old != datax2_clusterx).sum()
        xx2_unchanged[xx % ave_unchanged_iterwin] = num_changed
        ave_unchanged = xx2_unchanged.mean()
        #utool.print_('  (%d, %.2f, %d)\n' % (xx, ave_unchanged, num_changed))
        if ave_unchanged < ave_unchanged_thresh:
            break
        else:  # Iterate
            datax2_clusterx_old = datax2_clusterx
            #if xx % 5 == 0:
            #    sys.stdout.flush()
    if xx == max_iters:
        print('[akmeans]  * AKMEANS: converged in %d/%d iters' % (xx + 1, max_iters))
    else:
        print('[akmeans]  * AKMEANS: reached the maximum iterations after in %d/%d iters' % (xx + 1, max_iters))
    sys.stdout.flush()
    return (datax2_clusterx, centroids)


# ---------------
# Plotting Code
# ---------------

def plot_clusters(data, datax2_clusterx, centroids, num_pca_dims=3,
                  whiten=False):
    """ Plots centroids and datapoints. Plots accurately up to 3 dimensions.
    If there are more than 3 dimensions, PCA is used to recude the dimenionality
    to the <num_pca_dims> principal components
    """
    # http://www.janeriksolem.net/2012/03/isomap-with-scikit-learn.html
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
        pca_clusters = pca.transform(centroids)
        print('[akmeans] ...Finished PCA')
    else:
        # pca is not necessary
        print('[akmeans] No need for PCA')
        pca_data = data
        pca_clusters = centroids
    K = len(centroids)
    print(pca_data.shape)
    # Make a color for each cluster
    colors = np.array(df2.distinct_colors(K, brightness=.95))
    data_x = pca_data[:, 0]
    data_y = pca_data[:, 1]
    data_colors = colors[np.array(datax2_clusterx, dtype=np.int32)]
    clus_x = pca_clusters[:, 0]
    clus_y = pca_clusters[:, 1]
    clus_colors = colors
    # Create a figure
    fig = df2.figure(1, doclf=True, docla=True)
    if show_dims == 2:
        ax = df2.plt.gca()
        df2.plt.scatter(data_x, data_y, s=20,  c=data_colors, marker='o', alpha=.2)
        df2.plt.scatter(clus_x, clus_y, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        df2.dark_background(ax)
    if show_dims == 3:
        from mpl_toolkits.mplot3d import Axes3D  # NOQA
        ax = fig.add_subplot(111, projection='3d')
        data_z = pca_data[:, 2]
        clus_z = pca_clusters[:, 2]
        ax.scatter(data_x, data_y, data_z, s=20,  c=data_colors, marker='o', alpha=.2)
        ax.scatter(clus_x, clus_y, clus_z, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        df2.dark_background(ax)
        #ax.set_alpha(.1)
        #utool.embed()
        #ax.set_frame_on(False)
    ax = df2.plt.gca()
    waswhitestr = ' +whitening' * whiten
    titlestr = ('AKmeans: K={K}.'
                'PCA projection {data_dims}D -> {show_dims}D'
                '{waswhitestr}').format(**locals())
    ax.set_title(titlestr)
    return fig
