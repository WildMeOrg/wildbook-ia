# LICENCE
from __future__ import absolute_import, division, print_function
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
    Description: Quickly partitions data into K=num_clusters clusters.  Cluster
    centers are randomly assigned to datapoints.  Each datapoint is assigned to
    its approximate nearest cluster center.  The cluster centers are recomputed.
    Repeat until approximate convergence."""

    # Setup iterations
    #data   = np.array(data, __BOW_DTYPE__)
    num_data = data.shape[0]
    index_dtype = np.uint32  # specify cluster index datatype
    # Initialize to random cluster clusters
    datax_rand = np.arange(0, num_data, dtype=index_dtype)
    np.random.shuffle(datax_rand)
    clusterx2_datax     = datax_rand[0:num_clusters]
    clusters            = np.copy(data[clusterx2_datax])
    datax2_clusterx_old = -np.ones(len(data), dtype=datax_rand.dtype)
    # This function does the work
    (datax2_clusterx, clusters) = _akmeans_iterate(data, clusters,
                                                   datax2_clusterx_old,
                                                   max_iters, flann_params,
                                                   ave_unchanged_thresh,
                                                   ave_unchanged_iterwin)
    return (datax2_clusterx, clusters)


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
        clusters        = target_frame.f_locals['clusters']
        datax2_clusterx = target_frame.f_locals['datax2_clusterx']
        utool.save_npz(fpath + '.earlystop', datax2_clusterx, clusters)
    except Exception as ex:
        print(repr(ex))
        exec(utool.IPYTHON_EMBED_STR)


def _compute_cluster_centers(num_data, num_clusters, data, clusters, datax2_clusterx):
    """ Computes the cluster centers and stores output in the outvar: clusters.
    This outvar is also returned """
    # sort data by cluster
    datax_sort    = datax2_clusterx.argsort()
    clusterx_sort = datax2_clusterx[datax_sort]
    # group datapoints by cluster using a sliding grouping algorithm
    _L = 0
    clusterx2_dataLRx = utool.alloc_nones(num_clusters)
    for _R in xrange(len(datax_sort) + 1):  # Slide R
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
        clusters[clusterx] = np.mean(data[datax_sort[_L:_R]], axis=0)
        #clusters[clusterx] = np.array(np.round(clusters[clusterx]), dtype=np.uint8)
    return clusters


#@profile
def _akmeans_iterate(data, clusters, datax2_clusterx_old, max_iters,
                     flann_params, ave_unchanged_thresh, ave_unchanged_iterwin):
    """ Helper function which continues the iterations of akmeans """
    num_data = data.shape[0]
    num_clusters = clusters.shape[0]
    # Keep track of how many points have changed in each iteration
    xx2_unchanged = np.zeros(ave_unchanged_iterwin, dtype=clusters.dtype) + len(data)
    print('[akmeans] Running akmeans: data.shape=%r ; num_clusters=%r' %
          (data.shape, num_clusters))
    print('[akmeans] * max_iters = %r ' % max_iters)
    print('[akmeans] * ave_unchanged_iterwin=%r ; ave_unchanged_thresh=%r' %
          (ave_unchanged_thresh, ave_unchanged_iterwin))
    #print('[akmeans] Printing akmeans info in format: time (iterx, ave(#changed), #unchanged)')
    xx = 0
    for xx in xrange(0, max_iters):
        tt = utool.tic()
        utool.print_('...tic')
        # 1) Find each datapoints nearest cluster center
        (datax2_clusterx, _dist) = nn.ann_flann_once(clusters, data, 1, flann_params)
        ellapsed = utool.toc(tt)
        utool.print_('...toc(%.2fs)' % ellapsed)
        # 2) Compute new cluster centers
        clusters = _compute_cluster_centers(num_data, num_clusters, data, clusters, datax2_clusterx)
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
    return (datax2_clusterx, clusters)


def refine_akmeans(data, datax2_clusterx, clusters, max_iters=5,
                   flann_params={}, cache_dir=None, cfgstr='',
                   use_data_hash=True):
    """ Refines the approximates clusters """
    print('[akmeans.precompute] refining:')
    if use_data_hash:
        # compute a hashstr based on the data
        data_hashstr = utool.hashstr_arr(data, '_dID')
        cfgstr += data_hashstr
    datax2_clusterx_old = datax2_clusterx
    (datax2_clusterx, clusters) = _akmeans_iterate(data, clusters, datax2_clusterx_old, max_iters, flann_params, 0, 10)
    utool.save_cache(cache_dir, CLUSTERS_FNAME, cfgstr, clusters)
    utool.save_cache(cache_dir, DATAX2CL_FNAME, cfgstr, datax2_clusterx)
    return (datax2_clusterx, clusters)


def precompute_akmeans(data, num_clusters, max_iters=5, flann_params={},
                       cache_dir=None, force_recomp=False, use_data_hash=True,
                       cfgstr='', refine=False):
    """ precompute aproximate kmeans with builtin caching """
    print('[akmeans] pre_akmeans()')
    # filename prefix constants
    assert cache_dir is not None, 'choose a cache directory'
    if use_data_hash:
        # compute a hashstr based on the data
        data_hashstr = utool.hashstr_arr(data, '_dID')
        cfgstr += data_hashstr
    try:
        # Try and load a previous clustering
        if force_recomp:
            raise UserWarning('forceing recommpute')
        clusters        = utool.load_cache(cache_dir, CLUSTERS_FNAME, cfgstr)
        datax2_clusterx = utool.load_cache(cache_dir, DATAX2CL_FNAME, cfgstr)
        print('[akmeans.precompute] load successful')
        if refine:
            # Refines the cluster centers if specified
            (datax2_clusterx, clusters) =\
                refine_akmeans(data, datax2_clusterx, clusters,
                               max_iters=max_iters, flann_params=flann_params,
                               cache_dir=cache_dir, cfgstr=cfgstr,
                               use_data_hash=False)
        return (datax2_clusterx, clusters)
    except IOError as ex:
        utool.printex(ex, 'cache miss', iswarning=True)
    except UserWarning:
        pass
    # First time computation
    print('[akmeans.precompute] pre_akmeans(): calling akmeans')
    (datax2_clusterx, clusters) = akmeans(data, num_clusters, max_iters, flann_params)
    print('[akmeans.precompute] save and return')
    utool.save_cache(cache_dir, CLUSTERS_FNAME, cfgstr, clusters)
    utool.save_cache(cache_dir, DATAX2CL_FNAME, cfgstr, datax2_clusterx)
    return (datax2_clusterx, clusters)


def plot_clusters(data, datax2_clusterx, clusters, num_pca_dims=3,
                  whiten=False):
    # http://www.janeriksolem.net/2012/03/isomap-with-scikit-learn.html
    print('[akmeans] Doing PCA')
    from plottool import draw_func2 as df2
    data_dims = data.shape[1]
    num_pca_dims = min(num_pca_dims, data_dims)
    pca = None
    #pca = sklearn.decomposition.PCA(copy=True, n_components=num_pca_dims,
                                    #whiten=whiten).fit(data)
    pca_data = pca.transform(data)
    pca_clusters = pca.transform(clusters)
    K = len(clusters)
    print('[akmeans] ...Finished PCA')
    fig = df2.plt.figure(1)
    fig.clf()
    #cmap = plt.get_cmap('hsv')
    data_x = pca_data[:, 0]
    data_y = pca_data[:, 1]
    colors = np.array(df2.distinct_colors(K))
    print(colors)
    print(datax2_clusterx)
    data_colors = colors[np.array(datax2_clusterx, dtype=np.int32)]
    clus_x = pca_clusters[:, 0]
    clus_y = pca_clusters[:, 1]
    clus_colors = colors
    if num_pca_dims == 2:
        ax = df2.plt.gca()
        df2.plt.scatter(data_x, data_y, s=20,  c=data_colors, marker='o', alpha=.2)
        df2.plt.scatter(clus_x, clus_y, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
    if num_pca_dims == 3:
        from mpl_toolkits.mplot3d import Axes3D  # NOQA
        ax = fig.add_subplot(111, projection='3d')
        data_z = pca_data[:, 2]
        clus_z = pca_clusters[:, 2]
        ax.scatter(data_x, data_y, data_z, s=20,  c=data_colors, marker='o', alpha=.2)
        ax.scatter(clus_x, clus_y, clus_z, s=500, c=clus_colors, marker='*')
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
    ax = df2.plt.gca()
    ax.set_title('AKmeans clustering. K=%r. PCA projection %dD -> %dD%s' %
                 (K, data_dims, num_pca_dims, ' +whitening' * whiten))
    return fig
