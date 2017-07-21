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
        cache_dir = ut.get_app_resource_dir('vtool')
        ut.ensuredir(cache_dir)
    if akmeans_cfgstr is None:
        akmeans_cfgstr = nntool.get_flann_cfgstr(
            data, flann_params, cfgstr, use_data_hash)
    centroids = akmeans_iterations(data, centroids, max_iters, flann_params, 0, 10)
    ut.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
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
        >>> # DISABLE_DOCTEST
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
        >>> ut.qtensure()
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
    for count in ut.ProgIter(range(0, max_iters), length=max_iters, lbl='Akmeans: '):
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


CLUSTERS_FNAME = 'akmeans_centroids'


def testdata_kmeans():
    # import utool as ut
    from sklearn.utils import check_array
    # from sklearn.utils.extmath import row_norms, squared_norm
    # from sklearn.metrics.pairwise import euclidean_distances
    # import warnings
    # import scipy.sparse as sp
    rng = np.random.RandomState(42)
    K = 1000
    check_inputs = True
    nump = 10000
    dims = 128
    dtype = np.uint8
    data = rng.randint(0, 255, (nump, dims)).astype(dtype)
    n_local_trials = 1
    verbose = True
    X = data
    n_clusters = K
    random_state = rng
    import numpy as np
    rng = np.random.RandomState(42)
    nump, dims = K ** 2, 128
    # dtype = np.uint8
    dtype = np.float32
    data = rng.randint(0, 255, (nump, dims)).astype(dtype)
    num_samples = None
    flann_params = None
    X = data
    X = check_array(X, accept_sparse="csr", order='C',
                    dtype=[np.float32])
    data = X
    return locals()


def kmeans_plusplus_sklearn(X, K, **kwargs):
    import sklearn.cluster
    from sklearn.utils.extmath import row_norms
    from sklearn.utils import check_array
    from sklearn.utils import check_random_state

    self = sklearn.cluster.MiniBatchKMeans(n_clusters=K, **kwargs)

    random_state = check_random_state(self.random_state)
    X = check_array(X, accept_sparse="csr", order='C',
                    dtype=[np.float64, np.float32])
    n_samples, n_features = X.shape
    if n_samples < self.n_clusters:
        raise ValueError("Number of samples smaller than number "
                         "of clusters.")

    x_squared_norms = row_norms(X, squared=True)

    if self.tol > 0.0:
        old_center_buffer = np.zeros(n_features, dtype=X.dtype)
    else:
        old_center_buffer = np.zeros(0, dtype=X.dtype)

    init_size = self.init_size
    if init_size is None:
        init_size = 3 * self.batch_size
    if init_size > n_samples:
        init_size = n_samples
    self.init_size_ = init_size

    validation_indices = random_state.choice(n_samples, init_size,
                                             replace=False)
    X_valid = X[validation_indices]
    x_squared_norms_valid = x_squared_norms[validation_indices]

    n_init = self.n_init

    # perform several inits with random sub-sets
    best_inertia = None
    for init_idx in range(n_init):
        if self.verbose:
            print("Init %d/%d with method: %s"
                  % (init_idx + 1, n_init, self.init))
        counts = np.zeros(self.n_clusters, dtype=np.int32)

        # TODO: once the `k_means` function works with sparse input we
        # should refactor the following init to use it instead.

        # Initialize the centers using only a fraction of the data as we
        # expect n_samples to be very large when using MiniBatchKMeans
        cluster_centers = sklearn.cluster.k_means_._init_centroids(
            X, self.n_clusters, self.init,
            random_state=random_state,
            x_squared_norms=x_squared_norms,
            init_size=init_size, check_inputs=False)

        # Compute the label assignment on the init dataset
        _t = sklearn.cluster.k_means_._mini_batch_step(
            X_valid, x_squared_norms[validation_indices],
            cluster_centers, counts, old_center_buffer, False,
            distances=None, verbose=self.verbose)
        batch_inertia, centers_squared_diff = _t

        # Keep only the best cluster centers across independent inits on
        # the common validation set
        _, inertia = sklearn.cluster.k_means_._labels_inertia(
            X_valid, x_squared_norms_valid, cluster_centers)

        if self.verbose:
            print("Inertia for init %d/%d: %f"
                  % (init_idx + 1, n_init, inertia))
        if best_inertia is None or inertia < best_inertia:
            self.cluster_centers_ = cluster_centers
            self.counts_ = counts
            best_inertia = inertia
    centers = self.cluster_centers_
    return centers


def k_means_pp_cv2(data, K):
    # define criteria and apply kmeans()
    # Crieteria is a 3-tuple:
    #  (type, max_iter, epsilon)
    import cv2
    max_iter = 100
    n_init = 1
    # with ut.Timer('sklearn2km'):
    #     est = sklearn.cluster.KMeans(n_clusters=K, max_iter=max_iter,
    #                                  n_init=n_init)
    #     est.fit(data)

    with ut.Timer('sklearn2km++'):
        clusters = kmeans_plusplus_sklearn(data, K)  # NOQA

    criteria_type = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER
    max_iter = 0
    epsilon = 0
    criteria = (criteria_type, max_iter, epsilon)
    with ut.Timer('cv2km++'):
        loss, label, center = cv2.kmeans(data=data, K=K, bestLabels=None,
                                         criteria=criteria, attempts=n_init,
                                         flags=cv2.KMEANS_PP_CENTERS)


def akmeans_plusplus_init(data, K, num_samples=None, flann_params=None,
                          rng=None):
    """
    Referencs:
        http://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

    Example:
        >>> # SLOW_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> K = 1000
        >>> nump, dims = K ** 2, 128
        >>> dtype = np.uint8
        >>> data = rng.randint(0, 255, (nump, dims)).astype(dtype)
        >>> num_samples = None
        >>> flann_params = None
        >>> centers = akmeans_plusplus_init(data, K, num_samples, flann_params)

    Example:
        >>> # SLOW_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import sklearn.cluster
        >>> from sklearn.cluster import *
        >>> from sklearn.utils import check_array
        >>> from sklearn.utils.extmath import row_norms, squared_norm
        >>> from sklearn.metrics.pairwise import euclidean_distances
        >>> import warnings
        >>> import scipy.sparse as sp
        >>> rng = np.random.RandomState(42)
        >>> K = 6400
        >>> check_inputs=True
        >>> nump, dims = K * 10, 128
        >>> dtype = np.uint8
        >>> data = rng.randint(0, 255, (nump, dims)).astype(dtype)
        >>> n_local_trials = 1
        >>> verbose = True
        >>> X = data
        >>> n_clusters = K
        >>> random_state = rng
        >>> X = check_array(X, accept_sparse="csr", order='C',
        >>>                 dtype=[np.float32])
        >>> x_squared_norms = row_norms(X, squared=True)[np.newaxis, :]
        >>> centers = k_means_plus_plus(data, K, rng)

    Example2:
        >>> # SLOW_DOCTEST
        >>> from vtool.clustering2 import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> data = np.vstack(ibs.get_annot_vecs(ibs.get_valid_aids()))
        >>> flann_params = None
        >>> num_samples = 1000
        >>> K = 8000  # 64000
        >>> initial_centers = akmeans_plusplus_init(data, K, num_samples,
        >>>                                         flann_params)

    CommandLine:
        python -m vtool akmeans_plusplus_init:0
        python -m vtool akmeans_plusplus_init:1
        python -m vtool akmeans_plusplus_init:0 --profile

        vt
        cd vtool
        wget https://gist.githubusercontent.com/dwf/2200359/raw/aa3c79c6f432ad630cc6e01f1ba2dfbef238bfeb/kmeans.pyx
        wget https://gist.githubusercontent.com/dwf/2200359/raw/aa3c79c6f432ad630cc6e01f1ba2dfbef238bfeb/setup.py

        import numpy as np
        import cv2
        from matplotlib import pyplot as plt

        X = np.random.randint(25,50,(25,2))
        Y = np.random.randint(60,85,(25,2))
        Z = np.vstack((X,Y))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria and apply kmeans()
        # Crieteria is a 3-tuple:
        #  (type, max_iter, epsilon)
        criteria_type = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER
        max_iter = 0
        epsilon = 0
        criteria = (criteria_type, max_iter, epsilon)
        loss, label, center = cv2.kmeans(
                data=Z, K=2, bestLabels=None, criteria=criteria, attempts=1,
                flags=cv2.KMEANS_PP_CENTERS)

        # ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        https://github.com/dbelll/bell_d_project/blob/master/cuda_kmeans.py

        http://mloss.org/software/view/48/

        export PATH=$PATH:/home/joncrall/venv2/lib/python2.7/site-packages/numpy/core/include
        https://github.com/argriffing/pyvqcore
        export CFLAGS="$CFLAGS -I/home/joncrall/venv2/lib/python2.7/site-packages/numpy/core/include/"
        pip install git+https://github.com/argriffing/pyvqcore.git

        python ~/code/vtool/vtool/clustering2.py --test-akmeans_plusplus_init

        python -m plottool.draw_func2 --exec-plot_func --show --range=0,64000 \
                --func="lambda K: 64 / K "

        python -m plottool.draw_func2 --exec-plot_func --show --range=0,1E7 \
                --func="lambda N: N * (64 / 65000) "


    """
    raise NotImplementedError('use sklearn or opencv')
    return kmeans_plusplus_sklearn(data, K)
    # # import pyflann
    # # import six

    # # rng = ut.ensure_rng(rng)

    # # if num_samples is None:
    #     # num_samples = 8192
    # # num_samples = min(num_samples, len(data))
    # # do_sampling = num_samples < len(data)

    # # print('akmeans++ on %r points. K=%r, num_samples=%r. ' % (
    #     # len(data), K, num_samples))

    # # if len(data) == K:
    #     # print('Warning, K is the same size as data')
    #     # return data

    # # assert len(data) > K

    # # # Allocate data for centers
    # # centers = np.empty((K, data.shape[1]), dtype=data.dtype)

    # # # Create a mask denoting all unused elements
    # # num_unused = len(data)
    # # num_used = 0
    # # is_unused = np.ones(len(data), dtype=np.bool)
    # # unused_didxs = list(range(len(data)))
    # # used_didxs = []

    # # prog = ut.ProgPartial(lbl='akmeans++ init', freq=1, adjust=True, bs=True)
    # # _iter = iter(prog(range(K)))

    # # six.next(_iter)

    # # # Choose an index and "use" it
    # # chosen_datax = rng.randint(0, num_unused)
    # # is_unused[chosen_datax] = False
    # # centers[num_used] = data[chosen_datax]
    # # num_used += 1
    # # num_unused -= 1

    # # # initalize flann index for approximate nn calculation
    # # if flann_params is None:
    #     # flann_params = {}
    #     # #flann_params['algorithm'] = 'linear'
    #     # flann_params['algorithm'] = 'kdtree'
    #     # flann_params['trees'] = 1
    #     # flann_params['checks'] = 8

    # # flann = None

    # # import vtool as vt
    # # try:
    #     # for count in _iter:

    #         # # Randomly sample choose a set of data vectors
    #         # if do_sampling:
    #             # unused_didx = np.where(is_unused)[0]
    #             # sx_to_uidx = rng.randint(num_unused, size=num_samples)
    #             # sx_to_didx = unused_didx[sx_to_uidx]
    #             # sx_to_data = data.take(sx_to_didx, axis=0)
    #         # else:
    #             # sx_to_data = data.compress(is_unused, axis=0)

    #         # flann_on = num_used > 128

    #         # if flann_on:
    #             # if flann is None:
    #                 # flann = pyflann.FLANN()
    #                 # flann.build_index(centers[:num_used], **flann_params)
    #             # # Distance from the sample data to current centers
    #             # # (this call takes 98% of the time.)
    #             # sx2_cidx, sx2_cdist = flann.nn_index(
    #                 # sx_to_data, 1, checks=flann_params['checks'])
    #         # else:
    #             # # Dont use flann when number of centers is small
    #             # # vt.L2(sx_to_data, centers[:num_used])
    #             # import sklearn
    #             # dists = sklearn.metrics.pairwise.euclidean_distances(
    #                 # sx_to_data, centers[:num_used])

    #         # # Choose unused sample with probability proportional to the squared
    #         # # distance to the closest existing center
    #         # rand_vals = rng.random_sample()
    #         # sx2_cumdist = sx2_cdist.cumsum()
    #         # sx2_cumprob = sx2_cumdist / sx2_cumdist[-1]
    #         # chosen_sx = np.searchsorted(sx2_cumprob, rand_vals)
    #         # if do_sampling:
    #             # chosen_datax = sx_to_didx[chosen_sx]
    #         # else:
    #             # chosen_datax = chosen_sx

    #         # # Remove the chosen index from unused indices
    #         # is_unused[chosen_datax] = False
    #         # chosen_data = data[chosen_datax]
    #         # centers[num_used] = chosen_data
    #         # num_used += 1
    #         # num_unused -= 1

    #         # if flann_on:
    #             # # Append new center to data and flann index
    #             # flann.add_points(chosen_data)

    # # except KeyboardInterrupt:
    #     # print('\n\n')
    #     # raise

    # # center_indices = np.where(~is_unused)[0]
    # # # array(center_indices)
    # # centers = data.take(center_indices, axis=0)
    # # print('len(center_indices) = %r' % len(center_indices))
    # # print('len(set(center_indices)) = %r' % len(set(center_indices)))
    # return centers


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
        ut.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
        if not clip_centroids:
            raise
    try:
        assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
    except Exception as ex:
        ut.printex(ex, keys=dbgkeys)
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
        ut.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
        if not clip_centroids:
            raise ex
        else:
            nCentroids = data.shape[0]
    print('+--- START CACHED AKMEANS')
    # filename prefix constants
    if cache_dir == 'default':
        print('[akmeans] using default cache dir')
        cache_dir = ut.get_app_resource_dir(appname)
        ut.ensuredir(cache_dir)
    # Build a cfgstr if the full one is not specified
    akmeans_cfgstr = get_akmeans_cfgstr(data, nCentroids, max_iters,
                                        initmethod, flann_params,
                                        use_data_hash, cfgstr, akmeans_cfgstr) + initmethod
    try:
        # Try and load a previous centroiding
        if not use_cache or force_recomp:
            raise UserWarning('forceing recommpute')
        centroids = ut.load_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr)
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
            ut.printex(ex, keys=dbgkeys, iswarning=clip_centroids)
            if not clip_centroids:
                raise
        try:
            assert centroids.shape[1] == data.shape[1], 'bad dimensionality'
        except Exception as ex:
            ut.printex(ex, keys=dbgkeys)
            raise
        print('L___ END CACHED AKMEANS')
        return centroids
    except IOError as ex:
        ut.printex(ex, 'cache miss', iswarning=True)
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
    #    with ut.Timer('testing time of 1 kmeans iteration') as timer:
    #        centroids = flann.kmeans(data, nCentroids, max_iterations=1)
    #    estimated_time = max_iters * timer.ellapsed
    #    print('Current time:            ' + ut.get_timestamp('printable'))
    #    print('Estimated Total Time:    ' + ut.get_unix_timedelta_str(estimated_time))
    #    print('Estimated finish time:   ' + ut.get_timestamp('printable', delta_seconds=estimated_time))
    #    print('Begining computation...')
    #    centroids = flann.kmeans(data, nCentroids, max_iterations=max_iters)
    #    print('The true finish time is: ' + ut.get_timestamp('printable'))
    #else:
    centroids = akmeans(data, nCentroids, max_iters, initmethod, flann_params)
    assert_centroids(centroids, data, nCentroids, clip_centroids)
    print('[akmeans.precompute] save and return')
    ut.save_cache(cache_dir, CLUSTERS_FNAME, akmeans_cfgstr, centroids)
    print('L___ END CACHED AKMEANS')
    return centroids


def approximate_distances(centroids, data, K, flann_params):
    import pyflann
    (_, qdist2_sdist) = pyflann.FLANN().nn(centroids, data, K, **flann_params)
    return qdist2_sdist


def approximate_assignments(seachedvecs, queryvecs, K, flann_params):
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


def double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy=False):
    """
    Takes corresponding lists as input and builds a double mapping.

    DEPRICATE

    Args:
        inner_key_list (list): each value_i is a scalar key.
        outer_keys_list (list): each value_i list of scalar keys
        items_list (list): value_i is a list that corresponds to outer_keys_i

    Returns:
        ut.ddict of dicts: outerkey2_innerkey2_items

    Examples:
        >>> from vtool.clustering2 import *  # NOQA
        >>> inner_key_list = [100, 200, 300, 400]
        >>> outer_keys_list = [[10, 20, 20], [30], [30, 10], [20]]
        >>> items_list = [[1, 2, 3], [4], [5, 6], [7]]
        >>> ensure_numpy = True
        >>> outerkey2_innerkey2_items = double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
        >>> print(ut.repr2(outerkey2_innerkey2_items))
        {
            10: {300: array([6]), 100: array([1])},
            20: {400: array([7]), 100: array([2, 3])},
            30: {200: array([4]), 300: array([5])},
        }

        >>> from vtool.clustering2 import *  # NOQA
        >>> len_ = 3000
        >>> incrementer = ut.make_incrementer()
        >>> nOuterList = [np.random.randint(300) for _ in range(len_)]
        >>> # Define big double_group input
        >>> inner_key_list = np.random.randint(100, size=len_) * 1000 + 1000
        >>> outer_keys_list = [np.random.randint(100, size=nOuter_) for nOuter_ in nOuterList]
        >>> items_list = [np.array([incrementer() for _ in range(nOuter_)]) for nOuter_ in nOuterList]
        >>> ensure_numpy = False
        >>> outerkey2_innerkey2_items = double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
        >>> print(ut.repr2(outerkey2_innerkey2_items))
        >>> print(ut.repr2(outerkey2_innerkey2_items[0]))

    Timeit:
        %timeit double_group(inner_key_list, outer_keys_list, items_list, ensure_numpy)
    """
    if ensure_numpy:
        inner_key_list = np.array(inner_key_list)
        outer_keys_list = np.array(map(np.array, outer_keys_list))
        items_list = np.array(map(np.array, items_list))
    outerkey2_innerkey2_items = ut.ddict(dict)
    _iter =  zip(inner_key_list, outer_keys_list, items_list)
    for inner_key, outer_keys, items in _iter:
        group_outerkeys, groupxs = group_indices(outer_keys)
        subitem_iter = (items.take(xs, axis=0) for xs in groupxs)
        for outer_key, subitems in zip(group_outerkeys, subitem_iter):
            outerkey2_innerkey2_items[outer_key][inner_key] = subitems
    return outerkey2_innerkey2_items
    #daid2_wx2_drvecs = ut.ddict(lambda: ut.ddict(list))
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


