from __future__ import absolute_import, division, print_function
import utool


def test_akmeans(full_test=False, plot_test=False, num_pca_dims=2, data_dim=2,
                 nump=1000):
    import numpy as np
    from vtool import clustering
    nump = nump
    dims = data_dim  # 128
    dtype = np.uint8
    print('Make %d random %d-dimensional %s points.' % (nump, dims, dtype))
    # Seed for a determenistic test
    np.random.seed(42)
    data = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)

    num_clusters = 10
    max_iters = 2
    ave_unchanged_thresh = 0
    ave_unchanged_iterwin = 10
    flann_params = {}

    cache_dir = utool.get_app_resource_dir('vtool', 'test_cache')
    utool.ensuredir(cache_dir)

    # Test precomputing
    dx2_label, centers = clustering.precompute_akmeans(data, num_clusters,
                                                       max_iters=max_iters,
                                                       cache_dir=cache_dir)
    # internal names
    datax2_clusterx, centroids = dx2_label, centers

    if plot_test:
        clustering.plot_clusters(data, datax2_clusterx, centroids, num_pca_dims=num_pca_dims)

    assert centers.shape == (num_clusters, dims), 'sanity check'
    assert dx2_label.shape == (nump,), 'sanity check'

    # Test regular computing
    if full_test:
        dx2_label, centers = clustering.akmeans(data, num_clusters, max_iters=max_iters)
        assert centers.shape == (num_clusters, dims), 'sanity check'
        assert dx2_label.shape == (nump,), 'sanity check'

    if False:
        # other test (development)
        import pyflann
        flann_lib_inst = pyflann.flann
        flann_class_inst = pyflann.FLANN()
        flann_class_inst.build_index(data)
    return locals()


if __name__ == '__main__':
    testkw = {
        'plot_test': utool.get_argflag('--plot-test'),
        'full_test': utool.get_argflag('--full-test'),
        'num_pca_dims': utool.get_argval('--num-pca-dims', type_=int, default=2),
        'data_dim': utool.get_argval('--data-dim', type_=int, default=2),
        'nump': utool.get_argval('--nump', type_=int, default=2000),
    }
    test_locals = utool.run_test(test_akmeans, **testkw)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    if testkw['plot_test']:
        from plottool import draw_func2 as df2
        exec(df2.present())
    else:
        exec(utool.ipython_execstr())
