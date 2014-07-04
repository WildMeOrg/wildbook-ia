from __future__ import absolute_import, division, print_function
import utool


def test_akmeans():
    import numpy as np
    from vtool import clustering
    nump = 10000
    dims = 128
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

    # Test regular computing
    dx2_label, centers = clustering.akmeans(data, num_clusters, max_iters=max_iters)

    assert centers.shape == (num_clusters, dims), 'sanity check'
    assert dx2_label.shape == (nump,), 'sanity check'

    # other test
    import pyflann
    flann_lib_inst = pyflann.flann
    flann_class_inst = pyflann.FLANN()
    flann_class_inst.build_index(data)
    return locals()


if __name__ == '__main__':
    test_locals = utool.run_test(test_akmeans)
    exec(utool.execstr_dict('test_locals'))
    exec(utool.ipython_execstr)
