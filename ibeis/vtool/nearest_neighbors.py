"""
Wrapper around flann (with caching)

python -c "import vtool, doctest; print(doctest.testmod(vtool.nearest_neighbors))"
"""
from __future__ import absolute_import, division, print_function
from os.path import exists, normpath, join
import sys
import utool as ut
import numpy as np
(print, rrr, profile) = ut.inject2(__name__)

try:
    import pyflann
except ImportError:
    print('Warning: pyflann failed to import')


class AnnoyWrapper(object):
    """
    Wrapper for annoy to use the FLANN api
    """
    def __init__(self):
        self.ann = None
        self.params = {
            'trees': 8,
            'checks': 512,
        }

    def build_index(self, dvecs, **kwargs):
        import annoy
        self.params.update(kwargs)
        self.ann = annoy.AnnoyIndex(f=dvecs.shape[1], metric='euclidean')
        for i, dvec in enumerate(dvecs):
            ann.add_item(i, dvec)
        ann.build(n_trees=self.params['trees'])

    def nn_index(self, qvecs, num_neighbs, checks=None):
        if checks is None:
            checks = self.params['checks']
        idxs = np.empty((len(qvecs), num_neighbs), dtype=np.int)
        dists = np.empty((len(qvecs), num_neighbs), dtype=np.float)
        for i, qvec in enumerate(qvecs):
            idxs[i], dists[i] = ann.get_nns_by_vector(
                qvec, n=num_neighbs, search_k=checks, include_distances=True)
        return idxs, dists


def test_annoy():
    from vtool.tests import dummy
    import pyflann
    import annoy
    import utool
    qvecs = dummy.testdata_dummy_sift(2 * 1000)
    dvecs = dummy.testdata_dummy_sift(100 * 1000)
    dim = dpts.shape[1]

    checks = 200
    num_neighbs = 10
    num_trees = 8

    trials = 10

    for timer in utool.Timerit(trials, label='build annoy'):
        with timer:
            ann = annoy.AnnoyIndex(dim, metric='euclidean')
            for i, vec in enumerate(dvecs):
                ann.add_item(i, vec)
            ann.build(n_trees=num_trees)

    for timer in utool.Timerit(trials, label='annoy query'):
        with timer:
            for qvec in qvecs:
                ann.get_nns_by_vector(qvec, n=num_neighbs, search_k=checks,
                                      include_distances=True)

    # ---------------

    for timer in utool.Timerit(trials, label='build flann'):
        with timer:
            flann = pyflann.FLANN()
            flann.build_index(dvecs, algorithm='kdtree', trees=num_trees,
                              checks=checks, cores=1)

    for timer in utool.Timerit(trials, label='flann query'):
        with timer:
            flann.nn_index(qvecs, num_neighbs, checks=checks)

    # ---------------

    for timer in utool.Timerit(trials, label='build annoy wrapper'):
        with timer:
            index = AnnoyWrapper()
            index.build_index(dvecs, trees=num_trees, checks=checks)

    for timer in utool.Timerit(trials, label='query annoy wrapper'):
        with timer:
            index.nn_index(qvecs, num_neighbs, checks=checks)





def test_cv2_flann():
    """
    Ignore:
        [name for name in dir(cv2) if 'create' in name.lower()]
        [name for name in dir(cv2) if 'stereo' in name.lower()]

        ut.grab_zipped_url('https://priithon.googlecode.com/archive/a6117f5e81ec00abcfb037f0f9da2937bb2ea47f.tar.gz', download_dir='.')
    """
    import cv2
    from vtool.tests import dummy
    import plottool as pt
    import vtool as vt
    img1 = vt.imread(ut.grab_test_imgpath('easy1.png'))
    img2 = vt.imread(ut.grab_test_imgpath('easy2.png'))

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1, img2)
    pt.imshow(disparity)
    pt.show()

    #cv2.estima

    flow = cv2.createOptFlow_DualTVL1()
    img1, img2 = vt.convert_image_list_colorspace([img1, img2], 'gray', src_colorspace='bgr')
    img2 = vt.resize(img2, img1.shape[0:2][::-1])
    out = img1.copy()
    flow.calc(img1, img2, out)

    orb = cv2.ORB_create()
    kp1, vecs1 = orb.detectAndCompute(img1, None)
    kp2, vecs2 = orb.detectAndCompute(img2, None)

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img1)
    skp, sd = descriptor.compute(img1, skp)

    tkp = detector.detect(img2)
    tkp, td = descriptor.compute(img2, tkp)

    out = img1.copy()
    cv2.drawKeypoints(img1, kp1, outImage=out)
    pt.imshow(out)

    vecs1 = dummy.testdata_dummy_sift(10)
    vecs2 = dummy.testdata_dummy_sift(10)  # NOQA

    FLANN_INDEX_KDTREE = 0  # bug: flann enums are missing
    #flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # NOQA

    cv2.flann.Index(vecs1, index_params)

    #cv2.FlannBasedMatcher(flann_params)

    cv2.flann.Index(vecs1, flann_params)  # NOQA

    #def match_flann(desc1, desc2, r_threshold = 0.6):
    #    flann = cv2.flann_Index(desc2, flann_params)
    #    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    #    mask = dist[:,0] / dist[:,1] < r_threshold
    #    idx1 = np.arange(len(desc1))
    #    pairs = np.int32( zip(idx1, idx2[:,0]) )
    #    return pairs[mask]


def ann_flann_once(dpts, qpts, num_neighbors, flann_params={}):
    """
    Finds the approximate nearest neighbors of qpts in dpts


    CommandLine:
        python -m vtool.nearest_neighbors --test-ann_flann_once:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> np.random.seed(1)
        >>> dpts = np.random.randint(0, 255, (5, 128)).astype(np.uint8)
        >>> qpts = np.random.randint(0, 255, (5, 128)).astype(np.uint8)
        >>> qx2_dx, qx2_dist = ann_flann_once(dpts, qpts, 2)
        >>> result = ut.list_str((qx2_dx.T, qx2_dist.T), precision=2)
        >>> print(result)
        (
            np.array([[3, 3, 3, 3, 0],
                      [2, 0, 1, 4, 4]], dtype=np.int32),
            np.array([[ 1037329.,  1235876.,  1168550.,  1286435.,  1075507.],
                      [ 1038324.,  1243690.,  1304896.,  1320598.,  1369036.]], dtype=np.float32),
        )

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Test upper bounds on sift descriptors
        >>> # SeeAlso distance.understanding_pseudomax_props
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> import vtool as vt
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> # get points on unit sphere
        >>> nDpts = 5000 # 5
        >>> nQpts = 10000 # 10
        >>> dpts = vt.normalize_rows(np.random.rand(nDpts, 128))
        >>> qpts = vt.normalize_rows(np.random.rand(nQpts, 128))
        >>> qmag = np.sqrt(np.power(qpts, 2).sum(1))
        >>> dmag = np.sqrt(np.power(dpts, 2).sum(1))
        >>> assert np.all(np.allclose(qmag, 1)), 'not on unit sphere'
        >>> assert np.all(np.allclose(dmag, 1)), 'not on unit sphere'
        >>> # cast to uint8
        >>> uint8_max = 512  # hack
        >>> uint8_min = 0  # hack
        >>> K = 100 # 2

        >>> qpts8 = np.clip(np.round(qpts * uint8_max), uint8_min, uint8_max).astype(np.uint8)
        >>> dpts8 = np.clip(np.round(dpts * uint8_max), uint8_min, uint8_max).astype(np.uint8)
        >>> qmag8 = np.sqrt(np.power(qpts8.astype(np.float32), 2).sum(1))
        >>> dmag8 = np.sqrt(np.power(dpts8.astype(np.float32), 2).sum(1))
        >>> # test
        >>> qx2_dx, qx2_dist = ann_flann_once(dpts8, qpts8, K)
        >>> biggest_dist = np.sqrt(qx2_dist.max())
        >>> print('biggest_dist = %r' % (biggest_dist))
        >>> # Get actual distance by hand
        >>> hand_dist = np.sum((qpts8 - dpts8[qx2_dx.T[0]]) ** 2, 0)
        >>> # Seems like flann returns squared distance. makes sense
        >>> result = utool.hashstr27(repr((qx2_dx, qx2_dist)))
        >>> print(result)
        8zdwd&q0mu+ez4gp

     Example2:
        >>> # Build theoretically maximally distant vectors
        >>> b = 512
        >>> D = 128
        >>> x = np.sqrt((float(b) ** 2) / float(D - 1))
        >>> dpts = np.ones((2, 128)) * x
        >>> qpts = np.zeros((2, 128))
        >>> dpts[:, 0] = 0
        >>> qpts[:, 0] = 512
        >>> qpts[:, 0::2] = 1
        >>> dpts[:, 1::2] = 1
        >>> qpts[:, 1::2] = 0
        >>> dpts[:, 0::2] = 0
        >>> qmag = np.sqrt(np.power(qpts.astype(np.float64), 2).sum(1))
        >>> dmag = np.sqrt(np.power(dpts.astype(np.float64), 2).sum(1))
        >>> # FIX TO ACTUALLY BE AT THE RIGHT NORM
        >>> dpts = dpts * (512 / np.linalg.norm(dpts, axis=1))[:, None]
        >>> qpts = qpts * (512 / np.linalg.norm(qpts, axis=1))[:, None]
        >>> print(np.linalg.norm(dpts))
        >>> print(np.linalg.norm(qpts))
        >>> dist = np.sqrt(np.sum((qpts - dpts) ** 2, 1))
        >>> # Because of norm condition another maximally disant pair of vectors
        >>> # is [1, 0, 0, ... 0] and [0, 1, .. 0, 0, 0]
        >>> # verifythat this gives you same dist.
        >>> dist2 = np.sqrt((512 ** 2 + 512 ** 2))
        >>> print(dist2)
        >>> print(dist)
    """
    # qx2_dx   = query_index -> nearest database index
    # qx2_dist = query_index -> distance
    (qx2_dx, qx2_dist) = pyflann.FLANN().nn(
        dpts, qpts, num_neighbors, **flann_params)
    return (qx2_dx, qx2_dist)


def assign_to_centroids(dpts, qpts, num_neighbors=1, flann_params={}):
    """ Helper for akmeans """
    (qx2_dx, qx2_dist) = pyflann.FLANN().nn(
        dpts, qpts, num_neighbors, **flann_params)
    return qx2_dx


def get_flann_params_cfgstr(flann_params):
    if True:
        # Ensure consistent ordering
        flann_vals = list(flann_params.values())
        flann_keys = list(flann_params.keys())
        # reverse to maintain backwards compatibility
        flann_valsig_ = str(ut.sortedby(flann_vals, flann_keys, reverse=True))
    else:
        flann_valsig_ = str(list(flann_params.values()))
    flann_valsig = ut.remove_chars(flann_valsig_, ', \'[]')
    return flann_valsig


def get_flann_cfgstr(dpts, flann_params, cfgstr='', use_params_hash=True,
                     use_data_hash=True):
    """

    CommandLine:
        python -m vtool.nearest_neighbors --test-get_flann_cfgstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> rng = np.random.RandomState(1)
        >>> dpts = rng.randint(0, 255, (10, 128)).astype(np.uint8)
        >>> cache_dir = '.'
        >>> cfgstr = '_FEAT(alg=heshes)'
        >>> flann_params = get_kdtree_flann_params()
        >>> result = get_flann_cfgstr(dpts, flann_params, cfgstr)
        >>> print(result)
        _FEAT(alg=heshes)_FLANN(4kdtree)_DPTS((10,128)xxaotseonmfjkzcr)
    """
    flann_cfgstr = cfgstr
    if use_params_hash:
        flann_valsig = get_flann_params_cfgstr(flann_params)
        flann_cfgstr += '_FLANN(' + flann_valsig + ')'
    # Generate a unique filename for dpts and flann parameters
    if use_data_hash:
        # flann is dependent on the dpts
        data_hashstr = ut.hashstr_arr27(dpts, '_DPTS')
        flann_cfgstr += data_hashstr
    return flann_cfgstr


def get_flann_fpath(dpts, cache_dir='default', cfgstr='', flann_params={},
                    use_params_hash=True, use_data_hash=True, appname='vtool',
                    verbose=True):
    """ returns filepath for flann index """
    if cache_dir == 'default':
        if verbose:
            print('[flann] using default cache dir')
        cache_dir = ut.get_app_resource_dir(appname)
        ut.ensuredir(cache_dir)
    flann_cfgstr = get_flann_cfgstr(dpts, flann_params, cfgstr,
                                    use_params_hash=use_params_hash,
                                    use_data_hash=use_data_hash)
    if verbose:
        print('...flann_cache cfgstr = %r: ' % flann_cfgstr)
    # Append any user labels
    flann_fname = 'flann_index' + flann_cfgstr + '.flann'
    flann_fpath = normpath(join(cache_dir, flann_fname))
    return flann_fpath


def flann_cache(dpts, cache_dir='default', cfgstr='', flann_params={},
                use_cache=True, save=True, use_params_hash=True,
                use_data_hash=True, appname='vtool', verbose=None):
    """
    Tries to load a cached flann index before doing anything
    from vtool.nn
    """
    if verbose is None:
        verbose = int(ut.NOT_QUIET)
    if verbose is True:
        verbose = 2
    if verbose > 1:
        print('+--- START CACHED FLANN INDEX ')
    if len(dpts) == 0:
        raise ValueError(
            'cannot build flann when len(dpts) == 0. (prevents a segfault)')
    flann_fpath = get_flann_fpath(dpts, cache_dir, cfgstr, flann_params,
                                  use_params_hash=use_params_hash,
                                  use_data_hash=use_data_hash, appname=appname,
                                  verbose=verbose)
    # Load the index if it exists
    flann = pyflann.FLANN()
    flann.flann_fpath = flann_fpath
    if use_cache and exists(flann_fpath):
        try:
            flann.load_index(flann_fpath, dpts)
            if verbose > 0:
                print('...flann cache hit: %d vectors' % (len(dpts)))
            if verbose > 1:
                print('L___ END FLANN INDEX ')
            return flann
        except Exception as ex:
            ut.printex(ex, '... cannot load index', iswarning=True)
    # Rebuild the index otherwise
    if verbose > 0:
        print('...flann cache miss.')
    num_dpts = len(dpts)
    if flann is None:
        flann = pyflann.FLANN()
    if verbose > 1 or (verbose > 0 and num_dpts > 1E6):
        print('...building kdtree over %d points (this may take a sec).' % num_dpts)
    if num_dpts == 0:
        print('WARNING: CANNOT BUILD FLANN INDEX OVER 0 POINTS. THIS MAY BE A SIGN OF A DEEPER ISSUE')
        return flann
    flann.build_index(dpts, **flann_params)
    if verbose > 1:
        print('flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=2))
    if save:
        flann.save_index(flann_fpath)
    if verbose > 1:
        print('L___ END CACHED FLANN INDEX ')
    return flann


def flann_augment(dpts, new_dpts, cache_dir, cfgstr, new_cfgstr, flann_params,
                  use_cache=True, save=True):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> import vtool.tests.dummy as dummy  # NOQA
        >>> dpts = dummy.get_dummy_dpts(ut.get_nth_prime(10))
        >>> new_dpts = dummy.get_dummy_dpts(ut.get_nth_prime(9))
        >>> cache_dir = ut.get_app_resource_dir('vtool')
        >>> cfgstr = '_testcfg'
        >>> new_cfgstr = '_new_testcfg'
        >>> flann_params = get_kdtree_flann_params()
        >>> use_cache = False
        >>> save = False
    """
    flann = flann_cache(dpts, cache_dir, cfgstr, flann_params)
    flann.add_points(new_dpts)
    if save:
        aug_dpts = np.vstack((dpts, new_dpts))
        new_flann_fpath = get_flann_fpath(
            aug_dpts, cache_dir, new_cfgstr, flann_params)
        flann.save_index(new_flann_fpath)
    return flann


def get_kdtree_flann_params():
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 4
    }
    return flann_params


def get_flann_params(algorithm='kdtree', **kwargs):
    """
    Returns flann params that are relvant tothe algorithm

    References:
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf

    Args:
        algorithm (str): (default = 'kdtree')

    Returns:
        dict: flann_params

    CommandLine:
        python -m vtool.nearest_neighbors --test-get_flann_params --algo=kdtree
        python -m vtool.nearest_neighbors --test-get_flann_params --algo=kmeans

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> algorithm = ut.get_argval('--algo', default='kdtree')
        >>> flann_params = get_flann_params(algorithm)
        >>> result = ('flann_params = %s' % (ut.dict_str(flann_params),))
        >>> print(result)
    """
    _algorithm_options = [
        'linear',
        'kdtree',
        'kmeans',
        'composite',
        'kdtree_single'
    ]
    _centersinit_options = [
        'random',
        'gonzales',
        'kmeanspp',
    ]
    # Search params (for all algos)
    assert algorithm in _algorithm_options
    flann_params = {
        'algorithm': algorithm
    }
    if algorithm != 'linear':
        flann_params.update({
            'random_seed': -1
        })
    if algorithm in ['kdtree', 'composite']:
        # kdtree index parameters
        flann_params.update({
            'algorithm': _algorithm_options[1],
            'trees': 4,
            'checks': 32,  # how many leafs (features) to check in one search
        })
    elif algorithm in ['kmeans', 'composite']:
        # Kmeans index parametrs
        flann_params.update({
            'branching': 32,
            'iterations': 5,
            'centers_init': _centersinit_options[2],
            'cb_index': 0.5,  # cluster boundary index for searching kmeanms tree
            'checks': 32,  # how many leafs (features) to check in one search
        })
    elif algorithm == 'autotuned':
        flann_params.update({
            'algorithm'        : 'autotuned',
            'target_precision' : .01,    # precision desired (used for autotuning, -1 otherwise)
            'build_weight'     : 0.01,   # build tree time weighting factor
            'memory_weight'    : 0.0,    # index memory weigthing factor
            'sample_fraction'  : 0.001,  # what fraction of the dataset to use for autotuning
        })
    elif algorithm == 'lsh':
        flann_params.update({
            'table_number_': 12,
            'key_size_': 20,
            'multi_probe_level_': 2,
        })

    flann_params = ut.update_existing(flann_params, kwargs, assert_exists=True)
    return flann_params


def tune_flann(dpts,
               target_precision=.90,
               build_weight=0.50,
               memory_weight=0.00,
               sample_fraction=0.01):
    r"""

    References:
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_pami2014.pdf
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
        http://docs.opencv.org/trunk/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html

    Math::
        cost of an algorithm is:

        LaTeX:
            \cost = \frac
                {\search + build_weight * \build }
                { \minoverparams( \search + build_weight \build)} +
                memory_weight * \memory

    Args:
        dpts (ndarray):

        target_precision (float): number between 0 and 1 representing desired
            accuracy. Higher values are more accurate.

        build_weight (float): importance weight given to minimizing build time
            relative to search time. This number can range from 0 to infinity.
            typically because building is a more complex computation you want
            to keep the number relatively low, (less than 1) otherwise you'll
            end up getting a linear search (no build time).

        memory_weight (float): Importance of memory relative to total speed.
            A value less than 1 gives more importance to the time spent and a
            value greater than 1 gives more importance to the memory usage.

        sample_fraction (float): number between 0 and 1 representing the
            fraction of the input data to use in the optimization. A higher
            number uses more data.

    Returns:
        dict: tuned_params

    CommandLine:
        python -m vtool.nearest_neighbors --test-tune_flann

    """
    with ut.Timer('tuning flann'):
        print('Autotuning flann with %d %dD vectors' % (dpts.shape[0], dpts.shape[1]))
        print('a sample of %d vectors will be used' % (int(dpts.shape[0] * sample_fraction)))
        flann = pyflann.FLANN()
        #num_data = len(dpts)
        flann_atkwargs = dict(algorithm='autotuned',
                              target_precision=target_precision,
                              build_weight=build_weight,
                              memory_weight=memory_weight,
                              sample_fraction=sample_fraction)
        suffix = repr(flann_atkwargs)
        badchar_list = ',{}\': '
        for badchar in badchar_list:
            suffix = suffix.replace(badchar, '')
        print('flann_atkwargs:')
        print(ut.dict_str(flann_atkwargs))
        print('starting optimization')
        tuned_params = flann.build_index(dpts, **flann_atkwargs)
        print('finished optimization')

        # The algorithm is sometimes returned as default which is
        # very unuseful as the default name is embeded in the pyflann
        # module where most would not care to look. This finds the default
        # name for you.
        for key in ['algorithm', 'centers_init', 'log_level']:
            val = tuned_params.get(key, None)
            if val == 'default':
                dict_ = pyflann.FLANNParameters._translation_[key]
                other_algs = ut.dict_find_other_sameval_keys(dict_, 'default')
                assert len(other_algs) == 1, 'more than 1 default for key=%r' % (key,)
                tuned_params[key] = other_algs[0]

        common_params = [
            'algorithm',
            'checks',
        ]
        relevant_params_dict = dict(
            linear=['algorithm'],
            #---
            kdtree=[
                'trees'
            ],
            #---
            kmeans=[
                'branching',
                'iterations',
                'centers_init',
                'cb_index',
            ],
            #---
            lsh=[
                'table_number',
                'key_size',
                'multi_probe_level',
            ],
        )
        relevant_params_dict['composite'] = relevant_params_dict['kmeans'] + relevant_params_dict['kdtree'] + common_params
        relevant_params_dict['kmeans'] += common_params
        relevant_params_dict['kdtree'] += common_params
        relevant_params_dict['lsh'] += common_params

        #kdtree_single_params = [
        #    'leaf_max_size',
        #]
        #other_params = [
        #    'build_weight',
        #    'sorted',
        #]
        out_file = 'flann_tuned' + suffix
        ut.write_to(out_file, ut.dict_str(tuned_params, sorted_=True, newlines=True))
        flann.delete_index()
        if tuned_params['algorithm'] in relevant_params_dict:
            print('relevant_params=')
            relevant_params = relevant_params_dict[tuned_params['algorithm']]
            print(ut.dict_str(ut.dict_subset(tuned_params, relevant_params),
                              sorted_=True, newlines=True))
            print('irrelevant_params=')
            print(ut.dict_str(ut.dict_setdiff(tuned_params, relevant_params),
                              sorted_=True, newlines=True))
        else:
            print('unknown tuned algorithm=%r' % (tuned_params['algorithm'],))

        print('all_tuned_params=')
        print(ut.dict_str(tuned_params, sorted_=True, newlines=True))
    return tuned_params


def flann_index_time_experiment():
    r"""

    Shows a plot of how long it takes to build a flann index for a given number of KD-trees

    CommandLine:
        python -m vtool.nearest_neighbors --test-flann_index_time_experiment

    Example:
        >>> # SLOW_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> result = flann_index_time_experiment()
        >>> print(result)
    """
    import vtool as vt
    import pyflann
    import itertools

    class TestDataPool(object):
        """
        Perform only a few allocations of test data
        """
        def __init__(self):
            self.num = 10000
            self.data_pool = None
            self.alloc_pool(1000000)

        def alloc_pool(self, num):
            print('[alloc] num = %r' % (num,))
            self.num = num
            self.data_pool = vt.tests.dummy.testdata_dummy_sift(num)
            print('[alloc] object size ' + ut.get_object_size_str(self.data_pool, 'data_pool'))

        def get_testdata(self, num):
            if len(self.data_pool) < num:
                self.alloc_pool(2 * self.num)
            return self.data_pool[0:num]

    pool = TestDataPool()

    def get_buildtime_data(**kwargs):
        flann_params = vt.get_flann_params(**kwargs)
        print('flann_params = %r' % (ut.dict_str(flann_params),))
        data_list = []
        num = 1000
        print('-----')
        for count in ut.ProgressIter(itertools.count(), nTotal=-1, freq=1, autoadjust=False):
            num = int(num * 1.2)
            print('num = %r' % (num,))
            #if num > 1E6:
            #    break
            data = pool.get_testdata(num)
            print('object size ' + ut.get_object_size_str(data, 'data'))
            flann = pyflann.FLANN(**flann_params)
            with ut.Timer(verbose=False) as t:
                flann.build_index(data)
            print('t.ellapsed = %r' % (t.ellapsed,))
            if t.ellapsed > 5 or count > 1000:
                break
            data_list.append((count, num, t.ellapsed))
            print('-----')
        return data_list, flann_params

    data_list1, params1 = get_buildtime_data(trees=1)

    data_list2, params2 = get_buildtime_data(trees=2)

    data_list4, params4 = get_buildtime_data(trees=4)

    data_list8, params8 = get_buildtime_data(trees=8)

    data_list16, params16 = get_buildtime_data(trees=16)

    import plottool as pt

    def plotdata(data_list):
        count_arr = ut.get_list_column(data_list, 1)
        time_arr  = ut.get_list_column(data_list, 2)
        pt.plot2(count_arr, time_arr, marker='-o', equal_aspect=False,
                 x_label='num_vectors', y_label='FLANN build time')

    plotdata(data_list1)
    plotdata(data_list2)
    plotdata(data_list4)
    plotdata(data_list8)
    plotdata(data_list16)

    pt.iup()


def invertible_stack(vecs_list, label_list):
    """
    Stacks descriptors into a flat structure and returns inverse mapping from
    flat database descriptor indexes (dx) to annotation ids (label) and feature
    indexes (fx). Feature indexes are w.r.t. annotation indexes.

    Output:
        idx2_desc - flat descriptor stack
        idx2_label  - inverted index into annotations
        idx2_fx   - inverted index into features

    # Example with 2D Descriptors

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> DESC_TYPE = np.uint8
        >>> label_list  = [1, 2, 3, 4, 5]
        >>> vecs_list = [
        ...     np.array([[0, 0], [0, 1]], dtype=DESC_TYPE),
        ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
        ...     np.empty((0, 2), dtype=DESC_TYPE),
        ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
        ...     np.array([[3, 3], [42, 42], [2, 6]], dtype=DESC_TYPE),
        ...     ]
        >>> idx2_vec, idx2_label, idx2_fx = invertible_stack(vecs_list, label_list)
        >>> print(repr(idx2_vec.T))
        array([[ 0,  0,  5,  2,  1,  5,  2,  1,  3, 42,  2],
               [ 0,  1,  3, 30,  1,  3, 30,  1,  3, 42,  6]], dtype=uint8)
        >>> print(repr(idx2_label))
        array([1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 5])
        >>> print(repr(idx2_fx))
        array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    """
    # INFER DTYPE? dtype = vecs_list[0].dtype
    # Build inverted index of (label, fx) pairs
    nFeats = sum(list(map(len, vecs_list)))
    nFeat_iter = map(len, vecs_list)
    label_nFeat_iter = zip(label_list, map(len, vecs_list))
    # generate featx inverted index for each feature in each annotation
    _ax2_fx = [list(range(nFeat)) for nFeat in nFeat_iter]
    # generate label inverted index for each feature in each annotation
    '''
    # this is not a real test the code just happened to be here. syntax is good though
    #-ifdef CYTH_TEST_SWAP
    _ax2_label = [[label] * nFeat for (label, nFeat) in label_nFeat_iter]
    #-else
    '''
    _ax2_label = [[label] * nFeat for (label, nFeat) in label_nFeat_iter]
    # endif is optional. the end of the functionscope counts as an #endif
    '#-endif'
    # Flatten generators into the inverted index
    _flatlabels = ut.iflatten(_ax2_label)
    _flatfeatxs = ut.iflatten(_ax2_fx)

    idx2_label = np.fromiter(_flatlabels, np.int32, nFeats)
    idx2_fx = np.fromiter(_flatfeatxs, np.int32, nFeats)
    # Stack vecsriptors into numpy array corresponding to inverted inexed
    # This might throw a MemoryError
    idx2_vec = np.vstack(vecs_list)
    '#pragma cyth_returntup'
    return idx2_vec, idx2_label, idx2_fx

#import cyth
#if cyth.DYNAMIC:
#    exec(cyth.import_cyth_execstr(__name__))
#else:
#    # <AUTOGEN_CYTH>
#    # Regen command: python -c "import vtool.nearest_neighbors" --cyth-write
#    try:
#        if not cyth.WITH_CYTH:
#            raise ImportError('no cyth')
#        import vtool._nearest_neighbors_cyth
#        _invertible_stack_cyth = vtool._nearest_neighbors_cyth._invertible_stack_cyth
#        invertible_stack_cyth = vtool._nearest_neighbors_cyth._invertible_stack_cyth
#        CYTHONIZED = True
#    except ImportError:
#        invertible_stack_cyth = invertible_stack
#        CYTHONIZED = False
#    # </AUTOGEN_CYTH>
#    pass

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.nearest_neighbors
        python -m vtool.nearest_neighbors --allexamples
        python -m vtool.nearest_neighbors --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
