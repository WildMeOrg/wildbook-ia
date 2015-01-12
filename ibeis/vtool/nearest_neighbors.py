"""
Wrapper around flann (with caching)

python -c "import vtool, doctest; print(doctest.testmod(vtool.nearest_neighbors))"
"""
from __future__ import absolute_import, division, print_function
from os.path import exists, normpath, join
import sys
import pyflann
import utool
import utool as ut  # NOQA
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[nneighbs]')


def ann_flann_once(dpts, qpts, num_neighbors, flann_params={}):
    """
    Finds the approximate nearest neighbors of qpts in dpts


    CommandLine:
        python -m vtool.nearest_neighbors --test-ann_flann_once

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> np.random.seed(1)
        >>> dpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
        >>> qpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
        >>> qx2_dx, qx2_dist = ann_flann_once(dpts, qpts, 2)
        >>> result = utool.hashstr(repr((qx2_dx, qx2_dist)))
        >>> print(result)
        8zdwd&q0mu+ez4gp

    Example2:
        >>> # ENABLE_DOCTEST
        >>> # Test upper bounds on sift descriptors
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
        >>> result = utool.hashstr(repr((qx2_dx, qx2_dist)))
        >>> print(result)
        8zdwd&q0mu+ez4gp

     Example:
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
    flann_valsig_ = str(list(flann_params.values()))
    flann_valsig = utool.remove_chars(flann_valsig_, ', \'[]')
    return flann_valsig


def get_flann_cfgstr(dpts, flann_params, cfgstr='', use_params_hash=True, use_data_hash=True):
    """

    CommandLine:
        python -m vtool.nearest_neighbors --test-get_flann_cfgstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> np.random.seed(1)
        >>> dpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
        >>> cache_dir = '.'
        >>> cfgstr = '_FEAT(alg=heshes)'
        >>> flann_params = get_kdtree_flann_params()
        >>> result = get_flann_cfgstr(dpts, flann_params, cfgstr)
        >>> print(result)
        _FEAT(alg=heshes)_FLANN(4kdtree)_DPTS((10,128)b+oqb%cnuo&oxk7h)
    """
    flann_cfgstr = cfgstr
    if use_params_hash:
        flann_valsig = get_flann_params_cfgstr(flann_params)
        flann_cfgstr += '_FLANN(' + flann_valsig + ')'
    # Generate a unique filename for dpts and flann parameters
    if use_data_hash:
        # flann is dependent on the dpts
        data_hashstr = utool.hashstr_arr(dpts, '_DPTS')
        flann_cfgstr += data_hashstr
    return flann_cfgstr


#@utool.indent_func
def get_flann_fpath(dpts, cache_dir='default', cfgstr='', flann_params={},
                    use_params_hash=True, use_data_hash=True, appname='vtool',
                    verbose=True):
    """ returns filepath for flann index """
    if cache_dir == 'default':
        print('[flann] using default cache dir')
        cache_dir = utool.get_app_resource_dir(appname)
        utool.ensuredir(cache_dir)
    flann_cfgstr = get_flann_cfgstr(dpts, flann_params, cfgstr,
                                    use_params_hash=use_params_hash,
                                    use_data_hash=use_data_hash)
    if verbose:
        print('...flann_cache cfgstr = %r: ' % flann_cfgstr)
    # Append any user labels
    flann_fname = 'flann_index' + flann_cfgstr + '.flann'
    flann_fpath = normpath(join(cache_dir, flann_fname))
    return flann_fpath


def build_flann_index(dpts, flann_params, quiet=False, verbose=True, flann=None):
    """ build flann with some verbosity """
    num_dpts = len(dpts)
    if flann is None:
        flann = pyflann.FLANN()
    if not quiet:
        print('...flann cache miss.')
    if verbose or (not quiet and num_dpts > 1E6):
        print('...building kdtree over %d points (this may take a sec).' % num_dpts)
    sys.stdout.flush()
    ut.util_logging.__UTOOL_FLUSH__()
    flann.build_index(dpts, **flann_params)
    return flann


#@utool.indent_func
def flann_cache(dpts, cache_dir='default', cfgstr='', flann_params={},
                use_cache=True, save=True, use_params_hash=True,
                use_data_hash=True, appname='vtool', verbose=utool.NOT_QUIET,
                quiet=utool.QUIET):
    """
    Tries to load a cached flann index before doing anything
    from vtool.nn
    """
    if verbose:
        print('+--- START CACHED FLANN INDEX ')
    if len(dpts) == 0:
        raise AssertionError(
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
            if not quiet:
                print('...flann cache hit')
            if verbose:
                print('L___ END FLANN INDEX ')
            return flann
        except Exception as ex:
            utool.printex(ex, '... cannot load index', iswarning=True)
    # Rebuild the index otherwise
    flann = build_flann_index(dpts, flann_params, verbose=verbose, quiet=quiet, flann=flann)
    if verbose:
        print('flann.save_index(%r)' % utool.path_ndir_split(flann_fpath, n=2))
    if save:
        flann.save_index(flann_fpath)
    if verbose:
        print('L___ END CACHED FLANN INDEX ')
    return flann


def flann_augment(dpts, new_dpts, cache_dir, cfgstr, new_cfgstr, flann_params,
                  use_cache=True, save=True):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.nearest_neighbors import *  # NOQA
        >>> import vtool.tests.dummy as dummy  # NOQA
        >>> dpts = dummy.get_dummy_dpts(utool.get_nth_prime(10))
        >>> new_dpts = dummy.get_dummy_dpts(utool.get_nth_prime(9))
        >>> cache_dir = utool.get_app_resource_dir('vtool')
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


def get_flann_params(type_='kdtree'):
    """
    References:
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
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
    flann_params = {
        'checks': 32,  # how many leafs (features) to check in one search
        'cb_index': 0.5,  # cluster boundary index for searching kdtrees
    }
    if type_ == 'kdtree':
        # kdtree index parameters
        flann_params.update({
            'algorithm': _algorithm_options[1],
            'trees': 4,
        })
        # Kmeans index parametrs
        flann_params.update({
            'breanching': 32,
            'iterations': 5,
            'centers_init': _centersinit_options[2],
        })
    elif type_ == 'autotuned':
        flann_params.update({
            'algorithm'        : 'autotuned',
            'target_precision' : .01,    # precision desired (used for autotuning, -1 otherwise)
            'build_weight'     : 0.01,   # build tree time weighting factor
            'memory_weight'    : 0.0,    # index memory weigthing factor
            'sample_fraction'  : 0.001,  # what fraction of the dataset to use for autotuning
        })
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
        print(utool.dict_str(flann_atkwargs))
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
        utool.write_to(out_file, ut.dict_str(tuned_params, sorted_=True, newlines=True))
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


def invertable_stack(vecs_list, label_list):
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
        >>> idx2_vec, idx2_label, idx2_fx = invertable_stack(vecs_list, label_list)
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
    _flatlabels = utool.iflatten(_ax2_label)
    _flatfeatxs = utool.iflatten(_ax2_fx)

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
#        _invertable_stack_cyth = vtool._nearest_neighbors_cyth._invertable_stack_cyth
#        invertable_stack_cyth = vtool._nearest_neighbors_cyth._invertable_stack_cyth
#        CYTHONIZED = True
#    except ImportError:
#        invertable_stack_cyth = invertable_stack
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
