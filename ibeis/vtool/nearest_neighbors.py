"""
Wrapper around flann (with caching)

python -c "import vtool, doctest; print(doctest.testmod(vtool.nearest_neighbors))"
"""
from __future__ import absolute_import, division, print_function
from os.path import exists, normpath, join
import pyflann
import utool
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[nneighbs]', DEBUG=False)


def ann_flann_once(dpts, qpts, num_neighbors, flann_params={}):
    """
    Finds the approximate nearest neighbors of qpts in dpts
    >>> from vtool.nearest_neighbors import *
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> dpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
    >>> qpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
    >>> qx2_dx, qx2_dist = ann_flann_once(dpts, qpts, 2)
    >>> print(utool.hashstr(repr((qx2_dx, qx2_dist))))
    8zdwd&q0mu+ez4gp
    """
    # qx2_dx   = query_index -> nearest database index
    # qx2_dist = query_index -> distance
    (qx2_dx, qx2_dist) = pyflann.FLANN().nn(dpts, qpts, num_neighbors, **flann_params)
    return (qx2_dx, qx2_dist)


def get_flann_cfgstr(dpts, flann_params, cfgstr='', use_data_hash=True):
    flann_valsig_ = str(list(flann_params.values()))
    flann_valsig = utool.remove_chars(flann_valsig_, ', \'[]')
    flann_cfgstr = '_FLANN(' + flann_valsig  + ')'
    # Generate a unique filename for dpts and flann parameters
    if use_data_hash:
        data_hashstr = utool.hashstr_arr(dpts, '_dID')  # flann is dependent on the dpts
        flann_cfgstr += data_hashstr
    return flann_cfgstr


#@utool.indent_func
def get_flann_fpath(dpts, cache_dir=None, cfgstr='', flann_params={}):
    """
    >>> from vtool.nearest_neighbors import *
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> dpts = np.random.randint(0, 255, (10, 128)).astype(np.uint8)
    >>> cache_dir = '.'
    >>> cfgstr = '_FEAT(alg=heshes)'
    >>> flann_params = get_kdtree_flann_params()
    >>> get_flann_fpath(dpts, cache_dir, cfgstr, flann_params)
    8zdwd&q0mu+ez4gp
    """
    #cache_dir = '.' if cache_dir is None else cache_dir
    assert cache_dir is not None, 'no cache dir specified'
    flann_cfgstr = get_flann_cfgstr(dpts, flann_params, cfgstr)
    # Append any user labels
    flann_fname = 'flann_index_' + flann_cfgstr + '.flann'
    flann_fpath = normpath(join(cache_dir, flann_fname))
    return flann_fpath


#@utool.indent_func
def flann_cache(dpts, cache_dir=None, cfgstr='', flann_params=None,
                use_cache=True, save=True):
    """
    Tries to load a cached flann index before doing anything
    from vtool.nn
    """
    if utool.NOT_QUIET:
        print('...flann_cache cfgstr = %r: ' % cfgstr)
    if len(dpts) == 0:
        raise AssertionError('cannot build flann when len(dpts) == 0. (prevents a segfault)')
    flann_fpath = get_flann_fpath(dpts, cache_dir, cfgstr, flann_params)
    # Load the index if it exists
    flann = pyflann.FLANN()
    if use_cache and exists(flann_fpath):
        try:
            flann.load_index(flann_fpath, dpts)
            if utool.NOT_QUIET:
                print('...flann cache hit')
            return flann
        except Exception as ex:
            utool.printex(ex, '... cannot load index', iswarning=True)
    # Rebuild the index otherwise
    print('...flann cache miss.')
    print('...building kdtree over %d points (this may take a sec).' % len(dpts))
    flann.build_index(dpts, **flann_params)
    print('flann.save_index(%r)' % utool.path_ndir_split(flann_fpath, n=2))
    if save:
        flann.save_index(flann_fpath)
    return flann


def flann_augment(dpts, new_dpts, cache_dir, cfgstr, new_cfgstr, flann_params,
                  use_cache=True, save=True):
    flann = flann_cache(dpts, cache_dir, cfgstr, flann_params)
    flann.add_points(new_dpts)
    if save:
        aug_dpts = np.vstack((dpts, new_dpts))
        new_flann_fpath = get_flann_fpath(aug_dpts, cache_dir, new_cfgstr, flann_params)
        flann.save_index(new_flann_fpath)
    return flann


def get_kdtree_flann_params():
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 4
    }
    return flann_params


def tune_flann(dpts, **kwargs):
    flann = pyflann.FLANN()
    #num_data = len(dpts)
    flann_atkwargs = dict(algorithm='autotuned',
                          target_precision=.01,
                          build_weight=0.01,
                          memory_weight=0.0,
                          sample_fraction=0.001)
    flann_atkwargs.update(kwargs)
    suffix = repr(flann_atkwargs)
    badchar_list = ',{}\': '
    for badchar in badchar_list:
        suffix = suffix.replace(badchar, '')
    print(flann_atkwargs)
    tuned_params = flann.build_index(dpts, **flann_atkwargs)
    utool.myprint(tuned_params)
    out_file = 'flann_tuned' + suffix
    utool.write_to(out_file, repr(tuned_params))
    flann.delete_index()
    return tuned_params
