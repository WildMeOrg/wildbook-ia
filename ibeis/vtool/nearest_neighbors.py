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
    >>> from vtool.nearest_neighbors import *  # NOQA
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
    >>> from vtool.nearest_neighbors import *  # NOQA
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
    """
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


def map_vecx_to_rowids(vecs_list, rowid_list):
    """
    Stacks descriptors into a flat structure and returns inverse mapping from
    flat database descriptor indexes (dx) to annotation ids (rowid) and feature
    indexes (fx). Feature indexes are w.r.t. annotation indexes.

    Output:
        dx2_desc - flat descriptor stack
        dx2_rowid  - inverted index into annotations
        dx2_fx   - inverted index into features

    # Example with 2D Descriptors
    >>> from vtool.nearest_neighbors import *  # NOQA
    >>> DESC_TYPE = np.uint8
    >>> rowid_list  = [1, 2, 3, 4, 5]
    >>> vecs_list = [
    ...     np.array([[0, 0], [0, 1]], dtype=DESC_TYPE),
    ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
    ...     np.empty((0, 2), dtype=DESC_TYPE),
    ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
    ...     np.array([[3, 3], [42, 42], [2, 6]], dtype=DESC_TYPE),
    ...     ]
    >>> dx2_vecs, dx2_rowid, dx2_fx = _build_inverted_vecsriptor_index(rowid_list, vecs_list)
    >>> print(repr(dx2_vecs.T))
    array([[ 0,  0,  5,  2,  1,  5,  2,  1,  3, 42,  2],
           [ 0,  1,  3, 30,  1,  3, 30,  1,  3, 42,  6]], dtype=uint8)
    >>> print(repr(dx2_rowid))
    array([1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 5])
    >>> print(repr(dx2_fx))
    array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    #if cyth:
        cdef:
            list rowid_list, vecs_list
            long nFeat, rowid
            iter rowid_nFeat_iter, nFeat_iter, _ax2_rowid, _ax2_fx
            np.ndarray dx2_rowid, dx2_fx, dx2_vecs
    #endif

    --- vs ---

    <CYTH>
    cdef:
        list rowid_list, vecs_list
        long nFeat, rowid
        iter rowid_nFeat_iter, nFeat_iter, _ax2_rowid, _ax2_fx
        np.ndarray dx2_rowid, dx2_fx, dx2_vecs
    </CYTH>

    --- consider ---
    SYNTAX:

    {pyth_code} <- python code | pass
    {cyth_code} <- cython code | pass

    {code_block} = {pyth_code} | {cyth_code}

    {block} := {code_block} | {parse_block}
    {prev_block} := {block}
    {next_block} := {block}
    {cyth_block} := {cyth_code}
    {pyth_block} := {pyth_code}

    {parse_block} :=
        {prev_block}
        #-ifdef CYTH
        {cyth_block}
        #-elif
        {pyth_block}
        #-endif
        {next_block}
    ---

    * Cyth becomes a python preparser. It converts annotated python code into
        cython code. It also provides a packages to semi-dynamically replace
        your python code with the generated and compiled cython code.

        "setup.py script not included"  // Check out utool for that
    """

    # INFER DTYPE? dtype = vecs_list[0].dtype
    # Build inverted index of (rowid, fx) pairs
    nFeats = sum(list(map(len, vecs_list)))
    nFeat_iter = map(len, vecs_list)
    rowid_nFeat_iter = zip(rowid_list, map(len, vecs_list))
    # generate featx inverted index for each feature in each annotation
    _ax2_fx  = [list(xrange(nFeat)) for nFeat in nFeat_iter]
    # generate rowid inverted index for each feature in each annotation
    '''
    # this is not a real test the code just happened to be here. syntax is good though
    #-ifdef CYTH_TEST_SWAP
    _ax2_rowid = [[rowid] * nFeat for (rowid, nFeat) in rowid_nFeat_iter]
    #-else
    '''
    _ax2_rowid = [[rowid] * nFeat for (rowid, nFeat) in rowid_nFeat_iter]
    '#-endif'  # endif is optional. the end of the functionscope counts as an #endif
    # Flatten generators into the inverted index
    _flatrowids = utool.iflatten(_ax2_rowid)
    _flatfeatxs = utool.iflatten(_ax2_fx)

    dtype = np.uint8
    count = nFeats
    dx2_rowid = np.fromiter(_flatrowids, dtype, count)
    dx2_fx  = np.fromiter(_flatfeatxs, dtype, count)
    # Stack vecsriptors into numpy array corresponding to inverted inexed
    # This might throw a MemoryError
    dx2_vecs = np.vstack(vecs_list)
    '#pragma cyth_returntup'
    return dx2_vecs, dx2_rowid, dx2_fx
