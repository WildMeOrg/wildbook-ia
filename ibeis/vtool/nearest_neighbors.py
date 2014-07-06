from __future__ import absolute_import, division, print_function
from os.path import exists, normpath, join, split
import pyflann
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[nneighbs]', DEBUG=False)


def ann_flann_once(dpts, qpts, num_neighbors, flann_params={}):
    """
    Finds the approximate nearest neighbors of qpts in dpts
    """
    flann = pyflann.FLANN()
    flann.build_index(dpts, **flann_params)
    checks = flann_params.get('checks', 1024)
    # qx2_dx   = query_index -> nearest database index
    # qx2_dist = query_index -> distance
    (qx2_dx, qx2_dist) = flann.nn_index(qpts, num_neighbors, checks=checks)
    return (qx2_dx, qx2_dist)


#@utool.indent_func
def get_flann_fpath(data, cache_dir=None, cfgstr='', flann_params={}):
    #cache_dir = '.' if cache_dir is None else cache_dir
    assert cache_dir is not None, 'no cache dir specified'
    flann_cfgstr = get_flann_cfgstr(data, flann_params, cfgstr)
    # Append any user labels
    flann_fname = 'flann_index_' + flann_cfgstr + '.flann'
    flann_fpath = normpath(join(cache_dir, flann_fname))
    return flann_fpath


def get_flann_cfgstr(data, flann_params, cfgstr='', use_data_hash=True):
    flann_cfgstr = '_FLANN(' + utool.remove_chars(str(flann_params.values()), ', \'[]') + ')'
    # Generate a unique filename for data and flann parameters
    if use_data_hash:
        data_hashstr = utool.hashstr_arr(data, '_dID')  # flann is dependent on the data
        flann_cfgstr += data_hashstr
    return flann_cfgstr


#@utool.indent_func
def flann_cache(data, cache_dir=None, cfgstr='', flann_params=None,
                force_recompute=False):
    """ Tries to load a cached flann index before doing anything """
    print('...flann_cache cfgstr = %r: ' % cfgstr)
    flann_fpath = get_flann_fpath(data, cache_dir, cfgstr, flann_params)
    flann = pyflann.FLANN()
    load_success = False
    if len(data) == 0:
        raise AssertionError('cannot build flann with 0 datapoints. (there would be a segfault')
    # Load the index if it exists
    if exists(flann_fpath) and not force_recompute:
        try:
            flann.load_index(flann_fpath, data)
            print('...flann cache hit')
            load_success = True
        except Exception as ex:
            print('...cannot load index')
            print('...caught ex=\n%r' % (ex,))
    # Rebuild the index otherwise
    if not load_success:
        print('...flann cache miss.')
        print('...building kdtree over %d points (this may take a sec).' % len(data))
        flann.build_index(data, **flann_params)
        print('flann.save_index(%r)' % split(flann_fpath)[1])
        flann.save_index(flann_fpath)
    return flann


def tune_flann(data, **kwargs):
    flann = pyflann.FLANN()
    #num_data = len(data)
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
    tuned_params = flann.build_index(data, **flann_atkwargs)
    utool.myprint(tuned_params)
    out_file = 'flann_tuned' + suffix
    utool.write_to(out_file, repr(tuned_params))
    flann.delete_index()
    return tuned_params

"""
#def __tune():
    #tune_flann(sample_fraction=.03, target_precision=.9, build_weight=.01)
    #tune_flann(sample_fraction=.03, target_precision=.8, build_weight=.5)
    #tune_flann(sample_fraction=.03, target_precision=.8, build_weight=.9)
    #tune_flann(sample_fraction=.03, target_precision=.98, build_weight=.5)
    #tune_flann(sample_fraction=.03, target_precision=.95, build_weight=.01)
    #tune_flann(sample_fraction=.03, target_precision=.98, build_weight=.9)

    #tune_flann(sample_fraction=.3, target_precision=.9, build_weight=.01)
    #tune_flann(sample_fraction=.3, target_precision=.8, build_weight=.5)
    #tune_flann(sample_fraction=.3, target_precision=.8, build_weight=.9)
    #tune_flann(sample_fraction=.3, target_precision=.98, build_weight=.5)
    #tune_flann(sample_fraction=.3, target_precision=.95, build_weight=.01)
    #tune_flann(sample_fraction=.3, target_precision=.98, build_weight=.9)

    #tune_flann(sample_fraction=1, target_precision=.9, build_weight=.01)
    #tune_flann(sample_fraction=1, target_precision=.8, build_weight=.5)
    #tune_flann(sample_fraction=1, target_precision=.8, build_weight=.9)
    #tune_flann(sample_fraction=1, target_precision=.98, build_weight=.5)
    #tune_flann(sample_fraction=1, target_precision=.95, build_weight=.01)
    #tune_flann(sample_fraction=1, target_precision=.98, build_weight=.9)

# Look at /flann/algorithms/dist.h for distance clases

#distance_translation = {"euclidean"        : 1,
                        #"manhattan"        : 2,
                        #"minkowski"        : 3,
                        #"max_dist"         : 4,
                        #"hik"              : 5,
                        #"hellinger"        : 6,
                        #"chi_square"       : 7,
                        #"cs"               : 7,
                        #"kullback_leibler" : 8,
                        #"kl"               : 8,
                        #"hamming"          : 9,
                        #"hamming_lut"      : 10,
                        #"hamming_popcnt"   : 11,
                        #"l2_simple"        : 12,}

# MAKE SURE YOU EDIT index.py in pyflann

#flann_algos = {
    #'linear'        : 0,
    #'kdtree'        : 1,
    #'kmeans'        : 2,
    #'composite'     : 3,
    #'kdtree_single' : 4,
    #'hierarchical'  : 5,
    #'lsh'           : 6, # locality sensitive hashing
    #'kdtree_cuda'   : 7,
    #'saved'         : 254, # dont use
    #'autotuned'     : 255,
#}

#multikey_dists = {
    ## Huristic distances
    #('euclidian', 'l2')        :  1,
    #('manhattan', 'l1')        :  2,
    #('minkowski', 'lp')        :  3, # I guess p is the order?
    #('max_dist' , 'linf')      :  4,
    #('l2_simple')              : 12, # For low dimensional points
    #('hellinger')              :  6,
    ## Nonparametric test statistics
    #('hik','histintersect')    :  5,
    #('chi_square', 'cs')       :  7,
    ## Information-thoery divergences
    #('kullback_leibler', 'kl') :  8,
    #('hamming')                :  9, # xor and bitwise sum
    #('hamming_lut')            : 10, # xor (sums with lookup table;if nosse2)
    #('hamming_popcnt')         : 11, # population count (number of 1 bits)
#}


 #Hamming distance functor - counts the bit differences between two strings -
 #useful for the Brief descriptor
 #bit count of A exclusive XOR'ed with B

#flann_distances = {"euclidean"        : 1,
                   #"manhattan"        : 2,
                   #"minkowski"        : 3,
                   #"max_dist"         : 4,
                   #"hik"              : 5,
                   #"hellinger"        : 6,
                   #"chi_square"       : 7,
                   #"cs"               : 7,
                   #"kullback_leibler" : 8,
                   #"kl"               : 8 }

#pyflann.set_distance_type('hellinger', order=0)
"""
