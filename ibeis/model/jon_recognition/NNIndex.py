from __future__ import division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[nnindex]', DEBUG=False)
# Standard
from itertools import izip, chain, imap
import utool
from ibeis.dev import params
import numpy as np
from os.path import join, normpath, split
import pyflann


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


def get_flann_fpath(data, cache_dir, uid='', flann_params=None):
    cache_dir = '.' if cache_dir is None else cache_dir
    # Generate a unique filename for data and flann parameters
    fparams_uid = utool.remove_chars(str(flann_params.values()), ', \'[]')
    data_uid = utool.hashstr_arr(data, 'dID')  # flann is dependent on the data
    flann_suffix = '_' + fparams_uid + '_' + data_uid + '.flann'
    # Append any user labels
    flann_fname = 'flann_index_' + uid + flann_suffix
    flann_fpath = normpath(join(cache_dir, flann_fname))
    return flann_fpath


#@profile
def precompute_flann(data, cache_dir=None, uid='', flann_params=None,
                     force_recompute=False):
    ''' Tries to load a cached flann index before doing anything'''
    print('[flann] precompute_flann(%r): ' % uid)
    # Load the index if it exists
    flann_fpath = get_flann_fpath(data, cache_dir, uid, flann_params)
    flann = pyflann.FLANN()
    load_success = False
    if utool.checkpath(flann_fpath) and not force_recompute:
        try:
            #print('[flann] precompute_flann():
                #trying to load: %r ' % flann_fname)
            flann.load_index(flann_fpath, data)
            print('[flann]...flann cache hit')
            load_success = True
        except Exception as ex:
            print('[flann] precompute_flann(): ...cannot load index')
            print('[flann] precompute_flann(): ...caught ex=\n%r' % (ex,))
    if not load_success:
        # Rebuild the index otherwise
        with utool.Timer(msg='compute FLANN', newline=False):
            flann.build_index(data, **flann_params)
        print('[flann] precompute_flann(): save_index(%r)' % split(flann_fpath)[1])
        flann.save_index(flann_fpath)
    return flann


def get_flann_uid(ibs, cid_list):
    feat_uid   = ibs.prefs.feat_cfg.get_uid()
    sample_uid = utool.hashstr_arr(cid_list, 'dcxs')
    uid = '_' + sample_uid + feat_uid
    return uid


@profile
def build_flann_inverted_index(ibs, cid_list, return_info=False):
    cid2_desc  = ibs.feats.cid2_desc
    assert max(cid_list) < len(cid2_desc)
    uid = get_flann_uid(ibs, cid_list)
    # Make unique id for indexed descriptors
    # Number of features per sample chip
    nFeat_iter1 = imap(lambda cid: len(cid2_desc[cid]), iter(cid_list))
    nFeat_iter2 = imap(lambda cid: len(cid2_desc[cid]), iter(cid_list))
    nFeat_iter3 = imap(lambda cid: len(cid2_desc[cid]), iter(cid_list))
    # Inverted index from indexed descriptor to chipx and featx
    _ax2_cx = ([cid] * nFeat for (cid, nFeat) in izip(cid_list, nFeat_iter1))
    _ax2_fx = (xrange(nFeat) for nFeat in iter(nFeat_iter2))
    ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    # Aggregate indexed descriptors into continuous structure
    try:
        # sanatize cid_list
        cid_list = [cid for cid, nFeat in izip(iter(cid_list), nFeat_iter3) if nFeat > 0]
        if isinstance(cid2_desc, list):
            ax2_desc = np.vstack((cid2_desc[cid] for cid in cid_list))
        elif isinstance(cid2_desc, np.ndarray):
            ax2_desc = np.vstack(cid2_desc[cid_list])
    except MemoryError as ex:
        with utool.Indenter('[mem error]'):
            print(ex)
            print('len(cid_list) = %r' % (len(cid_list),))
            print('len(cid_list) = %r' % (len(cid_list),))
        raise
    except Exception as ex:
        with utool.Indenter('[unknown error]'):
            print(ex)
            print('cid_list = %r' % (cid_list,))
        raise
    # Build/Load the flann index
    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    precomp_kwargs = {'cache_dir': ibs.dirs.cache_dir,
                      'uid': uid,
                      'flann_params': flann_params,
                      'force_recompute': params.args.nocache_flann}
    flann = precompute_flann(ax2_desc, **precomp_kwargs)
    if not return_info:
        return ax2_cx, ax2_fx, ax2_desc, flann
    else:
        return ax2_cx, ax2_fx, ax2_desc, flann, precomp_kwargs


class NNIndex(object):
    'Nearest Neighbor (FLANN) Index Class'
    def __init__(nn_index, ibs, cid_list):
        print('[ds] building NNIndex object')
        ax2_cx, ax2_fx, ax2_desc, flann = build_flann_inverted_index(ibs, cid_list)
        #----
        # Agg Data
        nn_index.ax2_cx   = ax2_cx
        nn_index.ax2_fx   = ax2_fx
        nn_index.ax2_data = ax2_desc
        nn_index.flann = flann

    def __getstate__(nn_index):
        printDBG('get state NNIndex')
        #if 'flann' in nn_index.__dict__ and nn_index.flann is not None:
            #nn_index.flann.delete_index()
            #nn_index.flann = None
        # This class is not pickleable
        return None

    def __del__(nn_index):
        printDBG('deleting NNIndex')
        if 'flann' in nn_index.__dict__ and nn_index.flann is not None:
            nn_index.flann.delete_index()
            nn_index.flann = None
