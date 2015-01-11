"""
External mechanism for computing feature distinctiveness

stores some set of vectors which lose their association with
their parent.
"""
from __future__ import absolute_import, division, print_function
import utool
#from os.path import join
#import numpy as np
import utool as ut
import numpy as np
#import vtool as vt
import six  # NOQA
from ibeis import constants as const
import pyflann
from ibeis.model.hots import hstypes
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[distinctnorm]', DEBUG=False)


DISTINCTIVENESS_NORMALIZER_CACHE = {}


@six.add_metaclass(ut.ReloadingMetaclass)
class DistinctivnessNormalizer(ut.Cachable):
    ext    = '.cPkl'
    prefix = 'distinctivness'

    def __init__(dstcnvs_normer, species):
        """ cfgstring should be the species trained on """
        dstcnvs_normer.vecs = None
        dstcnvs_normer.species = species
        dstcnvs_normer.flann_params = {'algorithm': 'kdtree', 'trees': 8, 'checks': 800}
        dstcnvs_normer.checks = dstcnvs_normer.flann_params.get('checks')
        dstcnvs_normer.cores = dstcnvs_normer.flann_params.get('cores', 0)

    def get_prefix(dstcnvs_normer):
        return DistinctivnessNormalizer.prefix + '_'

    def get_cfgstr(dstcnvs_normer):
        assert dstcnvs_normer.species is not None
        cfgstr = dstcnvs_normer.species
        return cfgstr

    def get_flann_fpath(dstcnvs_normer, cachedir):
        flann_fpath = dstcnvs_normer.get_fpath(cachedir, ext='.flann')
        return flann_fpath

    def init_support(dstcnvs_normer, vecs, verbose=True):
        dstcnvs_normer.flann = pyflann.FLANN()  # Approximate search structure
        dstcnvs_normer.vecs = vecs
        num_vecs = len(vecs)
        notify_num = 1E6
        if verbose or (not ut.QUIET and num_vecs > notify_num):
            print('...building kdtree over %d points (this may take a sec).' % num_vecs)
        dstcnvs_normer.flann.build_index(dstcnvs_normer.vecs, **dstcnvs_normer.flann_params)
        if dstcnvs_normer.vecs.dtype == hstypes.VEC_TYPE:
            dstcnvs_normer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
            dstcnvs_normer.max_distance_sqrd = dstcnvs_normer.max_distance ** 2

    def add_support(dstcnvs_normer, new_vecs):
        """
        """
        pass

    def load(dstcnvs_normer, cachedir, verbose=True, *args, **kwargs):
        # Inherited method
        kwargs['ignore_keys'] = ['flann']
        super(DistinctivnessNormalizer, dstcnvs_normer).load(cachedir, *args, **kwargs)
        load_success = False
        # Load Flann
        flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
        if ut.checkpath(flann_fpath, verbose=ut.VERBOSE):
            try:
                dstcnvs_normer.flann = pyflann.FLANN()
                dstcnvs_normer.flann.load_index(flann_fpath, dstcnvs_normer.vecs)
                load_success = True
            except Exception as ex:
                ut.printex(ex, '... cannot load distinctiveness flann', iswarning=True)
        else:
            raise IOError('cannot load distinctiveness flann')
        return load_success
        if ut.VERBOSE:
            print('[nnindex] load_success = %r' % (load_success,))

    def save(dstcnvs_normer, cachedir, verbose=True, *args, **kwargs):
        """
        args = tuple()
        kwargs = {}
        """
        # Inherited method
        kwargs['ignore_keys'] = ['flann']
        # Save everything but flann
        super(DistinctivnessNormalizer, dstcnvs_normer).save(cachedir, *args, **kwargs)
        # Save flann
        if dstcnvs_normer.flann is not None:
            flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
            if verbose:
                print('flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
            dstcnvs_normer.flann.save_index(flann_fpath)

    def get_distinctiveness(dstcnvs_normer, qfx2_vec):
        K = 5
        assert K > 0 and K < len(dstcnvs_normer.vecs)
        if len(qfx2_vec) == 0:
            (qfx2_idx, qfx2_dist) = dstcnvs_normer.empty_neighbors(0, K)
        else:
            # perform nearest neighbors
            (qfx2_idx, qfx2_dist) = dstcnvs_normer.flann.nn_index(
                qfx2_vec, K, checks=dstcnvs_normer.checks, cores=dstcnvs_normer.cores)
            # Ensure that distance returned are between 0 and 1
            #qfx2_dist = qfx2_dist / (dstcnvs_normer.max_distance ** 2)
            qfx2_dist = np.divide(qfx2_dist, dstcnvs_normer.max_distance_sqrd)
            #qfx2_dist = np.sqrt(qfx2_dist) / dstcnvs_normer.max_distance
        norm_sqared_dist = qfx2_dist.T[-1].T
        qfx2_dstncvs = compute_distinctiveness_from_dist(norm_sqared_dist)
        return qfx2_dstncvs


def compute_distinctiveness_from_dist(norm_sqared_dist):
    """
    Compute distinctiveness from distance to K+1 nearest neighbor
    """
    # TODO: paramaterize
    # expondent to augment distinctiveness scores.
    p = 1.0
    # clip the distinctiveness at this fraction
    clip_fraction = .2
    wd_cliped = np.divide(norm_sqared_dist, clip_fraction)
    wd_cliped[np.greater(wd_cliped, 1.0)] = 1.0
    dstncvs = np.power(wd_cliped, p)
    return dstncvs


def request_ibeis_distinctiveness_normalizer(qreq_, verbose=True):
    r"""
    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-request_ibeis_distinctiveness_normalizer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qaids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(qaids, daids)
        >>> # execute function
        >>> result = request_ibeis_distinctiveness_normalizer(qreq_)
        >>> # verify results
        >>> print(result)
    """
    global DISTINCTIVENESS_NORMALIZER_CACHE
    unique_species = qreq_.get_unique_species()
    assert len(unique_species) == 1
    species = unique_species[0]
    if species in DISTINCTIVENESS_NORMALIZER_CACHE:
        dstcnvs_normer = DISTINCTIVENESS_NORMALIZER_CACHE[species]
    else:
        dstcnvs_normer = DistinctivnessNormalizer(species)
        global_distinctdir = qreq_.ibs.get_global_distinctiveness_modeldir()
        cachedir = global_distinctdir
        dstcnvs_normer.load(cachedir)
        DISTINCTIVENESS_NORMALIZER_CACHE[species] = dstcnvs_normer
        return dstcnvs_normer


def dev_train_distinctiveness(species=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        species (None):

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness
        python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness --gz
        python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness --pz

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import ibeis
        >>> if ut.get_argflag('--gz'):
        ...     species = const.Species.ZEB_GREVY
        ... elif ut.get_argflag('--pz'):
        ...     species = const.Species.ZEB_PLAIN
        ... else:
        ...     raise AssertionError('no species')
        >>> dev_train_distinctiveness(species)
    """
    import ibeis
    if 'species' not in vars() or species is None:
        species = const.Species.ZEB_GREVY
    if species == const.Species.ZEB_GREVY:
        dbname = 'GZ_ALL'
    elif species == const.Species.ZEB_PLAIN:
        dbname = 'PZ_Master0'
    dstcnvs_normer = DistinctivnessNormalizer(species)
    ibs = ibeis.opendb(dbname)
    global_distinctdir = ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    try:
        with ut.Timer('loading distinctiveness'):
            dstcnvs_normer.load(cachedir)
        # Cache hit
        print('cache hit')
    except IOError:
        print('cache miss')
        with ut.Timer('training distinctiveness'):
            # Need to train
            # Add one example from each name
            # TODO: add one exemplar per viewpoint for each name
            #max_vecs = 1E6
            max_annots = 975
            nid_list = ibs.get_valid_nids()
            aids_list = ibs.get_name_aids(nid_list)
            num_annots_list = map(len, aids_list)
            aids_list = ut.sortedby(aids_list, num_annots_list, reverse=True)
            aid_list = ut.get_list_column(aids_list, 0)
            # Keep only a certain number of annots for distinctiveness mapping
            aid_list_ = ut.listclip(aid_list, max_annots)
            print('total num named annots = %r' % (sum(num_annots_list)))
            print('training distinctiveness using %d/%d singleton annots' % (len(aid_list_), len(aid_list)))
            # vec
            vecs_list = ibs.get_annot_vecs(aid_list_)
            num_vecs = sum(map(len, vecs_list))
            print('num_vecs = %r' % (num_vecs,))
            vecs = np.vstack(vecs_list)
            print('vecs size = %r' % (ut.get_object_size_str(vecs),))
            dstcnvs_normer.init_support(vecs)
            dstcnvs_normer.save(global_distinctdir)
    #vsone_
    #inct

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer
        python -m ibeis.model.hots.distinctiveness_normalizer --allexamples
        python -m ibeis.model.hots.distinctiveness_normalizer --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
