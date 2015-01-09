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
#import vtool as vt
import six  # NOQA
from ibeis import constants as const
import pyflann
from ibeis.model.hots import hstypes
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[distinctnorm]', DEBUG=False)


@six.add_metaclass(ut.ReloadingMetaclass)
class DistinctivnessNormalizer(ut.Cachable):
    ext    = '.cPkl'
    prefix = 'distinctivness'

    def __init__(self, cfgstr):
        """ cfgstring should be the species trained on """
        self.vecs = None
        self.cfgstr = cfgstr
        self.flann_params = {'algorithm': 'kdtree', 'trees': 8, 'checks': 800}

    def get_prefix(self):
        return DistinctivnessNormalizer.prefix + '_'

    def get_cfgstr(self):
        assert self.cfgstr is not None
        return self.cfgstr

    def get_flann_fpath(self, cachedir):
        return self.get_fpath(cachedir, ext='.flann')

    def init_support(self, vecs, verbose=True):
        self.flann = pyflann.FLANN()  # Approximate search structure
        self.vecs = vecs
        num_vecs = len(vecs)
        notify_num = 1E6
        if verbose or (not ut.QUIET and num_vecs > notify_num):
            print('...building kdtree over %d points (this may take a sec).' % num_vecs)
        self.flann.build_index(self.vecs, **self.flann_params)
        if self.vecs.dtype == hstypes.VEC_TYPE:
            self.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE

    def add_support(self, new_vecs):
        """
        """
        pass

    def load(self, cachedir, verbose=True, *args, **kwargs):
        # Inherited method
        kwargs['ignore_keys'] = ['flann']
        super(DistinctivnessNormalizer, self).load(cachedir, *args, **kwargs)
        load_success = False
        # Load Flann
        flann_fpath = self.get_flann_fpath(cachedir)
        if ut.checkpath(flann_fpath, verbose=ut.VERBOSE):
            try:
                self.flann = pyflann.FLANN()
                self.flann.load_index(flann_fpath, self.vecs)
                load_success = True
            except Exception as ex:
                ut.printex(ex, '... cannot load distinctiveness flann', iswarning=True)
        else:
            raise IOError('cannot load distinctiveness flann')
        return load_success
        if ut.VERBOSE:
            print('[nnindex] load_success = %r' % (load_success,))

    def save(self, cachedir, verbose=True, *args, **kwargs):
        """
        args = tuple()
        kwargs = {}
        """
        # Inherited method
        kwargs['ignore_keys'] = ['flann']
        # Save everything but flann
        super(DistinctivnessNormalizer, self).save(cachedir, *args, **kwargs)
        # Save flann
        if self.flann is not None:
            flann_fpath = self.get_flann_fpath(cachedir)
            if verbose:
                print('flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
            self.flann.save_index(flann_fpath)

    def get_distinctiveness(self, qfx2_vec):
        K = 3
        assert K > 0 and K < len(self.vecs)
        if len(qfx2_vec) == 0:
            (qfx2_idx, qfx2_dist) = self.empty_neighbors(0, K)
        else:
            # perform nearest neighbors
            (qfx2_idx, qfx2_dist) = self.flann.nn_index(
                qfx2_vec, K, checks=self.checks, cores=self.cores)
            # Ensure that distance returned are between 0 and 1
            qfx2_dist = qfx2_dist / (self.max_distance ** 2)
            #qfx2_dist = np.sqrt(qfx2_dist) / self.max_distance
        return qfx2_dist.T[-1].T


def dev_train_distinctiveness(species=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        species (None):

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import ibeis
        >>> species = const.Species.ZEB_GREVY
        >>> species = const.Species.ZEB_PLAIN
        >>> dev_train_distinctiveness(species)
    """
    import ibeis
    import numpy as np
    if 'species' not in vars() or species is None:
        species = const.Species.ZEB_GREVY
    if species == const.Species.ZEB_GREVY:
        dbname = 'GZ_ALL'
    elif species == const.Species.ZEB_PLAIN:
        dbname = 'PZ_Master0'
    self = DistinctivnessNormalizer(species)
    ibs = ibeis.opendb(dbname)
    global_distinctdir = ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    try:
        self.load(cachedir)
        # Cache hit
        print('cache hit')
    except IOError:
        print('cache miss')
        # Need to train
        # Add one example from each name
        # TODO: add one exemplar per viewpoint for each name
        nid_list = ibs.get_valid_nids()
        aids_list = ibs.get_name_aids(nid_list)
        aid_list = ut.get_list_column(aids_list, 0)
        # vec
        vecs_list = ibs.get_annot_vecs(aid_list)
        vecs = np.vstack(vecs_list)
        self.init_support(vecs)
        self.save(global_distinctdir)
    #vsone_
    #inct
