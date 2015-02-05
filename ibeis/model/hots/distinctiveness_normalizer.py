"""
External mechanism for computing feature distinctiveness

stores some set of vectors which lose their association with
their parent.
"""

from __future__ import absolute_import, division, print_function
import utool
#from os.path import join
#import numpy as np
import vtool as vt
import utool as ut
import numpy as np
#import vtool as vt
import six  # NOQA
from ibeis import constants as const
import pyflann
from ibeis import sysres
from ibeis.model.hots import hstypes
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[distinctnorm]', DEBUG=False)


DISTINCTIVENESS_NORMALIZER_CACHE = {}
BASELINE_DISTINCTIVNESS_URLS = {
    # TODO: Populate
    const.Species.ZEB_GREVY: const.ZIPPED_URLS.GZ_DISTINCTIVE,
    const.Species.ZEB_PLAIN: const.ZIPPED_URLS.PZ_DISTINCTIVE,
}
PUBLISH_DIR = ut.unixpath('~/Dropbox/IBEIS')


def testdata_distinctiveness(species=None):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> dstcnvs_normer, qreq_ = testdata_distinctiveness()
    """
    import ibeis
    # build test data
    ibs = ibeis.opendb('testdb1')
    if species is None:
        species = ibeis.const.Species.ZEB_PLAIN
    daids = ibs.get_valid_aids(species=species)
    qaids = daids
    qreq_ = ibs.new_query_request(qaids, daids)
    dstcnvs_normer = request_ibeis_distinctiveness_normalizer(qreq_)
    return dstcnvs_normer, qreq_


@six.add_metaclass(ut.ReloadingMetaclass)
class DistinctivnessNormalizer(ut.Cachable):
    ext    = '.cPkl'
    prefix = 'distinctivness'

    def __init__(dstcnvs_normer, species, cachedir=None):
        """ cfgstring should be the species trained on """
        dstcnvs_normer.vecs = None
        dstcnvs_normer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
        dstcnvs_normer.max_distance_sqrd = dstcnvs_normer.max_distance ** 2
        dstcnvs_normer.cachedir = cachedir
        dstcnvs_normer.species = species
        dstcnvs_normer.flann_params = {'algorithm': 'kdtree', 'trees': 8, 'checks': 800}
        dstcnvs_normer.checks = dstcnvs_normer.flann_params.get('checks')
        dstcnvs_normer.cores  = dstcnvs_normer.flann_params.get('cores', 0)

    def get_prefix(dstcnvs_normer):
        return DistinctivnessNormalizer.prefix + '_'

    def get_cfgstr(dstcnvs_normer):
        assert dstcnvs_normer.species is not None
        cfgstr = dstcnvs_normer.species
        return cfgstr

    def add_support(dstcnvs_normer, new_vecs):
        """
        """
        raise NotImplementedError()
        pass

    def archive(dstcnvs_normer, cachedir=None, overwrite=False):
        cachedir      = dstcnvs_normer.cachedir if cachedir is None else cachedir
        data_fpath    = dstcnvs_normer.get_fpath(cachedir)
        #flann_fpath   = dstcnvs_normer.get_flann_fpath(cachedir)
        archive_fpath = dstcnvs_normer.get_fpath(cachedir, ext='.zip')
        fpath_list = [
            data_fpath,
            #flann_fpath
        ]
        ut.archive_files(archive_fpath, fpath_list, overwrite=overwrite)
        return archive_fpath

    def publish(dstcnvs_normer, cachedir=None):
        """
        Sets this as the default normalizer available for download
        ONLY DEVELOPERS CAN PERFORM THIS OPERATION

        Args:
            cachedir (str):

        CommandLine:
            python -m ibeis.model.hots.distinctiveness_normalizer --test-publish

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
            >>> dstcnvs_normer = testdata_distinctiveness()[0]
            >>> dstcnvs_normer.rebuild()
            >>> dstcnvs_normer.save()
            >>> result = dstcnvs_normer.publish(cachedir)
            >>> # verify results
            >>> print(result)
        """
        from os.path import basename, join
        assert ut.is_developer(), 'ONLY DEVELOPERS CAN PERFORM THIS OPERATION'
        cachedir      = dstcnvs_normer.cachedir if cachedir is None else cachedir
        archive_fpath = dstcnvs_normer.archive(cachedir, overwrite=True)
        archive_fname = basename(archive_fpath)
        publish_dpath = PUBLISH_DIR
        publish_fpath = join(publish_dpath, archive_fname)
        if ut.checkpath(publish_fpath, verbose=True):
            print('Overwriting model')
            print('old nBytes(publish_fpath) = %s' % (ut.get_file_nBytes_str(publish_fpath),))
            print('new nBytes(archive_fpath) = %s' % (ut.get_file_nBytes_str(archive_fpath),))
        else:
            print('Publishing model')
        print('publish_fpath = %r' % (publish_fpath,))
        ut.copy(archive_fpath, publish_fpath)

    def get_flann_fpath(dstcnvs_normer, cachedir):
        flann_fpath = dstcnvs_normer.get_fpath(cachedir, ext='.flann')
        return flann_fpath

    def exists(dstcnvs_normer, cachedir=None, verbose=True, need_flann=False, *args, **kwargs):
        r"""
        Args:
            cachedir (str): cache directory
            verbose (bool):  verbosity flag

        Returns:
            flag: load_success

        CommandLine:
            python -m ibeis.model.hots.distinctiveness_normalizer --test-exists

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
            >>> # build test data
            >>> dstcnvs_normer = testdata_distinctiveness()[0]
            >>> assert dstcnvs_normer.exists()
        """
        from os.path import exists
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        cpkl_fpath = dstcnvs_normer.get_fpath(cachedir)
        flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
        fpath_list = [cpkl_fpath]
        if need_flann:
            fpath_list.append(flann_fpath)
        flag = all([exists(fpath) for fpath in fpath_list])
        return flag

    def load(dstcnvs_normer, cachedir=None, verbose=True, *args, **kwargs):
        # Inherited method
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        kwargs['ignore_keys'] = ['flann']
        super(DistinctivnessNormalizer, dstcnvs_normer).load(cachedir, *args, **kwargs)
        dstcnvs_normer.load_or_build_flann(cachedir, verbose, *args, **kwargs)
        ## Load Flann
        #if ut.VERBOSE:
        #    print('[nnindex] load_success = %r' % (load_success,))

    def load_or_build_flann(dstcnvs_normer, cachedir=None, verbose=True, *args, **kwargs):
        flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
        if ut.checkpath(flann_fpath, verbose=ut.VERBOSE):
            try:
                dstcnvs_normer.flann = pyflann.FLANN()
                dstcnvs_normer.flann.load_index(flann_fpath, dstcnvs_normer.vecs)
                #load_success = True
            except Exception as ex:
                ut.printex(ex, '... cannot load distinctiveness flann', iswarning=True)
        else:
            dstcnvs_normer.ensure_flann(cachedir)
            #raise IOError('cannot load distinctiveness flann')
        #return load_success

    def save(dstcnvs_normer, cachedir=None, verbose=True, *args, **kwargs):
        """
        args = tuple()
        kwargs = {}
        """
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        # Inherited method
        kwargs['ignore_keys'] = ['flann']
        # Save everything but flann
        super(DistinctivnessNormalizer, dstcnvs_normer).save(cachedir, *args, **kwargs)
        # Save flann
        if dstcnvs_normer.flann is not None:
            dstcnvs_normer.save_flann(cachedir, verbose=verbose)

    def save_flann(dstcnvs_normer, cachedir=None, verbose=True):
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
        if verbose:
            print('flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
        dstcnvs_normer.flann.save_index(flann_fpath)

    def init_support(dstcnvs_normer, vecs, verbose=True):
        dstcnvs_normer.vecs = vecs
        dstcnvs_normer.rebuild(verbose=verbose)

    def rebuild(dstcnvs_normer, verbose=True, quiet=False):
        dstcnvs_normer.flann = vt.build_flann_index(
            dstcnvs_normer.vecs, dstcnvs_normer.flann_params, verbose=verbose)
        if dstcnvs_normer.vecs.dtype == hstypes.VEC_TYPE:
            dstcnvs_normer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
            dstcnvs_normer.max_distance_sqrd = dstcnvs_normer.max_distance ** 2

    def ensure_flann(dstcnvs_normer, cachedir=None):
        if not ut.checkpath(dstcnvs_normer.get_flann_fpath(cachedir)):
            dstcnvs_normer.rebuild(cachedir)
            dstcnvs_normer.save_flann(cachedir)

    def get_distinctiveness(dstcnvs_normer, qfx2_vec, **kwargs):
        r"""
        Args:
            qfx2_vec (ndarray):  mapping from query feature index to vec

        CommandLine:
            python -m ibeis.model.hots.distinctiveness_normalizer --test-get_distinctiveness

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
            >>> dstcnvs_normer, qreq_ = testdata_distinctiveness()
            >>> qaid = qreq_.get_external_qaids()[0]
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(qaid)
            >>> qfx2_dstncvs = dstcnvs_normer.get_distinctiveness(qfx2_vec)
            >>> ut.assert_eq(len(qfx2_dstncvs.shape), 1)
            >>> assert np.all(qfx2_dstncvs) <= 1
            >>> assert np.all(qfx2_dstncvs) >= 0
        """
        K = kwargs.get('K', 5)
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
        qfx2_dstncvs = compute_distinctiveness_from_dist(norm_sqared_dist, **kwargs)
        return qfx2_dstncvs


def compute_distinctiveness_from_dist(norm_sqared_dist, **kwargs):
    """
    Compute distinctiveness from distance to K+1 nearest neighbor

    Ignore:
        norm_sqared_dist = np.random.rand(1000)

        import numexpr

        %timeit np.divide(norm_sqared_dist, clip_fraction)
        %timeit numexpr.evaluate('norm_sqared_dist / clip_fraction', local_dict=dict(norm_sqared_dist=norm_sqared_dist, clip_fraction=clip_fraction))
        wd_cliped = np.divide(norm_sqared_dist, clip_fraction)

        %timeit numexpr.evaluate('wd_cliped > 1.0', local_dict=locals())
        %timeit np.greater(wd_cliped, 1.0)

        %timeit np.power(wd_cliped, p)
        %timeit numexpr.evaluate('wd_cliped ** p', local_dict=locals())

        %timeit
    """
    # TODO: paramaterize
    # expondent to augment distinctiveness scores.
    #p = kwargs.get('p', .5)
    p = kwargs.get('p', .25)
    #1.0)
    # clip the distinctiveness at this fraction
    #clip_fraction = kwargs.get('clip_fraction', .2)
    #clip_fraction = kwargs.get('clip_fraction', .4)
    clip_fraction = kwargs.get('clip_fraction', .5)
    wd_cliped = np.divide(norm_sqared_dist, clip_fraction)
    wd_cliped[np.greater(wd_cliped, 1.0)] = 1.0
    dstncvs = np.power(wd_cliped, p)
    return dstncvs


def download_baseline_distinctiveness_normalizer(cachedir, species):
    zipped_url = BASELINE_DISTINCTIVNESS_URLS[species]
    utool.grab_zipped_url(zipped_url, ensure=True, download_dir=cachedir)
    #ut.assert_eq(ut.unixpath(cachedir), dir_)


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
        >>> dstcnvs_normer = request_ibeis_distinctiveness_normalizer(qreq_)
        >>> # verify results
        >>> assert dstcnvs_normer is not None
    """
    global DISTINCTIVENESS_NORMALIZER_CACHE
    unique_species = qreq_.get_unique_species()
    assert len(unique_species) == 1
    species = unique_species[0]
    global_distinctdir = qreq_.ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    dstcnvs_normer = request_species_distinctiveness_normalizer(species, cachedir, verbose=False)
    return dstcnvs_normer


def request_species_distinctiveness_normalizer(species, cachedir=None, verbose=False):
    """
    helper function to get distinctivness model independent of IBEIS.
    """
    if species in DISTINCTIVENESS_NORMALIZER_CACHE:
        dstcnvs_normer = DISTINCTIVENESS_NORMALIZER_CACHE[species]
    else:
        if cachedir is None:
            cachedir = sysres.get_global_distinctiveness_modeldir(ensure=True)
        dstcnvs_normer = DistinctivnessNormalizer(species, cachedir=cachedir)
        if not dstcnvs_normer.exists(cachedir):
            # download normalizer if it doesn't exist
            download_baseline_distinctiveness_normalizer(cachedir, species)
        dstcnvs_normer.load(cachedir)
        print(ut.get_object_size_str(dstcnvs_normer, 'dstcnvs_normer = '))
        print('Loaded distinctivness normalizer')
        #dstcnvs_normer.ensure_flann(cachedir)
        assert dstcnvs_normer.exists(cachedir, need_flann=True), 'normalizer should have been downloaded, but it doesnt exist'
        DISTINCTIVENESS_NORMALIZER_CACHE[species] = dstcnvs_normer
    return dstcnvs_normer


def clear_distinctivness_cache():
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    ut.remove_files_in_dir(global_distinctdir)


def list_distinctivness_cache():
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    print(ut.list_str(ut.ls(global_distinctdir)))


def list_published_distinctivness():
    r"""
    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-list_published_distinctivness

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> published_fpaths = list_published_distinctivness()
        >>> print(ut.list_str(published_fpaths))
    """
    published_fpaths = ut.ls(PUBLISH_DIR)
    return published_fpaths


def view_distinctiveness_model_dir():
    r"""
    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-view_distinctiveness_model_dir

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> view_distinctiveness_model_dir()
    """
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    ut.vd(global_distinctdir)


def view_publish_dir():
    r"""
    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-view_publish_dir

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> view_publish_dir()
    """
    ut.vd(PUBLISH_DIR)


def test_single_annot_distinctiveness_params(ibs, aid):
    r"""

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show
        python -m ibeis.model.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db GZ_ALL

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import plottool as pt
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(ut.get_argval('--db', type_=str, default='PZ_MTEST'))
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> # execute function
        >>> test_single_annot_distinctiveness_params(ibs, aid)
        >>> pt.show_if_requested()
    """
    ####
    # TODO: Also paramatarize the downweighting based on the keypoint size
    ####
    # HACK IN ABILITY TO SET CONFIG
    from ibeis.dev.main_commands import postload_commands
    postload_commands(ibs, None)

    from vtool import coverage_image
    import plottool as pt
    from plottool import interact_impaint

    #cfglbl_list = cfgdict_list
    #ut.all_dict_combinations_lbls(varied_dict)

    # Get info to find distinctivness of
    species_text = ibs.get_annot_species(aid)
    vecs = ibs.get_annot_vecs(aid)
    kpts = ibs.get_annot_kpts(aid)
    print(kpts)
    chip = ibs.get_annot_chips(aid)
    chipsize = ibs.get_annot_chipsizes(aid)

    # Paramater space to search
    # TODO: use slicing to control the params being varied
    # Use GridSearch class to modify paramaters as you go.

    gauss_patch_varydict = {
        'gauss_shape': [(7, 7), (19, 19), (41, 41), (5, 5), (3, 3)],
        'gauss_sigma_frac': [.2, .5, .7, .95],
    }
    cov_blur_varydict = {
        'cov_blur_on': [True, False],
        'cov_blur_ksize': [(5, 5,),  (7, 7), (17, 17)],
        'cov_blur_sigma': [5.0, 1.2],
    }
    dstncvs_varydict = {
        'p': [.01, .1, .5, 1.0],
        'clip_fraction': [.05, .1, .2, .5],
        'K': [2, 3, 5],
    }
    size_penalty_varydict = {
        'remove_affine_information': [False, True],
        'constant_scaling': [False, True],
        'size_penalty_on': [True, False],
        'size_penalty_power': [.5, .1, 1.0],
        'size_penalty_scale': [.1, 1.0],
    }
    keyval_iter = ut.iflatten([
        dstncvs_varydict.items(),
        gauss_patch_varydict.items(),
        cov_blur_varydict.items(),
        size_penalty_varydict.items(),
    ])

    # Dont vary most paramaters, specify how much of their list can be used
    param_slice_dict = {
        'p'                  : slice(0, 2),
        'K'                  : slice(0, 2),
        'clip_fraction'      : slice(0, 2),
        'clip_fraction'      : slice(0, 2),
        #'gauss_shape'        : slice(0, 3),
        'gauss_sigma_frac'   : slice(0, 2),
        'remove_affine_information' : slice(0, 2),
        'constant_scaling' : slice(0, 2),
        'size_penalty_on' : slice(0, 2),
        #'cov_blur_on'        : slice(0, 2),
        #'cov_blur_ksize'     : slice(0, 2),
        #'cov_blur_sigma'     : slice(0, 1),
        #'size_penalty_power' : slice(0, 2),
        #'size_penalty_scale' : slice(0, 2),
    }
    varied_dict = {
        key: val[param_slice_dict.get(key, slice(0, 1))]
        for key, val in keyval_iter
    }

    def constrain_config(cfg):
        """ encode what makes a configuration feasible """
        if cfg['cov_blur_on'] is False:
            cfg['cov_blur_ksize'] = None
            cfg['cov_blur_sigma'] = None
        if cfg['constant_scaling'] is True:
            cfg['remove_affine_information'] = True
            cfg['size_penalty_on'] = False
        if cfg['remove_affine_information'] is True:
            cfg['gauss_shape'] = (41, 41)
        if cfg['size_penalty_on'] is False:
            cfg['size_penalty_power'] = None
            cfg['size_penalty_scale'] = None

    print('Varied Dict: ')
    print(ut.dict_str(varied_dict))

    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, constrain_config)

    # Get groundtruthish distinctivness map
    # for objective function
    GT_IS_DSTNCVS = 255
    GT_NOT_DSTNCVS = 100
    GT_UNKNOWN = 0
    label_colors = [GT_IS_DSTNCVS, GT_NOT_DSTNCVS, GT_UNKNOWN]
    gtmask = interact_impaint.cached_impaint(chip, 'dstncvnss',
                                             label_colors=label_colors,
                                             aug=True, refine=ut.get_argflag('--refine'))
    true_dstncvs_mask = gtmask == GT_IS_DSTNCVS
    false_dstncvs_mask = gtmask == GT_NOT_DSTNCVS

    true_dstncvs_mask_sum = true_dstncvs_mask.sum()
    false_dstncvs_mask_sum = false_dstncvs_mask.sum()

    def distinctiveness_objective_function(dstncvs_mask):
        true_mask  = true_dstncvs_mask * dstncvs_mask
        false_mask = false_dstncvs_mask * dstncvs_mask
        true_score = true_mask.sum() / true_dstncvs_mask_sum
        false_score = false_mask.sum() / false_dstncvs_mask_sum
        score = true_score * (1 - false_score)
        return score

    # Load distinctivness normalizer
    with ut.Timer('Loading Distinctivness Normalizer for %s' % (species_text)):
        dstcvnss_normer = request_species_distinctiveness_normalizer(species_text)

    # Get distinctivness over all params
    dstncvs_list = [dstcvnss_normer.get_distinctiveness(vecs, **cfgdict)
                    for cfgdict in ut.ProgressIter(cfgdict_list, lbl='get dstcvns')]

    # Then compute the distinctinvess coverage map
    #gauss_shape = kwargs.get('gauss_shape', (19, 19))
    #sigma_frac = kwargs.get('sigma_frac', .3)
    dstncvs_mask_list = [
        coverage_image.make_coverage_mask(
            kpts, chipsize, fx2_score=dstncvs, mode='max', return_patch=False, **cfg)
        for cfg, dstncvs in ut.ProgressIter(zip(cfgdict_list, dstncvs_list), lbl='Warping Image')
    ]
    score_list = [distinctiveness_objective_function(dstncvs_mask) for dstncvs_mask in dstncvs_mask_list]

    fnum = 1

    def show_covimg_result(img, fnum=None, pnum=None):
        pt.imshow(255 * img, fnum=fnum, pnum=pnum)

    ut.interact_gridsearch_result_images(
        show_covimg_result, cfgdict_list, cfglbl_list, dstncvs_mask_list,
        score_list=score_list, fnum=fnum, figtitle='dstncvs gridsearch')

    # Show subcomponents of grid search
    gauss_patch_cfgdict_list, gauss_patch_cfglbl_list = ut.get_cfgdict_lbl_list_subset(cfgdict_list, gauss_patch_varydict)
    patch_list = [coverage_image.get_gaussian_weight_patch(**cfgdict)
                  for cfgdict in ut.ProgressIter(gauss_patch_cfgdict_list, lbl='patch cfg')]

    ut.interact_gridsearch_result_images(
        show_covimg_result, gauss_patch_cfgdict_list, gauss_patch_cfglbl_list,
        patch_list, fnum=fnum + 1, figtitle='gaussian patches')

    patch = patch_list[0]
    #ut.embed()

    # Show the first mask in more depth
    dstncvs = dstncvs_list[0]
    dstncvs_mask = dstncvs_mask_list[0]
    coverage_image.show_coverage_map(chip, dstncvs_mask, patch, kpts, fnum=fnum + 2, ell_alpha=.2, show_mask_kpts=False)

    pt.imshow(gtmask, fnum=fnum + 3, pnum=(1, 2, 1), title='ground truth distinctiveness')
    pt.imshow(chip, fnum=fnum + 3, pnum=(1, 2, 2))
    pt.present()
    #ut.embed()
    #pt.iup()

    #ut.print_resource_usage()
    #pt.set_figtitle(mode)
    #pass


#def test_example():
#    import scipy.linalg as spl
#    M = np.array([
#        [1.0, 0.6, 0. , 0. , 0. ],
#        [0.6, 1.0, 0.5, 0.2, 0. ],
#        [0. , 0.5, 1.0, 0. , 0. ],
#        [0. , 0.2, 0. , 1.0, 0.8],
#        [0. , 0. , 0. , 0.8, 1.0],
#    ])
#    M_ = M / M.sum(axis=0)[:, None]
#    #eigvals, eigvecs = np.linalg.eigh(M_)
#    #, left=True, right=False)
#    eigvals, eigvecs = spl.eig(M_, left=True, right=False)
#    index = np.where(np.isclose(eigvals, 1))[0]
#    pi = stationary_vector = eigvecs.T[index]
#    pi_test = pi.dot(M_)
#    pi / pi.sum()
#    print(pi / np.linalg.norm(pi))
#    print(pi_test / np.linalg.norm(pi_test))

#    M = np.array([
#        [1.0, 0.6],
#        [0.6, 1.0],
#    ])
#    M_ = M / M.sum(axis=0)[:, None]
#    #eigvals, eigvecs = np.linalg.eigh(M_)
#    #, left=True, right=False)
#    eigvals, eigvecs = spl.eig(M_, left=True, right=False)
#    index = np.where(np.isclose(eigvals, 1))[0]
#    pi = stationary_vector = eigvecs.T[index]
#    pi_test = pi.dot(M_)
#    pi / pi.sum()
#    print(pi / np.linalg.norm(pi))
#    print(pi_test / np.linalg.norm(pi_test))
#    #pi = pi / pi.sum()


def dev_train_distinctiveness(species=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        species (None):

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness
        alias dev_train_distinctiveness='python -m ibeis.model.hots.distinctiveness_normalizer --test-dev_train_distinctiveness'
        dev_train_distinctiveness --species GZ --publish
        dev_train_distinctiveness --species PZ --publish

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import ibeis
        >>> species_code = ut.get_argval('--species', str, 'GZ')
        >>> species = sysres.resolve_species(species_code)
        >>> dev_train_distinctiveness(species)
    """
    import ibeis
    #if 'species' not in vars() or species is None:
    #    species = const.Species.ZEB_GREVY
    if species == const.Species.ZEB_GREVY:
        dbname = 'GZ_ALL'
    elif species == const.Species.ZEB_PLAIN:
        dbname = 'PZ_Master0'
    ibs = ibeis.opendb(dbname)
    global_distinctdir = ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    dstcnvs_normer = DistinctivnessNormalizer(species, cachedir=cachedir)
    try:
        with ut.Timer('loading distinctiveness'):
            dstcnvs_normer.load(cachedir)
        # Cache hit
        print('distinctivness model cache hit')
    except IOError:
        print('distinctivness model cache miss')
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

    if ut.get_argflag('--publish'):
        dstcnvs_normer.publish()
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
