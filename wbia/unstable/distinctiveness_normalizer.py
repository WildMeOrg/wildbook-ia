# -*- coding: utf-8 -*-
"""
External mechanism for computing feature distinctiveness

DEPRICATE

stores some set of vectors which lose their association with
their parent.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
from six.moves import map
import six  # NOQA
from wbia import constants as const
from wbia.init import sysres
from wbia.algo.hots import hstypes

print, rrr, profile = ut.inject2(__name__)


DCVS_DEFAULT = ut.ParamInfoList(
    'distinctivness',
    [
        ut.ParamInfo('dcvs_power', 1.0, 'p', varyvals=[0.5, 1.0, 1.5, 2.0]),
        ut.ParamInfo('dcvs_min_clip', 0.2, 'mn', varyvals=[0.2, 0.02, 0.03][0:1]),
        ut.ParamInfo(
            'dcvs_max_clip', 0.5, 'mx', varyvals=[0.05, 0.3, 0.4, 0.45, 0.5, 1.0][1:4]
        ),
        ut.ParamInfo('dcvs_K', 5, 'dcvsK', varyvals=[5, 7, 15][0:1]),
    ],
)


DISTINCTIVENESS_NORMALIZER_CACHE = {}
BASELINE_DISTINCTIVNESS_URLS = {
    # TODO: Populate
    'zebra_grevys': const.ZIPPED_URLS.GZ_DISTINCTIVE,
    'zebra_plains': const.ZIPPED_URLS.PZ_DISTINCTIVE,
}
PUBLISH_DIR = ut.unixpath('~/Dropbox/IBEIS')


def testdata_distinctiveness():
    """
    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> dstcnvs_normer, qreq_ = testdata_distinctiveness()
    """
    import wbia

    # build test data
    db = ut.get_argval('--db', str, 'testdb1')
    species = ut.get_argval('--species', str, None)
    aid = ut.get_argval('--aid', int, None)
    ibs = wbia.opendb(db)
    if aid is not None:
        species = ibs.get_annot_species_texts(aid)
    if species is None:
        if db == 'testdb1':
            species = wbia.const.TEST_SPECIES.ZEB_PLAIN
    daids = ibs.get_valid_aids(species=species)
    qaids = [aid] if aid is not None else daids
    qreq_ = ibs.new_query_request(qaids, daids)
    dstcnvs_normer = request_wbia_distinctiveness_normalizer(qreq_)
    return dstcnvs_normer, qreq_


# @six.add_metaclass(ut.ReloadingMetaclass)
class DistinctivnessNormalizer(ut.Cachable):
    ext = '.cPkl'
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
        dstcnvs_normer.cores = dstcnvs_normer.flann_params.get('cores', 0)

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
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        data_fpath = dstcnvs_normer.get_fpath(cachedir)
        # flann_fpath   = dstcnvs_normer.get_flann_fpath(cachedir)
        archive_fpath = dstcnvs_normer.get_fpath(cachedir, ext='.zip')
        fpath_list = [
            data_fpath,
            # flann_fpath
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
            python -m wbia.algo.hots.distinctiveness_normalizer --test-publish

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
            >>> dstcnvs_normer = testdata_distinctiveness()[0]
            >>> dstcnvs_normer.rebuild()
            >>> dstcnvs_normer.save()
            >>> result = dstcnvs_normer.publish(cachedir)
            >>> # verify results
            >>> print(result)
        """
        from os.path import basename, join

        assert ut.is_developer(), 'ONLY DEVELOPERS CAN PERFORM THIS OPERATION'
        cachedir = dstcnvs_normer.cachedir if cachedir is None else cachedir
        archive_fpath = dstcnvs_normer.archive(cachedir, overwrite=True)
        archive_fname = basename(archive_fpath)
        publish_dpath = PUBLISH_DIR
        publish_fpath = join(publish_dpath, archive_fname)
        if ut.checkpath(publish_fpath, verbose=True):
            print('Overwriting model')
            print(
                'old nBytes(publish_fpath) = %s'
                % (ut.get_file_nBytes_str(publish_fpath),)
            )
            print(
                'new nBytes(archive_fpath) = %s'
                % (ut.get_file_nBytes_str(archive_fpath),)
            )
        else:
            print('Publishing model')
        print('publish_fpath = %r' % (publish_fpath,))
        ut.copy(archive_fpath, publish_fpath)

    def get_flann_fpath(dstcnvs_normer, cachedir):
        flann_fpath = dstcnvs_normer.get_fpath(cachedir, ext='.flann')
        return flann_fpath

    def exists(
        dstcnvs_normer, cachedir=None, verbose=True, need_flann=False, *args, **kwargs
    ):
        r"""
        Args:
            cachedir (str): cache directory
            verbose (bool):  verbosity flag

        Returns:
            flag: load_success

        CommandLine:
            python -m wbia.algo.hots.distinctiveness_normalizer --test-exists

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
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
        # # Load Flann
        # if ut.VERBOSE:
        #    print('[nnindex] load_success = %r' % (load_success,))

    def load_or_build_flann(dstcnvs_normer, cachedir=None, verbose=True, *args, **kwargs):
        from vtool._pyflann_backend import pyflann as pyflann

        flann_fpath = dstcnvs_normer.get_flann_fpath(cachedir)
        if ut.checkpath(flann_fpath, verbose=ut.VERBOSE):
            try:
                dstcnvs_normer.flann = pyflann.FLANN()
                dstcnvs_normer.flann.load_index(flann_fpath, dstcnvs_normer.vecs)
                assert dstcnvs_normer.flann._FLANN__curindex is not None
                # load_success = True
            except Exception as ex:
                ut.printex(ex, '... cannot load distinctiveness flann', iswarning=True)
                dstcnvs_normer.rebuild(cachedir)
        else:
            dstcnvs_normer.ensure_flann(cachedir)
            # raise IOError('cannot load distinctiveness flann')
        # return load_success

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
        # dstcnvs_normer.flann = vt.build_flann_index(
        #     dstcnvs_normer.vecs, dstcnvs_normer.flann_params, verbose=verbose)
        if dstcnvs_normer.vecs.dtype == hstypes.VEC_TYPE:
            dstcnvs_normer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
            dstcnvs_normer.max_distance_sqrd = dstcnvs_normer.max_distance ** 2

    def ensure_flann(dstcnvs_normer, cachedir=None):
        if not ut.checkpath(dstcnvs_normer.get_flann_fpath(cachedir)):
            dstcnvs_normer.rebuild(cachedir)
            dstcnvs_normer.save_flann(cachedir)

    def get_distinctiveness(
        dstcnvs_normer,
        qfx2_vec,
        dcvs_K=2,
        dcvs_power=1.0,
        dcvs_max_clip=1.0,
        dcvs_min_clip=0.0,
    ):
        r"""
        Args:
            qfx2_vec (ndarray):  mapping from query feature index to vec

        CommandLine:
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --db GZ_ALL --show
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_power .25
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_power .5
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_power .1
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_K 1&
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_K 2&
            python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --dcvs_K 3&

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
            >>> dstcnvs_normer, qreq_ = testdata_distinctiveness()
            >>> qaid = qreq_.qaids[0]
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(qaid, config2_=qreq_.qparams)
            >>> default_dict = {'dcvs_power': .25, 'dcvs_K': 5, 'dcvs_max_clip': .5}
            >>> kwargs = ut.argparse_dict(default_dict)
            >>> qfx2_dstncvs = dstcnvs_normer.get_distinctiveness(qfx2_vec, **kwargs)
            >>> ut.assert_eq(len(qfx2_dstncvs.shape), 1)
            >>> assert np.all(qfx2_dstncvs) <= 1
            >>> assert np.all(qfx2_dstncvs) >= 0
            >>> ut.quit_if_noshow()
            >>> # Show distinctivness on an animal and a corresponding graph
            >>> import wbia.plottool as pt
            >>> chip = qreq_.ibs.get_annot_chips(qaid)
            >>> qfx2_kpts = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.qparams)
            >>> show_chip_distinctiveness_plot(chip, qfx2_kpts, qfx2_dstncvs)
            >>> #pt.figure(2)
            >>> #pt.show_all_colormaps()
            >>> pt.show_if_requested()

        Ignore:
            %s/\(^ *\)\(.*\)/\1>>> \2/c

        """
        # ut.embed()
        assert dcvs_K > 0 and dcvs_K < len(dstcnvs_normer.vecs), 'dcvs_K=%r' % (dcvs_K,)
        if len(qfx2_vec) == 0:
            # (qfx2_idx, qfx2_dist_sqrd) = dstcnvs_normer.empty_neighbors(0, dcvs_K)
            qfx2_idx = np.empty((0, dcvs_K), dtype=np.int32)
            qfx2_dist = np.empty((0, dcvs_K), dtype=np.float64)
        else:
            # perform nearest neighbors
            (qfx2_idx, qfx2_dist_sqrd) = dstcnvs_normer.flann.nn_index(
                qfx2_vec, dcvs_K, checks=dstcnvs_normer.checks, cores=dstcnvs_normer.cores
            )
            # Ensure that distance returned are between 0 and 1
            # qfx2_dist = qfx2_dist / (dstcnvs_normer.max_distance ** 2)
            qfx2_dist = (
                np.sqrt(qfx2_dist_sqrd.astype(np.float64))
                / hstypes.VEC_PSEUDO_MAX_DISTANCE
            )
            # qfx2_dist32 = np.sqrt(np.divide(qfx2_dist_sqrd, dstcnvs_normer.max_distance_sqrd))
            # qfx2_dist =
            # qfx2_dist = np.sqrt(qfx2_dist) / dstcnvs_normer.max_distance
        if dcvs_K == 1:
            qfx2_dist = qfx2_dist[:, None]
        norm_dist = qfx2_dist.T[dcvs_K - 1].T
        qfx2_dstncvs = compute_distinctiveness_from_dist(
            norm_dist, dcvs_power, dcvs_max_clip, dcvs_min_clip
        )
        return qfx2_dstncvs


def compute_distinctiveness_from_dist(
    norm_dist, dcvs_power, dcvs_max_clip, dcvs_min_clip
):
    """
    Compute distinctiveness from distance to dcvs_K+1 nearest neighbor

    Ignore:
        norm_dist = np.random.rand(1000)

        import numexpr

        %timeit np.divide(norm_dist, dcvs_max_clip)
        %timeit numexpr.evaluate('norm_dist / dcvs_max_clip', local_dict=dict(norm_dist=norm_dist, dcvs_max_clip=dcvs_max_clip))
        wd_cliped = np.divide(norm_dist, dcvs_max_clip)

        %timeit numexpr.evaluate('wd_cliped > 1.0', local_dict=locals())
        %timeit np.greater(wd_cliped, 1.0)

        %timeit np.power(wd_cliped, dcvs_power)
        %timeit numexpr.evaluate('wd_cliped ** dcvs_power', local_dict=locals())

        %timeit
    """
    # expondent to augment distinctiveness scores.
    # clip the distinctiveness at this fraction
    clip_range = dcvs_max_clip - dcvs_min_clip
    # apply distinctivness normalization
    _tmp = np.clip(norm_dist, dcvs_min_clip, dcvs_max_clip)
    np.subtract(_tmp, dcvs_min_clip, out=_tmp)
    np.divide(_tmp, clip_range, out=_tmp)
    np.power(_tmp, dcvs_power, out=_tmp)
    dstncvs = _tmp
    return dstncvs


def show_chip_distinctiveness_plot(chip, kpts, dstncvs, fnum=1, pnum=None):
    import wbia.plottool as pt

    pt.figure(fnum, pnum=pnum)
    ax = pt.gca()
    divider = pt.ensure_divider(ax)
    # ax1 = divider.append_axes("left", size="50%", pad=0)
    ax1 = ax
    ax2 = divider.append_axes('bottom', size='100%', pad=0.05)
    # f, (ax1, ax2) = pt.plt.subplots(1, 2, sharex=True)
    cmapstr = 'rainbow'  # 'hot'
    color_list = pt.df2.plt.get_cmap(cmapstr)(ut.norm_zero_one(dstncvs))
    sortx = dstncvs.argsort()
    # pt.df2.plt.plot(qfx2_dstncvs[sortx], c=color_list[sortx])
    pt.plt.sca(ax1)
    pt.colorline(np.arange(len(sortx)), dstncvs[sortx], cmap=pt.plt.get_cmap(cmapstr))
    pt.gca().set_xlim(0, len(sortx))
    pt.dark_background()
    pt.plt.sca(ax2)
    pt.imshow(chip, darken=0.2)
    # MATPLOTLIB BUG CANNOT SHOW DIFFERENT ALPHA FOR POINTS AND KEYPOINTS AT ONCE
    # pt.draw_kpts2(kpts, pts_color=color_list, ell_color=color_list, ell_alpha=.1, ell=True, pts=True)
    # pt.draw_kpts2(kpts, color_list=color_list, pts_alpha=1.0, pts_size=1.5,
    #              ell=True, ell_alpha=.1, pts=False)
    ell = ut.get_argflag('--ell')
    pt.draw_kpts2(
        kpts,
        color_list=color_list,
        pts_alpha=1.0,
        pts_size=1.5,
        ell=ell,
        ell_alpha=0.3,
        pts=not ell,
    )
    pt.plt.sca(ax)
    # pt.figure(fnum, pnum=pnum)


def download_baseline_distinctiveness_normalizer(cachedir, species):
    zipped_url = BASELINE_DISTINCTIVNESS_URLS[species]
    ut.grab_zipped_url(zipped_url, ensure=True, download_dir=cachedir)
    # ut.assert_eq(ut.unixpath(cachedir), dir_)


def request_wbia_distinctiveness_normalizer(qreq_, verbose=True):
    r"""
    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-request_wbia_distinctiveness_normalizer

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> daids = ibs.get_valid_aids(species=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> qaids = ibs.get_valid_aids(species=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(qaids, daids)
        >>> # execute function
        >>> dstcnvs_normer = request_wbia_distinctiveness_normalizer(qreq_)
        >>> # verify results
        >>> assert dstcnvs_normer is not None
    """
    global DISTINCTIVENESS_NORMALIZER_CACHE
    unique_species = qreq_.get_unique_species()
    assert len(unique_species) == 1
    species = unique_species[0]
    global_distinctdir = qreq_.ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    dstcnvs_normer = request_species_distinctiveness_normalizer(
        species, cachedir, verbose=False
    )
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
        # dstcnvs_normer.ensure_flann(cachedir)
        assert dstcnvs_normer.exists(
            cachedir, need_flann=True
        ), 'normalizer should have been downloaded, but it doesnt exist'
        DISTINCTIVENESS_NORMALIZER_CACHE[species] = dstcnvs_normer
    return dstcnvs_normer


def clear_distinctivness_cache(j):
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    ut.remove_files_in_dir(global_distinctdir)


def list_distinctivness_cache():
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    print(ut.repr2(ut.ls(global_distinctdir)))


def list_published_distinctivness():
    r"""
    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-list_published_distinctivness

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> published_fpaths = list_published_distinctivness()
        >>> print(ut.repr2(published_fpaths))
    """
    published_fpaths = ut.ls(PUBLISH_DIR)
    return published_fpaths


def view_distinctiveness_model_dir():
    r"""
    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-view_distinctiveness_model_dir

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> view_distinctiveness_model_dir()
    """
    global_distinctdir = sysres.get_global_distinctiveness_modeldir()
    ut.vd(global_distinctdir)


def view_publish_dir():
    r"""
    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-view_publish_dir

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> view_publish_dir()
    """
    ut.vd(PUBLISH_DIR)


def tst_single_annot_distinctiveness_params(ibs, aid):
    r"""

    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show
        python -m wbia.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db GZ_ALL

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb(ut.get_argval('--db', type_=str, default='PZ_MTEST'))
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> # execute function
        >>> test_single_annot_distinctiveness_params(ibs, aid)
        >>> pt.show_if_requested()
    """
    ####
    # TODO: Also paramatarize the downweighting based on the keypoint size
    ####
    # HACK IN ABILITY TO SET CONFIG
    from wbia.init.main_commands import postload_commands
    from wbia.algo import Config

    postload_commands(ibs, None)

    import wbia.plottool as pt

    # cfglbl_list = cfgdict_list
    # ut.all_dict_combinations_lbls(varied_dict)

    # Get info to find distinctivness of
    species_text = ibs.get_annot_species(aid)
    # FIXME; qreq_ params for config rowid
    vecs = ibs.get_annot_vecs(aid)
    kpts = ibs.get_annot_kpts(aid)
    chip = ibs.get_annot_chips(aid)

    # Paramater space to search
    # TODO: use slicing to control the params being varied
    # Use GridSearch class to modify paramaters as you go.

    varied_dict = Config.DCVS_DEFAULT.get_varydict()

    print('Varied Dict: ')
    print(ut.repr2(varied_dict))

    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict)

    # Get groundtruthish distinctivness map
    # for objective function
    # Load distinctivness normalizer
    with ut.Timer('Loading Distinctivness Normalizer for %s' % (species_text)):
        dstcvnss_normer = request_species_distinctiveness_normalizer(species_text)

    # Get distinctivness over all params
    dstncvs_list = [
        dstcvnss_normer.get_distinctiveness(vecs, **cfgdict)
        for cfgdict in ut.ProgIter(cfgdict_list, lbl='get dstcvns')
    ]

    # fgweights = ibs.get_annot_fgweights([aid])[0]
    # dstncvs_list = [x * fgweights for x in dstncvs_list]
    fnum = 1

    import functools

    show_func = functools.partial(show_chip_distinctiveness_plot, chip, kpts)

    ut.interact_gridsearch_result_images(
        show_func,
        cfgdict_list,
        cfglbl_list,
        dstncvs_list,
        score_list=None,
        fnum=fnum,
        figtitle='dstncvs gridsearch',
    )

    pt.present()


def dev_train_distinctiveness(species=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        species (None):

    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer --test-dev_train_distinctiveness

        alias dev_train_distinctiveness='python -m wbia.algo.hots.distinctiveness_normalizer --test-dev_train_distinctiveness'
        # Publishing (uses cached normalizers if available)
        dev_train_distinctiveness --species GZ --publish
        dev_train_distinctiveness --species PZ --publish
        dev_train_distinctiveness --species PZ --retrain

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.distinctiveness_normalizer import *  # NOQA
        >>> import wbia
        >>> species = ut.get_argval('--species', str, 'zebra_grevys')
        >>> dev_train_distinctiveness(species)
    """
    import wbia

    # if 'species' not in vars() or species is None:
    #    species = 'zebra_grevys'
    if species == 'zebra_grevys':
        dbname = 'GZ_ALL'
    elif species == 'zebra_plains':
        dbname = 'PZ_Master0'
    ibs = wbia.opendb(dbname)
    global_distinctdir = ibs.get_global_distinctiveness_modeldir()
    cachedir = global_distinctdir
    dstcnvs_normer = DistinctivnessNormalizer(species, cachedir=cachedir)
    try:
        if ut.get_argflag('--retrain'):
            raise IOError('force cache miss')
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
            # max_vecs = 1E6
            # max_annots = 975
            max_annots = 975
            # ibs.fix_and_clean_database()
            nid_list = ibs.get_valid_nids()
            aids_list = ibs.get_name_aids(nid_list)
            # remove junk
            aids_list = ibs.unflat_map(ibs.filter_junk_annotations, aids_list)
            # remove empty
            aids_list = [aids for aids in aids_list if len(aids) > 0]
            num_annots_list = list(map(len, aids_list))
            aids_list = ut.sortedby(aids_list, num_annots_list, reverse=True)
            # take only one annot per name
            aid_list = ut.get_list_column(aids_list, 0)
            # Keep only a certain number of annots for distinctiveness mapping
            aid_list_ = ut.listclip(aid_list, max_annots)
            print('total num named annots = %r' % (sum(num_annots_list)))
            print(
                'training distinctiveness using %d/%d singleton annots'
                % (len(aid_list_), len(aid_list))
            )
            # vec
            # FIXME: qreq_ params for config rowid
            vecs_list = ibs.get_annot_vecs(aid_list_)
            num_vecs = sum(list(map(len, vecs_list)))
            print('num_vecs = %r' % (num_vecs,))
            vecs = np.vstack(vecs_list)
            print('vecs size = %r' % (ut.get_object_size_str(vecs),))
            dstcnvs_normer.init_support(vecs)
            dstcnvs_normer.save(global_distinctdir)

    if ut.get_argflag('--publish'):
        dstcnvs_normer.publish()
    # vsone_
    # inct


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.distinctiveness_normalizer
        python -m wbia.algo.hots.distinctiveness_normalizer --allexamples
        python -m wbia.algo.hots.distinctiveness_normalizer --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
