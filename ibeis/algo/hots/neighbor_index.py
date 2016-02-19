# -*- coding: utf-8 -*-
"""
TODO:
    Remove Bloat
multi_index.py as well
"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import utool as ut
import vtool as vt
import pyflann
#import lockfile
from os.path import basename
from six.moves import range, zip, map  # NOQA
from ibeis.algo.hots import hstypes
from ibeis.algo.hots import _pipeline_helpers as plh  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[neighbor_index]', DEBUG=False)

USE_HOTSPOTTER_CACHE = not ut.get_argflag('--nocache-hs')
NOSAVE_FLANN = ut.get_argflag('--nosave-flann')
NOCACHE_FLANN = ut.get_argflag('--nocache-flann') and USE_HOTSPOTTER_CACHE


def get_support_data(qreq_, daid_list):
    # TODO: look into ut.cached_func
    vecs_list = qreq_.ibs.get_annot_vecs(daid_list, config2_=qreq_.get_internal_data_config2())
    # <HACK:featweight>
    #hack to get  feature weights. returns None if feature weights are turned
    #off in config settings
    if qreq_.qparams.fg_on:
        fgws_list = qreq_.ibs.get_annot_fgweights(
            daid_list, config2_=qreq_.get_internal_data_config2(), ensure=True)
    else:
        fgws_list = None
    # </HACK:featweight>
    return vecs_list, fgws_list


def prepare_index_data(aid_list, vecs_list, fgws_list, verbose=True):
    r"""
    flattens vecs_list and builds a reverse index from the flattened indices
    (idx) to the original aids and fxs
    """
    print('[nnindex] Preparing data for indexing / loading index')
    # Check input
    assert len(aid_list) == len(vecs_list), 'invalid input. bad len'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')
    # Create indexes into the input aids
    ax_list = np.arange(len(aid_list))
    idx2_vec, idx2_ax, idx2_fx = invert_index(vecs_list, ax_list, verbose=verbose)
    # <HACK:fgweights>
    if fgws_list is not None:
        idx2_fgw = np.hstack(fgws_list)
        try:
            assert len(idx2_fgw) == len(idx2_vec), 'error. weights and vecs do not correspond'
        except Exception as ex:
            ut.printex(ex, keys=[(len, 'idx2_fgw'), (len, 'idx2_vec')])
            raise
    else:
        idx2_fgw = None
    # </HACK:fgweights>
    ax2_aid = np.array(aid_list)
    _preptup = (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx)
    return _preptup


def invert_index(vecs_list, ax_list, verbose=ut.NOT_QUIET):
    r"""
    Aggregates descriptors of input annotations and returns inverted information

    Args:
        vecs_list (list):
        ax_list (list):
        verbose (bool):  verbosity flag(default = True)

    Returns:
        tuple: (idx2_vec, idx2_ax, idx2_fx)

    CommandLine:
        python -m ibeis.algo.hots.neighbor_index --test-invert_index

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
        >>> import vtool as vt
        >>> num = 100
        >>> rng = np.random.RandomState(0)
        >>> ax_list = np.arange(num)
        >>> vecs_list = [vt.tests.dummy.get_dummy_dpts(rng.randint(100)) for ax in ax_list]
        >>> verbose = True
        >>> (idx2_vec, idx2_ax, idx2_fx) = invert_index(vecs_list, ax_list, verbose)
    """
    if ut.VERYVERBOSE:
        print('[nnindex] stacking descriptors from %d annotations' % len(ax_list))
    try:
        idx2_vec, idx2_ax, idx2_fx = vt.invertible_stack(vecs_list, ax_list)
        assert idx2_vec.shape[0] == idx2_ax.shape[0]
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
    except MemoryError as ex:
        ut.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if ut.VERYVERBOSE or verbose:
        print('[nnindex] stacked nVecs={nVecs} from nAnnots={nAnnots}'.format(
            nVecs=len(idx2_vec), nAnnots=len(ax_list)))
        print('[nnindex] idx2_vecs.dtype = {}'.format(idx2_vec.dtype))
        print('[nnindex] memory(idx2_vecs) = {}'.format(
            ut.byte_str2(idx2_vec.size * idx2_vec.dtype.itemsize)))
    return idx2_vec, idx2_ax, idx2_fx


@six.add_metaclass(ut.ReloadingMetaclass)
class NeighborIndex(object):
    r"""
    wrapper class around flann
    stores flann index and data it needs to index into

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
    """
    ext     = '.flann'
    prefix1 = 'flann'

    def __init__(nnindexer, flann_params, cfgstr):
        r"""
        initialize an empty neighbor indexer
        """
        nnindexer.flann    = None  # Approximate search structure
        nnindexer.ax2_aid  = None  # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = None  # (M x D) Descriptors to index
        nnindexer.idx2_fgw = None  # (M x 1) Descriptor forground weight
        nnindexer.idx2_ax  = None  # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = None  # (M x 1) Index into the annot's features
        nnindexer.cfgstr   = cfgstr  # configuration id
        if 'random_seed' not in flann_params:
            # Make flann determenistic for the same data
            flann_params['random_seed'] = 42
        nnindexer.flann_params = flann_params
        nnindexer.cores  = flann_params.get('cores', 0)
        nnindexer.checks = flann_params.get('checks', 1028)
        nnindexer.num_indexed = None
        nnindexer.flann_fpath = None
        nnindexer.max_distance_sqrd = None  # max possible distance^2 for normalization

    def init_support(nnindexer, aid_list, vecs_list, fgws_list, verbose=True):
        r"""
        prepares inverted indicies and FLANN data structure
        """
        assert nnindexer.flann is None, 'already initalized'
        _preptup = prepare_index_data(aid_list, vecs_list, fgws_list, verbose=verbose)
        (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = _preptup
        nnindexer.flann    = pyflann.FLANN()  # Approximate search structure
        nnindexer.ax2_aid  = ax2_aid   # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        nnindexer.idx2_fgw = idx2_fgw  # (M x 1) Descriptor forground weight
        nnindexer.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        nnindexer.num_indexed = nnindexer.idx2_vec.shape[0]
        if nnindexer.idx2_vec.dtype == hstypes.VEC_TYPE:
            # these are sift descriptors
            nnindexer.max_distance_sqrd = hstypes.VEC_PSEUDO_MAX_DISTANCE_SQRD
        else:
            # FIXME: hacky way to support siam128 descriptors.
            #raise AssertionError(
            #'NNindexer should get uint8s right now unless the algorithm has changed')
            nnindexer.max_distance_sqrd = None

    @ut.tracefunc_xml
    def add_ibeis_support(nnindexer, qreq_, new_daid_list, verbose=ut.NOT_QUIET):
        r"""
        # TODO: ensure that the memcache changes appropriately
        """
        from ibeis.algo.hots.neighbor_index import clear_memcache
        clear_memcache()
        if verbose:
            print('[nnindex] request add %d annots to single-indexer' % (len(new_daid_list)))
        duplicate_aids = set(new_daid_list).intersection(nnindexer.get_indexed_aids())
        if len(duplicate_aids) > 0:
            if verbose:
                print('[nnindex] request has %d annots that are already indexed. ignore those'
                      % (len(duplicate_aids),))
            new_daid_list_ = np.array(sorted(list(set(new_daid_list) - duplicate_aids)))
        else:
            new_daid_list_ = new_daid_list
        if len(new_daid_list_) == 0:
            if verbose:
                print('[nnindex] Nothing to do')
        else:
            new_vecs_list, new_fgws_list = get_support_data(qreq_, new_daid_list_)
            nnindexer.add_support(new_daid_list_, new_vecs_list, new_fgws_list, verbose=verbose)

    @ut.tracefunc_xml
    def remove_ibeis_support(nnindexer, qreq_, remove_daid_list, verbose=ut.NOT_QUIET):
        r"""
        # TODO: ensure that the memcache changes appropriately
        """
        if verbose:
            print('[nnindex] request remove %d annots from single-indexer' %
                  (len(remove_daid_list)))
        from ibeis.algo.hots.neighbor_index import clear_memcache
        clear_memcache()
        nnindexer.remove_support(remove_daid_list, verbose=verbose)

    def remove_support(nnindexer, remove_daid_list, verbose=ut.NOT_QUIET):
        r"""
        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --test-remove_support

        SeeAlso:
            ~/code/flann/src/python/pyflann/index.py

        Example:
            >>> # SLOW_DOCTEST
            >>> # (IMPORTANT)
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer(use_memcache=False)
            >>> remove_daid_list = [8, 9, 10, 11]
            >>> K = 2
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> # get before data
            >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K)
            >>> # execute test function
            >>> nnindexer.remove_support(remove_daid_list)
            >>> # test before data vs after data
            >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K)
            >>> ax2_nvecs = ut.dict_take(ut.dict_hist(nnindexer.idx2_ax), range(len(nnindexer.ax2_aid)))
            >>> assert qfx2_idx2.max() < ax2_nvecs[0], 'should only get points from aid 7'
            >>> assert qfx2_idx1.max() > ax2_nvecs[0], 'should get points from everyone'
        """
        if ut.DEBUG2:
            print('REMOVING POINTS')
        # TODO: ensure no duplicates
        ax2_remove_flag = np.in1d(nnindexer.ax2_aid, remove_daid_list)
        remove_ax_list = np.nonzero(ax2_remove_flag)[0]
        idx2_remove_flag = np.in1d(nnindexer.idx2_ax, remove_ax_list)
        remove_idx_list = np.nonzero(idx2_remove_flag)[0]
        if verbose:
            print('[nnindex] Found %d / %d annots that need removing' %
                  (len(remove_ax_list), len(remove_daid_list)))
            print('[nnindex] Removing %d indexed features' % (len(remove_idx_list),))
        # FIXME: indicies may need adjustment after remove points
        # Currently this is not being done and the data is just being left alone
        # This should be ok temporarilly because removed ids should not
        # be returned by the flann object
        nnindexer.flann.remove_points(remove_idx_list)

        # FIXME:
        #nnindexer.ax2_aid
        if True:
            nnindexer.ax2_aid[remove_ax_list] = -1
            nnindexer.idx2_fx[remove_idx_list] = -1
            nnindexer.idx2_vec[remove_idx_list] = 0
            if nnindexer.idx2_fgw is not None:
                nnindexer.idx2_fgw[remove_idx_list] = np.nan

        # FIXME: This will definitely bug out if you remove points and then try
        # to add the same points back again.

        if ut.DEBUG2:
            print('DONE REMOVE POINTS')

    def add_support(nnindexer, new_daid_list, new_vecs_list, new_fgws_list,
                    verbose=ut.NOT_QUIET):
        r"""
        adds support data (aka data to be indexed)

        Args:
            new_daid_list (list): list of annotation ids that are being added
            new_vecs_list (list): list of descriptor vectors for each annotation
            new_fgws_list (list): list of weights per vector for each annotation
            verbose (bool):  verbosity flag(default = True)

        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --test-add_support

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer(use_memcache=False)
            >>> new_daid_list = [2, 3, 4]
            >>> K = 2
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> # get before data
            >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K)
            >>> new_vecs_list, new_fgws_list = get_support_data(qreq_, new_daid_list)
            >>> # execute test function
            >>> nnindexer.add_support(new_daid_list, new_vecs_list, new_fgws_list)
            >>> # test before data vs after data
            >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K)
            >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        # TODO: ensure no duplicates
        nAnnots = nnindexer.num_indexed_annots()
        nVecs = nnindexer.num_indexed_vecs()
        nNewAnnots = len(new_daid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNewAnnots)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list, verbose=verbose)
        nNewVecs = len(new_idx2_vec)
        if verbose or ut.VERYVERBOSE:
            print(('[nnindex] Adding %d vecs from %d annots to nnindex '
                   'with %d vecs and %d annots') %
                  (nNewVecs, nNewAnnots, nVecs, nAnnots))
        if ut.DEBUG2:
            print('STACKING')
        # Stack inverted information
        old_idx2_vec = nnindexer.idx2_vec
        if nnindexer.idx2_fgw is not None:
            new_idx2_fgw = np.hstack(new_fgws_list)
            #nnindexer.old_vecs.append(new_idx2_fgw)
        ##---
        _ax2_aid = np.hstack((nnindexer.ax2_aid, new_daid_list))
        _idx2_ax = np.hstack((nnindexer.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((nnindexer.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((old_idx2_vec, new_idx2_vec))
        if nnindexer.idx2_fgw is not None:
            _idx2_fgw = np.hstack((nnindexer.idx2_fgw, new_idx2_fgw))
        if ut.DEBUG2:
            print('REPLACING')
        nnindexer.ax2_aid  = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        if nnindexer.idx2_fgw is not None:
            nnindexer.idx2_fgw = _idx2_fgw
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        if ut.DEBUG2:
            print('ADD POINTS (FIXME: SOMETIMES SEGFAULT OCCURS)')
            print('new_idx2_vec.dtype = %r' % new_idx2_vec.dtype)
            print('new_idx2_vec.shape = %r' % (new_idx2_vec.shape,))
        nnindexer.flann.add_points(new_idx2_vec)
        if ut.DEBUG2:
            print('DONE ADD POINTS')

    def ensure_indexer(nnindexer, cachedir, verbose=True, force_rebuild=False, memtrack=None):
        r"""
        Ensures that you get a neighbor indexer. It either loads a chached
        indexer or rebuilds a new one.
        """
        #with ut.PrintStartEndContext(msg='CACHED NNINDEX', verbose=verbose):
        if NOCACHE_FLANN or force_rebuild:
            print('...nnindex flann cache is forced off')
            load_success = False
        else:
            load_success = nnindexer.load(cachedir, verbose=verbose)
        if load_success:
            if not ut.QUIET:
                nVecs   = nnindexer.num_indexed_vecs()
                nAnnots = nnindexer.num_indexed_annots()
                print('...nnindex flann cache hit: %d vectors, %d annots' %
                      (nVecs, nAnnots))
        else:
            if not ut.QUIET:
                nVecs   = nnindexer.num_indexed_vecs()
                nAnnots = nnindexer.num_indexed_annots()
                print('...nnindex flann cache miss: %d vectors, %d annots' %
                      (nVecs, nAnnots))
            nnindexer.build_and_save(cachedir, verbose=verbose, memtrack=memtrack)

    def build_and_save(nnindexer, cachedir, verbose=True, memtrack=None):
        nnindexer.reindex(memtrack=memtrack)
        nnindexer.save(cachedir, verbose=verbose)

    def reindex(nnindexer, verbose=True, memtrack=None):
        r""" indexes all vectors with FLANN. """
        num_vecs = nnindexer.num_indexed
        notify_num = 1E6
        verbose_ = ut.VERYVERBOSE or verbose or (not ut.QUIET and num_vecs > notify_num)
        if verbose_:
            print('[nnindex] ...building kdtree over %d points (this may take a sec).' % num_vecs)
            tt = ut.tic(msg='Building index')
        idx2_vec = nnindexer.idx2_vec
        flann_params = nnindexer.flann_params
        if num_vecs == 0:
            print('WARNING: CANNOT BUILD FLANN INDEX OVER 0 POINTS. THIS MAY BE A SIGN OF A DEEPER ISSUE')
        else:
            if memtrack is not None:
                memtrack.report('BEFORE BUILD FLANN INDEX')
            nnindexer.flann.build_index(idx2_vec, **flann_params)
            if memtrack is not None:
                memtrack.report('AFTER BUILD FLANN INDEX')
        if verbose_:
            ut.toc(tt)

    # ---- <cachable_interface> ---

    def save(nnindexer, cachedir, verbose=True):
        r"""
        Caches a neighbor indexer to disk
        """
        if NOSAVE_FLANN:
            if ut.VERYVERBOSE or verbose:
                print('[nnindex] flann save is deactivated')
            return False
        flann_fpath = nnindexer.get_fpath(cachedir)
        nnindexer.flann_fpath = flann_fpath
        if ut.VERYVERBOSE or verbose:
            print('[nnindex] flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
        nnindexer.flann.save_index(flann_fpath)

    def load(nnindexer, cachedir, verbose=True):
        r"""
        Loads a cached neighbor indexer from disk
        """
        load_success = False
        flann_fpath = nnindexer.get_fpath(cachedir)
        nnindexer.flann_fpath = flann_fpath
        if ut.checkpath(flann_fpath, verbose=verbose):
            idx2_vec = nnindexer.idx2_vec
            # Warning: Loading a FLANN index with old headers may silently fail.
            try:
                nnindexer.flann.load_index(flann_fpath, idx2_vec)
            except (IOError, pyflann.FLANNException) as ex:
                ut.printex(ex, '... cannot load nnindex flann', iswarning=True)
            else:
                load_success = True
        return load_success

    def get_prefix(nnindexer):
        return nnindexer.prefix1

    def get_cfgstr(nnindexer, noquery=False):
        r""" returns string which uniquely identified configuration and support data

        Args:
            noquery (bool): if True cfgstr is only relevant to building the
                index. No search params are returned (default = False)

        Returns:
            str: flann_cfgstr

        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --test-get_cfgstr

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> cfgdict = dict(fg_on=False)
            >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='testdb1', preload=True, cfgdict=cfgdict)
            >>> nnindexer = qreq_.indexer
            >>> noquery = True
            >>> flann_cfgstr = nnindexer.get_cfgstr(noquery)
            >>> result = ('flann_cfgstr = %s' % (str(flann_cfgstr),))
            >>> print(result)
            flann_cfgstr = _FLANN((algo=kdtree,seed=42,t=8,))_VECS((5232,128)4mu3cl+!se1x13je)
        """
        flann_cfgstr_list = []
        use_params_hash = True
        if use_params_hash:
            flann_defaults = vt.get_flann_params(nnindexer.flann_params['algorithm'])
            flann_params_clean = flann_defaults.copy()
            ut.updateif_haskey(flann_params_clean, nnindexer.flann_params)
            if noquery:
                ut.delete_dict_keys(flann_params_clean, ['checks'])
            shortnames = dict(algorithm='algo', checks='chks', random_seed='seed', trees='t')
            short_params = dict([(shortnames.get(key, key), str(val)[0:7])
                                 for key, val in six.iteritems(flann_params_clean)])
            #  if key == 'algorithm'])  # or val != flann_defaults.get(key, None)])
            flann_valsig_ = ut.dict_str(
                short_params, nl=False, explicit=True, strvals=True)
            flann_valsig_ = flann_valsig_.lstrip('dict').replace(' ', '')
            #flann_valsig_ = str(list(flann_params.values()))
            #flann_valsig = ut.remove_chars(flann_valsig_, ', \'[]')
            flann_cfgstr_list.append('_FLANN(' + flann_valsig_ + ')')
        use_data_hash = True
        if use_data_hash:
            idx2_vec = nnindexer.idx2_vec
            vecs_hashstr = ut.hashstr_arr(idx2_vec, '_VECS')
            flann_cfgstr_list.append(vecs_hashstr)
        flann_cfgstr = ''.join(flann_cfgstr_list)
        return flann_cfgstr

    def get_fname(nnindexer):
        return basename(nnindexer.get_fpath(''))

    def get_fpath(nnindexer, cachedir, cfgstr=None):
        _args2_fpath = ut.util_cache._args2_fpath
        dpath  = cachedir
        prefix = nnindexer.get_prefix()
        cfgstr = nnindexer.get_cfgstr(noquery=True)
        ext    = nnindexer.ext
        fpath  = _args2_fpath(dpath, prefix, cfgstr, ext, write_hashtbl=False)
        print('flann fpath = %r' % (fpath,))
        return fpath

    # ---- </cachable_interface> ---

    def get_dtype(nnindexer):
        return nnindexer.idx2_vec.dtype

    def knn(nnindexer, qfx2_vec, K):
        r"""
        Returns the indices and squared distance to the nearest K neighbors.
        The distance is noramlized between zero and one using
        VEC_PSEUDO_MAX_DISTANCE = (np.sqrt(2) * VEC_PSEUDO_MAX)

        Args:
            qfx2_vec : (N x D) an array of N, D-dimensional query vectors

            K: number of approximate nearest neighbors to find

        Returns: tuple of (qfx2_idx, qfx2_dist)
            ndarray : qfx2_idx[n][k] (N x K) is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            ndarray : qfx2_dist[n][k] (N x K) is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]
                        distance is normalized squared euclidean distance.

        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --test-knn:0 --debug2
            python -m ibeis.algo.hots.neighbor_index --test-knn:1

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> K = 2
            >>> nnindexer.debug_nnindexer()
            >>> assert vt.check_sift_validity(qfx2_vec), 'bad SIFT properties'
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K)
            >>> result = str(qfx2_idx.shape) + ' ' + str(qfx2_dist.shape)
            >>> print('qfx2_vec.dtype = %r' % (qfx2_vec.dtype,))
            >>> print('nnindexer.max_distance_sqrd = %r' % (nnindexer.max_distance_sqrd,))
            >>> assert np.all(qfx2_dist < 1.0), (
            >>>    'distance should be less than 1. got %r' % (qfx2_dist,))
            >>> # Ensure distance calculations are correct
            >>> qfx2_dvec = nnindexer.idx2_vec[qfx2_idx.T]
            >>> targetdist = vt.L2_sift(qfx2_vec, qfx2_dvec).T ** 2
            >>> rawdist    = vt.L2_sqrd(qfx2_vec, qfx2_dvec).T
            >>> assert np.all(qfx2_dist * nnindexer.max_distance_sqrd == rawdist), (
            >>>    'inconsistant distance calculations')
            >>> assert np.allclose(targetdist, qfx2_dist), (
            >>>    'inconsistant distance calculations')

        Example2:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = np.empty((0, 128), dtype=nnindexer.get_dtype())
            >>> K = 2
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K)
            >>> result = str(qfx2_idx.shape) + ' ' + str(qfx2_dist.shape)
            >>> print(result)
            (0, 2) (0, 2)
        """
        if K == 0:
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(len(qfx2_vec), 0)
        if K > nnindexer.num_indexed or K == 0:
            # If we want more points than there are in the database
            # FLANN will raise an exception. This corner case
            # will hopefully only be hit if using the multi-indexer
            # so try this workaround which should seemlessly integrate
            # when the multi-indexer stacks the subindxer results.
            # There is a very strong possibility that this will cause errors
            # If this corner case is used in non-multi-indexer code
            K = nnindexer.num_indexed
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(len(qfx2_vec), 0)
        elif len(qfx2_vec) == 0:
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(0, K)
        else:
            try:
                # perform nearest neighbors
                (qfx2_idx, qfx2_raw_dist) = nnindexer.flann.nn_index(
                    qfx2_vec, K, checks=nnindexer.checks, cores=nnindexer.cores)
            except pyflann.FLANNException as ex:
                ut.printex(ex, 'probably misread the cached flann_fpath=%r' % (nnindexer.flann_fpath,))
                #ut.embed()
                # Uncomment and use if the flan index needs to be deleted
                #ibs = ut.search_stack_for_localvar('ibs')
                #cachedir = ibs.get_flann_cachedir()
                #flann_fpath = nnindexer.get_fpath(cachedir)
                raise
            # Ensure that distance returned are between 0 and 1
            if nnindexer.max_distance_sqrd is not None:
                qfx2_dist = np.divide(qfx2_raw_dist, nnindexer.max_distance_sqrd)
            else:
                qfx2_dist = qfx2_raw_dist
            if ut.DEBUG2:
                # Ensure distance calculations are correct
                qfx2_dvec = nnindexer.idx2_vec[qfx2_idx.T]
                targetdist = vt.L2_sift(qfx2_vec, qfx2_dvec).T ** 2
                rawdist    = vt.L2_sqrd(qfx2_vec, qfx2_dvec).T
                assert np.all(qfx2_raw_dist == rawdist), 'inconsistant distance calculations'
                assert np.allclose(targetdist, qfx2_dist), 'inconsistant distance calculations'
            #qfx2_dist = np.sqrt(qfx2_dist) / nnindexer.max_distance_sqrd
        return (qfx2_idx, qfx2_dist)

    def debug_nnindexer(nnindexer):
        r"""
        Makes sure the indexer has valid SIFT descriptors
        """
        # FIXME: they might not agree if data has been added / removed
        init_data, extra_data = nnindexer.flann.get_indexed_data()
        with ut.Indenter('[NNINDEX_DEBUG]'):
            print('extra_data = %r' % (extra_data,))
            print('init_data = %r' % (init_data,))
            print('nnindexer.max_distance_sqrd = %r' % (nnindexer.max_distance_sqrd,))
            data_agrees = nnindexer.idx2_vec is nnindexer.flann.get_indexed_data()[0]
            if data_agrees:
                print('indexed_data agrees')
            assert vt.check_sift_validity(init_data), 'bad SIFT properties'
            assert data_agrees, 'indexed data does not agree'

    def empty_neighbors(nnindexer, nQfx, K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def num_indexed_vecs(nnindexer):
        return nnindexer.idx2_vec.shape[0]

    def num_indexed_annots(nnindexer):
        #invalid_idxs = (nnindexer.ax2_aid[nnindexer.idx2_ax] == -1)
        return (nnindexer.ax2_aid != -1).sum()

    def get_indexed_aids(nnindexer):
        return nnindexer.ax2_aid[nnindexer.ax2_aid != -1]

    def get_indexed_vecs(nnindexer):
        valid_idxs = (nnindexer.ax2_aid[nnindexer.idx2_ax] != -1)
        valid_idx2_vec = nnindexer.idx2_vec.compress(valid_idxs, axis=0)
        return valid_idx2_vec

    def get_removed_idxs(nnindexer):
        r"""
        __removed_ids = nnindexer.flann._FLANN__removed_ids
        invalid_idxs = nnindexer.get_removed_idxs()
        assert len(np.intersect1d(invalid_idxs, __removed_ids)) == len(__removed_ids)
        """
        invalid_idxs = np.nonzero(nnindexer.ax2_aid[nnindexer.idx2_ax] == -1)[0]
        return invalid_idxs

    def get_nn_vecs(nnindexer, qfx2_nnidx):
        r""" gets matching vectors """
        return nnindexer.idx2_vec.take(qfx2_nnidx, axis=0)

    def get_nn_axs(nnindexer, qfx2_nnidx):
        r""" gets matching internal annotation indices """
        return nnindexer.idx2_ax.take(qfx2_nnidx)

    def get_nn_aids(nnindexer, qfx2_nnidx):
        r"""
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_aid : (N x K) qfx2_fx[n][k] is the annotation id index of the
                                kth approximate nearest data vector

        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --exec-get_nn_aids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> cfgdict = dict(fg_on=False)
            >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='testdb1',
            >>>                                        cfgdict=cfgdict,
            >>>                                        preload=True)
            >>> nnindexer = qreq_.indexer
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(
            >>>     qreq_.get_internal_qaids()[0],
            >>>     config2_=qreq_.get_internal_query_config2())
            >>> num_neighbors = 4
            >>> (qfx2_nnidx, qfx2_dist) = nnindexer.knn(qfx2_vec, num_neighbors)
            >>> qfx2_aid = nnindexer.get_nn_aids(qfx2_nnidx)
            >>> assert qfx2_aid.shape[1] == num_neighbors
            >>> result = ('qfx2_aid.shape = %r' % (qfx2_aid.shape,))
            >>> print(result)
            qfx2_aid.shape = (1257, 4)
        """
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nnidx):
        r"""
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fx : (N x K) qfx2_fx[n][k] is the feature index (w.r.t the
                               source annotation) of the kth approximate
                               nearest data vector
        """
        qfx2_fx = nnindexer.idx2_fx.take(qfx2_nnidx)
        return qfx2_fx

    def get_nn_fgws(nnindexer, qfx2_nnidx):
        r"""
        Gets forground weights of neighbors

        CommandLine:
            python -m ibeis.algo.hots.neighbor_index --exec-NeighborIndex.get_nn_fgws

        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fgw : (N x K) qfx2_fgw[n][k] is the annotation id index of the
                                kth forground weight
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer(dbname='testdb1')
            >>> qfx2_nnidx = np.array([[0, 1, 2], [3, 4, 5]])
            >>> qfx2_fgw = nnindexer.get_nn_fgws(qfx2_nnidx)
        """
        if nnindexer.idx2_fgw is None:
            qfx2_fgw = np.ones(qfx2_nnidx.shape)
        else:
            qfx2_fgw = nnindexer.idx2_fgw.take(qfx2_nnidx)
        return qfx2_fgw


def test_nnindexer(*args, **kwargs):
    from ibeis.algo.hots.neighbor_index_cache import test_nnindexer
    return test_nnindexer(*args, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.neighbor_index
        python -m ibeis.algo.hots.neighbor_index --allexamples
        utprof.sh ibeis/algo/hots/neighbor_index.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
