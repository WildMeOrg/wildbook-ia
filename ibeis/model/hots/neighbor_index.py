from __future__ import absolute_import, division, print_function
import six
import numpy as np
import utool as ut
from os.path import join
import vtool.nearest_neighbors as nntool
from ibeis.model.hots import hstypes
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[neighbor_index]', DEBUG=False)

NOCACHE_FLANN = ut.get_argflag('--nocache-flann')


# cache for heavyweight nn structures.
# ensures that only one is in memory
NEIGHBOR_CACHE = {}
MAX_NEIGHBOR_CACHE_SIZE = 8
CURRENT_THREAD = None


def get_nnindexer_uuid_map_fpath(ibs):
    flann_cachedir = ibs.get_flann_cachedir()
    uuid_map_fname = 'uuid_map.shelf'
    uuid_map_fpath = join(flann_cachedir, uuid_map_fname)
    return uuid_map_fpath


@profile
def request_ibeis_nnindexer(qreq_, verbose=True, use_cache=True):
    """

    CALLED BY QUERYREQUST::LOAD_INDEXER

    TODO: naming convetion of new_ sucks. use request_ or
    anything else.

    FIXME: and use params from qparams instead of ibs.cfg

    IBEIS interface into neighbor_index

    Args:
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer(None)
        >>> nnindexer = request_ibeis_nnindexer(qreq_)
    """
    daid_list = qreq_.get_internal_daids()
    nnindexer = internal_request_ibeis_nnindexer(qreq_, daid_list,
                                                 verbose=verbose,
                                                 use_cache=use_cache)
    return nnindexer


def clear_uuid_cache(ibs):
    """

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-clear_uuid_cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = clear_uuid_cache(ibs)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(ibs)
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        uuid_map.clear()


def request_background_nnindexer(qreq_, daid_list):
    """ FIXME: Duplicate code """
    global CURRENT_THREAD
    if CURRENT_THREAD is not None and not CURRENT_THREAD.is_alive():
        # Make sure this function doesn't run if it is already running
        return False
    print('Requesting background reindex')
    daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)
    flann_cfgstr = qreq_.qparams.flann_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    #feat_cfgstr  = qreq_.qparams.feat_cfgstr
    # HACK: feature weights should probably have their own config
    #fw_cfgstr = 'weighted=%r' % (qreq_.qparams.fg_weight != 0)
    #indexer_cfgstr = ''.join((daids_hashid, flann_cfgstr, feat_cfgstr, fw_cfgstr))
    indexer_cfgstr = ''.join((daids_hashid, flann_cfgstr, featweight_cfgstr))
    flann_cachedir = qreq_.ibs.get_flann_cachedir()
    # Save inverted cache uuid mappings for
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    # Grab the keypoints names and image ids before query time?
    flann_params =  qreq_.qparams.flann_params
    # Get annot descriptors to index
    vecs_list = qreq_.ibs.get_annot_vecs(daid_list)
    fgws_list = get_fgweights_hack(qreq_, daid_list)
    preptup = prepare_index_data(daid_list, vecs_list, fgws_list, verbose=True)
    (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = preptup
    use_cache = True
    use_params_hash = False
    # Dont hash rowids when given enough info in indexer_cfgstr
    flann_params['cores'] = 2  # Only ues a few cores in the background
    flannkw = dict(cache_dir=flann_cachedir, cfgstr=indexer_cfgstr,
                   flann_params=flann_params, use_cache=use_cache,
                   use_params_hash=use_params_hash)
    #cores = flann_params.get('cores', 0)
    # Build/Load the flann index
    #flann = nntool.flann_cache(idx2_vec, verbose=verbose, **flannkw)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_.ibs)
    visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)

    threadobj = ut.spawn_background_process(
        background_flann_func, idx2_vec, flannkw, uuid_map_fpath, daids_hashid,
        visual_uuid_list, min_reindex_thresh)
    CURRENT_THREAD = threadobj


def background_flann_func(idx2_vec, flannkw, uuid_map_fpath, daids_hashid,
                          visual_uuid_list, min_reindex_thresh):
    """ FIXME: Duplicate code """
    print('Starting Background FLANN')
    nntool.flann_cache(idx2_vec, **flannkw)
    if len(visual_uuid_list) > min_reindex_thresh:
        # let the multi-indexer know about any big caches we've made
        # multi-indexer
        with ut.shelf_open(uuid_map_fpath) as uuid_map:
            uuid_map[daids_hashid] = visual_uuid_list
    print('Finished Background FLANN')


@profile
def internal_request_ibeis_nnindexer(qreq_, daid_list, verbose=True,
                                     use_cache=True):
    """ FOR INTERNAL USE ONLY
    takes custom daid list. might not be the same as what is in qreq_

    _aids (list): for multiindexer use only
        lets multindexer avoid shallow copies

    """
    global NEIGHBOR_CACHE
    daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)
    flann_cfgstr = qreq_.qparams.flann_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    #feat_cfgstr  = qreq_.qparams.feat_cfgstr
    # HACK: feature weights should probably have their own config
    #fw_cfgstr = 'weighted=%r' % (qreq_.qparams.fg_weight != 0)
    #indexer_cfgstr = ''.join((daids_hashid, flann_cfgstr, feat_cfgstr, fw_cfgstr))
    indexer_cfgstr = ''.join((daids_hashid, flann_cfgstr, featweight_cfgstr))

    # neighbor memory cache
    if use_cache and indexer_cfgstr in NEIGHBOR_CACHE:
        nnindexer = NEIGHBOR_CACHE[indexer_cfgstr]
    else:
        flann_cachedir = qreq_.ibs.get_flann_cachedir()
        # Grab the keypoints names and image ids before query time?
        flann_params =  qreq_.qparams.flann_params
        # Get annot descriptors to index
        vecs_list = qreq_.ibs.get_annot_vecs(daid_list)
        fgws_list = get_fgweights_hack(qreq_, daid_list)
        try:
            nnindexer = new_neighbor_index(
                daid_list, vecs_list, fgws_list, flann_params, flann_cachedir,
                cfgstr=indexer_cfgstr, use_params_hash=False, verbose=verbose)
        except Exception as ex:
            ut.printex(ex, True, msg_='cannot build inverted index',
                            key_list=['ibs.get_infostr()'])
            raise
        # Save inverted cache uuid mappings for
        min_reindex_thresh = qreq_.qparams.min_reindex_thresh
        if len(daid_list) > min_reindex_thresh:
            # let the multi-indexer know about any big caches we've made
            # multi-indexer
            uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_.ibs)
            with ut.shelf_open(uuid_map_fpath) as uuid_map:
                visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
                uuid_map[daids_hashid] = visual_uuid_list
        if len(NEIGHBOR_CACHE) > MAX_NEIGHBOR_CACHE_SIZE:
            NEIGHBOR_CACHE.clear()
        NEIGHBOR_CACHE[indexer_cfgstr] = nnindexer
    return nnindexer


def get_fgweights_hack(qreq_, daid_list):
    """
    hack to get  feature weights. returns None if feature weights are turned off
    in config settings
    """
    # <HACK:featweight>
    if qreq_.qparams.fg_weight != 0:
        fgws_list = qreq_.ibs.get_annot_fgweights(
            daid_list, qreq_=qreq_, ensure=True)
    else:
        fgws_list = None
    return fgws_list
    # </HACK:featweight>


@profile
def new_neighbor_index(aid_list, vecs_list, fgws_list=None, flann_params={},
                       flann_cachedir=None, cfgstr='',
                       use_cache=not NOCACHE_FLANN,
                       use_params_hash=False, verbose=True):
    """
    Args:
        aid_list (list):
        vecs_list (list):
        fgws_list (list):
        flann_params (dict):
        flann_cachedir (None):
        indexer_cfgstr (str):
        use_cache (bool):
        use_params_hash (bool):

    Returns:
        nnindexer
    """
    if verbose:
        print('[nnindexer] building NeighborIndex object')
    preptup = prepare_index_data(aid_list, vecs_list, fgws_list, verbose=True)
    (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = preptup
    # Dont hash rowids when given enough info in indexer_cfgstr
    flannkw = dict(cache_dir=flann_cachedir, cfgstr=cfgstr,
                   flann_params=flann_params, use_cache=use_cache,
                   use_params_hash=use_params_hash)
    cores = flann_params.get('cores', 0)
    # Build/Load the flann index
    flann = nntool.flann_cache(idx2_vec, verbose=verbose, **flannkw)
    nnindexer = NeighborIndex(ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx,
                              flann, cores, cfgstr)
    return nnindexer


def prepare_index_data(aid_list, vecs_list, fgws_list, verbose=True):
    _check_input(aid_list, vecs_list)
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
    preptup = (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx)
    return preptup


def _check_input(aid_list, vecs_list):
    assert len(aid_list) == len(vecs_list), 'invalid input. bad len'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


@six.add_metaclass(ut.ReloadingMetaclass)
class NeighborIndex(object):
    """
    wrapper class around flann
    stores flann index and data it needs to index into

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
    """

    def __init__(nnindexer, ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx,
                 flann, cores, cfgstr):
        nnindexer.ax2_aid  = ax2_aid   # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        nnindexer.idx2_fgw = idx2_fgw  # (M x 1) Descriptor forground weight
        nnindexer.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        nnindexer.flann    = flann     # Approximate search structure
        nnindexer.cfgstr   = cfgstr    # configuration id
        nnindexer.cores    = cores
        nnindexer.num_indexed = len(nnindexer.idx2_vec)
        if nnindexer.idx2_vec.dtype == hstypes.VEC_TYPE:
            nnindexer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
        else:
            assert False, 'NNindexer should get uint8s right now unless the algorithm has changed'

    def get_dtype(nnindexer):
        return nnindexer.idx2_vec.dtype

    #@profile
    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        Args:
            qfx2_vec : (N x D) an array of N, D-dimensional query vectors

            K: number of approximate nearest neighbors to find

        Returns: tuple of (qfx2_idx, qfx2_dist)
            qfx2_idx : (N x K) qfx2_idx[n][k] is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            qfx2_dist : (N x K) qfx2_dist[n][k] is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]
                        distance is normalized squared euclidean distance.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
            >>> result = str(qfx2_idx.shape) + ' ' + str(qfx2_dist.shape)
            >>> assert np.all(qfx2_dist < 1.0), 'distance should be less than 1'
            >>> print(result)
            (1257, 2) (1257, 2)

        Example2:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = np.empty((0, 128), dtype=nnindexer.get_dtype())
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
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
            # perform nearest neighbors
            (qfx2_idx, qfx2_dist) = nnindexer.flann.nn_index(
                qfx2_vec, K, checks=checks, cores=nnindexer.cores)
            # Ensure that distance returned are between 0 and 1
            qfx2_dist = qfx2_dist / (nnindexer.max_distance ** 2)
            #qfx2_dist = np.sqrt(qfx2_dist) / nnindexer.max_distance
        return (qfx2_idx, qfx2_dist)

    def empty_neighbors(nnindexer, nQfx, K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def add_points(nnindexer, new_aid_list, new_vecs_list, new_fgws_list):
        """
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_vecs(1)
        >>> new_vecs_list = ibs.get_annot_vecs(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K, checks=checks)
        >>> nnindexer.add_points(new_aid_list, new_vecs_list)
        >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K, checks=checks)
        >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = nnindexer.num_indexed_annots()
        nNew    = len(new_aid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNew)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        new_idx2_fgw = np.vstack(new_fgws_list)
        # Stack inverted information
        _ax2_aid = np.hstack((nnindexer.ax2_aid, new_aid_list))
        _idx2_ax = np.hstack((nnindexer.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((nnindexer.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((nnindexer.idx2_vec, new_idx2_vec))
        _idx2_fgw = np.hstack((nnindexer.idx2_fgw, new_idx2_fgw))
        nnindexer.ax2_aid  = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        nnindexer.idx2_fgw = _idx2_fgw
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        nnindexer.flann.add_points(new_idx2_vec)

    def num_indexed_vecs(nnindexer):
        return len(nnindexer.idx2_vec)

    def num_indexed_annots(nnindexer):
        return len(nnindexer.ax2_aid)

    def get_nn_vecs(nnindexer, qfx2_nnidx):
        """ gets matching vectors """
        return nnindexer.idx2_vec.take(qfx2_nnidx, axis=0)

    def get_nn_axs(nnindexer, qfx2_nnidx):
        """ gets matching internal annotation indicies """
        return nnindexer.idx2_ax.take(qfx2_nnidx)

    @profile
    def get_nn_aids(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_aid : (N x K) qfx2_fx[n][k] is the annotation id index of the
                                kth approximate nearest data vector

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots import pipeline
            >>> cfgdict = dict()
            >>> dbname = 'testdb1'
            >>> ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname, cfgdict=cfgdict)
            >>> nnindexer = qreq_.indexer
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(qreq_.get_internal_qaids()[0])
            >>> num_neighbors = 4
            >>> checks = 1024
            >>> (qfx2_nnidx, qfx2_dist) = nnindexer.knn(qfx2_vec, num_neighbors, checks)
            >>> qfx2_aid = nnindexer.get_nn_aids(qfx2_nnidx)
            >>> result = qfx2_aid.shape
            >>> print(result)
            (1257, 4)
        """
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nnidx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fx : (N x K) qfx2_fx[n][k] is the feature index (w.r.t the
                               source annotation) of the kth approximate
                               nearest data vector
        """
        #return nnindexer.idx2_fx[qfx2_nnidx]
        qfx2_fx = nnindexer.idx2_fx.take(qfx2_nnidx)
        return qfx2_fx

    def get_nn_fgws(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fgw : (N x K) qfx2_fgw[n][k] is the annotation id index of the
                                kth forground weight
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer(dbname='testdb1')
            >>> qfx2_nnidx = np.array([[0, 1, 2], [3, 4, 5]])
            >>> qfx2_fgw = nnindexer.get_nn_fgws(qfx2_nnidx)
        """
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nnidx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        if nnindexer.idx2_fgw is None:
            qfx2_fgw = np.ones(qfx2_nnidx.shape)
        else:
            qfx2_fgw = nnindexer.idx2_fgw.take(qfx2_nnidx)
        return qfx2_fgw


@profile
def invert_index(vecs_list, ax_list, verbose=ut.NOT_QUIET):
    """
    Aggregates descriptors of input annotations and returns inverted information
    """
    if ut.VERYVERBOSE:
        print('[hsnbrx] stacking descriptors from %d annotations' % len(ax_list))
    try:
        idx2_vec, idx2_ax, idx2_fx = nntool.invertable_stack(vecs_list, ax_list)
        assert idx2_vec.shape[0] == idx2_ax.shape[0]
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
    except MemoryError as ex:
        ut.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if verbose:
        print('[hsnbrx] stacked nVecs={nVecs} from nAnnots={nAnnots}'.format(
            nVecs=len(idx2_vec), nAnnots=len(ax_list)))
    return idx2_vec, idx2_ax, idx2_fx


def test_nnindexer(dbname='testdb1', with_indexer=True):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
    """
    import ibeis
    daid_list = [7, 8, 9, 10, 11]
    ibs = ibeis.opendb(db=dbname)
    qreq_ = ibs.new_query_request(daid_list, daid_list)
    if with_indexer:
        nnindexer = request_ibeis_nnindexer(qreq_)
    else:
        nnindexer = None
    return nnindexer, qreq_, ibs


def test_incremental_add(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-test_incremental_add

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = test_incremental_add(ibs)
        >>> print(result)
    """
    sample_aids = ibs.get_annot_rowid_sample()
    aids1 = sample_aids[::2]
    aids2 = sample_aids[0:5]
    aids3 = sample_aids[:-1]  # NOQA
    daid_list = aids1  # NOQA
    nnindexer1 = request_ibeis_nnindexer(ibs.new_query_request(aids1, aids1))  # NOQA
    nnindexer2 = request_ibeis_nnindexer(ibs.new_query_request(aids2, aids2))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(aids3)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(ibs)
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {key: set(val) for key, val in six.iteritems(uuid_map)}
    candidate_sets = candidate_uuids
    uncovered_items, covered_items_list, accepted_keys = ut.greedy_max_inden_setcover(candidate_sets, items)
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    nnindexer3 = request_ibeis_nnindexer(ibs.new_query_request(uncovered_aids, uncovered_aids))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(sample_aids)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(ibs)
    #contextlib.closing(shelve.open(uuid_map_fpath)) as uuid_map:
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {key: set(val) for key, val in six.iteritems(uuid_map)}
    candidate_sets = candidate_uuids
    uncovered_items, covered_items_list, accepted_keys = ut.greedy_max_inden_setcover(candidate_sets, items)
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))  # NOQA
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    uuid_map
    #uuid_map_fpath = join(flann_cachedir, 'uuid_map.shelf')
    #uuid_map = shelve.open(uuid_map_fpath)
    #uuid_map[daids_hashid] = visual_uuid_list
    #visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
    #visual_uuid_list
    #%timeit request_ibeis_nnindexer(qreq_, use_cache=False)
    #%timeit request_ibeis_nnindexer(qreq_, use_cache=True)

    #for uuids in uuid_set
    #    if


def subindexer_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment
    """
    import ibeis
    import utool as ut
    import pyflann
    import numpy as np
    import plottool as pt
    ibs = ibeis.opendb(db='PZ_Master0')
    daid_list = ibs.get_valid_aids()
    count_list = []
    time_list = []
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()
    for count in ut.ProgressIter(range(1, 301)):
        daids_ = daid_list[:]
        np.random.shuffle(daids_)
        daids = daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = pyflann.FLANN()
            flann.build_index(vecs, **flann_params)
        count_list.append(count)
        time_list.append(t.ellapsed)
    count_arr = np.array(count_list)
    time_arr = np.array(time_list)
    pt.plot2(count_arr, time_arr, marker='-', equal_aspect=False,
             x_label='num_annotations', y_label='FLANN build time')
    pt.update()


def subindexer_add_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment

    CommandLine:
        python -m ibeis.model.hots.multi_index --test-subindexer_add_time_experiment

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.multi_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = subindexer_add_time_experiment()
        >>> # verify results
        >>> print(result)
    """
    import ibeis
    import utool as ut
    import pyflann
    import numpy as np
    import plottool as pt
    #ibs = ibeis.opendb(db='PZ_Master0')
    ibs = ibeis.opendb(db='PZ_MTEST')
    daid_list = ibs.get_valid_aids()
    # Rebuild Part Part
    count_list = []
    time_list = []
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()
    max_num = min(301, len(daid_list))
    daids_ = daid_list[:]
    np.random.shuffle(daids_)
    for count in ut.ProgressIter(range(1, max_num)):
        daids_ = daid_list[:]
        np.random.shuffle(daids_)
        daids = daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = pyflann.FLANN()
            flann.build_index(vecs, **flann_params)
        count_list.append(count)
        time_list.append(t.ellapsed)
    count_arr = np.array(count_list)
    time_arr = np.array(time_list)
    pt.plot2(count_arr, time_arr, marker='-', equal_aspect=False,
             x_label='num_annotations', y_label='FLANN build time')
    # Add Part
    count_list2 = []
    time_list2 = []
    daids = daids_[0:count]
    vecs = np.vstack(ibs.get_annot_vecs(daids))
    flann = pyflann.FLANN()
    flann.build_index(vecs, **flann_params)
    for count in ut.ProgressIter(range(2, max_num)):
        daids_ = daid_list[:]
        np.random.shuffle(daids_)
        daids = daids_[count:count + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann.add_points(vecs)
        count_list2.append(count)
        time_list2.append(t.ellapsed)
    count_arr2 = np.array(count_list2)
    time_arr2 = np.array(time_list2)
    pt.plot2(count_arr2, time_arr2, marker='-', equal_aspect=False,
             x_label='num_annotations', y_label='FLANN add time')
    pt.update()


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.neighbor_index
        python -m ibeis.model.hots.neighbor_index --allexamples
        python -m ibeis.model.hots.neighbor_index --allexamples --noface --nosrc

        profiler.sh ibeis/model/hots/neighbor_index.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
