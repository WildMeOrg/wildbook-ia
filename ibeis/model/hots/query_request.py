"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.query_request))"
"""
from __future__ import absolute_import, division, print_function
from ibeis.model.hots import neighbor_index as hsnbrx
import six
# UTool
import utool
import numpy as np
import atexit
(print, print_, printDBG, rrr_, profile) = utool.inject(__name__, '[query_request]', DEBUG=False)


# cache for heavyweight nn structures.
# ensures that only one is in memory
NEIGHBOR_CACHE = {}


def rrr():
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()
    rrr_()


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()
    try:
        del NEIGHBOR_CACHE
    except NameError:
        pass


def new_ibeis_query_request(ibs, qaid_list, daid_list):
    """
    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> import ibeis
    >>> qaid_list = [1]
    >>> daid_list = [1, 2, 3, 4, 5]
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    >>> qreq_ = new_ibeis_query_request(ibs, qaid_list, daid_list)
    """
    if utool.NOT_QUIET:
        print(' --- New IBEIS QueryRequest --- ')
    cfg     = ibs.cfg.query_cfg
    qresdir = ibs.get_qres_cachedir()
    qparams = QueryParams(cfg)
    # Neighbor Indexer
    qreq_ = QueryRequest(qaid_list, daid_list, qparams, qresdir)
    return qreq_


def init_neighbor_indexer(qreq_, ibs, flann_cachedir):
    """
    IBEIS interface into neighbor_index

    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> aid_list = daid_list
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    >>> nnindexer = init_neighbor_indexer(ibs, daid_list)
    """
    global NEIGHBOR_CACHE
    indexer_cfgstr = qreq_.get_indexer_cfgstr(ibs)
    try:
        # neighbor cache
        if indexer_cfgstr in NEIGHBOR_CACHE:
            nnindexer = NEIGHBOR_CACHE[indexer_cfgstr]
            return nnindexer
        else:
            # Grab the keypoints names and image ids before query time
            #rx2_kpts = ibs.get_annot_kpts(daid_list)
            #rx2_gid  = ibs.get_annot_gids(daid_list)
            #rx2_nid  = ibs.get_annot_nids(daid_list)
            flann_params = qreq_.qparams.flann_params
            # Get annotation descriptors that will be searched
            aid_list = qreq_.get_internal_daids()
            vecs_list = ibs.get_annot_desc(aid_list)
            nnindexer = hsnbrx.NeighborIndex(aid_list, vecs_list, flann_params,
                                             flann_cachedir, indexer_cfgstr,
                                             hash_rowids=True,
                                             use_params_hash=False)
            if len(NEIGHBOR_CACHE) > 2:
                NEIGHBOR_CACHE.clear()
            NEIGHBOR_CACHE[indexer_cfgstr] = nnindexer
            return nnindexer
    except Exception as ex:
        utool.printex(ex, True, msg_='cannot build inverted index',
                        key_list=['ibs.get_infostr()'])
        raise


class QueryRequest(object):

    def rrr(qreq_):
        from ibeis.model.hots import query_request as hsqreq
        hsqreq.rrr()
        print('reloading QueryRequest_')
        utool.reload_class_methods(qreq_, hsqreq.QueryRequest)

    def __init__(qreq_, qaid_list, daid_list, qparams, qresdir):
        qreq_.qparams = qparams
        qreq_.qresdir = qresdir
        qreq_.internal_qaids = None
        qreq_.internal_daids = None
        qreq_.internal_qidx = None
        qreq_.internal_didx = None
        qreq_.indexer = None
        qreq_.internal_qvecs_list = None
        qreq_.internal_qkpts_list = None
        qreq_.internal_dkpts_list = None
        qreq_.internal_qgid_list  = None
        qreq_.internal_qnid_list  = None
        qreq_.aid2_nid = None
        qreq_.set_external_daids(daid_list)
        qreq_.set_external_qaids(qaid_list)

    # --- State Modification ---

    def set_external_daids(qreq_, daid_list):
        if qreq_.qparams.vsmany:
            qreq_.internal_daids = np.array(daid_list)
        else:
            qreq_.internal_qaids = np.array(daid_list)  # flip on vsone
        # Index the annotation ids for fast internal lookup
        qreq_.internal_didx = np.arange(len(daid_list))

    def set_external_qaids(qreq_, qaid_list):
        if qreq_.qparams.vsmany:
            qreq_.internal_qaids = np.array(qaid_list)
        else:
            qreq_.internal_daids = np.array(qaid_list)  # flip on vsone
        # Index the annotation ids for fast internal lookup
        qreq_.internal_qidx = np.arange(len(qaid_list))

    # --- Lazy Loading ---

    def load_oris(qreq_, ibs):
        if qreq_.idx2_oris is not None:
            return
        from vtool import keypoint as ktool
        qreq_.load_kpts(ibs)
        idx2_oris = ktool.get_oris(qreq_.idx2_kpts)
        assert len(idx2_oris) == len(qreq_.num_indexed_vecs())
        qreq_.idx2_oris = idx2_oris

    #def load_kpts(qreq_, ibs):
    #    if qreq_.idx2_kpts is not None:
    #        return
    #    aid_list = qreq_.indexer.aid_list
    #    kpts_list = qreq_.ibs.get_annot_kpts(aid_list)
    #    idx2_kpts = np.vstack(kpts_list)
    #    qreq_.idx2_kpts = idx2_kpts

    #def load_query_queryx(qreq_):
    #    qaids = qreq_.get_internal_qaids()
    #    qaid2_queryx = {aid: queryx for queryx, aid in enumerate(qaids)}
    #    qreq_.qaid2_queryx = qaid2_queryx

    #def load_data_datax(qreq_):
    #    daids = qreq_.get_internal_daids()
    #    daid2_datax = {aid: datax for datax, aid in enumerate(daids)}
    #    qreq_.daid2_datax = daid2_datax

    #def load_query_gids(qreq_, ibs):
    #    if qreq_.internal_qgid_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    gid_list = ibs.get_annot_gids(aid_list)
    #    qreq_.internal_qgid_list = gid_list

    #def load_query_nids(qreq_, ibs):
    #    if qreq_.internal_qnid_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    nid_list = ibs.get_annot_nids(aid_list)
    #    qreq_.internal_qnid_list = nid_list

    def load_query_vectors(qreq_, ibs):
        if qreq_.internal_qvecs_list is not None:
            return False
        aid_list = qreq_.get_internal_qaids()
        vecs_list = ibs.get_annot_desc(aid_list)
        qreq_.internal_qvecs_list = vecs_list

    def load_query_keypoints(qreq_, ibs):
        if qreq_.internal_qkpts_list is not None:
            return False
        aid_list = qreq_.get_internal_qaids()
        kpts_list = ibs.get_annot_kpts(aid_list)
        qreq_.internal_qkpts_list = kpts_list

    def load_data_keypoints(qreq_, ibs):
        if qreq_.internal_dkpts_list is not None:
            return False
        aid_list = qreq_.get_internal_daids()
        kpts_list = ibs.get_annot_kpts(aid_list)
        qreq_.internal_dkpts_list = kpts_list

    def load_indexer(qreq_, ibs):
        if qreq_.indexer is not None:
            return False
        flann_cachedir = ibs.get_flann_cachedir()
        indexer = init_neighbor_indexer(qreq_, ibs, flann_cachedir)
        qreq_.indexer = indexer

    def lazy_load(qreq_, ibs):
        qreq_.load_indexer(ibs)
        qreq_.load_query_vectors(ibs)
        qreq_.load_query_keypoints(ibs)
        qreq_.ibs = ibs  # HACK

    def load_annot_nameids(qreq_, ibs):
        aids = list(set(utool.chain(qreq_.qaids, qreq_.daids)))
        nids = ibs.get_annot_nids(aids)
        qreq_.aid2_nid = dict(zip(aids, nids))

    # --- Indexer Interface ----

    def knn(qreq_, qfx2_vec, K, checks=1028):
        return qreq_.indexer.knn(qfx2_vec, K, checks)

    def get_nn_kpts(qreq_, qfx2_nndx):
        return qreq_.idx2_kpts[qfx2_nndx]

    def get_nn_oris(qreq_, qfx2_nndx):
        return qreq_.idx2_oris[qfx2_nndx]

    #def get_indexed_rowids(qreq_):
    #    return qreq_.indexer.idx2_rowid

    def get_nn_aids(qreq_, qfx2_nndx):
        return qreq_.indexer.get_nn_aids(qfx2_nndx)

    def get_nn_featxs(qreq_, qfx2_nndx):
        return qreq_.indexer.get_nn_featxs(qfx2_nndx)

    # --- IBEISControl Transition ---

    def get_annot_nids(qreq_, aids):
        return qreq_.ibs.get_annot_nids(aids)

    def get_annot_gids(qreq_, aids):
        return qreq_.ibs.get_annot_gids(aids)

    def get_annot_kpts(qreq_, aids):
        return qreq_.ibs.get_annot_kpts(aids)

    def get_annot_chipsizes(qreq_, aids):
        return qreq_.ibs.get_annot_chipsizes(qreq_, aids)

    # --- Internal Interface ---

    def get_internal_qvecs(qreq_):
        return qreq_.internal_qvecs_list

    def get_internal_data_hashid(qreq_, ibs=None):
        if qreq_.qparams.vsone:
            return qreq_.get_data_hashid(ibs)
        else:
            return qreq_.get_query_hashid(ibs)

    def get_indexer_cfgstr(qreq_, ibs=None):
        daids_hashid = qreq_.get_internal_data_hashid(ibs)
        flann_cfgstr = qreq_.qparams.flann_cfgstr
        feat_cfgstr  = qreq_.qparams.feat_cfgstr
        indexer_cfgstr = daids_hashid + flann_cfgstr + feat_cfgstr
        return indexer_cfgstr

    def get_internal_daids(qreq_):
        """ For within pipeline use """
        return qreq_.internal_daids

    def get_internal_qaids(qreq_):
        """ For within pipeline use """
        return qreq_.internal_qaids

    # --- External Interface ---

    def get_external_daids(qreq_):
        """ These are the users daids in vsone mode """
        daids = qreq_.internal_daids if qreq_.qparams.vsmany else qreq_.internal_qaids
        return daids

    def get_external_qaids(qreq_):
        """ These are the users qaids in vsone mode """
        qaids = qreq_.internal_qaids if qreq_.qparams.vsmany else qreq_.internal_daids
        return qaids

    def get_data_hashid(qreq_, ibs=None):
        daids = qreq_.get_external_daids()
        assert len(daids) > 0, 'QueryRequest not populated. len(daids)=0'
        if ibs is None:
            data_hashid = utool.hashstr_arr(daids, '_DAIDS')
        else:
            duuid_list    = ibs.get_annot_uuids(daids)
            data_hashid  = utool.hashstr_arr(duuid_list, '_DUUIDS')
        return data_hashid

    def get_query_hashid(qreq_, ibs=None):
        qaids = qreq_.get_external_qaids()
        assert len(qaids) > 0, 'QueryRequest not populated. len(qaids)=0'
        if ibs is None:
            query_hashid = utool.hashstr_arr(qaids, '_QAIDS')
        else:
            quuid_list    = ibs.get_annot_uuids(qaids)
            query_hashid  = utool.hashstr_arr(quuid_list, '_QUUIDS')
        return query_hashid

    def get_cfgstr(qreq_, ibs=None):
        daids_hashid = qreq_.get_data_hashid(ibs)
        cfgstr = daids_hashid + qreq_.qparams.query_cfgstr
        return cfgstr

    def get_qresdir(qreq):
        return qreq.qresdir


class QueryParams(object):
    def __init__(qparams, cfg):
        K                  = cfg.nn_cfg.K
        Knorm              = cfg.nn_cfg.Knorm
        checks             = cfg.nn_cfg.checks
        normalizer_rule    = cfg.nn_cfg.normalizer_rule
        Krecip             = cfg.filt_cfg.Krecip
        can_match_sameimg  = cfg.filt_cfg.can_match_sameimg
        can_match_samename = cfg.filt_cfg.can_match_samename
        filt_on            = cfg.filt_cfg.filt_on
        gravity_weighting  = cfg.filt_cfg.gravity_weighting
        active_filter_list = cfg.filt_cfg.get_active_filters()
        #active_filter_list = cfg.filt_cfg._valid_filters
        filt2_stw          = {filt: cfg.filt_cfg.get_stw(filt) for filt in active_filter_list}
        # Correct dumb Pref bugs
        for key, val in six.iteritems(filt2_stw):
            #print(val)
            if val[1] == 'None':
                val[1] = None
            if val[1] is not None and not isinstance(val[1], (float, int)):
                val[1] = float(val[1])
        isWeighted         = cfg.agg_cfg.isWeighted
        max_alts           = cfg.agg_cfg.max_alts
        query_type         = cfg.agg_cfg.query_type
        vsmany             = query_type == 'vsmany'
        vsone              = query_type == 'vsone'
        score_method       = cfg.agg_cfg.score_method
        min_nInliers       = cfg.sv_cfg.min_nInliers
        nShortlist         = cfg.sv_cfg.nShortlist
        ori_thresh         = cfg.sv_cfg.ori_thresh
        prescore_method    = cfg.sv_cfg.prescore_method
        scale_thresh       = cfg.sv_cfg.scale_thresh
        use_chip_extent    = cfg.sv_cfg.use_chip_extent
        xy_thresh          = cfg.sv_cfg.xy_thresh
        sv_on              = cfg.sv_cfg.sv_on
        flann_params       = cfg.flann_cfg.get_dict_args()

        # cfgstrs
        feat_cfgstr = cfg._feat_cfg.get_cfgstr()
        nn_cfgstr = cfg.nn_cfg.get_cfgstr()
        filt_cfgstr = cfg.filt_cfg.get_cfgstr()
        sv_cfgstr = cfg.sv_cfg.get_cfgstr()
        flann_cfgstr = cfg.flann_cfg.get_cfgstr()
        query_cfgstr = cfg.get_cfgstr()

        # Dynamically set members
        for key, val in locals().iteritems():
            if key not in ['qparams', 'cfg', 'filt', 'key', 'val']:
                setattr(qparams, key, val)
