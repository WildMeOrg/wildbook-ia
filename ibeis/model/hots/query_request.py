"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.query_request))"
"""
from __future__ import absolute_import, division, print_function
from ibeis.model.hots import neighbor_index as hsnbrx
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[query_request]', DEBUG=False)


def get_ibeis_query_request(ibs, qaid_list, daid_list):
    """
    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> import ibeis
    >>> qaid_list = [1]
    >>> daid_list = [1, 2, 3, 4, 5]
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    >>> qreq = get_ibeis_query_request(ibs, qaid_list, daid_list)
    """
    if utool.NOT_QUIET:
        print(' --- Prep QueryRequest --- ')
    # Request params
    cfg = ibs.cfg.query_cfg
    # Request directories
    qresdir = ibs.get_qres_cachedir()
    # Lazy descriptors
    qparams = QueryParams(cfg)
    # Neighbor Indexer
    qreq = QueryRequest(qaid_list, daid_list, qparams, qresdir)
    return qreq


NEIGHBOR_CACHE = {}


def init_neighbor_indexer(qreq, ibs, flann_cachedir):
    """
    IBEIS interface into neighbor_index

    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> rowid_list = daid_list
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    >>> nnindexer = init_neighbor_indexer(ibs, daid_list)
    """
    global NEIGHBOR_CACHE
    indexer_cfgstr = qreq.get_indexer_cfgstr(ibs)
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
            flann_params = qreq.qparams.flann_params
            # Get annotation descriptors that will be searched
            rowid_list = qreq.get_internal_daids()
            vecs_list = ibs.get_annot_desc(rowid_list)
            nnindexer = hsnbrx.NeighborIndex(rowid_list, vecs_list, flann_params,
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
    def __init__(qreq, qaid_list, daid_list, qparams, qresdir):
        qreq.qaids = qaid_list
        qreq.daids = daid_list
        qreq.qparams = qparams
        qreq.qresdir = qresdir
        qreq.indexer = None
        qreq.internal_qvecs_list = None
        qreq.aid2_nid = None

    def rrr(qreq):
        from ibeis.model.hots import query_request as hsqreq
        hsqreq.rrr()
        print('reloading QueryRequest')
        utool.reload_class_methods(qreq, hsqreq.QueryRequest)

    def load_query_vectors(qreq, ibs):
        rowid_list = qreq.get_internal_qaids()
        vecs_list = ibs.get_annot_desc(rowid_list)
        qreq.internal_qvecs_list = vecs_list

    def load_indexer(qreq, ibs):
        flann_cachedir = ibs.get_flann_cachedir()
        indexer = init_neighbor_indexer(qreq, ibs, flann_cachedir)
        qreq.indexer = indexer

    def load_annot_nameids(qreq, ibs):
        aids = list(set(utool.chain(qreq.qaids, qreq.daids)))
        nids = ibs.get_annot_nids(aids)
        qreq.aid2_nid = dict(zip(aids, nids))

    def get_annot_nids(qreq, aids):
        return None

    def get_data_hashid(qreq, ibs=None):
        assert len(qreq.daids) > 0, 'QueryRequest not populated. len(daids)=0'
        if ibs is None:
            data_hashid = utool.hashstr_arr(qreq.daids, '_DAIDS')
        else:
            duuid_list    = ibs.get_annot_uuids(qreq.daids)
            data_hashid  = utool.hashstr_arr(duuid_list, '_DUUIDS')
        return data_hashid

    def get_query_hashid(qreq, ibs=None):
        assert len(qreq.qaids) > 0, 'QueryRequest not populated. len(qaids)=0'
        if ibs is None:
            query_hashid = utool.hashstr_arr(qreq.qaids, '_QAIDS')
        else:
            quuid_list    = ibs.get_annot_uuids(qreq.qaids)
            query_hashid  = utool.hashstr_arr(quuid_list, '_QUUIDS')
        return query_hashid

    def get_internal_daids(qreq):
        """ These are not the users daids in vsone mode """
        daids = qreq.daids if qreq.qparams.vsmany else qreq.qaids
        return daids

    def get_internal_qaids(qreq):
        """ These are not the users qaids in vsone mode """
        qaids = qreq.qaids if qreq.qparams.vsmany else qreq.daids
        return qaids

    def get_internal_qvecs(qreq):
        return qreq.internal_qvecs_list

    def get_indexer(qreq):
        return qreq.indexer

    def get_internal_data_hashid(qreq, ibs=None):
        if qreq.qparams.vsone:
            return qreq.get_data_hashid(ibs)
        else:
            return qreq.get_query_hashid(ibs)

    def get_indexer_cfgstr(qreq, ibs=None):
        daids_hashid = qreq.get_internal_data_hashid(ibs)
        flann_cfgstr = qreq.qparams.flann_cfgstr
        feat_cfgstr  = qreq.qparams.feat_cfgstr
        indexer_cfgstr = daids_hashid + flann_cfgstr + feat_cfgstr
        return indexer_cfgstr

    def get_cfgstr(qreq, ibs=None):
        daids_hashid = qreq.get_data_hashid(ibs)
        cfgstr = daids_hashid + qreq.qparams.query_cfgstr
        return cfgstr


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
        filt2_stwt         = {filt: cfg.filt_cfg.get_stw(filt) for filt in active_filter_list}
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
            if key not in ['qparams', 'cfg']:
                setattr(qparams, key, val)
