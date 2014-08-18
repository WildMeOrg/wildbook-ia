"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.query_request))"
"""
from __future__ import absolute_import, division, print_function
# UTool
import utool
# VTool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[query_request]', DEBUG=False)


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
        score_method       = cfg.agg_cfg.score_method
        min_nInliers       = cfg.sv_cfg.min_nInliers
        nShortlist         = cfg.sv_cfg.nShortlist
        ori_thresh         = cfg.sv_cfg.ori_thresh
        prescore_method    = cfg.sv_cfg.prescore_method
        scale_thresh       = cfg.sv_cfg.scale_thresh
        use_chip_extent    = cfg.sv_cfg.use_chip_extent
        xy_thresh          = cfg.sv_cfg.xy_thresh

        # cfgstrs
        nn_cfgstr = cfg.nn_cfg.get_cfgstr()
        filt_cfgstr = cfg.filt_cfg.get_cfgstr()
        sv_cfgstr = cfg.sv_cfg.get_cfgstr()

        # Dynamically set members
        for key, val in locals().iteritems():
            if key not in ['qparams', 'cfg']:
                setattr(qparams, key, val)


def QueryRequest(object):
    def __init__(qreq, qaid_list, qvecs_list, indexer, qparams, qresdir, bigcache_dir):
        qreq.indexer = indexer
        qreq.qvecs_list = qvecs_list
        qreq.qaids = qaid_list
        qreq.daids = indexer
        qreq.qparams = qparams
        qreq.qresdir = qresdir
        qreq.bigcache_dir = bigcache_dir

    def get_internal_daids(qreq):
        """ These are not the users daids in vsone mode """
        daids = qreq.daids if qreq.vsmany else qreq.qaids
        return daids

    def get_internal_qaids(qreq):
        """ These are not the users qaids in vsone mode """
        qaids = qreq.qaids if qreq.vsmany else qreq.daids
        return qaids

    def get_daids_hashid(qreq):
        assert len(qreq.daids) > 0, 'QueryRequest not populated. len(daids)=0'
        daids_hashid = utool.hashstr_arr(qreq.daids, '_daids')
        return daids_hashid

    def get_qaids_hashid(qreq):
        assert len(qreq.qaids) > 0, 'QueryRequest not populated. len(qaids)=0'
        qaids_hashid = utool.hashstr_arr(qreq.qaids, '_qaids')
        return qaids_hashid
