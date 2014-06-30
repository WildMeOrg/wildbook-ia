from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[qreq]', DEBUG=False)
# Standard
from itertools import izip
# Scientific
import utool

__REQUEST_BASE__ = utool.DynStruct if utool.get_flag('--debug') else object


class QueryRequest(__REQUEST_BASE__):
    # This will allow for a pipelining structure of requests and results
    def __init__(qreq, qresdir, bigcachedir):
        super(QueryRequest, qreq).__init__()
        qreq.cfg = None  # Query config pointer
        qreq.qaids = []
        qreq.daids = []
        qreq.data_index = None  # current index
        qreq.dftup2_index = {}  # cached indexes
        qreq.vsmany = False
        qreq.vsone  = False
        qreq.qresdir = qresdir  # Where to cache individual results
        qreq.bigcachedir = bigcachedir  # Where to cache large results

    #def __del__(qreq):
    #    for key in qreq.dftup2_index.keys():
    #         del qreq.dftup2_index[key]
    #    qreq.data_index = None

    def set_aids(qreq, qaids, daids):
        qreq.qaids = qaids
        qreq.daids = daids

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def get_daids_hashid(qreq):
        assert len(qreq.daids) > 0, 'QueryRequest not populated. len(daids)=0'
        daids_hashid = utool.hashstr_arr(qreq.daids, '_daids')
        return daids_hashid

    def get_qaids_hashid(qreq):
        assert len(qreq.qaids) > 0, 'QueryRequest not populated. len(qaids)=0'
        qaids_hashid = utool.hashstr_arr(qreq.qaids, '_qaids')
        return qaids_hashid

    def get_cfgstr_list(qreq, use_daids=True, use_qaids=False, **kwargs):
        cfgstr_list = []
        if use_daids:
            cfgstr_list.append(qreq.get_daids_hashid())
        if use_qaids:
            cfgstr_list.append(qreq.get_qaids_cfgstr())
        cfgstr_list.extend(qreq.cfg.get_cfgstr_list(**kwargs))
        return cfgstr_list

    def get_cfgstr(qreq, **kwargs):
        return ''.join(qreq.get_cfgstr_list(**kwargs))

    def get_internal_daids(qreq):
        """ These are not the users daids in vsone mode """
        daids = qreq.daids if qreq.vsmany else qreq.qaids
        return daids

    def get_internal_qaids(qreq):
        """ These are not the users qaids in vsone mode """
        qaids = qreq.qaids if qreq.vsmany else qreq.daids
        return qaids

    def get_aidfx_enum(qreq):
        ax2_aids = qreq.data_index.ax2_aid
        ax2_fxs = qreq.data_index.ax2_fx
        aidfx_enum = enumerate(izip(ax2_aids, ax2_fxs))
        return aidfx_enum
