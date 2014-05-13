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
        qreq.qrids = []
        qreq.drids = []
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

    def set_rids(qreq, qrids, drids):
        qreq.qrids = qrids
        qreq.drids = drids

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def get_drids_uid(qreq):
        assert len(qreq.drids) > 0, 'QueryRequest not populated. len(drids)=0'
        drids_uid = utool.hashstr_arr(qreq.drids, '_drids')
        return drids_uid

    def get_qrids_uid(qreq):
        assert len(qreq.qrids) > 0, 'QueryRequest not populated. len(qrids)=0'
        qrids_uid = utool.hashstr_arr(qreq.qrids, '_qrids')
        return qrids_uid

    def get_uid_list(qreq, use_drids=True, use_qrids=False, **kwargs):
        uid_list = []
        if use_drids:
            uid_list.append(qreq.get_drids_uid())
        if use_qrids:
            uid_list.append(qreq.get_qrids_uid())
        uid_list.extend(qreq.cfg.get_uid_list(**kwargs))
        return uid_list

    def get_uid(qreq, **kwargs):
        return ''.join(qreq.get_uid_list(**kwargs))

    def get_internal_drids(qreq):
        """ These are not the users drids in vsone mode """
        drids = qreq.drids if qreq.vsmany else qreq.qrids
        return drids

    def get_internal_qrids(qreq):
        """ These are not the users qrids in vsone mode """
        qrids = qreq.qrids if qreq.vsmany else qreq.drids
        return qrids

    def get_ridfx_enum(qreq):
        ax2_rids = qreq.data_index.ax2_rid
        ax2_fxs = qreq.data_index.ax2_fx
        ridfx_enum = enumerate(izip(ax2_rids, ax2_fxs))
        return ridfx_enum
