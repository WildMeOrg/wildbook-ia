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
    def __init__(qreq):
        super(QueryRequest, qreq).__init__()
        qreq.cfg = None  # Query config pointer
        qreq.qrids = []
        qreq.drids = []
        qreq.data_index = None  # current index
        qreq.dftup2_index = {}  # cached indexes
        qreq.vsmany = False
        qreq.vsone = False

    def set_rids(qreq, qrids, drids):
        qreq.qrids = qrids
        qreq.drids = drids

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def get_uid_list(qreq, *args, **kwargs):
        uid_list = qreq.cfg.get_uid_list(*args, **kwargs)
        if not 'noDCXS' in args:
            if len(qreq.drids) == 0:
                raise Exception('QueryRequest not populated. len(drids)=0')
            # In case you don't search the entire dataset
            drids_uid = utool.hashstr_arr(qreq.drids, '_drids')
            uid_list += [drids_uid]
        return uid_list

    def get_uid(qreq, *args, **kwargs):
        return ''.join(qreq.get_uid_list(*args, **kwargs))

    def get_query_uid(qreq, ibs, qrids):
        query_uid = qreq.get_uid()
        hs_uid    = ibs.get_db_name()
        qrids_uid  = utool.hashstr_arr(qrids, lbl='_qids')
        test_uid  = hs_uid + query_uid + qrids_uid
        return test_uid

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
