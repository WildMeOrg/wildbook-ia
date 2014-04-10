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
        qreq.qcids = []
        qreq.dcids = []
        qreq.data_index = None  # current index
        qreq.dftup2_index = {}  # cached indexes
        qreq.vsmany = False
        qreq.vsone = False

    def set_cids(qreq, qcids, dcids):
        qreq.qcids = qcids
        qreq.dcids = dcids

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def get_uid_list(qreq, *args, **kwargs):
        uid_list = qreq.cfg.get_uid_list(*args, **kwargs)
        if not 'noDCXS' in args:
            if len(qreq.dcids) == 0:
                raise Exception('QueryRequest not populated. len(dcids)=0')
            # In case you don't search the entire dataset
            dcids_uid = utool.hashstr_arr(qreq.dcids, '_dcids')
            uid_list += [dcids_uid]
        return uid_list

    def get_uid(qreq, *args, **kwargs):
        return ''.join(qreq.get_uid_list(*args, **kwargs))

    def get_query_uid(qreq, ibs, qcids):
        query_uid = qreq.get_uid()
        hs_uid    = ibs.get_db_name()
        qcids_uid  = utool.hashstr_arr(qcids, lbl='_qids')
        test_uid  = hs_uid + query_uid + qcids_uid
        return test_uid

    def get_internal_dcids(qreq):
        """ These are not the users dcids in vsone mode """
        dcids = qreq.dcids if qreq.vsmany else qreq.qcids
        return dcids

    def get_internal_qcids(qreq):
        """ These are not the users qcids in vsone mode """
        qcids = qreq.qcids if qreq.vsmany else qreq.dcids
        return qcids

    def get_cidfx_enum(qreq):
        ax2_cids = qreq.data_index.ax2_cid
        ax2_fxs = qreq.data_index.ax2_fx
        cidfx_enum = enumerate(izip(ax2_cids, ax2_fxs))
        return cidfx_enum
