from __future__ import division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[qreq]', DEBUG=False)
# Standard
from itertools import izip
# Scientific
import numpy as np
import utool

ID_DTYPE = np.int32  # id datatype
X_DTYPE  = np.int32  # indeX datatype

__OBJECT_BASE__ = object if not utool.get_flag('--debug') else utool.DynStrucct


class QueryRequest(__OBJECT_BASE__):
    # This will allow for a pipelining structure of requests and results
    def __init__(qreq):
        super(QueryRequest, qreq).__init__()
        qreq.cfg = None  # Query Config
        qreq._qcxs = []
        qreq._dcxs = []
        qreq._data_index = None  # current index
        qreq._dftup2_index = {}   # cached indexes
        qreq.query_uid = None
        qreq.featchip_uid = None
        qreq.vsmany = False
        qreq.vsone = False

    def set_cxs(qreq, qcxs, dcxs):
        qreq._qcxs = qcxs
        qreq._dcxs = dcxs

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def unload_data(qreq):
        # Data TODO: Separate this
        printDBG('[qreq] unload_data()')
        qreq._data_index  = None  # current index
        qreq._dftup2_index = {}  # cached indexes
        printDBG('[qreq] unload_data(success)')

    def get_uid_list(qreq, *args, **kwargs):
        uid_list = qreq.cfg.get_uid_list(*args, **kwargs)
        if not 'noDCXS' in args:
            if len(qreq._dcxs) == 0:
                raise Exception('QueryRequest has not been populated. len(dcxs)=0')
            # In case you don't search the entire dataset
            dcxs_uid = utool.hashstr_arr(qreq._dcxs, '_dcxs')
            uid_list += [dcxs_uid]
        return uid_list

    def get_uid(qreq, *args, **kwargs):
        return ''.join(qreq.get_uid_list(*args, **kwargs))

    def get_query_uid(qreq, ibs, qcxs):
        query_uid = qreq.get_uid()
        hs_uid    = ibs.get_db_name()
        qcxs_uid  = utool.hashstr_arr(qcxs, lbl='_qcxs')
        test_uid  = hs_uid + query_uid + qcxs_uid
        return test_uid

    def get_internal_dcxs(qreq):
        dcxs = qreq._dcxs if qreq.vsmany else qreq._qcxs
        return dcxs

    def get_internal_qcxs(qreq):
        dcxs = qreq._qcxs if qreq.vsmany else qreq._dcxs
        return dcxs

    def get_cxfx_enum(qreq):
        ax2_cxs = qreq._data_index.ax2_cx
        ax2_fxs = qreq._data_index.ax2_fx
        cidfx_enum = enumerate(izip(ax2_cxs, ax2_fxs))
        return cidfx_enum
