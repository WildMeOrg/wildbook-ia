from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[qreq]', DEBUG=False)
# Standard
from itertools import izip
from os.path import join
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

    def set_rids(qreq, qrids, drids):
        qreq.qrids = qrids
        qreq.drids = drids

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def get_uid_list(qreq, use_drids=True, use_qrids=False, **kwargs):
        uid_list = []
        if use_drids:
            # Append a hash of the database rois
            assert len(qreq.drids) > 0, 'QueryRequest not populated. len(drids)=0'
            drids_uid = utool.hashstr_arr(qreq.drids, '_drids')
            uid_list.append(drids_uid)
        if use_qrids:
            # Append a hash of the query rois
            assert len(qreq.qrids) > 0, 'QueryRequest not populated. len(qrids)=0'
            qrids_uid = utool.hashstr_arr(qreq.qrids, '_qrids')
            uid_list.append(qrids_uid)
        uid_list.extend(qreq.cfg.get_uid_list(**kwargs))
        return uid_list

    def get_uid(qreq, **kwargs):
        return ''.join(qreq.get_uid_list(**kwargs))

    def get_bigcache_fpath(qreq, ibs):
        query_uid = qreq.get_uid(use_drids=True, use_qrids=True)
        dbname    = ibs.get_dbname()
        bigcache_fname = 'qrid2_qres_' + dbname + query_uid
        bigcache_dpath = ibs.bigcachedir
        bigcache_fpath = join(bigcache_dpath, bigcache_fname)
        return bigcache_fpath

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
