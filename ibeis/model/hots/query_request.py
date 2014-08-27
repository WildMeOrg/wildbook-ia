"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.query_request))"
"""
from __future__ import absolute_import, division, print_function
from ibeis.model.hots import neighbor_index as hsnbrx
import six
# UTool
import utool
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[query_request]', DEBUG=False)


def get_test_qreq():
    import ibeis
    qaid_list = [1]
    daid_list = [1, 2, 3, 4, 5]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, qaid_list, daid_list)
    return qreq_, ibs


def new_ibeis_query_request(ibs, qaid_list, daid_list):
    """
    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> qreq_, ibs = get_test_qreq()   #doctest: +ELLIPSIS
    """
    if utool.NOT_QUIET:
        print(' --- New IBEIS QRequest --- ')
    cfg     = ibs.cfg.query_cfg
    qresdir = ibs.get_qres_cachedir()
    qparams = QueryParams(cfg)
    # Neighbor Indexer
    quuid_list = ibs.get_annot_uuids(qaid_list)
    duuid_list = ibs.get_annot_uuids(daid_list)
    qreq_ = QueryRequest(qaid_list, quuid_list,
                         daid_list, duuid_list,
                         qparams, qresdir)
    return qreq_


@six.add_metaclass(utool.ReloadingMetaclass)
class QueryRequest(object):
    def __init__(qreq_,
                 qaid_list, quuid_list,
                 daid_list, duuid_list,
                 qparams, qresdir):
        qreq_.qparams = qparams
        qreq_.qresdir = qresdir
        qreq_.internal_qaids = None
        qreq_.internal_daids = None
        qreq_.internal_quuids = None
        qreq_.internal_duuids = None
        qreq_.internal_qidx = None
        qreq_.internal_didx = None
        qreq_.indexer = None
        qreq_.internal_qvecs_list = None
        qreq_.internal_qkpts_list = None
        qreq_.internal_dkpts_list = None
        qreq_.internal_qgid_list  = None
        qreq_.internal_qnid_list  = None
        qreq_.aid2_nid = None
        qreq_.set_external_daids(daid_list, duuid_list)
        qreq_.set_external_qaids(qaid_list, quuid_list)

    # --- State Modification ---

    def set_external_daids(qreq_, daid_list, duuid_list):
        assert len(daid_list) == len(duuid_list), 'inconsistent external daids'
        if qreq_.qparams.vsmany:
            qreq_.set_internal_daids(daid_list, duuid_list)
        else:
            qreq_.set_internal_qaids(daid_list, duuid_list)  # flip on vsone

    def set_external_qaids(qreq_, qaid_list, quuid_list):
        assert len(qaid_list) == len(quuid_list), 'inconsistent internal qaids'
        if qreq_.qparams.vsmany:
            qreq_.set_internal_qaids(qaid_list, quuid_list)
        else:
            qreq_.set_internal_daids(qaid_list, quuid_list)  # flip on vsone

    def set_internal_daids(qreq_, daid_list, duuid_list):
        assert len(daid_list) == len(duuid_list), 'inconsistent internal daids'
        qreq_.internal_daids = np.array(daid_list)
        qreq_.internal_duuids = np.array(duuid_list)
        # Index the annotation ids for fast internal lookup
        #qreq_.internal_didx = np.arange(len(daid_list))

    def set_internal_qaids(qreq_, qaid_list, quuid_list):
        assert len(qaid_list) == len(quuid_list), 'inconsistent internal qaids'
        qreq_.internal_qaids = np.array(qaid_list)
        qreq_.internal_quuids = np.array(quuid_list)
        # Index the annotation ids for fast internal lookup
        #qreq_.internal_qidx = np.arange(len(qaid_list))

    # --- Internal Interface ---
    # For within pipeline use only

    def get_internal_qvecs(qreq_):
        return qreq_.internal_qvecs_list

    def get_internal_data_hashid(qreq_, ibs=None):
        if qreq_.qparams.vsone:
            return qreq_.get_data_hashid(ibs)
        else:
            return qreq_.get_query_hashid(ibs)

    def get_internal_daids(qreq_):
        return qreq_.internal_daids

    def get_internal_qaids(qreq_):
        return qreq_.internal_qaids

    def get_internal_duuids(qreq_):
        return qreq_.internal_duuids

    def get_internal_quuids(qreq_):
        return qreq_.internal_quuids

    # --- External Interface ---

    def get_external_daids(qreq_):
        """ These are the users daids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_daids()
        # Flip on vsone
        return qreq_.get_internal_qaids()

    def get_external_qaids(qreq_):
        """ These are the users qaids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_qaids()
        # Flip on vsone
        return qreq_.get_internal_daids()

    def get_external_quuids(qreq_):
        """ These are the users qauuids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_quuids()
        # Flip on vsone
        return qreq_.get_internal_duuids()

    def get_external_duuids(qreq_):
        """ These are the users qauuids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_duuids()
        # Flip on vsone
        return qreq_.get_internal_quuids()

    def get_data_hashid(qreq_, ibs=None):
        daids = qreq_.get_external_daids()
        assert len(daids) > 0, 'QRequest not populated. len(daids)=0'
        if ibs is None:
            data_hashid = utool.hashstr_arr(daids, '_DAIDS')
        else:
            data_hashid = ibs.get_annot_uuid_hashid(daids, '_DUUIDS')
        return data_hashid

    def get_query_hashid(qreq_, ibs=None):
        qaids = qreq_.get_external_qaids()
        assert len(qaids) > 0, 'QRequest not populated. len(qaids)=0'
        if ibs is None:
            query_hashid = utool.hashstr_arr(qaids, '_QAIDS')
        else:
            query_hashid = ibs.get_annot_uuid_hashid(qaids, '_QUUIDS')
        return query_hashid

    def get_cfgstr(qreq_, ibs=None):
        daids_hashid = qreq_.get_data_hashid(ibs)
        cfgstr = daids_hashid + qreq_.qparams.query_cfgstr
        return cfgstr

    def get_qresdir(qreq_):
        return qreq_.qresdir

    # --- IBEISControl Transition ---

    def get_annot_nids(qreq_, aids):
        return qreq_.ibs.get_annot_nids(aids)

    def get_annot_gids(qreq_, aids):
        return qreq_.ibs.get_annot_gids(aids)

    def get_annot_kpts(qreq_, aids):
        return qreq_.ibs.get_annot_kpts(aids)

    def get_annot_chipsizes(qreq_, aids):
        return qreq_.ibs.get_annot_chipsizes(qreq_, aids)

    # --- Lazy Loading ---

    #def load_oris(qreq_, ibs):
    #    if qreq_.idx2_oris is not None:
    #        return
    #    from vtool import keypoint as ktool
    #    qreq_.load_kpts(ibs)
    #    idx2_oris = ktool.get_oris(qreq_.idx2_kpts)
    #    assert len(idx2_oris) == len(qreq_.num_indexed_vecs())
    #    qreq_.idx2_oris = idx2_oris

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
        indexer = hsnbrx.new_ibeis_nnindexer(ibs, qreq_.get_internal_daids())
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

    def assert_self(qreq_, ibs):
        qaids = qreq_.get_external_qaids()
        qauuids = qreq_.get_external_quuids()
        daids = qreq_.get_external_daids()
        dauuids = qreq_.get_external_duuids()
        _qaids = qreq_.get_internal_qaids()
        _qauuids = qreq_.get_internal_quuids()
        _daids = qreq_.get_internal_daids()
        _dauuids = qreq_.get_internal_duuids()
        def assert_uuids(aids, uuids):
            if utool.NOT_QUIET:
                print('[qreq_] asserting %s ids' % len(aids))
            assert len(aids) == len(uuids)
            assert all([u1 == u2 for u1, u2 in
                        zip(ibs.get_annot_uuids(aids), uuids)])
        assert_uuids(qaids, qauuids)
        assert_uuids(daids, dauuids)
        assert_uuids(_qaids, _qauuids)
        assert_uuids(_daids, _dauuids)


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
