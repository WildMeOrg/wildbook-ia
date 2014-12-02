from __future__ import absolute_import, division, print_function
from ibeis.model.hots import neighbor_index, score_normalization
from ibeis.model import Config
import six
import utool
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[qreq]')


def get_test_qreq():
    import ibeis
    qaid_list = [1]
    daid_list = [1, 2, 3, 4, 5]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, qaid_list, daid_list)
    return qreq_, ibs


def apply_species_with_detector_hack(ibs, cfgdict):
    """ HACK """
    from ibeis import constants
    unique_species = ibs.get_database_species()
    # turn off featureweights when not absolutely sure they are ok to us,)
    species_with_detectors = (
        constants.Species.ZEB_GREVY,
        constants.Species.ZEB_PLAIN,
    )
    candetect = (
        len(unique_species) == 1 and
        unique_species[0] in species_with_detectors
    )
    if not candetect:
        print('HACKING FG_WEIGHT OFF (database species is not supported)')
        if len(unique_species) != 1:
            print('  * len(unique_species) = %r' % len(unique_species))
        else:
            print('  * unique_species = %r' % (unique_species,))
        print('  * valid species = %r' % (species_with_detectors,))
        #cfg._featweight_cfg.featweight_on = 'ERR'
        cfgdict['featweight_on'] = 'ERR'
    return unique_species


def new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict=None):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.query_request import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_MTEST')
        >>> qaids = [1]
        >>> daids = [1, 2, 3, 4, 5]
        >>> cfgdict = {'sv_on': False, 'fg_weight': 1.0, 'featweight_on': True}
        >>> qreq_ = new_ibeis_query_request(ibs, qaids, daids, cfgdict=cfgdict)
        >>> print(qreq_.qparams.query_cfgstr)
        >>> assert qreq_.qparams.fg_weight == 1.0, (
        ...    'qreq_.qparams.fg_weight = %r ' % qreq_.qparams.fg_weight)
        >>> assert qreq_.qparams.sv_on is False, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
    """
    if utool.NOT_QUIET:
        print(' --- New IBEIS QRequest --- ')
    cfg     = ibs.cfg.query_cfg
    qresdir = ibs.get_qres_cachedir()
    if cfgdict is None:
        cfgdict = {}
    cfgdict = cfgdict.copy()
    # <HACK>
    unique_species = apply_species_with_detector_hack(ibs, cfgdict)
    # </HACK>
    qparams = QueryParams(cfg, cfgdict)
    quuid_list = ibs.get_annot_uuids(qaid_list)
    duuid_list = ibs.get_annot_uuids(daid_list)
    qreq_ = QueryRequest(qaid_list, quuid_list,
                         daid_list, duuid_list,
                         qparams, qresdir)
    if utool.NOT_QUIET:
        print(' * query_cfgstr = %s' % (qreq_.qparams.query_cfgstr,))
    qreq_.unique_species = unique_species  # HACK
    return qreq_


def qreq_shallow_copy(qreq_, qx=None, dx=None):
    """
    Creates a copy of qreq with the same qparams object and a subset of the qx
    and dx objects.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.query_request import *  # NOQA
        >>> import ibeis
        >>> qreq_, ibs = get_test_qreq()
        >>> qreq2_ = qreq_shallow_copy(qreq_, 0)
    """
    qaid_list  = qreq_.get_external_qaids()
    quuid_list = qreq_.get_external_quuids()
    daid_list  = qreq_.get_external_daids()
    duuid_list = qreq_.get_external_duuids()
    #[qx:qx + 1]
    #[qx:qx + 1]
    qaid_list  =  qaid_list if qx is None else  qaid_list[qx:qx + 1]
    quuid_list = quuid_list if qx is None else quuid_list[qx:qx + 1]
    daid_list  =  daid_list if dx is None else  daid_list[dx:dx + 1]
    duuid_list = duuid_list if dx is None else duuid_list[dx:dx + 1]
    qreq_copy  = QueryRequest(qaid_list, quuid_list, daid_list, duuid_list, qreq_.qparams, qreq_.qresdir)
    qreq_copy.unique_species = qreq_.unique_species  # HACK
    qreq_copy.ibs = qreq_.ibs
    return qreq_copy


@six.add_metaclass(utool.ReloadingMetaclass)
class QueryRequest(object):
    def __init__(qreq_, qaid_list, quuid_list, daid_list, duuid_list, qparams, qresdir):
        # Reminder:
        # lists and other objects are functionally equivalent to pointers
        qreq_.unique_species = None  # num categories
        qreq_.internal_qspeciesid_list = None  # category species id label list
        qreq_.internal_qnid_list = None  # individual name id label list
        qreq_.internal_qaids = None
        qreq_.internal_daids = None
        qreq_.internal_qidx  = None
        qreq_.internal_didx  = None
        qreq_.internal_qvecs_list = None
        qreq_.internal_qkpts_list = None
        qreq_.internal_dkpts_list = None
        qreq_.internal_qgid_list  = None
        qreq_.internal_qnid_list  = None
        # Handle to parent IBEIS Controller
        qreq_.ibs = None
        # The nearest neighbor mechanism
        qreq_.indexer = None
        # The scoring normalization mechanism
        qreq_.normalizer = None
        # DEPRICATE?
        qreq_.aid2_nid = None
        qreq_.hasloaded = False
        qreq_.internal_quuids = None
        qreq_.internal_duuids = None

        # Set values
        qreq_.qparams = qparams   # Parameters relating to pipeline execution
        qreq_.qresdir = qresdir
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

    def get_unique_species(qreq_):
        return qreq_.unique_species

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

    def get_annot_name_rowids(qreq_, aids):
        return qreq_.ibs.get_annot_name_rowids(aids)

    def get_annot_gids(qreq_, aids):
        assert qreq_.ibs is not qreq_
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
    #    nid_list = ibs.get_annot_name_rowids(aid_list)
    #    qreq_.internal_qnid_list = nid_list

    def load_query_vectors(qreq_, ibs):
        if qreq_.internal_qvecs_list is not None:
            return False
        aid_list = qreq_.get_internal_qaids()
        vecs_list = ibs.get_annot_vecs(aid_list)
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
        indexer = neighbor_index.new_ibeis_nnindexer(ibs, qreq_)
        qreq_.indexer = indexer

    def load_score_normalizer(qreq_, ibs):
        if qreq_.normalizer is not None:
            return False
        normalizer = score_normalization.request_ibeis_normalizer(ibs, qreq_)
        qreq_.normalizer = normalizer

    def lazy_load(qreq_, ibs):
        print('[qreq] lazy loading')
        qreq_.hasloaded = True
        qreq_.ibs = ibs  # HACK
        if qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
            qreq_.load_indexer(ibs)
            # FIXME: not sure if this is even used
            qreq_.load_query_vectors(ibs)
            qreq_.load_query_keypoints(ibs)
        if qreq_.qparams.fg_weight != 0:
            # Hacky way to ensure fgweights exist
            #ibs.get_annot_fgweights(qreq_.get_internal_daids())
            ibs.get_annot_fgweights(qreq_.get_internal_qaids(), ensure=True)
        if qreq_.qparams.score_normalization:
            qreq_.load_score_normalizer(ibs)

    def load_annot_nameids(qreq_, ibs):
        aids = list(set(utool.chain(qreq_.qaids, qreq_.daids)))
        nids = ibs.get_annot_name_rowids(aids)
        qreq_.aid2_nid = dict(zip(aids, nids))

    def assert_self(qreq_, ibs):
        print('[qreq] ASSERT SELF')
        qaids    = qreq_.get_external_qaids()
        qauuids  = qreq_.get_external_quuids()
        daids    = qreq_.get_external_daids()
        dauuids  = qreq_.get_external_duuids()
        _qaids   = qreq_.get_internal_qaids()
        _qauuids = qreq_.get_internal_quuids()
        _daids   = qreq_.get_internal_daids()
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


def test_cfg_deepcopy():
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.query_request import *  # NOQA
        >>> result = test_cfg_deepcopy()
        >>> print(result)
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    cfg1 = ibs.cfg.query_cfg
    cfg2 = cfg1.deepcopy()
    cfg3 = cfg2
    assert cfg1.get_cfgstr() == cfg2.get_cfgstr()
    assert cfg2.sv_cfg is not cfg1.sv_cfg
    assert cfg3.sv_cfg is cfg2.sv_cfg
    cfg2.update_query_cfg(sv_on=False)
    assert cfg1.get_cfgstr() != cfg2.get_cfgstr()
    assert cfg2.get_cfgstr() == cfg3.get_cfgstr()


class QueryParams(object):
    """
    Structure to store static query pipeline parameters

    CommandLine:
        python -m ibeis.model.hots.query_request --test-QueryParams

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import query_request
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> cfg = ibs.cfg.query_cfg
        >>> #cfg.pipeline_root = 'asmk'
        >>> cfgdict = {'pipeline_root': 'asmk', 'sv_on': False,
        ...            'fg_weight': 1.0, 'featweight_on': True}
        >>> qparams = query_request.QueryParams(cfg, cfgdict)
        >>> assert qparams.fg_weight == 1.0
        >>> assert qparams.pipeline_root == 'smk'
        >>> assert qparams.featweight_on is True
        >>> result = qparams.query_cfgstr
        >>> print(')_\n'.join(result.split(')_')))
        _smk_SMK(agg=True,t=0.0,a=3.0,idf)_
        VocabAssign(nAssign=10,a=1.2,s=None,eqw=T)_
        VocabTrain(nWords=8000,init=akmeans++,nIters=128,taids=all)_
        SV(OFF)_
        FEATWEIGHT(ON,uselabel,rf)_
        FEAT(hesaff+sift_)_
        CHIP(sz450)
    """

    def __init__(qparams, cfg, cfgdict=None):
        """
        Args:
            cfg (QueryConfig): query_config
            cfgdict (dict or None): dictionary to update cfg with
        """
        # Ensures that at least everything exits
        # pares nested config structure into this flat one
        if cfgdict is not None:
            cfg = cfg.deepcopy()
            cfg.update_query_cfg(**cfgdict)
        param_list = Config.parse_config_items(cfg)
        seen_ = set()
        for key, val in param_list:
            if key not in seen_:
                seen_.add(key)
            else:
                raise AssertionError('Configs have duplicate names: %r' % key)
            setattr(qparams, key, val)
        del seen_, key, val, param_list

        # Then we explicitly add some items as well that might not be explicit
        # in the configs.
        pipeline_root      = cfg.pipeline_root
        # Params not explicitly represented in Config objects
        ###
        flann_params       = cfg.flann_cfg.get_dict_args()
        vsmany             = pipeline_root == 'vsmany'
        vsone              = pipeline_root == 'vsone'
        ###
        active_filter_list = cfg.filt_cfg.get_active_filters()
        filt2_stw          = {filt: cfg.filt_cfg.get_stw(filt)
                              for filt in active_filter_list}
        # Correct dumb Pref bugs
        for key, val in six.iteritems(filt2_stw):
            if val[1] == 'None':
                val[1] = None
            if val[1] is not None and not isinstance(val[1], (float, int)):
                val[1] = float(val[1])
        ####

        # cfgstrs
        featweight_cfgstr = cfg._featweight_cfg.get_cfgstr()
        feat_cfgstr  = cfg._featweight_cfg._feat_cfg.get_cfgstr()
        nn_cfgstr    = cfg.nn_cfg.get_cfgstr()
        filt_cfgstr  = cfg.filt_cfg.get_cfgstr()
        sv_cfgstr    = cfg.sv_cfg.get_cfgstr()
        flann_cfgstr = cfg.flann_cfg.get_cfgstr()
        query_cfgstr = cfg.get_cfgstr()
        vocabtrain_cfgstr = cfg.smk_cfg.vocabtrain_cfg.get_cfgstr()

        # Dynamically set members
        for key, val in locals().iteritems():
            if key not in ['qparams', 'cfg', 'filt', 'key', 'val']:
                setattr(qparams, key, val)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.query_request
        python -m ibeis.model.hots.query_request --allexamples
        python -m ibeis.model.hots.query_request --allexamples --noface --nosrc
        python -m ibeis.model.hots.query_request --test-QueryParams
        profiler.sh -m ibeis.model.hots.query_request --test-QueryParams
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
