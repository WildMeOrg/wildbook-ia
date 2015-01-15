from __future__ import absolute_import, division, print_function
from ibeis.model.hots import neighbor_index
from ibeis.model.hots import multi_index
from ibeis.model.hots import score_normalization
from ibeis.model.hots import distinctiveness_normalizer
from ibeis.model import Config
import vtool as vt
import copy
import six
import utool as ut
import numpy as np
from ibeis.model.hots import hots_query_result
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[qreq]')


@profile
def new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict=None,
                            verbose=ut.NOT_QUIET, unique_species=None):
    """
    ibeis entry point to create a new query request object

    CommandLine:
        python -m ibeis.model.hots.query_request --test-new_ibeis_query_request

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.query_request import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_MTEST')
        >>> qaids = [1]
        >>> daids = [1, 2, 3, 4, 5]
        >>> cfgdict = {'sv_on': False, 'fg_weight': 1.0, 'featweight_on': True}
        >>> # Execute test
        >>> qreq_ = new_ibeis_query_request(ibs, qaids, daids, cfgdict=cfgdict)
        >>> # Check Results
        >>> print(qreq_.qparams.query_cfgstr)
        >>> assert qreq_.qparams.fg_weight == 1.0, (
        ...    'qreq_.qparams.fg_weight = %r ' % qreq_.qparams.fg_weight)
        >>> assert qreq_.qparams.sv_on is False, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
        >>> datahashid = qreq_.get_data_hashid()
        >>> dbname = ibs.get_dbname()
        >>> result = dbname + datahashid
        >>> print(result)
        PZ_MTEST_DSUUIDS((5)q87ho9a0@9s02imh)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.query_request import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='NAUT_test')
        >>> qaids = [1]
        >>> daids = [1, 2, 3, 4, 5]
        >>> cfgdict = {'sv_on': True, 'fg_weight': 1.0, 'featweight_on': True}
        >>> # Execute test
        >>> qreq_ = new_ibeis_query_request(ibs, qaids, daids, cfgdict=cfgdict)
        >>> # Check Results.
        >>> # Featweight should be off because there is no Naut detector
        >>> print(qreq_.qparams.query_cfgstr)
        >>> assert qreq_.qparams.fg_weight == 0.0, (
        ...    'qreq_.qparams.fg_weight = %r ' % qreq_.qparams.fg_weight)
        >>> assert qreq_.qparams.sv_on is True, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
        >>> datahashid = qreq_.get_data_hashid()
        >>> dbname = ibs.get_dbname()
        >>> result = dbname + datahashid
        >>> print(result)
        NAUT_test_DSUUIDS((5)4e972cjxcj30a8u1)

    """
    if verbose:
        print(' --- New IBEIS QRequest --- ')
    cfg     = ibs.cfg.query_cfg
    qresdir = ibs.get_qres_cachedir()
    cfgdict = {} if cfgdict is None else cfgdict.copy()
    # <HACK>
    if unique_species is None:
        unique_species_ = apply_species_with_detector_hack(ibs, cfgdict, qaid_list, daid_list)
    else:
        unique_species_ = unique_species
    # </HACK>
    qparams = QueryParams(cfg, cfgdict)
    qreq_ = QueryRequest(qaid_list, daid_list, qparams, qresdir, ibs)
    if verbose:
        print(' * query_cfgstr = %s' % (qreq_.qparams.query_cfgstr,))
    qreq_.unique_species = unique_species_  # HACK
    return qreq_


@profile
def apply_species_with_detector_hack(ibs, cfgdict, qaids, daids):
    """
    HACK turns of featweights if they cannot be applied
    """
    # Only apply the hack with repsect to the queried annotations
    aid_list = np.hstack((qaids, daids)).tolist()
    unique_species = ibs.get_database_species(aid_list)
    # turn off featureweights when not absolutely sure they are ok to us,)
    candetect = (len(unique_species) == 1 and
                 ibs.has_species_detector(unique_species[0]))
    if not candetect:
        print('HACKING FG_WEIGHT OFF (database species is not supported)')
        if len(unique_species) != 1:
            print('  * len(unique_species) = %r' % len(unique_species))
        else:
            print('  * unique_species = %r' % (unique_species,))
        print('  * valid species = %r' % (ibs.get_species_with_detectors(),))
        #cfg._featweight_cfg.featweight_on = 'ERR'
        cfgdict['featweight_on'] = 'ERR'
    else:
        #print(ibs.get_annot_species_texts(aid_list))
        print('HACK FG_WEIGHT NOT APPLIED, unique_species=%r' % (unique_species,))
        #, aid_list=%r' % (unique_species, aid_list))
    return unique_species


#def qreq_shallow_copy(qreq_, qx=None, dx=None):
#    #[qx:qx + 1]
#    #[qx:qx + 1]
#    qreq_copy  = QueryRequest(qaid_list, quuid_list, daid_list, duuid_list,
#                              qreq_.qparams, qreq_.qresdir, qreq_.ibs)
#    qreq_copy.unique_species = qreq_.unique_species  # HACK
#    qreq_copy.ibs = qreq_.ibs
#    return qreq_copy


@six.add_metaclass(ut.ReloadingMetaclass)
class QueryRequest(object):
    def __init__(qreq_, qaid_list, daid_list, qparams,
                 qresdir, ibs):
        # Reminder:
        # lists and other objects are functionally equivalent to pointers
        #
        # Conceptually immutable State
        qreq_.unique_species = None  # num categories
        qreq_.internal_qspeciesid_list = None  # category species id label list
        qreq_.internal_qaids = None
        qreq_.internal_daids = None
        # Conceptually mutable state
        qreq_.internal_qaids_mask = None
        qreq_.internal_daids_mask = None
        #qreq_.internal_qnid_list = None  # individual name id label list
        #qreq_.internal_qidx  = None
        #qreq_.internal_didx  = None
        #qreq_.internal_qvecs_list = None
        #qreq_.internal_qkpts_list = None
        #qreq_.internal_dkpts_list = None
        #qreq_.internal_qgid_list  = None
        #qreq_.internal_qnid_list  = None
        # Loaded Objects
        # Handle to parent IBEIS Controller
        # THIS SHOULD BE OK BUT MAYBE IBS SHOULD BE REMOVED FROM THE
        # PICTURE AFTER THE QREQ IS BUILT?
        qreq_.ibs = ibs
        # The nearest neighbor mechanism
        qreq_.indexer = None
        # The scoring normalization mechanism
        qreq_.normalizer = None
        qreq_.dstcnvs_normer = None
        # Hacky metadata
        qreq_.metadata = {}
        # DEPRICATE?
        #qreq_.aid2_nid = None
        qreq_.hasloaded = False
        #qreq_.internal_quuids = None
        #qreq_.internal_duuids = None

        # Set values
        qreq_.qparams = qparams   # Parameters relating to pipeline execution
        qreq_.qresdir = qresdir
        qreq_.set_external_daids(daid_list)
        qreq_.set_external_qaids(qaid_list)

    @profile
    def remove_internal_daids(qreq_, remove_daids):
        r"""
        State Modification: remove daids from the query request.  Do not call
        this function often. It invalidates the indexer, which is very slow to
        rebuild.  Should only be done between query pipeline runs.

        CommandLine:
            python -m ibeis.model.hots.query_request --test-remove_internal_daids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.query_request import *  # NOQA
            >>> import ibeis
            >>> # build test data
            >>> ibs = ibeis.opendb('testdb1')
            >>> species = ibeis.const.Species.ZEB_PLAIN
            >>> daids = ibs.get_valid_aids(species=species, is_exemplar=True)
            >>> qaids = ibs.get_valid_aids(species=species, is_exemplar=False)
            >>> qreq_ = ibs.new_query_request(qaids, daids)
            >>> remove_daids = daids[0:1]
            >>> # execute function
            >>> assert len(qreq_.internal_daids) == 4, 'bad setup data'
            >>> qreq_.remove_internal_daids(remove_daids)
            >>> # verify results
            >>> assert len(qreq_.internal_daids) == 3, 'did not remove'
        """
        # Invalidate the current indexer, mask and metadata
        qreq_.indexer = None
        qreq_.internal_daids_mask = None
        qreq_.metadata = {}
        # Find indicies to remove
        delete_flags = vt.get_covered_mask(qreq_.internal_daids, remove_daids)
        delete_indices = np.where(delete_flags)[0]
        assert len(delete_indices) == len(remove_daids), 'requested removal of nonexistant daids'
        # Remove indicies
        qreq_.internal_daids = np.delete(qreq_.internal_daids, delete_indices)

    @profile
    def add_internal_daids(qreq_, new_daids):
        """
        State Modification: add new daid to query request. Should only be
        done between query pipeline runs
        """
        if ut.DEBUG2:
            species = qreq_.ibs.get_annot_species(new_daids)
            assert set(qreq_.unique_species) == set(species), 'inconsistent species'
        qreq_.internal_daids_mask = None
        qreq_.metadata = {}
        qreq_.internal_daids = np.append(qreq_.internal_daids, new_daids)
        # TODO: multi-indexer add_support
        qreq_.indexer.add_ibeis_support(qreq_, new_daids)

    @profile
    def shallowcopy(qreq_, qx=None, dx=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of the qx
        and dx objects.
        used to generate chunks of vsone queries

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.query_request import *  # NOQA
            >>> import ibeis
            >>> qreq_, ibs = get_test_qreq()
            >>> qreq2_ = qreq_.shallowcopy(qx=0)
            >>> assert qreq_.get_external_daids() is qreq2_.get_external_daids()
            >>> assert len(qreq_.get_external_qaids()) != len(qreq2_.get_external_qaids())
            >>> assert qreq_.metadata is not qreq2_.metadata
        """
        qreq2_ = copy.copy(qreq_)
        if qx is not None:
            qaid_list  = qreq2_.get_external_qaids()
            qaid_list  = qaid_list[qx:qx + 1]
            #quuid_list = qreq2_.get_external_quuids()
            #quuid_list = quuid_list[qx:qx + 1]
            qreq2_.set_external_qaids(qaid_list)  # , quuid_list)
        if dx is not None:
            daid_list  = qreq2_.get_external_daids()
            daid_list  = daid_list[dx:dx + 1]
            #duuid_list = qreq2_.get_external_duuids()
            #duuid_list = duuid_list[dx:dx + 1]
            qreq2_.set_external_daids(daid_list)
        # The shallow copy does not bring over output / query data
        qreq2_.indexer = None
        qreq2_.metadata = {}
        qreq2_.hasloaded = False
        return qreq2_

    # --- State Modification ---
    # TODO: Dont modify state

    @profile
    def set_external_daids(qreq_, daid_list):
        # DO NOT USE
        #assert len(daid_list) == len(duuid_list), 'unequal len external daids'
        if qreq_.qparams.vsmany:
            qreq_.set_internal_daids(daid_list)
        else:
            qreq_.set_internal_qaids(daid_list)

    @profile
    def set_external_qaids(qreq_, qaid_list):
        # TODO make shallow copy instead
        # DO NOT USE
        #assert len(qaid_list) == len(quuid_list), 'unequal len internal qaids'
        if qreq_.qparams.vsmany:
            qreq_.set_internal_qaids(qaid_list)
        else:
            qreq_.set_internal_daids(qaid_list)

    @profile
    def set_external_qaid_mask(qreq_, masked_qaid_list):
        r"""
        Args:
            qaid_list (list):

        CommandLine:
            python -m ibeis.model.hots.query_request --test-set_external_qaid_mask

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.query_request import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(db='testdb1')
            >>> qaid_list = [1, 2, 3, 4, 5]
            >>> daid_list = [1, 2, 3, 4, 5]
            >>> qreq_ = ibs.new_query_request(qaid_list, daid_list)
            >>> masked_qaid_list = [2, 4, 5]
            >>> qreq_.set_external_qaid_mask(masked_qaid_list)
            >>> result = np.array_str(qreq_.get_external_qaids())
            >>> print(result)
            [1 3]
        """
        if qreq_.qparams.vsmany:
            qreq_.set_internal_masked_qaids(masked_qaid_list)
        else:
            qreq_.set_internal_masked_daids(masked_qaid_list)

    # --- Internal Annotation ID Masks ----

    @profile
    def set_internal_masked_daids(qreq_, masked_daid_list):
        """ used by the pipeline to execute a subset of the query request
        without modifying important state """
        if masked_daid_list is None or len(masked_daid_list) == 0:
            qreq_.internal_daids_mask = None
        else:
            #with ut.EmbedOnException():
            # input denotes invalid elements mark all elements not in that
            # list as True
            flags = vt.get_uncovered_mask(qreq_.internal_daids, masked_daid_list)
            assert len(flags) == len(qreq_.internal_daids), 'unequal len internal daids'
            qreq_.internal_daids_mask = flags

    @profile
    def set_internal_masked_qaids(qreq_, masked_qaid_list):
        """
        used by the pipeline to execute a subset of the query request
        without modifying important state

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> from ibeis.model.hots import pipeline
            >>> cfgdict1 = dict(codename='vsone', sv_on=True)
            >>> qaid_list = [1, 2, 3, 4]
            >>> daid_list = [1, 2, 3, 4]
            >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict1,
            ...     qaid_list=qaid_list, daid_list=daid_list)
            >>> qaids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(qaid_list, qaids)
            >>> masked_qaid_list = [1, 2, 3,]
            >>> qreq_.set_internal_masked_qaids(masked_qaid_list)
            >>> new_internal_aids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(new_internal_aids, [4])
        """
        if masked_qaid_list is None or len(masked_qaid_list) == 0:
            qreq_.internal_qaids_mask = None
        else:
            #with ut.EmbedOnException():
            # input denotes invalid elements mark all elements not in that
            # list as True
            flags = vt.get_uncovered_mask(qreq_.internal_qaids, masked_qaid_list)
            assert len(flags) == len(qreq_.internal_qaids), 'unequal len internal qaids'
            qreq_.internal_qaids_mask = flags

    @profile
    def set_internal_unmasked_qaids(qreq_, unmasked_qaid_list):
        """
        used by the pipeline to execute a subset of the query request
        without modifying important state

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> from ibeis.model.hots import pipeline
            >>> cfgdict1 = dict(codename='vsone', sv_on=True)
            >>> qaid_list = [1, 2, 3, 4]
            >>> daid_list = [1, 2, 3, 4]
            >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict1,
            ...     qaid_list=qaid_list, daid_list=daid_list)
            >>> qaids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(qaid_list, qaids)
            >>> unmasked_qaid_list = [1, 2, 3,]
            >>> qreq_.set_internal_unmasked_qaids(unmasked_qaid_list)
            >>> new_internal_aids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(new_internal_aids, unmasked_qaid_list)
        """
        if unmasked_qaid_list is None:
            qreq_.internal_qaids_mask = None
        else:
            # input denotes valid elements mark all elements not in that list as False
            flags = vt.get_covered_mask(qreq_.internal_qaids, unmasked_qaid_list)
            assert len(flags) == len(qreq_.internal_qaids), 'unequal len internal qaids'
            qreq_.internal_qaids_mask = flags

    # --- Internal Annotation IDs ----

    @profile
    def set_internal_daids(qreq_, daid_list):
        qreq_.internal_daids_mask = None  # Invalidate mask
        qreq_.internal_daids = np.array(daid_list)
        #assert len(daid_list) == len(duuid_list), 'unequal len internal daids'
        #qreq_.internal_duuids = np.array(duuid_list)
        # Index the annotation ids for fast internal lookup
        #qreq_.internal_didx = np.arange(len(daid_list))

    @profile
    def set_internal_qaids(qreq_, qaid_list):
        qreq_.internal_qaids_mask = None  # Invalidate mask
        qreq_.internal_qaids = np.array(qaid_list)
        #assert len(qaid_list) == len(quuid_list), 'unequal len internal qaids'
        #qreq_.internal_quuids = np.array(quuid_list)
        # Index the annotation ids for fast internal lookup
        #qreq_.internal_qidx = np.arange(len(qaid_list))

    # --- INTERNAL INTERFACE ---
    # For within pipeline use only

    @profile
    def get_internal_daids(qreq_):
        if qreq_.internal_daids_mask is None:
            return qreq_.internal_daids
        else:
            return qreq_.internal_daids[qreq_.internal_daids_mask]

    @profile
    def get_internal_qaids(qreq_):
        if qreq_.internal_qaids_mask is None:
            return qreq_.internal_qaids
        else:
            return qreq_.internal_qaids[qreq_.internal_qaids_mask]

    @profile
    def get_internal_duuids(qreq_):
        return qreq_.ibs.get_annot_semantic_uuids(qreq_.get_internal_daids())
        #return qreq_.internal_duuids

    @profile
    def get_internal_quuids(qreq_):
        return qreq_.ibs.get_annot_semantic_uuids(qreq_.get_internal_qaids())
        #return qreq_.internal_quuids

    # --- EXTERNAL INTERFACE ---

    def get_unique_species(qreq_):
        return qreq_.unique_species

    # External id-lists

    @profile
    def get_external_daids(qreq_):
        """ These are the users daids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_daids()
        else:
            return qreq_.get_internal_qaids()

    @profile
    def get_external_qaids(qreq_):
        """ These are the users qaids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_qaids()
        else:
            return qreq_.get_internal_daids()

    @profile
    def get_external_quuids(qreq_):
        """ These are the users qauuids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_quuids()
        else:
            return qreq_.get_internal_duuids()

    @profile
    def get_external_duuids(qreq_):
        """ These are the users qauuids in vsone mode """
        if qreq_.qparams.vsmany:
            return qreq_.get_internal_duuids()
        else:
            return qreq_.get_internal_quuids()

    # External id-hashes

    @profile
    def get_data_hashid(qreq_):
        daids = qreq_.get_external_daids()
        try:
            assert len(daids) > 0, 'QRequest not populated. len(daids)=0'
        except AssertionError as ex:
            ut.printex(ex, iswarning=True)
        # TODO: SYSTEM : semantic should only be used if name scoring is on
        data_hashid = qreq_.ibs.get_annot_hashid_semantic_uuid(daids, prefix='D')
        return data_hashid

    @profile
    def get_query_hashid(qreq_):
        qaids = qreq_.get_external_qaids()
        assert len(qaids) > 0, 'QRequest not populated. len(qaids)=0'
        # TODO: SYSTEM : semantic should only be used if name scoring is on
        query_hashid = qreq_.ibs.get_annot_hashid_semantic_uuid(qaids, prefix='Q')
        return query_hashid

    @profile
    def get_query_cfgstr(qreq_):
        query_cfgstr = qreq_.qparams.query_cfgstr
        return query_cfgstr

    @profile
    def get_cfgstr(qreq_):
        """ main cfgstring used to identify the 'querytype' """
        data_hashid = qreq_.get_data_hashid()
        query_cfgstr = qreq_.get_query_cfgstr()
        cfgstr = data_hashid + query_cfgstr
        return cfgstr

    def get_qresdir(qreq_):
        return qreq_.qresdir

    # --- IBEISControl Transition ---

    #def get_annot_name_rowids(qreq_, aids):
    #    return qreq_.ibs.get_annot_name_rowids(aids)

    #def get_annot_gids(qreq_, aids):
    #    assert qreq_.ibs is not qreq_
    #    return qreq_.ibs.get_annot_gids(aids)

    #def get_annot_kpts(qreq_, aids):
    #    return qreq_.ibs.get_annot_kpts(aids)

    #def get_annot_chipsizes(qreq_, aids):
    #    return qreq_.ibs.get_annot_chipsizes(qreq_, aids)

    # --- Lazy Loading ---

    @profile
    def lazy_preload(qreq_, verbose=True):
        """
        feature weights and normalizers should be loaded before vsone queries
        are issued. They do not depened only on qparams

        Load non-query specific normalizers / weights
        """
        if verbose:
            print('[qreq] lazy preloading')
        if qreq_.qparams.featweight_on is True:
            qreq_.ensure_featweights(verbose=verbose)
        if qreq_.qparams.score_normalization is True:
            qreq_.load_score_normalizer(verbose=verbose)
        if qreq_.qparams.use_external_distinctiveness:
            qreq_.load_distinctiveness_normalizer(verbose=verbose)

    @profile
    def lazy_load(qreq_, verbose=True):
        """
        Performs preloading of all data needed for a batch of queries
        """
        print('[qreq] lazy loading')
        #with ut.Indenter('[qreq.lazy_load]'):
        qreq_.hasloaded = True
        #qreq_.ibs = ibs  # HACK
        qreq_.lazy_preload(verbose=verbose)
        if qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
            qreq_.load_indexer(verbose=verbose)
            # FIXME: not sure if this is even used
            #qreq_.load_query_vectors()
            #qreq_.load_query_keypoints()
        #if qreq_.qparams.pipeline_root in ['smk']:
        #    # TODO load vocabulary indexer

    # load query data structures

    @profile
    def ensure_featweights(qreq_, verbose=True):
        """ ensure feature weights are computed """
        #with ut.EmbedOnException():
        internal_qaids = qreq_.get_internal_qaids()
        internal_daids = qreq_.get_internal_daids()
        # TODO: pass qreq_ down so the right parameters are computed
        qreq_.ibs.get_annot_fgweights(internal_qaids, ensure=True, qreq_=qreq_)
        qreq_.ibs.get_annot_fgweights(internal_daids, ensure=True, qreq_=qreq_)

    @profile
    def load_indexer(qreq_, verbose=True, force=False):
        if not force and qreq_.indexer is not None:
            return False
        else:
            index_method = qreq_.qparams.index_method
            if index_method == 'single':
                # TODO: SYSTEM updatable indexer
                if verbose:
                    print('loading single indexer normalizer')
                indexer = neighbor_index.request_ibeis_nnindexer(qreq_, verbose=verbose)
            else:
                if verbose:
                    print('loading multi indexer normalizer')
                indexer = multi_index.request_ibeis_mindexer(qreq_, verbose=verbose)
            qreq_.indexer = indexer
            return True

    @profile
    def load_score_normalizer(qreq_, verbose=True):
        if qreq_.normalizer is not None:
            return False
        if verbose:
            print('loading score normalizer')
        # TODO: SYSTEM updatable normalizer
        normalizer = score_normalization.request_ibeis_normalizer(qreq_, verbose=verbose)
        qreq_.normalizer = normalizer

    @profile
    def load_distinctiveness_normalizer(qreq_, verbose=True):
        """
        Example:
            >>> from ibeis.model.hots import distinctiveness_normalizer
            >>> verbose = True
        """
        if qreq_.dstcnvs_normer is not None:
            return False
        if verbose:
            print('loading external distinctiveness normalizer')
        # TODO: SYSTEM updatable dstcnvs_normer
        dstcnvs_normer = distinctiveness_normalizer.request_ibeis_distinctiveness_normalizer(qreq_, verbose=verbose)
        qreq_.dstcnvs_normer = dstcnvs_normer
        if verbose:
            print('qreq_.dstcnvs_normer = %r' % (qreq_.dstcnvs_normer,))

    # load data lists
    # see _broken/broken_qreq.py

    #def load_query_vectors(qreq_, ibs):
    #    if qreq_.internal_qvecs_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    vecs_list = ibs.get_annot_vecs(aid_list)
    #    qreq_.internal_qvecs_list = vecs_list

    #def load_query_keypoints(qreq_, ibs):
    #    if qreq_.internal_qkpts_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    kpts_list = ibs.get_annot_kpts(aid_list)
    #    qreq_.internal_qkpts_list = kpts_list

    #def load_data_keypoints(qreq_, ibs):
    #    if qreq_.internal_dkpts_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_daids()
    #    kpts_list = ibs.get_annot_kpts(aid_list)
    #    qreq_.internal_dkpts_list = kpts_list

    #def load_annot_nameids(qreq_, ibs):
    #    import itertools
    #    aids = list(set(itertools.chain(qreq_.qaids, qreq_.daids)))
    #    nids = ibs.get_annot_name_rowids(aids)
    #    qreq_.aid2_nid = dict(zip(aids, nids))

    def get_infostr(qreq_):
        infostr_list = []
        app = infostr_list.append
        qaid_internal = qreq_.get_internal_qaids()
        daid_internal = qreq_.get_internal_daids()
        qd_intersection = ut.intersect_ordered(daid_internal, qaid_internal)
        app(' * len(internal_qaids) = %r' % len(daid_internal))
        app(' * len(internal_daids) = %r' % len(qaid_internal))
        app(' * len(qd_intersection) = %r' % len(qd_intersection))
        infostr = '\n'.join(infostr_list)
        return infostr

    @profile
    def get_external_query_groundtruth(qreq_, qaids):
        """ gets groundtruth that are accessible via this query """
        external_daids = qreq_.get_external_daids()
        gt_aids = qreq_.ibs.get_annot_groundtruth(qaids, daid_list=external_daids)
        return gt_aids

    @profile
    def get_internal_query_groundtruth(qreq_, qaids):
        """ gets groundtruth that are accessible via this query """
        internal_daids = qreq_.get_internal_daids()
        gt_aids = qreq_.ibs.get_annot_groundtruth(qaids, daid_list=internal_daids)
        return gt_aids

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
            if ut.NOT_QUIET:
                print('[qreq_] asserting %s aids' % len(aids))
            assert len(aids) == len(uuids)
            assert all([u1 == u2 for u1, u2 in zip(ibs.get_annot_semantic_uuids(aids), uuids)])
        assert_uuids(qaids, qauuids)
        assert_uuids(daids, dauuids)
        assert_uuids(_qaids, _qauuids)
        assert_uuids(_daids, _dauuids)

    def make_empty_query_results(qreq_):
        """ returns empty query results for each external qaid """
        external_qaids   = qreq_.get_external_qaids()
        external_qauuids = qreq_.get_external_quuids()
        daids  = qreq_.get_external_daids()
        cfgstr = qreq_.get_cfgstr()
        qres_list = [hots_query_result.QueryResult(qaid, qauuid, cfgstr, daids)
                     for qaid, qauuid in zip(external_qaids, external_qauuids)]
        return qres_list


class QueryParams(object):
    """
    Structure to store static query pipeline parameters
    parses nested config structure into this flat one

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
        # if given custom settings update the config and ensure feasibilty
        if cfgdict is not None:
            cfg = cfg.deepcopy()
            cfg.update_query_cfg(**cfgdict)
        # Get flat item list
        param_list = Config.parse_config_items(cfg)
        # Assert that there are no config conflicts
        duplicate_keys = ut.find_duplicate_items(ut.get_list_column(param_list, 0))
        assert len(duplicate_keys) == 0, 'Configs have duplicate names: %r' % duplicate_keys
        # Set nexted config attributes as flat qparam properties
        for key, val in param_list:
            setattr(qparams, key, val)
        # Add params not implicitly represented in Config object
        pipeline_root      = cfg.pipeline_root
        active_filter_list = cfg.filt_cfg.get_active_filters()
        filt2_stw          = {filt: cfg.filt_cfg.get_stw(filt)
                               for filt in active_filter_list}
        # Correct dumb filt2_stw Pref bugs
        for key, val in six.iteritems(filt2_stw):
            if val[1] == 'None':
                val[1] = None
            if val[1] is not None and not isinstance(val[1], (float, int)):
                val[1] = float(val[1])
        qparams.active_filter_list = active_filter_list
        qparams.filt2_stw          = filt2_stw
        qparams.flann_params       = cfg.flann_cfg.get_flann_params()
        qparams.pipeline_root      = pipeline_root
        qparams.vsmany             = pipeline_root == 'vsmany'
        qparams.vsone              = pipeline_root == 'vsone'
        # Add custom strings to the mix as well
        qparams.featweight_cfgstr = cfg._featweight_cfg.get_cfgstr()
        qparams.feat_cfgstr  = cfg._featweight_cfg._feat_cfg.get_cfgstr()
        qparams.nn_cfgstr    = cfg.nn_cfg.get_cfgstr()
        qparams.filt_cfgstr  = cfg.filt_cfg.get_cfgstr()
        qparams.sv_cfgstr    = cfg.sv_cfg.get_cfgstr()
        qparams.flann_cfgstr = cfg.flann_cfg.get_cfgstr()
        qparams.query_cfgstr = cfg.get_cfgstr()
        qparams.vocabtrain_cfgstr = cfg.smk_cfg.vocabtrain_cfg.get_cfgstr()


def get_test_qreq():
    import ibeis
    qaid_list = [1, 2]
    daid_list = [1, 2, 3, 4, 5]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, qaid_list, daid_list)
    return qreq_, ibs


def test_cfg_deepcopy():
    """
    TESTING FUNCTION

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


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.query_request --test-QueryParams
        utprof.sh -m ibeis.model.hots.query_request --test-QueryParams

        python -m ibeis.model.hots.query_request
        python -m ibeis.model.hots.query_request --allexamples
        python -m ibeis.model.hots.query_request --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
