"""
Hotspotter pipeline module

TODO:
    We need to remove dictionaries from the pipeline
    We can easily use parallel lists

Module Concepts::
    PREFIXES:
    qaid2_XXX - prefix mapping query chip index to
    qfx2_XXX  - prefix mapping query chip feature index to

     * nns    - a (qfx2_idx, qfx2_dist) tuple

     * idx    - the index into the nnindexers descriptors
     * qfx    - query feature index wrt the query chip
     * dfx    - query feature index wrt the database chip
     * dist   - the distance to a corresponding feature
     * fm     - a list of featur match pairs (qfx, dfx)
     * fsv    - a score vector of a corresponding feature
     * valid  - a valid bit for a corresponding feature

    PIPELINE_VARS::
    nns_list - maping from query chip index to nns
    {
     * qfx2_idx   - ranked list of query feature indexes to database feature indexes
     * qfx2_dist - ranked list of query feature indexes to database feature indexes
    }

    * qaid2_norm_weight - mapping from qaid to (qfx2_normweight, qfx2_selnorm)
             = qaid2_nnfiltagg[qaid]

CommandLine:
    To see the ouput of a complete pipeline run use

    # Set to whichever database you like
    python main.py --db PZ_MTEST --setdb
    python main.py --db NAUT_test --setdb
    python main.py --db testdb1 --setdb

    # Then run whichever configuration you like
    python main.py --verbose --noqcache --cfg codename:vsone --query 1
    python main.py --verbose --noqcache --cfg codename:vsone_norm --query 1
    python main.py --verbose --noqcache --cfg codename:vsmany --query 1
    python main.py --verbose --noqcache --cfg codename:vsmany_nsum  --query 1
"""

from __future__ import absolute_import, division, print_function
from six.moves import zip, range
import six
import numpy as np
import vtool as vt
from vtool import keypoint as ktool
from vtool import spatial_verification as sver
from ibeis.model.hots import hots_query_result
from ibeis.model.hots import hstypes  # NOQA
from ibeis.model.hots import chip_match
from ibeis.model.hots import nn_weights
from ibeis.model.hots import scoring
from ibeis.model.hots import exceptions as hsexcept
from ibeis.model.hots import _pipeline_helpers as plh
import utool as ut
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[pipeline]', DEBUG=False)


#=================
# Globals
#=================

TAU = 2 * np.pi  # References: tauday.com
NOT_QUIET = ut.NOT_QUIET and not ut.get_argflag('--quiet-query')
DEBUG_PIPELINE = ut.get_argflag(('--debug-pipeline', '--debug-pipe'))
VERB_PIPELINE =  NOT_QUIET and (ut.VERBOSE or ut.get_argflag(('--verbose-pipeline', '--verb-pipe')))
VERYVERBOSE_PIPELINE = ut.get_argflag(('--very-verbose-pipeline', '--very-verb-pipe'))


NN_LBL      = 'Assign NN:       '
FILT_LBL    = 'Filter NN:       '
BUILDCM_LBL = 'Build Chipmatch: '
SVER_LVL    = 'SVER:            '


# Query Level 0
#@ut.indent_func('[Q0]')
@profile
def request_ibeis_query_L0(ibs, qreq_, verbose=VERB_PIPELINE):
    r"""
    Driver logic of query pipeline

    Note:
        Make sure _pipeline_helpres.testrun_pipeline_upto reflects what happens
        in this function

    Args:
        ibs   (IBEISController): IBEIS database object to be queried.
            technically this object already lives inside of qreq_.
        qreq_ (QueryRequest): hyper-parameters. use ``ibs.new_query_request`` to create one

    Returns:
        (dict[int, QueryResult]): qaid2_qres maps query annotid to QueryResult

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0

    Example1:
        >>> # one-vs-many:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> import ibeis
        >>> cfgdict = dict(codename='vsmany')
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict)
        >>> print(qreq_.qparams.query_cfgstr)
        >>> verbose = True
        >>> qaid2_qres = request_ibeis_query_L0(ibs, qreq_, verbose=verbose)
        >>> qres = qaid2_qres[list(qaid2_qres.keys())[0]]
        >>> if ut.get_argflag('--show') or ut.inIPython():
        ...     qres.show_analysis(ibs, fnum=0, make_figtitle=True)
        >>> print(qres.get_inspect_str())

    Example2:
        >>> # one-vs-one:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> import ibeis  # NOQA
        >>> cfgdict1 = dict(codename='vsone', sv_on=False)
        >>> ibs1, qreq_1 = plh.get_pipeline_testdata(cfgdict=cfgdict1)
        >>> print(qreq_1.qparams.query_cfgstr)
        >>> qaid2_qres1 = request_ibeis_query_L0(ibs1, qreq_1)
        >>> qres1 = qaid2_qres1[list(qaid2_qres1.keys())[0]]
        >>> if ut.get_argflag('--show') or ut.inIPython():
        ...     qres1.show_analysis(ibs1, fnum=1, make_figtitle=True)
        >>> print(qres1.get_inspect_str())

    """
    # Load data for nearest neighbors
    if verbose:
        assert ibs is qreq_.ibs
        print('\n\n[hs] +--- STARTING HOTSPOTTER PIPELINE ---')
        print(qreq_.get_infostr())

    qreq_.lazy_load(verbose=verbose)

    if qreq_.qparams.pipeline_root == 'smk':
        from ibeis.model.hots.smk import smk_match
        # Alternative to naive bayes matching:
        # Selective match kernel
        qaid2_scores, qaid2_chipmatch_FILT_ = smk_match.execute_smk_L5(qreq_)
    elif qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
        impossible_daids_list, Kpad_list = build_impossible_daids_list(qreq_)

        # Nearest neighbors (nns_list)
        # a nns object is a tuple(ndarray, ndarray) - (qfx2_dx, qfx2_dist)
        # * query descriptors assigned to database descriptors
        # * FLANN used here
        nns_list = nearest_neighbors(qreq_, Kpad_list, verbose=verbose)

        # Remove Impossible Votes
        # a nnfilt object is an ndarray qfx2_valid
        # * marks matches to the same image as invalid
        nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list, impossible_daids_list, verbose=verbose)

        # Nearest neighbors weighting / scoring (filtweights_list)
        # filtweights_list maps qaid to filtweights which is a dict
        # that maps a filter name to that query's weights for that filter
        filtkey_list, filtweights_list, filtvalids_list = weight_neighbors(qreq_, nns_list, nnvalid0_list, verbose=verbose)

        # Nearest neighbors to chip matches (cm_list)
        # * Inverted index used to create aid2_fmfsvfk (TODO: aid2_fmfv)
        # * Initial scoring occurs
        # * vsone un-swapping occurs here
        cm_list_FILT = build_chipmatches(qreq_, nns_list, nnvalid0_list,
                                         filtkey_list, filtweights_list, filtvalids_list,
                                         verbose=verbose)
    else:
        print('invalid pipeline root %r' % (qreq_.qparams.pipeline_root))

    # Spatial verification (cm_list) (TODO: cython)
    # * prunes chip results and feature matches
    # TODO: allow for reweighting of feature matches to happen.
    qaid2_chipmatch_SVER_ = spatial_verification(qreq_, cm_list_FILT,
                                                 verbose=verbose)

    # We might just put this check inside the function like it is for SVER.
    # or just not do that and use some good pipeline framework
    if qreq_.qparams.rrvsone_on:
        # VSONE RERANKING
        qaid2_chipmatch_ = vsone_reranking(qreq_, qaid2_chipmatch_SVER_, verbose=verbose)
    else:
        qaid2_chipmatch_ = qaid2_chipmatch_SVER_

    # Query results format (qaid2_qres)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qaid2_qres_ = chipmatch_to_resdict(qreq_, qaid2_chipmatch_,
                                       verbose=verbose)

    # <HACK>
    # FOR VSMANY DISTINCTIVENSS
    if qreq_.qparams.return_expanded_nns:
        assert qreq_.qparams.vsmany, ' must be in a special vsmany mode'
        # MAJOR HACK TO RETURN ALL QUERY NEAREST NEIGHBORS
        # BREAKS PIPELINE CACHING ASSUMPTIONS
        # SHOULD ONLY BE CALLED BY SPECIAL_QUERY
        # CAUSES TOO MUCH DATA TO BE SAVED
        for qaid, nns in zip(qreq_.get_external_qaids(), nns_list):
            # TODO: hook up external neighbor mechanism?
            # No, not here, further down.
            (qfx2_idx, qfx2_dist) = nns
            qres = qaid2_qres_[qaid]
            qres.qfx2_dist = qfx2_dist
            msg_list = [
                #'qres.qfx2_daid = ' + ut.get_object_size_str(qres.qfx2_daid),
                #'qres.qfx2_dfx = ' + ut.get_object_size_str(qres.qfx2_dfx),
                'qres.qfx2_dist = ' + ut.get_object_size_str(qres.qfx2_dist),
            ]
            print('\n'.join(msg_list))
    # </HACK>

    if VERB_PIPELINE:
        print('[hs] L___ FINISHED HOTSPOTTER PIPELINE ___')

    return qaid2_qres_

#============================
# 0) Nearest Neighbors
#============================


def build_impossible_daids_list(qreq_, verbose=VERB_PIPELINE):
    r"""
    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-build_impossible_daids_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> species = ibeis.const.Species.ZEB_PLAIN
        >>> daids = ibs.get_valid_aids(species=species)
        >>> qaids = ibs.get_valid_aids(species=species)
        >>> qreq_ = ibs.new_query_request(qaids, daids, cfgdict=dict(codename='vsmany', can_match_sameimg=False, can_match_samename=False))
        >>> # execute function
        >>> impossible_daids_list, Kpad_list = build_impossible_daids_list(qreq_)
        >>> # verify results
        >>> result = str((impossible_daids_list, Kpad_list))
        >>> print(result)
        ([array([1]), array([2, 3]), array([2, 3]), array([4]), array([5, 6]), array([5, 6])], [1, 2, 2, 1, 2, 2])
    """
    if verbose:
        print('[hs] Step 0) Build impossible matches')

    cant_match_sameimg  = not qreq_.qparams.can_match_sameimg
    cant_match_samename = not qreq_.qparams.can_match_samename
    cant_match_self     = True
    internal_qaids = qreq_.get_internal_qaids()
    internal_daids = qreq_.get_internal_daids()
    internal_data_nids  = qreq_.ibs.get_annot_nids(internal_daids)

    _impossible_daid_lists = []
    if cant_match_self:
        if not (cant_match_sameimg or cant_match_samename):
            # we can skip this if sameimg or samename is specified.
            # it will cover this case for us
            _impossible_daid_lists.append([[qaid] for qaid in internal_qaids])
    if cant_match_sameimg:
        # slow way of getting contact_aids
        #contact_aids_list = qreq_.ibs.get_annot_contact_aids(internal_qaids)
        # Faster way
        internal_data_gids  = qreq_.ibs.get_annot_gids(internal_daids)
        internal_query_gids = qreq_.ibs.get_annot_gids(internal_qaids)
        contact_aids_list = [
            internal_daids.compress(internal_data_gids == gid)
            for gid in internal_query_gids
        ]
        _impossible_daid_lists.append(contact_aids_list)
        EXTEND_TO_OTHER_CONTACT_GT = False
        # Also cannot match any aids with a name of an annotation in this image
        if EXTEND_TO_OTHER_CONTACT_GT:
            # TODO: need a test set that can accomidate testing this case
            # testdb1 might cut it if we spruced it up
            nonself_contact_aids = [np.setdiff1d(aids, qaid) for aids, qaid in zip(contact_aids_list, internal_qaids)]
            nonself_contact_nids = qreq_.ibs.unflat_map(qreq_.ibs.get_annot_nids, nonself_contact_aids)
            contact_aids_gt_list = [
                internal_daids.compress(vt.get_covered_mask(internal_data_nids, nids))
                for nids in nonself_contact_nids
            ]
            _impossible_daid_lists.append(contact_aids_gt_list)

    if cant_match_samename:
        #internal_daids = qreq_.get_internal_daids()
        # slow way of getting gt_aids
        #gt_aids = qreq_.ibs.get_annot_groundtruth(internal_qaids, daid_list=internal_daids)
        #faster way
        internal_data_nids  = qreq_.ibs.get_annot_nids(internal_daids)
        internal_query_nids = qreq_.ibs.get_annot_nids(internal_qaids)
        gt_aids = [
            internal_daids.compress(internal_data_nids == nid)
            for nid in internal_query_nids
        ]
        _impossible_daid_lists.append(gt_aids)
    # TODO: add explicit not a match case in here
    _impossible_daids_list = list(map(ut.flatten, zip(*_impossible_daid_lists)))
    impossible_daids_list = [np.unique(impossible_daids) for impossible_daids in _impossible_daids_list]
    #impossible_daids_list = [
    #    np.intersect1d(np.unique(impossible_daids), internal_daids, assume_unique=True)
    #    for impossible_daids in _impossible_daids_list]

    # TODO: we need to pad K for each bad annotation
    if qreq_.qparams.vsone:
        # dont pad vsone
        Kpad_list = [0 for _ in range(len(impossible_daids_list))]
    else:
        Kpad_list = list(map(len, impossible_daids_list))  # NOQA
    return impossible_daids_list, Kpad_list

#============================
# 1) Nearest Neighbors
#============================


#@ut.indent_func('[nn]')
@profile
def nearest_neighbors(qreq_, Kpad_list, verbose=VERB_PIPELINE):
    """
    Plain Nearest Neighbors

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-nearest_neighbors

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> verbose = True
        >>> ibs, qreq_ = plh.get_pipeline_testdata(dbname='testdb1')
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'nearest_neighbors')
        >>> Kpad_list, = ut.dict_take(locals_, ['Kpad_list'])
        >>> # execute function
        >>> nn_list = nearest_neighbors(qreq_, Kpad_list, verbose=verbose)
        >>> (qfx2_idx, qfx2_dist) = nn_list[0]
        >>> num_neighbors = Kpad_list[0] + qreq_.qparams.K + qreq_.qparams.Knorm
        >>> # Assert nns tuple is valid
        >>> ut.assert_eq(qfx2_idx.shape, qfx2_dist.shape)
        >>> ut.assert_eq(qfx2_idx.shape[1], num_neighbors)
        >>> ut.assert_inbounds(qfx2_idx.shape[0], 1000, 2000)
    """
    # Neareset neighbor configuration
    K      = qreq_.qparams.K
    Knorm  = qreq_.qparams.Knorm
    #checks = qreq_.qparams.checks
    # Get both match neighbors (including padding) and normalizing neighbors
    num_neighbors_list = [K + Kpad + Knorm for Kpad in Kpad_list]
    if verbose:
        print('[hs] Step 1) Assign nearest neighbors: %s' %
              (qreq_.qparams.nn_cfgstr,))
    # For each internal query annotation
    internal_qaids = qreq_.get_internal_qaids()
    # Find the nearest neighbors of each descriptor vector
    qvecs_list = qreq_.ibs.get_annot_vecs(internal_qaids, qreq_=qreq_)
    # Mark progress ane execute nearest indexer nearest neighbor code
    progkw = dict(freq=20, time_thresh=2.5)
    qvec_iter = ut.ProgressIter(qvecs_list, lbl=NN_LBL, **progkw)
    nns_list = [qreq_.indexer.knn(qfx2_vec, num_neighbors)
                for qfx2_vec, num_neighbors in zip(qvec_iter, num_neighbors_list)]
    # Verbose statistics reporting
    if verbose:
        plh.print_nearest_neighbor_assignments(qvecs_list, nns_list)
    #if qreq_.qparams.with_metadata:
    #    qreq_.metadata['nns'] = nns_list
    return nns_list


#============================
# 2) Remove Impossible Weights
#============================


@profile
def baseline_neighbor_filter(qreq_, nns_list, impossible_daids_list, verbose=VERB_PIPELINE):
    """
    Removes matches to self, the same image, or the same name.

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-baseline_neighbor_filter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *   # NOQA
        >>> qreq_, nns_list, impossible_daids_list = plh.testdata_pre_baselinefilter(qaid_list=[1, 2, 3, 4], codename='vsmany')
        >>> nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list, impossible_daids_list)
        >>> ut.assert_eq(len(nnvalid0_list), len(qreq_.get_external_qaids()))
        >>> #ut.assert_eq(nnvalid0_list[0].shape[1], qreq_.qparams.K, 'does not match k')
        >>> #ut.assert_eq(qreq_.qparams.K, 4, 'k is not 4')
        >>> assert not np.any(nnvalid0_list[0][:, 0]), (
        ...    'first col should be all invalid because of self match')
        >>> assert not np.all(nnvalid0_list[0][:, 1]), (
        ...    'second col should have some good matches')
        >>> ut.assert_inbounds(nnvalid0_list[0].sum(), 2000, 3000)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *   # NOQA
        >>> qreq_, nns_list, impossible_daids_list = plh.testdata_pre_baselinefilter(codename='vsone')
        >>> nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list, impossible_daids_list)
        >>> ut.assert_eq(len(nnvalid0_list), len(qreq_.get_external_daids()))
        >>> ut.assert_eq(qreq_.qparams.K, 1, 'k is not 1')
        >>> ut.assert_eq(nnvalid0_list[0].shape[1], qreq_.qparams.K, 'does not match k')
        >>> ut.assert_eq(nnvalid0_list[0].sum(), 0, 'no self matches')
        >>> ut.assert_inbounds(nnvalid0_list[1].sum(), 800, 1100)
    """
    if verbose:
        print('[hs] Step 2) Baseline neighbor filter')

    Knorm = qreq_.qparams.Knorm
    nnidx_iter = (qfx2_idx.T[0:-Knorm].T for (qfx2_idx, _) in nns_list)
    qfx2_aid_list = [qreq_.indexer.get_nn_aids(qfx2_nnidx) for qfx2_nnidx in nnidx_iter]
    nnvalid0_list = [
        vt.get_uncovered_mask(qfx2_aid, impossible_daids)
        for qfx2_aid, impossible_daids in zip(qfx2_aid_list, impossible_daids_list)
    ]
    return nnvalid0_list


#============================
# 3) Nearest Neighbor weights
#============================


#@ut.indent_func('[wn]')
@profile
def weight_neighbors(qreq_, nns_list, nnvalid0_list, verbose=VERB_PIPELINE):
    """
    assigns weights to feature matches based on the active filter list

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-weight_neighbors

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> args = plh.testdata_pre_weight_neighbors('testdb1', qaid_list=[1, 2, 3])
        >>> ibs, qreq_, nns_list, nnvalid0_list = args
        >>> # execute function
        >>> filtkey_list, filtweights_list, filtvalids_list = weight_neighbors(qreq_, nns_list, nnvalid0_list)
        >>> nInternAids = len(qreq_.get_internal_qaids())
        >>> nFiltKeys = len(filtkey_list)
        >>> filtweight_depth = ut.depth_profile(filtweights_list)
        >>> filtvalid_depth = ut.depth_profile(filtvalids_list)
        >>> ut.assert_eq(nInternAids, len(filtweights_list))
        >>> ut.assert_eq(nInternAids, len(filtvalids_list))
        >>> ut.assert_eq(ut.get_list_column(filtweight_depth, 0), [nFiltKeys] * nInternAids)
        >>> ut.assert_eq(filtvalid_depth, (nInternAids, nFiltKeys))
        >>> ut.assert_eq(filtvalids_list, [[None, None], [None, None], [None, None]])
        >>> ut.assert_eq(filtkey_list, [hstypes.FiltKeys.LNBNN, hstypes.FiltKeys.FG])

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> args = plh.testdata_pre_weight_neighbors('testdb1', codename='vsone')
        >>> ibs, qreq_, nns_list, nnvalid0_list = args
        >>> # execute function
        >>> filtkey_list, filtweights_list, filtvalids_list = weight_neighbors(qreq_, nns_list, nnvalid0_list)
        >>> nFiltKeys = len(filtkey_list)
        >>> nInternAids = len(qreq_.get_internal_qaids())
        >>> filtweight_depth = ut.depth_profile(filtweights_list)
        >>> filtvalid_depth = ut.depth_profile(filtvalids_list)
        >>> ut.assert_eq(nInternAids, len(filtweights_list))
        >>> ut.assert_eq(nInternAids, len(filtvalids_list))
        >>> ut.assert_eq(ut.get_list_column(filtweight_depth, 0), [nFiltKeys] * nInternAids)
        >>> assert filtvalids_list[0][0] is not None
        >>> assert filtvalids_list[0][1] is None
        >>> ut.assert_eq(filtkey_list, [hstypes.FiltKeys.RATIO, hstypes.FiltKeys.FG])
    """
    if verbose:
        print('[hs] Step 3) Weight neighbors: ' + qreq_.qparams.nnweight_cfgstr)
    # Build weights for each active filter
    filtkey_list    = []
    _filtweight_list = []
    _filtvalid_list  = []
    if qreq_.qparams.lnbnn_on:
        lnbnn_weight_list = nn_weights.NN_WEIGHT_FUNC_DICT['lnbnn'](nns_list, nnvalid0_list, qreq_)
        _filtweight_list.append(lnbnn_weight_list)
        _filtvalid_list.append(None)  # None means all valid
        filtkey_list.append('lnbnn')
    if qreq_.qparams.ratio_thresh:
        ratio_weight_list = nn_weights.NN_WEIGHT_FUNC_DICT['ratio'](nns_list, nnvalid0_list, qreq_)
        ratio_isvalid   = [qfx2_ratio <= qreq_.qparams.ratio_thresh for qfx2_ratio in ratio_weight_list]
        ratioscore_list = [np.subtract(1, qfx2_ratio) for qfx2_ratio in ratio_weight_list]
        _filtweight_list.append(ratioscore_list)
        _filtvalid_list.append(ratio_isvalid)
        filtkey_list.append('ratio')
    if qreq_.qparams.fg_on:
        fgvote_weight_list = nn_weights.NN_WEIGHT_FUNC_DICT['fg'](nns_list, nnvalid0_list, qreq_)
        _filtweight_list.append(fgvote_weight_list)
        _filtvalid_list.append(None)  # None means all valid
        filtkey_list.append('fg')

    # Switch nested list structure from [filt, qaid] to [qaid, filt]
    nInternAids = len(nns_list)
    filtweights_list = [ut.get_list_column(_filtweight_list, index) for index in range(nInternAids)]
    filtvalids_list = [
        [
            None if filtvalid is None else filtvalid[index]
            for filtvalid in _filtvalid_list
        ]
        for index in range(nInternAids)
    ]
    return filtkey_list, filtweights_list, filtvalids_list


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> aid2
#============================


#@ut.indent_func('[bc]')
#@profile
def build_chipmatches(qreq_, nns_list, nnvalid0_list, filtkey_list, filtweights_list, filtvalids_list, verbose=VERB_PIPELINE):
    """
    pipeline step 4 - builds sparse chipmatches

    Takes the dense feature matches from query feature to (what could be any)
    database features and builds sparse matching pairs for each annotation to
    annotation match.

    Ignore:
        python -c "import utool; print(utool.auto_docstr('ibeis.model.hots.pipeline', 'build_chipmatches'))"

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-build_chipmatches
        python -m ibeis.model.hots.pipeline --test-build_chipmatches:0 --show
        python -m ibeis.model.hots.pipeline --test-build_chipmatches:1 --show

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> ibs, qreq_, args = plh.testdata_pre_build_chipmatch('testdb1', codename='vsmany')
        >>> nns_list, nnvalid0_list, filtkey_list, filtweights_list, filtvalids_list = args
        >>> verbose = True
        >>> # execute function
        >>> cm_list = build_chipmatches(qreq_, *args, verbose=verbose)
        >>> # verify results
        >>> [cm.assert_self(qreq_) for cm in cm_list]
        >>> fm = cm_list[0].fm_list[cm_list[0].daid2_idx[2]]
        >>> num_matches = len(fm)
        >>> print('vsone num_matches = %r' % num_matches)
        >>> ut.assert_inbounds(num_matches, 750, 800, 'vsmany nmatches out of bounds')
        >>> cm_list[0].testshow_single(qreq_)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> verbose = True
        >>> ibs, qreq_, args = plh.testdata_pre_build_chipmatch('testdb1', codename='vsone')
        >>> nns_list, nnvalid0_list, filtkey_list, filtweights_list, filtvalids_list = args
        >>> # execute function
        >>> cm_list = build_chipmatches(qreq_, *args, verbose=verbose)
        >>> # verify results
        >>> [cm.assert_self(qreq_) for cm in cm_list]
        >>> fm = cm_list[0].fm_list[cm_list[0].daid2_idx[2]]
        >>> num_matches = len(fm)
        >>> print('vsone num_matches = %r' % num_matches)
        >>> ut.assert_inbounds(num_matches, 33, 42, 'vsone nmatches out of bounds')
        >>> cm_list[0].testshow_single(qreq_)
    """
    is_vsone =  qreq_.qparams.vsone
    if verbose:
        pipeline_root = qreq_.qparams.pipeline_root
        print('[hs] Step 4) Building chipmatches %s' % (pipeline_root,))
    idx_list = [qfx2_idx for (qfx2_idx, _) in nns_list]
    #nnvalid0_list
    valid_match_tup_list = [
        get_sparse_matchinfo_nonagg(qreq_, qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list)
        for qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list in
        zip(idx_list, nnvalid0_list, filtweights_list, filtvalids_list)
    ]
    # Iterate over INTERNAL query annotation ids
    internal_qaids = qreq_.get_internal_qaids()
    external_qaids = qreq_.get_external_qaids()
    external_daids = qreq_.get_external_daids()
    #progkw = dict(freq=20, time_thresh=2.0)
    #intern_qaid_iter = ut.ProgressIter(internal_qaids, lbl=BUILDCM_LBL, **progkw)
    intern_qaid_iter = internal_qaids

    if is_vsone:
        # VSONE build one cmtup_old
        assert len(external_qaids) == 1, 'vsone can only accept one external qaid'
        assert np.all(external_daids == internal_qaids)
        # build vsone dict output
        qaid = external_qaids[0]
        cm_list = [
            chip_match.ChipMatch2.from_vsone_match_tup(
                valid_match_tup_list, daid_list=external_daids, qaid=qaid,
                fsv_col_lbls=filtkey_list)
        ]
    else:
        # VSMANY build many cmtup_olds
        cm_list = [chip_match.ChipMatch2.from_vsmany_match_tup(
            valid_match_tup, qaid=qaid, fsv_col_lbls=filtkey_list)
            for valid_match_tup, qaid in zip(valid_match_tup_list, intern_qaid_iter)]
    return cm_list
    # build vsmany dict output
    #cm_list = dict(zip(external_qaids, cm_list))
    #return cm_list


def get_sparse_matchinfo_nonagg(qreq_, qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list):
    """
    builds sparse iterator that generates feature match pairs, scores, and ranks

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-get_sparse_matchinfo_nonagg

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> verbose = True
        >>> qreq_, qaid, daid, args = plh.testdata_sparse_matchinfo_nonagg(codename='vsmany')
        >>> qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list = args
        >>> # execute function
        >>> valid_match_tup = get_sparse_matchinfo_nonagg(qreq_, *args)
        >>> # check results
        >>> (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank) = valid_match_tup
        >>> assert ut.list_allsame(list(map(len, valid_match_tup))), 'need same num rows'
        >>> ut.assert_inbounds(valid_qfx, -1, qreq_.ibs.get_annot_num_feats(qaid, qreq_=qreq_))
        >>> ut.assert_inbounds(valid_dfx, -1, np.array(qreq_.ibs.get_annot_num_feats(valid_daid, qreq_=qreq_)))

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> verbose = True
        >>> qreq_, qaid, daid, args = plh.testdata_sparse_matchinfo_nonagg(codename='vsone')
        >>> qfx2_idx, qfx2_valid0, qfx2_score_list, qfx2_valid_list = args
        >>> # execute function
        >>> valid_match_tup = get_sparse_matchinfo_nonagg(qreq_, *args)
        >>> # check results
        >>> (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank) = valid_match_tup
        >>> assert ut.list_allsame(list(map(len, valid_match_tup))), 'need same num rows'
        >>> ut.assert_inbounds(valid_dfx, -1, qreq_.ibs.get_annot_num_feats(qaid, qreq_=qreq_))
        >>> ut.assert_inbounds(valid_qfx, -1, qreq_.ibs.get_annot_num_feats(daid, qreq_=qreq_))
    """
    # TODO: unpacking can be external
    Knorm = qreq_.qparams.Knorm
    # Unpack neighbor ids, indices, filter scores, and flags
    qfx2_nnidx = qfx2_idx.T[:-Knorm].T
    K = len(qfx2_nnidx.T)
    qfx2_daid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    qfx2_dfx = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
    # And all valid lists together to get a final mask
    qfx2_valid_agg = vt.and_lists(qfx2_valid0, *ut.filter_Nones(qfx2_valid_list))
    # We fill filter each relavant matrix by aggregate validity
    flat_validx = np.flatnonzero(qfx2_valid_agg)
    # Infer the valid internal query feature indexes and ranks
    valid_qfx   = np.floor_divide(flat_validx, K, dtype=hstypes.INDEX_TYPE)
    valid_rank  = np.mod(flat_validx, K, dtype=hstypes.FK_DTYPE)
    # Then take the valid indices from internal database
    # annot_rowids, feature indexes, and all scores
    valid_daid  = qfx2_daid.take(flat_validx, axis=None)
    valid_dfx   = qfx2_dfx.take(flat_validx, axis=None)
    valid_scorevec = np.vstack([qfx2_score.take(flat_validx)
                                for qfx2_score in qfx2_score_list]).T
    # The q/d's are all internal here, thus in vsone they swap
    valid_match_tup = (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank)
    return valid_match_tup


#============================
# 4.5) Shortlisting
#============================


#@ut.indent_func('[scm]')


#============================
# 5) Spatial Verification
#============================


def spatial_verification(qreq_, cm_list, verbose=VERB_PIPELINE):
    """
    Returns:
        dict or tuple(dict, dict)

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-spatial_verification --show
        python -m ibeis.model.hots.pipeline --test-spatial_verification:0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
        >>> scoring.score_chipmatch_list(qreq_, cm_list, qreq_.qparams.prescore_method)  # HACK
        >>> cm = cm_list[0]
        >>> top_nids = cm.get_top_nids(6)
        >>> verbose = True
        >>> # Execute Function
        >>> cm_list_SVER = spatial_verification(qreq_, cm_list)
        >>> # Test Results
        >>> cmSV = cm_list_SVER[0]
        >>> scoring.score_chipmatch_list(qreq_, cm_list_SVER, qreq_.qparams.score_method)  # HACK
        >>> top_nids_SV = cmSV.get_top_nids(6)
        >>> cm.print_csv(sort=True)
        >>> cmSV.print_csv(sort=False)
        >>> gt_daids  = np.intersect1d(cm.get_groundtruth_daids(), cmSV.get_groundtruth_daids())
        >>> fm_list   = cm.get_annot_fm(gt_daids)
        >>> fmSV_list = cmSV.get_annot_fm(gt_daids)
        >>> maplen = lambda list_: np.array(list(map(len, list_)))
        >>> assert len(gt_daids) > 0, 'ground truth did not survive'
        >>> ut.assert_lessthan(maplen(fmSV_list), maplen(fm_list)), 'feature matches were not filtered'
        >>> cmSV.show_daids_matches(qreq_, gt_daids)
        """
    if not qreq_.qparams.sv_on or qreq_.qparams.xy_thresh is None:
        if verbose:
            print('[hs] Step 5) Spatial verification: off')
        return cm_list
    else:
        cm_list_SVER = _spatial_verification(qreq_, cm_list, verbose=verbose)
        return cm_list_SVER


#@ut.indent_func('[_sv]')
@profile
def _spatial_verification(qreq_, cm_list, verbose=VERB_PIPELINE):
    """
    make only spatially valid features survive
    """
    if verbose:
        print('[hs] Step 5) Spatial verification: ' + qreq_.qparams.sv_cfgstr)

    # TODO: move rerank out of theis pipeline node
    #with_metadata = qreq_.qparams.with_metadata
    # dbg info (can remove if there is a speed issue)
    score_method    = qreq_.qparams.prescore_method
    nNameShortList  = qreq_.qparams.nNameShortlistSVER
    nAnnotPerName   = qreq_.qparams.nAnnotPerNameSVER

    #qaid2_svtups = {} if with_metadata else None
    scoring.score_chipmatch_list(qreq_, cm_list, score_method)
    cm_shortlist = scoring.make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName)
    cm_progiter = ut.ProgressIter(cm_shortlist, nTotal=len(cm_shortlist), lbl=SVER_LVL,
                                  freq=20, time_thresh=2.0)
    cm_list_SVER = [sver_single_chipmatch(qreq_, cm) for cm in cm_progiter]
    #cm_list_SVER = []
    #for cm in cm_progiter:
    #    pass
    #    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    #    # Get information for sver, query keypoints, diaglen
    #    #cmSV, daid2_svtup = sver_single_chipmatch(qreq_, cm)
    #    cmSV = sver_single_chipmatch(qreq_, cm)
    #    #if with_metadata:
    #    #    qaid2_svtups[cm.qaid] = daid2_svtup
    #    # Rebuild the feature match / score arrays to be consistent
    #    cm_list_SVER.append(cmSV)
    #if with_metadata:
    #    qreq_.metadata['qaid2_svtups'] = qaid2_svtups
    return cm_list_SVER


@profile
def sver_single_chipmatch(qreq_, cm):
    """
    loops over a shortlist of results for a specific query annotation

    python -m ibeis.model.hots.pipeline --test-chipmatch_to_resdict:1

    """
    qaid = cm.qaid
    use_chip_extent = qreq_.qparams.use_chip_extent
    xy_thresh       = qreq_.qparams.xy_thresh
    scale_thresh    = qreq_.qparams.scale_thresh
    ori_thresh      = qreq_.qparams.ori_thresh
    min_nInliers    = qreq_.qparams.min_nInliers
    sver_weighting  = qreq_.qparams.sver_weighting
    # Precompute sver cmtup_old
    #daid2_svtup = {} if qreq_.qparams.with_metadata else None
    kpts1 = qreq_.ibs.get_annot_kpts(qaid, qreq_=qreq_)
    kpts2_list = qreq_.ibs.get_annot_kpts(cm.daid_list, qreq_=qreq_)
    if use_chip_extent:
        top_dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlensqrd(cm.daid_list, qreq_=qreq_)
    else:
        top_dlen_sqrd_list = compute_matching_dlen_extent(qreq_, cm.fm_list, kpts2_list)
    #
    _iter1 = zip(cm.daid_list, cm.fm_list, cm.fsv_list, cm.fk_list, kpts2_list, top_dlen_sqrd_list)
    svtup_list = []
    for daid, fm, fsv, fk, kpts2, dlen_sqrd2 in _iter1:
        if len(fm) == 0:
            # skip results without any matches
            sv_tup = None
        else:
            try:
                # Compute homography from chip2 to chip1
                # returned homography maps image1 space into image2 space
                # image1 is a query chip and image2 is a database chip
                sv_tup = sver.spatially_verify_kpts(
                    kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
                    dlen_sqrd2, min_nInliers)
                # returnAff=qreq_.qparams.with_metadata)
            except Exception as ex:
                ut.printex(ex, 'Unknown error in spatial verification.',
                              keys=['kpts1', 'kpts2',  'fm', 'xy_thresh',
                                    'scale_thresh', 'dlen_sqrd2', 'min_nInliers'])
                sv_tup = None
        svtup_list.append(sv_tup)

    # Remove all matches that failed spatial verification
    isnone_list = ut.flag_None_items(svtup_list)
    svtup_list_ = ut.filterfalse_items(svtup_list, isnone_list)
    daid_list   = ut.filterfalse_items(cm.daid_list, isnone_list)
    dnid_list   = ut.filterfalse_items(cm.dnid_list, isnone_list)
    fm_list     = ut.filterfalse_items(cm.fm_list, isnone_list)
    fsv_list    = ut.filterfalse_items(cm.fsv_list, isnone_list)
    fk_list     = ut.filterfalse_items(cm.fk_list, isnone_list)

    #if qreq_.qparams.with_metadata:
    #    for sv_tup, daid in zip(svtup_list_, daid_list):
    #        daid2_svtup[daid] = sv_tup

    sver_matchtup_list = []
    fsv_col_lbls = cm.fsv_col_lbls[:]
    if sver_weighting:
        fsv_col_lbls += [hstypes.FiltKeys.HOMOGERR]

    for sv_tup, daid, fm, fsv, fk in zip(svtup_list_, daid_list, fm_list, fsv_list, fk_list):
        # Return the inliers to the homography from chip2 to chip1
        (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = sv_tup
        fm_SV  = fm.take(homog_inliers, axis=0)
        fsv_SV = fsv.take(homog_inliers, axis=0)
        fk_SV  = fk.take(homog_inliers, axis=0)
        if sver_weighting:
            # Rescore based on homography errors
            #xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
            xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
            homog_xy_errors = homog_errors[0].take(homog_inliers, axis=0)
            homog_err_weight = (1.0 - np.sqrt(homog_xy_errors / xy_thresh_sqrd))
            #with ut.EmbedOnException():
            homog_err_weight.shape = (homog_err_weight.size, 1)
            fsv_SV = np.concatenate((fsv_SV, homog_err_weight), axis=1)
            #fsv_SV = np.hstack((fsv_SV, homog_err_weight))
        sver_matchtup_list.append((fm_SV, fsv_SV, fk_SV, H))

    fm_list_SV  = ut.get_list_column(sver_matchtup_list, 0)
    fsv_list_SV = ut.get_list_column(sver_matchtup_list, 1)
    fk_list_SV  = ut.get_list_column(sver_matchtup_list, 2)
    H_list_SV   = ut.get_list_column(sver_matchtup_list, 3)

    cmSV = chip_match.ChipMatch2(
        qaid=cm.qaid, daid_list=daid_list,
        fm_list=fm_list_SV, fsv_list=fsv_list_SV, fk_list=fk_list_SV,
        H_list=H_list_SV, dnid_list=dnid_list, qnid=cm.qnid,
        fsv_col_lbls=fsv_col_lbls)
    return cmSV  # , daid2_svtup


def compute_matching_dlen_extent(qreq_, fm_list, kpts_list):
    """
    helper for spatial verification, computes the squared diagonal length of
    matching chips

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-compute_matching_dlen_extent

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST')
        >>> verbose = True
        >>> cm = cm_list[0]
        >>> cm.sortself()
        >>> fm_list = cm.fm_list
        >>> kpts_list = qreq_.ibs.get_annot_kpts(cm.daid_list, qreq_=qreq_)
        >>> topx2_dlen_sqrd = compute_matching_dlen_extent(qreq_, fm_list, kpts_list)
        >>> ut.assert_inbounds(np.sqrt(topx2_dlen_sqrd)[0:5], 600, 800)

    """
    # Use extent of matching keypoints
    # first get matching keypoints
    fx2_list = [fm.T[1] for fm in fm_list]
    kpts2_m_list = [kpts.take(fx2, axis=0)
                    for (kpts, fx2) in zip(kpts_list, fx2_list)]
    dlen_sqrd_list = [ktool.get_kpts_dlen_sqrd(kpts2_m)
                      for kpts2_m in kpts2_m_list]
    return dlen_sqrd_list


#============================
# 5.5ish) Vsone Reranking
#============================


def vsone_reranking(qreq_, cm_list, verbose=VERB_PIPELINE):
    """
    CommandLine:
        python -m ibeis.model.hots.pipeline --test-vsone_reranking
        python -m ibeis.model.hots.pipeline --test-vsone_reranking --show

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> cfgdict = dict(prescore_method='nsum', score_method='nsum', vsone_reranking=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict, qaid_list=[2])
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'vsone_reranking')
        >>> cm_list = locals_['cm_list_SVER']
        >>> verbose = True
        >>> cm_list_VSONE = vsone_reranking(qreq_, cm_list, verbose=verbose)
        >>> if ut.show_was_requested():
        >>>     from ibeis.model.hots import vsone_pipeline
        >>>     import plottool as pt
        >>>     # NOTE: the aid2_score field must have been hacked
        >>>     vsone_pipeline.show_top_chipmatches(ibs, cm_list, 0,  'prescore')
        >>>     vsone_pipeline.show_top_chipmatches(ibs, cm_list_VSONE,   1, 'vsone-reranked')
        >>>     pt.show_if_requested()
    """
    from ibeis.model.hots import vsone_pipeline
    if verbose:
        print('Step 5.5ish) vsone reranking')
    cm_list_VSONE = vsone_pipeline.vsone_reranking(qreq_, cm_list, verbose)
    return cm_list_VSONE


#============================
# 6) Query Result Format
#============================


@profile
def chipmatch_to_resdict(qreq_, cm_list, verbose=VERB_PIPELINE):
    """
    Converts a dictionary of cmtup_old tuples into a dictionary of query results

    Args:
        cm_list (dict):
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        qaid2_qres

    CommandLine:
        utprof.py -m ibeis.model.hots.pipeline --test-chipmatch_to_resdict
        python -m ibeis.model.hots.pipeline --test-chipmatch_to_resdict
        python -m ibeis.model.hots.pipeline --test-chipmatch_to_resdict:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1, 5])
        >>> qaid2_qres = chipmatch_to_resdict(qreq_, cm_list)
        >>> qres = qaid2_qres[1]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> cfgdict = dict(sver_weighting=True)
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1, 2], cfgdict=cfgdict)
        >>> qaid2_qres = chipmatch_to_resdict(qreq_, cm_list)
        >>> qres = qaid2_qres[1]
        >>> num_filtkeys = len(qres.filtkey_list)
        >>> ut.assert_eq(num_filtkeys, qres.aid2_fsv[2].shape[1])
        >>> ut.assert_eq(num_filtkeys, 3)
        >>> ut.assert_inbounds(qres.aid2_fsv[2].shape[0], 105, 150)
        >>> assert np.all(qres.aid2_fs[2] == qres.aid2_fsv[2].prod(axis=1)), 'math is broken'

    """
    if verbose:
        print('[hs] Step 6) Convert cmtup_old -> qres')
    # Matchable daids
    external_qaids   = qreq_.get_external_qaids()
    # Create the result structures for each query.
    qres_list = qreq_.make_empty_query_results()
    # Perform final scoring
    # TODO: only score if already unscored
    score_method = qreq_.qparams.score_method
    scoring.score_chipmatch_list(qreq_, cm_list, score_method)
    # Normalize scores if requested
    if qreq_.qparams.score_normalization:
        normalizer = qreq_.normalizer
        for cm in cm_list:
            cm.prob_list = normalizer.normalize_score_list(cm.score_list)
    for qaid, qres, cm in zip(external_qaids, qres_list, cm_list):
        assert qaid == cm.qaid
        assert qres.qaid == qaid
        #ut.assert_eq(qaid, cm.qaid)
        qres.filtkey_list = cm.fsv_col_lbls
        aid2_fm    = dict(zip(cm.daid_list, cm.fm_list))
        aid2_fsv   = dict(zip(cm.daid_list, cm.fsv_list))
        aid2_fs    = dict(zip(cm.daid_list, [fsv.prod(axis=1) for fsv in cm.fsv_list]))
        aid2_fk    = dict(zip(cm.daid_list, cm.fk_list))
        aid2_score = dict(zip(cm.daid_list, cm.score_list))
        aid2_H     = None if cm.H_list is None else dict(zip(cm.daid_list, cm.H_list))
        aid2_prob  = None if cm.prob_list is None else dict(zip(cm.daid_list, cm.p))
        qres.aid2_fm    = aid2_fm
        qres.aid2_fsv   = aid2_fsv
        qres.aid2_fs    = aid2_fs
        qres.aid2_fk    = aid2_fk
        qres.aid2_score = aid2_score
        qres.aid2_prob = aid2_prob
        qres.aid2_H = aid2_H
    # Build dictionary structure to maintain functionality
    qaid2_qres = {qaid: qres for qaid, qres in zip(external_qaids, qres_list)}
    return qaid2_qres


#============================
# Result Caching
#============================


#@ut.indent_func('[tlr]')
@profile
def try_load_resdict(qreq_, force_miss=False, verbose=VERB_PIPELINE):
    """
    Try and load the result structures for each query.
    returns a list of failed qaids
    """
    qaids   = qreq_.get_external_qaids()
    qauuids = qreq_.get_external_quuids()
    daids   = qreq_.get_external_daids()

    cfgstr = qreq_.get_cfgstr()
    qresdir = qreq_.get_qresdir()
    qaid2_qres_hit = {}
    #cachemiss_qaids = []
    # TODO: could prefiler paths that don't exist
    for qaid, qauuid in zip(qaids, qauuids):
        qres = hots_query_result.QueryResult(qaid, qauuid, cfgstr, daids)
        try:
            qres.load(qresdir, force_miss=force_miss, verbose=verbose)  # 77.4 % time
        except (hsexcept.HotsCacheMissError, hsexcept.HotsNeedsRecomputeError) as ex:
            if ut.VERYVERBOSE:
                ut.printex(ex, iswarning=True)
            #cachemiss_qaids.append(qaid)  # cache miss
        else:
            qaid2_qres_hit[qaid] = qres  # cache hit
    return qaid2_qres_hit  # , cachemiss_qaids


@profile
def save_resdict(qreq_, qaid2_qres, verbose=VERB_PIPELINE):
    """
    Saves a dictionary of query results to disk
    """
    qresdir = qreq_.get_qresdir()
    if verbose:
        print('[hs] saving %d query results' % len(qaid2_qres))
    for qres in six.itervalues(qaid2_qres):
        qres.save(qresdir)


#============================
# Testdata
#============================


if __name__ == '__main__':
    """
    python -m ibeis.model.hots.pipeline --verb-test
    python -m ibeis.model.hots.pipeline --test-build_chipmatches
    python -m ibeis.model.hots.pipeline --test-spatial-verification
    python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0 --show
    python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0:0 --show
    python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0:1 --show --db NAUT_test
    python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0:1 --db NAUT_test --noindent
    python -m ibeis.model.hots.pipeline --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
    if ut.get_argflag('--show'):
        from plottool import df2
        exec(df2.present())
