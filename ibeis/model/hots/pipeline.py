"""
Hotspotter pipeline module

Module Concepts::

    PREFIXES:
    qaid2_XXX - prefix mapping query chip index to
    qfx2_XXX  - prefix mapping query chip feature index to

    TUPLES::
     * nns    - a (qfx2_idx, qfx2_dist) tuple
     * nnfilt - a (qfx2_fs, qfx2_valid) tuple

    SCALARS::
     * idx    - the index into the nnindexers descriptors
     * dist   - the distance to a corresponding feature
     * fs     - a score of a corresponding feature
     * valid  - a valid bit for a corresponding feature

    PIPELINE_VARS::
    qaid2_nns - maping from query chip index to nns
    {
     * qfx2_idx   - ranked list of query feature indexes to database feature indexes
     * qfx2_dist - ranked list of query feature indexes to database feature indexes
    }

    * qaid2_norm_weight - mapping from qaid to (qfx2_normweight, qfx2_selnorm)
             = qaid2_nnfilt[qaid]


"""

from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
import six
from collections import defaultdict
# Scientific
import numpy as np
import vtool as vt
from vtool import keypoint as ktool
from vtool import spatial_verification as sver
# Hotspotter
from ibeis.model.hots import hots_query_result
from ibeis.model.hots import hstypes
#from ibeis.model.hots import coverage_image
from ibeis.model.hots import nn_weights
from ibeis.model.hots import voting_rules2 as vr2
from ibeis.model.hots import exceptions as hsexcept
import utool as ut
from functools import partial
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[hs]', DEBUG=False)


TAU = 2 * np.pi  # References: tauday.com
NOT_QUIET = ut.NOT_QUIET and not ut.get_argflag('--quiet-query')
DEBUG_PIPELINE = ut.get_argflag(('--debug-pipeline', '--debug-pipe'))
VERB_PIPELINE =  NOT_QUIET and (ut.VERBOSE or ut.get_argflag(('--verbose-pipeline', '--verb-pipe')))
VERYVERBOSE_PIPELINE = ut.get_argflag(('--very-verbose-pipeline', '--very-verb-pipe'))

#=================
# Globals
#=================

START_AFTER = 2


# specialized progress func
log_progress = partial(ut.log_progress, startafter=START_AFTER, disable=ut.QUIET)


# Query Level 0
#@profile
#@ut.indent_func('[Q0]')
@profile
def request_ibeis_query_L0(ibs, qreq_, verbose=VERB_PIPELINE):
    r"""
    Driver logic of query pipeline

    Args:
        ibs   (IBEISController): IBEIS database object to be queried
        qreq_ (QueryRequest): hyper-parameters. use ``prep_qreq`` to create one

    Returns:
        (dict of QueryResult): qaid2_qres mapping from query indexes to Query Result Objects

    Example:
        >>> # one-vs-many:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict = dict(codename='vsmany')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict)
        >>> print(qreq_.qparams.query_cfgstr)
        >>> qaid2_qres = pipeline.request_ibeis_query_L0(ibs, qreq_)
        >>> qres = qaid2_qres[list(qaid2_qres.keys())[0]]
        >>> if ut.get_argflag('--show') or ut.inIPython():
        ...     qres.show_analysis(ibs, fnum=0, make_figtitle=True)
        >>> print(qres.get_inspect_str())

    Example:
        >>> # one-vs-one:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.model.hots import pipeline
        >>> # pipeline.rrr()
        >>> cfgdict1 = dict(codename='vsone', sv_on=False)
        >>> ibs1, qreq_1 = pipeline.get_pipeline_testdata(cfgdict=cfgdict1)
        >>> print(qreq_1.qparams.query_cfgstr)
        >>> qaid2_qres1 = pipeline.request_ibeis_query_L0(ibs1, qreq_1)
        >>> qres1 = qaid2_qres1[list(qaid2_qres1.keys())[0]]
        >>> if ut.get_argflag('--show') or ut.inIPython():
        ...     qres1.show_analysis(ibs1, fnum=1, make_figtitle=True)
        >>> print(qres1.get_inspect_str())

    python -m ibeis.model.hots.pipeline --test-request_ibeis_query_L0

    """
    # Load data for nearest neighbors
    #if qreq_.qparams.pipeline_root == 'vsone':
    #    ut.embed()

    qreq_.lazy_load(verbose=verbose)

    if verbose:
        print('\n\n[hs] +--- STARTING HOTSPOTTER PIPELINE ---')
        print('[hs] * len(internal_qaids) = %r' % len(qreq_.internal_qaids))
        print('[hs] * len(internal_daids) = %r' % len(qreq_.internal_daids))

    if qreq_.qparams.pipeline_root == 'smk':
        from ibeis.model.hots.smk import smk_match
        # Alternative to naive bayes matching:
        # Selective match kernel
        qaid2_scores, qaid2_chipmatch_FILT_ = smk_match.execute_smk_L5(qreq_)
    elif qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
        # Nearest neighbors (qaid2_nns)
        # * query descriptors assigned to database descriptors
        # * FLANN used here
        qaid2_nns_ = nearest_neighbors(qreq_, verbose=verbose)

        # Remove Impossible Votes
        qaid2_nnfilt0_ = baseline_neighbor_filter(qaid2_nns_, qreq_, verbose=verbose)

        # Nearest neighbors weighting / scoring (filt2_weights)
        # * feature matches are weighted
        filt2_weights_ = weight_neighbors(qaid2_nns_, qaid2_nnfilt0_, qreq_, verbose=verbose)

        # Thresholding and combine weights into a score
        # * nnfilt = (qfx2_valid, qfx2_score)
        qaid2_nnfilt_ = filter_neighbors(qaid2_nns_, qaid2_nnfilt0_,
                                         filt2_weights_, qreq_, verbose=verbose)

        # Nearest neighbors to chip matches (qaid2_chipmatch)
        # * Inverted index used to create aid2_fmfsfk (TODO: aid2_fmfv)
        # * Initial scoring occurs
        # * vsone inverse swapping occurs here
        qaid2_chipmatch_FILT_ = build_chipmatches(qaid2_nns_, qaid2_nnfilt_,
                                                  qreq_, verbose=verbose)
    else:
        print('invalid pipeline root %r' % (qreq_.qparams.pipeline_root))

    # Spatial verification (qaid2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    qaid2_chipmatch_SVER_ = spatial_verification(qaid2_chipmatch_FILT_, qreq_,
                                                 verbose=verbose)

    # Query results format (qaid2_qres)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qaid2_qres_ = chipmatch_to_resdict(qaid2_chipmatch_SVER_, qreq_,
                                       verbose=verbose)

    if VERB_PIPELINE:
        print('[hs] L___ FINISHED HOTSPOTTER PIPELINE ___')

    return qaid2_qres_

#============================
# 1) Nearest Neighbors
#============================


#@ut.indent_func('[nn]')
@profile
def nearest_neighbors(qreq_, verbose=VERB_PIPELINE):
    """
    Plain Nearest Neighbors

    Args:
        qreq_  (QueryRequest): hyper-parameters

    Returns:
        dict: qaid2_nnds - a dict mapping query annnotation-ids to a nearest
            neighbor tuple (indexes, dists). indexes and dist have the shape
            (nDesc x K) where nDesc is the number of descriptors in the
            annotation, and K is the number of approximate nearest neighbors.


    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> pipeline.rrr()
        >>> #cfgdict = dict(codename='vsone')
        >>> cfgdict = dict(codename='nsum')
        >>> dbname = 'testdb1'
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname, cfgdict=cfgdict)
        >>> qaid2_nns = pipeline.nearest_neighbors(qreq_)
        >>> # Asserts
        >>> assert list(qaid2_nns.keys()) == qreq_.get_external_qaids().tolist()
        >>> tup = list(qaid2_nns.values())[0]
        >>> assert tup[0].shape == tup[1].shape
    """
    # Neareset neighbor configuration
    K      = qreq_.qparams.K
    Knorm  = qreq_.qparams.Knorm
    checks = qreq_.qparams.checks
    if verbose:
        print('[hs] Step 1) Assign nearest neighbors: ' + qreq_.qparams.nn_cfgstr)
    num_neighbors = K + Knorm  # number of nearest neighbors
    qvecs_list = qreq_.ibs.get_annot_vecs(qreq_.get_internal_qaids())  # query descriptors
    # Allocate numpy array for each query annotation
    # TODO: dtype=np.ndarray is just an object, might be useful to use
    # pointers?
    nQAnnots = len(qvecs_list)
    nn_idxs_arr   = np.empty(nQAnnots, dtype=np.ndarray)  # database indexes
    nn_dists_arr = np.empty(nQAnnots, dtype=np.ndarray)  # corresponding distance
    # Internal statistics reporting
    nTotalNN, nTotalDesc = 0, 0
    mark_, end_ = log_progress('Assign NN: ', len(qvecs_list))
    qvec_iter = ut.ProgressIter(qvecs_list, lbl='Assign NN: ', freq=20, time_thresh=2.0)
    for count, qfx2_vec in enumerate(qvec_iter):
        # Check that we can query this annotation
        if len(qfx2_vec) == 0:
            # Assign empty nearest neighbors
            (qfx2_idx, qfx2_dist) = qreq_.indexer.empty_neighbors(num_neighbors)
        else:
            # Find Neareset Neighbors nntup = (indexes, dists)
            (qfx2_idx, qfx2_dist) = qreq_.indexer.knn(qfx2_vec, num_neighbors, checks)
            nTotalNN += qfx2_idx.size
            nTotalDesc += len(qfx2_vec)
        # record number of query and result desc
        nn_idxs_arr[count]   = qfx2_idx
        nn_dists_arr[count] = qfx2_dist
    if verbose:
        print('[hs] * assigned %d desc (from %d annots) to %r nearest neighbors'
              % (nTotalDesc, nQAnnots, nTotalNN))
    #return nn_idxs_arr, nn_dists_arr
    # Return old style dicts for now
    qaids = qreq_.get_internal_qaids()
    qaid2_nns_ = {aid: (qfx2_idx, qfx2_dist) for (aid, qfx2_idx, qfx2_dist) in
                  zip(qaids, nn_idxs_arr, nn_dists_arr)}

    if qreq_.qparams.with_metadata:
        qreq_.metadata['nns'] = qaid2_nns_
    return qaid2_nns_

#============================
# 1.5) Remove Impossible Weights
#============================


def baseline_neighbor_filter(qaid2_nns, qreq_, verbose=VERB_PIPELINE):
    """
    Returns:
        qaid2_nnfilt0 : mapping from qaid to (qfx2_valid, qfx2_score) tuple

    Example:
        >>> from ibeis.model.hots.pipeline import *   # NOQA
        >>> from ibeis.model.hots import pipeline
        >>> import ibeis
        >>> cfgdict = dict(codename='nsum')
        >>> dbname = 'testdb1'
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname, cfgdict=cfgdict)
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> pipeline.test_pipeline_upto(ibs, qreq_, stop_node='baseline_neighbor_filter')
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'baseline_neighbor_filter')
        >>> args = [locals_[key] for key in ['qaid2_nns']]
        >>> qaid2_nns, = args
        >>> qaid2_nnfilt0 = baseline_neighbor_filter(qaid2_nns, qreq_)

    Removes matches to self, the same image, or the same name.
    """
    if verbose:
        print('[hs] Step 1.5) Baseline neighbor filter')
    cant_match_sameimg  = not qreq_.qparams.can_match_sameimg
    cant_match_samename = not qreq_.qparams.can_match_samename
    cant_match_self     = not cant_match_sameimg
    K = qreq_.qparams.K

    qaid2_nnfilt0 = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        (qfx2_idx, _) = qaid2_nns[qaid]
        qfx2_nnidx = qfx2_idx.T[0:K].T
        nnfilt = remove_impossible_votes(qaid, qfx2_nnidx, qreq_, cant_match_self,
                                         cant_match_sameimg,
                                         cant_match_samename, verbose=verbose)
        qaid2_nnfilt0[qaid] = nnfilt
    return qaid2_nnfilt0


def remove_impossible_votes(qaid, qfx2_nnidx, qreq_, cant_match_self,
                            cant_match_sameimg, cant_match_samename,
                            verbose=VERB_PIPELINE):
    """
    Remove matches to self or same image
    """
    # Baseline is all matches have score 1 and all matches are valid
    qfx2_valid = np.ones(qfx2_nnidx.shape, dtype=np.bool)
    qfx2_score = np.ones(qfx2_nnidx.shape, dtype=hstypes.FS_DTYPE)

    # Get neighbor annotation information
    qfx2_aid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    # dont vote for yourself or another chip in the same image
    if cant_match_self:
        qfx2_notsamechip = qfx2_aid != qaid
        if DEBUG_PIPELINE:
            __self_verbose_check(qfx2_notsamechip, qfx2_valid)
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
    if cant_match_sameimg:
        qfx2_gid = qreq_.ibs.get_annot_gids(qfx2_aid)
        qgid     = qreq_.ibs.get_annot_gids(qaid)
        qfx2_notsameimg = qfx2_gid != qgid
        if DEBUG_PIPELINE:
            __sameimg_verbose_check(qfx2_notsameimg, qfx2_valid)
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
    if cant_match_samename:
        # This should probably be off
        qfx2_nid = qreq_.ibs.get_annot_name_rowids(qfx2_aid)
        qnid = qreq_.ibs.get_annot_name_rowids(qaid)
        qfx2_notsamename = qfx2_nid != qnid
        if DEBUG_PIPELINE:
            __samename_verbose_check(qfx2_notsamename, qfx2_valid)
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
    nnfilt = (qfx2_score, qfx2_valid)
    return nnfilt


def __self_verbose_check(qfx2_notsamechip, qfx2_valid):
    nInvalidChips = ((True - qfx2_notsamechip)).sum()
    nNewInvalidChips = (qfx2_valid * (True - qfx2_notsamechip)).sum()
    total = qfx2_valid.size
    print('[hs] * self invalidates %d/%d assignments' % (nInvalidChips, total))
    print('[hs] * %d are newly invalided by self' % (nNewInvalidChips))


def __samename_verbose_check(qfx2_notsamename, qfx2_valid):
    nInvalidNames = ((True - qfx2_notsamename)).sum()
    nNewInvalidNames = (qfx2_valid * (True - qfx2_notsamename)).sum()
    total = qfx2_valid.size
    print('[hs] * nid invalidates %d/%d assignments' % (nInvalidNames, total))
    print('[hs] * %d are newly invalided by nid' % nNewInvalidNames)


def __sameimg_verbose_check(qfx2_notsameimg, qfx2_valid):
    nInvalidImgs = ((True - qfx2_notsameimg)).sum()
    nNewInvalidImgs = (qfx2_valid * (True - qfx2_notsameimg)).sum()
    total = qfx2_valid.size
    print('[hs] * gid invalidates %d/%d assignments' % (nInvalidImgs, total))
    print('[hs] * %d are newly invalided by gid' % nNewInvalidImgs)


def identity_filter(qaid2_nns, qreq_):
    """ testing function returns unfiltered nearest neighbors
    this does check that you are not matching yourself
    """
    K = qreq_.qparams.K
    qaid2_nnfilt = {}
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        (qfx2_idx, _) = qaid2_nns[qaid]
        qfx2_nnidx = qfx2_idx[:, 0:K]
        qfx2_score = np.ones(qfx2_nnidx.shape, dtype=hstypes.FS_DTYPE)
        qfx2_valid = np.ones(qfx2_nnidx.shape, dtype=np.bool)
        # Check that you are not matching yourself
        qfx2_aid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
        qfx2_notsamechip = qfx2_aid != qaid
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        qaid2_nnfilt[qaid] = (qfx2_score, qfx2_valid)
    return qaid2_nnfilt


#============================
# 2) Nearest Neighbor weights
#============================


#@ut.indent_func('[wn]')
def weight_neighbors(qaid2_nns, qaid2_nnfilt0, qreq_, verbose=VERB_PIPELINE):
    """
    PIPELINE NODE 3

    Args:
        qaid2_nns (dict):
        qaid2_nnfilt0 (dict):
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        dict : filt2_weights

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> pipeline.rrr()
        >>> #cfgdict = dict(codename='vsone')
        >>> cfgdict = dict(codename='nsum')
        >>> # dbname = 'GZ_ALL'  # 'testdb1'
        >>> dbname = 'testdb1'
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname, cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'weight_neighbors')
        >>> args = [locals_[key] for key in ['qaid2_nns', 'qaid2_nnfilt0']]
        >>> qaid2_nns, qaid2_nnfilt0  = args
        >>> filt2_weights = pipeline.weight_neighbors(qaid2_nns, qaid2_nnfilt0, qreq_)
    """
    if verbose:
        print('[hs] Step 2) Weight neighbors: ' + qreq_.qparams.filt_cfgstr)
    if not qreq_.qparams.filt_on:
        filt2_weights = {}
    else:
        # Remove impossible votes (votes for chips is the same image)
        nnweight_list = qreq_.qparams.active_filter_list
        # Prealloc output
        filt2_weights = {nnweight: None for nnweight in nnweight_list}
        # Buidl output
        for nnweightkey in nnweight_list:
            nn_filter_fn = nn_weights.NN_WEIGHT_FUNC_DICT[nnweightkey]
            # Apply [nnweightkey] weight to each nearest neighbor
            qaid2_norm_weight = nn_filter_fn(qaid2_nns, qaid2_nnfilt0, qreq_)
            filt2_weights[nnweightkey] = qaid2_norm_weight
    return filt2_weights


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


#@ut.indent_func('[fn]')
@profile
def filter_neighbors(qaid2_nns, qaid2_nnfilt0, filt2_weights, qreq_, verbose=VERB_PIPELINE):
    """
    Args:
        qaid2_nns (dict):
        filt2_weights (dict):
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        qaid2_nnfilt

    Example:
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> pipeline.rrr()
        >>> cfgdict = dict(dupvote_weight=1.0)
        >>> cfgdict = dict(codename='vsone')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'filter_neighbors')
        >>> args = [locals_[key] for key in ['qaid2_nns', 'qaid2_nnfilt0', 'filt2_weights']]
        >>> qaid2_nns, qaid2_nnfilt0, filt2_weights = args
        >>> qaid2_nnfilt = filter_neighbors(qaid2_nns, qaid2_nnfilt0, filt2_weights, qreq_)
    """
    qaid2_nnfilt = {}
    # Configs
    K = qreq_.qparams.K
    if verbose:
        print('[hs] Step 3) Filter neighbors: ')
    # Filter matches based on config and weights
    mark_, end_ = log_progress('Filter NN: ', len(qaid2_nns))
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        mark_(count)  # progress
        (qfx2_idx, _) = qaid2_nns[qaid]
        qfx2_nnidx = qfx2_idx.T[0:K].T
        # Baseline is all matches have score 1 and all matches are valid

        #if qreq_.qparams.score_method == 'nsum':
        qfx2_score0, qfx2_valid0 = qaid2_nnfilt0[qaid]

        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _threshold_and_scale_weights(qaid, qfx2_nnidx, filt2_weights,
                                                              qfx2_score0, qfx2_valid0, qreq_)
        if DEBUG_PIPELINE:
            print('\n[hs] * %d assignments are invalid by filter thresholds' %
                  ((True - qfx2_valid).sum()))
        if qreq_.qparams.gravity_weighting:
            raise NotImplementedError('have not finished gv weighting')
            #from vtool import linalg as ltool
            #qfx2_nnkpts = qreq_.indexer.get_nn_kpts(qfx2_nnidx)
            #qfx2_nnori = ktool.get_oris(qfx2_nnkpts)
            #qfx2_kpts  = qreq_.ibs.get_annot_kpts(qaid)  # FIXME: Highly inefficient
            #qfx2_oris  = ktool.get_oris(qfx2_kpts)
            ## Get the orientation distance
            #qfx2_oridist = ltool.rowwise_oridist(qfx2_nnori, qfx2_oris)
            ## Normalize into a weight (close orientations are 1, far are 0)
            #qfx2_gvweight = (TAU - qfx2_oridist) / TAU
            ## Apply gravity vector weight to the score
            #qfx2_score *= qfx2_gvweight
        #printDBG('[hs] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        nnfilt = (qfx2_score, qfx2_valid)
        qaid2_nnfilt[qaid] = nnfilt
    end_()
    return qaid2_nnfilt


#@ut.indent_func('[_tsw]')
@profile
def _threshold_and_scale_weights(qaid, qfx2_nnidx, filt2_weights, qfx2_score0,
                                 qfx2_valid0, qreq_):
    """
    helper function _threshold_and_scale_weights

    qfx2_score is an ndarray containing the score of individual feature matches.
    qfx2_valid marks if that score will be thresholded.

    Args:
        qaid (int): query annotation id
        qfx2_nnidx (dict):
        filt2_weights (dict):
        qreq_ (QueryRequest): hyper-parameters

    Return:
        tuple : (qfx2_score, qfx2_valid)
    """
    qfx2_score = qfx2_score0.copy()
    qfx2_valid = qfx2_valid0.copy()
    # Apply the filter weightings to determine feature validity and scores
    for filt, aid2_weights in six.iteritems(filt2_weights):
        qfx2_weights = aid2_weights[qaid]
        sign, thresh, weight = qreq_.qparams.filt2_stw[filt]  # stw = sign, thresh, weight
        if thresh is not None:
            # Filter if threshold is specified
            qfx2_trueweights = sign * qfx2_weights
            truethresh = sign * thresh
            qfx2_passed = qfx2_trueweights <= truethresh
            #if DEBUG_PIPELINE:
            #    print('[pipe.thresh] TEST(%r): (truethresh:%r) <= (qfx2_trueweights.min()=%r)' %
            #            (filt, truethresh, qfx2_trueweights.min()))
            #    print('[pipe.thresh] * %d matches passed' % (qfx2_passed.sum()))
            #    print('[pipe.thresh] * and %d matches are now valid' % (qfx2_valid.sum(),))
            qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if weight != 0:
            # Score if weight is specified
            qfx2_score *= (weight * qfx2_weights)
    nnfilt = (qfx2_score, qfx2_valid)
    return nnfilt


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> aid2
#============================


@profile
def _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk):
    minMatches = 2  # TODO: paramaterize
    # Convert to numpy
    fm_dtype = hstypes.FM_DTYPE
    fs_dtype = hstypes.FS_DTYPE
    fk_dtype = hstypes.FK_DTYPE
    # FIXME: This is slow
    aid2_fm_ = {aid: np.array(fm, fm_dtype)
                for aid, fm in six.iteritems(aid2_fm)
                if len(fm) > minMatches}
    aid2_fs_ = {aid: np.array(fs, fs_dtype)
                for aid, fs in six.iteritems(aid2_fs)
                if len(fs) > minMatches}
    aid2_fk_ = {aid: np.array(fk, fk_dtype)
                for aid, fk in six.iteritems(aid2_fk)
                if len(fk) > minMatches}
    # Ensure shape
    for aid, fm in six.iteritems(aid2_fm_):
        fm.shape = (fm.size // 2, 2)
    chipmatch = (aid2_fm_, aid2_fs_, aid2_fk_)
    return chipmatch


def new_fmfsfk():
    aid2_fm = defaultdict(list)
    aid2_fs = defaultdict(list)
    aid2_fk = defaultdict(list)
    return aid2_fm, aid2_fs, aid2_fk


#@ut.indent_func('[bc]')
@profile
def build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq_, verbose=VERB_PIPELINE):
    """
    Args:
        qaid2_nns    : dict of assigned nearest features (only indexes are used here)
        qaid2_nnfilt : dict of (featmatch_scores, featmatch_mask)
                        where the scores and matches correspond to the assigned
                        nearest features
        qreq_(QueryRequest) : hyper-parameters

    Returns:
        qaid2_chipmatch : feat match, feat score, feat rank

    Notes:
        The prefix ``qaid2_`` denotes a mapping where keys are query-annotation-id

        vsmany/vsone counts here. also this is where the filter
        weights and thershold are applied to the matches. Essientally
        nearest neighbors are converted into weighted assignments

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict = dict(dupvote_weight=1.0)
        >>> cfgdict = dict(codename='vsone')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('NAUT_test', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'build_chipmatches')
        >>> qaid2_nns, qaid2_nnfilt = [locals_[key] for key in ['qaid2_nns', 'qaid2_nnfilt']]
        >>> qaid2_chipmatch = build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq_)
    """

    # Config
    K = qreq_.qparams.K
    is_vsone =  qreq_.qparams.vsone
    if verbose:
        pipeline_root = qreq_.qparams.pipeline_root
        print('[hs] Step 4) Building chipmatches %s' % (pipeline_root,))
    # Return var
    qaid2_chipmatch = {}
    nFeatMatches = 0
    #Vsone
    if is_vsone:
        assert len(qreq_.get_external_qaids()) == 1
        assert len(qreq_.get_internal_daids()) == 1
        aid2_fm, aid2_fs, aid2_fk = new_fmfsfk()
    # Iterate over chips with nearest neighbors
    mark_, end_ = log_progress('Build Chipmatch: ', len(qaid2_nns))

    # Iterate over INTERNAL query annotation ids
    qaid_iter = ut.ProgressIter(six.iterkeys(qaid2_nns), nTotal=len(qaid2_nns), lbl='Build Chipmatch: ', freq=20, time_thresh=2.0)
    #for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
    #    mark_(count)  # Mark progress
    for qaid in qaid_iter:
        (qfx2_idx, _) = qaid2_nns[qaid]
        (qfx2_fs, qfx2_valid) = qaid2_nnfilt[qaid]
        nQKpts = qfx2_idx.shape[0]
        # Build feature matches
        qfx2_nnidx = qfx2_idx.T[0:K].T
        qfx2_aid  = qreq_.indexer.get_nn_aids(qfx2_nnidx)
        qfx2_fx   = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
        # FIXME: Can probably get away without using tile here
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        # Pack valid feature matches into an interator
        valid_lists = (qfx2[qfx2_valid] for qfx2 in (qfx2_qfx, qfx2_aid, qfx2_fx, qfx2_fs, qfx2_k,))
        # TODO: Sorting the valid lists by aid might help the speed of this
        # code. Also, consolidating fm, fs, and fk into one vector will reduce
        # the amount of appends.
        match_iter = zip(*valid_lists)

        #-----
        # Vsmany - Append query feature matches to database aids
        if not is_vsone:
            aid2_fm, aid2_fs, aid2_fk = new_fmfsfk()
            for qfx, aid, fx, fs, fk in match_iter:
                aid2_fm[aid].append((qfx, fx))  # Note the difference
                aid2_fs[aid].append(fs)
                aid2_fk[aid].append(fk)
                nFeatMatches += 1
            chipmatch = _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk)
            qaid2_chipmatch[qaid] = chipmatch
            #if DEBUG_PIPELINE:
            #    nFeats_in_matches = [len(fm) for fm in six.itervalues(aid2_fm)]
            #    print('nFeats_in_matches_stats = ' +
            #          ut.dict_str(ut.get_stats(nFeats_in_matches)))
        #L_____

        #-----
        # Vsone - Append database feature matches to query aids
        else:
            for qfx, aid, fx, fs, fk in match_iter:
                # Remember in vsone internal qaids = external daids
                aid2_fm[qaid].append((fx, qfx))  # Note the difference
                aid2_fs[qaid].append(fs)
                aid2_fk[qaid].append(fk)
                nFeatMatches += 1
        #L_____
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk)
        qaid = qreq_.get_external_qaids()[0]
        qaid2_chipmatch[qaid] = chipmatch
    if verbose:
        print('[hs] * made %d feat matches' % nFeatMatches)
    return qaid2_chipmatch


def assert_qaid2_chipmatch(ibs, qreq_, qaid2_chipmatch):
    """ Runs consistency check """
    external_qaids = qreq_.get_external_qaids().tolist()
    external_daids = qreq_.get_external_daids().tolist()

    if len(external_qaids) == 1 and qreq_.qparams.pipeline_root == 'vsone':
        nExternalQVecs = ibs.get_annot_vecs(external_qaids[0]).shape[0]
        assert qreq_.indexer.idx2_vec.shape[0] == nExternalQVecs, 'did not index query descriptors properly'

    assert external_qaids == list(qaid2_chipmatch.keys()), 'bad external qaids'
    # Loop over internal qaids
    for qaid, chipmatch in qaid2_chipmatch.iteritems():
        nQVecs = ibs.get_annot_vecs(qaid).shape[0]  # NOQA
        (daid2_fm, daid2_fs, daid2_fk) = chipmatch
        assert external_daids.tolist() == list(daid2_fm.keys())
        daid2_fm

        pass


#============================
# 5) Spatial Verification
#============================


#@ut.indent_func('[sv]')
def spatial_verification(qaid2_chipmatch, qreq_, verbose=VERB_PIPELINE):
    """
    Args:
        qaid2_chipmatch (dict):
        qreq_ (QueryRequest): hyper-parameters
        dbginfo (bool):

    Returns:
        dict or tuple(dict, dict)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'spatial_verification')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_FILT']
        >>> spatial_verification(qaid2_chipmatch, qreq_)
    """
    if not qreq_.qparams.sv_on or qreq_.qparams.xy_thresh is None:
        if verbose:
            print('[hs] Step 5) Spatial verification: off')
        return qaid2_chipmatch
    else:
        qaid2_chipmatchSV = _spatial_verification(qaid2_chipmatch, qreq_, verbose=verbose)
        return qaid2_chipmatchSV


def get_prescore_shortlist(qaid, chipmatch, qreq_):
    """
    computes which of the annotations should continue in the next pipeline step
    """
    prescore_method = qreq_.qparams.prescore_method
    nShortlist      = qreq_.qparams.nShortlist
    daid2_prescore = score_chipmatch(qaid, chipmatch, prescore_method, qreq_)
    #print('Prescore: %r' % (daid2_prescore,))
    # HACK FOR NAME PRESCORING
    if prescore_method == 'nsum':
        topx2_aid = prescore_nsum(qreq_, daid2_prescore, nShortlist)
        nRerank = len(topx2_aid)
    else:
        topx2_aid = ut.util_dict.keys_sorted_by_value(daid2_prescore)[::-1]
        nRerank = min(len(topx2_aid), nShortlist)
    return topx2_aid, nRerank


#@ut.indent_func('[_sv]')
@profile
def _spatial_verification(qaid2_chipmatch, qreq_, verbose=VERB_PIPELINE):
    """
    make only spatially valid features survive

    Dev:
        >>> import pyflann
        >>> qaid = 1
        >>> daid = ibs.get_annot_groundtruth(qaid)[0]
        >>> qvecs = ibs.get_annot_vecs(qaid)
        >>> dvecs = ibs.get_annot_vecs(daid)
        >>> # Simple ratio-test matching
        >>> flann = pyflann.FLANN()
        >>> flann.build_index(dvecs)
        >>> qfx2_dfx, qfx2_dist = flann.nn_index(qvecs, 2)
        >>> ratio = (qfx2_dist.T[1] / qfx2_dist.T[0])
        >>> valid = ratio < 1.2
        >>> valid_qfx = np.where(valid)[0]
        >>> valid_dfx = qfx2_dfx.T[0][valid]
        >>> fm = np.vstack((valid_qfx, valid_dfx)).T
        >>> fs = ratio[valid]
        >>> fk = np.ones(fs.size)
        >>> qaid2_chipmatch = {qaid: ({daid: fm}, {daid: fs}, {daid: fk})}
        >>> qreq_ = query_request.new_ibeis_query_request(ibs, [qaid], [daid])
        >>> qreq_.ibs = ibs

    Example:
        >>> # one-vs-one
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> from ibeis.model.hots import pipeline
        >>> import ibeis
        >>> cfgdict = dict(codename='vsone')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('NAUT_test', cfgdict=cfgdict, daid_list='all')

    Example:
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> from ibeis.model.hots import pipeline
        >>> import ibeis
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict, daid_list='all')

    """
    # TODO: Make sure vsone isn't being messed up by some stupid assumption here
    # spatial verification
    if verbose:
        print('[hs] Step 5) Spatial verification: ' + qreq_.qparams.sv_cfgstr)
    xy_thresh       = qreq_.qparams.xy_thresh
    scale_thresh    = qreq_.qparams.scale_thresh
    ori_thresh      = qreq_.qparams.ori_thresh
    use_chip_extent = qreq_.qparams.use_chip_extent
    min_nInliers    = qreq_.qparams.min_nInliers
    sver_weighting  = qreq_.qparams.sver_weighting
    qaid2_chipmatchSV = {}
    nFeatSVTotal = 0
    nFeatMatchSV = 0
    if qreq_.qparams.with_metadata:
        qaid2_svtups = {}  # dbg info (can remove if there is a speed issue)
    qaid_iter = ut.ProgressIter(six.iterkeys(qaid2_chipmatch),
                                nTotal=len(qaid2_chipmatch), lbl='SVER: ',
                                freq=20, time_thresh=2.0)
    for qaid in qaid_iter:
        # Find a transform from chip2 to chip1 (the old way was 1 to 2)
        chipmatch = qaid2_chipmatch[qaid]
        topx2_aid, nRerank = get_prescore_shortlist(qaid, chipmatch, qreq_)
        (daid2_fm, daid2_fs, daid2_fk) = chipmatch
        # Precompute output container
        daid2_fm_V, daid2_fs_V, daid2_fk_V = new_fmfsfk()
        # Query Keypoints
        kpts1 = qreq_.ibs.get_annot_kpts(qaid)
        topx2_kpts = qreq_.ibs.get_annot_kpts(topx2_aid)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = precompute_topx2_dlen_sqrd(
            qreq_, daid2_fm, topx2_aid, topx2_kpts, nRerank, use_chip_extent)
        if qreq_.qparams.with_metadata:
            daid2_svtup = {}  # dbg info (can remove if there is a speed issue)
        # spatially verify the top __NUM_RERANK__ results
        for topx in range(nRerank):
            daid = topx2_aid[topx]
            fm = daid2_fm[daid]
            if len(fm) == 0:
                # skip results without any matches
                continue
            dlen_sqrd2 = topx2_dlen_sqrd[topx]
            kpts2 = topx2_kpts[topx]
            fs    = daid2_fs[daid]
            fk    = daid2_fk[daid]
            try:
                # Compute homography from chip2 to chip1
                sv_tup = sver.spatially_verify_kpts(
                    kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
                    dlen_sqrd2, min_nInliers,
                    returnAff=qreq_.qparams.with_metadata)
            except Exception as ex:
                ut.printex(ex, 'Unknown error in spatial verification.',
                              keys=['kpts1', 'kpts2',  'fm', 'xy_thresh',
                                    'scale_thresh', 'dlen_sqrd2', 'min_nInliers'])
                sv_tup = None
            nFeatSVTotal += len(fm)
            if sv_tup is not None:
                # Return the inliers to the homography from chip2 to chip1
                homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff = sv_tup
                if qreq_.qparams.with_metadata:
                    daid2_svtup[daid] = sv_tup
                fm_SV = fm[homog_inliers]
                fs_SV = fs[homog_inliers]
                fk_SV = fk[homog_inliers]
                if sver_weighting:
                    # Rescore based on homography errors
                    #xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
                    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
                    homog_xy_errors = homog_errors[0][homog_inliers]
                    homog_err_weight = (1.0 - np.sqrt(homog_xy_errors / xy_thresh_sqrd))
                    fs_SV *= homog_err_weight
                daid2_fm_V[daid] = fm_SV
                daid2_fs_V[daid] = fs_SV
                daid2_fk_V[daid] = fk_SV
                nFeatMatchSV += len(homog_inliers)
                #nFeatMatchSVAff += len(aff_inliers)
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(daid2_fm_V, daid2_fs_V, daid2_fk_V)
        if qreq_.qparams.with_metadata:
            qaid2_svtups[qaid] = daid2_svtup
        qaid2_chipmatchSV[qaid] = chipmatchSV
    if verbose:
        #print('[hs] * Affine verified %d/%d feat matches' % (nFeatMatchSVAff, nFeatSVTotal))
        print('[hs] * Homog  verified %d/%d feat matches' % (nFeatMatchSV, nFeatSVTotal))
    if qreq_.qparams.with_metadata:
        qreq_.metadata['qaid2_svtups'] = qaid2_svtups
    return qaid2_chipmatchSV


def prescore_nsum(qreq_, daid2_prescore, nShortlist):
    """ # TODO : rectify with code in hots_query_result """
    daid_list = np.array(daid2_prescore.keys())
    dnid_list = np.array(qreq_.ibs.get_annot_name_rowids(daid_list))
    prescore_arr = np.array(daid2_prescore.values())
    unique_nids, groupxs = vt.group_indicies(dnid_list)
    grouped_prescores = vt.apply_grouping(prescore_arr, groupxs)
    dnid2_prescore = dict(zip(unique_nids, [arr.max() for arr in grouped_prescores]))
    # Ensure that you verify each member of the top shortlist names
    topx2_nid = ut.util_dict.keys_sorted_by_value(dnid2_prescore)[::-1]
    # Use shortlist of names instead of annots
    nNamesRerank = min(len(topx2_nid), nShortlist)
    topx2_aids = [daid_list[dnid_list == nid] for nid in topx2_nid[:nNamesRerank]]
    # override shortlist because we already selected a subset of names
    topx2_aid = ut.flatten(topx2_aids)
    return topx2_aid


#@ut.indent_func('[pdls]')
def precompute_topx2_dlen_sqrd(qreq_, aid2_fm, topx2_aid, topx2_kpts,
                                nRerank, use_chip_extent):
    """
    helper for spatial verification, computes the squared diagonal length of
    matching chips

    Args:
        qreq_ (QueryRequest): hyper-parameters
        aid2_fm (dict):
        topx2_aid (dict):
        topx2_kpts (dict):
        nRerank (int):
        use_chip_extent (bool):

    Returns:
        topx2_dlen_sqrd

    # TODO: decouple example

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> import utool
        >>> cfgdict = dict()
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'spatial_verification')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_FILT']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> topx2_aid, nRerank = pipeline.get_prescore_shortlist(qaid, chipmatch, qreq_)
        >>> (daid2_fm, daid2_fs, daid2_fk) = chipmatch
        >>> kpts1 = qreq_.ibs.get_annot_kpts(qaid)
        >>> topx2_kpts = qreq_.ibs.get_annot_kpts(topx2_aid)
        >>> use_chip_extent = True
        >>> topx2_dlen_sqrd = pipeline.precompute_topx2_dlen_sqrd(qreq_, daid2_fm, topx2_aid, topx2_kpts, nRerank, use_chip_extent)

    """
    if use_chip_extent:
        #topx2_chipsize = list(qreq_.ibs.get_annot_chipsizes(topx2_aid))
        #def chip_dlen_sqrd(tx):
        #    (chipw, chiph) = topx2_chipsize[tx]
        #    dlen_sqrd = chipw ** 2 + chiph ** 2
        #    return dlen_sqrd
        #topx2_dlen_sqrd = [chip_dlen_sqrd(tx) for tx in range(nRerank)]
        #topx2_chipsize = np.array(qreq_.ibs.get_annot_chipsizes(topx2_aid))
        #topx2_chipsize = np.array(qreq_.ibs.get_annot_chipsizes(topx2_aid[:nRerank]))
        #(np.array(qreq_.ibs.get_annot_chipsizes(topx2_aid[:nRerank])) ** 2).sum(1)
        #[w ** 2 + h ** 2
        #                   for (w, h) in qreq_.ibs.get_annot_chipsizes(topx2_aid)]
        topx2_dlen_sqrd = [
            ((w ** 2) + (h ** 2))
            for (w, h) in qreq_.ibs.get_annot_chipsizes(topx2_aid[:nRerank])
        ]
        return topx2_dlen_sqrd
    else:
        # Use extent of matching keypoints
        def kpts_dlen_sqrd(tx):
            kpts2 = topx2_kpts[tx]
            aid = topx2_aid[tx]
            fm  = aid2_fm[aid]
            # This dosent make sense when len(fm) == 0
            if len(fm) == 0:
                return -1
            x_m, y_m = ktool.get_xys(kpts2[fm[:, 1]])
            dlensqrd = (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            return dlensqrd
        topx2_dlen_sqrd = [kpts_dlen_sqrd(tx) for tx in range(nRerank)]
    return topx2_dlen_sqrd


#============================
# 6) Query Result Format
#============================


#@ut.indent_func('[ctr]')
@profile
def chipmatch_to_resdict(qaid2_chipmatch, qreq_, verbose=VERB_PIPELINE):
    """
    Converts a dictionary of chipmatch tuples into a dictionary of query results

    Args:
        qaid2_chipmatch (dict):
        metadata (dict):
        qreq_ (QueryRequest): hyper-parameters
        qaid2_scores (dict): optional

    Returns:
        qaid2_qres

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid2_qres = pipeline.chipmatch_to_resdict(qaid2_chipmatch, qreq_)
        >>> qres = qaid2_qres[1]
    """
    if verbose:
        print('[hs] Step 6) Convert chipmatch -> qres')
    qaids   = qreq_.get_external_qaids()
    qauuids = qreq_.get_external_quuids()
    # Matchable daids
    daids   = qreq_.get_external_daids()
    cfgstr = qreq_.get_cfgstr()
    score_method = qreq_.qparams.score_method
    # Create the result structures for each query.
    qaid2_qres = {}
    # Currently not looping over the keys so we have access to uuids
    # using qreq_ externals aids should be equivalent
    #for qaid in six.iterkeys(qaid2_chipmatch):
    for qaid, qauuid in zip(qaids, qauuids):
        # Create a query result structure
        qres = hots_query_result.QueryResult(qaid, qauuid, cfgstr, daids)
        qaid2_qres[qaid] = qres

    for qaid, qres in six.iteritems(qaid2_qres):
        pass
        # For each query's chipmatch
        chipmatch = qaid2_chipmatch[qaid]
        if chipmatch is not None:
            try:
                aid2_fm, aid2_fs, aid2_fk = chipmatch
            except Exception as ex:
                ut.printex(ex, 'error converting chipmatch',
                              keys=['chipmatch'])
                raise
            qres.aid2_fm = aid2_fm
            qres.aid2_fs = aid2_fs
            qres.aid2_fk = aid2_fk

        # Perform final scoring
        qaid2_scores = None  # HACK. Used to be arg, still in in case smk breaks
        if qaid2_scores is None:
            daid2_score = score_chipmatch(qaid, chipmatch, score_method, qreq_)
        else:
            daid2_score = qaid2_scores[qaid]
            if not isinstance(daid2_score, dict):
                # Pandas hack
                daid2_score = daid2_score.to_dict()
        if qreq_.qparams.score_normalization:
            normalizer = qreq_.normalizer
            score_list = list(six.itervalues(daid2_score))
            prob_list = normalizer.normalize_score_list(score_list)
            daid2_prob = dict(zip(six.iterkeys(daid2_score), prob_list))
            qres.aid2_prob = daid2_prob
        # Populate query result fields
        qres.aid2_score = daid2_score  # FIXME fig qreq name

        metadata = qreq_.metadata
        qres.metadata = {}  # dbgstats
        for key, qaid2_meta in six.iteritems(metadata):
            qres.metadata[key] = qaid2_meta[qaid]  # things like k+1th
    # Retain original score method
    return qaid2_qres


#============================
# Scoring Mechanism
#============================

#@ut.indent_func('[scm]')
@profile
def score_chipmatch(qaid, chipmatch, score_method, qreq_):
    """
    Assigns scores to database annotation ids for a particualry query's
    chipmatch

    DOES NOT APPLY SCORE NORMALIZATION

    Args:
        qaid (int): query annotation id
        chipmatch (tuple):
        score_method (str):
        qreq_ (QueryRequest): hyper-parameters
        isprescore (bool): flag

    Returns:
        daid2_score : scores for each database id w.r.t. a single query

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-score_chipmatch

    Example:
        >>> # DISABLE_ENABLE
        >>> # PRESCORE
        >>> from ibeis.model.hots.pipeline import *
        >>> from ibeis.model.hots import pipeline
        >>> import utool as ut
        >>> cfgdict = dict(codename='nsum')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'spatial_verification')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_FILT']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> score_method = qreq_.qparams.prescore_method
        >>> daid2_score_pre = pipeline.score_chipmatch(qaid, chipmatch, score_method, qreq_)

    Example2:
        >>> # POSTSCORE
        >>> from ibeis.model.hots.pipeline import *
        >>> from ibeis.model.hots import pipeline
        >>> import utool as ut
        >>> cfgdict = dict(codename='nsum')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> score_method = qreq_.qparams.score_method
        >>> daid2_score_post = pipeline.score_chipmatch(qaid, chipmatch, score_method, qreq_)
    """
    #(aid2_fm, aid2_fs, aid2_fk) = chipmatch
    # HACK: Im not even sure if the 'w' suffix is correctly handled anymore
    if score_method.find('w') == len(score_method) - 1:
        score_method = score_method[:-1]
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        (aid_list, score_list) = vr2.score_chipmatch_csum(qaid, chipmatch, qreq_)
    elif score_method == 'nsum':
        (aid_list, score_list) = vr2.score_chipmatch_nsum(qaid, chipmatch, qreq_)
    #elif score_method == 'pl':
    #    daid2_score, nid2_score = vr2.score_chipmatch_PL(qaid, chipmatch, qreq_)
    #elif score_method == 'borda':
    #    daid2_score, nid2_score = vr2.score_chipmatch_pos(qaid, chipmatch, qreq_, 'borda')
    #elif score_method == 'topk':
    #    daid2_score, nid2_score = vr2.score_chipmatch_pos(qaid, chipmatch, qreq_, 'topk')
    #elif score_method.startswith('coverage'):
    #    # Method num is at the end of coverage
    #    method = int(score_method.replace('coverage', '0'))
    #    daid2_score = coverage_image.score_chipmatch_coverage(qaid, chipmatch, qreq_, method=method)
    else:
        raise Exception('[hs] unknown scoring method:' + score_method)

    # HACK: should not use dicts in pipeline if it can be helped
    daid2_score = dict(zip(aid_list, score_list))
    return daid2_score


#============================
# Result Caching
#============================


#@ut.indent_func('[tlr]')
@profile
def try_load_resdict(qreq_, force_miss=False):
    """
    Try and load the result structures for each query.
    returns a list of failed qaids

    Args:
        qreq_ (QueryRequest): hyper-parameters
        force_miss (bool):

    Returns:
        tuple : (qaid2_qres_hit, cachemiss_qaids)
    """
    qaids   = qreq_.get_external_qaids()
    qauuids = qreq_.get_external_quuids()
    daids   = qreq_.get_external_daids()

    cfgstr = qreq_.get_cfgstr()
    qresdir = qreq_.get_qresdir()
    qaid2_qres_hit = {}
    cachemiss_qaids = []
    for qaid, qauuid in zip(qaids, qauuids):
        try:
            qres = hots_query_result.QueryResult(qaid, qauuid, cfgstr, daids)
            qres.load(qresdir, force_miss=force_miss)  # 77.4 % time
            qaid2_qres_hit[qaid] = qres  # cache hit
        except (hsexcept.HotsCacheMissError, hsexcept.HotsNeedsRecomputeError):
            cachemiss_qaids.append(qaid)  # cache miss
    return qaid2_qres_hit, cachemiss_qaids


def save_resdict(qreq_, qaid2_qres, verbose=VERB_PIPELINE):
    """
    Saves a dictionary of query results to disk

    Args:
        qreq_ (QueryRequest): hyper-parameters
        qaid2_qres (dict):

    Returns:
        None
    """
    qresdir = qreq_.get_qresdir()
    if verbose:
        print('[hs] saving %d query results' % len(qaid2_qres))
    for qres in six.itervalues(qaid2_qres):
        qres.save(qresdir)


def testrun_pipeline_upto(qreq_, stop_node=None):
    """ convinience: runs pipeline for tests """
    #---
    if stop_node == 'nearest_neighbors':
        return locals()
    qaid2_nns = nearest_neighbors(qreq_)
    #---
    if stop_node == 'baseline_neighbor_filter':
        return locals()
    qaid2_nnfilt0 = baseline_neighbor_filter(qaid2_nns, qreq_)
    #---
    if stop_node == 'weight_neighbors':
        return locals()
    filt2_weights = weight_neighbors(qaid2_nns, qaid2_nnfilt0, qreq_)
    #---
    if stop_node == 'filter_neighbors':
        return locals()
    qaid2_nnfilt = filter_neighbors(qaid2_nns, qaid2_nnfilt0, filt2_weights, qreq_)
    #---
    if stop_node == 'build_chipmatches':
        return locals()
    qaid2_chipmatch_FILT = build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq_)
    #---
    if stop_node == 'spatial_verification':
        return locals()
    qaid2_chipmatch_SVER = spatial_verification(qaid2_chipmatch_FILT, qreq_)
    #---
    if stop_node == 'chipmatch_to_resdict':
        return locals()
    qaid2_qres = chipmatch_to_resdict(qaid2_chipmatch_SVER, qreq_)

    #qaid2_svtups = qreq_.metadata['qaid2_svtups']
    return locals()


#============================
# Testdata
#============================


def get_pipeline_testdata(dbname=None, cfgdict={}, qaid_list=None,
                          daid_list=None):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict = dict(pipeline_root='vsone', codename='vsone')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict)
        >>> print(qreq_.qparams.query_cfgstr)
    """
    import ibeis
    from ibeis.model.hots import query_request
    if dbname is None:
        dbname = ut.get_argval('--db', str, 'testdb1')
    ibs = ibeis.opendb(dbname)
    if qaid_list is None:
        if dbname == 'testdb1':
            qaid_list = [1]
        if dbname == 'GZ_ALL':
            qaid_list = [1032]
        if dbname == 'PZ_ALL':
            qaid_list = [1, 3, 5, 9]
        else:
            qaid_list = [1]
    if daid_list is None:
        daid_list = ibs.get_valid_aids()
        if dbname == 'testdb1':
            daid_list = daid_list[0:min(5, len(daid_list))]
    elif daid_list == 'all':
        daid_list = ibs.get_valid_aids()
    ibs = ibeis.test_main(db=dbname)
    if 'with_metadata' not in cfgdict:
        cfgdict['with_metadata'] = True
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, cfgdict)
    qreq_.lazy_load()
    return ibs, qreq_


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
