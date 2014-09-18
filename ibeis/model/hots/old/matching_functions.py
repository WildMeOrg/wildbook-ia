"""
#=================
# matching_functions:
# Module Concepts
#=================

PREFIXES:
qaid2_XXX - prefix mapping query chip index to
qfx2_XXX  - prefix mapping query chip feature index to

TUPLES:
 * nns    - a (qfx2_dx, qfx2_dist) tuple
 * nnfilt - a (qfx2_fs, qfx2_valid) tuple

SCALARS
 * dx     - the index into the database of features
 * dist   - the distance to a corresponding feature
 * fs     - a score of a corresponding feature
 * valid  - a valid bit for a corresponding feature

REALIZATIONS:
qaid2_nns - maping from query chip index to nns
{
 * qfx2_dx   - ranked list of query feature indexes to database feature indexes
 * qfx2_dist - ranked list of query feature indexes to database feature indexes
}

* qaid2_norm_weight - mapping from qaid to (qfx2_normweight, qfx2_selnorm)
         = qaid2_nnfilt[qaid]
"""
# TODO: Remove ibs control as much as possible or abstract it away
from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
import six
from collections import defaultdict
import sys
# Scientific
import numpy as np
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import spatial_verification as sver
# Hotspotter
from ibeis.model.hots import hots_query_result
from ibeis.model.hots import exceptions as hsexcept
from ibeis.model.hots import coverage_image
from ibeis.model.hots import nn_filters
from ibeis.model.hots import voting_rules2 as vr2
import utool
from functools import partial
#profile = utool.profile
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[mf]', DEBUG=False)


np.tau = 2 * np.pi  # tauday.com
NOT_QUIET = utool.NOT_QUIET and not utool.get_flag('--quiet-query')
VERBOSE = utool.VERBOSE or utool.get_flag('--verbose-query')


#=================
# Cython Metadata
#=================
"""
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t
ctypedef np.uint8_t uint8_t
ctypedef np.uint8_t desc_t
ctypedef ktool.KPTS_T kpts_t
ctypedef ktool.DESC_T desc_t

cdef int MARK_AFTER
cdef double tau
"""

#=================
# Globals
#=================

START_AFTER = 2


# specialized progress func
log_prog = partial(utool.log_progress, startafter=START_AFTER)

#=================
# Helpers
#=================


#def compare(qreq, qreq_):
#    qaid = 1
#    qvecs_list = qreq_.get_internal_qvecs()
#    qaids = qreq.get_internal_qaids()
#    qdesc_list = qreq_.get_annot_desc(qaids)  # Get descriptors
#    assert np.all(qvecs_list[0] == qdesc_list[0])
#    assert np.all(qreq_.indexer.dx2_vec == qreq.data_index.dx2_data)
#    assert np.all(qreq_.indexer.dx2_rowid == qreq.data_index.dx2_aid)
#    assert np.all(qreq_.indexer.dx2_fx == qreq.data_index.dx2_fx)
#    qfx2_dx_, qfx2_dist_ = qaid2_nns_[qaid]
#    qfx2_dx, qfx2_dist = qaid2_nns[qaid]
#    assert id(qaid2_nns) != id(qaid2_nns_)
#    assert np.all(qfx2_dx_ == qfx2_dx)
#    assert np.all(qfx2_dist_ == qfx2_dist)
#    index = np.where(qfx2_dx_ != qfx2_dx)
#    qfx2_dx.shape == qfx2_dx.shape
#    qfx2_dx_[index]
#    qfx2_dx[index]


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qaid):
    msg = ('QUERY ERROR IN %s: qaid=%r has no descriptors!' +
           'Please delete it.') % (ibs.get_dbname(), qaid)
    ex = QueryException(msg)
    return ex


#============================
# 1) Nearest Neighbors
#============================


@profile
def nearest_neighbors(ibs, qaids, qreq):
    """ Plain Nearest Neighbors
    Input:
        ibs   - an IBEIS Controller
        qaids - query annotation-ids
        qreq  - a QueryRequest object
    Output:
        qaid2_nnds - a dict mapping query annnotation-ids to a nearest neighbor
                     tuple (indexes, dists). indexes and dist have the shape
                     (nDesc x K) where nDesc is the number of descriptors in the
                     annotation, and K is the number of approximate nearest
                     neighbors.
        cdef:
            dict qaid2_nns
            object ibs
            object qreq
    """
    # Neareset neighbor configuration
    nn_cfg = qreq.cfg.nn_cfg
    K      = nn_cfg.K
    Knorm  = nn_cfg.Knorm
    checks = nn_cfg.checks
    if NOT_QUIET:
        cfgstr_   = nn_cfg.get_cfgstr()
        print('[mf] Step 1) Assign nearest neighbors: ' + cfgstr_)
    num_neighbors = K + Knorm  # number of nearest neighbors
    qdesc_list = ibs.get_annot_desc(qaids)  # Get descriptors
    nn_func = qreq.data_index.flann.nn_index  # Approx nearest neighbor func
    # Call a tighter (hopefully cythonized) nearest neighbor function
    qaid2_nns = _nearest_neighbors(nn_func, qaids, qdesc_list, num_neighbors, checks)
    return qaid2_nns


def _nearest_neighbors(nn_func, qaids, qdesc_list, num_neighbors, checks):
    """ Helper worker function for nearest_neighbors
        cdef:
            list qaids, qdesc_list
            long num_neighbors, checks
            dict qaid2_nns
            long nTotalNN, nTotalDesc
            np.ndarray[desc_t, ndim=2] qfx2_desc
            np.ndarray[int32_t, ndim=2] qfx2_dx
            np.ndarray[float64_t, ndim=2] qfx2_dist
            np.ndarray[int32_t, ndim=2] qfx2_dx
            np.ndarray[float64_t, ndim=2] qfx2_dist
    """
    # Output
    qaid2_nns = {}
    # Internal statistics reporting
    nTotalNN, nTotalDesc = 0, 0
    mark_, end_ = log_prog('Assign NN: ', len(qaids))
    for count, qaid in enumerate(qaids):
        mark_(count)  # progress
        qfx2_desc = qdesc_list[count]
        # Check that we can query this annotation
        if len(qfx2_desc) == 0:
            # Assign empty nearest neighbors
            qfx2_dx   = np.empty((0, num_neighbors), dtype=np.int32)
            qfx2_dist = np.empty((0, num_neighbors), dtype=np.float64)
            qaid2_nns[qaid] = (qfx2_dx, qfx2_dist)
            continue
        # Find Neareset Neighbors nntup = (indexes, dists)
        (qfx2_dx, qfx2_dist) = nn_func(qfx2_desc, num_neighbors, checks=checks)
        # Associate query annotation with its nearest descriptors
        qaid2_nns[qaid] = (qfx2_dx, qfx2_dist)
        # record number of query and result desc
        nTotalNN += qfx2_dx.size
        nTotalDesc += len(qfx2_desc)
    end_()
    if NOT_QUIET:
        print('[mf] * assigned %d desc from %d chips to %r nearest neighbors'
              % (nTotalDesc, len(qaids), nTotalNN))
    return qaid2_nns


#============================
# 2) Nearest Neighbor weights
#============================


def weight_neighbors(ibs, qaid2_nns, qreq):
    if NOT_QUIET:
        print('[mf] Step 2) Weight neighbors: ' + qreq.cfg.filt_cfg.get_cfgstr())
    if qreq.cfg.filt_cfg.filt_on:
        return _weight_neighbors(ibs, qaid2_nns, qreq)
    else:
        return  {}


@profile
def _weight_neighbors(ibs, qaid2_nns, qreq):
    nnfilter_list = qreq.cfg.filt_cfg.get_active_filters()
    filt2_weights = {}
    filt2_meta = {}
    for nnfilter in nnfilter_list:
        nn_filter_fn = nn_filters.NN_FILTER_FUNC_DICT[nnfilter]
        # Apply [nnfilter] weight to each nearest neighbor
        # TODO FIX THIS!
        qaid2_norm_weight, qaid2_selnorms = nn_filter_fn(ibs, qaid2_nns, qreq)
        filt2_weights[nnfilter] = qaid2_norm_weight
        filt2_meta[nnfilter] = qaid2_selnorms
    return filt2_weights, filt2_meta


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


@profile
def _apply_filter_scores(qaid, qfx2_nndx, filt2_weights, filt_cfg):
    qfx2_score = np.ones(qfx2_nndx.shape, dtype=hots_query_result.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nndx.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, aid2_weights in six.iteritems(filt2_weights):
        qfx2_weights = aid2_weights[qaid]
        sign, thresh, weight = filt_cfg.get_stw(filt)  # stw = sign, thresh, weight
        if thresh is not None and thresh != 'None':
            thresh = float(thresh)  # corrects for thresh being strings sometimes
            if isinstance(thresh, (int, float)):
                qfx2_passed = sign * qfx2_weights <= sign * thresh
                qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if not weight == 0:
            qfx2_score += weight * qfx2_weights
    return qfx2_score, qfx2_valid


@profile
def filter_neighbors(ibs, qaid2_nns, filt2_weights, qreq):
    qaid2_nnfilt = {}
    # Configs
    filt_cfg = qreq.cfg.filt_cfg
    cant_match_sameimg  = not filt_cfg.can_match_sameimg
    cant_match_samename = not filt_cfg.can_match_samename
    K = qreq.cfg.nn_cfg.K
    if NOT_QUIET:
        print('[mf] Step 3) Filter neighbors: ')
    if filt_cfg.gravity_weighting:
        # We dont have an easy way to access keypoints from nearest neighbors yet
        aid_list = np.unique(qreq.data_index.dx2_aid)  # FIXME: Highly inefficient
        kpts_list = ibs.get_annot_kpts(aid_list)
        dx2_kpts = np.vstack(kpts_list)
        dx2_oris = ktool.get_oris(dx2_kpts)
        assert len(dx2_oris) == len(qreq.data_index.dx2_data)
    # Filter matches based on config and weights
    mark_, end_ = log_prog('Filter NN: ', len(qaid2_nns))
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        mark_(count)  # progress
        (qfx2_dx, _) = qaid2_nns[qaid]
        qfx2_nndx = qfx2_dx[:, 0:K]
        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _apply_filter_scores(qaid, qfx2_nndx, filt2_weights, filt_cfg)
        qfx2_aid = qreq.data_index.dx2_aid[qfx2_nndx]
        if VERBOSE:
            print('[mf] * %d assignments are invalid by thresh' %
                  ((True - qfx2_valid).sum()))
        if filt_cfg.gravity_weighting:
            qfx2_nnori = dx2_oris[qfx2_nndx]
            qfx2_kpts  = ibs.get_annot_kpts(qaid)  # FIXME: Highly inefficient
            qfx2_oris  = ktool.get_oris(qfx2_kpts)
            # Get the orientation distance
            qfx2_oridist = ltool.rowwise_oridist(qfx2_nnori, qfx2_oris)
            # Normalize into a weight (close orientations are 1, far are 0)
            qfx2_gvweight = (np.tau - qfx2_oridist) / np.tau
            # Apply gravity vector weight to the score
            qfx2_score *= qfx2_gvweight
        # Remove Impossible Votes:
        # dont vote for yourself or another chip in the same image
        cant_match_self = not cant_match_sameimg
        if cant_match_self:
            ####DBG
            qfx2_notsamechip = qfx2_aid != qaid
            if VERBOSE:
                nChip_all_invalid = ((True - qfx2_notsamechip)).sum()
                nChip_new_invalid = (qfx2_valid * (True - qfx2_notsamechip)).sum()
                print('[mf] * %d assignments are invalid by self' % nChip_all_invalid)
                print('[mf] * %d are newly invalided by self' % nChip_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        if cant_match_sameimg:
            qfx2_gid = ibs.get_annot_gids(qfx2_aid)
            qgid     = ibs.get_annot_gids(qaid)
            qfx2_notsameimg = qfx2_gid != qgid
            ####DBG
            if VERBOSE:
                nImg_all_invalid = ((True - qfx2_notsameimg)).sum()
                nImg_new_invalid = (qfx2_valid * (True - qfx2_notsameimg)).sum()
                print('[mf] * %d assignments are invalid by gid' % nImg_all_invalid)
                print('[mf] * %d are newly invalided by gid' % nImg_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
        if cant_match_samename:
            qfx2_nid = ibs.get_annot_nids(qfx2_aid)
            qnid = ibs.get_annot_nids(qaid)
            qfx2_notsamename = qfx2_nid != qnid
            ####DBG
            if VERBOSE:
                nName_all_invalid = ((True - qfx2_notsamename)).sum()
                nName_new_invalid = (qfx2_valid * (True - qfx2_notsamename)).sum()
                print('[mf] * %d assignments are invalid by nid' % nName_all_invalid)
                print('[mf] * %d are newly invalided by nid' % nName_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
        #printDBG('[mf] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        qaid2_nnfilt[qaid] = (qfx2_score, qfx2_valid)
    end_()
    return qaid2_nnfilt


@profile
def identity_filter(qaid2_nns, qreq):
    """ testing function returns unfiltered nearest neighbors
    this does check that you are not matching yourself
    """
    qaid2_nnfilt = {}
    K = qreq.cfg.nn_cfg.K
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        (qfx2_dx, _) = qaid2_nns[qaid]
        qfx2_nndx = qfx2_dx[:, 0:K]
        qfx2_score = np.ones(qfx2_nndx.shape, dtype=hots_query_result.FS_DTYPE)
        qfx2_valid = np.ones(qfx2_nndx.shape, dtype=np.bool)
        # Check that you are not matching yourself
        qfx2_aid = qreq.data_index.dx2_aid[qfx2_nndx]
        qfx2_notsamechip = qfx2_aid != qaid
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        qaid2_nnfilt[qaid] = (qfx2_score, qfx2_valid)

    return qaid2_nnfilt


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> aid2
#============================


@profile
def _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk):
    minMatches = 2  # TODO: paramaterize
    # Convert to numpy
    fm_dtype = hots_query_result.FM_DTYPE
    fs_dtype = hots_query_result.FS_DTYPE
    fk_dtype = hots_query_result.FK_DTYPE
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


@profile
def build_chipmatches(qaid2_nns, qaid2_nnfilt, qreq):
    """
    Input:
        qaid2_nns    - dict of assigned nearest features (only indexes are used here)
        qaid2_nnfilt - dict of (featmatch_scores, featmatch_mask)
                        where the scores and matches correspond to the assigned
                        nearest features
        qreq         - QueryRequest object

    Output:
        qaid2_chipmatch - dict of (

    Notes:
        The prefix qaid2_ denotes a mapping where keys are query-annotation-id

    vsmany/vsone counts here. also this is where the filter
    weights and thershold are applied to the matches. Essientally
    nearest neighbors are converted into weighted assignments
    """

    # Config
    K = qreq.cfg.nn_cfg.K
    query_type = qreq.cfg.agg_cfg.query_type
    is_vsone = query_type == 'vsone'
    if NOT_QUIET:
        print('[mf] Step 4) Building chipmatches %s' % (query_type,))
    # Return var
    qaid2_chipmatch = {}
    nFeatMatches = 0
    #Vsone
    if is_vsone:
        assert len(qreq.qaids) == 1
        aid2_fm, aid2_fs, aid2_fk = new_fmfsfk()
    # Iterate over chips with nearest neighbors
    mark_, end_ = log_prog('Build Chipmatch: ', len(qaid2_nns))
    for count, qaid in enumerate(six.iterkeys(qaid2_nns)):
        mark_(count)  # Mark progress
        (qfx2_dx, _) = qaid2_nns[qaid]
        (qfx2_fs, qfx2_valid) = qaid2_nnfilt[qaid]
        nQKpts = len(qfx2_dx)
        # Build feature matches
        qfx2_nndx = qfx2_dx[:, 0:K]
        qfx2_aid  = qreq.data_index.dx2_aid[qfx2_nndx]
        qfx2_fx   = qreq.data_index.dx2_fx[qfx2_nndx]
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        # Pack valid feature matches into an interator
        valid_lists = [qfx2[qfx2_valid] for qfx2 in (qfx2_qfx, qfx2_aid, qfx2_fx, qfx2_fs, qfx2_k,)]
        # TODO: Sorting the valid lists by aid might help the speed of this
        # code. Also, consolidating fm, fs, and fk into one vector will reduce
        # the amount of appends.
        match_iter = zip(*valid_lists)
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
            #if not QUIET:
            #    nFeats_in_matches = [len(fm) for fm in six.itervalues(aid2_fm)]
            #    print('nFeats_in_matches_stats = ' +
            #          utool.dict_str(utool.mystats(nFeats_in_matches)))
        # Vsone - Append database feature matches to query aids
        else:
            for qfx, aid, fx, fs, fk in match_iter:
                aid2_fm[qaid].append((fx, qfx))  # Note the difference
                aid2_fs[qaid].append(fs)
                aid2_fk[qaid].append(fk)
                nFeatMatches += 1
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk)
        qaid = qreq.qaids[0]
        qaid2_chipmatch[qaid] = chipmatch
    end_()
    if NOT_QUIET:
        print('[mf] * made %d feat matches' % nFeatMatches)
    return qaid2_chipmatch


#============================
# 5) Spatial Verification
#============================


def spatial_verification(ibs, qaid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        print('[mf] Step 5) Spatial verification: off')
        return (qaid2_chipmatch, {}) if dbginfo else qaid2_chipmatch
    else:
        return _spatial_verification(ibs, qaid2_chipmatch, qreq, dbginfo=dbginfo)


@profile
def _spatial_verification(ibs, qaid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    print('[mf] Step 5) Spatial verification: ' + sv_cfg.get_cfgstr())
    prescore_method = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    scale_thresh    = sv_cfg.scale_thresh
    ori_thresh      = sv_cfg.ori_thresh
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    qaid2_chipmatchSV = {}
    nFeatSVTotal = 0
    nFeatMatchSV = 0
    nFeatMatchSVAff = 0
    if dbginfo:
        qaid2_svtups = {}  # dbg info (can remove if there is a speed issue)
    def print_(msg, count=0):
        """ temp print_. Using count in this way is a hack """
        if NOT_QUIET:
            if count % 25 == 0:
                sys.stdout.write(msg)
            count += 1
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qaid in six.iterkeys(qaid2_chipmatch):
        chipmatch = qaid2_chipmatch[qaid]
        aid2_prescore = score_chipmatch(ibs, qaid, chipmatch, prescore_method, qreq)
        #print('Prescore: %r' % (aid2_prescore,))
        (aid2_fm, aid2_fs, aid2_fk) = chipmatch
        topx2_aid = utool.util_dict.keys_sorted_by_value(aid2_prescore)[::-1]
        nRerank = min(len(topx2_aid), nShortlist)
        # Precompute output container
        if dbginfo:
            aid2_svtup = {}  # dbg info (can remove if there is a speed issue)
        aid2_fm_V, aid2_fs_V, aid2_fk_V = new_fmfsfk()
        # Query Keypoints
        kpts1 = ibs.get_annot_kpts(qaid)
        topx2_kpts = ibs.get_annot_kpts(topx2_aid)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = _precompute_topx2_dlen_sqrd(ibs, aid2_fm, topx2_aid,
                                                      topx2_kpts, nRerank,
                                                      use_chip_extent)
        # spatially verify the top __NUM_RERANK__ results
        for topx in range(nRerank):
            aid = topx2_aid[topx]
            fm = aid2_fm[aid]
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = topx2_kpts[topx]
            fs    = aid2_fs[aid]
            fk    = aid2_fk[aid]
            sv_tup = sver.spatial_verification(kpts1, kpts2, fm,
                                               xy_thresh, scale_thresh, ori_thresh, dlen_sqrd,
                                               min_nInliers)
            nFeatSVTotal += len(fm)
            if sv_tup is None:
                    print_('o')  # sv failure
            else:
                # Return the inliers to the homography
                homog_inliers, H, aff_inliers, Aff = sv_tup
                if dbginfo:
                    aid2_svtup[aid] = sv_tup
                aid2_fm_V[aid] = fm[homog_inliers, :]
                aid2_fs_V[aid] = fs[homog_inliers]
                aid2_fk_V[aid] = fk[homog_inliers]
                nFeatMatchSV += len(homog_inliers)
                nFeatMatchSVAff += len(aff_inliers)
                if NOT_QUIET:
                    #print(inliers)
                    print_('.')  # verified something
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(aid2_fm_V, aid2_fs_V, aid2_fk_V)
        if dbginfo:
            qaid2_svtups[qaid] = aid2_svtup
        qaid2_chipmatchSV[qaid] = chipmatchSV
    print_('\n')
    if NOT_QUIET:
        print('[mf] * Affine verified %d/%d feat matches' % (nFeatMatchSVAff, nFeatSVTotal))
        print('[mf] * Homog  verified %d/%d feat matches' % (nFeatMatchSV, nFeatSVTotal))
    if dbginfo:
        return qaid2_chipmatchSV, qaid2_svtups
    else:
        return qaid2_chipmatchSV


def _precompute_topx2_dlen_sqrd(ibs, aid2_fm, topx2_aid, topx2_kpts,
                                nRerank, use_chip_extent):
    """ helper for spatial verification, computes the squared diagonal length of
    matching chips
    """
    if use_chip_extent:
        topx2_chipsize = list(ibs.get_annot_chipsizes(topx2_aid))
        def chip_dlen_sqrd(tx):
            (chipw, chiph) = topx2_chipsize[tx]
            dlen_sqrd = chipw ** 2 + chiph ** 2
            return dlen_sqrd
        topx2_dlen_sqrd = [chip_dlen_sqrd(tx) for tx in range(nRerank)]
    else:
        # Use extent of matching keypoints
        def kpts_dlen_sqrd(tx):
            kpts2 = topx2_kpts[tx]
            aid = topx2_aid[tx]
            fm    = aid2_fm[aid]
            x_m, y_m = ktool.get_xys(kpts2[fm[:, 1]])
            dlensqrd = (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            return dlensqrd
        topx2_dlen_sqrd = [kpts_dlen_sqrd(tx) for tx in range(nRerank)]
    return topx2_dlen_sqrd


#============================
# 6) QueryResult Format
#============================


@profile
def chipmatch_to_resdict(ibs, qaid2_chipmatch, filt2_meta, qreq):
    if NOT_QUIET:
        print('[mf] Step 6) Convert chipmatch -> qres')
    cfgstr = qreq.get_cfgstr()
    score_method = qreq.cfg.agg_cfg.score_method
    # Create the result structures for each query.
    qaid2_qres = {}
    for qaid in six.iterkeys(qaid2_chipmatch):
        # For each query's chipmatch
        chipmatch = qaid2_chipmatch[qaid]
        # Perform final scoring
        aid2_score = score_chipmatch(ibs, qaid, chipmatch, score_method, qreq)
        # Create a query result structure
        qres = hots_query_result.QueryResult(qaid, cfgstr)
        qres.aid2_score = aid2_score
        (qres.aid2_fm, qres.aid2_fs, qres.aid2_fk) = chipmatch
        qres.filt2_meta = {}  # dbgstats
        for filt, qaid2_meta in six.iteritems(filt2_meta):
            qres.filt2_meta[filt] = qaid2_meta[qaid]  # things like k+1th
        qaid2_qres[qaid] = qres
    # Retain original score method
    return qaid2_qres


@profile
def try_load_resdict(qreq):
    """ Try and load the result structures for each query.
    returns a list of failed qaids
    cdef:
        object qreq
        list qaids
        dict qaid2_qres
        list failed_qaids
    """
    qaids = qreq.qaids
    #cfgstr = qreq.get_cfgstr()  # NEEDS FIX TAKES 21.9 % time of this function
    cfgstr = qreq.get_cfgstr2()  # hack of a fix
    qaid2_qres = {}
    failed_qaids = []
    for qaid in qaids:
        try:
            qres = hots_query_result.QueryResult(qaid, cfgstr)
            qres.load(qreq.get_qresdir())  # 77.4 % time
            qaid2_qres[qaid] = qres
        except hsexcept.HotsCacheMissError:
            failed_qaids.append(qaid)
        except hsexcept.HotsNeedsRecomputeError:
            failed_qaids.append(qaid)
    return qaid2_qres, failed_qaids


#============================
# Scoring Mechanism
#============================

@profile
def score_chipmatch(ibs, qaid, chipmatch, score_method, qreq=None):
    (aid2_fm, aid2_fs, aid2_fk) = chipmatch
    # HACK: Im not even sure if the 'w' suffix is correctly handled anymore
    if score_method.find('w') == len(score_method) - 1:
        score_method = score_method[:-1]
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        aid2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'pl':
        aid2_score, nid2_score = vr2.score_chipmatch_PL(ibs, qaid, chipmatch, qreq)
    elif score_method == 'borda':
        aid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qaid, chipmatch, qreq, 'borda')
    elif score_method == 'topk':
        aid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qaid, chipmatch, qreq, 'topk')
    elif score_method.startswith('coverage'):
        # Method num is at the end of coverage
        method = int(score_method.replace('coverage', '0'))
        aid2_score = coverage_image.score_chipmatch_coverage(ibs, qaid, chipmatch, qreq, method=method)
    else:
        raise Exception('[mf] unknown scoring method:' + score_method)
    return aid2_score
