from __future__ import division, print_function
import utool
print, print_,  printDBG, rrr, profile =\
    utool.inject(__name__, '[mf]', DEBUG=False)
# Python
from itertools import izip
import sys
# Scientific
import numpy as np
# Hotspotter
import QueryResult as qr
import coverage
import nn_filters
import spatial_verification2 as sv2
import voting_rules2 as vr2
import utool


#=================
# Module Concepts
#=================
'''
PREFIXES:
qcx2_XXX - prefix mapping query chip index to
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
qcx2_nns - maping from query chip index to nns
{
  * qfx2_dx    - a ranked list of query feature indexes to database feature indexes
  * qfx2_dist  - a ranked list of query feature indexes to database feature indexes
}

* qcx2_norm_weight - mapping from qcx to (qfx2_normweight, qfx2_selnorm)

         = qcx2_nnfilt[qcx]


'''
#=================
# Globals
#=================

# TODO: Make a more elegant way of mapping weighting parameters to weighting
# function. A dict is better than eval, but there may be a better way.
NN_FILTER_FUNC_DICT = {
    'scale':   nn_filters.nn_scale_weight,
    'roidist': nn_filters.nn_roidist_weight,
    'recip':   nn_filters.nn_recip_weight,
    'bursty':  nn_filters.nn_bursty_weight,
    'lnrat':   nn_filters.nn_lnrat_weight,
    'lnbnn':   nn_filters.nn_lnbnn_weight,
    'ratio':   nn_filters.nn_ratio_weight,
}
MARK_AFTER = 2

#=================
# Helpers
#=================


def progress_func(maxval=0):
    mark_progress, end_progress = utool.progress_func(maxval, mark_after=MARK_AFTER, progress_type='simple')
    #if maxval > MARK_AFTER:
        #print('')
    return mark_progress, end_progress


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qcx):
    dbname = ibs.get_db_name()
    cidstr = ibs.cidstr(qcx)
    ex = QueryException(('QUERY ERROR IN %s: Query Chip q%s has no descriptors!' +
                         'Please delete it.') % (dbname, cidstr))
    return ex


#============================
# 1) Nearest Neighbors
#============================


@profile
def nearest_neighbors(ibs, qcxs, qreq):
    'Plain Nearest Neighbors'
    # Neareset neighbor configuration
    nn_cfg = qreq.cfg.nn_cfg
    K      = nn_cfg.K
    Knorm  = nn_cfg.Knorm
    checks = nn_cfg.checks
    uid_   = nn_cfg.get_uid()
    print('[mf] Step 1) Assign nearest neighbors: ' + uid_)
    # Grab descriptors
    cid2_desc = ibs.feats.cid2_desc
    # NNIndex
    flann = qreq._data_index.flann
    # Output
    qcx2_nns = {}
    nNN, nDesc = 0, 0
    mark_progress, end_progress = progress_func(len(qcxs))
    for count, qcx in enumerate(qcxs):
        mark_progress(count)
        qfx2_desc = cid2_desc[qcx]
        # Check that we can query this chip
        if len(qfx2_desc) == 0:
            # Raise error if strict
            if '--strict' in sys.argv:
                raise NoDescriptorsException(ibs, qcx)
            # Assign empty nearest neighbors
            empty_qfx2_dx   = np.empty((0, K + Knorm), dtype=np.int)
            empty_qfx2_dist = np.empty((0, K + Knorm), dtype=np.float)
            qcx2_nns[qcx] = (empty_qfx2_dx, empty_qfx2_dist)
            continue
        # Find Neareset Neighbors
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        # Store nearest neighbors
        qcx2_nns[qcx] = (qfx2_dx, qfx2_dist)
        # record number of query and result desc
        nNN += qfx2_dx.size
        nDesc += len(qfx2_desc)
    end_progress()
    print('[mf] * assigned %d desc from %d chips to %r nearest neighbors' %
          (nDesc, len(qcxs), nNN))
    return qcx2_nns


#============================
# 2) Nearest Neighbor weights
#============================


@profile
def weight_neighbors(ibs, qcx2_nns, qreq):
    filt_cfg = qreq.cfg.filt_cfg
    print('[mf] Step 2) Weight neighbors: ' + filt_cfg.get_uid())
    if not filt_cfg.filt_on:
        return  {}
    nnfilter_list = filt_cfg.get_active_filters()
    filt2_weights = {}
    filt2_meta = {}
    for nnfilter in nnfilter_list:
        nn_filter_fn = NN_FILTER_FUNC_DICT[nnfilter]
        # Apply [nnfilter] weight to each nearest neighbor
        # TODO FIX THIS!
        qcx2_norm_weight, qcx2_selnorms = nn_filter_fn(ibs, qcx2_nns, qreq)
        filt2_weights[nnfilter] = qcx2_norm_weight
        filt2_meta[nnfilter] = qcx2_selnorms
    return filt2_weights, filt2_meta


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


@profile
def filter_neighbors(ibs, qcx2_nns, filt2_weights, qreq):
    qcx2_nnfilter = {}
    # Configs
    filt_cfg = qreq.cfg.filt_cfg
    cant_match_sameimg = not filt_cfg.can_match_sameimg
    cant_match_samename = not filt_cfg.can_match_samename
    K = qreq.cfg.nn_cfg.K
    print('[mf] Step 3) Filter neighbors: ')
    #+ filt_cfg.get_uid())
    # NNIndex
    # Database feature index to chip index
    dx2_cx = qreq._data_index.ax2_cx
    # Filter matches based on config and weights
    mark_progress, end_progress = progress_func(len(qcx2_nns))
    for count, qcx in enumerate(qcx2_nns.iterkeys()):
        mark_progress(count)
        (qfx2_dx, _) = qcx2_nns[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt_cfg)
        qfx2_cx = dx2_cx[qfx2_nn]
        printDBG('[mf] * %d assignments are invalid by thresh' % ((True - qfx2_valid).sum()))
        # Remove Impossible Votes:
        # dont vote for yourself or another chip in the same image
        qfx2_notsamechip = qfx2_cx != qcx
        cant_match_self = True
        if cant_match_self:
            ####DBG
            nChip_all_invalid = ((True - qfx2_notsamechip)).sum()
            nChip_new_invalid = (qfx2_valid * (True - qfx2_notsamechip)).sum()
            printDBG('[mf] * %d assignments are invalid by self' % nChip_all_invalid)
            printDBG('[mf] * %d are newly invalided by self' % nChip_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        if cant_match_sameimg:
            qfx2_notsameimg  = ibs.tables.cid2_gx[qfx2_cx] != ibs.tables.cid2_gx[qcx]
            ####DBG
            nImg_all_invalid = ((True - qfx2_notsameimg)).sum()
            nImg_new_invalid = (qfx2_valid * (True - qfx2_notsameimg)).sum()
            printDBG('[mf] * %d assignments are invalid by gid' % nImg_all_invalid)
            printDBG('[mf] * %d are newly invalided by gid' % nImg_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
        if cant_match_samename:
            qfx2_notsamename = ibs.tables.cid2_nx[qfx2_cx] != ibs.tables.cid2_nx[qcx]
            ####DBG
            nName_all_invalid = ((True - qfx2_notsamename)).sum()
            nName_new_invalid = (qfx2_valid * (True - qfx2_notsamename)).sum()
            printDBG('[mf] * %d assignments are invalid by nid' % nName_all_invalid)
            printDBG('[mf] * %d are newly invalided by nid' % nName_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
        printDBG('[mf] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        qcx2_nnfilter[qcx] = (qfx2_score, qfx2_valid)
    end_progress()
    return qcx2_nnfilter


def _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt_cfg):
    qfx2_score = np.ones(qfx2_nn.shape, dtype=qr.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, cid2_weights in filt2_weights.iteritems():
        qfx2_weights = cid2_weights[qcx]
        sign, thresh, weight = filt_cfg.get_stw(filt)
        if thresh is not None and thresh != 'None':
            thresh = float(thresh)  # corrects for thresh being strings sometimes
            if isinstance(thresh, (int, float)):
                qfx2_passed = sign * qfx2_weights <= sign * thresh
                qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if not weight == 0:
            qfx2_score += weight * qfx2_weights
    return qfx2_score, qfx2_valid


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> cid2
#============================


@profile
def build_chipmatches(ibs, qcx2_nns, qcx2_nnfilt, qreq):
    '''vsmany/vsone counts here. also this is where the filter
    weights and thershold are applied to the matches. Essientally
    nearest neighbors are converted into weighted assignments'''
    # Config
    K = qreq.cfg.nn_cfg.K
    query_type = qreq.cfg.agg_cfg.query_type
    is_vsone = query_type == 'vsone'
    print('[mf] Step 4) Building chipmatches %s' % (query_type,))
    # Data Index
    dx2_cx = qreq._data_index.ax2_cx
    dx2_fx = qreq._data_index.ax2_fx
    # Return var
    qcx2_chipmatch = {}

    #Vsone
    if is_vsone:
        assert len(qreq._qcxs) == 1
        cid2_fm, cid2_fs, cid2_fk = new_fmfsfk(ibs)

    # Iterate over chips with nearest neighbors
    mark_progress, end_progress = progress_func(len(qcx2_nns))
    for count, qcx in enumerate(qcx2_nns.iterkeys()):
        mark_progress(count)
        #print('[mf] * scoring q' + ibs.cidstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        (qfx2_fs, qfx2_valid) = qcx2_nnfilt[qcx]
        nQuery = len(qfx2_dx)
        # Build feature matches
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_qfx = np.tile(np.arange(nQuery), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQuery, 1))
        # Pack feature matches into an interator
        match_iter = izip(*[qfx2[qfx2_valid] for qfx2 in
                            (qfx2_qfx, qfx2_cx, qfx2_fx, qfx2_fs, qfx2_k)])
        # Vsmany - Iterate over feature matches
        if not is_vsone:
            cid2_fm, cid2_fs, cid2_fk = new_fmfsfk(ibs)
            for qfx, cid, fx, fs, fk in match_iter:
                cid2_fm[cid].append((qfx, fx))  # Note the difference
                cid2_fs[cid].append(fs)
                cid2_fk[cid].append(fk)
            chipmatch = _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk)
            qcx2_chipmatch[qcx] = chipmatch
        # Vsone - Iterate over feature matches
        else:
            for qfx, cid, fx, fs, fk in match_iter:
                cid2_fm[qcx].append((fx, qfx))  # Note the difference
                cid2_fs[qcx].append(fs)
                cid2_fk[qcx].append(fk)
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk)
        qcx = qreq._qcxs[0]
        qcx2_chipmatch[qcx] = chipmatch

    end_progress()
    return qcx2_chipmatch


#============================
# 5) Spatial Verification
#============================


@profile
def spatial_verification(ibs, qcx2_chipmatch, qreq):
    sv_cfg = qreq.cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        print('[mf] Step 5) Spatial verification: off')
        return qcx2_chipmatch
    print('[mf] Step 5) Spatial verification: ' + sv_cfg.get_uid())
    prescore_method = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    min_scale = sv_cfg.scale_thresh_low
    max_scale = sv_cfg.scale_thresh_high
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    just_affine     = sv_cfg.just_affine
    cid2_rchip_size  = ibs.cpaths.cid2_rchip_size
    cid2_kpts = ibs.feats.cid2_kpts
    qcx2_chipmatchSV = {}
    #printDBG(qreq._dcxs)
    dcxs_ = set(qreq._dcxs)
    USE_1_to_2 = True
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qcx in qcx2_chipmatch.iterkeys():
        #printDBG('[mf] verify qcx=%r' % qcx)
        chipmatch = qcx2_chipmatch[qcx]
        cid2_prescore = score_chipmatch(ibs, qcx, chipmatch, prescore_method, qreq)
        (cid2_fm, cid2_fs, cid2_fk) = chipmatch
        topx2_cx = cid2_prescore.argsort()[::-1]  # Only allow indexed cids to be in the top results
        topx2_cx = [cid for cid in iter(topx2_cx) if cid in dcxs_]
        nRerank = min(len(topx2_cx), nShortlist)
        # Precompute output container
        cid2_fm_V, cid2_fs_V, cid2_fk_V = new_fmfsfk(ibs)
        # Query Keypoints
        kpts1 = cid2_kpts[qcx]
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = _precompute_topx2_dlen_sqrd(cid2_rchip_size, cid2_kpts,
                                                      cid2_fm, topx2_cx, nRerank,
                                                      use_chip_extent,
                                                      USE_1_to_2)
        # Override print function temporarilly
        def print_(msg, count=0):
            if count % 50 == 0:
                sys.stdout.write(msg)
            count += 1
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            cid = topx2_cx[topx]
            fm = cid2_fm[cid]
            #printDBG('[mf] vs topcx=%r, score=%r' % (cid, cid2_prescore[cid]))
            #printDBG('[mf] len(fm)=%r' % (len(fm)))
            if len(fm) >= min_nInliers:
                dlen_sqrd = topx2_dlen_sqrd[topx]
                kpts2 = cid2_kpts[cid]
                fs    = cid2_fs[cid]
                fk    = cid2_fk[cid]
                #printDBG('[mf] computing homog')
                sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh,
                                                max_scale, min_scale, dlen_sqrd,
                                                min_nInliers, just_affine)
                #printDBG('[mf] sv_tup = %r' % (sv_tup,))
                if sv_tup is None:
                    print_('o')  # sv failure
                else:
                    # Return the inliers to the homography
                    (H, inliers) = sv_tup
                    cid2_fm_V[cid] = fm[inliers, :]
                    cid2_fs_V[cid] = fs[inliers]
                    cid2_fk_V[cid] = fk[inliers]
                    print_('.')  # verified something
            else:
                print_('x')  # not enough initial matches
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(cid2_fm_V, cid2_fs_V, cid2_fk_V)
        qcx2_chipmatchSV[qcx] = chipmatchSV
    print_('\n')
    print('[mf] Finished sv')
    return qcx2_chipmatchSV


def _precompute_topx2_dlen_sqrd(cid2_rchip_size, cid2_kpts, cid2_fm, topx2_cx,
                                nRerank, use_chip_extent, USE_1_to_2):
    '''helper for spatial verification, computes the squared diagonal length of
    matching chips'''
    if use_chip_extent:
        def cid2_chip_dlensqrd(cid):
            (chipw, chiph) = cid2_rchip_size[cid]
            dlen_sqrd = chipw ** 2 + chiph ** 2
            return dlen_sqrd
        if USE_1_to_2:
            topx2_dlen_sqrd = [cid2_chip_dlensqrd(cid) for cid in iter(topx2_cx[:nRerank])]
        #else:
            #topx2_dlen_sqrd = [cid2_chip_dlensqrd(cid)] * nRerank
    else:
        if USE_1_to_2:
            def cid2_kpts2_dlensqrd(cid):
                kpts2 = cid2_kpts[cid]
                fm    = cid2_fm[cid]
                if len(fm) == 0:
                    return 1
                x_m = kpts2[fm[:, 1], 0].T
                y_m = kpts2[fm[:, 1], 1].T
                return (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            topx2_dlen_sqrd = [cid2_kpts2_dlensqrd(cid) for cid in iter(topx2_cx[:nRerank])]
        #else:
            #def cid2_kpts1_dlensqrd(cid):
                #kpts2 = cid2_kpts[cid]
                #fm    = cid2_fm[cid]
                #if len(fm) == 0:
                    #return 1
                #x_m = kpts2[fm[:, 0], 0].T
                #y_m = kpts2[fm[:, 0], 1].T
                #return (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            #topx2_dlen_sqrd = [cid2_kpts1_dlensqrd(cid) for cid in iter(topx2_cx[:nRerank])]
    return topx2_dlen_sqrd


def _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk):
    # Convert to numpy
    fm_dtype_ = qr.FM_DTYPE
    fs_dtype_ = qr.FS_DTYPE
    fk_dtype_ = qr.FK_DTYPE
    cid2_fm = [np.array(fm, fm_dtype_) for fm in iter(cid2_fm)]
    cid2_fs = [np.array(fs, fs_dtype_) for fs in iter(cid2_fs)]
    cid2_fk = [np.array(fk, fk_dtype_) for fk in iter(cid2_fk)]
    # Ensure shape
    for cid in xrange(len(cid2_fm)):
        cid2_fm[cid].shape = (cid2_fm[cid].size // 2, 2)
    # Cast lists
    cid2_fm = np.array(cid2_fm, list)
    cid2_fs = np.array(cid2_fs, list)
    cid2_fk = np.array(cid2_fk, list)
    chipmatch = (cid2_fm, cid2_fs, cid2_fk)
    return chipmatch


def new_fmfsfk(ibs):
    num_chips = ibs.get_num_chips()
    cid2_fm = [[] for _ in xrange(num_chips)]
    cid2_fs = [[] for _ in xrange(num_chips)]
    cid2_fk = [[] for _ in xrange(num_chips)]
    return cid2_fm, cid2_fs, cid2_fk


#============================
# 6) QueryResult Format
#============================


@profile
def chipmatch_to_resdict(ibs, qcx2_chipmatch, filt2_meta, qreq):
    print('[mf] Step 6) Convert chipmatch -> res')
    uid = qreq.get_uid()
    score_method = qreq.cfg.agg_cfg.score_method
    # Create the result structures for each query.
    qcx2_res = {}
    for qcx in qcx2_chipmatch.iterkeys():
        # For each query's chipmatch
        chipmatch = qcx2_chipmatch[qcx]
        # Perform final scoring
        cid2_score = score_chipmatch(ibs, qcx, chipmatch, score_method, qreq)
        # Create a query result structure
        res = qr.QueryResult(qcx, uid)
        res.cid2_score = cid2_score
        (res.cid2_fm, res.cid2_fs, res.cid2_fk) = chipmatch
        res.filt2_meta = {}  # dbgstats
        for filt, qcx2_meta in filt2_meta.iteritems():
            res.filt2_meta[filt] = qcx2_meta[qcx]  # things like k+1th
        qcx2_res[qcx] = res
    # Retain original score method
    return qcx2_res


def load_resdict(ibs, qreq):
    # Load the result structures for each query.
    qcxs = qreq._qcxs
    uid = qreq.get_uid()
    ##IF DICT_COMPREHENSION
    qcx2_res = {qcx: qr.QueryResult(qcx, uid) for qcx in iter(qcxs)}
    [res.load(ibs) for res in qcx2_res.itervalues()]
    ##ELSE
    #qcx2_res = {}
    #for qcx in qcxs:
        #res = qr.QueryResult(qcx, uid)
        #res.load(ibs)
        #qcx2_res[qcx] = res
    ##ENDIF
    return qcx2_res


def try_load_resdict(ibs, qreq):
    # Load the result structures for each query.
    qcxs = qreq._qcxs
    uid = qreq.get_uid()
    qcx2_res = {}
    failed_qcxs = []
    for qcx in qcxs:
        try:
            res = qr.QueryResult(qcx, uid)
            res.load(ibs)
            qcx2_res[qcx] = res
        except IOError:
            failed_qcxs.append(qcx)
    return qcx2_res, failed_qcxs


#============================
# Scoring Mechanism
#============================


@profile
def score_chipmatch(ibs, qcx, chipmatch, score_method, qreq=None):
    (cid2_fm, cid2_fs, cid2_fk) = chipmatch
    # HACK: Im not even sure if the 'w' suffix is correctly handled anymore
    if score_method.find('w') == len(score_method) - 1:
        score_method = score_method[:-1]
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        cid2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'pl':
        cid2_score, nid2_score = vr2.score_chipmatch_PL(ibs, qcx, chipmatch, qreq)
    elif score_method == 'borda':
        cid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qcx, chipmatch, qreq, 'borda')
    elif score_method == 'topk':
        cid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qcx, chipmatch, qreq, 'topk')
    elif score_method.startswith('coverage'):
        # Method num is at the end of coverage
        method = int(score_method.replace('coverage', '0'))
        cid2_score = coverage.score_chipmatch_coverage(ibs, qcx, chipmatch, qreq, method=method)
    else:
        raise Exception('[mf] unknown scoring method:' + score_method)
    cid2_nMatch = np.array(map(len, cid2_fm))
    # Autoremove chips with no match support
    cid2_score *= (cid2_nMatch != 0)
    return cid2_score
