from __future__ import division, print_function
import utool
print, print_,  printDBG, rrr, profile =\
    utool.inject(__name__, '[cov]', DEBUG=False)
# Standard
from itertools import izip
# Science
import cv2
import numpy as np
# HotSpotter
import utool
import matching_functions as mf
# VTool
import vtool.patch as ptool

SCALE_FACTOR_DEFAULT = .05
METHOD_DEFAULT = 0


def score_chipmatch_coverage(ibs, qcx, chipmatch, qreq, method=0):
    prescore_method = 'csum'
    nShortlist = 100
    dcxs_ = set(qreq._dcxs)
    (cid2_fm, cid2_fs, cid2_fk) = chipmatch
    cid2_prescore = mf.score_chipmatch(ibs, qcx, chipmatch, prescore_method, qreq)
    topx2_cx = cid2_prescore.argsort()[::-1]  # Only allow indexed cids to be in the top results
    topx2_cx = [cid for cid in iter(topx2_cx) if cid in dcxs_]
    nRerank = min(len(topx2_cx), nShortlist)
    cid2_score = [0 for _ in xrange(len(cid2_fm))]
    mark_progress, end_progress = utool.progress_func(nRerank, flush_after=10,
                                                      lbl='[cov] Compute coverage')
    for topx in xrange(nRerank):
        mark_progress(topx)
        cid2 = topx2_cx[topx]
        fm = cid2_fm[cid2]
        fs = cid2_fs[cid2]
        covscore = get_match_coverage_score(ibs, qcx, cid2, fm, fs, method=method)
        cid2_score[cid2] = covscore
    end_progress()
    return cid2_score


def get_match_coverage_score(ibs, cid1, cid2, fm, fs, **kwargs):
    if len(fm) == 0:
        return 0
    if not 'scale_factor' in kwargs:
        kwargs['scale_factor'] = SCALE_FACTOR_DEFAULT
    if not 'method' in kwargs:
        kwargs['method'] = METHOD_DEFAULT
    sel_fx1, sel_fx2 = fm.T
    method = kwargs.get('method', 0)
    score1 = get_cx_match_covscore(ibs, cid1, sel_fx1, fs, **kwargs)
    if method in [0, 2]:
        # 0 and 2 use both score
        score2 = get_cx_match_covscore(ibs, cid2, sel_fx2, fs, **kwargs)
        covscore = (score1 + score2) / 2
    elif method in [1, 3]:
        # 1 and 3 use just score 1
        covscore = score1
    else:
        raise NotImplemented('[cov] method=%r' % method)
    return covscore


def get_cx_match_covscore(ibs, cid, sel_fx, mx2_score, **kwargs):
    dstimg = get_cx_match_covimg(ibs, cid, sel_fx, mx2_score, **kwargs)
    score = dstimg.sum() / (dstimg.shape[0] * dstimg.shape[1])
    return score


def get_cx_match_covimg(ibs, cid, sel_fx, mx2_score, **kwargs):
    chip = ibs.get_chip(cid)
    kpts = ibs.get_kpts(cid)
    mx2_kp = kpts[sel_fx]
    srcimg = ptool.gaussian_patch()
    # 2 and 3 are scale modes
    if kwargs.get('method', 0) in [2, 3]:
        # Bigger keypoints should get smaller weights
        mx2_scale = np.sqrt([a * d for (x, y, a, c, d) in mx2_kp])
        mx2_score = mx2_score / mx2_scale
    dstimg = warp_srcimg_to_kpts(mx2_kp, srcimg, chip.shape[0:2],
                                 fx2_score=mx2_score, **kwargs)
    return dstimg


def get_match_coverage_images(ibs, cid1, cid2, fm, mx2_score, **kwargs):
    sel_fx1, sel_fx2 = fm.T
    dstimg1 = get_cx_match_covimg(ibs, cid1, sel_fx1, mx2_score, **kwargs)
    dstimg2 = get_cx_match_covimg(ibs, cid1, sel_fx1, mx2_score, **kwargs)
    return dstimg1, dstimg2


def warp_srcimg_to_kpts(fx2_kp, srcimg, chip_shape, fx2_score=None, **kwargs):
    if len(fx2_kp) == 0:
        return None
    if fx2_score is None:
        fx2_score = np.ones(len(fx2_kp))
    scale_factor = kwargs.get('scale_Factor', SCALE_FACTOR_DEFAULT)
    # Build destination image
    (h, w) = map(int, (chip_shape[0] * scale_factor, chip_shape[1] * scale_factor))
    dstimg = np.zeros((h, w), dtype=np.float32)
    dst_copy = dstimg.copy()
    src_shape = srcimg.shape
    # Build keypoint transforms
    fx2_M = build_kpts_transforms(fx2_kp, (h, w), src_shape, scale_factor)
    # cv2 warp flags
    dsize = (w, h)
    flags = cv2.INTER_LINEAR  # cv2.INTER_LANCZOS4
    boderMode = cv2.BORDER_CONSTANT
    # mark prooress
    mark_progress, end_progress = utool.progress_func(len(fx2_M),
                                                      flush_after=20,
                                                      mark_after=1000,
                                                      lbl='coverage warp ')
    # For each keypoint warp a gaussian scaled by the feature score
    # into the image
    count = 0
    for count, (M, score) in enumerate(izip(fx2_M, fx2_score)):
        mark_progress(count)
        warped = cv2.warpAffine(srcimg * score, M, dsize,
                                dst=dst_copy,
                                flags=flags, borderMode=boderMode,
                                borderValue=0).T
        catmat = np.dstack((warped.T, dstimg))
        dstimg = catmat.max(axis=2)
    mark_progress(count)
    end_progress()
    return dstimg


def build_kpts_transforms(kpts, chip_shape, src_shape, scale_factor):
    (h, w) = chip_shape
    (h_, w_) = src_shape
    T1 = np.array(((1, 0, -w_ / 2),
                   (0, 1, -h_ / 2),
                   (0, 0,       1),))
    S1 = np.array(((1 / w_,      0,  0),
                   (0,      1 / h_,  0),
                   (0,           0,  1),))
    invVR_aff2Ds = [np.array(((a, 0, x),
                              (c, d, y),
                              (0, 0, 1),)) for (x, y, a, c, d, ori) in kpts]
    S2 = np.array(((scale_factor,      0,  0),
                   (0,      scale_factor,  0),
                   (0,           0,  1),))
    perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
    transform_list = [M[0:2] for M in perspective_list]
    return transform_list


def get_coverage_map(kpts, chip_shape, **kwargs):
    # Create gaussian image to warp
    np.tau = 2 * np.pi
    srcimg = ptool.gaussian_patch()
    dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, **kwargs)
    return dstimg
