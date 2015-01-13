from __future__ import absolute_import, division, print_function
from six.moves import zip, range, map  # NOQA
import cv2
import numpy as np
import utool as ut
from vtool import patch as ptool
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[cov]', DEBUG=False)

SCALE_FACTOR_DEFAULT = .05
METHOD_DEFAULT = 0


def score_chipmatch_coverage(ibs, qaid, chipmatch, qreq, method=0):
    from ibeis.model.hots import matching_functions as mf
    prescore_method = 'csum'
    nShortlist = 100
    daids_ = set(qreq._daids)
    (aid2_fm, aid2_fs, aid2_fk) = chipmatch
    aid2_prescore = mf.score_chipmatch(ibs, qaid, chipmatch, prescore_method, qreq)
    topx2_aid = aid2_prescore.argsort()[::-1]  # Only allow indexed aids to be in the top results
    topx2_aid = [aid for aid in iter(topx2_aid) if aid in daids_]
    nRerank = min(len(topx2_aid), nShortlist)
    aid2_score = [0 for _ in range(len(aid2_fm))]
    mark_progress, end_progress = ut.progress_func(nRerank, flush_after=10,
                                                      lbl='[cov] Compute coverage')
    for topx in range(nRerank):
        mark_progress(topx)
        aid2 = topx2_aid[topx]
        fm = aid2_fm[aid2]
        fs = aid2_fs[aid2]
        covscore = get_match_coverage_score(ibs, qaid, aid2, fm, fs, method=method)
        aid2_score[aid2] = covscore
    end_progress()
    return aid2_score


def get_match_coverage_score(ibs, aid1, aid2, fm, fs, **kwargs):
    if len(fm) == 0:
        return 0
    if 'scale_factor' not in kwargs:
        kwargs['scale_factor'] = SCALE_FACTOR_DEFAULT
    if 'method' not in kwargs:
        kwargs['method'] = METHOD_DEFAULT
    sel_fx1, sel_fx2 = fm.T
    method = kwargs.get('method', 0)
    score1 = get_annot_match_covscore(ibs, aid1, sel_fx1, fs, **kwargs)
    if method in [0, 2]:
        # 0 and 2 use both score
        score2 = get_annot_match_covscore(ibs, aid2, sel_fx2, fs, **kwargs)
        covscore = (score1 + score2) / 2
    elif method in [1, 3]:
        # 1 and 3 use just score 1
        covscore = score1
    else:
        raise NotImplemented('[cov] method=%r' % method)
    return covscore


def get_annot_match_covscore(ibs, aid, sel_fx, mx2_score, **kwargs):
    dstimg = get_annot_match_covimg(ibs, aid, sel_fx, mx2_score, **kwargs)
    score = dstimg.sum() / (dstimg.shape[0] * dstimg.shape[1])
    return score


def get_annot_match_covimg(ibs, aid, sel_fx, mx2_score, **kwargs):
    chip = ibs.get_annot_chips(aid)
    kpts = ibs.get_annot_kpts(aid)
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


def get_match_coverage_images(ibs, aid1, aid2, fm, mx2_score, **kwargs):
    sel_fx1, sel_fx2 = fm.T
    dstimg1 = get_annot_match_covimg(ibs, aid1, sel_fx1, mx2_score, **kwargs)
    dstimg2 = get_annot_match_covimg(ibs, aid1, sel_fx1, mx2_score, **kwargs)
    return dstimg1, dstimg2


def build_kpts_transforms(kpts, chip_shape, src_shape, scale_factor):
    """
    builds a transform for each keypoint which will map an image with shape
    src_shape into the keypoint location on a chip of shape chip_shape

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        chip_shape (tuple):
        src_shape (tuple):
        scale_factor (float):

    Returns:
        ndarray: transform_list

    CommandLine:
        python -m vtool.coverage_image --test-build_kpts_transforms

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.dummy.get_dummy_kpts(1)
        >>> chip = vt.dummy.get_kpts_dummy_img(kpts, intensity=10)
        >>> srcimg = ptool.gaussian_patch()
        >>> # build test data
        >>> chip_shape = chip.shape
        >>> src_shape = srcimg.shape
        >>> scale_factor = 1.0
        >>> # execute function
        >>> transform_list = build_kpts_transforms(kpts, chip_shape, src_shape, scale_factor)
        >>> # verify results
        >>> result = vt.kpts_repr(transform_list)
        >>> print(result)
        array([[[  0.75,   0.  ,  17.39],
                [ -0.73,   3.45,  15.48]],
               [[  0.34,   0.  ,  27.82],
                [ -0.73,   3.45,  15.48]],
               [[  1.75,   0.  ,  23.89],
                [  1.72,   1.5 ,  18.73]],
               [[  1.91,   0.  ,  24.32],
                [  2.52,   2.01,  13.13]],
               [[  2.29,   0.  ,  23.97],
                [  0.49,   1.68,  23.43]]])


    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> invVR_aff2Ds = [np.array(((a, 0, x),
        >>>                           (c, d, y),
        >>>                           (0, 0, 1),))
        >>>                 for (x, y, a, c, d, ori) in kpts]
        >>> invVR_3x3 = vt.get_invVR_mats3x3(kpts)
        >>> invV_3x3 = vt.get_invV_mats3x3(kpts)
        >>> assert np.all(np.array(invVR_aff2Ds) == invVR_3x3)
        >>> assert np.all(np.array(invVR_aff2Ds) == invV_3x3)

    Timeit:
        %timeit [np.array(((a, 0, x), (c, d, y), (0, 0, 1),)) for (x, y, a, c, d, ori) in kpts]
        %timeit vt.get_invVR_mats3x3(kpts)
        %timeit vt.get_invV_mats3x3(kpts) <- THIS IS ACTUALLY MUCH FASTER

    Ignore::
        %pylab qt4
        import plottool as pt
        pt.imshow(chip)
        pt.draw_kpts2(kpts)
        pt.update()


    """
    from vtool import keypoint as ktool
    patch_shape = src_shape
    perspective_list = ktool.get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
    transform_list = perspective_list[:, 0:2, :]
    #from vtool import linalg as ltool
    ##(h, w)   = chip_shape
    #(h_, w_) = src_shape
    #half_width  = w_ / 2.0
    #half_height = h_ / 2.0
    ## Center src image
    #T1 = ltool.translation_mat3x3(-half_width, -half_height)
    ## Scale src to the unit circle
    #S1 = ltool.scale_mat3x3(1 / w_, 1 / h_)
    ## Transform the source image to the keypoint ellipse
    #invVR_aff2Ds = ktool.get_invVR_mats3x3(kpts)
    ## Adjust for the requested scale factor
    #S2 = ltool.scale_mat3x3(scale_factor, scale_factor)
    ## Center the image
    ##T1 = np.array(((1, 0, -w_ / 2),
    ##               (0, 1, -h_ / 2),
    ##               (0, 0,       1),))
    ## Scale
    ##S1 = np.array(((1 / w_,      0,  0),
    ##               (0,      1 / h_,  0),
    ##               (0,           0,  1),))
    ##invVR_aff2Ds = [np.array(((a, 0, x),
    ##                          (c, d, y),
    ##                          (0, 0, 1),))
    ##                for (x, y, a, c, d, ori) in kpts]
    ##S2 = np.array(((scale_factor,           0,  0),
    ##               (0,           scale_factor,  0),
    ##               (0,                      0,  1),))
    #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
    #transform_list = [M[0:2] for M in perspective_list]
    return transform_list


def warp_srcimg_to_kpts(kpts, srcimg, chip_shape, fx2_score=None,
                        scale_factor=1.0, **kwargs):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        srcimg (?):
        chip_shape (?):
        fx2_score (ndarray):
        scale_factor (float):

    Returns:
        ?: None

    CommandLine:
        python -m vtool.coverage_image --test-warp_srcimg_to_kpts

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import pyhesaff
        >>> img_fpath    = ut.grab_test_imgpath('carl.jpg')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> chip = vt.imread(img_fpath)
        >>> kwargs = {}
        >>> chip_shape = chip.shape
        >>> fx2_score = np.ones(len(kpts))
        >>> scale_factor = 1.0
        >>> srcimg = ptool.gaussian_patch()
        >>> # execute function
        >>> dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, fx2_score, scale_factor)
        >>> # verify results
        >>> result = str(None)
        >>> print(result)
    """
    if len(kpts) == 0:
        return None
    if fx2_score is None:
        fx2_score = np.ones(len(kpts))
    # Allocate destination image
    (_h, _w) = chip_shape[0:2]
    (h, w) = int(_h * scale_factor), int(_w * scale_factor)
    dstimg = np.zeros((h, w), dtype=np.float32)
    dst_copy = dstimg.copy()
    src_shape = srcimg.shape
    # Scale keypoints into destination image
    fx2_M = build_kpts_transforms(kpts, (h, w), src_shape, scale_factor)
    # cv2 warp flags
    dsize = (w, h)
    flags = cv2.INTER_LINEAR  # cv2.INTER_LANCZOS4
    boderMode = cv2.BORDER_CONSTANT
    # mark prooress
    # For each keypoint warp a gaussian scaled by the feature score
    # into the image
    for (M, score) in zip(fx2_M, fx2_score):
        warped = cv2.warpAffine(srcimg * score, M, dsize,
                                dst=dst_copy,
                                flags=flags, borderMode=boderMode,
                                borderValue=0).T
        catmat = np.dstack((warped.T, dstimg))
        dstimg = catmat.max(axis=2)
    return dstimg


def get_coverage_map(kpts, chip_shape, **kwargs):
    # Create gaussian image to warp
    r"""
    Returns a intensity image denoting which pixels are covered by the input
    keypoints

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        chip_shape (?):

    Returns:
        ndarray: dstimg

    CommandLine:
        python -m vtool.coverage_image --test-get_coverage_map --show
        python -m vtool.coverage_image --test-get_coverage_map

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import plottool as pt
        >>> import pyhesaff
        >>> #img_fpath   = ut.grab_test_imgpath('carl.jpg')
        >>> img_fpath   = ut.grab_test_imgpath('lena.png')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> chip = vt.imread(img_fpath)
        >>> kwargs = {}
        >>> chip_shape = chip.shape
        >>> dstimg = get_coverage_map(kpts, chip_shape)
        >>> fnum = 1
        >>> pnum_ = pt.get_pnum_func(nRows=1, nCols=2)
        >>> pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(0))
        >>> pt.imshow(chip, fnum=fnum, pnum=pnum_(1))
        >>> pt.draw_kpts2(kpts)
        >>> pt.show_if_requested()
    """
    np.tau = 2 * np.pi
    srcimg = ptool.gaussian_patch()
    dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, **kwargs)
    return dstimg


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.coverage_image
        python -m vtool.coverage_image --allexamples
        python -m vtool.coverage_image --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
