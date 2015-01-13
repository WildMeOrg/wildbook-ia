from __future__ import absolute_import, division, print_function
from six.moves import zip, range, map  # NOQA
import cv2
import numpy as np
import utool as ut
from vtool import patch as ptool
from vtool import keypoint as ktool
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


def warp_srcimg_to_kpts(kpts, srcimg, chip_shape, fx2_score=None,
                        scale_factor=1.0, mode='sum', **kwargs):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        srcimg (ndarray): patch to warp (like gaussian)
        chip_shape (tuple):
        fx2_score (ndarray): score for every keypoint
        scale_factor (float):

    Returns:
        ?: None

    CommandLine:
        python -m vtool.coverage_image --test-warp_srcimg_to_kpts
        python -m vtool.coverage_image --test-warp_srcimg_to_kpts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import pyhesaff
        >>> img_fpath    = ut.grab_test_imgpath('carl.jpg')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> kwargs = {}
        >>> chip_shape = chip.shape
        >>> fx2_score = np.ones(len(kpts))
        >>> scale_factor = 1.0
        >>> sigma = 1.0
        >>> #srcshape = (3, 3)
        >>> srcshape = (11, 11)
        >>> #srcshape = (170, 170)
        >>> SQUARE = 0
        >>> if SQUARE:
        >>>     srcimg = np.ones(srcshape)
        >>> else:
        >>>     srcimg = ptool.gaussian_patch(shape=srcshape, sigma=sigma, norm_01=False)
        >>> #srcimg[int(srcimg.shape[0] / 2), int(srcimg.shape[1] / 2)] = 0
        >>> # execute function
        >>> dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, fx2_score, scale_factor)
        >>> # verify results
        >>> print(ut.get_stats_str(dstimg, axis=None))
        >>> assert np.all(ut.inbounds(dstimg, 0, 1, eq=True))
        >>> # show results
        >>> if ut.get_argflag('--show'):
        >>>     masked_chip = (chip * dstimg[:, :, None]).astype(np.uint8)
        >>>     import plottool as pt
        >>>     fnum = 1
        >>>     pnum_ = pt.get_pnum_func(nRows=2, nCols=2)
        >>>     pt.imshow(srcimg * 255, fnum=fnum, pnum=pnum_(0))
        >>>     pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(1))
        >>>     pt.draw_kpts2(kpts)
        >>>     pt.imshow(chip, fnum=fnum, pnum=pnum_(2))
        >>>     pt.draw_kpts2(kpts)
        >>>     pt.imshow(masked_chip, fnum=fnum, pnum=pnum_(3))
        >>>     #pt.draw_kpts2(kpts)
        >>>     pt.show_if_requested()

    Ignore::
        %pylab qt4
        import plottool as pt
        pt.imshow(chip)
        pt.draw_kpts2(kpts)
        pt.update()

        pt.imshow(warped * 255)
        pt.imshow(dstimg * 255)
    """
    #if len(kpts) == 0:
    #    return None
    if fx2_score is None:
        fx2_score = np.ones(len(kpts))
    chip_scale_h = int(np.ceil(chip_shape[0] * scale_factor))
    chip_scale_w = int(np.ceil(chip_shape[1] * scale_factor))
    dsize = (chip_scale_w, chip_scale_h)
    shape = dsize[::-1]
    # Allocate destination image
    dstimg = np.zeros(shape, dtype=np.float32)
    patch_shape = srcimg.shape
    # Scale keypoints into destination image
    M_list = ktool.get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
    affmat_list = M_list[:, 0:2, :]
    # cv2 warpAffine flags
    dst_copy = dstimg.copy()
    warpkw = dict(dst=dst_copy,
                  flags=cv2.INTER_LINEAR,
                  #flags=cv2.INTER_LANCZOS4,
                  borderMode=cv2.BORDER_CONSTANT,
                  borderValue=0)
    def warped_patch_generator():
        for (M, score) in zip(affmat_list, fx2_score):
            warped = cv2.warpAffine(srcimg * score, M, dsize, **warpkw).T
            yield warped
    # For each keypoint
    # warp a gaussian scaled by the feature score into the image
    # Either max or sum
    if mode == 'max':
        for warped in warped_patch_generator():
            dstimg = np.dstack((warped.T, dstimg)).max(axis=2)
    elif mode == 'sum':
        for warped in warped_patch_generator():
            dstimg += warped.T
        # HACK FOR SUM: DO NOT DO THIS FOR MAX
        dstimg[dstimg > 1.0] = 1.0
    else:
        raise AssertionError('Unknown mode=%r' % (mode,))
    return dstimg


def get_coverage_map(kpts, chip_shape, **kwargs):
    # Create gaussian image to warp
    r"""
    Returns a intensity image denoting which pixels are covered by the input
    keypoints

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        chip_shape (tuple):

    Returns:
        ndarray: dstimg

    CommandLine:
        python -m vtool.coverage_image --test-get_coverage_map --show
        python -m vtool.coverage_image --test-get_coverage_map

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import plottool as pt
        >>> import pyhesaff
        >>> #img_fpath   = ut.grab_test_imgpath('carl.jpg')
        >>> img_fpath   = ut.grab_test_imgpath('lena.png')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> kwargs = {}
        >>> chip_shape = chip.shape
        >>> # execute function
        >>> dstimg = get_coverage_map(kpts, chip_shape)
        >>> # show results
        >>> fnum = 1
        >>> pnum_ = pt.get_pnum_func(nRows=1, nCols=3)
        >>> pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(0))
        >>> pt.draw_kpts2(kpts)
        >>> pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(1))
        >>> pt.imshow(chip, fnum=fnum, pnum=pnum_(2))
        >>> pt.draw_kpts2(kpts)
        >>> pt.show_if_requested()
    """
    srcshape = (7, 7)
    #srcshape = (3, 3)
    #srcshape = (75, 75)
    srcimg = ptool.gaussian_patch(shape=srcshape, sigma=1.0, norm_01=False)
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
