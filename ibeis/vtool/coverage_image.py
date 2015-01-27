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
    patch = ptool.gaussian_patch()
    # 2 and 3 are scale modes
    if kwargs.get('method', 0) in [2, 3]:
        # Bigger keypoints should get smaller weights
        mx2_scale = np.sqrt([a * d for (x, y, a, c, d) in mx2_kp])
        mx2_score = mx2_score / mx2_scale
    dstimg = warp_patch_into_kpts(mx2_kp, patch, chip.shape[0:2],
                                  fx2_score=mx2_score, **kwargs)
    return dstimg


def get_match_coverage_images(ibs, aid1, aid2, fm, mx2_score, **kwargs):
    sel_fx1, sel_fx2 = fm.T
    dstimg1 = get_annot_match_covimg(ibs, aid1, sel_fx1, mx2_score, **kwargs)
    dstimg2 = get_annot_match_covimg(ibs, aid1, sel_fx1, mx2_score, **kwargs)
    return dstimg1, dstimg2


def warp_patch_into_kpts(kpts, patch, chip_shape, fx2_score=None,
                         scale_factor=1.0, mode='sum', **kwargs):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch (ndarray): patch to warp (like gaussian)
        chip_shape (tuple):
        fx2_score (ndarray): score for every keypoint
        scale_factor (float):

    Returns:
        ?: None

    CommandLine:
        python -m vtool.coverage_image --test-warp_patch_into_kpts
        python -m vtool.coverage_image --test-warp_patch_into_kpts --show
        python -m vtool.coverage_image --test-warp_patch_into_kpts --show --hole
        python -m vtool.coverage_image --test-warp_patch_into_kpts --show --square
        python -m vtool.coverage_image --test-warp_patch_into_kpts --show --square --hole

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
        >>> srcshape = (19, 19)
        >>> radius = srcshape[0] / 2.0
        >>> sigma = 0.95 * radius
        >>> #sigma = 1.6
        >>> #srcshape = (7, 7)
        >>> #srcshape = (7, 7)
        >>> #srcshape = (11, 11)
        >>> #srcshape = (170, 170)
        >>> SQUARE = ut.get_argflag('--square')
        >>> HOLE = ut.get_argflag('--hole')
        >>> if SQUARE:
        >>>     patch = np.ones(srcshape)
        >>> else:
        >>>     patch = ptool.gaussian_patch(shape=srcshape, sigma=sigma) #, norm_01=False)
        >>> if HOLE:
        >>>     patch[int(patch.shape[0] / 2), int(patch.shape[1] / 2)] = 0
        >>> # execute function
        >>> dstimg = warp_patch_into_kpts(kpts, patch, chip_shape, fx2_score, scale_factor)
        >>> # verify results
        >>> print('dstimg stats %r' % (ut.get_stats_str(dstimg, axis=None)),)
        >>> print('patch stats %r' % (ut.get_stats_str(patch, axis=None)),)
        >>> #print(patch.sum())
        >>> assert np.all(ut.inbounds(dstimg, 0, 1, eq=True))

        >>> # show results
        >>> if ut.get_argflag('--show'):
        >>>     import plottool as pt
        >>>     mask = dstimg
        >>>     show_coverage_map(chip, mask, patch, kpts)
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
    patch_shape = patch.shape
    # Scale keypoints into destination image
    M_list = ktool.get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
    affmat_list = M_list[:, 0:2, :]
    # cv2 warpAffine flags
    BIG_KEYPOINT_LOW_WEIGHT_HACK = True
    def warped_patch_generator():
        warped_dst = np.zeros(shape, dtype=np.float32)
        # each score is spread across its contributing pixels
        for (M, score) in zip(affmat_list, fx2_score):
            src = patch * score
            # References:
            #   http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
            # It seems that this will not operate in place even if a destination
            # array is passed in. Thus we need to find a way to work around this
            # massive memory usage.
            warped_dst = cv2.warpAffine(src, M, dsize,
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            if BIG_KEYPOINT_LOW_WEIGHT_HACK:
                total_weight = np.sqrt(warped_dst.sum()) * .1
                #divisor =  / 1000)
                #print(warp_sum)
                #print(warped_dst.max())
                if total_weight > 1:
                    # Whatever the size of the keypoint is it should
                    # contribute a total of 1 score
                    np.divide(warped_dst, total_weight, out=warped_dst)
                #print(warped_dst.max())
            yield warped_dst
    # For each keypoint
    # warp a gaussian scaled by the feature score into the image
    # Either max or sum
    dstimg = np.zeros(shape, dtype=np.float32)
    if mode == 'max':
        print(ut.get_resource_usage_str())
        for warped in warped_patch_generator():
            np.maximum(warped, dstimg, out=dstimg)
            del warped
            #dstimg = np.dstack((warped.T, dstimg)).max(axis=2)
    elif mode == 'sum':
        for warped in warped_patch_generator():
            np.add(warped, dstimg, out=dstimg)
            #dstimg += warped
        # HACK FOR SUM: DO NOT DO THIS FOR MAX
        dstimg[dstimg > 1.0] = 1.0
    else:
        raise AssertionError('Unknown mode=%r' % (mode,))
    return dstimg


def show_coverage_map(chip, mask, patch, kpts, fnum=None, ell_alpha=.6,
                      show_mask_kpts=False):
    import plottool as pt
    masked_chip = (chip * mask[:, :, None]).astype(np.uint8)
    if fnum is None:
        fnum = pt.next_fnum()
    pnum_ = pt.get_pnum_func(nRows=2, nCols=2)
    pt.imshow((patch * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(0), title='patch')
    #ut.embed()
    pt.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(1), title='mask')
    if show_mask_kpts:
        pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    pt.imshow(chip, fnum=fnum, pnum=pnum_(2), title='chip')
    pt.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    pt.imshow(masked_chip, fnum=fnum, pnum=pnum_(3), title='masked chip')
    #pt.draw_kpts2(kpts)


def make_coverage_mask(kpts, chip_shape, fx2_score=None, mode=None, **kwargs):
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
        python -m vtool.patch --test-test_show_gaussian_patches2 --show
        python -m vtool.coverage_image --test-make_coverage_mask --show
        python -m vtool.coverage_image --test-make_coverage_mask

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.coverage_image import *  # NOQA
        >>> import vtool as vt
        >>> import plottool as pt
        >>> import pyhesaff
        >>> #img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img_fpath = ut.grab_test_imgpath('lena.png')
        >>> (kpts, vecs) = pyhesaff.detect_kpts(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> kwargs = {}
        >>> chip_shape = chip.shape
        >>> # execute function
        >>> dstimg, patch = make_coverage_mask(kpts, chip_shape)
        >>> # show results
        >>> if ut.get_argflag('--show'):
        >>>     # FIXME:  params
        >>>     srcshape = (5, 5)
        >>>     sigma = 1.6
        >>>     #srcshape = (75, 75)
        >>>     mask = dstimg
        >>>     show_coverage_map(chip, mask, patch, kpts)
        >>>     pt.show_if_requested()

    #>>> fnum = 1
    #>>> pnum_ = pt.get_pnum_func(nRows=1, nCols=3)
    #>>> pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(0))
    #>>> pt.draw_kpts2(kpts)
    #>>> pt.imshow(dstimg * 255, fnum=fnum, pnum=pnum_(1))
    #>>> pt.imshow(chip, fnum=fnum, pnum=pnum_(2))
    #>>> pt.draw_kpts2(kpts, rect=True)
    #>>> pt.show_if_requested()
    """
    #srcshape = (7, 7)
    #srcshape = (3, 3)
    #srcshape = (5, 5)
    srcshape = (19, 19)
    #sigma = 1.6
    # Perdoch uses roughly .95 of the radius
    USE_PERDOCH_VALS = True
    if USE_PERDOCH_VALS:
        radius = srcshape[0] / 2.0
        sigma = 0.4 * radius
        sigma = 0.95 * radius
    #srcshape = (75, 75)
    # Similar to SIFT's computeCircularGaussMask in helpers.cpp
    # uses smmWindowSize=19 in hesaff for patch size. and 1.6 for sigma
    patch = ptool.gaussian_patch(shape=srcshape, sigma=sigma)
    norm_01 = True
    if mode is None:
        mode = 'sum'
    #mode = 'max'
    if norm_01:
        patch /= patch.max()
    #, norm_01=False)
    scale_factor = .25
    dstimg = warp_patch_into_kpts(kpts, patch, chip_shape, mode=mode,
                                  fx2_score=fx2_score, scale_factor=scale_factor,
                                  **kwargs)
    #cv2.GaussianBlur(dstimg, ksize=(9, 9,), sigmaX=5.0, sigmaY=5.0,
    #                 dst=dstimg, borderType=cv2.BORDER_CONSTANT)
    #import vtool as vt
    cv2.GaussianBlur(dstimg, ksize=(17, 17,), sigmaX=5.0, sigmaY=5.0,
                     dst=dstimg, borderType=cv2.BORDER_CONSTANT)
    dsize = tuple(chip_shape[0:2][::-1])
    dstimg = cv2.resize(dstimg, dsize)
    print(dstimg)
    return dstimg, patch


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
