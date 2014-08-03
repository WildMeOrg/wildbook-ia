#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
import vtool.spatial_verification as sver
from plottool import draw_sv
from plottool import draw_func2 as df2
import numpy as np
import vtool.tests.dummy as dummy
import vtool.keypoint as ktool  # NOQA
import vtool.linalg as ltool  # NOQA
from  vtool.keypoint import *  # NOQA
from  vtool.spatial_verification import *  # NOQA
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[tets_sv]', DEBUG=False)


xy_thresh = ktool.KPTS_DTYPE(.009)
scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
ori_thresh = ktool.KPTS_DTYPE(np.tau / 4)


def test_sver(chip1, chip2, kpts1, kpts2, fm, nShow=6):

    xy_thresh_sqrd = ktool.get_diag_extent_sqrd(kpts2) * xy_thresh

    def pack_errors(xy_err, scale_err, ori_err):
        """ makes human readable errors """
        def _pack(bits, errs, thresh):
            return utool.indentjoin(['%5s %f < %f' % (bit, err, thresh) for (bit, err) in zip(bits, errs)])
        xy_flag = xy_err < xy_thresh_sqrd
        scale_flag = scale_err < scale_thresh_sqrd
        ori_flag = ori_err < ori_thresh
        errors_dict = {
            'xy_err':     _pack(xy_flag, np.sqrt(xy_err), np.sqrt(xy_thresh_sqrd)),
            'scale_err':  _pack(scale_flag, np.sqrt(scale_err), np.sqrt(scale_thresh_sqrd)),
            'ori_err':    _pack(ori_flag, ori_err, ori_thresh),
        }
        return errors_dict

    # Test each affine hypothesis
    #assert kpts1.dtype == ktool.KPTS_DTYPE, 'bad cast somewhere kpts1.dtype=%r' % (kpts1.dtype)
    #assert kpts2.dtype == ktool.KPTS_DTYPE, 'bad cast somewhere kpts2.dtype=%r' % (kpts2.dtype)
    #assert xy_thresh_sqrd.dtype == ktool.KPTS_DTYPE, 'bad cast somewhere xy_thresh_sqrd.dtype=%r' % (xy_thresh_sqrd.dtype)
    aff_hypo_tups = sver.get_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd,
                                            scale_thresh_sqrd, ori_thresh)
    inliers_list, errors_list, Aff_mats = aff_hypo_tups

    # Determine best hypothesis
    nInliers_list = np.array(list(map(len, inliers_list)))
    best_mxs = nInliers_list.argsort()[::-1]

    for fnum, mx in enumerate(best_mxs[0:min(len(best_mxs), nShow)]):
        Aff = Aff_mats[mx]
        aff_inliers = inliers_list[mx]
        if utool.get_flag('--print-error'):
            errors = pack_errors(*errors_list[mx])  # NOQA
            print(utool.dict_str(errors, strvals=True))

        homog_inliers, H = sver.get_homography_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd)

        kpts1_At = ktool.transform_kpts(kpts1, Aff)
        kpts1_Ht = ktool.transform_kpts(kpts1, H)
        kpts = kpts1
        M = H

        homog_tup = (homog_inliers, H)
        aff_tup = (aff_inliers, Aff)

        _args = (chip1, chip2, kpts1, kpts2, fm)
        _kw = dict(show_assign=True, show_kpts=True, mx=mx, fnum=fnum * 3)
        draw_sv.show_sv(*_args, aff_tup=aff_tup, homog_tup=homog_tup, **_kw)
        #draw_sv.show_sv(*_args, aff_tup=aff_tup, mx=mx, fnum=fnum * 3)
        #draw_sv.show_sv(*_args, homog_tup=homog_tup, mx=mx, fnum=3)

        df2.set_figtitle('# %r inliers (in rects, hypo in bold)' % (nInliers_list[mx],))
    return locals()


def get_dummy_test_vars():
    kpts1 = dummy.pertebed_grid_kpts(seed=12, damping=1.2)
    kpts2 = dummy.pertebed_grid_kpts(seed=24, damping=1.6)
    assert kpts1.dtype == np.float32
    assert kpts2.dtype == np.float32
    chip1 = dummy.get_kpts_dummy_img(kpts1)
    chip2 = dummy.get_kpts_dummy_img(kpts2)
    #kpts2 = ktool.get_grid_kpts()
    fm = dummy.make_dummy_fm(len(kpts1))
    return chip1, chip2, kpts1, kpts2, fm


def get_stashed_test_vars():
    chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup = utool.load_testdata(
        'chip1', 'chip2', 'kpts1', 'kpts2', 'fm', 'homog_tup', 'aff_tup')
    return chip1, chip2, kpts1, kpts2, fm


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    utool.util_inject.inject_colored_exceptions()
    nShow = utool.get_arg('--nShow', int, 1)
    chip1, chip2, kpts1, kpts2, fm = get_dummy_test_vars()
    #chip1, chip2, kpts1, kpts2, fm = get_stashed_test_vars()
    test_locals = test_sver(chip1, chip2, kpts1, kpts2, fm, nShow=nShow)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(df2.present())
