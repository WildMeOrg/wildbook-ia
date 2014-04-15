#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import vtool.keypoint as ktool
import vtool.linalg as ltool
from drawtool import draw_sv
from drawtool import draw_func2 as df2
import numpy as np
import vtool.tests.dummy as dummy
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[tets_sv]', DEBUG=False)


def test_affine_inliers2(kpts1, kpts2, fm, nShow=6):
    chip1 = dummy.get_kpts_dummy_img(kpts1)
    chip2 = dummy.get_kpts_dummy_img(kpts2)
    xy_thresh_sqrd = 150 ** 2
    scale_thresh_sqrd = 1.2
    ori_thresh = np.tau

    kpts1_m = kpts1[fm.T[0]]
    kpts2_m = kpts2[fm.T[1]]

    # Get keypoints to project
    invVR1s_m = ktool.get_invV_mats(kpts1_m, with_trans=True, with_ori=True)
    V1s_m     = ktool.get_V_mats(kpts1_m, with_trans=True, with_ori=True)
    invVR2s_m = ktool.get_invV_mats(kpts2_m, with_trans=True, with_ori=True)
    # The transform from kp1 to kp2 is given as:
    # Aff = inv(invV2).dot(V1)
    Aff_mats = ktool.matrix_multiply(invVR2s_m, V1s_m)
    # Get components to test projects against
    det2_m = ktool.get_sqrd_scales(kpts2_m)  # PYX FLOAT_1D
    _xy2_m   = invVR2s_m[:, 0, 0:2]
    _ori2_m  = ktool.get_invVR_mats_oris(invVR2s_m)
    # Test all hypothesis
    errors_list = []
    def test_hypothosis_inliers(Aff):
        # Map keypoints from image 1 onto image 2
        invVR1s_mt = ktool.matrix_multiply(Aff, invVR1s_m)
        # Get projection components
        _xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
        _ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
        _det1_mt  = ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
        # Check for projection errors
        ori_err   = ltool.ori_distance(_ori1_mt, _ori2_m)
        xy_err    = ltool.L2_sqrd(_xy2_m, _xy1_mt)
        scale_err = ltool.det_distance(_det1_mt, det2_m)
        # Mark keypoints which are inliers to this hypothosis
        xy_inliers_flag = xy_err < xy_thresh_sqrd
        ori_inliers_flag = ori_err < ori_thresh
        scale_inliers_flag = scale_err < scale_thresh_sqrd
        hypo_inliers_flag = ltool.logical_and_many(xy_inliers_flag, ori_inliers_flag, scale_inliers_flag)
        hypo_inliers = np.where(hypo_inliers_flag)[0]
        # TODO Add uniqueness of matches constraint

        def packerrors(flag, err):
            return utool.indentjoin(['%5s %f' % tup for tup in zip(flag, err)])
        errors = {
            'scale_err': packerrors(scale_inliers_flag, np.sqrt(scale_err)),
            'ori_err': packerrors(ori_inliers_flag, ori_err),
            'xy_err': packerrors(xy_inliers_flag, np.sqrt(xy_err))
        }
        # TEST
        errors_list.append(errors)
        return hypo_inliers

    # Enumerate all hypothesis
    inliers_list = [test_hypothosis_inliers(Aff) for Aff in Aff_mats]
    # Determine best hypothesis
    nInliers_list = np.array(map(len, inliers_list))
    best_mxs = nInliers_list.argsort()[::-1]

    for fnum, mx in enumerate(best_mxs[0:min(len(best_mxs), nShow)]):
        Aff = Aff_mats[mx]
        aff_inliers = inliers_list[mx]
        #errors = errors_list[mx]
        #print(utool.dict_str(errors, strvals=True))
        draw_sv.show_sv_affine(chip1, chip2, kpts1, kpts2, fm,
                               Aff, aff_inliers, mx=mx, fnum=fnum)
        df2.set_figtitle('#inliers = %r' % (nInliers_list[mx],))
    # Enumerate all hypothesis
    # Determine best hypothesis


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    utool.util_inject._inject_colored_exception_hook()
    kpts1 = dummy.pertebed_grid_kpts(seed=1, damping=2)
    kpts2 = dummy.pertebed_grid_kpts(seed=2, damping=2)
    #kpts2 = ktool.get_grid_kpts()
    fm = dummy.make_dummy_fm(len(kpts1))
    nShow = utool.get_arg('--nShow', int, 1)
    test_affine_inliers2(kpts1, kpts2, fm, nShow=nShow)
    exec(df2.present(wh=(500, 300), num_rc=(3, 1)))
