#!/usr/bin/env python
import ctypes as C
import numpy as np
import vtool
import vtool.keypoint as ktool
import vtool.tests.dummy as dummy
import vtool.spatial_verification as sver
import utool as ut

TAU = 2 * np.pi  # tauday.org
c_double_p = C.POINTER(C.c_double)

# copied/adapted from _pyhesaff.py
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=FLAGS_RW)
fms_t = np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags=FLAGS_RW)

inliers_t = np.ctypeslib.ndpointer(dtype=np.bool, ndim=2, flags=FLAGS_RW)
errs_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=FLAGS_RW)
mats_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=FLAGS_RW)


def call_python_version(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    return sver.get_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)


c_sver = C.cdll['./sver.so']
c_getaffineinliers = c_sver['get_affine_inliers']
c_getaffineinliers.restype = None
c_getaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                kpts_t, C.c_size_t,
                                fms_t, C.c_size_t,
                                C.c_double, C.c_double, C.c_double,
                                inliers_t, errs_t, mats_t]


def call_cpp_version(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    out_inlier_flags = np.empty((len(fm), len(fm)), np.bool)
    out_errors = np.empty((len(fm), 3, len(fm)), np.float64)
    out_mats = np.empty((len(fm), 3, 3), np.float64)
    c_getaffineinliers(kpts1, 6*len(kpts1),
        kpts2, 6*len(kpts2),
        fm, 2*len(fm),
        xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh,
        out_inlier_flags, out_errors, out_mats)
    out_inliers = [np.where(row)[0] for row in out_inlier_flags]
    return out_inliers, out_errors, out_mats


def test_with_dummy_keypoints(f):
    kpts1, kpts2 = dummy.get_dummy_kpts_pair((100, 100))
    #fm = np.ascontiguousarray(dummy.make_dummy_fm(len(kpts1)).astype(np.uint))
    fm = np.ascontiguousarray(dummy.make_dummy_fm(len(kpts1)).astype(np.int64))
    xy_thresh_sqrd = ktool.KPTS_DTYPE(.009) ** 2
    scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
    ori_thresh = ktool.KPTS_DTYPE(TAU / 4)
    #print(repr([kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh]))
    return f(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)


def test_calling():
    py_results = test_with_dummy_keypoints(call_python_version)
    cpp_results = test_with_dummy_keypoints(call_cpp_version)
    assert np.allclose(py_results[0], cpp_results[0])
    assert np.allclose(py_results[1], cpp_results[1])
    assert np.allclose(py_results[2], cpp_results[2])
    ut.embed()


def call_hello():
    lib = C.cdll['./sver.so']
    hello = lib['hello_world']
    hello()


if __name__ == '__main__':
    test_calling()
    #call_hello()
