#!/usr/bin/env python
import ctypes as C
import numpy as np
import vtool.keypoint as ktool
import vtool.tests.dummy as dummy

TAU = 2 * np.pi  # tauday.org
c_double_p = C.POINTER(C.c_double)

# copied/adapted from _pyhesaff.py
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=FLAGS_RW)
fms_t = np.ctypeslib.ndpointer(dtype=np.uint, ndim=2, flags=FLAGS_RW)

def test_calling():
    kpts1, kpts2 = dummy.get_dummy_kpts_pair((100, 100))
    fm = np.ascontiguousarray(dummy.make_dummy_fm(len(kpts1)).astype(np.uint))
    xy_thresh_sqrd = ktool.KPTS_DTYPE(.009) ** 2
    scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
    ori_thresh = ktool.KPTS_DTYPE(TAU / 4)
    print(repr([kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh]))
    #output = get_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)
    c_sver = C.cdll['./sver.so']
    gai = c_sver['get_affine_inliers']
    gai.restype = None
    gai.argtypes = [kpts_t, C.c_size_t, kpts_t, C.c_size_t, fms_t, C.c_size_t, C.c_double, C.c_double, C.c_double]
    gai(kpts1, len(kpts1), kpts2, len(kpts2), fm, len(fm), xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)

def call_hello():
    lib = C.cdll['./sver.so']
    hello = lib['hello_world']
    hello()

if __name__ == '__main__':
    test_calling()
    #call_hello()
