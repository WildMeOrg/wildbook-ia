#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import ctypes as C
import numpy as np
#import vtool
import vtool.keypoint as ktool
import utool as ut
from os.path import dirname, join

TAU = 2 * np.pi  # tauday.com
c_double_p = C.POINTER(C.c_double)

# copied/adapted from _pyhesaff.py
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=FLAGS_RW)
fms_t = np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags=FLAGS_RW)

inliers_t = np.ctypeslib.ndpointer(dtype=np.bool, ndim=2, flags=FLAGS_RW)
errs_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=FLAGS_RW)
mats_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=FLAGS_RW)


dpath = dirname(__file__)
cpp_fname = join(dpath, 'sver.cpp')
lib_fname = join(dpath, 'sver.so')


if ut.get_argflag('--rebuild-sver'):
    cflags = '-shared -fPIC -O2 -ffast-math'
    cmd_fmtstr = 'g++ -Wall -Wextra {cpp_fname} -lopencv_core {cflags} -o {lib_fname}'
    cmd_str = cmd_fmtstr.format(**locals())
    ut.cmd(cmd_str)


c_sver = C.cdll[lib_fname]
c_getaffineinliers = c_sver['get_affine_inliers']
c_getaffineinliers.restype = None
c_getaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                kpts_t, C.c_size_t,
                                fms_t, C.c_size_t,
                                C.c_double, C.c_double, C.c_double,
                                inliers_t, errs_t, mats_t]


def get_affine_inliers_cpp(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    #np.ascontiguousarray(kpts1)
    #with ut.Timer('PreC'):
    num_matches = len(fm)
    fm = np.ascontiguousarray(fm, dtype=np.int64)
    out_inlier_flags = np.empty((num_matches, num_matches), np.bool)
    out_errors = np.empty((num_matches, 3, num_matches), np.float64)
    out_mats = np.empty((num_matches, 3, 3), np.float64)
    #with ut.Timer('C'):
    c_getaffineinliers(kpts1, kpts1.size,
                       kpts2, kpts2.size,
                       fm, fm.size,
                       xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh,
                       out_inlier_flags, out_errors, out_mats)
    #with ut.Timer('C'):
    out_inliers = [np.where(row)[0] for row in out_inlier_flags]
    return out_inliers, out_errors, out_mats


def test_calling():
    """
    Test to ensure cpp and python agree and that cpp is faster

    CommandLine:
        python -m vtool.sver_c_wrapper --test-test_calling
        python -m vtool.sver_c_wrapper --test-test_calling --rebuild-sver
        python -m vtool.sver_c_wrapper --test-test_calling --show
        python -m vtool.sver_c_wrapper --test-test_calling --show --dummy
        python -m vtool.sver_c_wrapper --test-test_calling --show --fname1=easy1.png --fname2=easy2.png
        python -m vtool.sver_c_wrapper --test-test_calling --show --fname1=easy1.png --fname2=hard3.png
        python -m vtool.sver_c_wrapper --test-test_calling --show --fname1=carl.jpg --fname2=hard3.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.sver_c_wrapper import *  # NOQA
        >>> test_calling()

    Ignore:
        %timeit call_python_version(*args)
        %timeit get_affine_inliers_cpp(*args)
    """
    import vtool.spatial_verification as sver
    import vtool.tests.dummy as dummy
    xy_thresh_sqrd = ktool.KPTS_DTYPE(.4)
    scale_thresh_sqrd = ktool.KPTS_DTYPE(2.0)
    ori_thresh = ktool.KPTS_DTYPE(TAU / 4.0)
    keys = 'xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh'.split(', ')
    print(ut.dict_str(ut.dict_subset(locals(), keys)))

    if ut.get_argflag('--dummy'):
        (kpts1, kpts2, fm_input, fs_input, rchip1, rchip2) = dummy.testdata_dummy_matches()
    else:
        fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
        fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
        (kpts1, kpts2, fm_input, fs_input, rchip1, rchip2) = dummy.testdata_ratio_matches(fname1, fname2)

    # pack up call to aff hypothesis
    args = (kpts1, kpts2, fm_input, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)

    # test both versions
    with ut.Timer('time aff hyothesis python') as t_py:
        out_inliers_py, out_errors_py, out_mats_py = sver.get_affine_inliers(*args)
    with ut.Timer('time aff hyothesis c') as t_c:
        out_inliers_c, out_errors_c, out_mats_c = get_affine_inliers_cpp(*args)

    print('speedup = %r' % (t_py.ellapsed / t_c.ellapsed))

    msg =  'c and python disagree'
    try:
        assert ut.lists_eq(out_inliers_c, out_inliers_py), msg
    except AssertionError as ex:
        ut.printex(ex)
        raise
    try:
        passed, error = ut.almost_eq(out_errors_c, out_errors_py, 1E-7, ret_error=True)
        assert np.all(passed), msg
    except AssertionError as ex:
        passed_flat = passed.ravel()
        error_flat = error.ravel()
        failed_indexes = np.where(~passed_flat)[0]
        failing_errors = error_flat.take(failed_indexes)
        print(failing_errors)
        ut.printex(ex)
        raise
    try:
        assert np.all(ut.almost_eq(out_mats_c, out_mats_py, 1E-9)), msg
    except AssertionError as ex:
        ut.printex(ex)
        raise

    num_inliers_list = np.array(map(len, out_inliers_c))
    best_argx = num_inliers_list.argmax()
    #best_inliers_py = out_inliers_py[best_argx]
    best_inliers_c = out_inliers_c[best_argx]

    fm_output = fm_input.take(best_inliers_c, axis=0)
    #ut.embed()

    import plottool as pt
    fnum = pt.next_fnum()
    pt.figure(fnum=fnum, doclf=True, docla=True)
    pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm_input, ell_linewidth=5, fnum=fnum, pnum=(2, 1, 1))
    pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm_output, ell_linewidth=5, fnum=fnum, pnum=(2, 1, 2))
    pt.show_if_requested()


def call_hello():
    lib = C.cdll['./sver.so']
    hello = lib['hello_world']
    hello()


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.sver_c_wrapper
        python -m vtool.sver_c_wrapper --allexamples
        python -m vtool.sver_c_wrapper --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
