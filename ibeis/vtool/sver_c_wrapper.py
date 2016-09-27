#!/usr/bin/env python
"""
wraps c implementations slower parts of spatial verification

CommandLine:
    python -m vtool.sver_c_wrapper --rebuild-sver
    python -m vtool.sver_c_wrapper --rebuild-sver --allexamples
    python -m vtool.sver_c_wrapper --allexamples

    python -m vtool.sver_c_wrapper --test-test_sver_wrapper --rebuild-sver
"""
from __future__ import absolute_import, division, print_function
import ctypes as C
import numpy as np
import vtool.keypoint as ktool
import utool as ut
from os.path import dirname, join, realpath
# TODO: move to utool?
from vtool.other import asserteq, compare_implementations  # NOQA
print, rrr, profile = ut.inject2(__name__)

TAU = 2 * np.pi  # References: tauday.com
c_double_p = C.POINTER(C.c_double)

# copied/adapted from _pyhesaff.py
kpts_dtype = np.float64
# this is because size_t is 32 bit on mingw even on 64 bit machines
fm_dtype = np.int32 if ut.WIN32 else np.int64
fs_dtype = np.float64
FLAGS_RW = 'aligned, c_contiguous, writeable'
FLAGS_RO = 'aligned, c_contiguous'

kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RO)
fm_t  = np.ctypeslib.ndpointer(dtype=fm_dtype, ndim=2, flags=FLAGS_RO)
fs_t  = np.ctypeslib.ndpointer(dtype=fs_dtype, ndim=1, flags=FLAGS_RO)


def inliers_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.bool, ndim=ndim, flags=FLAGS_RW)


def errs_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)


def mats_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)

dpath = dirname(__file__)

lib_fname = join(dpath, 'libsver' + ut.util_cplat.get_lib_ext())


if __name__ != '__main__':
    if ut.get_argflag('--rebuild-sver'):  # and __name__ != '__main__':
        USE_CMAKE = True
        if USE_CMAKE:
            root_dir = realpath(dirname(__file__))
            repo_dir = dirname(root_dir)
            ut.std_build_command(repo_dir)
        else:
            cpp_fname = join(dpath, 'sver.cpp')
            cflags = '-shared -fPIC -O2 -ffast-math'
            cmd_fmtstr = 'g++ -Wall -Wextra {cpp_fname} -lopencv_core {cflags} -o {lib_fname}'
            cmd_str = cmd_fmtstr.format(**locals())
            ut.cmd(cmd_str)

    try:
        c_sver = C.cdll[lib_fname]
    except Exception as ex:
        print('Failed to open lib_fname = %r' % (lib_fname,))
        ut.checkpath(lib_fname, verbose=True)
        raise
    c_getaffineinliers = c_sver['get_affine_inliers']
    c_getaffineinliers.restype = C.c_int
    # for every affine hypothesis, for every keypoint pair (is
    #  it an inlier, the error triples, the hypothesis itself)
    c_getaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                   kpts_t, C.c_size_t,
                                   fm_t, fs_t, C.c_size_t,
                                   C.c_double, C.c_double, C.c_double,
                                   inliers_t(2), errs_t(3), mats_t(3)]
    # for the best affine hypothesis, for every keypoint pair
    #  (is it an inlier, the error triples (transposed?), the
    #   hypothesis itself)
    c_getbestaffineinliers = c_sver['get_best_affine_inliers']
    c_getbestaffineinliers.restype = C.c_int
    c_getbestaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                       kpts_t, C.c_size_t,
                                       fm_t, fs_t, C.c_size_t,
                                       C.c_double, C.c_double, C.c_double,
                                       inliers_t(1), errs_t(2), mats_t(2)]


@profile
def get_affine_inliers_cpp(kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    #np.ascontiguousarray(kpts1)
    #with ut.Timer('PreC'):
    num_matches = len(fm)
    fm = np.ascontiguousarray(fm, dtype=fm_dtype)
    out_inlier_flags = np.empty((num_matches, num_matches), np.bool)
    out_errors = np.empty((num_matches, 3, num_matches), np.float64)
    out_mats = np.empty((num_matches, 3, 3), np.float64)
    #with ut.Timer('C'):
    c_getaffineinliers(kpts1, kpts1.size,
                       kpts2, kpts2.size,
                       fm, fs, len(fm),
                       xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh,
                       out_inlier_flags, out_errors, out_mats)
    #with ut.Timer('C'):
    out_inliers = [np.where(row)[0] for row in out_inlier_flags]
    out_errors = list(map(tuple, out_errors))
    return out_inliers, out_errors, out_mats


@profile
def get_best_affine_inliers_cpp(kpts1, kpts2, fm, fs, xy_thresh_sqrd,
                                scale_thresh_sqrd, ori_thresh):
    #np.ascontiguousarray(kpts1)
    #with ut.Timer('PreC'):
    fm = np.ascontiguousarray(fm, dtype=fm_dtype)
    out_inlier_flags = np.empty((len(fm),), np.bool)
    out_errors = np.empty((3, len(fm)), np.float64)
    out_mat = np.empty((3, 3), np.float64)
    #with ut.Timer('C'):
    c_getbestaffineinliers(kpts1, 6 * len(kpts1),
                           kpts2, 6 * len(kpts2),
                           fm, fs, len(fm),
                           xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh,
                           out_inlier_flags, out_errors, out_mat)
    #with ut.Timer('C'):
    out_inliers = np.where(out_inlier_flags)[0]
    out_errors = tuple(out_errors)
    return out_inliers, out_errors, out_mat


def test_sver_wrapper2():
    r"""
    CommandLine:
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper2
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper2 --no-c --quiet
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper2 --rebuild-sver

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.sver_c_wrapper import *  # NOQA
        >>> result = test_sver_wrapper2()
        >>> print(result)

    Ignore:
        C (Serial):
            unique cases affine inliers: [
                '[ 4 25 33 36 37 53]',
            ]
            unique cases homog inliers: [
                '[]',
            ]

        C (Parallel)
            unique cases affine inliers: [
                '[ 4 25 33 36 37 53]',
                '[10 19 25 29 36 39 53]',
            ]
            unique cases homog inliers: [
                '[10 43 53]',
                '[]',
            ]

        Python:
            unique cases affine inliers: [
                '[10 19 25 29 36 39 53]',
            ]
            unique cases homog inliers: [
                '[10 43 53]',
            ]
    """
    import vtool
    import vtool.tests.testdata_nondeterm_sver
    kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh, dlen_sqrd2, min_nInliers, match_weights, full_homog_checks = vtool.tests.testdata_nondeterm_sver.testdata_nondeterm_sver()
    inliers_list = []
    homog_inliers_list = []

    for x in range(10):
        sv_tup = vtool.spatially_verify_kpts(
            kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
            dlen_sqrd2, min_nInliers, match_weights=match_weights,
            full_homog_checks=full_homog_checks, returnAff=True)
        aff_inliers = sv_tup[3]
        inliers_list.append(str(aff_inliers))
        homog_inliers_list.append(str(sv_tup[0]))

        #print(sv_tup[0])
        #print(sv_tup[3])
    print('unique cases affine inliers: ' + ut.list_str(list(set(inliers_list))))
    print('unique cases homog inliers: ' + ut.list_str(list(set(homog_inliers_list))))


def test_sver_wrapper():
    """
    Test to ensure cpp and python agree and that cpp is faster

    CommandLine:
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --rebuild-sver
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --show
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --show --dummy
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --show --fname1=easy1.png --fname2=easy2.png
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --show --fname1=easy1.png --fname2=hard3.png
        python -m vtool.sver_c_wrapper --test-test_sver_wrapper --show --fname1=carl.jpg --fname2=hard3.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.sver_c_wrapper import *  # NOQA
        >>> test_sver_wrapper()

    Ignore:
        %timeit call_python_version(*args)
        %timeit get_affine_inliers_cpp(*args)
    """
    import vtool.spatial_verification as sver
    import vtool.tests.dummy as dummy
    xy_thresh_sqrd    = ktool.KPTS_DTYPE(.4)
    scale_thresh_sqrd = ktool.KPTS_DTYPE(2.0)
    ori_thresh        = ktool.KPTS_DTYPE(TAU / 4.0)
    keys = 'xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh'.split(', ')
    print(ut.dict_str(ut.dict_subset(locals(), keys)))

    def report_errors():
        pass

    if ut.get_argflag('--dummy'):
        testtup = dummy.testdata_dummy_matches()
        (kpts1, kpts2, fm_input, fs_input, rchip1, rchip2) = testtup
        fm_input = fm_input.astype(fm_dtype)
        #fm_input = fm_input[0:10].astype(fm_dtype)
        #fs_input = fs_input[0:10].astype(np.float32)
    else:
        fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
        fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
        testtup = dummy.testdata_ratio_matches(fname1, fname2)
        (kpts1, kpts2, fm_input, fs_input, rchip1, rchip2) = testtup

    # pack up call to aff hypothesis
    import vtool as vt
    import scipy.stats.mstats
    scales1 = vt.get_scales(kpts1.take(fm_input.T[0], axis=0))
    scales2 = vt.get_scales(kpts2.take(fm_input.T[1], axis=0))
    #fs_input = 1 / scipy.stats.mstats.gmean(np.vstack((scales1, scales2)))
    fs_input = scipy.stats.mstats.gmean(np.vstack((scales1, scales2)))
    print('fs_input = ' + ut.numpy_str(fs_input))
    #fs_input[0:-9] = 0
    #fs_input = np.ones(len(fm_input), dtype=fs_dtype)
    #ut.embed()
    #fs_input = scales1 * scales2
    args = (kpts1, kpts2, fm_input, fs_input, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)

    ex_list = []

    try:
        with ut.Indenter('[TEST1] '):
            inlier_tup = vt.compare_implementations(
                sver.get_affine_inliers,
                get_affine_inliers_cpp,
                args, lbl1='py', lbl2='c',
                output_lbl=('aff_inliers_list', 'aff_errors_list', 'Aff_mats')
            )
            out_inliers, out_errors, out_mats = inlier_tup
    except AssertionError as ex:
        ex_list.append(ex)
        raise

    try:
        import functools
        with ut.Indenter('[TEST2] '):
            bestinlier_tup = vt.compare_implementations(
                functools.partial(sver.get_best_affine_inliers, forcepy=True),
                get_best_affine_inliers_cpp,
                args, show_output=True, lbl1='py', lbl2='c',
                output_lbl=('bestinliers', 'besterror', 'bestmat')
            )
            bestinliers, besterror, bestmat = bestinlier_tup
    except AssertionError as ex:
        ex_list.append(ex)
        raise

    if len(ex_list) > 0:
        raise AssertionError('some tests failed. see previous stdout')

    #num_inliers_list = np.array(map(len, out_inliers_c))
    #best_argx = num_inliers_list.argmax()
    ##best_inliers_py = out_inliers_py[best_argx]
    #best_inliers_c = out_inliers_c[best_argx]
    if ut.show_was_requested():
        import plottool as pt
        fm_output = fm_input.take(bestinliers, axis=0)
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
