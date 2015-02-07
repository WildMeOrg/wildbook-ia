#!/usr/bin/env python
"""
wraps c implementations slower parts of spatial verification

CommandLine:
    python -m vtool.sver_c_wrapper --rebuild-sver
    python -m vtool.sver_c_wrapper --rebuild-sver --allexamples
"""
from __future__ import absolute_import, division, print_function
import ctypes as C
import numpy as np
#import vtool
import vtool.keypoint as ktool
import utool as ut
from os.path import dirname, join, realpath

print, print_, printDBG, rrr, profile = ut.inject(__name__, '[sver_c]')

TAU = 2 * np.pi  # References: tauday.com
c_double_p = C.POINTER(C.c_double)

# copied/adapted from _pyhesaff.py
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=FLAGS_RW)
fms_t = np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags=FLAGS_RW)

inliers_t = lambda ndim: np.ctypeslib.ndpointer(dtype=np.bool, ndim=ndim, flags=FLAGS_RW)
errs_t = lambda ndim: np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)
mats_t = lambda ndim: np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)

dpath = dirname(__file__)

lib_fname = join(dpath, 'libsver' + ut.util_cplat.get_lib_ext())


if ut.get_argflag('--rebuild-sver') and __name__ != '__main__':
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


c_sver = C.cdll[lib_fname]
c_getaffineinliers = c_sver['get_affine_inliers']
c_getaffineinliers.restype = None
# for every affine hypothesis, for every keypoint pair (is
#  it an inlier, the error triples, the hypothesis itself)
c_getaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                kpts_t, C.c_size_t,
                                fms_t, C.c_size_t,
                                C.c_double, C.c_double, C.c_double,
                                inliers_t(2), errs_t(3), mats_t(3)]
# for the best affine hypothesis, for every keypoint pair
#  (is it an inlier, the error triples (transposed?), the
#   hypothesis itself)
c_getbestaffineinliers = c_sver['get_best_affine_inliers']
c_getbestaffineinliers.restype = C.c_int
c_getbestaffineinliers.argtypes = [kpts_t, C.c_size_t,
                                    kpts_t, C.c_size_t,
                                    fms_t, C.c_size_t,
                                    C.c_double, C.c_double, C.c_double,
                                    inliers_t(1), errs_t(2), mats_t(2)]


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


def get_best_affine_inliers_cpp(kpts1, kpts2, fm, xy_thresh_sqrd,
                                scale_thresh_sqrd, ori_thresh):
    #np.ascontiguousarray(kpts1)
    #with ut.Timer('PreC'):
    fm = np.ascontiguousarray(fm, dtype=np.int64)
    out_inlier_flags = np.empty((len(fm),), np.bool)
    out_errors = np.empty((3, len(fm)), np.float64)
    out_mat = np.empty((3, 3), np.float64)
    #with ut.Timer('C'):
    c_getbestaffineinliers(kpts1, 6 * len(kpts1),
                           kpts2, 6 * len(kpts2),
                           fm, 2 * len(fm),
                           xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh,
                           out_inlier_flags, out_errors, out_mat)
    #with ut.Timer('C'):
    out_inliers = np.where(out_inlier_flags)[0]
    return out_inliers, out_errors, out_mat


def assert_output_equal(output1, output2, thresh=1E-7, nestpath=None, level=0,
        lbl1='', lbl2=''):
    """ recursive equality checks """
    # Setup
    if nestpath is None:
        # record the path through the nested structure as testing goes on
        nestpath = []
    # print out these variables in all error cases
    common_keys = ['lbl1', 'lbl2', 'level', 'nestpath']
    # CHECK: types
    try:
        assert type(output1) == type(output2), 'types are not equal'
    except AssertionError as ex:
        print(type(output1))
        print(type(output2))
        ut.printex(ex, 'FAILED TYPE CHECKS',
                   keys=common_keys + [(type, 'output1'), (type, 'output2'), ])
        raise
    # CHECK: length
    if hasattr(output1, '__len__'):
        try:
            assert len(output1) == len(output2), 'lens are not equal'
        except AssertionError as ex:
            keys = common_keys + [(len, 'output1'), (len, 'output2'), ]
            ut.printex(ex, 'FAILED LEN CHECKS. ', keys=keys)
            raise
    # CHECK: ndarrays
    if isinstance(output1, np.ndarray):
        ndarray_keys = ['output1.shape', 'output2.shape']
        # CHECK: ndarray shape
        try:
            assert output1.shape == output2.shape, 'ndarrays have different shapes'
        except AssertionError as ex:
            keys = common_keys + ndarray_keys
            ut.printex(ex, 'FAILED NUMPY SHAPE CHECKS.', keys=keys)
            raise
        # CHECK: ndarray equality
        try:
            passed, error = ut.almost_eq(output1, output2, thresh, ret_error=True)
            assert np.all(passed), 'ndarrays are unequal.'
        except AssertionError as ex:
            error_stats = ut.get_stats_str(error)  # NOQA
            keys = common_keys + ndarray_keys + [
                (len, 'output1'), (len, 'output2'), ('error_stats')
            ]
            ut.printex(ex, 'FAILED NUMPY CHECKS.', keys=keys)
            raise
    # CHECK: list/tuple items
    elif isinstance(output1, (tuple, list)):
        for count, (item1, item2) in enumerate(zip(output1, output2)):
            # recursive call
            try:
                assert_output_equal(item1, item2, lbl1=lbl2, lbl2=lbl1,
                        nestpath=nestpath + [count], level=level + 1)
            except AssertionError as ex:
                ut.printex(ex, 'recursive call failed', keys=common_keys + ['item1', 'item2', 'count'])
                raise
    # CHECK: scalars
    else:
        try:
            assert output1 == output2, 'output1 != output2'
        except AssertionError as ex:
            print('nestpath= %r' % (nestpath,))
            ut.printex(ex, 'FAILED SCALAR CHECK.', keys=common_keys + ['output1', 'output2'])
            raise


def compare_implementations(func1, func2, args, show_output=False, lbl1='', lbl2=''):
    """
    tests two different implementations of the same function
    """
    print('+ --- BEGIN COMPARE IMPLEMENTATIONS ---')
    func1_name = ut.get_funcname(func1)
    func2_name = ut.get_funcname(func2)
    print('func1_name = %r' % (func1_name,))
    print('func2_name = %r' % (func2_name,))
    # test both versions
    with ut.Timer('time func1=' + func1_name) as t1:
        output1 = func1(*args)
    with ut.Timer('time func2=' + func2_name) as t2:
        output2 = func2(*args)
    print('speedup = %r' % (t1.ellapsed / t2.ellapsed))
    try:
        assert_output_equal(output1, output2, lbl1=lbl1, lbl2=lbl2)
        print('implementations are in agreement :) ')
    except AssertionError as ex:
        # prints out a nested list corresponding to nested structure
        depth_profile1 = ut.depth_profile(output1)
        depth_profile2 = ut.depth_profile(output2)
        print('depth_profile1 = ' + ut.list_str(depth_profile1))
        print('depth_profile2 = ' + ut.list_str(depth_profile2))
        ut.printex(ex, 'IMPLEMENTATIONS DO NOT AGREE', keys=[
            ('func1_name'),
            ('func2_name'), ]
        )
        raise
    print('L ___ END COMPARE IMPLEMENTATIONS ___')
    return output1

    #out_inliers_py, out_errors_py, out_mats_py = py_output
    #out_inliers_c, out_errors_c, out_mats_c = c_output
    #if show_output:
    #    print('python output:')
    #    print(out_inliers_py)
    #    print(out_errors_py)
    #    print(out_mats_py)
    #    print('c output:')
    #    print(out_inliers_c)
    #    print(out_errors_c)
    #    print(out_mats_c)
    #msg =  'c and python disagree'
    #try:
    #    assert ut.lists_eq(out_inliers_c, out_inliers_py), msg
    #except AssertionError as ex:
    #    ut.printex(ex)
    #    raise
    #try:
    #    passed, error = ut.almost_eq(out_errors_c, out_errors_py, 1E-7, ret_error=True)
    #    assert np.all(passed), msg
    #except AssertionError as ex:
    #    passed_flat = passed.ravel()
    #    error_flat = error.ravel()
    #    failed_indexes = np.where(~passed_flat)[0]
    #    failing_errors = error_flat.take(failed_indexes)
    #    print(failing_errors)
    #    ut.printex(ex)
    #    raise
    #try:
    #    assert np.all(ut.almost_eq(out_mats_c, out_mats_py, 1E-9)), msg
    #except AssertionError as ex:
    #    ut.printex(ex)
    #    raise
    #return out_inliers_c


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
    xy_thresh_sqrd    = ktool.KPTS_DTYPE(.4)
    scale_thresh_sqrd = ktool.KPTS_DTYPE(2.0)
    ori_thresh        = ktool.KPTS_DTYPE(TAU / 4.0)
    keys = 'xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh'.split(', ')
    print(ut.dict_str(ut.dict_subset(locals(), keys)))

    def report_errors():
        pass

    if ut.get_argflag('--dummy'):
        testtup = dummy.testdata_dummy_matches()
    else:
        fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
        fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
        testtup = dummy.testdata_ratio_matches(fname1, fname2)

    (kpts1, kpts2, fm_input, fs_input, rchip1, rchip2) = testtup

    # pack up call to aff hypothesis
    args = (kpts1, kpts2, fm_input, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)

    ex_list = []

    try:
        with ut.Indenter('[TEST1] '):
            inlier_tup = compare_implementations(
                sver.get_affine_inliers,
                get_affine_inliers_cpp,
                args, lbl1='py', lbl2='c')
            out_inliers, out_errors, out_mats = inlier_tup
    except AssertionError as ex:
        ex_list.append(ex)
        #raise

    try:
        with ut.Indenter('[TEST2] '):
            bestinlier_tup = compare_implementations(
                sver.get_best_affine_inliers,
                get_best_affine_inliers_cpp,
                args, show_output=True, lbl1='py', lbl2='c')
            bestinliers, besterror, bestmat = bestinlier_tup
    except AssertionError as ex:
        ex_list.append(ex)
        #raise

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
