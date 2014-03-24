'''
Keypoints are stored in the invA format by default.
Unfortunately many places in the code reference this as A instead of invA
because I was confused when I first started writing this.

to rectify this I am changing terminology.

invV - maps from ucircle onto an ellipse (perdoch.invA)
   V - maps from ellipse to ucircle      (perdoch.A)
   Z - the conic matrix                  (perdoch.E)
'''
from __future__ import print_function, division
# Python
from itertools import izip
# Science
import numpy as np
import numpy.linalg as npl
from numpy.core.umath_tests import matrix_multiply
from numpy import (array, rollaxis, sqrt, zeros, ones, diag)
# VTool
import vtool.linalg as ltool


#PYX START
"""
// These are cython style comments for maintaining python compatibility
cimport numpy as np
ctypedef np.float64_t FLOAT64
"""
#PYX MAP FLOAT_2D np.ndarray[FLOAT64, ndim=2]
#PYX MAP FLOAT_1D np.ndarray[FLOAT64, ndim=1]
#PYX END

tau = np.tau = np.pi * 2  # tauday.com
GRAVITY_THETA = np.tau / 4
KPTS_DTYPE = np.float32

XDIM = 0
YDIM = 1
SCAX_DIM = 2
SKEW_DIM = 3
SCAY_DIM = 4
ORI_DIM = 5


# --- raw keypoint components ---
def get_xys(kpts):
    # Keypoint locations in chip space
    # TODO: _xys = kpts.T[0:2]
    _xs, _ys   = kpts.T[0:2]
    return _xs, _ys


def get_invVs(kpts):
    # Keypoint shapes (oriented with the gravity vector)
    _invVs = kpts.T[2:5]
    return _invVs


def get_oris(kpts):
    # Keypoint orientations
    # (in isotropic guassian space relative to the gravity vector)
    # (in simpler words: the orientation is is taken from keypoints warped to the unit circle)
    if kpts.shape[1] == 5:
        _oris = np.zeros(len(kpts), dtype=kpts.dtype)
    elif kpts.shape[1] == 6:
        _oris = kpts.T[5]
    else:
        raise AssertionError('[ktool] Invalid kpts.shape = %r' % (kpts.shape,))
    return _oris


def get_components(kpts):
    _xs, _ys = scale_xys(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _oris = get_oris(kpts)
    return _xs, _ys, _iv11s, _iv21s, _iv22s, _oris


# --- scaled and offset keypoint components ---

def scale_xys(kpts, scale_factor=1.0, offset=(0.0, 0.0)):
    # Keypoint location modified by an offset and scale
    __xs, __ys = get_xys(kpts)
    _xs = (__xs * scale_factor) + offset[0]
    _ys = (__ys * scale_factor) + offset[1]
    return _xs, _ys


def scale_invVs(kpts, scale_factor=1.0):
    # Keypoint location modified by an offset and scale
    __iv11s, __iv21s, __iv22s = get_invVs(kpts)
    _iv11s = __iv11s * scale_factor
    _iv21s = __iv21s * scale_factor
    _iv22s = __iv22s * scale_factor
    _iv12s = np.zeros(len(_iv11s), dtype=_iv11s.dtype)
    return _iv11s, _iv12s, _iv21s, _iv22s


def scale_kpts(kpts, scale_factor=1.0, offset=(0.0, 0.0)):
    # Returns keypoint components subject to a scale and offset
    (_xs, _ys) = scale_xys(kpts, scale_factor, offset)
    (_iv11s, _iv12s,
     _iv21s, _iv22s) = scale_invVs(kpts, scale_factor)
    _oris = get_oris(kpts)
    return _xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s, _oris


# --- keypoint properties ---

def get_sqrd_scales(kpts):
    # gets average squared scale (does not take into account elliptical shape
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _scales_sqrd = _iv11s * _iv22s
    return _scales_sqrd


def get_scales(kpts):
    # Gets average scale (does not take into account elliptical shape
    _scales = sqrt(get_sqrd_scales(kpts))
    return _scales


def get_invV_mats(kpts, ashomog=False, with_trans=False, ascontiguous=False):
    # packs keypoint shapes into affine invV matrixes
    nKpts = len(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _iv12s = zeros(nKpts)
    if ashomog:
        # Use homogenous coordinates
        if with_trans:
            _iv13s, _iv23s = get_xys(kpts)
        else:
            _iv13s = zeros(nKpts)
            _iv23s = zeros(nKpts)
        _iv31s = zeros(nKpts)
        _iv32s = zeros(nKpts)
        _iv33s = ones(nKpts)
        invV_tups = ((_iv11s, _iv12s, _iv13s),
                     (_iv21s, _iv22s, _iv23s),
                     (_iv31s, _iv32s, _iv33s))
    else:
        invV_tups = ((_iv11s, _iv12s),
                     (_iv21s, _iv22s))
    invV_arrs = array(invV_tups)        # R x C x N
    invV_mats = rollaxis(invV_arrs, 2)  # N x R x C
    if ascontiguous:
        invV_mats = np.ascontiguousarray(invV_mats)
    return invV_mats


def get_V_mats(invV_mats, **kwargs):
    # invert keypoint into V format
    V_mats = [npl.inv(invV) for invV in invV_mats]
    return V_mats


def get_E_mats(V_mats):
    # transform into conic matrix Z
    # Z = (V.T).dot(V)
    Vt_mats = array(map(np.transpose, V_mats))
    Z_mats = matrix_multiply(Vt_mats, V_mats)
    return Z_mats


def get_xy_axis_extents(invV_mats=None, kpts=None):
    'gets the scales of the major and minor elliptical axis'
    if invV_mats is None:
        assert kpts is not None
        invV_mats = get_invV_mats(kpts, ashomog=False)
    if invV_mats.shape[1] == 3:
        # Take the SVD of only the shape part
        invV_mats = invV_mats[:, 0:2, 0:2]
    Us_list = [ltool.svd(invV)[0:2] for invV in invV_mats]
    def Us_axis_extent(U, s):
        # Columns of U.dot(S) are in principle scaled directions
        return sqrt(U.dot(diag(s)) ** 2).T.sum(0)
    xyexnts = array([Us_axis_extent(U, s) for U, s in Us_list])
    return xyexnts


def diag_extent_sqrd(kpts):
    xs, ys = get_xys(kpts)
    x_extent_sqrd = (xs.max() - xs.min()) ** 2
    y_extent_sqrd = (ys.max() - ys.min()) ** 2
    extent_sqrd = x_extent_sqrd + y_extent_sqrd
    return extent_sqrd


def cast_split(kpts, dtype=KPTS_DTYPE):
    'breakup keypoints into location, shape, and orientation'
    kptsT = kpts.T
    _xs   = array(kptsT[0], dtype=dtype)
    _ys   = array(kptsT[1], dtype=dtype)
    _invVs = array(kptsT[2:5], dtype=dtype)
    if kpts.shape[1] == 6:
        _oris = array(kptsT[5:6], dtype=dtype)
    else:
        _oris = zeros(len(kpts))
    return _xs, _ys, _invVs, _oris


# --- strings ---

def get_xy_strs(kpts):
    _xs, _ys   = get_xys(kpts)
    xy_strs = [('xy=(%.1f, %.1f)' % (x, y,)) for x, y, in izip(_xs, _ys)]
    return xy_strs


def get_shape_strs(kpts):
    invVs = get_invVs(kpts)
    shape_strs  = [(('[(%3.1f,  0.00),\n' +
                     ' (%3.1f, %3.1f)]') % (iv11, iv21, iv22,))
                   for iv11, iv21, iv22 in izip(*invVs)]
    return shape_strs
