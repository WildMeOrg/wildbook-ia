'''
Keypoints are stored in the invA format by default.
Unfortunately many places in the code reference this as A instead of invA
because I was confused when I first started writing this.

to rectify this I am changing terminology.

invV - maps from ucircle onto an ellipse (perdoch.invA)
   V - maps from ellipse to ucircle      (perdoch.A)
   Z - the conic matrix                  (perdoch.E)


Data formats:
kpts = [x, y, iv11, iv21, iv22, ori]
invV = ((iv11, iv12, x),
        (iv21, iv22, y),
        (   0,    0, 1))
'''
from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
# Science
import numpy as np
import numpy.linalg as npl
from numpy.core.umath_tests import matrix_multiply
from numpy import (array, rollaxis, sqrt, zeros, ones, diag)
# VTool
from . import linalg as ltool
from . import chip as ctool
# UTool
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[kpts]', DEBUG=False)


np.tau = 2 * np.pi  # tauday.com


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
    """ Keypoint locations in chip space """
    # TODO: _xys = kpts.T[0:2]
    _xs, _ys   = kpts.T[0:2]
    return _xs, _ys


def get_invVs(kpts):
    """ Keypoint shapes (oriented with the gravity vector) """
    _invVs = kpts.T[2:5]
    return _invVs


def get_oris(kpts):
    """ Keypoint orientations
    (in isotropic guassian space relative to the gravity vector)
    (in simpler words: the orientation is is taken from keypoints warped to the unit circle)
    """
    if kpts.shape[1] == 5:
        _oris = np.zeros(len(kpts), dtype=kpts.dtype)
    elif kpts.shape[1] == 6:
        _oris = kpts.T[5]
    else:
        raise AssertionError('[ktool] Invalid kpts.shape = %r' % (kpts.shape,))
    return _oris


def get_components(kpts):
    """
    breaks up keypoints into its location, shape, and orientation components
    """
    _xs, _ys = scale_xys(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _oris = get_oris(kpts)
    return _xs, _ys, _iv11s, _iv21s, _iv22s, _oris


# --- scaled and offset keypoint components ---

def scale_xys(kpts, scale_factor=1.0, offset=(0.0, 0.0)):
    """ Keypoint location modified by an offset and scale """
    __xs, __ys = get_xys(kpts)
    _xs = (__xs * scale_factor) + offset[0]
    _ys = (__ys * scale_factor) + offset[1]
    return _xs, _ys


def scale_invVs(kpts, scale_factor=1.0):
    """ Keypoint location modified by an offset and scale """
    __iv11s, __iv21s, __iv22s = get_invVs(kpts)
    _iv11s = __iv11s * scale_factor
    _iv21s = __iv21s * scale_factor
    _iv22s = __iv22s * scale_factor
    _iv12s = np.zeros(len(_iv11s), dtype=_iv11s.dtype)
    return _iv11s, _iv12s, _iv21s, _iv22s


def scale_kpts(kpts, scale_factor=1.0, offset=(0.0, 0.0)):
    """ Returns keypoint components subject to a scale and offset """
    (_xs, _ys) = scale_xys(kpts, scale_factor, offset)
    (_iv11s, _iv12s,
     _iv21s, _iv22s) = scale_invVs(kpts, scale_factor)
    _oris = get_oris(kpts)
    return _xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s, _oris


# --- keypoint properties ---

def get_sqrd_scales(kpts):
    """ gets average squared scale (does not take into account elliptical shape """
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _scales_sqrd = _iv11s * _iv22s
    return _scales_sqrd


def get_scales(kpts):
    """  Gets average scale (does not take into account elliptical shape """
    _scales = sqrt(get_sqrd_scales(kpts))
    return _scales


def get_ori_mats(kpts):
    _oris = get_oris(kpts)
    ori_mats = [ltool.rotation_mat(ori) for ori in _oris]
    #print([ori for ori in ori_mats])
    return ori_mats


def get_invV_mats(kpts, ashomog=False, with_trans=False, with_ori=False, ascontiguous=False):
    """ packs keypoint shapes into affine invV matrixes """
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
    if with_ori:
        ori_mats = get_ori_mats(kpts)
        # FIXME: this does not produce a numpy array
        # Is there a way to dot product without list comprehension?
        # matrix_multiply(A, B) ?
        invV_mats = [invV.dot(orimat) for (invV, orimat) in izip(invV_mats, ori_mats)]
    if ascontiguous:
        invV_mats = np.ascontiguousarray(invV_mats)
    return invV_mats


def flatten_invV_mats_to_kpts(invV_mats):
    """ flattens a matrix into keypoint format """
    # TODO: Need to rectify M to point downward and have a rotation
    assert all([invV.shape == (3, 3) for invV in invV_mats]), 'need to be 3x3 matrixes'
    invV_mats, _oris = rectify_invV_is_up(invV_mats)
    _xs = invV_mats[:, 0, 2]
    _ys = invV_mats[:, 1, 2]
    _iv11s = invV_mats[:, 0, 0]
    _iv21s = invV_mats[:, 1, 0]
    _iv22s = invV_mats[:, 1, 1]
    kpts = np.vstack((_xs, _ys, _iv11s, _iv21s, _iv22s, _oris)).T
    return kpts


def transform_kpts_to_imgspace(kpts, bbox, bbox_theta, chipsz):
    """ Transforms keypoints so they are plotable in imagespace
        kpts   - xyacdo keypoints
        bbox   - chip bounding boxes in image space
        theta  - chip rotations
        chipsz - chip extent (in keypoint / chip space)
    """
    # Get keypoints in matrix format
    invV_list = get_invV_mats(kpts, ashomog=True, with_trans=True, with_ori=True)
    # Get chip to imagespace transform
    invC = ctool._get_chip_to_image_transform(bbox, chipsz, bbox_theta)
    # Apply transform to keypoints
    # FIXME: this does not produce a numpy array
    # Is there a way to dot product without list comprehension?
    invCinvV_mats = np.array([invC.dot(invV) for invV in invV_list])
    # Flatten back into keypoint format
    imgkpts = flatten_invV_mats_to_kpts(invCinvV_mats)
    return imgkpts


def get_V_mats(invV_mats, **kwargs):
    """ invert keypoint into V format """
    V_mats = [npl.inv(invV) for invV in invV_mats]
    return V_mats


def get_Z_mats(V_mats):
    """
        transform into conic matrix Z
        Z = (V.T).dot(V)
    """
    Vt_mats = array(map(np.transpose, V_mats))
    Z_mats = matrix_multiply(Vt_mats, V_mats)
    return Z_mats


def get_xy_axis_extents(invV_mats=None, kpts=None):
    """ gets the scales of the major and minor elliptical axis """
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
    """ Returns the diagonal extent of keypoint locations """
    xs, ys = get_xys(kpts)
    x_extent_sqrd = (xs.max() - xs.min()) ** 2
    y_extent_sqrd = (ys.max() - ys.min()) ** 2
    extent_sqrd = x_extent_sqrd + y_extent_sqrd
    return extent_sqrd


def cast_split(kpts, dtype=KPTS_DTYPE):
    """ breakup keypoints into location, shape, and orientation """
    kptsT = kpts.T
    _xs   = array(kptsT[0], dtype=dtype)
    _ys   = array(kptsT[1], dtype=dtype)
    _invVs = array(kptsT[2:5], dtype=dtype)
    if kpts.shape[1] == 6:
        _oris = array(kptsT[5:6], dtype=dtype)
    else:
        _oris = zeros(len(kpts))
    return _xs, _ys, _invVs, _oris


def rectify_invV_is_up(invV_mats):
    """
    rotates affine shape matrixes into downward (lower triangular) position
    """
    ##
    # Extract keypoint componetns
    a = invV_mats[:, 0, 0]
    b = invV_mats[:, 0, 1]
    c = invV_mats[:, 1, 0]
    d = invV_mats[:, 1, 1]
    # Get deterimant and whatever b2a2 is
    det_ = np.sqrt(np.abs((a * d) - (b * c)))
    b2a2 = np.sqrt((b ** 2) + (a ** 2))
    # Rectify the keypoint direction
    iv11 = b2a2 / det_
    iv21 = ((d * b) + (c * a)) / (b2a2 * det_)
    iv22 = det_ / b2a2
    # Rebuild the matrixes
    invV_mats2 = invV_mats.copy()
    invV_mats2[:, 0, 0] = iv11 * det_
    invV_mats2[:, 0, 1] = 0.0
    invV_mats2[:, 1, 0] = iv21 * det_
    invV_mats2[:, 1, 1] = iv22 * det_
    #
    _oris = np.arctan2(d, b)  # solve for keypoint orientation
    _oris -= (np.tau / 4)  # Adjust for gravity vector pointing downwards
    return invV_mats2, _oris


# --- strings ---

def get_xy_strs(kpts):
    """ strings for debugging and output """
    _xs, _ys   = get_xys(kpts)
    xy_strs = [('xy=(%.1f, %.1f)' % (x, y,)) for x, y, in izip(_xs, _ys)]
    return xy_strs


def get_shape_strs(kpts):
    """ strings for debugging and output """
    invVs = get_invVs(kpts)
    shape_strs  = [(('[(%3.1f,  0.00),\n' +
                     ' (%3.1f, %3.1f)]') % (iv11, iv21, iv22,))
                   for iv11, iv21, iv22 in izip(*invVs)]
    return shape_strs
