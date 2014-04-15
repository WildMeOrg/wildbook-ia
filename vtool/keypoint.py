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
from vtool import linalg as ltool
from vtool import chip as ctool
from vtool import trig
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[kpts]', DEBUG=False)


np.tau = 2 * np.pi  # tauday.com


#PYX START
"""
// These are cython style comments used to maintaining python compatibility
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


def get_dummy_kpts(num):
    """ Some testing data """
    kpts = array([[0, 0, 5.21657705, -5.11095951, 24.1498699, 0],
                  [0, 0, 2.35508823, -5.11095952, 24.1498692, 0],
                  [0, 0, 12.2165705, 12.01909553, 10.5286992, 0],
                  [0, 0, 13.3555705, 17.63429554, 14.1040992, 0],
                  [0, 0, 16.0527005, 3.407312351, 11.7353722, 0]])
    kpts = np.vstack([kpts] * num)
    return kpts


def get_dummy_invV_mats():
    invV_mats = np.array((((1.0, 0.0),
                           (0.0, 1.0),),

                          ((0.5, 0.0),
                           (0.0, 2.0),),

                          ((2.5, 0.0),
                           (0.5, 2.0),),

                          ((1.0, 0.0),
                           (0.5, 1.0),),))
    return invV_mats


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


# --- keypoint matrixes ---

def get_ori_mats(kpts):
    """ Returns keypoint orientation matrixes """
    _oris = get_oris(kpts)
    R_mats = [ltool.rotation_mat2x2(ori) for ori in _oris]
    return R_mats


def get_invV_mats2x2(kpts, with_ori=False):
    """ Returns keypoint shape matrixes
        (default orientation is down)
    """
    nKpts = len(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _zeros = zeros(nKpts)
    invV_arrs = array(((_iv11s, _zeros),
                       (_iv21s, _iv22s)))  # R x C x N
    invV_mats = rollaxis(invV_arrs, 2)     # N x R x C
    if with_ori:
        # You must apply rotations before you apply shape
        # This is because we are dealing with \emph{inv}(V).
        # numpy operates with data on the right (operate right-to-left)
        R_mats  = get_ori_mats(kpts)
        invV_mats = matrix_multiply(invV_mats, R_mats)
    return invV_mats


def get_invV_mats(kpts, with_trans=False, with_ori=False, ashomog=False, ascontiguous=False):
    """ packs keypoint shapes into affine invV matrixes
        (default is just the 2x2 shape. But translation, orientation,
        homogonous, and contiguous flags can be set.)
    """
    nKpts = len(kpts)
    invV_mats = get_invV_mats2x2(kpts, with_ori=with_ori)
    if with_trans or ashomog:
        _iv11s = invV_mats[:, 0, 0]
        _iv12s = invV_mats[:, 0, 1]
        _iv21s = invV_mats[:, 1, 0]
        _iv22s = invV_mats[:, 1, 1]
        # Use homogenous coordinates
        _zeros = zeros(nKpts)
        _ones = ones(nKpts)
        if with_trans:
            _iv13s, _iv23s = get_xys(kpts)
        else:
            _iv13s = _iv23s = _zeros
        invV_tups = ((_iv11s, _iv12s, _iv13s),
                     (_iv21s, _iv22s, _iv23s),
                     (_zeros, _zeros,  _ones))
        invV_arrs = array(invV_tups)        # R x C x N
        invV_mats = rollaxis(invV_arrs, 2)  # N x R x C
    if ascontiguous:
        invV_mats = np.ascontiguousarray(invV_mats)
    return invV_mats

# --- scaled and offset keypoint components ---


def transform_kpts_to_imgspace(kpts, bbox, bbox_theta, chipsz):
    """ Transforms keypoints so they are plotable in imagespace
        kpts   - xyacdo keypoints
        bbox   - chip bounding boxes in image space
        theta  - chip rotationsinvC
        chipsz - chip extent (in keypoint / chip space)
    """
    # Get keypoints in matrix format
    invV_mats = get_invV_mats(kpts, with_trans=True, with_ori=True)
    # Get chip to imagespace transform
    invC = ctool._get_chip_to_image_transform(bbox, chipsz, bbox_theta)
    # Apply transform to keypoints
    invCinvV_mats = matrix_multiply(invC, invV_mats)
    invCinvV_mats_ = np.array([invC.dot(invV) for invV in invV_mats])
    is_matmult_ok = np.all(invCinvV_mats == invCinvV_mats_)
    print('[trans_kpts_to_imagspace]: is_matmult_ok = %r' % is_matmult_ok)
    assert is_matmult_ok
    # Flatten back into keypoint format
    imgkpts = flatten_invV_mats_to_kpts(invCinvV_mats)
    return imgkpts

#---------------------
# invV_mats functions
#---------------------


def get_invVR_mats_shape(invVR_mats):
    """ Extracts keypoint shape components """
    _iv11s = invVR_mats[:, 0, 0]
    _iv12s = invVR_mats[:, 0, 1]
    _iv21s = invVR_mats[:, 1, 0]
    _iv22s = invVR_mats[:, 1, 1]
    return (_iv11s, _iv12s, _iv21s, _iv22s)


def get_invVR_mats_xys(invVR_mats):
    """ extracts xys from matrix encoding """
    _xys = invVR_mats[:, 0, 0:2]
    return _xys


def get_invVR_mats_oris(invVR_mats):
    """ extracts orientation from matrix encoding """
    (_iv11s, _iv12s,
     _iv21s, _iv22s) = get_invVR_mats_shape(invVR_mats)
    #
    # Solve for orientations. Adjust gravity vector pointing down
    _oris = (-trig.atan2(_iv12s, _iv11s)) % np.tau
    return _oris


def rectify_invV_mats_are_up(invVR_mats):
    """
    Useful if invVR_mats is no longer lower triangular
    rotates affine shape matrixes into downward (lower triangular) position
    """
    # Get orientation encoded in the matrix
    _oris = get_invVR_mats_oris(invVR_mats)
    # Extract keypoint shape components
    (_a, _b,
     _c, _d) = get_invVR_mats_shape(invVR_mats)
    #
    # Convert to lower triangular (rectify orientation downwards)
    det_ = np.sqrt(np.abs((_a * _d) - (_b * _c)))
    b2a2 = np.sqrt((_b ** 2) + (_a ** 2))
    iv11 = b2a2 / det_
    iv21 = ((_d * _b) + (_c * _a)) / (b2a2 * det_)
    iv22 = det_ / b2a2
    # Rebuild the matrixes
    invV_mats = invVR_mats.copy()
    invV_mats[:, 0, 0] = iv11 * det_
    invV_mats[:, 0, 1] = 0
    invV_mats[:, 1, 0] = iv21 * det_
    invV_mats[:, 1, 1] = iv22 * det_
    return invV_mats, _oris


def flatten_invV_mats_to_kpts(invV_mats):
    """ flattens invV matrices into kpts format """
    invV_mats, _oris = rectify_invV_mats_are_up(invV_mats)
    _xs    = invV_mats[:, 0, 2]
    _ys    = invV_mats[:, 1, 2]
    _iv11s = invV_mats[:, 0, 0]
    _iv21s = invV_mats[:, 1, 0]
    assert np.all(invV_mats[:, 0, 1] == 0), 'expected lower triangular matrix'
    _iv22s = invV_mats[:, 1, 1]
    kpts = np.vstack((_xs, _ys, _iv11s, _iv21s, _iv22s, _oris)).T
    return kpts


def get_V_mats(kpts, **kwargs):
    invV_mats = get_invV_mats(kpts, **kwargs)
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


def get_invV_xy_axis_extents(invV_mats):
    """ gets the scales of the major and minor elliptical axis.
        from invV_mats (faster)
    """
    if invV_mats.shape[1] == 3:
        # Take the SVD of only the shape part
        invV_mats = invV_mats[:, 0:2, 0:2]
    Us_list = [ltool.svd(invV)[0:2] for invV in invV_mats]
    def Us_axis_extent(U, s):
        """ Columns of U.dot(S) are in principle scaled directions """
        return sqrt(U.dot(diag(s)) ** 2).T.sum(0)
    xyexnts = array([Us_axis_extent(U, s) for U, s in Us_list])
    return xyexnts


def get_xy_axis_extents(kpts):
    """ gets the scales of the major and minor elliptical axis
        from kpts (slower due to conversion to invV_mats)
    """
    invV_mats = get_invV_mats(kpts, ashomog=False)
    return get_invV_xy_axis_extents(invV_mats)


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


# --- strings ---

def get_xy_strs(kpts):
    """ strings debugging and output """
    _xs, _ys   = get_xys(kpts)
    xy_strs = [('xy=(%.1f, %.1f)' % (x, y,)) for x, y, in izip(_xs, _ys)]
    return xy_strs


def get_shape_strs(kpts):
    """ strings debugging and output """
    invVs = get_invVs(kpts)
    shape_strs  = [(('[(%3.1f,  0.00),\n' +
                     ' (%3.1f, %3.1f)]') % (iv11, iv21, iv22,))
                   for iv11, iv21, iv22 in izip(*invVs)]
    shape_strs = ['invV=\n' +  _str for _str in shape_strs]
    return shape_strs


def get_ori_strs(kpts):
    _oris = get_oris(kpts)
    ori_strs = ['ori=' + utool.theta_str(ori) for ori in _oris]
    return ori_strs


def get_kpts_strs(kpts):
    xy_strs = get_xy_strs(kpts)
    shape_strs = get_shape_strs(kpts)
    ori_strs = get_ori_strs(kpts)
    kpts_strs = ['\n---\n'.join(tup) for tup in izip(xy_strs, shape_strs, ori_strs)]
    return kpts_strs
