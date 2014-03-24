from __future__ import print_function, division
# Science
import cv2
import numpy as np
import numpy.linalg as npl
from numpy import (array, sin, cos)

np.tau = 2 * np.pi  # tauday.com


def svd(M):
    # V is actually Vt
    flags = cv2.SVD_FULL_UV
    S, U, Vt = cv2.SVDecomp(M, flags=flags)
    s = S.flatten()
    return U, s, Vt


def OLD_pdf_norm2d(x_, y_):
    # DEPRICATED
    import math
    x = np.array([x_, y_])
    sigma = np.eye(2)
    mu = np.array([0, 0])
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError('The covariance matrix cant be singular')
    np.tau = 2 * np.pi
    norm_const = 1.0 / ( math.pow(np.tau, float(size) / 2) * math.pow(det, 1.0 / 2))
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result


def gauss2d_pdf(x_, y_, sigma=None, mu=None):
    '''
    Input: x and y coordinate of a 2D gaussian
           sigma, mu - covariance and mean vector
    Output: The probability density at that point
    '''
    if sigma is None:
        sigma = np.eye(2)
    if mu is None:
        mu = np.array([0, 0])
    x = array([x_, y_])
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = npl.det(sigma)
        if det == 0:
            raise NameError('The covariance matrix cant be singular')
    denom1 = np.tau ** (size / 2.0)
    denom2 = np.sqrt(det)
    norm_const = 1.0 / (denom1 * denom2)
    x_mu = x - mu  # deviation from mean
    invSigma = npl.inv(sigma)  # inverse covariance
    exponent = -0.5 * (x_mu.dot(invSigma).dot(x_mu.T))
    result = norm_const * np.exp(exponent)
    return result


def rotation_mat(radians):
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = array(((cos_, -sin_, 0),
               (sin_,  cos_, 0),
               (   0,      0, 1)))
    return R


def rotation2x2(theta):
    sin_ = sin(theta)
    cos_ = cos(theta)
    rot_ = array([[cos_, -sin_],
                  [sin_, cos_]])
    return rot_


def rotation3x3_around(theta, x, y):
    sin_ = sin(theta)
    cos_ = cos(theta)
    tr1_ = array([[1, 0, -x],
                  [0, 1, -y],
                  [0, 0, 1]])
    rot_ = array([[cos_, -sin_, 0],
                  [sin_, cos_,  0],
                  [   0,    0,  1]])
    tr2_ = array([[1, 0, x],
                  [0, 1, y],
                  [0, 0, 1]])
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot


def translation_mat(x, y):
    T = array([[1, 0,  x],
               [0, 1,  y],
               [0, 0,  1]])
    return T


def scale_mat(ss):
    S = array([[ss, 0, 0],
               [0, ss, 0],
               [0,  0, 1]])
    return S


# Ensure that a feature doesn't have multiple assignments
# --------------------------------
# Linear algebra functions on lower triangular matrices


#PYX DEFINE
def det_ltri(ltri):
    #cdef det_ltri(FLOAT_2D ltri):
    'Lower triangular determinant'
    #PYX CDEF FLOAT_1D det
    det = ltri[0] * ltri[2]
    return det


#PYX DEFINE
def inv_ltri(ltri, det):
    #cdef inv_ltri(FLOAT_2D ltri, FLOAT_1D det):
    'Lower triangular inverse'
    # PYX CDEF FLOAT_2D inv_ltri
    inv_ltri = array((ltri[2], -ltri[1], ltri[0]), dtype=ltri.dtype) / det
    return inv_ltri


#PYX BEGIN
def dot_ltri(ltri1, ltri2):
    #cdef dot_ltri(FLOAT_2D ltri1, FLOAT_2D ltri2):
    'Lower triangular dot product'
    # PYX FLOAT_1D m11, m21, m22
    # PYX FLOAT_1D n11, n21, n22
    # PYX FLOAT_1D o11, o21, o22
    # PYX FLOAT_2D ltri3
    # use m, n, and o as temporary matrixes
    m11, m21, m22 = ltri1
    n11, n21, n22 = ltri2
    o11 = (m11 * n11)
    o21 = (m21 * n11) + (m22 * n21)
    o22 = (m22 * n22)
    ltri3 = array((o11, o21, o22), dtype=ltri1.dtype)
    return ltri3
# PYX END CDEF
