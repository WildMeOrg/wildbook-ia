from __future__ import absolute_import, division, print_function
from numpy import (array)
from vtool import keypoint as ktool
from vtool import image as gtool
import vtool.math as mtool
import numpy as np
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[dummy]', DEBUG=False)


DEFAULT_DTYPE = ktool.KPTS_DTYPE


def get_dummy_kpts(num, dtype=DEFAULT_DTYPE):
    """ Some testing data """
    kpts = array([[0, 0, 5.21657705, -5.11095951, 24.1498699, 0],
                  [0, 0, 2.35508823, -5.11095952, 24.1498692, 0],
                  [0, 0, 12.2165705, 12.01909553, 10.5286992, 0],
                  [0, 0, 13.3555705, 17.63429554, 14.1040992, 0],
                  [0, 0, 16.0527005, 3.407312351, 11.7353722, 0]], dtype=dtype)
    kpts = np.vstack([kpts] * num)
    return kpts


def get_kpts_dummy_img(kpts, sf=1.0):
    w, h = ktool.get_kpts_bounds(kpts)
    img = gtool.dummy_img(w * sf, h * sf)
    return img


def get_dummy_invV_mats(dtype=DEFAULT_DTYPE):
    invV_mats = np.array((((1.0, 0.0),
                           (0.0, 1.0),),

                          ((0.5, 0.0),
                           (0.0, 2.0),),

                          ((2.5, 0.0),
                           (0.5, 2.0),),

                          ((1.0, 0.0),
                           (0.5, 1.0),),), dtype=np.float32)
    return invV_mats


def get_dummy_matching_kpts(dtype=DEFAULT_DTYPE):
    kpts1 = array([[  3.28846716e+02,   1.21753590e+02,   1.25637716e+01,  -1.18816503e+01,   1.00417606e+01,   1.14933074e-02],
                   [  3.01069363e+02,   1.31472857e+02,   1.29744163e+01,  -1.26813604e+01,   1.23529540e+01,   2.56660535e-01],
                   [  4.08240651e+02,   1.86241170e+02,   1.68589858e+01,  -1.33149406e+01,   1.42353870e+01,   1.20307208e-02],
                   [  3.88036559e+02,   3.76789856e+02,   1.47105962e+01,  -8.80294520e+00,   1.20673321e+01,   6.02803372e+00],
                   [  2.91987002e+02,   1.56273941e+02,   1.59259660e+01,  -7.87242786e+00,   2.02183781e+01,   1.82204839e-01],
                   [  2.76012028e+02,   3.58504941e+02,   1.55394668e+01,  -1.52683674e+00,   2.38340761e+01,   6.25445998e+00],
                   [  3.90943404e+02,   3.44728235e+02,   2.09021796e+01,  -1.29564521e+01,   5.58724982e+00,   6.08949162e+00],
                   [  3.13005196e+02,   3.63265206e+02,   9.32637456e+00,  -1.18802434e+01,   2.15849429e+01,   9.10939207e-02],
                   [  3.49344296e+02,   4.47038587e+02,   1.92400430e+01,  -8.61497903e+00,   1.94933361e+01,   5.97432787e+00],
                   [  4.12149038e+02,   3.85985588e+02,   3.25077165e+01,  -1.86535559e+01,   7.91420083e+00,   6.23798141e+00],
                   [  3.76955263e+02,   3.62479481e+02,   2.50461522e+01,  -6.14592975e+00,   1.45263025e+01,   2.53605042e-01],
                   [  3.86279299e+02,   3.98880867e+02,   3.92990884e+01,  -1.95396603e+01,   1.51900915e+01,   2.34707827e-01],
                   [  2.72267165e+02,   3.39945945e+02,   1.80826449e+01, 5.12738475e+00,   5.43937103e+01,   6.00411182e-01]],
                  dtype=dtype)
    ##
    kpts2 = array([[  3.09621904e+02,   1.01906467e+02,   1.01749649e+01,  -8.13185135e+00,   9.76679199e+00,   5.98161838e+00],
                   [  2.95794687e+02,   1.32017193e+02,   1.31069994e+01,  -1.24562360e+01,   1.12054328e+01,   2.65517495e-01],
                   [  3.78841354e+02,   2.90189530e+02,   1.65034851e+01,  -8.23787708e+00,   1.04872841e+01,   5.52809448e-01],
                   [  3.54223322e+02,   3.56999688e+02,   1.56782369e+01,  -9.34924382e+00,   1.39873022e+01,   1.82464324e-01],
                   [  3.08208701e+02,   1.10979910e+02,   1.23573375e+01,  -1.32017287e+01,   1.81341743e+01,   6.18054641e+00],
                   [  2.72120364e+02,   3.06051898e+02,   1.41604565e+01,  -6.41663265e-01,   2.40546104e+01,   9.96961592e-02],
                   [  3.99782152e+02,   3.84726911e+02,   2.54674401e+01,  -1.13622177e+01,   1.52564237e+01,   2.82467605e-01],
                   [  2.73039551e+02,   4.09726917e+02,   7.87345363e+00,  -1.11725136e+01,   2.50426481e+01,   6.34441839e-01],
                   [  3.49597555e+02,   4.07510942e+02,   2.37571881e+01,  -7.80666851e+00,   1.35347776e+01,   5.83382941e+00],
                   [  4.06788317e+02,   3.33934196e+02,   3.04510906e+01,  -1.78787314e+01,   1.28190063e+01,   1.59551101e-01],
                   [  3.55198510e+02,   3.52797004e+02,   2.31707674e+01,  -5.17600302e+00,   2.02805843e+01,   8.69594936e-02],
                   [  3.81289714e+02,   4.07277352e+02,   3.63608634e+01,  -2.03970279e+01,   1.69257319e+01,   7.12531509e-03],
                   [  2.66126484e+02,   4.21403317e+02,   2.23173801e+01,   6.71736273e+00,   5.26069906e+01,   5.96099364e+00]],
                  dtype=dtype)
    ##
    fm = array([[ 0,  0],
                [ 1,  1],
                [ 2,  2],
                [ 3,  3],
                [ 4,  4],
                [ 5,  5],
                [ 6,  6],
                [ 7,  7],
                [ 8,  8],
                [ 9,  9],
                [10, 10],
                [11, 11],
                [12, 12]], dtype=np.int32)
    return kpts1, kpts2, fm


def make_dummy_fm(nKpts):
    fx1_m = np.arange(nKpts)
    fx2_m = np.arange(nKpts)
    fm = np.vstack((fx1_m, fx2_m)).T
    return fm


def force_kpts_feasibility(kpts, xys_nonneg=False):
    # Fix locations to be above 0
    if xys_nonneg:
        kpts[:, ktool.XDIM] = np.abs(kpts[:, ktool.XDIM])
        kpts[:, ktool.XDIM] = np.abs(kpts[:, ktool.XDIM])
    # Fix shape to be pos-semidef
    kpts[:, ktool.SCAX_DIM] = np.abs(kpts[:, ktool.SCAX_DIM])
    kpts[:, ktool.SCAY_DIM] = np.abs(kpts[:, ktool.SCAY_DIM])
    # Fix oris between 0 and tau
    kpts[:, ktool.ORI_DIM] = kpts[:, ktool.ORI_DIM] % np.tau
    return kpts


def pertebed_grid_kpts(*args, **kwargs):
    grid_kpts = ktool.get_grid_kpts(*args, **kwargs)
    perterb_kwargs = dict(xy_std=(5, 5), invV_std=(3, 5, 3), ori_std=.25)
    perterb_kwargs.update(kwargs)
    return perterb_kpts(grid_kpts, **perterb_kwargs)


def perterb_kpts(kpts, xy_std=None, invV_std=None, ori_std=None, damping=None,
                 seed=None):
    """ Adds normally distributed pertibations to keypoints """
    # TODO: Move to ktool
    # Get standard deviations of pertibations
    if xy_std is None:
        xy_std   = ktool.get_xys(kpts).std(1) + mtool.eps
    if invV_std is None:
        invV_std = ktool.get_invVs(kpts).std(1) + mtool.eps
    if ori_std is None:
        ori_std  = ktool.get_oris(kpts).std() + mtool.eps
    xy_std = np.array(xy_std, dtype=ktool.KPTS_DTYPE)
    invV_std = np.array(invV_std, dtype=ktool.KPTS_DTYPE)
    if damping is not None:
        xy_std /= damping
        invV_std /= damping
        ori_std /= damping
    if seed is not None:
        np.random.seed(seed)
    # Create normally distributed pertibations
    xy_aug   = np.random.normal(0, scale=xy_std, size=(len(kpts), 2)).astype(ktool.KPTS_DTYPE)
    try:
        invV_aug = np.random.normal(0, scale=invV_std, size=(len(kpts), 3)).astype(ktool.KPTS_DTYPE)
    except ValueError as ex:
        utool.printex(ex, key_list=[(type, 'invV_std')])
        raise
    ori_aug = np.random.normal(0, scale=ori_std, size=(len(kpts), 1)).astype(ktool.KPTS_DTYPE)
    # Augment keypoints
    aug = np.hstack((xy_aug, invV_aug, ori_aug))
    kpts_ = kpts + aug
    # Ensure keypoint feasibility
    kpts_ = force_kpts_feasibility(kpts_)
    #print(utool.dict_str({key: type(val) if not isinstance(val, np.ndarray) else val.dtype for key, val in locals().items()}))
    assert kpts_.dtype == ktool.KPTS_DTYPE, 'bad cast somewhere kpts_.dtype=%r' % (kpts_.dtype)
    return kpts_
