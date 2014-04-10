from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import izip

DIST_LIST = ['L1', 'L2']


def compute_distances(hist1, hist2, dist_list=DIST_LIST):
    dtype_ = np.float64
    hist1 = np.array(hist1, dtype=dtype_)
    hist2 = np.array(hist2, dtype=dtype_)
    return {type_: globals()[type_](hist1, hist2) for type_ in dist_list}


def L1(hist1, hist2):
    return (np.abs(hist1 - hist2)).sum(-1)


def L2_sqrd(hist1, hist2):
    return (np.abs(hist1 - hist2) ** 2).sum(-1)


def L2(hist1, hist2):
    return np.sqrt((np.abs(hist1 - hist2) ** 2).sum(-1))


def hist_isect(hist1, hist2):
    numer = (np.dstack([hist1, hist2])).min(-1).sum(-1)
    denom = hist2.sum(-1)
    hisect_dist = 1 - (numer / denom)
    if len(hisect_dist) == 1:
        hisect_dist = hisect_dist[0]
    return hisect_dist


def emd(hist1, hist2):
    """
    earth mover's distance by robjects(lpSovle::lp.transport)
    import numpy as np
    hist1 = np.random.rand(128)
    hist2 = np.random.rand(128)
    require: lpsolve55-5.5.0.9.win32-py2.7.exe
    https://github.com/andreasjansson/python-emd
    http://stackoverflow.com/questions/15706339/how-to-compute-emd-for-2-numpy-arrays-i-e-histogram-using-opencv
    http://www.cs.huji.ac.il/~ofirpele/FastEMD/code/
    http://www.cs.huji.ac.il/~ofirpele/publications/ECCV2008.pdf
    """
    try:
        from cv2 import cv
    except ImportError as ex:
        print(repr(ex))
        print('Cannot import cv. Is opencv 2.4.9?')
        return -1

    # Stack weights into the first column
    def add_weight(hist):
        weights = np.ones(len(hist))
        stacked = np.ascontiguousarray(np.vstack([weights, hist]).T)
        return stacked

    def convertCV32(stacked):
        hist64 = cv.fromarray(stacked)
        hist32 = cv.CreateMat(hist64.rows, hist64.cols, cv.CV_32FC1)
        cv.Convert(hist64, hist32)
        return hist32

    def emd_(a32, b32):
        return cv.CalcEMD2(a32, b32, cv.CV_DIST_L2)

    # HACK
    if len(hist1.shape) == 1 and len(hist2.shape) == 1:
        a, b = add_weight(hist1), add_weight(hist2)
        a32, b32 = convertCV32(a), convertCV32(b)
        emd_dist = emd_(a32, b32)
        return emd_dist
    else:
        ab_list   = [(add_weight(a), add_weight(b)) for a, b in izip(hist1, hist2)]
        ab32_list = [(convertCV32(a), convertCV32(b)) for a, b in ab_list]
        emd_dists = [emd_(a32, b32) for a32, b32, in ab32_list]
        return emd_dists
