# -*- coding: utf-8 -*-
"""
hstypes
Todo:
* SIFT: Root_SIFT -> L2 normalized -> Centering.
# http://hal.archives-ouvertes.fr/docs/00/84/07/21/PDF/RR-8325.pdf
The devil is in the deatails
http://www.robots.ox.ac.uk/~vilem/bmvc2011.pdf
This says dont clip, do rootsift instead
# http://hal.archives-ouvertes.fr/docs/00/68/81/69/PDF/hal_v1.pdf
* Quantization of residual vectors
* Burstiness normalization for N-SMK
* Implemented A-SMK
* Incorporate Spatial Verification
* Implement correct cfgstrs based on algorithm input
for cached computations.
* Color by word
* Profile on hyrule
* Train vocab on paris
* Remove self matches.
* New SIFT parameters for pyhesaff (root, powerlaw, meanwhatever, output_dtype)


TODO:
    This needs to be less constant when using non-sift descriptors

Issues:
* 10GB are in use when performing query on Oxford 5K
* errors when there is a word without any database vectors.
currently a weight of zero is hacked in
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


# INTEGER_TYPE = np.int32
# INDEX_TYPE = np.int32
# INDEX_TYPE = np.int64
# The index type should be the native sytem int, otherwise np.take will fail
# due to the safe constraint.
INDEX_TYPE = np.int_

# INTEGER_TYPE = np.int64
# INTEGER_TYPE = np.int32
INTEGER_TYPE = np.int64

# FLOAT_TYPE = np.float64
FLOAT_TYPE = np.float64
# FLOAT_TYPE = np.float32

VEC_DIM = 128

VEC_TYPE = np.uint8
VEC_IINFO = np.iinfo(VEC_TYPE)
VEC_MAX = VEC_IINFO.max
VEC_MIN = VEC_IINFO.min
# Psuedo max values come from SIFT descriptors implementation
# Each component has a theoretical maximum of 512
VEC_PSEUDO_MAX = 512
# unit sphere points can only be twice the maximum descriptor magnitude away
# from each other. The pseudo max is 512, so 1024 is the upper bound
# FURTHERMORE SIFT Descriptors are constrained to be in the upper right quadrent
# which means any two vectors with one full component and zeros elsewhere are
# maximally distant. VEC_PSEUDO_MAX_DISTANCE = np.sqrt(2) * VEC_PSEUDO_MAX
if VEC_MIN == 0:
    # SIFT distances can be on only on one quadrent of unit sphere
    # hense the np.sqrt(2) coefficient on the component maximum
    # Otherwise it would be 2.
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * np.sqrt(2.0)
    VEC_PSEUDO_MAX_DISTANCE_SQRD = 2.0 * (512.0 ** 2.0)
elif VEC_MIN < 0:
    # Can be on whole unit sphere
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * 2
else:
    raise AssertionError('impossible state')

PSEUDO_UINT8_MAX_SQRD = float(VEC_PSEUDO_MAX) ** 2


"""
SeeAlso:
    vt.distance.understanding_pseudomax_props
"""


RVEC_TYPE = np.int8
# RVEC_TYPE = np.float16
if RVEC_TYPE == np.int8:
    # Unfortunatley int8 cannot represent NaN, maybe used a masked array
    RVEC_INFO = np.iinfo(RVEC_TYPE)
    RVEC_MAX = 128
    RVEC_MIN = -128
    # Psuedo max values is used for a quantization trick where you pack more data
    # into a smaller space than would normally be allowed. We are able to do this
    # because values will hardly ever be close to the true max.
    RVEC_PSEUDO_MAX = RVEC_MAX * 2
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)
elif RVEC_TYPE == np.float16:
    RVEC_INFO = np.finfo(RVEC_TYPE)
    RVEC_MAX = 1.0
    RVEC_MIN = -1.0
    RVEC_PSEUDO_MAX = RVEC_MAX
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)
else:
    raise AssertionError('impossible RVEC_TYPE')


# Feature Match datatype
FM_DTYPE = INTEGER_TYPE
# Feature Score datatype
FS_DTYPE = FLOAT_TYPE
# Feature Rank datatype
# FK_DTYPE  = np.int16
FK_DTYPE = np.int8


class FiltKeys(object):
    DISTINCTIVENESS = 'distinctiveness'
    FG = 'fg'
    RATIO = 'ratio'
    DIST = 'dist'
    BARL2 = 'bar_l2'
    LNBNN = 'lnbnn'
    HOMOGERR = 'homogerr'


# Denote which scores should be  used as weights
# the others are used as scores
WEIGHT_FILTERS = [FiltKeys.FG, FiltKeys.DISTINCTIVENESS, FiltKeys.HOMOGERR]


# Replace old cmtup_old with ducktype
# Keep this turned off for now until we actually start using it


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.hstypes
        python -m wbia.algo.hots.hstypes --allexamples
        python -m wbia.algo.hots.hstypes --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
