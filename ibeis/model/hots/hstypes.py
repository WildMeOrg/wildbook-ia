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

Issues:
* 10GB are in use when performing query on Oxford 5K
* errors when there is a word without any database vectors.
currently a weight of zero is hacked in
"""
from __future__ import absolute_import, division, print_function
import numpy as np

#INTEGER_TYPE = np.int32
INDEX_TYPE = np.int32

#INTEGER_TYPE = np.int64
INTEGER_TYPE = np.int32

#FLOAT_TYPE = np.float64
FLOAT_TYPE = np.float32

VEC_DIM = 128

VEC_TYPE = np.uint8
VEC_IINFO = np.iinfo(VEC_TYPE)
VEC_MAX = VEC_IINFO.max
VEC_MIN = VEC_IINFO.min
# Psuedo max values come from SIFT descriptors implementation
VEC_PSEUDO_MAX = 512
# unit sphere points can only be twice the maximum descriptor magnitude away
# from each other. The pseudo max is 512, so 1024 is the upper bound
# FURTHERMORE SIFT Descriptors are constrained to be in the upper right quadrent
# which means any two vectors with one full component and zeros elsewhere are
# maximally distant. VEC_PSEUDO_MAX_DISTANCE = np.sqrt(2) * VEC_PSEUDO_MAX
if VEC_MIN == 0:
    # Can be on only on one quadrent of unit sphere
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * np.sqrt(2)
elif VEC_MIN < 0:
    # Can be on whole unit sphere
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * 2
else:
    raise AssertionError('impossible state')

PSEUDO_UINT8_MAX_SQRD = float(VEC_PSEUDO_MAX) ** 2


RVEC_TYPE = np.int8
#RVEC_TYPE = np.float16
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
FM_DTYPE  = INTEGER_TYPE
# Feature Score datatype
FS_DTYPE  = FLOAT_TYPE
# Feature Rank datatype
FK_DTYPE  = np.int16


class FiltKeys(object):
    DISTINCTIVENESS = 'distinctiveness'
    FG = 'fg'
    RATIO = 'ratio'
    DIST = 'dist'
    LNBNN = 'lnbnn'
    DUPVOTE = 'dupvote'
    HOMOGERR = 'homogerr'

# Denote which scores should be  used as weights
# the others are used as scores
WEIGHT_FILTERS = [FiltKeys.FG, FiltKeys.DISTINCTIVENESS, FiltKeys.HOMOGERR]
