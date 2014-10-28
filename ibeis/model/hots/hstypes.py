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
* Go pandas all the way
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


RVEC_TYPE = np.int8
#RVEC_TYPE = np.float16
if RVEC_TYPE == np.int8:
    # Unfortunatley int8 cannot represent NaN, maybe used a masked array
    RVEC_INFO = np.iinfo(RVEC_TYPE)
    RVEC_MAX = 128
    RVEC_MIN = -128
    # Psuedo max values is used for a quantization tricks where you pack more data
    # into a smaller space than would normally be allowed. We are able to do this
    # because values will hardly ever be close to the true max.
    RVEC_PSEUDO_MAX = RVEC_MAX * 2
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)
else:
    RVEC_INFO = np.finfo(RVEC_TYPE)
    RVEC_MAX = 1.0
    RVEC_MIN = -1.0
    RVEC_PSEUDO_MAX = RVEC_MAX
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)


# Feature Match datatype
FM_DTYPE  = INTEGER_TYPE
# Feature Score datatype
FS_DTYPE  = FLOAT_TYPE
# Feature Rank datatype
FK_DTYPE  = np.int16
