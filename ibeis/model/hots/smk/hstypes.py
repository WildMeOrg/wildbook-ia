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

Paper Style Guidelines:
* use real code examples instead of pseudocode
(show off power of python)
* short and consice
* never cryptic

Paper outline:

abstract:
contributions:

algorithms:
lnbnn
a/smk
modification (name scoring? next level categorization)

parameters:
database size
sift threshold
vocabulary?

Databases:
pzall
gzall
oxford
paris
"""
from __future__ import absolute_import, division, print_function
import numpy as np

#FLOAT_TYPE = np.float32
#INTEGER_TYPE = np.int32
FLOAT_TYPE = np.float64
INTEGER_TYPE = np.int64
INDEX_TYPE = np.int32
VEC_TYPE = np.uint8
VEC_DIM = 128
