#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, abspath
import numpy as np

PROPOSALS = 64
ALPHA = 0.3
C = 0.01
EPSILON = 0.2

LEARNING_RATE = 0.1
BATCH_SIZE = 128
MAXIMUM_EPOCHS = 250
OUTPUT_PATH = abspath('.')

CHIP_SIZE = (227, 227)
# CHIP_SIZE = (128, 128)
# CHIP_SIZE = (138, 138)

PRIOR_MATCHING = True
APPLY_PRIOR_CENTERING = False
APPLY_PRIOR_TRANSFORMATION = False

CACHE_DATA = False
FILTER_EMPTY_IMAGES = False
# MAXIMUM_GT_PER_IMAGE = 6
MAXIMUM_GT_PER_IMAGE = None
APPLY_DATA_AUGMENTATION = False
APPLY_NORMALIZATION = False

MIRROR_LOSS_GRADIENT_WITH_ASSIGNMENT_ERROR = True

if PRIOR_MATCHING:
    if APPLY_PRIOR_TRANSFORMATION:
        prior_filename = 'priors.transformed.npy'
    else:
        prior_filename = 'priors.npy'
    prior_filepath = join('extracted', prior_filename)
    PRIORS = np.load(prior_filepath)
    assert PRIORS.shape[0] == PROPOSALS

    # Reshape centroids
    PRIORS = PRIORS.astype(np.float32)
    PRIORS_ = PRIORS.copy()
    PRIORS = PRIORS.reshape((1, -1))
else:
    APPLY_PRIOR_CENTERING = False
    APPLY_PRIOR_TRANSFORMATION = False
    PRIORS = None
    PRIORS_ = None
