# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
#import numpy as np
import sklearn  # NOQA
#import sklearn.datasets
#import sklearn.svm
#import sklearn.metrics
#from sklearn import preprocessing
#from ibeis_cnn.models import abstract_models
#from os.path import join
(print, rrr, profile) = ut.inject2(__name__, '[classify_shark]')


def thesis_smk():
    import ibeis
    ibs = ibeis.opendb('PZ_MTEST')
    annots = ibs.annots()

    import sklearn.cross_validation
    rng = 43432
    xvalkw = dict(n_folds=4, shuffle=True, random_state=rng)
    skf = sklearn.cross_validation.StratifiedKFold(annots.nids, **xvalkw)
