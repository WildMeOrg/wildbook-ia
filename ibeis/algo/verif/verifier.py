# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
import utool as ut
from ibeis.algo.verif import pairfeat
from ibeis.algo.verif import sklearn_utils
# import itertools as it
# import vtool as vt
# from os.path import join
print, rrr, profile = ut.inject2(__name__)


@ut.reloadable_class
class Verifier(ut.NiceRepr):
    """

    Example:
        >>> from ibeis.algo.verif.vsone import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> speceis = 'zebra_plains'
        >>> task_key = 'match_state'
        >>> verif = Deployer()._load_published(ibs, species, task_key)
    """

    def __init__(verif, ibs=None, deploy_info=None):
        verif.ibs = ibs
        verif.clf = None
        verif.metadata = None

        verif.class_names = None
        verif.extr = None

        if deploy_info:
            verif.clf = deploy_info['clf']
            verif.metadata = deploy_info['metadata']
            verif.class_names = verif.metadata['class_names']

            data_info = verif.metadata['data_info']
            feat_extract_config, feat_dims = data_info

            feat_extract_config = feat_extract_config
            feat_dims = feat_dims

            verif.extr = pairfeat.PairwiseFeatureExtractor(
                ibs, feat_dims=feat_dims, **feat_extract_config)

    def __nice__(verif):
        return '.'.join([verif.metadata['task_key'],
                         verif.metadata['clf_key']])

    def predict_proba_df(verif, edges):
        # TODO: if multiple verifiers have the same feature extractor we should
        # be able to cache it before we run the verification algo.
        # (we used to do this)
        X_df = verif.extr.transform(edges)
        probs_df = sklearn_utils.predict_proba_df(verif.clf, X_df,
                                                  verif.class_names)
        return probs_df
        # prev_data_info = None
        # task_keys = list(infr.classifiers.keys())
        # task_probs = {}
        # for task_key in task_keys:
        #     deploy_info = infr.classifiers[task_key]
        #     data_info = deploy_info['metadata']['data_info']
        #     class_names = deploy_info['metadata']['class_names']
        #     clf = deploy_info['clf']
        #     if prev_data_info != data_info:
        #         X_df = infr._cached_pairwise_features(edges, data_info)
        #         prev_data_info = data_info
        #     probs_df = sklearn_utils.predict_proba_df(clf, X_df, class_names)
        # task_probs[task_key] = probs_df

    # def fit(verif, edges):
    #     pass
