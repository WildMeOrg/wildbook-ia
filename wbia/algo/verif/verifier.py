# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import utool as ut
from wbia.algo.verif import pairfeat
from wbia.algo.verif import sklearn_utils
import vtool as vt

# import itertools as it
# from os.path import join
print, rrr, profile = ut.inject2(__name__)


@ut.reloadable_class
class BaseVerifier(ut.NiceRepr):
    def __nice__(verif):
        return '.'.join([verif.metadata['task_key'], verif.metadata['clf_key']])

    def predict_proba_df(verif, edges):
        raise NotImplementedError('abstract')

    def fit(verif, edges):
        """
        The vsone.OneVsOneProblem currently handles fitting a model based on
        edges. The actual fit call is in clf_helpers.py
        """
        raise NotImplementedError('Need to use OneVsOneProblem to do this')

    def predict(verif, edges, method='argmax', encoded=False):
        probs = verif.predict_proba_df(edges)
        target_names = verif.class_names
        pred_enc = sklearn_utils.predict_from_probs(
            probs, method=method, target_names=target_names
        )
        if encoded:
            pred = pred_enc
        else:
            pred = pred_enc.apply(verif.class_names.__getitem__)
        return pred

    def easiness(verif, edges, real):
        """
        Gets the probability of the class each edge is labeled as.  Indicates
        how easy it is to classify this example.
        """
        probs = verif.predict_proba_df(edges)
        target_names = probs.columns.tolist()
        real_enc = np.array([target_names.index(r) for r in real])
        easiness = np.array(ut.ziptake(probs.values, real_enc))
        # easiness = pd.Series(easiness, index=probs.index)
        return easiness


@ut.reloadable_class
class Verifier(BaseVerifier):
    """
    Notes:
        deploy_info should be a dict with the following keys:
            clf: sklearn classifier
            metadata: another dict with key:
                class_names - classes that clf predicts
                task_key - str
                clf_key - str
                data_info - tuple of (feat_extract_config, feat_dims)  # TODO: make feat dims part of feat_extract_config defaulted to None
                data_info - tuple of (feat_extract_config, feat_dims)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.vsone import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
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

            feat_extract_config = feat_extract_config.copy()
            feat_extract_config['feat_dims'] = feat_dims

            verif.extr = pairfeat.PairwiseFeatureExtractor(
                ibs, config=feat_extract_config
            )

    def predict_proba_df(verif, edges):
        # TODO: if multiple verifiers have the same feature extractor we should
        # be able to cache it before we run the verification algo.
        # (we used to do this)
        X_df = verif.extr.transform(edges)
        probs_df = sklearn_utils.predict_proba_df(verif.clf, X_df, verif.class_names)
        return probs_df
        # prev_data_info = None
        # task_keys = list(infr.verifiers.keys())
        # task_probs = {}
        # for task_key in task_keys:
        #     deploy_info = infr.verifiers[task_key]
        #     data_info = deploy_info['metadata']['data_info']
        #     class_names = deploy_info['metadata']['class_names']
        #     clf = deploy_info['clf']
        #     if prev_data_info != data_info:
        #         X_df = infr._cached_pairwise_features(edges, data_info)
        #         prev_data_info = data_info
        #     probs_df = sklearn_utils.predict_proba_df(clf, X_df, class_names)
        # task_probs[task_key] = probs_df


@ut.reloadable_class
class IntraVerifier(BaseVerifier):
    """
    Predicts cross-validated intra-training sample probs.

    Note:
        Requires the original OneVsOneProblem object.
        This classifier is for intra-dataset evaulation and is not meant to be
        pushlished for use on external datasets.
    """

    def __init__(verif, pblm, task_key, clf_key, data_key):
        verif.pblm = pblm
        verif.task_key = task_key
        verif.clf_key = clf_key
        verif.data_key = data_key

        verif.metadata = {
            'task_key': task_key,
            'clf_key': clf_key,
        }

        # Make an ensemble of the evaluation classifiers
        from wbia.algo.verif import deploy

        deployer = deploy.Deployer(pblm=verif.pblm)
        verif.ensemble = deployer._make_ensemble_verifier(
            verif.task_key, verif.clf_key, verif.data_key
        )

        verif.class_names = verif.ensemble.class_names

    def predict_proba_df(verif, want_edges):
        """
        Predicts task probabilities in one of two ways:
            (1) if the edge was in the training set then its cross-validated
                probability is returned.
            (2) if the edge was not in the training set, then the average
                prediction over all cross validated classifiers are used.
        """
        clf_key = verif.clf_key
        task_key = verif.task_key
        data_key = verif.data_key

        pblm = verif.pblm

        # Load pre-predicted probabilities for intra-training set edges
        res = pblm.task_combo_res[task_key][clf_key][data_key]

        # Normalize and align combined result sample edges
        train_uv = np.array(res.probs_df.index.tolist())
        assert np.all(
            train_uv.T[0] < train_uv.T[1]
        ), 'edges must be in lower triangular form'
        assert len(vt.unique_row_indexes(train_uv)) == len(
            train_uv
        ), 'edges must be unique'
        assert sorted(ut.emap(tuple, train_uv.tolist())) == sorted(
            ut.emap(tuple, pblm.samples.aid_pairs.tolist())
        )
        want_uv = np.array(want_edges)

        # Determine which edges need/have probabilities
        want_uv_, train_uv_ = vt.structure_rows(want_uv, train_uv)
        unordered_have_uv_ = np.intersect1d(want_uv_, train_uv_)
        need_uv_ = np.setdiff1d(want_uv_, unordered_have_uv_)
        flags = vt.flag_intersection(train_uv_, unordered_have_uv_)
        # Re-order have_edges to agree with test_idx
        have_uv_ = train_uv_[flags]
        need_uv, have_uv = vt.unstructure_rows(need_uv_, have_uv_)

        # Convert to tuples for pandas lookup. bleh...
        have_edges = ut.emap(tuple, have_uv.tolist())
        need_edges = ut.emap(tuple, need_uv.tolist())
        want_edges = ut.emap(tuple, want_uv.tolist())
        assert set(have_edges) & set(need_edges) == set([])
        assert set(have_edges) | set(need_edges) == set(want_edges)

        # Predict on unseen edges using an ensemble of evaluation classifiers
        print('Predicting %s probabilities' % (task_key,))
        eclf_probs = verif.ensemble.predict_proba_df(need_edges)

        # Combine probabilities --- get probabilites for each sample
        # edges = have_edges + need_edges
        have_probs = res.probs_df.loc[have_edges]
        assert (
            have_probs.index.intersection(eclf_probs.index).size == 0
        ), 'training (have) data was not disjoint from new (want) data '

        probs = pd.concat([have_probs, eclf_probs])
        return probs


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.verif.verifier
        python -m wbia.algo.verif.verifier --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
