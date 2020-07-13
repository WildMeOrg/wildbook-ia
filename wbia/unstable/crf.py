# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import utool as ut
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)


def crftest():
    """
    pip install pyqpbo
    pip install pystruct

    http://taku910.github.io/crfpp/#install

    cd ~/tmp
    #wget https://drive.google.com/folderview?id=0B4y35FiV1wh7fngteFhHQUN2Y1B5eUJBNHZUemJYQV9VWlBUb3JlX0xBdWVZTWtSbVBneU0&usp=drive_web#list
    7z x CRF++-0.58.tar.gz
    7z x CRF++-0.58.tar
    cd CRF++-0.58
    chmod +x configure
    ./configure
    make

    """
    import pystruct
    import pystruct.models

    inference_method_options = ['lp', 'max-product']
    inference_method = inference_method_options[1]

    # graph = pystruct.models.GraphCRF(
    #    n_states=None,
    #    n_features=None,
    #    inference_method=inference_method,
    #    class_weight=None,
    #    directed=False,
    # )

    num_annots = 5
    num_names = num_annots

    aids = np.arange(5)
    rng = np.random.RandomState(0)
    hidden_nids = rng.randint(0, num_names, num_annots)
    unique_nids, groupxs = ut.group_indices(hidden_nids)

    # Indicator vector indicating the name
    node_features = np.zeros((num_annots, num_names))
    node_features[(aids, hidden_nids)] = 1

    toy_params = {True: {'mu': 1.0, 'sigma': 2.2}, False: {'mu': 7.0, 'sigma': 0.9}}
    if False:
        import vtool as vt
        import wbia.plottool as pt

        pt.ensureqt()
        xdata = np.linspace(0, 100, 1000)
        tp_pdf = vt.gauss_func1d(xdata, **toy_params[True])
        fp_pdf = vt.gauss_func1d(xdata, **toy_params[False])
        pt.plot_probabilities([tp_pdf, fp_pdf], ['TP', 'TF'], xdata=xdata)

    def metric(aidx1, aidx2, hidden_nids=hidden_nids, toy_params=toy_params):
        if aidx1 == aidx2:
            return 0
        rng = np.random.RandomState(int(aidx1 + aidx2))
        same = hidden_nids[int(aidx1)] == hidden_nids[int(aidx2)]
        mu, sigma = ut.dict_take(toy_params[same], ['mu', 'sigma'])
        return np.clip(rng.normal(mu, sigma), 0, np.inf)

    pairwise_aidxs = list(ut.iprod(range(num_annots), range(num_annots)))
    pairwise_labels = np.array(  # NOQA
        [hidden_nids[a1] == hidden_nids[a2] for a1, a2 in pairwise_aidxs]
    )
    pairwise_scores = np.array([metric(*zz) for zz in pairwise_aidxs])
    pairwise_scores_mat = pairwise_scores.reshape(num_annots, num_annots)  # NOQA

    graph = pystruct.models.EdgeFeatureGraphCRF(  # NOQA
        n_states=num_annots,
        n_features=num_names,
        n_edge_features=1,
        inference_method=inference_method,
    )

    import opengm

    numVar = 10
    unaries = np.ones([numVar, 3], dtype=opengm.value_type)
    gm = opengm.gm(np.ones(numVar, dtype=opengm.label_type) * 3)
    unary_fids = gm.addFunctions(unaries)
    gm.addFactors(unary_fids, np.arange(numVar))
    infParam = opengm.InfParam(workflow=ut.ensure_ascii('(IC)(TTC-I,CC-I)'),)
    inf = opengm.inference.Multicut(gm, parameter=infParam)
    visitor = inf.verboseVisitor(printNth=1, multiline=False)
    inf.infer(visitor)
    arg = inf.arg()

    # gridVariableIndices = opengm.secondOrderGridVis(img.shape[0], img.shape[1])
    # fid = gm.addFunction(regularizer)
    # gm.addFactors(fid, gridVariableIndices)
    # regularizer = opengm.pottsFunction([3, 3], 0.0, beta)
    # gridVariableIndices = opengm.secondOrderGridVis(img.shape[0], img.shape[1])
    # fid = gm.addFunction(regularizer)
    # gm.addFactors(fid, gridVariableIndices)

    unaries = np.random.rand(10, 10, 2)
    potts = opengm.PottsFunction([2, 2], 0.0, 0.4)
    gm = opengm.grid2d2Order(unaries=unaries, regularizer=potts)

    inf = opengm.inference.GraphCut(gm)
    inf.infer()
    arg = inf.arg()  # NOQA


def chain_crf():
    from pystruct.datasets import load_letters

    letters = load_letters()
    X, y, folds = ut.take(letters, ['data', 'labels', 'folds'])
    X, y = np.array(X), np.array(y)
    X_train, X_test = X[folds == 1], X[folds != 1]  # NOQA
    y_train, y_test = y[folds == 1], y[folds != 1]  # NOQA
    len(X_train)
