# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[ibs]')


"""
Buidl probability curves of
* lnbnn, ratio, logratio, l2dist, etc for tp and tn of different db sizes

Rescore based on progressively increasing thresholds

"""


def chip_tester():
    import plottool as pt
    pt.ensure_pylab_qt4()

    # from ibeis.core_annots import *  # NOQA
    import ibeis
    defaultdb = 'GZ_ALL'
    ibs = ibeis.opendb(defaultdb=defaultdb)
    aid_list = ibs.get_valid_aids()
    depc = ibs.depc_annot

    chips_orig = depc.get_property('chips', aid_list, 'img', config={})

    chips_aeq = depc.get_property('chips', aid_list, 'img', config={'adapteq': True})
    chips_heq = depc.get_property('chips', aid_list, 'img', config={'histeq': True})


    import pyhesaff
    nkpts_list = np.array(list(ut.generate(pyhesaff.detect_num_kpts_in_image, chips_orig, force_serial=ibs.force_serial)))
    nkpts_list = np.array(nkpts_list)

    nfeats_orig = np.array(ibs.depc_annot.get('feat', aid_list, 'num_feats'))
    nfeats_hteq = np.array(ibs.depc_annot.get('feat', aid_list, 'num_feats', config={'histeq': True}))
    nfeats_ateq = np.array(ibs.depc_annot.get('feat', aid_list, 'num_feats', config={'adapteq': True}))

    sortx = np.array(nfeats_orig).argsort()
    sortx = np.array(nfeats_hteq).argsort()
    sortx = np.array(nfeats_ateq).argsort()

    aids = ut.take(aid_list, sortx)
    chips = chips_orig
    chips_bad = ut.take(chips, sortx)
    chips_good = ut.take(chips, sortx[::-1])

    import ibeis.viz.interact.interact_chip
    ibeis.viz.interact.interact_chip.interact_multichips(ibs, aids)

    iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips_bad, nPerPage=15)
    iteract_obj.start()

    iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips_good, nPerPage=15)
    iteract_obj.start()

    x = sklearn.cluster.KMeans(2)
    x.fit(np.nan_to_num(measures))

    import vtool.quality_classifier
    from vtool.quality_classifier import contrast_measures
    chips128 = depc.get_property('chips', aid_list, 'img', config={'dim_size': 256})
    gray_chips = [vt.convert_colorspace(x, 'GRAY') for x in ut.ProgIter(chips128)]
    measures = list(ut.generate(contrast_measures, gray_chips, force_serial=ibs.force_serial))
    measures = np.array(measures)
    measures = np.nan_to_num(measures)
    y = measures.T[3]
    sortx = y.argsort()
    ys = y.take(sortx)


    pca = sklearn.decomposition.PCA(1)
    pca.fit(measures)
    pca_measure = pca.transform(measures)
    nfeats_white = (nfeats_orig - nfeats_orig.mean()) / nfeats_orig.std()
    pca_white = (pca_measure - pca_measure.mean()) / pca_measure.std()
    sortx = nfeats_white.argsort()
    pt.plt.plot(pca_white[sortx], 'x')
    pt.plt.plot(nfeats_white[sortx], '.')


    pyhesaff.detect_feats_in_image

    svc = sklearn.svm.LinearSVC()
    svc.fit(measures, nfeats_orig > 500)
    svc.predict(measures) == (nfeats_orig > 500)

    svr = sklearn.svm.LinearSVR()
    svr.fit(measures, nfeats_orig)
    svr.predict(measures)

    depc['feat'].get_config_history(z1)
    depc['feat'].get_config_history(z2)
    pt.plt.plot(nfeats_hteq[sortx], 'x')
    pt.plt.plot(nfeats_orig[sortx], '.')
    pt.plt.plot(nfeats_ateq[sortx], 'o')

    z1 = ibs.depc_annot.get_rowids('feat', aid_list, config={'histeq': True})
    z2 = ibs.depc_annot.get_rowids('feat', aid_list)
    assert len(set(z1).intersection(z2)) == 0



def test_sharpness():
    import ibeis
    defaltdb = 'seaturtles'
    a = ['default']
    ibs = ibeis.opendb(defaultdb=defaltdb)
    ibs, qaids, daids = ibeis.testdata_expanded_aids(ibs=ibs, a=a)
    from vtool import quality_classifier

    contrast_list = [quality_classifier.compute_average_contrast(chip) for chip in ibs.get_annot_chips(qaids)]
    sortx = ut.argsort(contrast_list)[::-1]
    sharpest_qaids = ut.take(qaids, sortx)

    aid = sharpest_qaids[0]
    ut.ensure_pylab_qt4()
    from ibeis import viz
    import plottool as pt
    for aid in ut.InteractiveIter(sharpest_qaids):
        viz.show_chip(ibs, aid, annot=False, nokpts=True)
        pt.update()


def testdata_workflow(defaltdb='PZ_MTEST', t=['default'], a=['defualt']):
    # qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', a='default', t='default')
    # ibs = qreq_.ibs
    # qaids = qreq_.qaids
    # daids = qreq_.daids
    # from ibeis.init import filter_annots
    # from ibeis.expt import annotation_configs
    # import copy
    # acfg = copy.deepcopy(annotation_configs.default)
    # qaids, daids = filter_annots.expand_acfgs(ibs, acfg,
    #                                           initial_aids=None,
    #                                           use_cache=False,
    #                                           verbose=True)
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaltdb)
    ibs, qaids, daids = ibeis.testdata_expanded_aids(ibs=ibs, a=a)
    pipecfg = ibeis.testdata_pipecfg(t=t)
    qreq_ = ibs.new_query_request(qaids, daids, pipecfg)
    cm_list = qreq_.qreq_.execute()
    return qreq_, cm_list


def find_most_disitnctive_keypoints():
    r"""
    CommandLine:
        python -m ibeis.workflow --exec-find_most_disitnctive_keypoints --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.workspace import *  # NOQA
        >>> find_most_disitnctive_keypoints()
    """
    import ibeis
    cm_list, qreq_ = ibeis.testdata_cmlist(
        a=['default:has_none=mother,size=30'])
    ibs = qreq_.ibs  # NOQA
    cm = cm_list[0]  # NOQA
    # import plottool as pt
    # import ibeis.viz
    # pt.ensure_pylab_qt4()
    # ibeis.viz.viz_graph.make_name_graph_interaction(ibs, aids=qreq_.qaids)
    pass


def segmentation_example():
    import vigra
    import opengm
    import sklearn
    import sklearn.mixture
    import numpy as np
    from vigra import graphs
    import matplotlib as mpl
    import plottool as pt

    pt.ensure_pylab_qt4()

    # load image and convert to LAB
    img_fpath = str(ut.grab_test_imgpath(str('lena.png')))
    img = vigra.impex.readImage(img_fpath)
    imgLab = vigra.colors.transform_RGB2Lab(img)

    superpixelDiameter = 15   # super-pixel size
    slicWeight = 15.0        # SLIC color - spatial weight
    labels, nseg = vigra.analysis.slicSuperpixels(imgLab, slicWeight,
                                                  superpixelDiameter)
    labels = vigra.analysis.labelImage(labels) - 1

    # get 2D grid graph and RAG
    gridGraph = graphs.gridGraph(img.shape[0:2])
    rag = graphs.regionAdjacencyGraph(gridGraph, labels)

    # Node Features
    nodeFeatures = rag.accumulateNodeFeatures(imgLab)
    nodeFeaturesImg = rag.projectNodeFeaturesToGridGraph(nodeFeatures)
    nodeFeaturesImg = vigra.taggedView(nodeFeaturesImg, "xyc")
    nodeFeaturesImgRgb = vigra.colors.transform_Lab2RGB(nodeFeaturesImg)

    nCluster = 5
    g = sklearn.mixture.GMM(n_components=nCluster)
    g.fit(nodeFeatures[:, :])
    clusterProb = g.predict_proba(nodeFeatures)
    # https://github.com/opengm/opengm/blob/master/src/interfaces/python/examples/tutorial/Irregular%20Factor%20Graphs.ipynb
    # https://github.com/opengm/opengm/blob/master/src/interfaces/python/examples/tutorial/Hard%20and%20Soft%20Constraints.ipynb
    clusterProbImg = rag.projectNodeFeaturesToGridGraph(
        clusterProb.astype(np.float32))
    clusterProbImg = vigra.taggedView(clusterProbImg, "xyc")

    ndim_data = clusterProbImg.reshape((-1, nCluster))
    pca = sklearn.decomposition.PCA(n_components=3)
    print(ndim_data.shape)
    pca.fit(ndim_data)
    print(ut.repr2(pca.explained_variance_ratio_, precision=2))
    oldshape = (clusterProbImg.shape[0:2] + (-1,))
    clusterProgImg3 = pca.transform(ndim_data).reshape(oldshape)
    print(clusterProgImg3.shape)

    # graphical model with as many variables
    # as superpixels, each has 3 states
    gm = opengm.gm(np.ones(rag.nodeNum, dtype=opengm.label_type) * nCluster)
    # convert probabilites to energies
    probs = np.clip(clusterProb, 0.00001, 0.99999)
    costs = -1.0 * np.log(probs)
    # add ALL unaries AT ONCE
    fids = gm.addFunctions(costs)
    gm.addFactors(fids, np.arange(rag.nodeNum))
    # add a potts function
    beta = 40.0  # strength of potts regularizer
    regularizer = opengm.pottsFunction([nCluster] * 2, 0.0, beta)
    fid = gm.addFunction(regularizer)
    # get variable indices of adjacent superpixels
    # - or "u" and "v" node id's for edges
    uvIds = rag.uvIds()
    uvIds = np.sort(uvIds, axis=1)
    # add all second order factors at once
    gm.addFactors(fid, uvIds)

    # get super-pixels with slic on LAB image
    Inf = opengm.inference.BeliefPropagation
    parameter = opengm.InfParam(steps=10, damping=0.5, convergenceBound=0.001)
    inf = Inf(gm, parameter=parameter)

    class PyCallback(object):

        def __init__(self,):
            self.labels = []

        def begin(self, inference):
            print("begin of inference")

        def end(self, inference):
            self.labels.append(inference.arg())

        def visit(self, inference):
            gm = inference.gm()
            labelVector = inference.arg()
            print("energy  %r" % (gm.evaluate(labelVector),))
            self.labels.append(labelVector)

    callback = PyCallback()
    visitor = inf.pythonVisitor(callback, visitNth=1)

    inf.infer(visitor)

    pt.imshow(clusterProgImg3.swapaxes(0, 1))
    # plot superpixels
    cmap = mpl.colors.ListedColormap(np.random.rand(nseg, 3))
    pt.imshow(labels.swapaxes(0, 1).squeeze(), cmap=cmap)
    pt.imshow(nodeFeaturesImgRgb)

    cmap = mpl.colors.ListedColormap(np.random.rand(nCluster, 3))
    for arg in callback.labels:
        arg = vigra.taggedView(arg, "n")
        argImg = rag.projectNodeFeaturesToGridGraph(arg.astype(np.uint32))
        argImg = vigra.taggedView(argImg, "xy")
        # plot superpixels
        pt.imshow(argImg.swapaxes(0, 1).squeeze(), cmap=cmap)


def dummy_multicut():
    """ """
    # Places to look for the definition of PottsGFunction class
    # ~/code/opengm/src/interfaces/python/opengm/opengmcore/pyFunctionTypes.cxx
    # /src/interfaces/python/opengm/opengmcore/function_injector.py
    # A Comparative Study of Modern Inference Techniques for Structured Discrete Energy Minimization Problems
    # http://arxiv.org/pdf/1404.0533.pdf
    # __init__( (object)arg1, (object)shape [, (object)values=()]) -> object :
    # values = np.arange(1, ut.num_partitions(num_annots) + 1)
    # http://hci.iwr.uni-heidelberg.de/opengm2/doxygen/opengm-2.1.1/classopengm_1_1PottsGFunction.html
    import opengm
    import numpy as np
    from itertools import product
    cost_matrix = np.array([
        [ 1. ,  0.2, -0.6, -0.2],
        [ 0.2,  1. , -0.6,  0.8],
        [-0.6, -0.6,  1. , -0.8],
        [-0.2,  0.8, -0.8,  1. ]])
    num_vars = len(cost_matrix)

    # Enumerate undirected edges (node index pairs)
    var_indices = np.arange(num_vars)
    varindex_pairs = np.array(
        [(a1, a2) for a1, a2 in product(var_indices, var_indices)
         if a1 != a2 and a1 > a2], dtype=np.uint32)
    varindex_pairs.sort(axis=1)

    # Create nodes in the graphical model.  In this case there are <num_vars>
    # nodes and each node can be assigned to one of <num_vars> possible labels
    num_nodes = num_vars
    space = np.full((num_nodes,), fill_value=num_vars, dtype=np.int)
    gm = opengm.gm(space)

    # Use one potts function for each edge
    for varx1, varx2 in varindex_pairs:
        cost = cost_matrix[varx1, varx2]
        potts_func = opengm.PottsFunction((num_vars, num_vars), valueEqual=0, valueNotEqual=cost)
        potts_func_id = gm.addFunction(potts_func)
        var_indicies = np.array([varx1, varx2])
        gm.addFactor(potts_func_id, var_indicies)

    #opengm.visualizeGm(gm=gm)

    InfAlgo = opengm.inference.Multicut
    parameter = opengm.InfParam()
    inf = InfAlgo(gm, parameter=parameter)
    inf.infer()
    labels = inf.arg()
    print(labels)

    import plottool as pt

    #varindex_pairs = np.vstack(np.triu_indices_from(cost_matrix)).T

    # Dummy unaries
    #for varx in var_indices:
    #    unary_func = np.ones(num_vars)
    #    unary_func_id = gm.addFunction(unary_func)
    #    gm.addFactor(unary_func_id, varx1)

    #pt.ensure_pylab_qt4()

    # add a potts function
    #shape = [num_vars] * 2
    # num_parts = 5  # possible number paritions with 4 variables
    # num_parts = ut.get_nth_bell_number(num_vars - 1)
    # Causes a segfault if values is passed in
    # values = np.arange(1, num_parts + 1).astype(np.float64)
    # gpotts_func = opengm.PottsGFunction(shape, values)
    #gpotts_func = opengm.PottsGFunction(shape)
    #gpotts_fid = gm.addFunction(gpotts_func)
    # Commenting out the next line results in a segfault
    #gm.addFactors(gpotts_fid, varindex_pairs)

    # 2nd order function
    # Seems to cause OpenGM error: Invalid Model for Multicut-Solver! Solver requires a generalized potts model!
    # pair_fid = gm.addFunction(cost_matrix)
    # gm.addFactors(pair_fid, varindex_pairs)

    InfAlgo = opengm.inference.Multicut
    # Not sure what parameters are allowed to be passed here.
    parameter = opengm.InfParam()
    inf = InfAlgo(gm, parameter=parameter)
    inf.infer()

    class PyCallback(object):

        def __init__(self,):
            self.labels = []

        def begin(self, inference):
            print("begin of inference")

        def end(self, inference):
            self.labels.append(inference.arg())

        def visit(self, inference):
            gm = inference.gm()
            labelVector = inference.arg()
            print("energy  %r" % (gm.evaluate(labelVector),))
            self.labels.append(labelVector)

    callback = PyCallback()
    visitor = inf.pythonVisitor(callback, visitNth=1)
    inf.infer(visitor)
    print(callback.labels)

    print(cost_matrix)
    pt.imshow(cost_matrix, cmap='magma')
    opengm.visualizeGm(gm=gm)


def dummy_cut_example():
    r"""
    CommandLine:
        python -m ibeis.workflow --exec-dummy_cut_example --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.workflow import *  # NOQA
        >>> result = dummy_cut_example()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import opengm
    import numpy as np
    import plottool as pt
    pt.ensure_pylab_qt4()
    # Matching Graph
    cost_matrix = np.array([
        [0.5, 0.6, 0.2, 0.4],
        [0.0, 0.5, 0.2, 0.9],
        [0.0, 0.0, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.5],
    ])
    cost_matrix += cost_matrix.T
    number_of_labels = 4
    num_annots = 4
    #cost_matrix = (cost_matrix * 2) - 1

    #gm = opengm.gm(number_of_labels)
    gm = opengm.gm(np.ones(num_annots) * number_of_labels)
    aids = np.arange(num_annots)
    aid_pairs = np.array([(a1, a2) for a1, a2 in ut.iprod(
        aids, aids) if a1 != a2], dtype=np.uint32)
    aid_pairs.sort(axis=1)

    # add a potts function
    # penalizes neighbors for having different labels
    # beta = 0   # 0.1  # strength of potts regularizer
    #beta = 0.1   # 0.1  # strength of potts regularizer

    # Places to look for the definition of this stupid class
    # ~/code/opengm/src/interfaces/python/opengm/opengmcore/pyFunctionTypes.cxx
    # /src/interfaces/python/opengm/opengmcore/function_injector.py

    #shape = [number_of_labels] * 2
    #regularizer = opengm.PottsGFunction(shape, 0.0, beta)
    # __init__( (object)arg1, (object)shape [, (object)values=()]) -> object :

    # values = np.arange(1, ut.num_partitions(num_annots) + 1)
    #regularizer = opengm.PottsGFunction(shape)
    #reg_fid = gm.addFunction(regularizer)

    # A Comparative Study of Modern Inference Techniques for Structured Discrete Energy Minimization Problems
    # http://arxiv.org/pdf/1404.0533.pdf

    # regularizer1 = opengm.pottsFunction([number_of_labels] * 2, valueEqual=0.0, valueNotEqual=beta)

    # gm.addFactors(reg_fid, aid_pairs)

    # 2nd order function
    pair_fid = gm.addFunction(cost_matrix)
    gm.addFactors(pair_fid, aid_pairs)

    if False:
        Inf = opengm.inference.BeliefPropagation
        parameter = opengm.InfParam(steps=10, damping=0.5, convergenceBound=0.001)
    else:
        Inf = opengm.inference.Multicut
        parameter = opengm.InfParam()

    inf = Inf(gm, parameter=parameter)

    class PyCallback(object):

        def __init__(self,):
            self.labels = []

        def begin(self, inference):
            print("begin of inference")

        def end(self, inference):
            self.labels.append(inference.arg())

        def visit(self, inference):
            gm = inference.gm()
            labelVector = inference.arg()
            print("energy  %r" % (gm.evaluate(labelVector),))
            self.labels.append(labelVector)

    callback = PyCallback()
    visitor = inf.pythonVisitor(callback, visitNth=1)
    inf.infer(visitor)
    print(callback.labels)

    print(cost_matrix)
    pt.imshow(cost_matrix, cmap='magma')
    opengm.visualizeGm(gm=gm)
    pass


def intra_encounter_matching():
    import numpy as np
    from scipy.sparse import coo_matrix, csgraph
    qreq_, cm_list = testdata_workflow()
    # qaids = [cm.qaid for cm in cm_list]
    # top_aids = [cm.get_top_aids(5) for cm in cm_list]
    aid_pairs = np.array([(cm.qaid, daid)
                          for cm in cm_list for daid in cm.get_top_aids(5)])
    top_scores = ut.flatten([cm.get_top_scores(5) for cm in cm_list])

    N = aid_pairs.max() + 1
    mat = coo_matrix((top_scores, aid_pairs.T), shape=(N, N))
    csgraph.connected_components(mat)
    tree = csgraph.minimum_spanning_tree(mat)  # NOQA
    import plottool as pt
    dense = mat.todense()
    pt.imshow(dense / dense.max() * 255)
    pt.show_if_requested()

    # baseline jobid
    import opengm
    # https://github.com/opengm/opengm/blob/master/src/interfaces/python/examples/tutorial/OpenGM%20tutorial.ipynb
    numVar = 10
    unaries = np.ones([numVar, 3], dtype=opengm.value_type)
    gm = opengm.gm(np.ones(numVar, dtype=opengm.label_type) * 3)
    unary_fids = gm.addFunctions(unaries)
    gm.addFactors(unary_fids, np.arange(numVar))
    infParam = opengm.InfParam(
        workflow=ut.ensure_ascii('(IC)(TTC-I,CC-I)'),
    )
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
    """
    # TODO: multiway cut
    """


def vs_exemplar_matching():
    pass


def choose_exemplars():
    pass


def detect_photobombs():
    pass


def detect_merges():
    pass


def detect_splits():
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.workflow
        python -m ibeis.workflow --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
