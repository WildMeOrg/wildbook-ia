# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[ibs]')


"""
Buidl probability curves of
* lnbnn, ratio, logratio, l2dist, etc for tp and tn of different db sizes

Rescore based on progressively increasing thresholds

"""


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
    cm_list = qreq_.ibs.query_chips(qreq_=qreq_)
    return qreq_, cm_list


def find_most_disitnctive_keypoints():
    r"""
    CommandLine:
        python -m ibeis.workflow --exec-find_most_disitnctive_keypoints --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.workflow import *  # NOQA
        >>> find_most_disitnctive_keypoints()
    """
    import ibeis
    cm_list, qreq_ = ibeis.testdata_cmlist(a=['default:has_none=mother,size=30'])
    ibs = qreq_.ibs  # NOQA
    cm = cm_list[0]  # NOQA
    # import plottool as pt
    # import ibeis.viz
    # pt.ensure_pylab_qt4()
    # ibeis.viz.viz_graph.make_name_graph_interaction(ibs, aids=qreq_.qaids)
    pass


def intra_encounter_matching():
    qreq_, cm_list = testdata_workflow()
    # qaids = [cm.qaid for cm in cm_list]
    # top_aids = [cm.get_top_aids(5) for cm in cm_list]
    import numpy as np
    from scipy.sparse import coo_matrix, csgraph
    aid_pairs = np.array([(cm.qaid, daid) for cm in cm_list for daid in cm.get_top_aids(5)])
    top_scores = ut.flatten([cm.get_top_scores(5) for cm in cm_list])

    N = aid_pairs.max() + 1
    mat = coo_matrix((top_scores, aid_pairs.T), shape=(N, N))
    csgraph.connected_components(mat)
    tree = csgraph.minimum_spanning_tree(mat)  # NOQA
    import plottool as pt
    dense = mat.todense()
    pt.imshow(dense / dense.max() * 255)
    pt.show_if_requested()

    # load image and convert to LAB
    img_fpath = str(ut.grab_test_imgpath(str('lena.png')))
    img = vigra.impex.readImage(img_fpath)
    imgLab = vigra.colors.transform_RGB2Lab(img)

    superpixelDiameter = 15   # super-pixel size
    slicWeight = 15.0        # SLIC color - spatial weight
    labels, nseg = vigra.analysis.slicSuperpixels(imgLab, slicWeight,
                                                  superpixelDiameter)
    labels = vigra.analysis.labelImage(labels)-1

    # get 2D grid graph and RAG
    gridGraph = graphs.gridGraph(img.shape[0:2])
    rag = graphs.regionAdjacencyGraph(gridGraph, labels)

    nodeFeatures = rag.accumulateNodeFeatures(imgLab)
    nodeFeaturesImg = rag.projectNodeFeaturesToGridGraph(nodeFeatures)
    nodeFeaturesImg = vigra.taggedView(nodeFeaturesImg, "xyc")
    nodeFeaturesImgRgb = vigra.colors.transform_Lab2RGB(nodeFeaturesImg)

    #from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn import mixture
    nCluster   = 3
    g = mixture.GMM(n_components=nCluster)
    g.fit(nodeFeatures[:,:])
    clusterProb = g.predict_proba(nodeFeatures)

    import numpy
    #https://github.com/opengm/opengm/blob/master/src/interfaces/python/examples/tutorial/Irregular%20Factor%20Graphs.ipynb
    #https://github.com/opengm/opengm/blob/master/src/interfaces/python/examples/tutorial/Hard%20and%20Soft%20Constraints.ipynb

    clusterProbImg = rag.projectNodeFeaturesToGridGraph(clusterProb.astype(numpy.float32))
    clusterProbImg = vigra.taggedView(clusterProbImg, "xyc")

    # strength of potts regularizer
    beta = 40.0
    # graphical model with as many variables
    # as superpixels, each has 3 states
    gm = opengm.gm(numpy.ones(rag.nodeNum,dtype=opengm.label_type)*nCluster)
    # convert probabilites to energies
    probs = numpy.clip(clusterProb, 0.00001, 0.99999)
    costs = -1.0*numpy.log(probs)
    # add ALL unaries AT ONCE
    fids = gm.addFunctions(costs)
    gm.addFactors(fids,numpy.arange(rag.nodeNum))
    # add a potts function
    regularizer = opengm.pottsFunction([nCluster]*2,0.0,beta)
    fid = gm.addFunction(regularizer)
    # get variable indices of adjacent superpixels
    # - or "u" and "v" node id's for edges
    uvIds = rag.uvIds()
    uvIds = numpy.sort(uvIds,axis=1)
    # add all second order factors at once
    gm.addFactors(fid,uvIds)

    # get super-pixels with slic on LAB image

    import opengm
    # Matching Graph
    cost_matrix = np.array([
        [0.5, 0.6, 0.2, 0.4, 0.1],
        [0.0, 0.5, 0.2, 0.9, 0.2],
        [0.0, 0.0, 0.5, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.5],
    ])
    cost_matrix += cost_matrix.T
    number_of_labels = 5
    num_annots = 5
    cost_matrix = (cost_matrix * 2) - 1
    #gm = opengm.gm(number_of_labels)
    gm = opengm.gm(np.ones(num_annots) * number_of_labels)
    aids = np.arange(num_annots)
    aid_pairs = np.array([(a1, a2) for a1, a2 in ut.iprod(aids, aids) if a1 != a2], dtype=np.uint32)
    aid_pairs.sort(axis=1)
    # 2nd order function
    fid = gm.addFunction(cost_matrix)
    gm.addFactors(fid, aid_pairs)
    Inf = opengm.inference.BeliefPropagation
    #Inf = opengm.inference.Multicut
    parameter = opengm.InfParam(steps=10, damping=0.5, convergenceBound=0.001)
    parameter = opengm.InfParam()
    inf = Inf(gm, parameter=parameter)
    class PyCallback(object):
        def __init__(self,):
            self.labels=[]
            pass
        def begin(self,inference):
            print("begin of inference")
            pass
        def end(self,inference):
            self.labels.append(inference.arg())
            pass
        def visit(self,inference):
            gm=inference.gm()
            labelVector=inference.arg()
            print("energy  %r" % (gm.evaluate(labelVector),))
            self.labels.append(labelVector)
            pass
    callback=PyCallback()
    visitor=inf.pythonVisitor(callback,visitNth=1)
    inf.infer(visitor)
    print(callback.labels)
    # baseline jobid
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
