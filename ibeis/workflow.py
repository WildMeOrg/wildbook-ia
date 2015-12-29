# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[ibs]')


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
