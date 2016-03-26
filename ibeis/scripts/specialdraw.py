from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[specialdraw]')


def general_identify_flow():
    r"""
    Returns:
        ?: name

    CommandLine:
        python -m ibeis.scripts.specialdraw general_identify_flow --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.specialdraw import *  # NOQA
        >>> general_identify_flow()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import plottool as pt
    pt.ensure_pylab_qt4()
    import networkx as nx
    # pt.plt.xkcd()

    graph = nx.DiGraph()

    def makenode(name, **attrdict):
        graph.add_node(name, **attrdict)
        return name

    def makecluster(name, num, **attrdict):
        return [makenode(name + str(n), **attrdict) for n in range(num)]

    def add_edge2(u, v, *args, **kwargs):
        v = ut.ensure_iterable(v)
        u = ut.ensure_iterable(u)
        for _u, _v in ut.product(u, v):
            graph.add_edge(_u, _v, *args, **kwargs)

    annot1 = makenode('annotation X', width=700, height=400, groupid='annot')
    annot2 = makenode('annotation Y', width=700, height=400, groupid='annot')

    global_pairvec = makenode('global similarity\nviewpoint\nquality')
    local_pairvec = makenode('local similarities')
    prob = makenode('probability\nsame individual from similar viewpoint')
    classifier = makenode('classifier\nsvm/rf')
    agglocal = makenode('aggregate')
    catvecs = makenode('concatenate')

    graph.add_edge(annot1, global_pairvec)
    graph.add_edge(annot2, global_pairvec)

    featX = makecluster('featX', 1, width=200, height=400,
                        groupid='feats', shape='stack')
    featY = makecluster('featY', 1, width=200, height=400,
                        groupid='feats', shape='stack')

    add_edge2(annot1, featX)
    add_edge2(annot2, featY)

    findnn = makenode('find correspondences\n(nearest neighbors)')

    add_edge2(featX, findnn)
    add_edge2(featY, findnn)

    add_edge2(findnn, local_pairvec)

    graph.add_edge(local_pairvec, agglocal)
    graph.add_edge(global_pairvec, catvecs)
    graph.add_edge(agglocal, catvecs)

    pairvec = makenode('pairwise similarities')

    graph.add_edge(catvecs, pairvec)

    graph.add_edge(pairvec, classifier)
    graph.add_edge(classifier, prob)

    ut.set_default_node_attributes(graph, 'shape',  'rect')
    ut.set_default_node_attributes(graph, 'fixedsize', 'true')
    ut.set_default_node_attributes(graph, 'width', 300 * ut.PHI)
    ut.set_default_node_attributes(graph, 'height', 300)
    ut.set_default_node_attributes(graph, 'regular', False)

    layoutkw = {
        'prog': 'dot',
        'rankdir': 'LR',
        'splines': 'spline',
        # 'concentrate': 'true', # merges edge lines
        # 'splines': 'ortho',
        # 'aspect': 1,
        # 'ratio': 'compress',
        # 'size': '5,4000',
        # 'rank': 'max',
    }

    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw, fontsize=7)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.specialdraw
        python -m ibeis.scripts.specialdraw --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
