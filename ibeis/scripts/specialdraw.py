from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[specialdraw]')


def general_identify_flow():
    r"""
    Returns:
        ?: name

    CommandLine:
        python -m ibeis.scripts.specialdraw general_identify_flow --show --save pairsim.png --dpi=100 --diskshow --clipwhite

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

    def makenode(name, **attrkw):
        if 'size' in attrkw:
            attrkw['width'], attrkw['height'] = attrkw.pop('size')
        graph.add_node(name, **attrkw)
        return name

    def makecluster(name, num, **attrkw):
        return [makenode(name + str(n), **attrkw) for n in range(num)]

    def add_edge2(u, v, *args, **kwargs):
        v = ut.ensure_iterable(v)
        u = ut.ensure_iterable(u)
        for _u, _v in ut.product(u, v):
            graph.add_edge(_u, _v, *args, **kwargs)

    ns = 500

    annot1 = makenode('Annotation X', width=ns, height=ns, groupid='annot')
    annot2 = makenode('Annotation Y', width=ns, height=ns, groupid='annot')

    global_pairvec = makenode('Global similarity\n(viewpoint, quality, ...)', width=ns * ut.PHI * 1.2)
    local_pairvec = makenode('Local similarities\n(LNBNN, spatial error, ...)',
                             size=(ns * 2.2, ns))
    prob = makenode('Probability\n(same individual and\nsimilar viewpoint)')
    classifier = makenode('Classifier\n(SVM/RF/DNN)')
    agglocal = makenode('Aggregate', size=(ns / 1.1, ns / 2))
    catvecs = makenode('Concatenate', shape='box', size=(ns / 1.1, ns / 2))
    pairvec = makenode('Vector of\npairwise similarities')
    findnn = makenode('Find correspondences\n(nearest neighbors)')

    featX = makenode('Features X', size=(ns / 1.2, ns / 2),
                        groupid='feats', shape='rect')
    featY = makenode('Features Y', size=(ns / 1.2, ns / 2),
                        groupid='feats', shape='rect')

    graph.add_edge(annot1, global_pairvec)
    graph.add_edge(annot2, global_pairvec)

    add_edge2(annot1, featX)
    add_edge2(annot2, featY)

    add_edge2(featX, findnn)
    add_edge2(featY, findnn)

    add_edge2(findnn, local_pairvec)

    graph.add_edge(local_pairvec, agglocal, constraint=True)
    graph.add_edge(agglocal, catvecs, constraint=False)
    graph.add_edge(global_pairvec, catvecs)

    graph.add_edge(catvecs, pairvec)

    # graph.add_edge(annot1, classifier, style='invis')
    # graph.add_edge(pairvec, classifier , constraint=False)
    graph.add_edge(pairvec, classifier)
    graph.add_edge(classifier, prob)

    ut.set_default_node_attributes(graph, 'shape',  'rect')
    ut.set_default_node_attributes(graph, 'fixedsize', 'true')
    ut.set_default_node_attributes(graph, 'width', ns * ut.PHI)
    ut.set_default_node_attributes(graph, 'height', ns)
    ut.set_default_node_attributes(graph, 'regular', False)

    layoutkw = {
        'prog': 'dot',
        'rankdir': 'LR',
        # 'splines': 'curved',
        'splines': 'line',
        # 'splines': 'polyline',
        # 'splines': 'spline',
        'sep': 100 / 72,
        'nodesep': 300 / 72,
        'ranksep': 300 / 72,
        # 'concentrate': 'true', # merges edge lines
        # 'splines': 'ortho',
        # 'aspect': 1,
        # 'ratio': 'compress',
        # 'size': '5,4000',
        # 'rank': 'max',
    }

    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw, fontsize=12)


def graphcut_flow():
    r"""
    Returns:
        ?: name

    CommandLine:
        python -m ibeis.scripts.specialdraw graphcut_flow --show --save cutflow.png --dpi=100 --diskshow --clipwhite

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.specialdraw import *  # NOQA
        >>> graphcut_flow()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import plottool as pt
    pt.ensure_pylab_qt4()
    import networkx as nx
    # pt.plt.xkcd()

    graph = nx.DiGraph()

    def makenode(name, **attrkw):
        graph.add_node(name, **attrkw)
        return name

    def makecluster(name, num, **attrkw):
        return [makenode(name + str(n), **attrkw) for n in range(num)]

    def add_edge2(u, v, *args, **kwargs):
        v = ut.ensure_iterable(v)
        u = ut.ensure_iterable(u)
        for _u, _v in ut.product(u, v):
            graph.add_edge(_u, _v, *args, **kwargs)

    ns = 500

    annot1 = makenode('Unlabeled\nannotations\n(query)', width=ns, height=ns,
                      groupid='annot')
    annot2 = makenode('Labeled\nannotations\n(database)', width=ns, height=ns,
                      groupid='annot')
    occurprob = makenode('Fully connected\nprobabilities')
    cacheprob = makenode('Cached \nprobabilities')
    sparseprob = makenode('Sparse\nprobabilities')

    graph.add_edge(annot1, occurprob)

    graph.add_edge(annot1, sparseprob)
    graph.add_edge(annot2, sparseprob)
    graph.add_edge(annot2, cacheprob)

    matchgraph = makenode('Graph of\npotential matches')
    cutalgo = makenode('Graph cut algorithm')
    cc_names = makenode('Identifications,\n splits, and merges are\nconnected compoments')

    graph.add_edge(occurprob, matchgraph)
    graph.add_edge(sparseprob, matchgraph)
    graph.add_edge(cacheprob, matchgraph)

    graph.add_edge(matchgraph, cutalgo)
    graph.add_edge(cutalgo, cc_names)

    ut.set_default_node_attributes(graph, 'shape',  'rect')
    ut.set_default_node_attributes(graph, 'fixedsize', 'true')
    ut.set_default_node_attributes(graph, 'width', ns * ut.PHI)
    ut.set_default_node_attributes(graph, 'height', ns)
    ut.set_default_node_attributes(graph, 'regular', False)

    layoutkw = {
        'prog': 'dot',
        'rankdir': 'LR',
        'splines': 'line',
        'sep': 100 / 72,
        'nodesep': 300 / 72,
        'ranksep': 300 / 72,
    }

    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw, fontsize=12)


def merge_viewpoint_graph():
    r"""
    CommandLine:
        python -m ibeis.scripts.specialdraw merge_viewpoint_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.specialdraw import *  # NOQA
        >>> result = merge_viewpoint_graph()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import plottool as pt
    import ibeis
    import networkx as nx

    defaultdb = 'PZ_Master1'
    ibs = ibeis.opendb(defaultdb=defaultdb)

    #nids = None
    aids = ibs.get_name_aids(4875)
    ibs.print_annot_stats(aids)

    left_aids = ibs.filter_annots_general(aids, view='left')[0:3]
    right_aids = ibs.filter_annots_general(aids, view='right')
    right_aids = list(set(right_aids) - {14517})[0:3]
    back = ibs.filter_annots_general(aids, view='back')[0:4]
    backleft = ibs.filter_annots_general(aids, view='backleft')[0:4]
    backright = ibs.filter_annots_general(aids, view='backright')[0:4]

    right_graph = nx.DiGraph(ut.upper_diag_self_prodx(right_aids))
    left_graph = nx.DiGraph(ut.upper_diag_self_prodx(left_aids))
    back_edges = [
        tuple([back[0], backright[0]][::1]),
        tuple([back[0], backleft[0]][::1]),
    ]
    back_graph = nx.DiGraph(back_edges)

    # Let the graph be a bit smaller
    right_graph.edge[right_aids[1]][right_aids[2]]['constraint'] = False
    left_graph.edge[left_aids[1]][left_aids[2]]['constraint'] = False

    #right_graph = right_graph.to_undirected().to_directed()
    #left_graph = left_graph.to_undirected().to_directed()
    nx.set_node_attributes(right_graph, 'groupid', 'right')
    nx.set_node_attributes(left_graph, 'groupid', 'left')

    #nx.set_node_attributes(right_graph, 'scale', .2)
    #nx.set_node_attributes(left_graph, 'scale', .2)
    #back_graph.node[back[0]]['scale'] = 2.3

    nx.set_node_attributes(back_graph, 'groupid', 'back')

    view_graph = nx.compose_all([left_graph, back_graph, right_graph])
    view_graph.add_edges_from([
        [backright[0], right_aids[0]][::-1],
        [backleft[0], left_aids[0]][::-1],
    ])
    pt.ensure_pylab_qt4()
    graph = graph = view_graph  # NOQA
    #graph = graph.to_undirected()

    nx.set_edge_attributes(graph, 'color', pt.DARK_ORANGE[0:3])
    #nx.set_edge_attributes(graph, 'color', pt.BLACK)
    nx.set_edge_attributes(graph, 'color', {edge: pt.LIGHT_BLUE[0:3] for edge in back_edges})

    #pt.close_all_figures();
    from ibeis.viz import viz_graph
    viz_graph.viz_netx_chipgraph(ibs, graph, with_images=1, prog='dot', augment_graph=False)

    if False:
        """
        #view_graph = left_graph
        pt.close_all_figures(); viz_netx_chipgraph(ibs, view_graph, with_images=0, prog='neato')
        #viz_netx_chipgraph(ibs, view_graph, layout='pydot', with_images=False)
        #back_graph = make_name_graph_interaction(ibs, aids=back, with_all=False)

        aids = left_aids + back + backleft + backright + right_aids

        for aid, chip in zip(aids, ibs.get_annot_chips(aids)):
            fpath = ut.truepath('~/slides/merge/aid_%d.jpg' % (aid,))
            vt.imwrite(fpath, vt.resize_to_maxdims(chip, (400, 400)))

        ut.copy_files_to(, )

        aids = ibs.filterannots_by_tags(ibs.get_valid_aids(),
        dict(has_any_annotmatch='splitcase'))

        aid1 = ibs.group_annots_by_name_dict(aids)[252]
        aid2 = ibs.group_annots_by_name_dict(aids)[6791]
        aids1 = ibs.get_annot_groundtruth(aid1)[0][0:4]
        aids2 = ibs.get_annot_groundtruth(aid2)[0]

        make_name_graph_interaction(ibs, aids=aids1 + aids2, with_all=False)

        ut.ensuredir(ut.truthpath('~/slides/split/))

        for aid, chip in zip(aids, ibs.get_annot_chips(aids)):
            fpath = ut.truepath('~/slides/merge/aidA_%d.jpg' % (aid,))
            vt.imwrite(fpath, vt.resize_to_maxdims(chip, (400, 400)))
        """
    pass


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
