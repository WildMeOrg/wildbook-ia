# -*- coding: utf-8 -*-
"""
Displays the matching graph of individuals

WindowsDepends:
    pip install networkx
    wget http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
    graphviz-2.38.msi
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np  # NOQA
#import sys
#from os.path import join
try:
    import networkx as nx
    #if ut.WIN32:
    #    # Make sure graphviz is in the path
    #    win32_graphviz_bin_paths = [
    #        r'C:\Program Files (x86)\Graphviz2.38\bin',
    #        r'C:\Program Files\Graphviz2.38\bin',
    #    ]
    #    found_dot_exe = False
    #    for path in win32_graphviz_bin_paths:
    #        if ut.checkpath(path):
    #            if ut.checkpath(join(path, 'dot.exe')):
    #                sys.path.append(path)
    #                found_dot_exe = True
    #                break
    #    assert found_dot_exe, 'could not find graphviz'
except ImportError as ex:
    ut.printex(ex, 'Cannot import networkx. pip install networkx', iswarning=True)
#import itertools


ZOOM = ut.get_argval('--zoom', type_=float, default=.4)


def show_chipmatch_graph(ibs, cm_list, qreq_, fnum=None, pnum=None, **kwargs):
    r"""

    CommandLine:
        python -m ibeis --tf show_chipmatch_graph:0 --show
        python -m ibeis --tf show_chipmatch_graph:1 --show
        python -m ibeis --tf show_chipmatch_graph:1 --show --zoom=.15
        python -m ibeis --tf show_chipmatch_graph:1 --zoom=.25 --save foo.jpg --diskshow --figsize=12,6 --dpath=. --dpi=280


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> from ibeis.algo.hots.chip_match import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1, 2, 3, 4, 5, 6, 7, 18])
        >>> [cm.score_nsum(qreq_) for cm in cm_list]
        >>> ut.quit_if_noshow()
        >>> daid = cm.get_groundtruth_daids()[0]
        >>> zoom = ut.get_argval('--zoom', float, .4)
        >>> show_chipmatch_graph(ibs, cm_list, qreq_, zoom=zoom)
        >>> #cm.show_single_annotmatch(qreq_, daid)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> from ibeis.algo.hots.chip_match import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list='all')
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list='all')
        >>> [cm.score_nsum(qreq_) for cm in cm_list]
        >>> ut.quit_if_noshow()
        >>> daid = cm.get_groundtruth_daids()[0]
        >>> zoom = ut.get_argval('--zoom', float, .4)
        >>> show_chipmatch_graph(ibs, cm_list, qreq_, zoom=zoom)
        >>> #cm.show_single_annotmatch(qreq_, daid)
        >>> ut.show_if_requested()
    """
    pass
    # breaking
    num = 3
    qaid_list   = [cm.qaid for cm in cm_list]
    daids_list  = [cm.get_top_aids(num).tolist() for cm in cm_list]
    scores_list = [cm.get_annot_scores(daids) for cm, daids in zip(cm_list, daids_list)]
    #graph = nx.Graph()
    #node = graph.add_node(img)
    # FDSFDS!!!
    graph = make_ibeis_matching_graph(ibs, qaid_list, daids_list, scores_list)
    fnum = None
    zoom = kwargs.get('zoom', .4)
    viz_netx_chipgraph(ibs, graph, fnum=fnum, with_images=True, zoom=zoom)


def make_ibeis_matching_graph(ibs, qaid_list, daids_list, scores_list):
    print('make_ibeis_matching_graph')
    aid1_list = ut.flatten([[qaid] * len(daids)
                            for qaid, daids in zip(qaid_list, daids_list)])
    aid2_list = ut.flatten(daids_list)
    unique_aids = list(set(aid2_list + qaid_list))
    score_list = ut.flatten(scores_list)

    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aid1_list, aid2_list, score_list))
    node_lbls = [('aid', 'int')]
    edge_lbls = [('weight', 'float')]
    graph = make_netx_graph(nodes, edges, node_lbls, edge_lbls)
    return graph


def get_name_rowid_edges_from_nids(ibs, nids):
    aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def get_name_rowid_edges_from_aids2(ibs, aids_list):
    # grouped version
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def get_name_rowid_edges_from_aids(ibs, aid_list):
    aids_list, nids = ibs.group_annots_by_name(aid_list)
    #aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def make_netx_graph_from_aids(ibs, aids_list, full=False):
    #aids_list, nid_list = ibs.group_annots_by_name(aid_list)
    unique_aids = list(ut.flatten(aids_list))
    aids1, aids2 = get_name_rowid_edges_from_aids2(ibs, aids_list)
    if not full:
        annotmatch_rowids = ibs.get_annotmatch_rowid_from_superkey(aids1, aids2)
        annotmatch_rowids = ut.filter_Nones(annotmatch_rowids)
        aids1 = ibs.get_annotmatch_aid1(annotmatch_rowids)
        aids2 = ibs.get_annotmatch_aid2(annotmatch_rowids)
    return make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=unique_aids)


def make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=None):
    # Enumerate annotmatch properties
    import numpy as np  # NOQA
    rng = np.random.RandomState(0)
    edge_props = {
        'weight': rng.rand(len(aids1)),
        #'reviewer_confidence': rng.rand(len(aids1)),
        #'algo_confidence': rng.rand(len(aids1)),
    }
    edge_keys = list(edge_props.keys())
    edge_vals = ut.dict_take(edge_props, edge_keys)
    if unique_aids is None:
        unique_aids = list(set(aids1 + aids2))
    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aids1, aids2, *edge_vals))
    node_lbls = [('aid', 'int')]
    edge_lbls = [('weight', 'float')]
    graph = make_netx_graph(nodes, edges, node_lbls, edge_lbls)
    return graph


def make_netx_graph(nodes, edges, node_lbls=[], edge_lbls=[]):
    print('make_netx_graph')
    # Make a graph between the chips
    netx_nodes = [(ntup[0], {key[0]: val for (key, val) in zip(node_lbls, ntup[1:])})
                  for ntup in iter(nodes)]
    netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in zip(edge_lbls, etup[2:])})
                  for etup in iter(edges)]
    graph = nx.DiGraph()
    graph.add_nodes_from(netx_nodes)
    graph.add_edges_from(netx_edges)
    import plottool as pt
    nx.set_edge_attributes(graph, 'color', pt.DARK_ORANGE)
    return graph


def viz_netx_chipgraph(ibs, graph, fnum=None, with_images=False,
                       layout=None, zoom=ZOOM, prog='neato', as_directed=False,
                       **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        graph (nx.DiGraph):
        fnum (int):  figure number(default = None)
        with_images (bool): (default = False)
        zoom (float): (default = 0.4)

    Returns:
        ?: pos

    CommandLine:
        python -m ibeis --tf viz_netx_chipgraph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> nid_list = ibs.get_valid_nids()[0:10]
        >>> fnum = None
        >>> with_images = True
        >>> zoom = 0.4
        >>> #pos = viz_netx_chipgraph(ibs, graph, fnum, with_images, zoom)
        >>> make_name_graph_interaction(ibs, nid_list, prog='neato')
        >>> #make_name_graph_interaction(ibs, nid_list)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    print('[viz_graph] drawing chip graph')
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = pt.gca()

    aid_list = graph.nodes()

    IMPLICIT_LAYOUT = len(set(ibs.get_annot_nids(aid_list))) != 1
    # FIXME
    #print('zoom = %r' % (zoom,))

    if IMPLICIT_LAYOUT:
        # HACK:
        # Use name edge to make pos (very bad)
        aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
        netx_graph_hack = make_netx_graph_from_aidpairs(ibs, aids1, aids2,
                                                        unique_aids=aid_list)
        pos = nx.nx_agraph.graphviz_layout(netx_graph_hack)

        implicit_netx_graph = nx.Graph()
        implicit_netx_graph.add_nodes_from(aid_list)
        implicit_netx_graph.add_edges_from(list(zip(aids1, aids2)))
    else:
        pos = nx.nx_agraph.graphviz_layout(graph)

    #layout = 'spring' #layout = 'spectral' #layout = 'circular'
    #layout = 'shell' #layout = 'pydot' #layout = 'graphviz'
    if layout is None:
        layout = 'agraph'
    print('layout = %r' % (layout,))

    with_nid_edges = True
    if with_nid_edges:
        spantree_aids1_ = []
        spantree_aids2_ = []

        # Get tentative node positions
        initial_pos = pt.get_nx_layout(graph.to_undirected(), 'graphviz')['node_pos']

        # Add edges between all names
        aug_digraph = graph.copy()
        # Change all weights in initial graph to be small (likely to be part of mst)
        nx.set_edge_attributes(aug_digraph, 'weight', .0001)
        aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
        # Weight edges in the MST based on tenative distances
        edge_pts1 = ut.dict_take(initial_pos, aids1)
        edge_pts2 = ut.dict_take(initial_pos, aids2)
        edge_pts1 = vt.atleast_nd(np.array(edge_pts1, dtype=np.int32), 2)
        edge_pts2 = vt.atleast_nd(np.array(edge_pts2, dtype=np.int32), 2)
        edge_weights = vt.L2(edge_pts1, edge_pts2)
        # Create implicit fully connected (by name) graph
        aug_edges = [(a1, a2, {'weight': w})
                      for a1, a2, w in zip(aids1, aids2, edge_weights)]
        aug_digraph.add_edges_from(aug_edges)

        # Determine which edges need to be added to
        # make original graph connected by name
        aug_graph = aug_digraph.to_undirected()
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            mst_edges = mst_sub_graph.edges()
            for edge in mst_edges:
                redge = edge[::-1]
                #attr_dict = {'implicit': True, 'color': pt.DARK_ORANGE}
                attr_dict = {'color': pt.DARK_ORANGE}
                if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                    graph.add_edge(*redge, attr_dict=attr_dict)

            if len(mst_edges) > 0:
                min_aids1_, min_aids2_ = ut.list_transpose(mst_edges)
                spantree_aids1_.extend(min_aids1_)
                spantree_aids2_.extend(min_aids2_)

    #with_images = True

    # target_size = (300, 300)
    #target_size = (220, 220)
    #target_size = (100, 100)

    if with_images:
        imgpath_list = ibs.depc_annot.get_property('chips', aid_list, 'img',
                                             config=dict(dim_size=300), read_extern=False)
        # img_list = ibs.get_annot_chips(aid_list)
        # img_list = [vt.resize_thumb(img, target_size) for img in img_list]
        # img_list = [vt.cvt_BGR2RGB(img) for img in img_list]
        # size_dict = {aid: vt.get_size(img) for aid, img in zip(aid_list, img_list)}
        # img_dict = {aid: img for aid, img in zip(aid_list, img_list)}
        # import utool
        # utool.embed()
        # imgpath_list = ibs.get_annot_chip_fpath(aid_list)
        nx.set_node_attributes(graph, 'image', dict(zip(aid_list, imgpath_list)))
    else:
        pass
        # csize_list = ibs.get_annot_chip_sizes(aid_list)
        # size_list = [vt.resized_dims_and_ratio(size, target_size)[0]
        #              for size in csize_list]
        # size_dict = {aid: size for aid, size in zip(aid_list, size_list)}
        # img_dict = None

    # nx.set_node_attributes(graph, 'width',
    #                        ut.map_dict_vals(lambda x: x[0], size_dict))
    # nx.set_node_attributes(graph, 'height',
    #                        ut.map_dict_vals(lambda x: x[1], size_dict))
    nx.set_node_attributes(graph, 'shape', 'rect')

    old = False
    #as_directed = True

    layoutkw = {'prog': prog}
    layoutkw.update(kwargs)

    plotinfo = pt.show_nx(graph,
                          ax=ax,
                          # img_dict=img_dict,
                          layout=layout,
                          # hacknonode=bool(with_images),
                          layoutkw=layoutkw,
                          as_directed=as_directed
                          )

    pos = plotinfo['pos']

    if old and with_nid_edges:
        edge_pts1_ = np.array(ut.dict_take(pos, spantree_aids1_))
        edge_pts2_ = np.array(ut.dict_take(pos, spantree_aids2_))
        segments = list(zip(edge_pts1_, edge_pts2_))
        #pt.distinct_colors
        color_list = pt.DARK_ORANGE
        #color_list = pt.BLACK
        import matplotlib as mpl
        line_group = mpl.collections.LineCollection(segments, color=color_list,
                                                    alpha=.3, lw=4)
        ax.add_collection(line_group)

    if with_images:
        offset_img_list = plotinfo['imgdat']['offset_img_list']
        artist_list = plotinfo['imgdat']['artist_list']

        # TODO: move this to the interaction
        def _onresize(event):
            print('foo' + ut.get_timestamp())

        #pt.interact_helpers.connect_callback(fig, 'resize_event', _onresize)
        pt.zoom_factory(ax, offset_img_list)

        for artist, aid in zip(artist_list, aid_list):
            pt.set_plotdat(artist, 'aid', aid)
    else:
        pt.zoom_factory(ax, [])
    #pt.plt.tight_layout()
    ax.autoscale()
    return plotinfo


def special_viewpoint_graph():

    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph --exec-special_viewpoint_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> result = special_viewpoint_graph()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import plottool as pt
    import ibeis

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

    #right_graph = right_graph.to_undirected().to_directed()
    #left_graph = left_graph.to_undirected().to_directed()
    nx.set_node_attributes(right_graph, 'groupid', 'right')
    nx.set_node_attributes(left_graph, 'groupid', 'left')

    nx.set_node_attributes(right_graph, 'scale', .8)
    nx.set_node_attributes(left_graph, 'scale', .8)
    back_graph.node[back[0]]['scale'] = 1.3

    nx.set_node_attributes(back_graph, 'groupid', 'back')

    view_graph = nx.compose_all([left_graph, back_graph, right_graph])
    view_graph.add_edges_from([
        [backright[0], right_aids[0]][::-1],
        [backleft[0], left_aids[0]][::-1],
    ])
    pt.ensure_pylab_qt4()
    graph = graph = view_graph  # NOQA
    #graph = graph.to_undirected()

    nx.set_edge_attributes(graph, 'color', pt.DARK_ORANGE)
    #nx.set_edge_attributes(graph, 'color', pt.BLACK)
    nx.set_edge_attributes(graph, 'color', {edge: pt.LIGHT_BLUE for edge in back_edges})

    #pt.close_all_figures();
    viz_netx_chipgraph(ibs, graph, with_images=1, prog='dot')
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


def make_name_graph_interaction(ibs, nids=None, aids=None, selected_aids=[],
                                zoom=ZOOM, with_all=True, prog='neato', **kwargs):
    """
    CommandLine:
        python -m ibeis --tf make_name_graph_interaction --db PZ_Master1 \
            --aids 2068 1003 --show

        python -m ibeis --tf make_name_graph_interaction --db testdb1 --show

        python -m ibeis --tf make_name_graph_interaction --db PZ_Master1 \
            --aids 2068 1003 1342 758 --show
        python -m ibeis --tf make_name_graph_interaction --db PZ_Master1
        --aids 758 1342 --show

        python -m ibeis --tf make_name_graph_interaction --db PZ_Master1
        --aids 2068 1003 --show
        python -m ibeis --tf make_name_graph_interaction --aids 3 --show

        python -m ibeis --tf make_name_graph_interaction --aids 193 194 195 196 --show
        --db WD_Siva

        python -m ibeis --tf make_name_graph_interaction --aids 792 --show
        --db GZ_ALL --no-with-all

        python -m ibeis --tf make_name_graph_interaction --aids=788,789,790,791
        --show --db GZ_ALL --no-with-all
        python -m ibeis --tf make_name_graph_interaction --aids=782,783,792
        --show --db GZ_ALL --no-with-all


        Ignore:
            from ibeis.viz.viz_graph import *  # NOQA
            import ibeis
            import plottool as pt
            aids = [2068, 1003]
            defaultdb = 'PZ_Master1'
            ibs = ibeis.opendb(defaultdb=defaultdb)

            left_view = ibs.filter_annots_general(view='left')
            right_view = ibs.filter_annots_general(view='right')
            nid2_left = ut.group_items(left_view, ibs.get_annot_nids(left_view))
            nid2_right = ut.group_items(right_view, ibs.get_annot_nids(right_view))

            nids = ut.intersect_ordered(nid2_left.keys(), nid2_right.keys())
            aids_list = ibs.get_name_aids(nids)
            for nid, aids in zip(nids, aids_list):
                print(nid)
                ibs.print_annot_stats(aids)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> import plottool as pt
        >>> exec(ut.execstr_funckw(make_name_graph_interaction), globals())
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> defaultdb='testdb1'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> nids = None if aids is not None else ibs.get_valid_nids()[0:5]
        >>> with_all = not ut.get_argflag('--no-with-all')
        >>> make_name_graph_interaction(ibs, nids, aids, with_all=with_all)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    from plottool.abstract_interaction import AbstractInteraction
    print('aids = %r' % (aids,))

    class NameGraphInteraction(AbstractInteraction):
        def __init__(self, ibs, nids=None, aids=None, selected_aids=[]):
            super(NameGraphInteraction, self).__init__()
            self.ibs = ibs
            self.selected_aids = selected_aids
            self._nids = nids if nids is not None else []
            self._aids = aids if aids is not None else []
            self.with_images = True
            self._aids2 = None

        def update_netx_graph(self):
            nids_list = []

            if self._aids2 is not None:
                nids2 = ibs.get_annot_nids(self._aids2)
                nids_list += [nids2]

            if with_all:
                nids_list += [ibs.get_annot_nids(self._aids)]
                nids_list += [self._nids]
                nids = list(set(ut.flatten(nids_list)))
                aids_list = ibs.get_name_aids(nids)
            else:
                aids_list = ibs.group_annots_by_name(self._aids)[0]
            self.graph = make_netx_graph_from_aids(ibs, aids_list)
            # TODO: allow for a subset of grouped aids to be shown
            #self.graph = make_netx_graph_from_nids(ibs, nids)
            self._aids2 = self.graph.nodes()
            self.aid2_index = {key: val for val, key in enumerate(self._aids2)}

        def plot(self, fnum, pnum):
            from ibeis.viz.viz_graph import viz_netx_chipgraph
            self.update_netx_graph()
            self.plotinfo = viz_netx_chipgraph(self.ibs, self.graph,
                                               fnum=self.fnum,
                                               with_images=self.with_images,
                                               zoom=zoom, prog=prog, **kwargs)
            self.ax = pt.gca()

            # FIXME: this doesn't work anymore
            for aid in self.selected_aids:
                self.highlight_aid(aid)
                pass

            #self.static_plot(fnum, pnum)

        def highlight_aid(self, aid, color=pt.ORANGE):
            ax = self.ax
            index = self.aid2_index[aid]
            try:
                artist = ax.artists[index]
                artist.patch.set_facecolor(color)
            except IndexError:
                pass

        def toggle_images(self):
            self.with_images = not self.with_images
            self.show_page()

        def toggle_selected_aid(self, aid):
            if aid in self.selected_aids:
                self.selected_aids.remove(aid)
                self.highlight_aid(aid, pt.WHITE)
            else:
                self.selected_aids.append(aid)
                self.highlight_aid(aid, pt.ORANGE)
            self.draw()

        def on_key_press(self, event):
            print(event)

            if event.key == 'r':
                self.show_page()
                self.draw()

            if event.key == 'i':
                ut.embed()

            if len(self.selected_aids) == 2:
                ibs = self.ibs
                aid1, aid2 = self.selected_aids
                _rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])
                if _rowid is None:
                    _rowid = ibs.get_annotmatch_rowid_from_superkey([aid2], [aid1])
                rowid = _rowid  # NOQA

        def mark_pair_truth(self, truth):
            if len(len(self.selected_aids)) != 2:
                print('This funciton only work if exactly 2 are selected')
                return
            aid1, aid2 = self.selected_aids
            print('aid2 = %r' % (aid2,))
            print('aid1 = %r' % (aid1,))

        @ut.debug_function_exceptions
        def on_click_inside(self, event, ax):
            self.ax = ax
            self.event = event
            event = self.event
            #print(ax)
            #print(event.x)
            #print(event.y)
            pos = self.plotinfo['pos']
            aids = list(pos.keys())
            pos_list = ut.dict_take(pos, aids)

            # TODO: FIXME
            #x = 10
            #y = 10
            import numpy as np  # NOQA
            x, y = event.xdata, event.ydata
            point = np.array([x, y])
            pos_list = np.array(pos_list)
            index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
            print('dist = %r' % (dist,))

            aid = aids[index]
            context_shown = False

            CHECK_PAIR = True
            if CHECK_PAIR:
                if self.event.button == 3 and not context_shown:
                    if len(self.selected_aids) != 2:
                        print('This funciton only work if exactly 2 are selected')
                    else:
                        from ibeis.gui import inspect_gui
                        context_shown = True
                        aid1, aid2 = (self.selected_aids)
                        qres = None
                        qreq_ = None
                        options = inspect_gui.get_aidpair_context_menu_options(
                            ibs, aid1, aid2, qres, qreq_=qreq_)
                        self.show_popup_menu(options, event)

            SELECT_ANNOT = dist < 35
            if SELECT_ANNOT:
                print(ut.obj_str(ibs.get_annot_info(aid, default=True,
                                                    name=False, gname=False)))

                if self.event.button == 1:
                    self.toggle_selected_aid(aid)

                if self.event.button == 3 and not context_shown:
                    # right click
                    from ibeis.viz.interact import interact_chip
                    context_shown = True
                    #refresh_func = functools.partial(viz.show_name, ibs, nid,
                    #fnum=fnum, sel_aids=sel_aids)
                    refresh_func = None
                    config2_ = None
                    options = interact_chip.build_annot_context_options(
                        ibs, aid, refresh_func=refresh_func,
                        with_interact_name=False,
                        config2_=config2_)
                    self.show_popup_menu(options, event)
            else:
                if self.event.button == 3:
                    options = [
                        ('Toggle images', self.toggle_images),
                    ]
                    self.show_popup_menu(options, event)

    self = NameGraphInteraction(ibs, nids, aids, selected_aids=selected_aids)
    self.show_page()
    self.show()

    #ax = self.fig.axes[0]
    #for index in range(len(ax.artists)):
    #    artist = ax.artists[index]
    #    bbox = artist.patch
    #    bbox.set_facecolor(pt.ORANGE)
    #    #offset_img = artist.offsetbox
    #    #img = offset_img.get_data()
    #    #offset_img.set_data(vt.draw_border(img, thickness=5))

    #bbox = artist.patch

    #ax.figure.canvas.draw()  # force re-draw

    return self


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_graph
        python -m ibeis.viz.viz_graph --allexamples
        python -m ibeis.viz.viz_graph --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
