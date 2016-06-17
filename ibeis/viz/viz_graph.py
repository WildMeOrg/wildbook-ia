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
import itertools
from plottool.abstract_interaction import AbstractInteraction
import plottool as pt
#import sys
#from os.path import join
try:
    import networkx as nx
except ImportError as ex:
    ut.printex(ex, 'Cannot import networkx. pip install networkx', iswarning=True)


def get_name_rowid_edges_from_nids(ibs, nids):
    aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def get_name_rowid_edges_from_aids(ibs, aid_list):
    aids_list, nids = ibs.group_annots_by_name(aid_list)
    #aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=None):
    # Enumerate annotmatch properties
    import numpy as np  # NOQA
    #rng = np.random.RandomState(0)
    #edge_props = {
    #    'weight': rng.rand(len(aids1)),
    #    #'reviewer_confidence': rng.rand(len(aids1)),
    #    #'algo_confidence': rng.rand(len(aids1)),
    #}
    #edge_keys = list(edge_props.keys())
    #edge_vals = ut.dict_take(edge_props, edge_keys)
    if unique_aids is None:
        unique_aids = list(set(aids1 + aids2))
    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aids1, aids2))
    #, *edge_vals))
    node_lbls = [('aid', 'int')]
    #edge_lbls = [('weight', 'float')]

    # Make a graph between the chips
    netx_nodes = [(ntup[0], {key[0]: val for (key, val) in zip(node_lbls, ntup[1:])})
                  for ntup in iter(nodes)]
    netx_edges = [(etup[0], etup[1], {}) for etup in iter(edges)]
    #netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in zip(edge_lbls, etup[2:])})
    #              for etup in iter(edges)]
    graph = nx.DiGraph()
    graph.add_nodes_from(netx_nodes)
    graph.add_edges_from(netx_edges)
    #import plottool as pt
    #nx.set_edge_attributes(graph, 'color', pt.DARK_ORANGE)
    return graph


def ensure_names_are_connected(graph, aids_list):
    aug_graph = graph.copy().to_undirected()
    orig_edges = aug_graph.edges()
    unflat_edges = [list(itertools.product(aids, aids)) for aids in aids_list]
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    new_edges = ut.setdiff_ordered(aid_pairs, aug_graph.edges())

    preweighted_edges = nx.get_edge_attributes(aug_graph, 'weight')
    if preweighted_edges:
        orig_edges = ut.setdiff(orig_edges, list(preweighted_edges.keys()))

    aug_graph.add_edges_from(new_edges)
    # Ensure the largest possible set of original edges is in the MST
    nx.set_edge_attributes(aug_graph, 'weight', dict([(edge, 1.0) for edge in new_edges]))
    nx.set_edge_attributes(aug_graph, 'weight', dict([(edge, 0.1) for edge in orig_edges]))
    for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
        mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
        for edge in mst_sub_graph.edges():
            redge = edge[::-1]
            if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                graph.add_edge(*redge, attr_dict={})


def make_netx_graph_from_aid_groups(ibs, aids_list, only_reviewed_matches=True,
                                    invis_edges=None, ensure_edges=None,
                                    temp_nids=None, allow_directed=False):
    r"""
    Args:
        ibs (ibeis.IBEISController): image analysis api
        aids_list (list):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids_list = [[1, 2, 3, 4], [5, 6, 7]]
        >>> invis_edges = [(1, 5)]
        >>> only_reviewed_matches = True
        >>> graph = make_netx_graph_from_aid_groups(ibs, aids_list,
        >>>                                         only_reviewed_matches,
        >>>                                         invis_edges)
        >>> list(nx.connected_components(graph.to_undirected()))
    """
    #aids_list, nid_list = ibs.group_annots_by_name(aid_list)
    unique_aids = list(ut.flatten(aids_list))

    # grouped version
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)

    if only_reviewed_matches:
        annotmatch_rowids = ibs.get_annotmatch_rowid_from_superkey(aids1, aids2)
        annotmatch_rowids = ut.filter_Nones(annotmatch_rowids)
        aids1 = ibs.get_annotmatch_aid1(annotmatch_rowids)
        aids2 = ibs.get_annotmatch_aid2(annotmatch_rowids)

    graph = make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=unique_aids)

    if ensure_edges is not None:
        if ensure_edges == 'all':
            ensure_edges = list(ut.upper_diag_self_prodx(list(graph.nodes())))
        ensure_edges_ = []
        for edge in ensure_edges:
            edge = tuple(edge)
            redge = tuple(edge[::-1])  # HACK
            if graph.has_edge(*edge):
                ensure_edges_.append(edge)
                pass
                #nx.set_edge_attributes(graph, 'weight', {edge: .001})
            elif (not allow_directed) and graph.has_edge(*redge):
                ensure_edges_.append(redge)
                #nx.set_edge_attributes(graph, 'weight', {redge: .001})
                pass
            else:
                ensure_edges_.append(edge)
                #graph.add_edge(*edge, weight=.001)
                graph.add_edge(*edge)

    if temp_nids is None:
        unique_nids = ibs.get_annot_nids(list(graph.nodes()))
    else:
        # HACK
        unique_nids = [1] * len(list(graph.nodes()))
        #unique_nids = temp_nids

    nx.set_node_attributes(graph, 'nid', dict(zip(graph.nodes(), unique_nids)))

    import plottool as pt
    ensure_names_are_connected(graph, aids_list)

    # Color edges by nid
    color_by_nids(graph, unique_nids=unique_nids)
    if invis_edges:
        for edge in invis_edges:
            if graph.has_edge(*edge):
                nx.set_edge_attributes(graph, 'style', {edge: 'invis'})
                nx.set_edge_attributes(graph, 'invisible', {edge: True})
            else:
                graph.add_edge(*edge, style='invis', invisible=True)

    # Hack color images orange
    if ensure_edges:
        nx.set_edge_attributes(graph, 'color',
                               {tuple(edge): pt.ORANGE for edge in ensure_edges_})

    return graph


def ensure_graph_nid_labels(graph, unique_nids=None, ibs=None):
    if unique_nids is None:
        unique_nids = ibs.get_annot_nids(list(graph.nodes()))
    nodeattrs = dict(zip(graph.nodes(), unique_nids))
    ut.nx_set_default_node_attributes(graph, 'nid', nodeattrs)


def color_by_nids(graph, unique_nids=None, ibs=None, nid2_color_=None):
    """ Colors edges and nodes by nid """
    # TODO use ut.color_nodes
    import plottool as pt

    ensure_graph_nid_labels(graph, unique_nids, ibs=ibs)
    node_to_nid = nx.get_node_attributes(graph, 'nid')
    unique_nids = ut.unique(node_to_nid.values())
    ncolors = len(unique_nids)
    if (ncolors) == 1:
        unique_colors = [pt.UNKNOWN_PURP]
    else:
        if nid2_color_ is not None:
            unique_colors = pt.distinct_colors(ncolors + len(nid2_color_) * 2)
        else:
            unique_colors = pt.distinct_colors(ncolors)
    # Find edges and aids strictly between two nids
    nid_to_color = dict(zip(unique_nids, unique_colors))
    if nid2_color_ is not None:
        # HACK NEED TO ENSURE COLORS ARE NOT REUSED
        nid_to_color.update(nid2_color_)
    edge_aids = list(graph.edges())
    edge_nids = ut.unflat_take(node_to_nid, edge_aids)
    flags = [nids[0] == nids[1] for nids in edge_nids]
    flagged_edge_aids = ut.compress(edge_aids, flags)
    flagged_edge_nids = ut.compress(edge_nids, flags)
    flagged_edge_colors = [nid_to_color[nids[0]] for nids in flagged_edge_nids]
    edge_to_color = dict(zip(flagged_edge_aids, flagged_edge_colors))
    node_to_color = ut.map_dict_vals(ut.partial(ut.take, nid_to_color), node_to_nid)
    nx.set_edge_attributes(graph, 'color', edge_to_color)
    nx.set_node_attributes(graph, 'color', node_to_color)


def augment_graph_mst(ibs, graph):
    import plottool as pt
    #spantree_aids1_ = []
    #spantree_aids2_ = []
    # Add edges between all names
    aid_list = list(graph.nodes())
    aug_digraph = graph.copy()
    # Change all weights in initial graph to be small (likely to be part of mst)
    nx.set_edge_attributes(aug_digraph, 'weight', .0001)
    aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
    if False:
        # Weight edges in the MST based on tenative distances
        # Get tentative node positions
        initial_pos = pt.get_nx_layout(graph.to_undirected(), 'graphviz')['node_pos']
        #initial_pos = pt.get_nx_layout(graph.to_undirected(), 'agraph')['node_pos']
        edge_pts1 = ut.dict_take(initial_pos, aids1)
        edge_pts2 = ut.dict_take(initial_pos, aids2)
        edge_pts1 = vt.atleast_nd(np.array(edge_pts1, dtype=np.int32), 2)
        edge_pts2 = vt.atleast_nd(np.array(edge_pts2, dtype=np.int32), 2)
        edge_weights = vt.L2(edge_pts1, edge_pts2)
    else:
        edge_weights = [1.0] * len(aids1)
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
            #attr_dict = {'color': pt.DARK_ORANGE[0:3]}
            attr_dict = {'color': pt.BLACK[0:3]}
            if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                graph.add_edge(*redge, attr_dict=attr_dict)


def ensure_node_images(ibs, graph):
    node_to_aid = nx.get_node_attributes(graph, 'aid')
    node_list = sorted(list(graph.nodes()))
    aid_list = [node_to_aid.get(node, node) for node in node_list]
    #aid_list = sorted(list(graph.nodes()))
    imgpath_list = ibs.depc_annot.get_property('chips', aid_list, 'img',
                                               config=dict(dim_size=200),
                                               read_extern=False)
    nx.set_node_attributes(graph, 'image', dict(zip(node_list, imgpath_list)))
    if True:
        nx.set_node_attributes(graph, 'shape', 'rect')


def viz_netx_chipgraph(ibs, graph, fnum=None, use_image=False, layout=None,
                       zoom=None, prog='neato', as_directed=False,
                       augment_graph=True, layoutkw=None, framewidth=True, **kwargs):
    r"""
    DEPRICATE or improve

    Args:
        ibs (IBEISController):  ibeis controller object
        graph (nx.DiGraph):
        fnum (int):  figure number(default = None)
        use_image (bool): (default = False)
        zoom (float): (default = 0.4)

    Returns:
        ?: pos

    CommandLine:
        python -m ibeis --tf viz_netx_chipgraph --show

    Cand:
        ibeis review_tagged_joins --save figures4/mergecase.png --figsize=15,15
            --clipwhite --diskshow
        ibeis compute_occurrence_groups --save figures4/occurgraph.png
            --figsize=40,40 --clipwhite --diskshow
        ~/code/ibeis/ibeis/algo/preproc/preproc_occurrence.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> nid_list = ibs.get_valid_nids()[0:10]
        >>> fnum = None
        >>> use_image = True
        >>> zoom = 0.4
        >>> make_name_graph_interaction(ibs, nid_list, prog='neato')
        >>> ut.show_if_requested()
    """
    import plottool as pt
    print('[viz_graph] drawing chip graph')
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = pt.gca()

    if layout is None:
        layout = 'agraph'
    print('layout = %r' % (layout,))

    if use_image:
        ensure_node_images(ibs, graph)
    nx.set_node_attributes(graph, 'shape', 'rect')

    if layoutkw is None:
        layoutkw = {}
    layoutkw['prog'] = layoutkw.get('prog', prog)
    layoutkw.update(kwargs)

    if prog == 'neato':
        graph = graph.to_undirected()

    plotinfo = pt.show_nx(graph,
                          ax=ax,
                          # img_dict=img_dict,
                          layout=layout,
                          # hacknonode=bool(use_image),
                          layoutkw=layoutkw,
                          as_directed=as_directed,
                          framewidth=framewidth,
                          )
    return plotinfo


class AnnotGraphInteraction(AbstractInteraction):
    def __init__(self, ibs, aids=None, selected_aids=[], use_image=True,
                 nids=None, temp_nids=None):
        super(AnnotGraphInteraction, self).__init__()
        self.ibs = ibs
        self.aids = aids
        self.selected_aids = selected_aids
        self.use_image = use_image
        from ibeis.algo.hots import graph_iden
        self.infr = graph_iden.AnnotInference2(aids, nids, temp_nids)

    def make_hud(self):
        """ Creates heads up display """
        import plottool as pt
        hl_slot, hr_slot = pt.make_bbox_positioner(
            y=.02, w=.10, h=.03, xpad=.05, startx=0, stopx=1)
        # Create buttons
        self.append_button('Auto Infer', callback=self.make_inference, rect=hl_slot(0))
        self.append_button('Break', callback=self.break_links, rect=hl_slot(1))
        self.append_button('Link', callback=self.make_links, rect=hl_slot(2))
        self.append_button('Accept', callback=self.confirm, rect=hr_slot(0))
        self.append_button('Deselect', callback=self.unselect_all, rect=hr_slot(1))
        self.append_button('Show', callback=self.show_selected, rect=hr_slot(2))
        #self.append_button(next_text, callback=next_callback, rect=next_rect)

    def make_inference(self, event):
        print('self.selected_aids = %r' % (self.selected_aids,))
        self.infer_cut()
        self.show_page()

    def break_links(self, event):
        print('self.selected_aids = %r' % (self.selected_aids,))

    def make_links(self, event):
        print('self.selected_aids = %r' % (self.selected_aids,))

    def unselect_all(self, event):
        print('self.selected_aids = %r' % (self.selected_aids,))
        for aid in self.selected_aids[:]:
            self.toggle_selected_aid(aid)

    def confirm(self, event):
        print('Not done yet')

    def show_selected(self, event):
        import plottool as pt
        print('show_selected')
        from ibeis.viz import viz_chip
        fnum = pt.ensure_fnum(None)
        print('fnum = %r' % (fnum,))
        pt.figure(fnum=fnum)
        viz_chip.show_many_chips(self.ibs, self.selected_aids)
        pt.update()
        #fig.canvas.update()
        #pt.iup()

    def update_netx_graph(self):

        #self.graph = make_netx_graph_from_aid_groups(
        #    ibs, aids_list, invis_edges=invis_edges,
        #    ensure_edges=ensure_edges, temp_nids=temp_nids)
        pass
        # TODO: allow for a subset of grouped aids to be shown
        #self.graph = make_netx_graph_from_nids(ibs, nids)

    def plot(self, fnum, pnum):
        self.update_netx_graph()
        #if split_check:

        node2_aid = nx.get_node_attributes(self.infr.model.graph, 'aid')
        node2_label = {
            node: '%d:aid=%r' % (node, aid)
            for node, aid in node2_aid.items()
        }
        nx.set_node_attributes(self.infr.model.graph, 'label', node2_label)
        #{self.infr.model.graph}
        layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)
        self.plotinfo = pt.show_nx(self.infr.model.graph,
                                   as_directed=False, fnum=self.fnum,
                                   layoutkw=layoutkw,
                                   use_image=self.use_image, verbose=0)

        ax = pt.gca()
        self.enable_pan_and_zoom(ax)
        #ax.autoscale()

        for aid in self.selected_aids:
            self.highlight_aid(aid)
        self.make_hud()
        #self.static_plot(fnum, pnum)

    def highlight_aid(self, aid, color=None):
        import plottool as pt
        if color is None:
            color = pt.ORANGE
        node = self.aid2_node[aid]
        frame = self.plotinfo['patch_frame_dict'][node]
        frame.set_facecolor(color)
        frame.set_edgecolor(color)

    def toggle_images(self):
        self.use_image = not self.use_image
        self.show_page()

    def toggle_selected_aid(self, aid):
        import plottool as pt
        if aid in self.selected_aids:
            self.selected_aids.remove(aid)
            #self.highlight_aid(aid, pt.WHITE)
            self.highlight_aid(aid, pt.DARK_BLUE)
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
        pos = self.plotinfo['node']['pos']
        nodes = list(pos.keys())
        pos_list = ut.dict_take(pos, nodes)

        # TODO: FIXME
        #x = 10
        #y = 10
        import numpy as np  # NOQA
        x, y = event.xdata, event.ydata
        point = np.array([x, y])
        pos_list = np.array(pos_list)
        index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
        #print('dist = %r' % (dist,))
        node = nodes[index]
        aid = self.node2_aid[node]
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
                        self.ibs, aid1, aid2, qres, qreq_=qreq_)
                    self.show_popup_menu(options, event)

        SELECT_ANNOT = dist < 35
        if SELECT_ANNOT:
            #print(ut.obj_str(ibs.get_annot_info(aid, default=True,
            #                                    name=False, gname=False)))

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
                    self.ibs, aid, refresh_func=refresh_func,
                    with_interact_name=False,
                    config2_=config2_)
                self.show_popup_menu(options, event)
        else:
            if self.event.button == 3:
                options = [
                    ('Toggle images', self.toggle_images),
                ]
                self.show_popup_menu(options, event)


def make_name_graph_interaction(ibs, nids=None, aids=None, selected_aids=[],
                                with_all=True, invis_edges=None,
                                ensure_edges=None, use_image=True,
                                temp_nids=None, **kwargs):
    """
    CommandLine:
        python -m ibeis --tf make_name_graph_interaction --db PZ_MTEST \
            --aids=1,2,3,4,5,6,7,8,9 --show

        python -m ibeis --tf make_name_graph_interaction --db LEWA_splits --nids=1 --show --split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> import plottool as pt
        >>> exec(ut.execstr_funckw(make_name_graph_interaction), globals())
        >>> defaultdb='testdb1'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> nids = ut.get_argval('--nids', type_=list, default=ibs.get_valid_nids()[0:5])
        >>> nids = None if aids is not None else nids
        >>> with_all = not ut.get_argflag('--no-with-all')
        >>> make_name_graph_interaction(ibs, nids, aids, with_all=with_all)
        >>> #pt.zoom_factory()
        >>> ut.show_if_requested()
    """
    if aids is None and nids is not None:
        aids = ut.flatten(ibs.get_name_aids(nids))
    elif nids is not None and aids is not None:
        aids += ibs.get_name_aids(nids)
        aids = ut.unique(aids)
    nids = np.unique(ibs.get_annot_name_rowids(aids))
    if with_all:
        aids = ut.flatten(ibs.get_name_aids(nids))
    self = AnnotGraphInteraction(ibs, aids, selected_aids=selected_aids,
                                 nids=nids, temp_nids=temp_nids,
                                 use_image=use_image)
    self.show_page()
    self.show()
    return self

    #infr.exec_split_check()
    #if split_check:
    #infr = exec_split_check(ibs, aid_list)
    #ut.graph_info(graph, 1)
    #ut.graph_info(infr.model.graph, 1)
    #_aids2 = sorted(list(graph.nodes()))
    #aid2_node = {key: val for val, key in enumerate(aid_list)}
    #graph = infr.model.graph
    #node_to_aid = nx.get_node_attributes(graph, 'aid')
    #node_to_aid = dict(zip(graph.nodes(), graph.nodes()))
    #self._aids2 = sorted(list(self.graph.nodes()))
    #self.aid2_node = {key: val for val, key in enumerate(self._aids2)}

    #node_list = sorted(list(graph.nodes()))
    #_aids2 = [node_to_aid.get(node, node) for node in node_list]
    #aid2_node = dict(zip(_aids2, node_list))
    #node2_aid = node_to_aid

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
