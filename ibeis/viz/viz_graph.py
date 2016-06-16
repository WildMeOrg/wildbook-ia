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


def viz_netx_chipgraph(ibs, graph, fnum=None, with_images=False, layout=None,
                       zoom=None, prog='neato', as_directed=False,
                       augment_graph=True, layoutkw=None, framewidth=True, **kwargs):
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

    Cand:
        ibeis review_tagged_joins --save figures4/mergecase.png --figsize=15,15 --clipwhite --diskshow
        ibeis compute_occurrence_groups --save figures4/occurgraph.png --figsize=40,40 --clipwhite --diskshow
        ~/code/ibeis/ibeis/algo/preproc/preproc_occurrence.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> nid_list = ibs.get_valid_nids()[0:10]
        >>> fnum = None
        >>> with_images = True
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

    if with_images:
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
                          # hacknonode=bool(with_images),
                          layoutkw=layoutkw,
                          as_directed=as_directed,
                          framewidth=framewidth,
                          )

    #offset_img_list = []
    #if with_images:
    #    offset_img_list = plotinfo['imgdat']['offset_img_list']
    #    artist_list = plotinfo['imgdat']['artist_list']
    #    aid_list_ = plotinfo['imgdat']['node_list']
    #    for artist, aid in zip(artist_list, aid_list_):
    #        pt.set_plotdat(artist, 'aid', aid)
    # TODO; make part of interaction
    #pt.zoom_factory(ax, offset_img_list)
    #pt.plt.tight_layout()
    #ax.autoscale()
    return plotinfo


def make_name_graph_interaction(ibs, nids=None, aids=None, selected_aids=[],
                                with_all=True, invis_edges=None,
                                ensure_edges=None, with_images=True,
                                split_check=None,
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
    import plottool as pt
    from plottool.abstract_interaction import AbstractInteraction
    print('aids = %r' % (aids,))

    if split_check is None:
        split_check = ut.get_argflag('--split')

    def exec_split_check(ibs, aid_list):
        cfgdict = {
            'can_match_samename': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum'
        }
        qreq_ = ibs.new_query_request(aid_list, aid_list, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        from ibeis.algo.hots import graph_iden
        infr = graph_iden.AnnotInference(qreq_, cm_list)
        import utool
        utool.embed()
        infr.initialize_graph_and_model()
        #print("BUILT SPLIT GRAPH")
        return infr

    class NameGraphInteraction(AbstractInteraction):
        def __init__(self, ibs, nids=None, aids=None, selected_aids=[]):
            super(NameGraphInteraction, self).__init__()
            self.ibs = ibs
            self.selected_aids = selected_aids
            self._nids = nids if nids is not None else []
            self._aids = aids if aids is not None else []
            self.with_images = with_images
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

            self.graph = make_netx_graph_from_aid_groups(
                ibs, aids_list, invis_edges=invis_edges,
                ensure_edges=ensure_edges, temp_nids=temp_nids)
            aid_list = ut.flatten(aids_list)
            if split_check:
                self.infr = exec_split_check(ibs, aid_list)
                ut.graph_info(self.graph, 1)
                ut.graph_info(self.infr.graph, 1)
                self._aids2 = sorted(list(self.graph.nodes()))
                self.aid2_node = {key: val for val, key in enumerate(self._aids2)}
                graph = self.infr.graph
            else:
                graph = self.graph
                #self._aids2 = sorted(list(self.graph.nodes()))
                #self.aid2_node = {key: val for val, key in enumerate(self._aids2)}

            node_to_aid = nx.get_node_attributes(graph, 'aid')
            node_list = sorted(list(graph.nodes()))
            self._aids2 = [node_to_aid.get(node, node) for node in node_list]
            self.aid2_node = dict(zip(self._aids2, node_list))

            #self.graph = make_netx_graph_from_aid_groups(
            #    ibs, aids_list, invis_edges=invis_edges,
            #    ensure_edges=ensure_edges, temp_nids=temp_nids)
            pass
            # TODO: allow for a subset of grouped aids to be shown
            #self.graph = make_netx_graph_from_nids(ibs, nids)

        def plot(self, fnum, pnum):
            from ibeis.viz.viz_graph import viz_netx_chipgraph
            self.update_netx_graph()
            if split_check:
                self.plotinfo = pt.show_nx(self.infr_graph, as_directed=False,
                                           fnum=self.fnum,
                                           layoutkw=dict(prog='neato'),
                                           use_image=True, verbose=0)
                #ax = pt.gca()
                #pt.zoom_factory()
            else:
                self.plotinfo = viz_netx_chipgraph(self.ibs, self.graph,
                                                   fnum=self.fnum,
                                                   with_images=self.with_images,
                                                   **kwargs)
            ax = pt.gca()
            self.enable_pan_and_zoom(ax)
            ax.autoscale()

            # FIXME: this doesn't work anymore
            for aid in self.selected_aids:
                self.highlight_aid(aid)
                pass

            #self.static_plot(fnum, pnum)

        def highlight_aid(self, aid, color=pt.ORANGE):
            ax = self.ax
            index = self.aid2_node[aid]
            try:
                artist = ax.artists[index]
                import matplotlib as mpl
                if isinstance(artist, mpl.patches.Rectangle):
                    patch = artist
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
                else:
                    artist.patch.set_facecolor(color)
                    artist.patch.set_edgecolor(color)
            except IndexError:
                pass

        def toggle_images(self):
            self.with_images = not self.with_images
            self.show_page()

        def toggle_selected_aid(self, aid):
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
