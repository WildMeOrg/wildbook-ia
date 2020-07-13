# -*- coding: utf-8 -*-
"""
DEPRICATE

Displays the matching graph of individuals

WindowsDepends:
    pip install networkx
    wget http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
    graphviz-2.38.msi
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import utool as ut
import vtool as vt
from wbia import dtool
import numpy as np  # NOQA
import itertools
from wbia.plottool.abstract_interaction import AbstractInteraction
import wbia.plottool as pt
from wbia.algo.graph.state import POSTV, NEGTV, INCMP

# import sys
# from os.path import join
try:
    import networkx as nx
except ImportError as ex:
    ut.printex(ex, 'Cannot import networkx. pip install networkx', iswarning=True)

(print, rrr, profile) = ut.inject2(__name__)


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
    # aids_list = ibs.get_name_aids(nids)
    import itertools

    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=None):
    # Enumerate annotmatch properties
    import numpy as np  # NOQA

    # rng = np.random.RandomState(0)
    # edge_props = {
    #    'weight': rng.rand(len(aids1)),
    #    #'reviewer_confidence': rng.rand(len(aids1)),
    #    #'algo_confidence': rng.rand(len(aids1)),
    # }
    # edge_keys = list(edge_props.keys())
    # edge_vals = ut.dict_take(edge_props, edge_keys)
    if unique_aids is None:
        unique_aids = list(set(aids1 + aids2))
    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aids1, aids2))
    # , *edge_vals))
    node_lbls = [('aid', 'int')]
    # edge_lbls = [('weight', 'float')]

    # Make a graph between the chips
    netx_nodes = [
        (ntup[0], {key[0]: val for (key, val) in zip(node_lbls, ntup[1:])})
        for ntup in iter(nodes)
    ]
    netx_edges = [(etup[0], etup[1], {}) for etup in iter(edges)]
    # netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in zip(edge_lbls, etup[2:])})
    #              for etup in iter(edges)]
    graph = nx.DiGraph()
    graph.add_nodes_from(netx_nodes)
    graph.add_edges_from(netx_edges)
    # import wbia.plottool as pt
    # nx.set_edge_attributes(graph, name='color', values=pt.DARK_ORANGE)
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
    nx.set_edge_attributes(
        aug_graph, name='weight', values=dict([(edge, 1.0) for edge in new_edges])
    )
    nx.set_edge_attributes(
        aug_graph, name='weight', values=dict([(edge, 0.1) for edge in orig_edges])
    )
    for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
        mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
        for edge in mst_sub_graph.edges():
            redge = edge[::-1]
            if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                graph.add_edge(*redge, attr_dict={})


def make_netx_graph_from_aid_groups(
    ibs,
    aids_list,
    only_reviewed_matches=True,
    invis_edges=None,
    ensure_edges=None,
    temp_nids=None,
    allow_directed=False,
):
    r"""
    Args:
        ibs (wbia.IBEISController): image analysis api
        aids_list (list):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids_list = [[1, 2, 3, 4], [5, 6, 7]]
        >>> invis_edges = [(1, 5)]
        >>> only_reviewed_matches = True
        >>> graph = make_netx_graph_from_aid_groups(ibs, aids_list,
        >>>                                         only_reviewed_matches,
        >>>                                         invis_edges)
        >>> list(nx.connected_components(graph.to_undirected()))
    """
    # aids_list, nid_list = ibs.group_annots_by_name(aid_list)
    unique_aids = list(ut.flatten(aids_list))

    # grouped version
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)

    if only_reviewed_matches:
        annotmatch_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(
            aids1, aids2
        )
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
                # nx.set_edge_attributes(graph, name='weight', values={edge: .001})
            elif (not allow_directed) and graph.has_edge(*redge):
                ensure_edges_.append(redge)
                # nx.set_edge_attributes(graph, name='weight', values={redge: .001})
                pass
            else:
                ensure_edges_.append(edge)
                # graph.add_edge(*edge, weight=.001)
                graph.add_edge(*edge)

    if temp_nids is None:
        unique_nids = ibs.get_annot_nids(list(graph.nodes()))
    else:
        # HACK
        unique_nids = [1] * len(list(graph.nodes()))
        # unique_nids = temp_nids

    nx.set_node_attributes(graph, name='nid', value=ut.dzip(graph.nodes(), unique_nids))

    import wbia.plottool as pt

    ensure_names_are_connected(graph, aids_list)

    # Color edges by nid
    color_by_nids(graph, unique_nids=unique_nids)
    if invis_edges:
        for edge in invis_edges:
            if graph.has_edge(*edge):
                nx.set_edge_attributes(graph, name='style', values={edge: 'invis'})
                nx.set_edge_attributes(graph, name='invisible', values={edge: True})
            else:
                graph.add_edge(*edge, style='invis', invisible=True)

    # Hack color images orange
    if ensure_edges:
        nx.set_edge_attributes(
            graph,
            name='color',
            values={tuple(edge): pt.ORANGE for edge in ensure_edges_},
        )

    return graph


def ensure_graph_nid_labels(graph, unique_nids=None, ibs=None):
    if unique_nids is None:
        unique_nids = ibs.get_annot_nids(list(graph.nodes()))
    nodeattrs = dict(zip(graph.nodes(), unique_nids))
    ut.nx_set_default_node_attributes(graph, 'nid', nodeattrs)


def color_by_nids(graph, unique_nids=None, ibs=None, nid2_color_=None):
    """ Colors edges and nodes by nid """
    # TODO use ut.color_nodes
    import wbia.plottool as pt

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
    nx.set_edge_attributes(graph, name='color', values=edge_to_color)
    nx.set_node_attributes(graph, name='color', values=node_to_color)


def augment_graph_mst(ibs, graph):
    import wbia.plottool as pt

    # spantree_aids1_ = []
    # spantree_aids2_ = []
    # Add edges between all names
    aid_list = list(graph.nodes())
    aug_digraph = graph.copy()
    # Change all weights in initial graph to be small (likely to be part of mst)
    nx.set_edge_attributes(aug_digraph, name='weight', values=0.0001)
    aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
    if False:
        # Weight edges in the MST based on tenative distances
        # Get tentative node positions
        initial_pos = pt.get_nx_layout(graph.to_undirected(), 'graphviz')['node_pos']
        # initial_pos = pt.get_nx_layout(graph.to_undirected(), 'agraph')['node_pos']
        edge_pts1 = ut.dict_take(initial_pos, aids1)
        edge_pts2 = ut.dict_take(initial_pos, aids2)
        edge_pts1 = vt.atleast_nd(np.array(edge_pts1, dtype=np.int32), 2)
        edge_pts2 = vt.atleast_nd(np.array(edge_pts2, dtype=np.int32), 2)
        edge_weights = vt.L2(edge_pts1, edge_pts2)
    else:
        edge_weights = [1.0] * len(aids1)
    # Create implicit fully connected (by name) graph
    aug_edges = [(a1, a2, {'weight': w}) for a1, a2, w in zip(aids1, aids2, edge_weights)]
    aug_digraph.add_edges_from(aug_edges)

    # Determine which edges need to be added to
    # make original graph connected by name
    aug_graph = aug_digraph.to_undirected()
    for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
        mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
        mst_edges = mst_sub_graph.edges()
        for edge in mst_edges:
            redge = edge[::-1]
            # attr_dict = {'color': pt.DARK_ORANGE[0:3]}
            attr_dict = {'color': pt.BLACK[0:3]}
            if not (graph.has_edge(*edge) or graph.has_edge(*redge)):
                graph.add_edge(*redge, attr_dict=attr_dict)


def ensure_node_images(ibs, graph):
    node_to_aid = nx.get_node_attributes(graph, 'aid')
    node_list = sorted(list(graph.nodes()))
    aid_list = [node_to_aid.get(node, node) for node in node_list]
    # aid_list = sorted(list(graph.nodes()))
    imgpath_list = ibs.depc_annot.get_property(
        'chips', aid_list, 'img', config=dict(dim_size=200), read_extern=False
    )
    nx.set_node_attributes(graph, name='image', values=dict(zip(node_list, imgpath_list)))
    if True:
        nx.set_node_attributes(graph, name='shape', values='rect')


def viz_netx_chipgraph(
    ibs,
    graph,
    fnum=None,
    use_image=False,
    layout=None,
    zoom=None,
    prog='neato',
    as_directed=False,
    augment_graph=True,
    layoutkw=None,
    framewidth=True,
    **kwargs,
):
    r"""
    DEPRICATE or improve

    Args:
        ibs (IBEISController):  wbia controller object
        graph (nx.DiGraph):
        fnum (int):  figure number(default = None)
        use_image (bool): (default = False)
        zoom (float): (default = 0.4)

    Returns:
        ?: pos

    CommandLine:
        python -m wbia --tf viz_netx_chipgraph --show

    Cand:
        wbia review_tagged_joins --save figures4/mergecase.png --figsize=15,15
            --clipwhite --diskshow
        wbia compute_occurrence_groups --save figures4/occurgraph.png
            --figsize=40,40 --clipwhite --diskshow
        ~/code/wbia/wbia/algo/preproc/preproc_occurrence.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> nid_list = ibs.get_valid_nids()[0:10]
        >>> fnum = None
        >>> use_image = True
        >>> zoom = 0.4
        >>> make_name_graph_interaction(ibs, nid_list, prog='neato')
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt

    print('[viz_graph] drawing chip graph')
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = pt.gca()

    if layout is None:
        layout = 'agraph'
    print('layout = %r' % (layout,))

    if use_image:
        ensure_node_images(ibs, graph)
    nx.set_node_attributes(graph, name='shape', values='rect')

    if layoutkw is None:
        layoutkw = {}
    layoutkw['prog'] = layoutkw.get('prog', prog)
    layoutkw.update(kwargs)

    if prog == 'neato':
        graph = graph.to_undirected()

    plotinfo = pt.show_nx(
        graph,
        ax=ax,
        # img_dict=img_dict,
        layout=layout,
        # hacknonode=bool(use_image),
        layoutkw=layoutkw,
        as_directed=as_directed,
        framewidth=framewidth,
    )
    return plotinfo


class InferenceConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('min_labels', 1),
        ut.ParamInfo('max_labels', 5),
        ut.ParamInfo(
            'thresh_method', 'elbow', valid_values=['elbow', 'mean', 'median', 'custom']
        ),
        ut.ParamInfo('thresh', 0.5, hideif=lambda cfg: cfg['thresh_method'] != 'custom'),
    ]


class AnnotGraphInteraction(AbstractInteraction):
    def __init__(self, infr, selected_aids=[], use_image=False, temp_nids=None):
        super(AnnotGraphInteraction, self).__init__()
        self.infr = infr
        self.selected_aids = selected_aids
        self.node2_aid = nx.get_node_attributes(self.infr.graph, 'aid')
        self.aid2_node = ut.invert_dict(self.node2_aid)
        node2_label = {
            # node: '%d:aid=%r' % (node, aid)
            node: 'aid=%r' % (aid)
            for node, aid in self.node2_aid.items()
        }
        # self.show_cuts = False
        self.use_image = use_image
        self.show_cuts = False
        self.config = InferenceConfig()
        nx.set_node_attributes(self.infr.graph, name='label', values=node2_label)

    def make_hud(self):
        """ Creates heads up display """
        import wbia.plottool as pt

        hl_slot, hr_slot = pt.make_bbox_positioners(
            y=0.01, w=0.10, h=0.03, xpad=0.01, startx=0, stopx=1
        )

        hl_slot2, hr_slot2 = pt.make_bbox_positioners(
            y=0.05, w=0.10, h=0.03, xpad=0.01, startx=0, stopx=1
        )

        hl_slot3, hr_slot3 = pt.make_bbox_positioners(
            y=0.09, w=0.10, h=0.03, xpad=0.01, startx=0, stopx=1
        )

        def make_position_gen(slot_):
            gen_ = (slot_(x) for x in itertools.count(0))

            def gennext_():
                return six.next(gen_)

            return gennext_

        hl_next = make_position_gen(hl_slot)
        hr_next = make_position_gen(hr_slot)
        hl_next2 = make_position_gen(hl_slot2)
        hr_next2 = make_position_gen(hr_slot2)
        hl_next3 = make_position_gen(hl_slot3)
        hr_next3 = make_position_gen(hr_slot3)

        def _add_button(text, callback, rect):
            self.append_button(text, callback=callback, rect=rect)

        def _debug_button(func, r_next, refresh=True):
            def _simple_onevent(event):
                func()
                if refresh:
                    self.show_page()

            _add_button(ut.get_funcname(func), _simple_onevent, r_next())

        def _toggle_args(states, varname, r_next):
            state = getattr(self, varname)

            def _simple_onevent(event):
                setattr(self, varname, not getattr(self, varname))
                self.show_page()

            text = states[0] if state else states[1]
            return text, _simple_onevent, r_next()

        # Create buttons
        r_next = hl_next
        _add_button('Mark: Match', self.mark_match, r_next())
        _add_button('Mark: Match', self.mark_match, r_next())
        _add_button('Mark: Non-Match', self.mark_nomatch, r_next())
        _add_button('Mark: Not-Comp', self.mark_notcomp, r_next())

        r_next = hr_next
        _add_button('Accept', self.confirm, r_next())
        _add_button('Cut', self.cut, r_next())

        r_next = hr_next2
        _add_button('Params', self.edit_config, r_next())

        r_next = hl_next2
        _add_button('Deselect', self.unselect_all, r_next())
        _add_button('Show Annots', self.show_selected, r_next())

        _add_button(*_toggle_args(['Hide Cuts', 'Show Cuts'], 'show_cuts', r_next))
        # _add_button(*_toggle_args(['Hide Imgs', 'Show Imgs'], 'use_image', r_next))
        _add_button(
            ['Hide Imgs', 'Show Imgs'][self.use_image], self.toggle_imgs, r_next()
        )
        _add_button('Reset', self.reset, r_next())

        r_next = hl_next3
        # Debug row
        _debug_button(self.infr.apply_scores, r_next)
        # _debug_button(self.infr.apply_feedback, r_next)
        _debug_button(self.infr.apply_weights, r_next)
        _debug_button(self.infr.apply_all, r_next)

        r_next = hr_next3
        _debug_button(self.plot_weights, r_next, 0)
        _debug_button(self.print_weights, r_next, 0)

    def print_weights(self, event=None):
        scalars = self.infr.get_scalars()
        for px, (key, vals) in enumerate(scalars.items()):
            print(key + ' = ' + ut.get_stats_str(vals, use_nan=True))

    def plot_weights(self, event=None):
        scalars = self.infr.get_scalars()
        import wbia.plottool as pt

        inter = pt.ExpandableInteraction(fnum=1)
        for px, (key, vals) in enumerate(scalars.items()):
            print(key + ' = ' + ut.get_stats_str(vals, use_nan=True))
            args = (np.arange(len(vals)), sorted(vals))
            kw = dict(title=key, y_label=key, marker='-o', equal_aspect=False)
            inter.append_partial(pt.plot2, *args, **kw)
        inter.start()
        inter.show_page()
        pt.update()

    def edit_config(self, event):
        from wbia import guitool

        guitool.ensure_qtapp()
        from wbia.guitool import PrefWidget2

        self.widget = PrefWidget2.EditConfigWidget(config=self.config)
        self.widget.show()
        # dlg = guitool.ConfigConfirmWidget.as_dialog(None,
        #                                            title='Confirm Import Images',
        #                                            msg='New Settings',
        #                                            config=self.config)
        # dlg.resize(700, 500)
        # self = dlg.widget
        # dlg.exec_()
        # print('self.config = %r' % (self.config,))
        # updated_config = dlg.widget.config  # NOQA
        # print('updated_config = %r' % (updated_config,))
        # print('self.config = %r' % (self.config,))

    def cut(self, event):
        keys = ['min_labels', 'max_labels']
        infrkw = ut.dict_subset(self.config, keys)
        self.infr.relabel_using_inference(**infrkw)
        self.show_page()

    def reset(self, event):
        self.infr.initialize_graph()
        self.show_page()

    def mark_nomatch(self, event):
        print('BREAK LINK self.selected_aids = %r' % (self.selected_aids,))
        import itertools

        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback((aid1, aid2), NEGTV)
        self.show_page()

    def mark_match(self, event):
        print('MAKE LINK self.selected_aids = %r' % (self.selected_aids,))
        import itertools

        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback((aid1, aid2), POSTV)
        self.show_page()

    def mark_notcomp(self, event):
        print('MAKE LINK self.selected_aids = %r' % (self.selected_aids,))
        import itertools

        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback((aid1, aid2), INCMP)
        self.show_page()

    def unselect_all(self, event):
        print('self.selected_aids = %r' % (self.selected_aids,))
        for aid in self.selected_aids[:]:
            self.toggle_selected_aid(aid)

    def confirm(self, event):
        print('Not done yet')
        # print(self.infr.current_name_labels)

    def toggle_imgs(self, event=None):
        self.use_image = not self.use_image
        self.show_page()

    def show_selected(self, event):
        import wbia.plottool as pt

        print('show_selected')
        from wbia.viz import viz_chip

        fnum = pt.ensure_fnum(10)
        print('fnum = %r' % (fnum,))
        pt.figure(fnum=fnum)
        pt.update()
        viz_chip.show_many_chips(self.infr.ibs, self.selected_aids)
        pt.update()
        # fig.canvas.update()
        # pt.iup()

    def plot(self, fnum, pnum):
        self.infr.update_visual_attrs(self.show_cuts)

        layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)

        # draw_implicit=self.show_cuts)
        self.plotinfo = pt.show_nx(
            self.infr.graph,
            as_directed=False,
            fnum=self.fnum,
            layoutkw=layoutkw,
            # node_labels=True,
            modify_ax=False,
            use_image=self.use_image,
            verbose=0,
        )

        ut.util_graph.graph_info(self.infr.graph, verbose=True)

        # _, edge_weights, edge_colors = self.infr.get_colored_edge_weights()
        # pt.colorbar(edge_weights, edge_colors, lbl='weights')

        # _normal_ticks = np.linspace(0, 1, num=11)
        # _normal_scores = np.linspace(0, 1, num=500)
        # _normal_colors = self.infr.get_colored_weights(_normal_scores)
        # cb = pt.colorbar(_normal_scores, _normal_colors, lbl='weights',
        #                  ticklabels=_normal_ticks)

        # cb.ax.annotate('threshold',
        #                xy=(1, self.infr.thresh),
        #                xytext=(2.5, .3 if self.infr.thresh < .5 else .7),
        #                arrowprops=dict(
        #                    alpha=.5,
        #                    fc="0.6",
        #                    connectionstyle="angle3,angleA=90,angleB=0"),)

        ax = pt.gca()
        self.enable_pan_and_zoom(ax)
        # ax.autoscale()
        for aid in self.selected_aids:
            self.highlight_aid(aid, pt.ORANGE)
        # self.static_plot(fnum, pnum)
        self.make_hud()
        # print(ut.repr2(self.infr.graph.edges, nl=2))
        print('Finished Plot')

    def highlight_aid(self, aid, color=None):
        import wbia.plottool as pt

        node = self.aid2_node[aid]
        frame = self.plotinfo['patch_frame_dict'][node]
        framewidth = self.infr.graph.node[node]['framewidth']
        if color is True:
            color = pt.ORANGE
        if color is None:
            color = pt.DARK_BLUE
            color = self.infr.graph.node[node]['color']
            # color = fix_color(color)
            color = pt.ensure_nonhex_color(color)
            frame.set_linewidth(framewidth)
        else:
            frame.set_linewidth(framewidth * 2)
        frame.set_facecolor(color)
        frame.set_edgecolor(color)

    def toggle_selected_aid(self, aid):
        import wbia.plottool as pt

        if aid in self.selected_aids:
            self.selected_aids.remove(aid)
            # self.highlight_aid(aid, pt.WHITE)
            self.highlight_aid(aid, color=None)
        else:
            self.selected_aids.append(aid)
            self.highlight_aid(aid, pt.ORANGE)
        self.draw()

    def on_key_press(self, event):
        print(event)
        infr = self.infr  # NOQA

        if event.key == 'r':
            self.show_page()
            self.draw()

        if event.key == 'l':
            self.infr.rrr()
            self.show_page()

        if event.key == 'i':
            ut.embed()

        if len(self.selected_aids) == 2:
            ibs = self.infr.ibs
            aid1, aid2 = self.selected_aids
            _rowid = ibs.get_annotmatch_rowid_from_undirected_superkey([aid1], [aid2])
            if _rowid is None:
                _rowid = ibs.get_annotmatch_rowid_from_undirected_superkey([aid2], [aid1])
            rowid = _rowid  # NOQA

    @ut.debug_function_exceptions
    def on_click_inside(self, event, ax):
        self.ax = ax
        self.event = event
        event = self.event
        # print(ax)
        # print(event.x)
        # print(event.y)
        pos = self.plotinfo['node']['pos']
        nodes = list(pos.keys())
        pos_list = ut.dict_take(pos, nodes)

        # TODO: FIXME
        # x = 10
        # y = 10
        import numpy as np  # NOQA

        x, y = event.xdata, event.ydata
        point = np.array([x, y])
        pos_list = np.array(pos_list)
        index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
        # print('dist = %r' % (dist,))
        node = nodes[index]
        aid = self.node2_aid[node]
        context_shown = False

        CHECK_PAIR = True
        if CHECK_PAIR:
            if self.event.button == 3 and not context_shown:
                if len(self.selected_aids) != 2:
                    print('This funciton only work if exactly 2 are selected')
                else:
                    from wbia.gui import inspect_gui

                    context_shown = True
                    aid1, aid2 = self.selected_aids
                    qres = None
                    qreq_ = None
                    options = inspect_gui.get_aidpair_context_menu_options(
                        self.infr.ibs, aid1, aid2, qres, qreq_=qreq_
                    )
                    self.show_popup_menu(options, event)

        bbox = vt.bbox_from_center_wh(
            self.plotinfo['node']['pos'][node], self.plotinfo['node']['size'][node]
        )
        SELECT_ANNOT = vt.point_inside_bbox(point, bbox)
        # SELECT_ANNOT = dist < 35

        if SELECT_ANNOT:
            # print(ut.obj_str(ibs.get_annot_info(aid, default=True,
            #                                    name=False, gname=False)))

            if self.event.button == 1:
                self.toggle_selected_aid(aid)

            if self.event.button == 3 and not context_shown:
                # right click
                from wbia.viz.interact import interact_chip

                context_shown = True
                # refresh_func = functools.partial(viz.show_name, ibs, nid,
                # fnum=fnum, sel_aids=sel_aids)
                refresh_func = None
                config2_ = None
                options = interact_chip.build_annot_context_options(
                    self.infr.ibs,
                    aid,
                    refresh_func=refresh_func,
                    with_interact_name=False,
                    config2_=config2_,
                )
                self.show_popup_menu(options, event)
        else:
            if self.event.button == 3:
                options = [
                    ('Toggle images', self.toggle_imgs),
                ]
                self.show_popup_menu(options, event)


def make_name_graph_interaction(
    ibs,
    nids=None,
    aids=None,
    selected_aids=[],
    with_all=True,
    invis_edges=None,
    ensure_edges=None,
    use_image=False,
    temp_nids=None,
    **kwargs,
):
    r"""
    CommandLine:
        python -m wbia --tf make_name_graph_interaction --db PZ_MTEST \
            --aids=1,2,3,4,5,6,7,8,9 --show

        python -m wbia --tf make_name_graph_interaction --db LEWA_splits \
                --nids=1 --show --split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph import *  # NOQA
        >>> import wbia
        >>> import wbia.plottool as pt
        >>> exec(ut.execstr_funckw(make_name_graph_interaction), globals())
        >>> defaultdb='testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
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

    if with_all:
        nids = ut.unique(ibs.get_annot_name_rowids(aids))
        aids = ut.flatten(ibs.get_name_aids(nids))

    # aids = aids[0:10]

    nids = ibs.get_annot_name_rowids(aids)
    # from wbia.algo.graph import graph_iden
    # infr = graph_iden.AnnotInference(aids, nids, temp_nids)  # NOQA
    # import utool
    # utool.embed()

    from wbia.algo.graph import graph_iden

    infr = graph_iden.AnnotInference(ibs, aids, nids, temp_nids)
    infr.initialize_graph()
    # infr.apply_scores()
    # infr.apply_weights()
    if ut.get_argflag('--cut'):
        infr.apply_all()

    # import wbia.guitool as gt
    # gt.ensure_qtapp()
    # print('infr = %r' % (infr,))
    # win = test_qt_graphs(infr=infr, use_image=use_image)
    # self = win
    # gt.qtapp_loop(qwin=win, freq=10)

    self = AnnotGraphInteraction(infr, selected_aids=selected_aids, use_image=use_image)
    self.show_page()
    self.show()
    return self


def tryout_web_graphs(self, infr):
    """
    https://plot.ly/python/

    http://bokeh.pydata.org/en/latest/

    pip install bokeh

    Notes:
        http://www.coppelia.io/2014/07/an-a-to-z-of-extra-features-for-the-d3-force-layout/
        http://andrewmellor.co.uk/blog/articles/2014/12/14/d3-networks/
        pip install plotly  # eww need to sign up and get a key
        http://igraph.org/
    import mpld3
    mpld3.save_html(fig, open('fig.html', 'w'))
    mpld3.save_json(fig, open('fig.json', 'w'))
    fig = pt.gcf()
    """
    # import wbia.plottool as pt
    # http://andrewmellor.co.uk/blog/articles/2014/12/14/d3-networks/
    from networkx.readwrite import json_graph

    G = infr.graph
    data = json_graph.node_link_data(G)
    json_text = ut.to_json(data, pretty=True)
    ut.writeto('graph.json', json_text)
    ut.editfile('graph.json')

    ut.startfile('d3_example.html')
    # d3_location = ut.grab_zipped_url('https://github.com/d3/d3/releases/download/v3.5.17/d3.zip')
    # python -m SimpleHTTPServer 8000


def tryout_with_qt():
    import sys
    from PyQt4 import QtCore, QtWebKit, QtWidgets
    from os.path import join, dirname
    import wbia.viz

    class Browser(QtWebKit.QWebView):
        def __init__(self):
            super(Browser, self).__init__()
            self.loadFinished.connect(self._result_available)

        def _result_available(self, ok):
            pass
            # frame = self.page().mainFrame()
            # print(unicode(frame.toHtml()).encode('utf-8'))

    app = QtWidgets.QApplication(sys.argv)

    view = Browser()
    view.show()
    path = join(dirname(wbia.viz.__file__), 'd3_example.html')
    view.load(QtCore.QUrl(path))
    view.page().settings().setAttribute(
        QtWebKit.QWebSettings.DeveloperExtrasEnabled, True
    )

    insp = QtWebKit.QWebInspector()
    insp.setPage(view.page())
    insp.show()

    app.exec_()


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_graph
        python -m wbia.viz.viz_graph --allexamples
        python -m wbia.viz.viz_graph --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
