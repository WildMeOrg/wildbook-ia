# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import warnings
import utool as ut
import vtool as vt  # NOQA
import six
import networkx as nx
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from wbia.algo.graph.state import SAME, DIFF, NULL  # NOQA

print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class GraphVisualization(object):
    """ contains plotting related code """

    def _get_truth_colors(infr):
        import wbia.plottool as pt

        truth_colors = {
            POSTV: pt.TRUE_BLUE,
            NEGTV: pt.FALSE_RED,
            INCMP: pt.YELLOW,
            UNKWN: pt.UNKNOWN_PURP,
            UNREV: pt.GRAY,
        }
        return truth_colors

    @property
    def _error_color(infr):
        import wbia.plottool as pt

        return pt.ORANGE

    def _get_cmap(infr):
        import wbia.plottool as pt

        # return pt.plt.cm.RdYlBu
        if hasattr(infr, '_cmap'):
            return infr._cmap
        else:
            cpool = np.array(
                [
                    [0.98135718, 0.19697982, 0.02117342],
                    [1.0, 0.33971852, 0.0],
                    [1.0, 0.45278535, 0.0],
                    [1.0, 0.55483746, 0.0],
                    [1.0, 0.65106306, 0.0],
                    [1.0, 0.74359729, 0.0],
                    [1.0, 0.83348477, 0.0],
                    [0.98052302, 0.92128928, 0.0],
                    [0.95300175, 1.0, 0.0],
                    [0.59886986, 0.99652954, 0.23932718],
                    [0.2, 0.95791134, 0.44764457],
                    [0.2, 0.89937643, 0.63308702],
                    [0.2, 0.82686023, 0.7895433],
                    [0.2, 0.74361034, 0.89742738],
                    [0.2, 0.65085832, 0.93960823],
                    [0.2, 0.54946918, 0.90949295],
                    [0.25697101, 0.44185497, 0.8138502],
                ]
            )
            cmap = pt.mpl.colors.ListedColormap(cpool, 'indexed')
            # cmap = pt.interpolated_colormap([
            #     (pt.FALSE_RED, 0.0),
            #     (pt.YELLOW, 0.5),
            #     (pt.TRUE_BLUE, 1.0),
            # ], resolution=128)
            infr._cmap = cmap
            return infr._cmap

    def initialize_visual_node_attrs(infr, graph=None):
        infr.print('initialize_visual_node_attrs!!!')
        infr.print('initialize_visual_node_attrs', 3)
        # import networkx as nx
        if graph is None:
            graph = infr.graph

        # nx.set_node_attributes(graph, name='framewidth', values=3.0)
        # nx.set_node_attributes(graph, name='shape', values=ut.dzip(annot_nodes, ['rect']))
        ut.nx_delete_node_attr(graph, 'size')
        ut.nx_delete_node_attr(graph, 'width')
        ut.nx_delete_node_attr(graph, 'height')
        ut.nx_delete_node_attr(graph, 'radius')

        infr._viz_init_nodes = True
        infr._viz_image_config_dirty = False

    def update_node_image_config(infr, **kwargs):
        if not hasattr(infr, '_viz_image_config_dirty'):
            infr.initialize_visual_node_attrs()
        for key, val in kwargs.items():
            assert key in infr._viz_image_config
            if infr._viz_image_config[key] != val:
                infr._viz_image_config[key] = val
                infr._viz_image_config_dirty = True

    def update_node_image_attribute(infr, use_image=False, graph=None):
        if graph is None:
            graph = infr.graph
        if not hasattr(infr, '_viz_image_config_dirty'):
            infr.initialize_visual_node_attrs()
        aid_list = list(graph.nodes())

        if infr.ibs is not None:
            nx.set_node_attributes(graph, name='framewidth', values=3.0)
            nx.set_node_attributes(
                graph, name='shape', values=ut.dzip(aid_list, ['rect'])
            )
            if infr.ibs is None:
                raise ValueError('Cannot show images when ibs is None')
            imgpath_list = infr.ibs.depc_annot.get(
                'chipthumb',
                aid_list,
                'img',
                config=infr._viz_image_config,
                read_extern=False,
            )
            nx.set_node_attributes(
                graph, name='image', values=ut.dzip(aid_list, imgpath_list)
            )
        if graph is infr.graph:
            infr._viz_image_config_dirty = False

    def get_colored_edge_weights(infr, graph=None, highlight_reviews=True):
        # Update color and linewidth based on scores/weight
        import wbia.plottool as pt

        if graph is None:
            graph = infr.graph
        truth_colors = infr._get_truth_colors()
        if highlight_reviews:
            edges = []
            colors = []
            for edge in graph.edges():
                d = infr.get_edge_data(edge)
                state = d.get('evidence_decision', UNREV)
                meta = d.get('meta_decision', NULL)
                color = truth_colors[state]
                if state not in {POSTV, NEGTV}:
                    # Darken and saturated same/diff edges without visual
                    # evidence
                    if meta == SAME:
                        color = truth_colors[POSTV]
                    if meta == DIFF:
                        color = truth_colors[NEGTV]
                    color = pt.color_funcs.adjust_hsv_of_rgb(
                        color, sat_adjust=1, val_adjust=-0.3
                    )
                edges.append(edge)
                colors.append(color)
        else:
            edges = list(graph.edges())
            edge_to_weight = nx.get_edge_attributes(graph, 'normscore')
            weights = np.array(ut.dict_take(edge_to_weight, edges, np.nan))
            nan_idxs = []
            if len(weights) > 0:
                # give nans threshold value
                nan_idxs = np.where(np.isnan(weights))[0]
                thresh = 0.5
                weights[nan_idxs] = thresh
            colors = infr.get_colored_weights(weights)
            # print('!! weights = %r' % (len(weights),))
            # print('!! edges = %r' % (len(edges),))
            # print('!! colors = %r' % (len(colors),))
            if len(nan_idxs) > 0:
                import wbia.plottool as pt

                for idx in nan_idxs:
                    colors[idx] = pt.GRAY
        return edges, colors

    @profile
    def get_colored_weights(infr, weights):
        import wbia.plottool as pt

        # pt.rrrr()
        # cmap_ = 'viridis'
        # cmap_ = 'plasma'
        # cmap_ = pt.plt.cm.RdYlBu
        cmap_ = infr._get_cmap()
        # cmap_ = pt.plt.cm.RdYlBu
        # cmap_ = pt.plt.get_cmap(cmap_)
        thresh = 0.5
        weights[np.isnan(weights)] = thresh
        # colors = pt.scores_to_color(weights, cmap_=cmap_, logscale=True)
        colors = pt.scores_to_color(
            weights, cmap_=cmap_, score_range=(0, 1), logscale=False, cmap_range=None
        )
        return colors

    @property
    def visual_edge_attrs(infr):
        """ all edge visual attrs """
        return infr.visual_edge_attrs_appearance + infr.visual_edge_attrs_space

    @property
    def visual_edge_attrs_appearance(infr):
        """ attrs that pertain to edge color and style """
        # picker doesnt really belong here
        return [
            'alpha',
            'color',
            'implicit',
            'label',
            'linestyle',
            'lw',
            'pos',
            'stroke',
            'capstyle',
            'hatch',
            'style',
            'sketch',
            'shadow',
            'picker',
            'linewidth',
        ]

    @property
    def visual_edge_attrs_space(infr):
        """ attrs that pertain to edge positioning in a plot """
        return [
            'ctrl_pts',
            'end_pt',
            'head_lp',
            'headlabel',
            'lp',
            'start_pt',
            'tail_lp',
            'taillabel',
            'zorder',
        ]

    @property
    def visual_node_attrs(infr):
        return [
            'color',
            'framewidth',
            'image',
            'label',
            'pos',
            'shape',
            'size',
            'height',
            'width',
            'zorder',
        ]

    def simplify_graph(infr, graph=None, copy=True):
        if graph is None:
            graph = infr.graph
        simple = graph.copy() if copy else graph
        ut.nx_delete_edge_attr(simple, infr.visual_edge_attrs)
        ut.nx_delete_node_attr(simple, infr.visual_node_attrs + ['pin'])
        return simple

    @staticmethod
    def make_viz_config(use_image, small_graph):
        from wbia import dtool as dt

        class GraphVizConfig(dt.Config):
            _param_info_list = [
                # Appearance
                ut.ParamInfo('show_image', default=use_image),
                ut.ParamInfo(
                    'in_image',
                    default=use_image,
                    hideif=lambda cfg: not cfg['show_image'],
                ),
                ut.ParamInfo('pin_positions', default=use_image),
                # Visibility
                ut.ParamInfo('show_reviewed_edges', small_graph),
                ut.ParamInfo('show_unreviewed_edges', small_graph),
                ut.ParamInfo('show_inferred_same', small_graph),
                ut.ParamInfo('show_inferred_diff', small_graph),
                ut.ParamInfo('highlight_reviews', True),
                ut.ParamInfo('show_recent_review', False),
                ut.ParamInfo('show_labels', small_graph),
                ut.ParamInfo(
                    'splines',
                    'spline' if small_graph else 'line',
                    valid_values=['line', 'spline', 'ortho'],
                ),
                ut.ParamInfo('groupby', 'name_label', valid_values=['name_label', None]),
            ]

        return GraphVizConfig

    @profile
    def update_visual_attrs(
        infr,
        graph=None,
        show_reviewed_edges=True,
        show_unreviewed_edges=False,
        show_inferred_diff=True,
        show_inferred_same=True,
        show_recent_review=False,
        highlight_reviews=True,
        show_inconsistency=True,
        wavy=False,
        simple_labels=False,
        show_labels=True,
        reposition=True,
        use_image=False,
        edge_overrides=None,
        node_overrides=None,
        colorby='name_label',
        **kwargs
        # hide_unreviewed_inferred=True
    ):
        import wbia.plottool as pt

        infr.print('update_visual_attrs', 3)
        if graph is None:
            graph = infr.graph
        # if hide_cuts is not None:
        #     # show_unreviewed_cuts = not hide_cuts
        #     show_reviewed_cuts = not hide_cuts

        if not getattr(infr, '_viz_init_nodes', False):
            infr._viz_init_nodes = True
            nx.set_node_attributes(graph, name='shape', values='circle')
            # infr.set_node_attrs('shape', 'circle')

        if getattr(infr, '_viz_image_config_dirty', True):
            infr.update_node_image_attribute(graph=graph, use_image=use_image)

        def get_any(dict_, keys, default=None):
            for key in keys:
                if key in dict_:
                    return dict_[key]
            return default

        show_cand = get_any(
            kwargs, ['show_candidate_edges', 'show_candidates', 'show_cand']
        )
        if show_cand is not None:
            show_cand = True
            show_reviewed_edges = True
            show_unreviewed_edges = True
            show_inferred_diff = True
            show_inferred_same = True

        if kwargs.get('show_all'):
            show_cand = True

        # alpha_low = .5
        alpha_med = 0.9
        alpha_high = 1.0

        dark_background = graph.graph.get('dark_background', None)

        # Ensure we are starting from a clean slate
        # if reposition:
        ut.nx_delete_edge_attr(graph, infr.visual_edge_attrs_appearance)

        # Set annotation node labels
        node_to_nid = None
        if not show_labels:
            nx.set_node_attributes(
                graph, name='label', values=ut.dzip(graph.nodes(), [''])
            )
        else:
            if simple_labels:
                nx.set_node_attributes(
                    graph, name='label', values={n: str(n) for n in graph.nodes()}
                )
            else:
                if node_to_nid is None:
                    node_to_nid = nx.get_node_attributes(graph, 'name_label')
                node_to_view = nx.get_node_attributes(graph, 'viewpoint')
                if node_to_view:
                    annotnode_to_label = {
                        aid: 'aid=%r%s\nnid=%r'
                        % (aid, node_to_view[aid], node_to_nid[aid])
                        for aid in graph.nodes()
                    }
                else:
                    annotnode_to_label = {
                        aid: 'aid=%r\nnid=%r' % (aid, node_to_nid[aid])
                        for aid in graph.nodes()
                    }
                nx.set_node_attributes(graph, name='label', values=annotnode_to_label)

        # NODE_COLOR: based on name_label
        ut.color_nodes(
            graph, labelattr=colorby, outof=kwargs.get('outof', None), sat_adjust=-0.4
        )

        # EDGES:
        # Grab different types of edges
        edges, edge_colors = infr.get_colored_edge_weights(graph, highlight_reviews)

        # reviewed_states = nx.get_edge_attributes(graph, 'evidence_decision')
        reviewed_states = {e: infr.edge_decision(e) for e in infr.graph.edges()}
        edge_to_inferred_state = nx.get_edge_attributes(graph, 'inferred_state')
        # dummy_edges = [edge for edge, flag in
        #                nx.get_edge_attributes(graph, '_dummy_edge').items()
        #                if flag]
        edge_to_reviewid = nx.get_edge_attributes(graph, 'review_id')
        recheck_edges = [
            edge
            for edge, split in nx.get_edge_attributes(graph, 'maybe_error').items()
            if split
        ]
        decision_to_edge = ut.group_pairs(reviewed_states.items())
        neg_edges = decision_to_edge[NEGTV]
        pos_edges = decision_to_edge[POSTV]
        incomp_edges = decision_to_edge[INCMP]
        unreviewed_edges = decision_to_edge[UNREV]

        inferred_same = [
            edge for edge, state in edge_to_inferred_state.items() if state == 'same'
        ]
        inferred_diff = [
            edge for edge, state in edge_to_inferred_state.items() if state == 'diff'
        ]
        inconsistent_external = [
            edge
            for edge, state in edge_to_inferred_state.items()
            if state == 'inconsistent_external'
        ]
        inferred_notcomp = [
            edge for edge, state in edge_to_inferred_state.items() if state == 'notcomp'
        ]

        reviewed_edges = incomp_edges + pos_edges + neg_edges
        compared_edges = pos_edges + neg_edges
        uncompared_edges = ut.setdiff(edges, compared_edges)
        nontrivial_inferred_same = ut.setdiff(
            inferred_same, pos_edges + neg_edges + incomp_edges
        )
        nontrivial_inferred_diff = ut.setdiff(
            inferred_diff, pos_edges + neg_edges + incomp_edges
        )
        nontrivial_inferred_edges = nontrivial_inferred_same + nontrivial_inferred_diff

        # EDGE_COLOR: based on edge_weight
        nx.set_edge_attributes(graph, name='color', values=ut.dzip(edges, edge_colors))

        # LINE_WIDTH: based on review_state
        # unreviewed_width = 2.0
        # reviewed_width = 5.0
        unreviewed_width = 1.0
        reviewed_width = 2.0
        if highlight_reviews:
            nx.set_edge_attributes(
                graph, name='linewidth', values=ut.dzip(reviewed_edges, [reviewed_width]),
            )
            nx.set_edge_attributes(
                graph,
                name='linewidth',
                values=ut.dzip(unreviewed_edges, [unreviewed_width]),
            )
        else:
            nx.set_edge_attributes(
                graph, name='linewidth', values=ut.dzip(edges, [unreviewed_width])
            )

        # EDGE_STROKE: based on decision and maybe_error
        # fg = pt.WHITE if dark_background else pt.BLACK
        # nx.set_edge_attributes(graph, name='stroke', values=ut.dzip(reviewed_edges, [{'linewidth': 3, 'foreground': fg}]))
        if show_inconsistency:
            nx.set_edge_attributes(
                graph,
                name='stroke',
                values=ut.dzip(
                    recheck_edges, [{'linewidth': 5, 'foreground': infr._error_color}]
                ),
            )

        # Set linestyles to emphasize PCCs
        # Dash lines between PCCs inferred to be different
        nx.set_edge_attributes(
            graph, name='linestyle', values=ut.dzip(inferred_diff, ['dashed'])
        )

        # Treat incomparable/incon-external inference as different
        nx.set_edge_attributes(
            graph, name='linestyle', values=ut.dzip(inferred_notcomp, ['dashed'])
        )
        nx.set_edge_attributes(
            graph, name='linestyle', values=ut.dzip(inconsistent_external, ['dashed'])
        )

        # Dot lines that we are unsure of
        nx.set_edge_attributes(
            graph, name='linestyle', values=ut.dzip(unreviewed_edges, ['dotted'])
        )

        # Cut edges are implicit and dashed
        # nx.set_edge_attributes(graph, name='implicit', values=ut.dzip(cut_edges, [True]))
        # nx.set_edge_attributes(graph, name='linestyle', values=ut.dzip(cut_edges, ['dashed']))
        # nx.set_edge_attributes(graph, name='alpha', values=ut.dzip(cut_edges, [alpha_med]))

        nx.set_edge_attributes(
            graph, name='implicit', values=ut.dzip(uncompared_edges, [True])
        )

        # Only matching edges should impose constraints on the graph layout
        nx.set_edge_attributes(graph, name='implicit', values=ut.dzip(neg_edges, [True]))
        nx.set_edge_attributes(
            graph, name='alpha', values=ut.dzip(neg_edges, [alpha_med])
        )
        nx.set_edge_attributes(
            graph, name='implicit', values=ut.dzip(incomp_edges, [True])
        )
        nx.set_edge_attributes(
            graph, name='alpha', values=ut.dzip(incomp_edges, [alpha_med])
        )

        # Ensure reviewed edges are visible
        nx.set_edge_attributes(
            graph, name='implicit', values=ut.dzip(reviewed_edges, [False])
        )
        nx.set_edge_attributes(
            graph, name='alpha', values=ut.dzip(reviewed_edges, [alpha_high])
        )

        if True:
            # Infered same edges can be allowed to constrain in order
            # to make things look nice sometimes
            nx.set_edge_attributes(
                graph, name='implicit', values=ut.dzip(inferred_same, [False])
            )
            nx.set_edge_attributes(
                graph, name='alpha', values=ut.dzip(inferred_same, [alpha_high])
            )

        if not kwargs.get('show_same', True):
            nx.set_edge_attributes(
                graph, name='alpha', values=ut.dzip(inferred_same, [0])
            )

        if not kwargs.get('show_diff', True):
            nx.set_edge_attributes(
                graph, name='alpha', values=ut.dzip(inferred_diff, [0])
            )

        if not kwargs.get('show_positive_edges', True):
            nx.set_edge_attributes(graph, name='alpha', values=ut.dzip(pos_edges, [0]))

        if not kwargs.get('show_negative_edges', True):
            nx.set_edge_attributes(graph, name='alpha', values=ut.dzip(neg_edges, [0]))

        if not kwargs.get('show_incomparable_edges', True):
            nx.set_edge_attributes(graph, name='alpha', values=ut.dzip(incomp_edges, [0]))

        if not kwargs.get('show_between', True):
            if node_to_nid is None:
                node_to_nid = nx.get_node_attributes(graph, 'name_label')
            between_edges = [(u, v) for u, v in edges if node_to_nid[u] != node_to_nid[v]]
            nx.set_edge_attributes(
                graph, name='alpha', values=ut.dzip(between_edges, [0])
            )

        # SKETCH: based on inferred_edges
        # Make inferred edges wavy
        if wavy:
            # dict(scale=3.0, length=18.0, randomness=None)]
            nx.set_edge_attributes(
                graph,
                name='sketch',
                values=ut.dzip(
                    nontrivial_inferred_edges,
                    [dict(scale=10.0, length=64.0, randomness=None)],
                ),
            )

        # Make dummy edges more transparent
        # nx.set_edge_attributes(graph, name='alpha', values=ut.dzip(dummy_edges, [alpha_low]))
        selected_edges = kwargs.pop('selected_edges', None)

        # SHADOW: based on most recent
        # Increase visibility of nodes with the most recently changed timestamp
        if show_recent_review and edge_to_reviewid and selected_edges is None:
            review_ids = list(edge_to_reviewid.values())
            recent_idxs = ut.argmax(review_ids, multi=True)
            recent_edges = ut.take(list(edge_to_reviewid.keys()), recent_idxs)
            selected_edges = recent_edges

        if selected_edges is not None:
            # TODO: add photoshop-like parameters like
            # spread and size. offset is the same as angle and distance.
            nx.set_edge_attributes(
                graph,
                name='shadow',
                values=ut.dzip(
                    selected_edges,
                    [
                        {
                            'rho': 0.3,
                            'alpha': 0.6,
                            'shadow_color': 'w' if dark_background else 'k',
                            'offset': (0, 0),
                            'scale': 3.0,
                        }
                    ],
                ),
            )

        # Z_ORDER: make sure nodes are on top
        nodes = list(graph.nodes())
        nx.set_node_attributes(graph, name='zorder', values=ut.dzip(nodes, [10]))
        nx.set_edge_attributes(graph, name='zorder', values=ut.dzip(edges, [0]))
        nx.set_edge_attributes(graph, name='picker', values=ut.dzip(edges, [10]))

        # VISIBILITY: Set visibility of edges based on arguments
        if not show_reviewed_edges:
            infr.print('Making reviewed edges invisible', 10)
            nx.set_edge_attributes(
                graph, name='style', values=ut.dzip(reviewed_edges, ['invis'])
            )

        if not show_unreviewed_edges:
            infr.print('Making un-reviewed edges invisible', 10)
            nx.set_edge_attributes(
                graph, name='style', values=ut.dzip(unreviewed_edges, ['invis'])
            )

        if not show_inferred_same:
            infr.print('Making nontrivial_same edges invisible', 10)
            nx.set_edge_attributes(
                graph, name='style', values=ut.dzip(nontrivial_inferred_same, ['invis'])
            )

        if not show_inferred_diff:
            infr.print('Making nontrivial_diff edges invisible', 10)
            nx.set_edge_attributes(
                graph, name='style', values=ut.dzip(nontrivial_inferred_diff, ['invis'])
            )

        if selected_edges is not None:
            # Always show the most recent review (remove setting of invis)
            # infr.print('recent_edges = %r' % (recent_edges,))
            nx.set_edge_attributes(
                graph, name='style', values=ut.dzip(selected_edges, [''])
            )

        if reposition:
            # LAYOUT: update the positioning layout
            def get_layoutkw(key, default):
                return kwargs.get(key, graph.graph.get(key, default))

            layoutkw = dict(
                prog='neato',
                splines=get_layoutkw('splines', 'line'),
                fontsize=get_layoutkw('fontsize', None),
                fontname=get_layoutkw('fontname', None),
                sep=10 / 72,
                esep=1 / 72,
                nodesep=0.1,
            )
            layoutkw.update(kwargs)
            # print(ut.repr3(graph.edges))
            pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

        if edge_overrides:
            for key, edge_to_attr in edge_overrides.items():
                nx.set_edge_attributes(graph, name=key, values=edge_to_attr)
        if node_overrides:
            for key, node_to_attr in node_overrides.items():
                nx.set_node_attributes(graph, name=key, values=node_to_attr)

    @profile
    def show_graph(
        infr,
        graph=None,
        use_image=False,
        update_attrs=True,
        with_colorbar=False,
        pnum=(1, 1, 1),
        zoomable=True,
        pickable=False,
        **kwargs,
    ):
        r"""
        Args:
            infr (?):
            graph (None): (default = None)
            use_image (bool): (default = False)
            update_attrs (bool): (default = True)
            with_colorbar (bool): (default = False)
            pnum (tuple):  plot number(default = (1, 1, 1))
            zoomable (bool): (default = True)
            pickable (bool): (de = False)
            **kwargs: verbose, with_labels, fnum, layout, ax, pos, img_dict,
                      title, layoutkw, framewidth, modify_ax, as_directed,
                      hacknoedge, hacknode, node_labels, arrow_width, fontsize,
                      fontweight, fontname, fontfamilty, fontproperties

        CommandLine:
            python -m wbia.algo.graph.mixin_viz GraphVisualization.show_graph --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_viz import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> import wbia.plottool as pt
            >>> infr = demo.demodata_infr(ccs=ut.estarmap(
            >>>    range, [(1, 6), (6, 10), (10, 13), (13, 15), (15, 16),
            >>>            (17, 20)]))
            >>> pnum_ = pt.make_pnum_nextgen(nRows=1, nCols=3)
            >>> infr.show_graph(show_cand=True, simple_labels=True, pickable=True, fnum=1, pnum=pnum_())
            >>> infr.add_feedback((1, 5), INCMP)
            >>> infr.add_feedback((14, 18), INCMP)
            >>> infr.refresh_candidate_edges()
            >>> infr.show_graph(show_cand=True, simple_labels=True, pickable=True, fnum=1, pnum=pnum_())
            >>> infr.add_feedback((17, 18), NEGTV)  # add inconsistency
            >>> infr.apply_nondynamic_update()
            >>> infr.show_graph(show_cand=True, simple_labels=True, pickable=True, fnum=1, pnum=pnum_())
            >>> ut.show_if_requested()
        """
        import wbia.plottool as pt

        if graph is None:
            graph = infr.graph
        # kwargs['fontsize'] = kwargs.get('fontsize', 8)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # default_update_kw = ut.get_func_kwargs(infr.update_visual_attrs)
            # update_kw = ut.update_existing(default_update_kw, kwargs)
            # infr.update_visual_attrs(**update_kw)
            if update_attrs:
                infr.update_visual_attrs(graph=graph, **kwargs)
            verbose = kwargs.pop('verbose', infr.verbose)
            pt.show_nx(
                graph,
                layout='custom',
                as_directed=False,
                modify_ax=False,
                use_image=use_image,
                pnum=pnum,
                verbose=verbose,
                **kwargs,
            )
            if zoomable:
                pt.zoom_factory()
                pt.pan_factory(pt.gca())

        # if with_colorbar:
        #     # Draw a colorbar
        #     _normal_ticks = np.linspace(0, 1, num=11)
        #     _normal_scores = np.linspace(0, 1, num=500)
        #     _normal_colors = infr.get_colored_weights(_normal_scores)
        #     cb = pt.colorbar(_normal_scores, _normal_colors, lbl='weights',
        #                      ticklabels=_normal_ticks)

        #     # point to threshold location
        #     thresh = None
        #     if thresh is not None:
        #         xy = (1, thresh)
        #         xytext = (2.5, .3 if thresh < .5 else .7)
        #         cb.ax.annotate('threshold', xy=xy, xytext=xytext,
        #                        arrowprops=dict(
        #                            alpha=.5, fc="0.6",
        #                            connectionstyle="angle3,angleA=90,angleB=0"),)

        # infr.graph
        if graph.graph.get('dark_background', None):
            pt.dark_background(force=True)

        if pickable:
            fig = pt.gcf()
            fig.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

    def show_edge(infr, edge, fnum=None, pnum=None, **kwargs):
        import wbia.plottool as pt

        match = infr._exec_pairwise_match([edge])[0]
        fnum = pt.ensure_fnum(fnum)
        pt.figure(fnum=fnum, pnum=pnum)
        ax = pt.gca()
        showkw = dict(
            vert=False,
            heatmask=True,
            show_lines=False,
            show_ell=False,
            show_ori=False,
            show_eig=False,
            modifysize=True,
        )
        showkw.update(kwargs)
        match.show(ax, **showkw)

    def draw_aids(infr, aids, fnum=None):
        from wbia.viz import viz_chip
        import wbia.plottool as pt

        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(infr.ibs, aids, fnum=fnum)
        return fig

    def start_qt_interface(infr, loop=True):
        import wbia.guitool as gt
        from wbia.viz.viz_graph2 import AnnotGraphWidget
        from wbia.plottool import abstract_interaction
        import wbia.plottool as pt

        pt.qtensure()
        gt.ensure_qtapp()
        # win = AnnotGraphWidget(infr=infr, use_image=False, init_mode='review')
        win = AnnotGraphWidget(infr=infr, use_image=False, init_mode=None)
        abstract_interaction.register_interaction(win)
        if loop:
            gt.qtapp_loop(qwin=win, freq=10)
        else:
            win.show()
        return win

    def debug_edge_repr(infr):
        print('DEBUG EDGE REPR')
        for u, v, d in infr.graph.edges(data=True):
            print('edge = %r, %r' % (u, v))
            print(infr.repr_edge_data(d, visual=False))

    def repr_edge_data(infr, all_edge_data, visual=True):
        visual_edge_data = {
            k: v for k, v in all_edge_data.items() if k in infr.visual_edge_attrs
        }
        edge_data = ut.delete_dict_keys(all_edge_data.copy(), infr.visual_edge_attrs)
        lines = []
        if visual:
            lines += [('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1))]
        lines += [('edge_data: ' + ut.repr2(edge_data, nl=1))]
        return '\n'.join(lines)

    def show_error_case(infr, aids, edge=None, error_edges=None, colorby=None, fnum=1):
        """
        Example
        """
        import wbia.plottool as pt

        if error_edges is None:
            # compute a minimal set of edges to minimally fix the case
            pass

        sub_infr = infr.subgraph(aids)

        # err_graph.add_edges_from(missing_edges)
        subdf = sub_infr.get_edge_dataframe()
        mistake_edges = []
        if len(subdf) > 0:
            mistakes = subdf[
                (subdf.truth != subdf.evidence_decision)
                & (subdf.evidence_decision != UNREV)
            ]
            mistake_edges = mistakes.index.tolist()
        err_edges = mistake_edges + list(error_edges)
        missing = [e for e in err_edges if not sub_infr.has_edge(e)]

        # Hack, make sure you don't reuse
        sub_infr.graph.add_edges_from(missing)

        stroke = {'linewidth': 2.5, 'foreground': sub_infr._error_color}
        edge_overrides = {
            # 'alpha': {e: .05 for e in true_negatives},
            'alpha': {},
            'style': {e: '' for e in err_edges},
            'sketch': {e: None for e in err_edges},
            'linestyle': {e: 'dashed' for e in missing},
            'linewidth': {e: 2.0 for e in err_edges + missing},
            'stroke': {e: stroke for e in err_edges + missing},
        }
        selected_kw = {
            'stroke': {'linewidth': 5, 'foreground': sub_infr._error_color},
            'alpha': 1.0,
        }
        for k, v in selected_kw.items():
            if k not in edge_overrides:
                edge_overrides[k] = {}
            edge_overrides[k][edge] = selected_kw[k]

        sub_infr.show_edge(edge, fnum=1, pnum=(2, 1, 2))
        ax = pt.gca()
        xy, w, h = pt.get_axis_xy_width_height(ax=ax)

        nx.set_node_attributes(sub_infr.graph, name='framewidth', values=1.0)
        nx.set_node_attributes(sub_infr.graph, name='framealign', values='outer')
        nx.set_node_attributes(sub_infr.graph, name='framealpha', values=0.7)
        sub_infr.show_graph(
            fnum=fnum,
            pnum=(2, 1, 1),
            show_recent_review=False,
            zoomable=False,
            pickable=False,
            show_cand=False,
            splines='spline',
            simple_labels=True,
            colorby=colorby,
            use_image=True,
            edge_overrides=edge_overrides,
            # ratio=1 / abs(w / h)
        )

    show = show_graph


def on_pick(event, infr=None):
    import wbia.plottool as pt

    print('ON PICK: %r' % (event,))
    artist = event.artist
    plotdat = pt.get_plotdat_dict(artist)
    if plotdat:
        if 'node' in plotdat:
            all_node_data = ut.sort_dict(plotdat['node_data'].copy())
            visual_node_data = ut.dict_subset(all_node_data, infr.visual_node_attrs, None)
            node_data = ut.delete_dict_keys(all_node_data, infr.visual_node_attrs)
            node = plotdat['node']
            node_data['degree'] = infr.graph.degree(node)
            node_label = infr.pos_graph.node_label(node)
            print('visual_node_data: ' + ut.repr2(visual_node_data, nl=1))
            print('node_data: ' + ut.repr2(node_data, nl=1))
            ut.cprint('node: ' + ut.repr2(plotdat['node']), 'blue')
            print('(pcc) node_label = %r' % (node_label,))
            print('artist = %r' % (artist,))
        elif 'edge' in plotdat:
            all_edge_data = ut.sort_dict(plotdat['edge_data'].copy())
            print(infr.repr_edge_data(all_edge_data))
            ut.cprint('edge: ' + ut.repr2(plotdat['edge']), 'blue')
            print('artist = %r' % (artist,))
        else:
            print('???: ' + ut.repr2(plotdat))
    print(ut.get_timestamp())


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_viz
        python -m wbia.algo.graph.mixin_viz --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
