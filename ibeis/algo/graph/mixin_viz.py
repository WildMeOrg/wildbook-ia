# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import warnings
import utool as ut
import vtool as vt  # NOQA
import six
import networkx as nx
from ibeis.algo.graph.state import (POSTV, NEGTV, INCMP, UNREV)
print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class GraphVisualization(object):
    """ contains plotting related code """

    def _get_truth_colors(infr):
        import plottool as pt
        truth_colors = {
            # POSTV: pt.TRUE_GREEN,
            POSTV: pt.TRUE_BLUE,
            NEGTV: pt.FALSE_RED,
            INCMP: pt.YELLOW,
            UNREV: pt.UNKNOWN_PURP
        }
        return truth_colors

    @property
    def _error_color(infr):
        import plottool as pt
        return pt.ORANGE

    def _get_cmap(infr):
        import plottool as pt
        # return pt.plt.cm.RdYlBu
        if hasattr(infr, '_cmap'):
            return infr._cmap
        else:
            cpool = np.array([[ 0.98135718,  0.19697982,  0.02117342],
                              [ 1.        ,  0.33971852,  0.        ],
                              [ 1.        ,  0.45278535,  0.        ],
                              [ 1.        ,  0.55483746,  0.        ],
                              [ 1.        ,  0.65106306,  0.        ],
                              [ 1.        ,  0.74359729,  0.        ],
                              [ 1.        ,  0.83348477,  0.        ],
                              [ 0.98052302,  0.92128928,  0.        ],
                              [ 0.95300175,  1.        ,  0.        ],
                              [ 0.59886986,  0.99652954,  0.23932718],
                              [ 0.2       ,  0.95791134,  0.44764457],
                              [ 0.2       ,  0.89937643,  0.63308702],
                              [ 0.2       ,  0.82686023,  0.7895433 ],
                              [ 0.2       ,  0.74361034,  0.89742738],
                              [ 0.2       ,  0.65085832,  0.93960823],
                              [ 0.2       ,  0.54946918,  0.90949295],
                              [ 0.25697101,  0.44185497,  0.8138502 ]])
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
        infr._viz_image_config = dict(in_image=False,
                                      thumbsize=221)

        # nx.set_node_attributes(graph, 'framewidth', 3.0)
        # nx.set_node_attributes(graph, 'shape', ut.dzip(annot_nodes, ['rect']))
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
            nx.set_node_attributes(graph, 'framewidth', 3.0)
            nx.set_node_attributes(graph, 'shape', ut.dzip(aid_list, ['rect']))
            if infr.ibs is None:
                raise ValueError('Cannot show images when ibs is None')
            imgpath_list = infr.ibs.depc_annot.get('chipthumb', aid_list, 'img',
                                                   config=infr._viz_image_config,
                                                   read_extern=False)
            nx.set_node_attributes(graph, 'image', ut.dzip(aid_list,
                                                           imgpath_list))
        if graph is infr.graph:
            infr._viz_image_config_dirty = False

    def get_colored_edge_weights(infr, graph=None, highlight_reviews=True):
        # Update color and linewidth based on scores/weight
        if graph is None:
            graph = infr.graph
        edges = list(graph.edges())
        if highlight_reviews:
            edge_to_decision = nx.get_edge_attributes(graph, 'decision')
            state_to_weight = {
                POSTV: 1.0,
                INCMP: 0.5,
                NEGTV: 0.0,
                UNREV: np.nan
            }
            edge2_weight = ut.map_dict_vals(state_to_weight, edge_to_decision)
            # {e: state for e, state in edge_to_decision.items()}
            # edge2_weight = nx.get_edge_attributes(graph, infr.CUT_WEIGHT_KEY)
        else:
            edge2_weight = nx.get_edge_attributes(graph, 'normscore')
        #edges = list(edge2_weight.keys())
        weights = np.array(ut.dict_take(edge2_weight, edges, np.nan))
        nan_idxs = []
        if len(weights) > 0:
            # give nans threshold value
            nan_idxs = np.where(np.isnan(weights))[0]
            weights[nan_idxs] = infr.thresh
        #weights = weights.compress(is_valid, axis=0)
        #edges = ut.compress(edges, is_valid)
        colors = infr.get_colored_weights(weights)
        #print('!! weights = %r' % (len(weights),))
        #print('!! edges = %r' % (len(edges),))
        #print('!! colors = %r' % (len(colors),))
        if len(nan_idxs) > 0:
            import plottool as pt
            for idx in nan_idxs:
                colors[idx] = pt.GRAY
        return edges, weights, colors

    @profile
    def get_colored_weights(infr, weights):
        import plottool as pt
        #pt.rrrr()
        # cmap_ = 'viridis'
        # cmap_ = 'plasma'
        # cmap_ = pt.plt.cm.RdYlBu
        cmap_ = infr._get_cmap()
        # cmap_ = pt.plt.cm.RdYlBu
        #cmap_ = pt.plt.get_cmap(cmap_)
        weights[np.isnan(weights)] = infr.thresh
        #colors = pt.scores_to_color(weights, cmap_=cmap_, logscale=True)
        colors = pt.scores_to_color(weights, cmap_=cmap_, score_range=(0, 1),
                                    logscale=False, cmap_range=None)
        return colors

    @property
    def visual_edge_attrs(infr):
        """ all edge visual attrs """
        return infr.visual_edge_attrs_appearance + infr.visual_edge_attrs_space

    @property
    def visual_edge_attrs_appearance(infr):
        """ attrs that pertain to edge color and style """
        # picker doesnt really belong here
        return ['alpha', 'color', 'implicit', 'label', 'linestyle', 'lw',
                'pos', 'stroke', 'capstyle', 'hatch', 'style', 'sketch',
                'shadow', 'picker', 'linewidth']

    @property
    def visual_edge_attrs_space(infr):
        """ attrs that pertain to edge positioning in a plot """
        return ['ctrl_pts', 'end_pt', 'head_lp', 'headlabel', 'lp', 'start_pt',
                'tail_lp', 'taillabel', 'zorder']

    @property
    def visual_node_attrs(infr):
        return ['color', 'framewidth', 'image', 'label',
                'pos', 'shape', 'size', 'height', 'width', 'zorder']

    def simplify_graph(infr, graph=None, copy=True):
        if graph is None:
            graph = infr.graph
        simple = graph.copy() if copy else graph
        ut.nx_delete_edge_attr(simple, infr.visual_edge_attrs)
        ut.nx_delete_node_attr(simple, infr.visual_node_attrs + ['pin'])
        return simple

    @staticmethod
    def make_viz_config(use_image, small_graph):
        import dtool as dt
        class GraphVizConfig(dt.Config):
            _param_info_list = [
                # Appearance
                ut.ParamInfo('show_image', default=use_image),
                ut.ParamInfo('in_image', default=use_image, hideif=lambda cfg: not cfg['show_image']),
                ut.ParamInfo('pin_positions', default=use_image),

                # Visibility
                ut.ParamInfo('show_reviewed_edges', small_graph),
                ut.ParamInfo('show_unreviewed_edges', small_graph),
                ut.ParamInfo('show_inferred_same', small_graph),
                ut.ParamInfo('show_inferred_diff', small_graph),
                ut.ParamInfo('highlight_reviews', True),
                ut.ParamInfo('show_recent_review', False),
                ut.ParamInfo('show_labels', small_graph),
                ut.ParamInfo('splines', 'spline' if small_graph else 'line',
                             valid_values=['line', 'spline', 'ortho']),
                ut.ParamInfo('groupby', 'name_label',
                             valid_values=['name_label', None]),
            ]
        return GraphVizConfig

    @profile
    def update_visual_attrs(infr, graph=None,
                            show_reviewed_edges=True,
                            show_unreviewed_edges=False,
                            show_inferred_diff=True,
                            show_inferred_same=True,
                            show_recent_review=True,
                            highlight_reviews=True,
                            show_inconsistency=True,
                            wavy=True,
                            simple_labels=False,
                            show_labels=True,
                            reposition=True,
                            use_image=False,
                            **kwargs
                            # hide_unreviewed_inferred=True
                            ):
        import plottool as pt

        infr.print('update_visual_attrs', 3)
        if graph is None:
            graph = infr.graph
        # if hide_cuts is not None:
        #     # show_unreviewed_cuts = not hide_cuts
        #     show_reviewed_cuts = not hide_cuts

        if not getattr(infr, '_viz_init_nodes', False):
            infr._viz_init_nodes = True
            nx.set_node_attributes(graph, 'shape', 'circle')
            # infr.set_node_attrs('shape', 'circle')

        if getattr(infr, '_viz_image_config_dirty', True):
            infr.update_node_image_attribute(graph=graph, use_image=use_image)

        def get_any(dict_, keys, default=None):
            for key in keys:
                if key in dict_:
                    return dict_[key]
            return default
        show_cand = get_any(kwargs, ['show_candidate_edges', 'show_candidates',
                                     'show_cand'])
        if show_cand is not None:
            show_unreviewed_edges = show_cand

        # alpha_low = .5
        alpha_med = .9
        alpha_high = 1.0

        dark_background = graph.graph.get('dark_background', None)

        # Ensure we are starting from a clean slate
        # if reposition:
        ut.nx_delete_edge_attr(graph, infr.visual_edge_attrs_appearance)

        # Set annotation node labels
        if not show_labels:
            nx.set_node_attributes(graph, 'label', ut.dzip(graph.nodes(), ['']))
        else:
            if simple_labels:
                nx.set_node_attributes(graph, 'label', {n: str(n) for n in graph.nodes()})
            else:
                node_to_nid = nx.get_node_attributes(graph, 'name_label')
                node_to_view = nx.get_node_attributes(graph, 'viewpoint')
                if node_to_view:
                    annotnode_to_label = {
                        aid: 'aid=%r%s\nnid=%r' % (aid, node_to_view[aid],
                                                    node_to_nid[aid])
                        for aid in graph.nodes()
                    }
                else:
                    annotnode_to_label = {
                        aid: 'aid=%r\nnid=%r' % (aid, node_to_nid[aid])
                        for aid in graph.nodes()
                    }
                nx.set_node_attributes(graph, 'label', annotnode_to_label)

        # NODE_COLOR: based on name_label
        ut.color_nodes(graph, labelattr='name_label', sat_adjust=-.4)

        # EDGES:
        # Grab different types of edges
        edges, edge_weights, edge_colors = infr.get_colored_edge_weights(
            graph, highlight_reviews)

        # reviewed_states = nx.get_edge_attributes(graph, 'decision')
        reviewed_states = dict(ut.util_graph.nx_gen_edge_attrs(
            graph, 'decision', default=UNREV))
        edge_to_inferred_state = nx.get_edge_attributes(graph, 'inferred_state')
        # dummy_edges = [edge for edge, flag in
        #                nx.get_edge_attributes(graph, '_dummy_edge').items()
        #                if flag]
        edge_to_reviewid = nx.get_edge_attributes(graph, 'review_id')
        recheck_edges = [edge for edge, split in
                         nx.get_edge_attributes(graph, 'maybe_error').items()
                         if split]
        decision_to_edge = ut.group_pairs(reviewed_states.items())
        nomatch_edges = decision_to_edge[NEGTV]
        match_edges = decision_to_edge[POSTV]
        notcomp_edges = decision_to_edge[INCMP]
        unreviewed_edges = decision_to_edge[UNREV]

        inferred_same = [edge for edge, state in edge_to_inferred_state.items()
                         if state == 'same']
        inferred_diff = [edge for edge, state in edge_to_inferred_state.items()
                         if state == 'diff']
        reviewed_edges = notcomp_edges + match_edges + nomatch_edges
        compared_edges = match_edges + nomatch_edges
        uncompared_edges = ut.setdiff(edges, compared_edges)
        nontrivial_inferred_same = ut.setdiff(inferred_same, match_edges +
                                              nomatch_edges)
        nontrivial_inferred_diff = ut.setdiff(inferred_diff, match_edges +
                                              nomatch_edges)
        nontrivial_inferred_edges = (nontrivial_inferred_same +
                                     nontrivial_inferred_diff)
        nx_set_edge_attrs = nx.set_edge_attributes

        # EDGE_COLOR: based on edge_weight
        nx_set_edge_attrs(graph, 'color', ut.dzip(edges, edge_colors))

        # LINE_WIDTH: based on review_state
        # unreviewed_width = 2.0
        # reviewed_width = 5.0
        unreviewed_width = 1.0
        reviewed_width = 2.0
        if highlight_reviews:
            nx_set_edge_attrs(graph, 'linewidth', ut.dzip(
                reviewed_edges, [reviewed_width]))
            nx_set_edge_attrs(graph, 'linewidth', ut.dzip(
                unreviewed_edges, [unreviewed_width]))
        else:
            nx_set_edge_attrs(graph, 'linewidth', ut.dzip(
                edges, [unreviewed_width]))

        # EDGE_STROKE: based on decision and maybe_error
        # fg = pt.WHITE if dark_background else pt.BLACK
        # nx_set_edge_attrs(graph, 'stroke', ut.dzip(reviewed_edges, [
        #     {'linewidth': 3, 'foreground': fg}]))
        if show_inconsistency:
            nx_set_edge_attrs(graph, 'stroke', ut.dzip(recheck_edges, [
                {'linewidth': 5, 'foreground': infr._error_color}]))

        # Cut edges are implicit and dashed
        # nx_set_edge_attrs(graph, 'implicit', ut.dzip(cut_edges, [True]))
        # nx_set_edge_attrs(graph, 'linestyle', ut.dzip(cut_edges, ['dashed']))
        # nx_set_edge_attrs(graph, 'alpha', ut.dzip(cut_edges, [alpha_med]))

        nx_set_edge_attrs(graph, 'implicit', ut.dzip(uncompared_edges, [True]))

        # Only matching edges should impose constraints on the graph layout
        nx_set_edge_attrs(graph, 'implicit', ut.dzip(nomatch_edges, [True]))
        nx_set_edge_attrs(graph, 'alpha', ut.dzip(nomatch_edges, [alpha_med]))
        nx_set_edge_attrs(graph, 'implicit', ut.dzip(notcomp_edges, [True]))
        nx_set_edge_attrs(graph, 'alpha', ut.dzip(notcomp_edges, [alpha_med]))

        # Ensure reviewed edges are visible
        nx_set_edge_attrs(graph, 'implicit', ut.dzip(reviewed_edges, [False]))
        nx_set_edge_attrs(graph, 'alpha', ut.dzip(reviewed_edges,
                                                       [alpha_high]))

        if True:
            # Infered same edges can be allowed to constrain in order
            # to make things look nice sometimes
            nx_set_edge_attrs(graph, 'implicit', ut.dzip(inferred_same,
                                                              [False]))
            nx_set_edge_attrs(graph, 'alpha', ut.dzip(inferred_same,
                                                           [alpha_high]))

        # SKETCH: based on inferred_edges
        # Make inferred edges wavy
        if wavy:
            nx_set_edge_attrs(
                graph, 'sketch', ut.dzip(nontrivial_inferred_edges, [
                    dict(scale=10.0, length=64.0, randomness=None)]
                    # dict(scale=3.0, length=18.0, randomness=None)]
                ))

        # Make dummy edges more transparent
        # nx_set_edge_attrs(graph, 'alpha', ut.dzip(dummy_edges, [alpha_low]))

        # SHADOW: based on review_timestamp
        # Increase visibility of nodes with the most recently changed timestamp
        if show_recent_review and edge_to_reviewid:
            timestamps = list(edge_to_reviewid.values())
            recent_idxs = ut.where(ut.equal([max(timestamps)], timestamps))
            recent_edges = ut.take(list(edge_to_reviewid.keys()), recent_idxs)
            # TODO: add photoshop-like parameters like
            # spread and size. offset is the same as angle and distance.
            nx_set_edge_attrs(graph, 'shadow', ut.dzip(recent_edges, [{
                'rho': .3,
                'alpha': .6,
                'shadow_color': 'w' if dark_background else 'k',
                # 'offset': (2, -2),
                'offset': (0, 0),
                'scale': 3.0,
                # 'offset': (4, -4)
            }]))

        # Z_ORDER: make sure nodes are on top
        nodes = list(graph.nodes())
        nx.set_node_attributes(graph, 'zorder', ut.dzip(nodes, [10]))
        nx_set_edge_attrs(graph, 'zorder', ut.dzip(edges, [0]))
        nx_set_edge_attrs(graph, 'picker', ut.dzip(edges, [10]))

        # VISIBILITY: Set visibility of edges based on arguments
        if not show_reviewed_edges:
            infr.print('Making reviewed edges invisible', 10)
            nx_set_edge_attrs(graph, 'style',
                                   ut.dzip(reviewed_edges, ['invis']))

        if not show_unreviewed_edges:
            infr.print('Making un-reviewed edges invisible', 10)
            nx_set_edge_attrs(graph, 'style',
                                   ut.dzip(unreviewed_edges, ['invis']))

        if not show_inferred_same:
            infr.print('Making nontrivial_same edges invisible', 10)
            nx_set_edge_attrs(graph, 'style', ut.dzip(
                nontrivial_inferred_same, ['invis']))

        if not show_inferred_diff:
            infr.print('Making nontrivial_diff edges invisible', 10)
            nx_set_edge_attrs(graph, 'style', ut.dzip(
                nontrivial_inferred_diff, ['invis']))

        if show_recent_review and edge_to_reviewid:
            # Always show the most recent review (remove setting of invis)
            infr.print('recent_edges = %r' % (recent_edges,))
            nx_set_edge_attrs(graph, 'style',
                                   ut.dzip(recent_edges, ['']))

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
                nodesep=.1
            )
            layoutkw.update(kwargs)
            # print(ut.repr3(graph.edge))
            pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    @profile
    def show_graph(infr, graph=None, use_image=False, update_attrs=True,
                   with_colorbar=False, pnum=(1, 1, 1),
                   pickable=False, **kwargs):
        import plottool as pt
        if graph is None:
            graph = infr.graph
        # kwargs['fontsize'] = kwargs.get('fontsize', 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # default_update_kw = ut.get_func_kwargs(infr.update_visual_attrs)
            # update_kw = ut.update_existing(default_update_kw, kwargs)
            # infr.update_visual_attrs(**update_kw)
            if update_attrs:
                infr.update_visual_attrs(graph=graph, **kwargs)
            pt.show_nx(graph, layout='custom', as_directed=False,
                       modify_ax=False, use_image=use_image, verbose=2,
                       pnum=pnum, **kwargs)
            pt.zoom_factory()
            pt.pan_factory(pt.gca())

        if with_colorbar:
            # Draw a colorbar
            _normal_ticks = np.linspace(0, 1, num=11)
            _normal_scores = np.linspace(0, 1, num=500)
            _normal_colors = infr.get_colored_weights(_normal_scores)
            cb = pt.colorbar(_normal_scores, _normal_colors, lbl='weights',
                             ticklabels=_normal_ticks)

            # point to threshold location
            if infr.thresh is not None:
                xy = (1, infr.thresh)
                xytext = (2.5, .3 if infr.thresh < .5 else .7)
                cb.ax.annotate('threshold', xy=xy, xytext=xytext,
                               arrowprops=dict(
                                   alpha=.5, fc="0.6",
                                   connectionstyle="angle3,angleA=90,angleB=0"),)

        # infr.graph
        if graph.graph.get('dark_background', None):
            pt.dark_background(force=True)

        if pickable:
            fig = pt.gcf()
            fig.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

    def draw_aids(infr, aids, fnum=None):
        from ibeis.viz import viz_chip
        import plottool as pt
        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(infr.ibs, aids, fnum=fnum)
        return fig

    def start_qt_interface(infr, loop=True):
        import guitool as gt
        from ibeis.viz.viz_graph2 import AnnotGraphWidget
        from plottool import abstract_interaction
        import plottool as pt
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
        visual_edge_data = {k: v for k, v in all_edge_data.items()
                            if k in infr.visual_edge_attrs}
        edge_data = ut.delete_dict_keys(all_edge_data.copy(), infr.visual_edge_attrs)
        lines = []
        if visual:
            lines += [('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1))]
        lines += [('edge_data: ' + ut.repr2(edge_data, nl=1))]
        return '\n'.join(lines)

    show = show_graph


def on_pick(event, infr=None):
    import plottool as pt
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
