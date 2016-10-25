# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt  # NOQA
import six
import networkx as nx
import plottool as pt
print, rrr, profile = ut.inject2(__name__)


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    return ut.dzip(a, b)


@six.add_metaclass(ut.ReloadingMetaclass)
class _AnnotInfrViz(object):
    """ contains plotting related code """

    def _get_truth_colors(infr):
        truth_colors = {
            # 'match': pt.TRUE_GREEN,
            'match': pt.TRUE_BLUE,
            'nomatch': pt.FALSE_RED,
            'notcomp': pt.YELLOW,
            'unreviewed': pt.UNKNOWN_PURP
        }
        return truth_colors

    def _get_cmap(infr):
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
        if infr.verbose >= 3:
            print('[infr] initialize_visual_node_attrs')
        import networkx as nx
        if graph is None:
            graph = infr.graph
        aid_to_node = infr.aid_to_node
        aid_list = list(aid_to_node.keys())
        annot_nodes = ut.take(aid_to_node, aid_list)
        chip_width = 256
        imgpath_list = infr.ibs.depc_annot.get('chips', aid_list, 'img',
                                               config=dict(dim_size=chip_width),
                                               read_extern=False)
        nx.set_node_attributes(graph, 'framewidth', 3.0)
        #nx.set_node_attributes(graph, 'framecolor', pt.DARK_BLUE)
        nx.set_node_attributes(graph, 'shape', _dz(annot_nodes, ['rect']))
        nx.set_node_attributes(graph, 'image', _dz(annot_nodes, imgpath_list))

    def get_colored_edge_weights(infr, graph=None):
        # Update color and linewidth based on scores/weight
        if graph is None:
            graph = infr.graph
        edges = list(infr.graph.edges())
        edge2_weight = nx.get_edge_attributes(infr.graph, infr.CUT_WEIGHT_KEY)
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
        return ['alpha', 'color', 'implicit', 'label', 'linestyle', 'lw',
                'pos', 'stroke', 'capstyle', 'hatch', 'style']

    @property
    def visual_edge_attrs_space(infr):
        """ attrs that pertain to edge positioning in a plot """
        return ['ctrl_pts', 'end_pt', 'head_lp', 'headlabel', 'lp', 'start_pt',
                'tail_lp', 'taillabel', 'zorder']

    @property
    def visual_node_attrs(infr):
        return ['color', 'framewidth', 'image', 'label',
                'pos', 'shape', 'size', 'height', 'width', 'zorder']

    def simplify_graph(infr, graph=None):
        if graph is None:
            graph = infr.graph
        s = graph.copy()
        ut.nx_delete_edge_attr(s, infr.visual_edge_attrs)
        ut.nx_delete_node_attr(s, infr.visual_node_attrs + ['pin'])
        return s

    def update_visual_attrs(infr, graph=None, show_cuts=False,
                            show_reviewed_cuts=True, only_reviewed=False,
                            mode=None):
        if infr.verbose >= 3:
            print('[infr] update_visual_attrs')
            print(' * show_cuts = %r' % (show_cuts,))
            print(' * show_reviewed_cuts = %r' % (show_reviewed_cuts,))
        if graph is None:
            graph = infr.graph

        # Ensure we are starting from a clean slate
        ut.nx_delete_edge_attr(graph, infr.visual_edge_attrs_appearance)

        # Set annotation node labels
        node_to_aid = nx.get_node_attributes(graph, 'aid')
        node_to_nid = nx.get_node_attributes(graph, 'name_label')
        annotnode_to_label = {
            #node: '%d:aid=%r' % (node, aid)
            node: 'aid=%r\nnid=%r' % (aid, node_to_nid[node])
            for node, aid in node_to_aid.items()
        }
        nx.set_node_attributes(graph, 'label', annotnode_to_label)

        # Color nodes by name label
        ut.color_nodes(graph, labelattr='name_label', sat_adjust=-.8)

        reviewed_states = nx.get_edge_attributes(graph, 'reviewed_state')

        SPLIT_MODE = mode == 'split'
        if not SPLIT_MODE:
            # Base color on edge weights
            edges, edge_weights, edge_colors = infr.get_colored_edge_weights(graph)
            #nx.set_edge_attributes(graph, 'len', _dz(edges, [10]))
            nx.set_edge_attributes(graph, 'color', _dz(edges, edge_colors))
            # minlw, maxlw = 1.5, 4
            # lw = ((maxlw - minlw) * edge_weights + minlw)
            # nx.set_edge_attributes(graph, 'lw', _dz(edges, lw))
            # nx.set_edge_attributes(graph, 'lw', _dz(edges, [1.5]))

            # Base line width on if reviewed
            reviewed_states_full = infr.get_edge_attrs('reviewed_state',
                                                       default='unreviewed')
            edge_to_lw = {
                edge: 2.0 if state == 'unreviewed' else 5.0
                for edge, state in reviewed_states_full.items()
            }
            nx.set_edge_attributes(graph, 'lw', edge_to_lw)

            # Mark reviewed edges witha stroke
            # truth_colors = infr._get_truth_colors()
            edge_to_stroke = {
                # edge: {'linewidth': 3, 'foreground': truth_colors[state]}
                # edge: {'linewidth': 3, 'foreground': pt.WHITE}
                edge: {'linewidth': 3, 'foreground': pt.BLACK}
                for edge, state in reviewed_states.items()
            }
            nx.set_edge_attributes(graph, 'stroke', edge_to_stroke)

            # nx.set_edge_attributes(graph, 'hatch', {
            #     edge: 'O' for edge, state in
            #     reviewed_states.items()})

            nx.set_edge_attributes(graph, 'capstyle', {
                edge: 'round' for edge, state in
                reviewed_states.items()})

            # Mark edges that might be splits with strokes
            # possible_split_edges = infr.find_possible_binary_splits()
            # edge_to_stroke = {
            #     edge: {'linewidth': 3, 'foreground': pt.ORANGE}
            #     for edge in ut.unique(possible_split_edges)
            # }
            # nx.set_edge_attributes(graph, 'stroke', edge_to_stroke)
        else:
            # Mark reviewed edges with a color
            truth_colors = infr._get_truth_colors()
            edge_to_color = {
                edge: truth_colors[state]
                for edge, state in reviewed_states.items()
            }
            nx.set_edge_attributes(graph, 'color', edge_to_color)

            # Mark edges that might be splits with strokes
            possible_split_edges = infr.find_possible_binary_splits()
            edge_to_stroke = {
                edge: {'linewidth': 3, 'foreground': pt.ORANGE}
                for edge in ut.unique(possible_split_edges)
            }
            nx.set_edge_attributes(graph, 'stroke', edge_to_stroke)

        # Mark edges that might be splits with strokes
        edge_to_split = nx.get_edge_attributes(infr.graph, 'splitcase')
        edge_to_stroke = {
            edge: {'linewidth': 5, 'foreground': pt.ORANGE}
            for edge, split in edge_to_split.items() if split
        }
        nx.set_edge_attributes(graph, 'stroke', edge_to_stroke)

        # Are cuts visible or invisible?
        edge_to_cut = nx.get_edge_attributes(graph, 'is_cut')
        cut_edges = [edge for edge, cut in edge_to_cut.items() if cut]
        nx.set_edge_attributes(graph, 'implicit', _dz(cut_edges, [True]))
        nx.set_edge_attributes(graph, 'linestyle', _dz(cut_edges, ['dashed']))

        # Non-matching edges should not impose a constraint on the graph layout
        nonmatch_edges = {edge: state for edge, state in reviewed_states.items()
                          if state == 'nomatch'}
        nx.set_edge_attributes(graph, 'implicit', _dz(nonmatch_edges, [True]))

        edges = list(graph.edges())
        reviewed_edges = list(reviewed_states.keys())
        nx.set_edge_attributes(graph, 'alpha', _dz(reviewed_edges, [1.0]))

        edge_to_infered_review = nx.get_edge_attributes(graph, 'infered_review')
        infered_edges = [edge for edge, state in edge_to_infered_review.items()
                         if state in ['match', 'nomatch']]
        # infered_edges = graph.edges()
        # nx.set_edge_attributes(graph, 'shadow', _dz(infered_edges, [True]))
        # nx.set_edge_attributes(graph, 'sketch', _dz(infered_edges, [True]))
        nx.set_edge_attributes(
            graph, 'sketch', _dz(infered_edges, [
                dict(scale=10.0, length=64.0, randomness=None)]
                # dict(scale=3.0, length=18.0, randomness=None)]
            ))
        hide_infered = True
        if hide_infered:
            # Infered edges are hidden
            nx.set_edge_attributes(
                graph, 'style', _dz(infered_edges, ['invis']))

        if only_reviewed:
            # only reviewed edges contribute
            unreviewed_edges = ut.setdiff(edges, reviewed_edges)
            nx.set_edge_attributes(graph, 'implicit', _dz(unreviewed_edges, [True]))
            nx.set_edge_attributes(graph, 'style', _dz(unreviewed_edges, ['invis']))

        if show_cuts or show_reviewed_cuts:
            if not show_cuts:
                nonfeedback_cuts = ut.setdiff(cut_edges, reviewed_edges)
                nx.set_edge_attributes(graph, 'style', _dz(nonfeedback_cuts, ['invis']))
        else:
            nx.set_edge_attributes(graph, 'style', _dz(cut_edges, ['invis']))

        # Make dummy edges more transparent
        edge_to_isdummy = nx.get_edge_attributes(graph, '_dummy_edge')
        dummy_edges = [edge for edge, flag in edge_to_isdummy.items() if flag]
        nx.set_edge_attributes(graph, 'alpha', _dz(dummy_edges, [.5]))

        nodes = list(graph.nodes())
        nx.set_node_attributes(graph, 'zorder', _dz(nodes, [10]))
        nx.set_edge_attributes(graph, 'zorder', _dz(edges, [0]))

        # update the positioning layout
        layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)
        pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    def show_graph(infr, use_image=False, only_reviewed=False, show_cuts=False,
                   mode=None, with_colorbar=False, **kwargs):
        kwargs['fontsize'] = kwargs.get('fontsize', 8)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infr.update_visual_attrs(only_reviewed=only_reviewed,
                                     show_cuts=show_cuts, mode=mode)
            graph = infr.graph
            plotinfo = pt.show_nx(graph, layout='custom', as_directed=False,
                                  modify_ax=False, use_image=use_image, verbose=0,
                                  **kwargs)
            plotinfo  # NOQA
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
                ta = cb.ax.annotate('threshold', xy=xy, xytext=xytext,
                                    arrowprops=dict(
                                        alpha=.5, fc="0.6",
                                        connectionstyle="angle3,angleA=90,angleB=0"),)
                ta  # NOQA

    show = show_graph
