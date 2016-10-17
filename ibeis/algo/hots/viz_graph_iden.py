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
    if len(a) == 0 and len(b) == 1:
        # This introduces a corner case
        b = []
    elif len(b) == 1 and len(a) > 1:
        b = b * len(a)
    assert len(a) == len(b), 'out of alignment a=%r, b=%r' % (a, b)
    return dict(zip(a, b))


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInferenceVisualization(object):
    """ contains plotting related code """

    truth_colors = {
        'match': pt.TRUE_GREEN,
        #'match': pt.TRUE_BLUE,
        'nomatch': pt.FALSE_RED,
        'notcomp': pt.YELLOW,
        'unreviewed': pt.UNKNOWN_PURP
    }

    def initialize_visual_node_attrs(infr, graph=None):
        if infr.verbose:
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
        cmap_ = 'viridis'
        cmap_ = 'plasma'
        #cmap_ = pt.plt.get_cmap(cmap_)
        weights[np.isnan(weights)] = infr.thresh
        #colors = pt.scores_to_color(weights, cmap_=cmap_, logscale=True)
        colors = pt.scores_to_color(weights, cmap_=cmap_, score_range=(0, 1),
                                    logscale=False)
        return colors

    @property
    def visual_edge_attrs(infr):
        return ['implicit', 'style', 'tail_lp', 'taillabel', 'label', 'lp',
                'headlabel', 'linestyle', 'color', 'stroke', 'lw', 'end_pt',
                'start_pt', 'head_lp', 'alpha', 'ctrl_pts', 'pos', 'zorder']

    @property
    def visual_node_attrs(infr):
        return ['color', 'framewidth', 'image', 'label',
                'pos', 'shape', 'size', 'height', 'width', 'zorder']

    def simplify_graph(infr, graph):
        s = graph.copy()
        for attr in infr.visual_edge_attrs:
            ut.nx_delete_edge_attr(s, attr)
        for attr in infr.visual_node_attrs:
            ut.nx_delete_node_attr(s, attr)
        return s

    def update_visual_attrs(infr, graph=None, show_cuts=False,
                            show_reviewed_cuts=True, only_reviewed=False,
                            mode=None):
        if infr.verbose:
            print('[infr] update_visual_attrs')
        #edge2_weight = nx.get_edge_attributes(infr.graph, 'score')
        if graph is None:
            # Hack for name_graph
            graph = infr.graph
        ut.nx_delete_edge_attr(graph, 'style')
        ut.nx_delete_edge_attr(graph, 'implicit')
        ut.nx_delete_edge_attr(graph, 'color')
        ut.nx_delete_edge_attr(graph, 'lw')
        ut.nx_delete_edge_attr(graph, 'stroke')
        ut.nx_delete_edge_attr(graph, 'alpha')
        ut.nx_delete_edge_attr(graph, 'linestyle')
        ut.nx_delete_edge_attr(graph, 'label')

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
        ut.color_nodes(graph, labelattr='name_label')

        reviewed_states = nx.get_edge_attributes(graph, 'reviewed_state')

        SPLIT_MODE = mode == 'split'
        if not SPLIT_MODE:
            # Update color and linewidth based on scores/weight
            edges, edge_weights, edge_colors = infr.get_colored_edge_weights(graph)
            #nx.set_edge_attributes(graph, 'len', _dz(edges, [10]))
            nx.set_edge_attributes(graph, 'color', _dz(edges, edge_colors))
            minlw, maxlw = .5, 4
            lw = ((maxlw - minlw) * edge_weights + minlw)
            nx.set_edge_attributes(graph, 'lw', _dz(edges, lw))

            # Mark reviewed edges witha stroke
            edge_to_stroke = {
                edge: {'linewidth': 3, 'foreground': infr.truth_colors[state]}
                for edge, state in reviewed_states.items()
            }
            nx.set_edge_attributes(graph, 'stroke', edge_to_stroke)
        else:
            # Mark reviewed edges with a color
            edge_to_color = {
                edge: infr.truth_colors[state]
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

        # Are cuts visible or invisible?
        edge2_cut = nx.get_edge_attributes(graph, 'is_cut')
        cut_edges = [edge for edge, cut in edge2_cut.items() if cut]
        nx.set_edge_attributes(graph, 'implicit', _dz(cut_edges, [True]))
        if infr.verbose:
            print('show_cuts = %r' % (show_cuts,))
            print('show_reviewed_cuts = %r' % (show_reviewed_cuts,))
        nx.set_edge_attributes(graph, 'linestyle', _dz(cut_edges, ['dashed']))

        # Non-matching edges should not impose a constraint on the graph layout
        nonmatch_edges = {edge: state for edge, state in reviewed_states.items()
                          if state == 'nomatch'}
        nx.set_edge_attributes(graph, 'implicit', _dz(nonmatch_edges, [True]))

        edges = list(graph.edges())
        if only_reviewed:
            # only reviewed edges contribute
            unreviewed_edges = ut.setdiff(edges, reviewed_states.keys())
            nx.set_edge_attributes(graph, 'implicit', _dz(unreviewed_edges, [True]))
            nx.set_edge_attributes(graph, 'style', _dz(unreviewed_edges, ['invis']))

        if show_cuts or show_reviewed_cuts:
            if not show_cuts:
                nonfeedback_cuts = ut.setdiff(cut_edges, reviewed_states.keys())
                nx.set_edge_attributes(graph, 'style', _dz(nonfeedback_cuts, ['invis']))
        else:
            nx.set_edge_attributes(graph, 'style', _dz(cut_edges, ['invis']))

        # Make MST edge have more alpha
        edge_to_ismst = nx.get_edge_attributes(graph, '_mst_edge')
        mst_edges = [edge for edge, flag in edge_to_ismst.items() if flag]
        nx.set_edge_attributes(graph, 'alpha', _dz(mst_edges, [.5]))

        nodes = list(graph.nodes())
        nx.set_node_attributes(graph, 'zorder', _dz(nodes, [10]))
        nx.set_edge_attributes(graph, 'zorder', _dz(edges, [0]))

        # update the positioning layout
        layoutkw = dict(
            prog='neato',
            #defaultdist=100,
            splines='spline',
            sep=10 / 72,
            #esep=10 / 72
        )
        pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    def show_graph(infr, use_image=False, only_reviewed=False,
                   show_cuts=False, mode=None):
        infr.update_visual_attrs(only_reviewed=only_reviewed,
                                 show_cuts=show_cuts, mode=mode)
        graph = infr.graph
        plotinfo = pt.show_nx(graph, layout='custom', as_directed=False,
                              modify_ax=False, use_image=use_image, verbose=0)
        pt.zoom_factory()
        pt.pan_factory(pt.gca())

        with_colorbar = False
        if with_colorbar:
            # Draw a colorbar
            xy = (1, infr.thresh)
            xytext = (2.5, .3 if infr.thresh < .5 else .7)

            _normal_ticks = np.linspace(0, 1, num=11)
            _normal_scores = np.linspace(0, 1, num=500)
            _normal_colors = infr.get_colored_weights(_normal_scores)
            cb = pt.colorbar(_normal_scores, _normal_colors, lbl='weights',
                             ticklabels=_normal_ticks)
            ta = cb.ax.annotate('threshold', xy=xy, xytext=xytext,
                                arrowprops=dict(
                                    alpha=.5, fc="0.6",
                                    connectionstyle="angle3,angleA=90,angleB=0"),)
            #return cb, ta
            plotinfo, ta, cb
            #return plotinfo
