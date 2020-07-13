# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function
import networkx as nx
import utool as ut
import pandas as pd
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)

edges = {
    2234: {5383: {'decision': 'match', 'reviewed_tags': ['needswork']}},
    2265: {},
    2280: {},
    2654: {
        5334: {
            'decision': 'match',
            'reviewed_tags': ['needswork', 'viewpoint', 'correctable', 'orientation'],
        }
    },
    5334: {
        2654: {
            'decision': 'match',
            'reviewed_tags': ['needswork', 'viewpoint', 'correctable', 'orientation'],
        },
        5344: {'decision': 'match', 'reviewed_tags': []},
        5383: {'decision': 'match', 'reviewed_tags': []},
    },
    5338: {
        5344: {'decision': 'match', 'reviewed_tags': []},
        5383: {'decision': 'match', 'reviewed_tags': []},
    },
    5344: {
        5334: {'decision': 'match', 'reviewed_tags': []},
        5338: {'decision': 'match', 'reviewed_tags': []},
        5349: {'decision': 'match', 'reviewed_tags': []},
        5383: {'decision': 'match', 'reviewed_tags': []},
        5430: {'decision': 'match', 'reviewed_tags': []},
    },
    5349: {
        5344: {'decision': 'match', 'reviewed_tags': []},
        5399: {'decision': 'match', 'reviewed_tags': []},
    },
    5383: {
        2234: {'decision': 'match', 'reviewed_tags': ['needswork']},
        5334: {'decision': 'match', 'reviewed_tags': []},
        5338: {'decision': 'match', 'reviewed_tags': []},
        5344: {'decision': 'match', 'reviewed_tags': []},
        5430: {'decision': 'match', 'reviewed_tags': []},
    },
    5399: {5349: {'decision': 'match', 'reviewed_tags': []}},
    5430: {
        5344: {'decision': 'match', 'reviewed_tags': []},
        5383: {'decision': 'match', 'reviewed_tags': []},
    },
}

nodes = {
    2234: {'aid': 2234, 'name_label': 5977, 'orig_name_label': 5977},
    2265: {'aid': 2265, 'name_label': 5977, 'orig_name_label': 5977},
    2280: {'aid': 2280, 'name_label': 5977, 'orig_name_label': 5977},
    2654: {'aid': 2654, 'name_label': 5977, 'orig_name_label': 5977},
    5334: {'aid': 5334, 'name_label': 5977, 'orig_name_label': 5977},
    5338: {'aid': 5338, 'name_label': 5977, 'orig_name_label': 5977},
    5344: {'aid': 5344, 'name_label': 5977, 'orig_name_label': 5977},
    5349: {'aid': 5349, 'name_label': 5977, 'orig_name_label': 5977},
    5383: {'aid': 5383, 'name_label': 5977, 'orig_name_label': 5977},
    5399: {'aid': 5399, 'name_label': 5977, 'orig_name_label': 5977},
    5430: {'aid': 5430, 'name_label': 5977, 'orig_name_label': 5977},
}


graph = nx.Graph(edges)
graph.add_nodes_from(nodes.keys())

df = pd.DataFrame.from_dict(nodes, orient='index')
nx.set_node_attributes(
    graph, name='orig_name_label', values=ut.dzip(df['aid'], df['orig_name_label'])
)
nx.set_node_attributes(
    graph, name='name_label', values=ut.dzip(df['aid'], df['name_label'])
)

aug_graph = graph
node_to_label = nx.get_node_attributes(graph, 'name_label')


aid1, aid2 = 2265, 2280

label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())

aug_graph = graph.copy()

# remove cut edges from augmented graph
edge_to_iscut = nx.get_edge_attributes(aug_graph, 'is_cut')
cut_edges = [
    (u, v)
    for (u, v, d) in aug_graph.edges(data=True)
    if not (d.get('is_cut') or d.get('decision', 'unreviewed') in ['nomatch'])
]
cut_edges = [edge for edge, flag in edge_to_iscut.items() if flag]
aug_graph.remove_edges_from(cut_edges)


# Enumerate cliques inside labels
unflat_edges = [list(ut.itertwo(nodes)) for nodes in label_to_nodes.values()]
node_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]

# Remove candidate MST edges that exist in the original graph
orig_edges = list(aug_graph.edges())
candidate_mst_edges = [edge for edge in node_pairs if not aug_graph.has_edge(*edge)]
# randomness prevents chains and visually looks better
rng = np.random.RandomState(42)


def _randint():
    return 0
    return rng.randint(0, 100)


aug_graph.add_edges_from(candidate_mst_edges)
# Weight edges in aug_graph such that existing edges are chosen
# to be part of the MST first before suplementary edges.
nx.set_edge_attributes(
    aug_graph, name='weight', values={edge: 0.1 for edge in orig_edges}
)

try:
    # Try linking by time for lynx data
    nodes = list(set(ut.iflatten(candidate_mst_edges)))
    aids = ut.take(infr.node_to_aid, nodes)
    times = infr.ibs.annots(aids).time
    node_to_time = ut.dzip(nodes, times)
    time_deltas = np.array(
        [abs(node_to_time[u] - node_to_time[v]) for u, v in candidate_mst_edges]
    )
    # print('time_deltas = %r' % (time_deltas,))
    maxweight = vt.safe_max(time_deltas, nans=False, fill=0) + 1
    time_deltas[np.isnan(time_deltas)] = maxweight
    time_delta_weight = 10 * time_deltas / (time_deltas.max() + 1)
    is_comp = infr.guess_if_comparable(candidate_mst_edges)
    comp_weight = 10 * (1 - is_comp)
    extra_weight = comp_weight + time_delta_weight

    # print('time_deltas = %r' % (time_deltas,))
    nx.set_edge_attributes(
        aug_graph,
        name='weight',
        values={
            edge: 10.0 + extra for edge, extra in zip(candidate_mst_edges, extra_weight)
        },
    )
except Exception:
    print('FAILED WEIGHTING USING TIME')
    nx.set_edge_attributes(
        aug_graph,
        name='weight',
        values={edge: 10.0 + _randint() for edge in candidate_mst_edges},
    )
new_edges = []
for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
    mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
    # Only add edges not in the original graph
    for edge in mst_sub_graph.edges():
        if not graph.has_edge(*edge):
            new_edges.append(e_(*edge))
