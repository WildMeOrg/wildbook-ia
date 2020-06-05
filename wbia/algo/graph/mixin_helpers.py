# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import networkx as nx
import operator
import numpy as np
import utool as ut
import vtool as vt
from wbia import constants as const
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from wbia.algo.graph.state import SAME, DIFF, NULL  # NOQA
from wbia.algo.graph.nx_utils import e_
from wbia.algo.graph import nx_utils as nxu
import six

print, rrr, profile = ut.inject2(__name__)


DEBUG_INCON = True


class AttrAccess(object):
    """ Contains non-core helper functions """

    def gen_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_attrs(
            infr.graph, key, nodes=nodes, default=default
        )

    def gen_edge_attrs(infr, key, edges=None, default=ut.NoParam, on_missing=None):
        """ maybe change to gen edge items """
        return ut.util_graph.nx_gen_edge_attrs(
            infr.graph, key, edges=edges, default=default, on_missing=on_missing
        )

    def gen_node_values(infr, key, nodes, default=ut.NoParam):
        return ut.util_graph.nx_gen_node_values(infr.graph, key, nodes, default=default)

    def gen_edge_values(
        infr, key, edges=None, default=ut.NoParam, on_missing='error', on_keyerr='default'
    ):
        return ut.util_graph.nx_gen_edge_values(
            infr.graph,
            key,
            edges,
            default=default,
            on_missing=on_missing,
            on_keyerr=on_keyerr,
        )

    def get_node_attrs(infr, key, nodes=None, default=ut.NoParam):
        """ Networkx node getter helper """
        return dict(infr.gen_node_attrs(key, nodes=nodes, default=default))

    def get_edge_attrs(infr, key, edges=None, default=ut.NoParam, on_missing=None):
        """ Networkx edge getter helper """
        return dict(
            infr.gen_edge_attrs(key, edges=edges, default=default, on_missing=on_missing)
        )

    def _get_edges_where(
        infr, key, op, val, edges=None, default=ut.NoParam, on_missing=None
    ):
        edge_to_attr = infr.gen_edge_attrs(
            key, edges=edges, default=default, on_missing=on_missing
        )
        return (e for e, v in edge_to_attr if op(v, val))

    def get_edges_where_eq(
        infr, key, val, edges=None, default=ut.NoParam, on_missing=None
    ):
        return infr._get_edges_where(
            key, operator.eq, val, edges=edges, default=default, on_missing=on_missing
        )

    def get_edges_where_ne(
        infr, key, val, edges=None, default=ut.NoParam, on_missing=None
    ):
        return infr._get_edges_where(
            key, operator.ne, val, edges=edges, default=default, on_missing=on_missing
        )

    def set_node_attrs(infr, key, node_to_prop):
        """ Networkx node setter helper """
        return nx.set_node_attributes(infr.graph, name=key, values=node_to_prop)

    def set_edge_attrs(infr, key, edge_to_prop):
        """ Networkx edge setter helper """
        return nx.set_edge_attributes(infr.graph, name=key, values=edge_to_prop)

    def get_edge_attr(infr, edge, key, default=ut.NoParam, on_missing='error'):
        """ single edge getter helper """
        return infr.get_edge_attrs(key, [edge], default=default, on_missing=on_missing)[
            edge
        ]

    def set_edge_attr(infr, edge, attr):
        """ single edge setter helper """
        for key, value in attr.items():
            infr.set_edge_attrs(key, {edge: value})

    def get_annot_attrs(infr, key, aids):
        """ Wrapper around get_node_attrs specific to annotation nodes """
        attr_list = list(infr.get_node_attrs(key, aids).values())
        return attr_list

    def edges(infr, data=False):
        if data:
            return ((e_(u, v), d) for u, v, d in infr.graph.edges(data=True))
        else:
            return (e_(u, v) for u, v in infr.graph.edges())

    def has_edge(infr, edge):
        return infr.graph.has_edge(*edge)
        # redge = edge[::-1]
        # flag = infr.graph.has_edge(*edge) or infr.graph.has_edge(*redge)
        # return flag

    def get_edge_data(infr, edge):
        return infr.graph.get_edge_data(*edge)

    def get_nonvisual_edge_data(infr, edge, on_missing='filter'):
        data = infr.get_edge_data(edge)
        if data is not None:
            data = ut.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        else:
            if on_missing == 'filter':
                data = None
            elif on_missing == 'default':
                data = {}
            elif on_missing == 'error':
                raise KeyError('graph does not have edge %r ' % (edge,))
        return data

    def get_edge_dataframe(infr, edges=None, all=False):
        import pandas as pd

        if edges is None:
            edges = infr.edges()
        edge_datas = {e: infr.get_nonvisual_edge_data(e) for e in edges}
        edge_datas = {
            e: {k: None for k in infr.feedback_data_keys} if d is None else d
            for e, d in edge_datas.items()
        }
        edge_df = pd.DataFrame.from_dict(edge_datas, orient='index')

        part = ['evidence_decision', 'meta_decision', 'tags', 'user_id']
        neworder = ut.partial_order(edge_df.columns, part)
        edge_df = edge_df.reindex(neworder, axis=1)
        if not all:
            edge_df = edge_df.drop(
                [
                    'review_id',
                    'timestamp',
                    'timestamp_s1',
                    'timestamp_c2',
                    'timestamp_c1',
                ],
                axis=1,
            )
        # pd.DataFrame.from_dict(edge_datas, orient='list')
        return edge_df

    def get_edge_df_text(infr, edges=None, highlight=True):
        df = infr.get_edge_dataframe(edges)
        df_str = df.to_string()
        if highlight:
            df_str = ut.highlight_regex(df_str, ut.regex_word(SAME), color='blue')
            df_str = ut.highlight_regex(df_str, ut.regex_word(POSTV), color='blue')
            df_str = ut.highlight_regex(df_str, ut.regex_word(DIFF), color='red')
            df_str = ut.highlight_regex(df_str, ut.regex_word(NEGTV), color='red')
            df_str = ut.highlight_regex(df_str, ut.regex_word(INCMP), color='yellow')
        return df_str


class Convenience(object):
    @staticmethod
    def e_(u, v):
        return e_(u, v)

    @property
    def pos_graph(infr):
        return infr.review_graphs[POSTV]

    @property
    def neg_graph(infr):
        return infr.review_graphs[NEGTV]

    @property
    def incomp_graph(infr):
        return infr.review_graphs[INCMP]

    @property
    def unreviewed_graph(infr):
        return infr.review_graphs[UNREV]

    @property
    def unknown_graph(infr):
        return infr.review_graphs[UNKWN]

    def print_graph_info(infr):
        print(ut.repr3(ut.graph_info(infr.simplify_graph())))

    def print_graph_connections(infr, label='orig_name_label'):
        """
        label = 'orig_name_label'
        """
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())
        print('CC info')
        for name, cc in label_to_nodes.items():
            print('\nname = %r' % (name,))
            edges = list(nxu.edges_between(infr.graph, cc))
            print(infr.get_edge_df_text(edges))

        print('CC pair info')
        for (n1, cc1), (n2, cc2) in it.combinations(label_to_nodes.items(), 2):
            if n1 == n2:
                continue
            print('\nname_pair = {}-vs-{}'.format(n1, n2))
            edges = list(nxu.edges_between(infr.graph, cc1, cc2))
            print(infr.get_edge_df_text(edges))

    def print_within_connection_info(infr, edge=None, cc=None, aid=None, nid=None):
        if edge is not None:
            aid, aid2 = edge
        if nid is not None:
            cc = infr.pos_graph._ccs[nid]
        if aid is not None:
            cc = infr.pos_graph.connected_to(aid)
        # subgraph = infr.graph.subgraph(cc)
        # list(nxu.complement_edges(subgraph))
        edges = list(nxu.edges_between(infr.graph, cc))
        print(infr.get_edge_df_text(edges))

    def pair_connection_info(infr, aid1, aid2):
        """
        Helps debugging when ibs.nids has info that annotmatch/staging do not

        Examples:
            >>> from wbia.algo.graph.mixin_helpers import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='GZ_Master1')
            >>> infr = wbia.AnnotInference(ibs, 'all', autoinit=True)
            >>> infr.reset_feedback('staging', apply=True)
            >>> infr.relabel_using_reviews(rectify=False)
            >>> aid1, aid2 = 1349, 3087
            >>> aid1, aid2 = 1535, 2549
            >>> infr.pair_connection_info(aid1, aid2)


            >>> aid1, aid2 = 4055, 4286
            >>> aid1, aid2 = 6555, 6882
            >>> aid1, aid2 = 712, 803
            >>> aid1, aid2 = 3883, 4220
            >>> infr.pair_connection_info(aid1, aid2)
        """

        nid1, nid2 = infr.pos_graph.node_labels(aid1, aid2)
        cc1 = infr.pos_graph.connected_to(aid1)
        cc2 = infr.pos_graph.connected_to(aid2)
        ibs = infr.ibs

        # First check directly relationships

        def get_aug_df(edges):
            df = infr.get_edge_dataframe(edges)
            if len(df):
                df.index.names = ('aid1', 'aid2')
                nids = np.array(
                    [infr.pos_graph.node_labels(u, v) for u, v in list(df.index)]
                )
                df = df.assign(nid1=nids.T[0], nid2=nids.T[1])
                part = ['nid1', 'nid2', 'evidence_decision', 'tags', 'user_id']
                neworder = ut.partial_order(df.columns, part)
                df = df.reindex(neworder, axis=1)
                df = df.drop(['review_id', 'timestamp'], axis=1)
            return df

        def print_df(df, lbl):
            df_str = df.to_string()
            df_str = ut.highlight_regex(df_str, ut.regex_word(str(aid1)), color='blue')
            df_str = ut.highlight_regex(df_str, ut.regex_word(str(aid2)), color='red')
            if nid1 not in {aid1, aid2}:
                df_str = ut.highlight_regex(
                    df_str, ut.regex_word(str(nid1)), color='darkblue'
                )
            if nid2 not in {aid1, aid2}:
                df_str = ut.highlight_regex(
                    df_str, ut.regex_word(str(nid2)), color='darkred'
                )
            print('\n\n=====')
            print(lbl)
            print('=====')
            print(df_str)

        print('================')
        print('Pair Connection Info')
        print('================')

        nid1_, nid2_ = ibs.get_annot_nids([aid1, aid2])
        print('AIDS        aid1, aid2 = %r, %r' % (aid1, aid2))
        print('INFR NAMES: nid1, nid2 = %r, %r' % (nid1, nid2))
        if nid1 == nid2:
            print('INFR cc = %r' % (sorted(cc1),))
        else:
            print('INFR cc1 = %r' % (sorted(cc1),))
            print('INFR cc2 = %r' % (sorted(cc2),))

        if (nid1 == nid2) != (nid1_ == nid2_):
            ut.cprint('DISAGREEMENT IN GRAPH AND DB', 'red')
        else:
            ut.cprint('GRAPH AND DB AGREE', 'green')

        print('IBS  NAMES: nid1, nid2 = %r, %r' % (nid1_, nid2_))
        if nid1_ == nid2_:
            print('IBS CC: %r' % (sorted(ibs.get_name_aids(nid1_)),))
        else:
            print('IBS CC1: %r' % (sorted(ibs.get_name_aids(nid1_)),))
            print('IBS CC2: %r' % (sorted(ibs.get_name_aids(nid2_)),))

        # Does this exist in annotmatch?
        in_am = ibs.get_annotmatch_rowid_from_undirected_superkey([aid1], [aid2])
        print('in_am = %r' % (in_am,))

        # Does this exist in staging?
        staging_rowids = ibs.get_review_rowids_from_edges([(aid1, aid2)])[0]
        print('staging_rowids = %r' % (staging_rowids,))

        if False:
            # Make absolutely sure
            stagedf = ibs.staging.get_table_as_pandas('reviews')
            aid_cols = ['annot_1_rowid', 'annot_2_rowid']
            has_aid1 = (stagedf[aid_cols] == aid1).any(axis=1)
            from_aid1 = stagedf[has_aid1]
            conn_aid2 = (from_aid1[aid_cols] == aid2).any(axis=1)
            print('# connections = %r' % (conn_aid2.sum(),))

        # Next check indirect relationships
        graph = infr.graph
        if cc1 != cc2:
            edge_df1 = get_aug_df(nxu.edges_between(graph, cc1))
            edge_df2 = get_aug_df(nxu.edges_between(graph, cc2))
            print_df(edge_df1, 'Inside1')

            print_df(edge_df2, 'Inside1')

            out_df1 = get_aug_df(nxu.edges_outgoing(graph, cc1))
            print_df(out_df1, 'Outgoing1')

            out_df2 = get_aug_df(nxu.edges_outgoing(graph, cc2))
            print_df(out_df2, 'Outgoing2')
        else:
            subgraph = infr.pos_graph.subgraph(cc1)
            print('Shortest path between endpoints')
            print(nx.shortest_path(subgraph, aid1, aid2))

        edge_df3 = get_aug_df(nxu.edges_between(graph, cc1, cc2))
        print_df(edge_df3, 'Between')

    def node_tag_hist(infr):
        tags_list = infr.ibs.get_annot_case_tags(infr.aids)
        tag_hist = ut.util_tags.tag_hist(tags_list)
        return tag_hist

    def edge_tag_hist(infr):
        tags_list = list(infr.gen_edge_values('tags', None))
        tag_hist = ut.util_tags.tag_hist(tags_list)
        # ut.util_tags.tag_coocurrence(tags_list)
        return tag_hist


@six.add_metaclass(ut.ReloadingMetaclass)
class DummyEdges(object):
    def ensure_mst(infr, label='name_label', meta_decision=SAME):
        """
        Ensures that all names are names are connected.

        Args:
            label (str): node attribute to use as the group id to form the mst.
            meta_decision (str): if specified adds clique edges as feedback
                items with this decision. Otherwise the edges are only
                explicitly added to the graph.  This makes feedback items with
                user_id=algo:mst and with a confidence of guessing.

        Ignore:
            annots = ibs.annots(infr.aids)
            def fix_name(n):
                import re
                n = re.sub('  *', ' ', n)
                return re.sub(' *-? *BBQ[0-9]*', '', n)

            ut.fix_embed_globals()
            new_names = [fix_name(n) for n in annots.names]
            set(new_names)

            annots.names = new_names

            infr.set_node_attrs('name_fix', ut.dzip(infr.aids, new_names))
            label = 'name_fix'
            infr.ensure_mst(label)

            infr.set_node_attrs('name_label', ut.dzip(infr.aids, annots.nids))

        Ignore:
            label = 'name_label'

        Doctest:
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=3, size=4)
            >>> assert infr.status()['nCCs'] == 3
            >>> infr.clear_edges()
            >>> assert infr.status()['nCCs'] == 12
            >>> infr.ensure_mst()
            >>> assert infr.status()['nCCs'] == 3

        Doctest:
            >>> from wbia.algo.graph.mixin_dynamic import *  # NOQA
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', 'all', autoinit=True)
            >>> infr.reset_feedback('annotmatch', apply=True)
            >>> assert infr.status()['nInconsistentCCs'] == 0
            >>> assert infr.status()['nCCs'] == 41
            >>> label = 'name_label'
            >>> new_edges = infr.find_mst_edges(label=label)
            >>> assert len(new_edges) == 0
            >>> infr.clear_edges()
            >>> assert infr.status()['nCCs'] == 119
            >>> infr.ensure_mst()
            >>> assert infr.status()['nCCs'] == 41

        """
        infr.print('ensure_mst', 1)
        new_edges = infr.find_mst_edges(label=label)
        # Add new MST edges to original graph
        infr.print('adding %d MST edges' % (len(new_edges)), 2)
        infr.add_feedback_from(
            new_edges,
            meta_decision=SAME,
            confidence=const.CONFIDENCE.CODE.GUESSING,
            user_id='algo:mst',
            verbose=False,
        )

    def ensure_cliques(infr, label='name_label', meta_decision=None):
        """
        Force each name label to be a clique.

        Args:
            label (str): node attribute to use as the group id to form the
                cliques.
            meta_decision (str): if specified adds clique edges as feedback
                items with this decision. Otherwise the edges are only
                explicitly added to the graph.

        Args:
            infr (?):
            label (str): (default = 'name_label')
            decision (str): (default = 'unreviewed')

        CommandLine:
            python -m wbia.algo.graph.mixin_helpers ensure_cliques

        Doctest:
            >>> from wbia.algo.graph.mixin_helpers import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> label = 'name_label'
            >>> infr = demo.demodata_infr(num_pccs=3, size=5)
            >>> print(infr.status())
            >>> assert infr.status()['nEdges'] < 33
            >>> infr.ensure_cliques()
            >>> print(infr.status())
            >>> assert infr.status()['nEdges'] == 33
            >>> assert infr.status()['nUnrevEdges'] == 12
            >>> assert len(list(infr.find_clique_edges(label))) > 0
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> assert infr.status()['nUnrevEdges'] == 0
            >>> assert len(list(infr.find_clique_edges(label))) == 0
        """
        infr.print('ensure_cliques', 1)
        new_edges = infr.find_clique_edges(label)
        infr.print('ensuring %d clique edges' % (len(new_edges)), 2)
        if meta_decision is None:
            infr.ensure_edges_from(new_edges)
        else:
            infr.add_feedback_from(
                new_edges,
                meta_decision=SAME,
                confidence=const.CONFIDENCE.CODE.GUESSING,
                user_id='algo:clique',
                verbose=False,
            )
        # infr.assert_disjoint_invariant()

    def ensure_full(infr):
        """
        Explicitly places all edges, but does not make any feedback items
        """
        infr.print('ensure_full with %d nodes' % (len(infr.graph)), 2)
        new_edges = list(nx.complement(infr.graph).edges())
        infr.ensure_edges_from(new_edges)

    def find_clique_edges(infr, label='name_label'):
        """
        Augmenting edges that would complete each the specified cliques.
        (based on the group inferred from `label`)

        Args:
            label (str): node attribute to use as the group id to form the
                cliques.
        """
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in it.combinations(nodes, 2):
                if infr.edge_decision(edge) == UNREV:
                    new_edges.append(edge)
                # if infr.has_edge(edge):
                # else:
                #     new_edges.append(edge)
        return new_edges

    @profile
    def find_mst_edges(infr, label='name_label'):
        """
        Returns edges to augment existing PCCs (by label) in order to ensure
        they are connected with positive edges.

        CommandLine:
            python -m wbia.algo.graph.mixin_helpers find_mst_edges --profile

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_helpers import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
            >>> infr = wbia.AnnotInference(ibs, 'all', autoinit=True)
            >>> label = 'orig_name_label'
            >>> label = 'name_label'
            >>> infr.find_mst_edges()
            >>> infr.ensure_mst()

        Ignore:
            old_mst_edges = [
                e for e, d in infr.edges(data=True)
                if d.get('user_id', None) == 'algo:mst'
            ]
            infr.graph.remove_edges_from(old_mst_edges)
            infr.pos_graph.remove_edges_from(old_mst_edges)
            infr.neg_graph.remove_edges_from(old_mst_edges)
            infr.incomp_graph.remove_edges_from(old_mst_edges)

        """
        # Find clusters by labels
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())

        weight_heuristic = infr.ibs is not None
        if weight_heuristic:
            annots = infr.ibs.annots(infr.aids)
            node_to_time = ut.dzip(annots, annots.time)
            node_to_view = ut.dzip(annots, annots.viewpoint_code)
            enabled_heuristics = {
                'view_weight',
                'time_weight',
            }

        def _heuristic_weighting(nodes, avail_uv):
            avail_uv = np.array(avail_uv)
            weights = np.ones(len(avail_uv))

            if 'view_weight' in enabled_heuristics:
                from vtool import _rhomb_dist

                view_edge = [(node_to_view[u], node_to_view[v]) for (u, v) in avail_uv]
                view_weight = np.array(
                    [_rhomb_dist.VIEW_CODE_DIST[(v1, v2)] for (v1, v2) in view_edge]
                )
                # Assume comparable by default and prefer undefined
                # more than probably not, but less than definately so.
                view_weight[np.isnan(view_weight)] = 1.5
                # Prefer viewpoint 10x more than time
                weights += 10 * view_weight

            if 'time_weight' in enabled_heuristics:
                # Prefer linking annotations closer in time
                times = ut.take(node_to_time, nodes)
                maxtime = vt.safe_max(times, fill=1, nans=False)
                mintime = vt.safe_min(times, fill=0, nans=False)
                time_denom = maxtime - mintime
                # Try linking by time for lynx data
                time_delta = np.array(
                    [abs(node_to_time[u] - node_to_time[v]) for u, v in avail_uv]
                )
                time_weight = time_delta / time_denom
                weights += time_weight

            weights = np.array(weights)
            weights[np.isnan(weights)] = 1.0

            avail = [(u, v, {'weight': w}) for (u, v), w in zip(avail_uv, weights)]
            return avail

        new_edges = []
        prog = ut.ProgIter(
            list(label_to_nodes.keys()),
            label='finding mst edges',
            enabled=infr.verbose > 0,
        )
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            if len(nodes) == 1:
                continue
            # We want to make this CC connected
            pos_sub = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = set(
                it.starmap(
                    e_,
                    it.chain(
                        nxu.edges_inside(infr.neg_graph, nodes),
                        nxu.edges_inside(infr.incomp_graph, nodes),
                        # nxu.edges_inside(infr.unknown_graph, nodes),
                    ),
                )
            )
            if len(impossible) == 0 and not weight_heuristic:
                # Simple mst augmentation
                aug_edges = list(nxu.k_edge_augmentation(pos_sub, k=1))
            else:
                complement = it.starmap(e_, nxu.complement_edges(pos_sub))
                avail_uv = [(u, v) for u, v in complement if (u, v) not in impossible]
                if weight_heuristic:
                    # Can do heuristic weighting to improve the MST
                    avail = _heuristic_weighting(nodes, avail_uv)
                else:
                    avail = avail_uv
                # print(len(pos_sub))
                try:
                    aug_edges = list(nxu.k_edge_augmentation(pos_sub, k=1, avail=avail))
                except nx.NetworkXUnfeasible:
                    print('Warning: MST augmentation is not feasible')
                    print('explicit negative edges might disconnect a PCC')
                    aug_edges = list(
                        nxu.k_edge_augmentation(pos_sub, k=1, avail=avail, partial=True)
                    )
            new_edges.extend(aug_edges)
        prog.ensure_newline()

        for edge in new_edges:
            assert not infr.graph.has_edge(*edge), 'alrady have edge={}'.format(edge)
        return new_edges

    def find_connecting_edges(infr):
        """
        Searches for a small set of edges, which if reviewed as positive would
        ensure that each PCC is k-connected.  Note that in somes cases this is
        not possible
        """
        label = 'name_label'
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())

        # k = infr.params['redun.pos']
        k = 1
        new_edges = []
        prog = ut.ProgIter(
            list(label_to_nodes.keys()),
            label='finding connecting edges',
            enabled=infr.verbose > 0,
        )
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            G = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = nxu.edges_inside(infr.neg_graph, nodes)
            impossible |= nxu.edges_inside(infr.incomp_graph, nodes)

            candidates = set(nx.complement(G).edges())
            candidates.difference_update(impossible)

            aug_edges = nxu.k_edge_augmentation(G, k=k, avail=candidates)
            new_edges += aug_edges
        prog.ensure_newline()
        return new_edges


class AssertInvariants(object):
    def assert_edge(infr, edge):
        import utool

        with utool.embed_on_exception_context:
            assert (
                edge[0] < edge[1]
            ), 'edge={} does not satisfy ordering constraint'.format(edge)

    def assert_invariants(infr, msg=''):
        infr.assert_disjoint_invariant(msg)
        infr.assert_union_invariant(msg)
        infr.assert_consistency_invariant(msg)
        infr.assert_recovery_invariant(msg)
        infr.assert_neg_metagraph()

    def assert_neg_metagraph(infr):
        """
        Checks that the negative metgraph is correctly book-kept.
        """
        # The total weight of all edges in the negative metagraph should equal
        # the total number of negative edges.
        neg_weight = sum(nx.get_edge_attributes(infr.neg_metagraph, 'weight').values())
        n_neg_edges = infr.neg_graph.number_of_edges()
        assert neg_weight == n_neg_edges

        # Self loops should correspond to the number of inconsistent components
        neg_self_loop_nids = sorted(
            [ne[0] for ne in list(infr.neg_metagraph.selfloop_edges())]
        )
        incon_nids = sorted(infr.nid_to_errors.keys())
        assert neg_self_loop_nids == incon_nids

    def assert_union_invariant(infr, msg=''):
        edge_sets = {
            key: set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        edge_union = set.union(*edge_sets.values())
        all_edges = set(it.starmap(e_, infr.graph.edges()))
        if edge_union != all_edges:
            print('ERROR STATUS DUMP:')
            print(ut.repr4(infr.status()))
            raise AssertionError(
                'edge sets must have full union. Found union=%d vs all=%d'
                % (len(edge_union), len(all_edges))
            )

    def assert_disjoint_invariant(infr, msg=''):
        # infr.print('assert_disjoint_invariant', 200)
        edge_sets = {
            key: set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        for es1, es2 in it.combinations(edge_sets.values(), 2):
            assert es1.isdisjoint(es2), 'edge sets must be disjoint'

    def assert_consistency_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_consistency_invariant', 200)
        if infr.params['inference.enabled']:
            incon_ccs = list(infr.inconsistent_components())
            if len(incon_ccs) > 0:
                raise AssertionError('The graph is not consistent. ' + msg)

    def assert_recovery_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_recovery_invariant', 200)
        inconsistent_ccs = list(infr.inconsistent_components())
        incon_cc = set(ut.flatten(inconsistent_ccs))  # NOQA
        # import utool
        # with utool.embed_on_exception_context:
        #     assert infr.recovery_cc.issuperset(incon_cc), 'diff incon'
        #     if False:
        #         # nid_to_cc2 = ut.group_items(
        #         #     incon_cc,
        #         #     map(pos_graph.node_label, incon_cc))
        #         infr.print('infr.recovery_cc = %r' % (infr.recovery_cc,))
        #         infr.print('incon_cc = %r' % (incon_cc,))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_helpers
        python -m wbia.algo.graph.mixin_helpers --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
