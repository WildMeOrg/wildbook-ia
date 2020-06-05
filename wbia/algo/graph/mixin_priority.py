# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import networkx as nx
from wbia import constants as const
from wbia.algo.graph import nx_utils as nxu
from wbia.algo.graph.state import POSTV, NEGTV
from wbia.algo.graph.state import SAME, DIFF, NULL  # NOQA
from wbia.algo.graph import mixin_loops

print, rrr, profile = ut.inject2(__name__)


ENABLE_PRIORITY_PCC_CORRECTION = True


class Priority(object):
    """
    Handles prioritization of edges for review.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.mixin_priority import *  # NOQA
        >>> from wbia.algo.graph import demo
        >>> infr = demo.demodata_infr(num_pccs=20)
    """

    def remaining_reviews(infr):
        assert infr.queue is not None
        return len(infr.queue)

    def _pop(infr, *args):
        """ Wraps queue so ordering is determenistic """
        (e, (p, _)) = infr.queue.pop(*args)
        return (e, -p)

    def _push(infr, edge, priority):
        """ Wraps queue so ordering is determenistic """
        PSEUDO_RANDOM_TIEBREAKER = False
        if PSEUDO_RANDOM_TIEBREAKER:
            # Make it so tiebreakers have a pseudo-random order
            chaotic = int.from_bytes(ut.digest_data(edge, alg='sha1')[:8], 'big')
            tiebreaker = (chaotic,) + edge
        else:
            tiebreaker = edge
        infr.assert_edge(edge)

        if mixin_loops.PRINCETON_KAIA_EDGE_LIST is not None:
            # print('[Priority._push] FILTERING EDGES FOR KAIA')
            # Sanity check, make sure that one of the edges is in the tier 1 dataset
            include_filter_set = set(mixin_loops.PRINCETON_KAIA_EDGE_LIST)

            u, v = edge
            if u not in include_filter_set and v not in include_filter_set:
                return

        # tiebreaker = (chaotic(chaotic(u) + chaotic(v)), u, v)
        infr.queue[edge] = (-priority, tiebreaker)

    def _peek_many(infr, n):
        """ Wraps queue so ordering is determenistic """
        return [(k, -p) for (k, (p, _)) in infr.queue.peek_many(n)]

    def _remove_edge_priority(infr, edges):
        if infr.queue is None:
            return
        edges_ = [edge for edge in edges if edge in infr.queue]
        if len(edges_) > 0:
            infr.print('removed priority from {} edges'.format(len(edges_)), 5)
            infr.queue.delete_items(edges_)

    def _reinstate_edge_priority(infr, edges):
        if infr.queue is None:
            return
        edges_ = [edge for edge in edges if edge not in infr.queue]
        if len(edges_) > 0:
            # TODO: use whatever the current metric is
            metric = 'prob_match'
            infr.print('reprioritize {} edges'.format(len(edges_)), 5)
            priorities = infr.gen_edge_values(metric, edges_, default=1e-9)
            for edge, priority in zip(edges_, priorities):
                infr._push(edge, priority)

    def _increase_priority(infr, edges, amount=10):
        if infr.queue is None:
            return
        infr.print('increase priority of {} edges'.format(len(edges)), 5)
        metric = 'prob_match'
        priorities = infr.gen_edge_values(metric, edges, default=1e-9)
        for edge, base in zip(edges, priorities):
            infr.push(edge, base + amount)

    def remove_internal_priority(infr, cc):
        if infr.queue is not None:
            infr._remove_edge_priority(nxu.edges_inside(infr.graph, cc))

    def remove_external_priority(infr, cc):
        if infr.queue is not None:
            infr._remove_edge_priority(nxu.edges_outgoing(infr.graph, cc))

    def remove_between_priority(infr, cc1, cc2):
        if infr.queue is not None:
            infr._remove_edge_priority(nxu.edges_cross(infr.graph, cc1, cc2))

    def reinstate_between_priority(infr, cc1, cc2):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = nxu.edges_cross(infr.unreviewed_graph, cc1, cc2)
            infr._reinstate_edge_priority(edges)

    def reinstate_internal_priority(infr, cc):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = nxu.edges_inside(infr.unreviewed_graph, cc)
            infr._reinstate_edge_priority(edges)

    def reinstate_external_priority(infr, cc):
        if infr.queue is not None:
            # Reinstate the appropriate edges into the queue
            edges = nxu.edges_outgoing(infr.unreviewed_graph, cc)
            infr._reinstate_edge_priority(edges)

    def _correct_priorities(infr, edge, priority):
        corrected_priority = None

        if not ENABLE_PRIORITY_PCC_CORRECTION:
            return corrected_priority

        if priority < 10:
            try:
                primary_task = 'match_state'
                decision_probs = infr.task_probs[primary_task][edge]
            except KeyError:
                decision_probs = None

            if decision_probs is not None:
                u, v = edge
                nid1, nid2 = infr.node_labels(u, v)
                if nid1 == nid2:
                    if priority != decision_probs[NEGTV]:
                        corrected_priority = decision_probs[NEGTV]
                else:
                    if priority != decision_probs[POSTV]:
                        corrected_priority = decision_probs[POSTV]

        return corrected_priority

    @profile
    def prioritize(
        infr, metric=None, edges=None, scores=None, force_inconsistent=True, reset=False
    ):
        """
        Adds edges to the priority queue

        Doctest:
            >>> from wbia.algo.graph.mixin_priority import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=7, size=5)
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> # Add a negative edge inside a PCC
            >>> ccs = list(infr.positive_components())
            >>> edge1 = tuple(list(ccs[0])[0:2])
            >>> edge2 = tuple(list(ccs[1])[0:2])
            >>> infr.add_feedback(edge1, NEGTV)
            >>> infr.add_feedback(edge2, NEGTV)
            >>> num_new = infr.prioritize(reset=True)
            >>> order = infr._peek_many(np.inf)
            >>> scores = ut.take_column(order, 1)
            >>> assert scores[0] > 10
            >>> assert len(scores) == num_new, 'should prioritize two hypotheis edges'
            >>> unrev_edges = set(infr.unreviewed_graph.edges())
            >>> err_edges = set(ut.flatten(infr.nid_to_errors.values()))
            >>> edges = set(list(unrev_edges - err_edges)[0:2])
            >>> edges.update(list(err_edges)[0:2])
            >>> num_new = infr.prioritize(edges=edges, reset=True)
            >>> order2 = infr._peek_many(np.inf)
            >>> scores2 = np.array(ut.take_column(order2, 1))
            >>> assert np.all(scores2[0:2] > 10)
            >>> assert np.all(scores2[2:] < 10)

        Example:
            import wbia
            infr = wbia.AnnotInference('PZ_MTEST', aids='all', autoinit='staging')
            infr.verbose = 1000
            infr.load_published()
            incon_edges = set(ut.iflatten(infr.nid_to_errors.values()))
            assert len(incon_edges) > 0
            edges = list(infr.find_pos_redun_candidate_edges())
            assert len(set(incon_edges).intersection(set(edges))) == 0
            infr.add_candidate_edges(edges)

            infr.prioritize()
            print(ut.repr4(infr.status()))
        """
        if reset or infr.queue is None:
            infr.queue = ut.PriorityQueue()
        low = 1e-9
        if metric is None:
            metric = 'prob_match'

        # If edges are not explicilty specified get unreviewed and error edges
        # that are not redundant
        if edges is None:
            if scores is not None:
                raise ValueError('must provide edges with scores')
            unrev_edges = infr.unreviewed_graph.edges()
            edges = set(infr.filter_edges_flagged_as_redun(unrev_edges))

        infr.print('ensuring {} edge(s) get priority'.format(len(edges)), 5)

        if infr.params['inference.enabled'] and force_inconsistent:
            # Ensure that maybe_error edges are always prioritized
            maybe_error_edges = set(infr.maybe_error_edges())
            extra_edges = set(maybe_error_edges).difference(set(edges))
            extra_edges = list(extra_edges)
            infr.print(
                'ensuring {} inconsistent edge(s) get priority'.format(len(extra_edges)),
                5,
            )

            if scores is not None:
                pgen = list(infr.gen_edge_values(metric, extra_edges, default=low))
                extra_scores = np.array(pgen)
                extra_scores[np.isnan(extra_scores)] = low

                scores = ut.aslist(scores) + ut.aslist(extra_scores)
            edges = ut.aslist(edges) + extra_edges

        # Ensure edges are in some arbitrary order
        edges = list(edges)

        # Ensure given scores do not have nan values
        if scores is None:
            pgen = infr.gen_edge_values(metric, edges, default=low)
            priorities = np.array(list(pgen))
            priorities[np.isnan(priorities)] = low
        else:
            priorities = np.asarray(scores)
            if np.any(np.isnan(priorities)):
                priorities[np.isnan(priorities)] = low

        if infr.params['inference.enabled']:
            # Increase priority of any flagged maybe_error edges
            err_flags = [e in maybe_error_edges for e in edges]
            priorities[err_flags] += 10

        # Push new items into the priority queue
        num_new = 0
        for edge, priority in zip(edges, priorities):
            if edge not in infr.queue:
                num_new += 1
            corrected_priority = infr._correct_priorities(edge, priority)
            if corrected_priority is not None:
                priority = corrected_priority
            infr._push(edge, priority)

        infr.print('added %d edges to the queue' % (num_new,), 1)
        return num_new

    def push(infr, edge, priority=None):
        """
        Push an edge back onto the queue
        """
        if priority is None:
            priority = 'prob_match'
        if isinstance(priority, six.string_types):
            prob_match = infr.get_edge_attr(edge, priority, default=1e-9)
            priority = prob_match
        corrected_priority = infr._correct_priorities(edge, priority)
        if corrected_priority is not None:
            priority = corrected_priority
        # Use edge-nids to break ties for determenistic behavior
        infr._push(edge, priority)

    @profile
    def pop(infr):
        """
        Main interface to the priority queue used by the algorithm loops.
        Pops the highest priority edge from the queue.
        """
        # The outer loop simulates recursive calls without using the stack
        SIZE_THRESH_ENABLED = False
        SIZE_THRESH = 10

        while True:
            try:
                edge, priority = infr._pop()
            except IndexError:
                raise StopIteration('no more to review!')
            else:
                # Re-prioritize positive or negative relative to PCCs
                corrected_priority = infr._correct_priorities(edge, priority)
                if corrected_priority is not None:
                    infr.push(edge, corrected_priority)
                    continue

                if SIZE_THRESH_ENABLED and infr.phase > 0:
                    u, v = edge
                    nid1, nid2 = infr.node_labels(u, v)
                    cc1 = infr.pos_graph.component(nid1)
                    cc2 = infr.pos_graph.component(nid2)
                    size1 = len(cc1)
                    size2 = len(cc2)

                    if nid1 == nid2 and not (size1 > SIZE_THRESH and size2 > SIZE_THRESH):
                        continue

                if infr.params['redun.enabled']:
                    u, v = edge
                    nid1, nid2 = infr.node_labels(u, v)
                    pos_graph = infr.pos_graph
                    pos_graph[nid1]
                    if nid1 == nid2:
                        if nid1 not in infr.nid_to_errors:
                            # skip edges that increase local connectivity beyond
                            # redundancy thresholds.
                            k_pos = infr.params['redun.pos']
                            # Much faster to compute local connectivity on subgraph
                            cc = infr.pos_graph.component(nid1)
                            pos_subgraph = infr.pos_graph.subgraph(cc)
                            pos_conn = nx.connectivity.local_edge_connectivity(
                                pos_subgraph, u, v, cutoff=k_pos
                            )
                            # Compute local connectivity
                            if pos_conn >= k_pos:
                                continue  # Loop instead of recursion
                                # return infr.pop()
                if infr.params['queue.conf.thresh'] is not None:
                    # Ignore reviews that would re-enforce a relationship that
                    # already has high confidence.
                    thresh_code = infr.params['queue.conf.thresh']
                    thresh = const.CONFIDENCE.CODE_TO_INT[thresh_code]
                    # FIXME: at the time of writing a hard coded priority of 10
                    # or more means that this is part of an inconsistency
                    if priority < 10:
                        u, v = edge
                        nid1, nid2 = infr.node_labels(u, v)
                        if nid1 == nid2:
                            if infr.confidently_connected(u, v, thresh):
                                infr.pop()
                        else:
                            if infr.confidently_separated(u, v, thresh):
                                infr.pop()

                if getattr(infr, 'fix_mode_split', False):
                    # only checking edges within a name
                    nid1, nid2 = infr.pos_graph.node_labels(*edge)
                    if nid1 != nid2:
                        continue  # Loop instead of recursion
                        # return infr.pop()
                if getattr(infr, 'fix_mode_merge', False):
                    # only checking edges within a name
                    nid1, nid2 = infr.pos_graph.node_labels(*edge)
                    if nid1 == nid2:
                        continue  # Loop instead of recursive (infr.pop())
                if getattr(infr, 'fix_mode_predict', False):
                    # No longer needed.
                    pred = infr.get_edge_data(edge).get('pred', None)
                    # only report cases where the prediction differs
                    # FIXME: at the time of writing a hard coded priority of 10
                    # or more means that this is part of an inconsistency
                    if priority < 10:
                        nid1, nid2 = infr.node_labels(*edge)
                        if nid1 == nid2:
                            u, v = edge
                            # Don't re-review confident CCs
                            thresh = const.CONFIDENCE.CODE_TO_INT['pretty_sure']
                            if infr.confidently_connected(u, v, thresh):
                                continue  # Loop instead of recursive (infr.pop())
                        if pred == POSTV and nid1 == nid2:
                            continue  # Loop instead of recursive (infr.pop())
                        if pred == NEGTV and nid1 != nid2:
                            continue  # Loop instead of recursive (infr.pop())
                    else:
                        print('in error recover mode')
                infr.assert_edge(edge)
                return edge, priority

    def peek(infr):
        return infr.peek_many(n=1)[0]

    def peek_many(infr, n):
        """
        Peeks at the top n edges in the queue.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.graph.mixin_priority import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_infr(num_pccs=7, size=5)
            >>> infr.refresh_candidate_edges()
            >>> infr.peek_many(50)
        """
        # Do pops that may invalidate pos redun edges internal to PCCs
        items = []
        count = 0
        # Pop the top n edges off the queue
        while len(infr.queue) > 0 and count < n:
            items.append(infr.pop())
            count += 1
        # Push them back because we are just peeking
        # (although we may have invalidated things based on local connectivity)
        for edge, priority in items:
            infr.push(edge, priority)
        return items

    def confidently_connected(infr, u, v, thresh=2):
        """
        Checks if u and v are conneted by edges above a confidence threshold
        """

        def satisfied(G, child, edge):
            decision = infr.edge_decision(edge)
            if decision != POSTV:
                return False
            data = G.get_edge_data(*edge)
            conf = data.get('confidence', 'unspecified')
            conf_int = const.CONFIDENCE.CODE_TO_INT[conf]
            conf_int = 0 if conf_int is None else conf_int
            return conf_int >= thresh

        for node in ut.bfs_conditional(
            infr.graph, u, yield_if=satisfied, continue_if=satisfied
        ):
            if node == v:
                return True
        return False

    def confidently_separated(infr, u, v, thresh=2):
        """
        Checks if u and v are conneted by edges above a confidence threshold

        Doctest:
            >>> from wbia.algo.graph.mixin_priority import *  # NOQA
            >>> from wbia.algo.graph import demo
            >>> infr = demo.make_demo_infr(ccs=[(1, 2), (3, 4), (5, 6), (7, 8)])
            >>> infr.add_feedback((1, 5), NEGTV)
            >>> infr.add_feedback((5, 8), NEGTV)
            >>> infr.add_feedback((6, 3), NEGTV)
            >>> u, v = (1, 4)
            >>> thresh = 0
            >>> assert not infr.confidently_separated(u, v, thresh)
            >>> infr.add_feedback((2, 3), NEGTV)
            >>> assert not infr.confidently_separated(u, v, thresh)
        """

        def can_cross(G, edge, n_negs):
            """
            DFS state condition

            Args:
                edge (tuple): the edge we are trying to cross
                n_negs (int): the number of negative edges crossed so far

            Returns:
                flag, new_state -
                   flag (bool): True if the edge can be crossed
                   new_state: new state for future decisions in this path.
            """
            decision = infr.edge_decision(edge)
            # only cross positive or negative edges
            if decision in {POSTV, NEGTV}:
                # only cross a negative edge once
                willcross = decision == NEGTV
                if willcross and n_negs == 0:
                    data = G.get_edge_data(*edge)
                    # only cross edges above a threshold
                    conf = data.get('confidence', 'unspecified')
                    conf_int = const.CONFIDENCE.CODE_TO_INT[conf]
                    conf_int = 0 if conf_int is None else conf_int
                    flag = conf_int >= thresh
                    num = n_negs + willcross
                    return flag, num
            return False, n_negs

        # need to do DFS check for this. Make DFS only allowed to
        # cross a negative edge once.
        # def dfs_cond_rec(G, parent, state, visited=None):
        #     if visited is None:
        #         visited = set()
        #     visited.add(parent)
        #     for child in G.neighbors(parent):
        #         if child not in visited:
        #             edge = (parent, child)
        #             flag, new_state = can_cross(G, edge, state)
        #             if flag:
        #                 yield child
        #                 for _ in dfs_cond_rec(G, child, new_state, visited):
        #                     yield _

        # need to do DFS check for this. Make DFS only allowed to
        # cross a negative edge once.
        def dfs_cond_stack(G, source, state):
            # stack based version
            visited = {source}
            stack = [(source, iter(G[source]), state)]
            while stack:
                parent, children, state = stack[-1]
                try:
                    child = next(children)
                    if child not in visited:
                        edge = (parent, child)
                        flag, new_state = can_cross(G, edge, state)
                        if flag:
                            yield child
                            visited.add(child)
                            stack.append((child, iter(G[child]), new_state))
                except StopIteration:
                    stack.pop()

        for node in dfs_cond_stack(infr.graph, u, 0):
            if node == v:
                return True
        return False

    def generate_reviews(infr, pos_redun=None, neg_redun=None, data=False):
        """
        Dynamic generator that yeilds high priority reviews
        """
        if pos_redun is not None:
            infr.params['redun.pos'] = pos_redun
        if neg_redun is not None:
            infr.params['redun.neg'] = neg_redun
        infr.prioritize()
        return infr._generate_reviews(data=data)

    def _generate_reviews(infr, data=False):
        if data:
            while True:
                edge, priority = infr.pop()
                yield edge, priority
        else:
            while True:
                edge, priority = infr.pop()
                yield edge


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_priority
        python -m wbia.algo.graph.mixin_priority --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
