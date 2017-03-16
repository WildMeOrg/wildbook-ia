# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import pandas as pd
import itertools as it
import networkx as nx
import vtool as vt
from ibeis.algo.hots.graph_iden_utils import e_
from ibeis.algo.hots.graph_iden_utils import (
    bridges_inside, bridges_cross)
print, rrr, profile = ut.inject2(__name__)


class TerminationCriteria2(object):
    def __init__(term, phis):
        term.phis = phis


class RefreshCriteria2(object):
    """
    Determine when to re-query for candidate edges
    """
    def __init__(refresh):
        refresh.window = 100
        refresh.manual_decisions = []
        refresh.num_pos = 0
        refresh.frac_thresh = 3 / refresh.window
        refresh.pos_thresh = 2

    def reset(refresh):
        refresh.manual_decisions = []
        refresh.num_pos = 0

    def add(refresh, decision, user_id):
        decision_code = 1 if decision == 'match' else 0
        if user_id is not None and not user_id.startswith('auto'):
            refresh.manual_decisions.append(decision_code)
        if decision_code:
            refresh.num_pos += 1

    @property
    def pos_frac(refresh):
        return np.mean(refresh.manual_decisions[-refresh.window:])

    def check(refresh):
        return (refresh.pos_frac < refresh.frac_thresh and
                refresh.num_pos > refresh.pos_thresh)


class UserOracle(object):
    def __init__(oracle, accuracy, rng):
        oracle.accuracy = accuracy
        oracle.rng = rng
        oracle.states = {'match', 'nomatch', 'notcomp'}

    def review(oracle, edge, truth):
        feedback = {
            'user_id': 'oracle',
            'confidence': 'absolutely_sure',
            'decision': None,
            'tags': [],
        }
        error = oracle.accuracy < oracle.rng.random()
        if error:
            error_options = list(oracle.states - {truth} - {'notcomp'})
            observed = oracle.rng.choice(list(error_options))
        else:
            observed = truth
        if oracle.accuracy < 1.0:
            feedback['confidence'] = 'pretty_sure'
        if oracle.accuracy < .5:
            feedback['confidence'] = 'guessing'
        feedback['decision'] = observed
        if error:
            ut.cprint('MADE MANUAL ERROR', 'red')
        return feedback


class CandidateSearch2(object):
    """ Search for candidate edges """

    @profile
    def filter_nonredun_edges(infr, edges):
        for u, v in edges:
            pos_graph = infr.pos_graph
            nidu, nidv = pos_graph.node_labels(u, v)
            if nidu == nidv:
                if nidu not in infr.pos_redun_nids:
                    yield (u, v)
            elif nidu != nidv:
                if not infr.neg_redun_nids.has_edge(nidu, nidv):
                    yield (u, v)

    @profile
    def find_lnbnn_candidate_edges(infr):
        # do LNBNN query for new edges
        # Use one-vs-many to establish candidate edges to classify
        infr.exec_matching(cfgdict={
            'resize_dim': 'width',
            'dim_size': 700,
            'condknn': True,
            'can_match_samename': False,
            'can_match_sameimg': False,
        })
        # infr.apply_match_edges(review_cfg={'ranks_top': 5})
        candidate_edges = infr._cm_breaking(review_cfg={'ranks_top': 5})
        already_reviewed = set(infr.get_edges_where_ne(
            'decision', 'unreviewed', edges=candidate_edges,
            default='unreviewed'))
        candidate_edges = set(candidate_edges) - already_reviewed

        if infr.enable_inference:
            candidate_edges = set(infr.filter_nonredun_edges(candidate_edges))

        # if infr.method == 'graph':
        #     # need to remove inferred candidates as well
        #     # hacking this in bellow
        #     pass

        if infr.verbose:
            print('[infr] vsmany found %d/%d new edges' % (
                len(candidate_edges), len(candidate_edges) + len(already_reviewed)))
        return candidate_edges

    @profile
    def find_pos_redun_candidate_edges(infr):
        # Add random edges between exisiting non-redundant PCCs
        candidate_edges = set([])
        for pcc in infr.non_pos_redundant_pccs(relax_size=True):
            sub = infr.graph.subgraph(pcc)
            # Very agressive, need to tone down
            check_edges = set(it.starmap(e_, nx.complement(sub).edges()))
            candidate_edges.update(check_edges)
        return candidate_edges

    def find_neg_redun_candidate_edges(infr):
        candidate_edges = set([])
        for c1, c2, check_edges in infr.non_complete_pcc_pairs():
            candidate_edges.update(check_edges)
        return candidate_edges

    def find_new_candidate_edges(infr, ranking=True):
        if ranking:
            candidate_edges = infr.find_lnbnn_candidate_edges()
        else:
            candidate_edges = set([])
        if infr.enable_inference:
            if False:
                new_neg = set(infr.find_neg_redun_candidate_edges())
                candidate_edges.update(new_neg)
            if not ranking:
                new_pos = set(infr.find_pos_redun_candidate_edges())
                candidate_edges.update(new_pos)
        import utool
        with utool.embed_on_exception_context:
            new_edges = {
                edge for edge in candidate_edges if not infr.graph.has_edge(*edge)
            }
        return new_edges

    def add_new_candidate_edges(infr, new_edges):
        new_edges = list(new_edges)
        if len(new_edges) == 0:
            return

        infr.graph.add_edges_from(new_edges)

        if infr.test_mode:
            edge_truth_df = infr.match_state_df(new_edges)
            edge_truth = edge_truth_df.idxmax(axis=1).to_dict()
            infr.set_edge_attrs('truth', edge_truth)
            infr.edge_truth.update(edge_truth)

        if infr.classifiers:
            if infr.verbose > 1:
                print('Prioritizing edges with one-vs-one probabilities')
            # Construct pairwise features on edges in infr
            # needs_probs = infr.get_edges_where_eq('task_probs', None,
            #                                       edges=new_edges,
            #                                       default=None)
            task_probs = infr._make_task_probs(new_edges)

            primary_task = 'match_state'
            primary_probs = task_probs[primary_task]
            primary_thresh = infr.task_thresh[primary_task]
            prob_match = primary_probs['match']

            # Give negatives that pass automatic thresholds high priority
            default_priority = prob_match.copy()
            nomatch_probs = task_probs[primary_task]['nomatch']
            flags = nomatch_probs > primary_thresh['nomatch']
            default_priority[flags] = np.maximum(default_priority[flags],
                                                 nomatch_probs[flags])

            # Pack into edge attributes
            edge_task_probs = {edge: {} for edge in new_edges}
            for task, probs in task_probs.items():
                for edge, val in probs.to_dict(orient='index').items():
                    edge_task_probs[edge][task] = val

            infr.set_edge_attrs('prob_match', prob_match.to_dict())
            infr.set_edge_attrs('task_probs', edge_task_probs)
            infr.set_edge_attrs('default_priority', default_priority.to_dict())

            # Insert all the new edges into the priority queue
            infr.queue.update((-default_priority).to_dict())
        else:
            if infr.verbose > 1:
                print('Prioritizing edges with one-vs-vsmany scores')
            # Not given any deploy classifier, this is the best we can do
            infr.task_probs = None
            scores = infr._make_lnbnn_scores(new_edges)
            infr.set_edge_attrs('normscore', ut.dzip(new_edges, scores))
            infr.queue.update(ut.dzip(new_edges, -scores))

    @profile
    def refresh_candidate_edges2(infr, ranking=True):
        """
        Search for candidate edges.
        Assign each edge a priority and add to queue.
        """
        if infr.verbose:
            print('[infr] refresh_candidate_edges2')
        infr.refresh.reset()
        new_edges = infr.find_new_candidate_edges(ranking=ranking)
        infr.add_new_candidate_edges(new_edges)

    def _make_task_probs(infr, edges):
        pblm = infr.classifiers
        task_keys = list(pblm.samples.subtasks.keys())
        data_key = pblm.default_data_key
        X = pblm.make_deploy_features(infr, edges, data_key)
        task_probs = pblm.predict_proba_deploy(X, task_keys)
        return task_probs

    def _make_lnbnn_scores(infr, edges):
        edge_to_data = infr._get_cm_edge_data(edges)
        edges = list(edge_to_data.keys())
        edge_scores = list(ut.take_column(edge_to_data.values(), 'score'))
        edge_scores = ut.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        # take the inf-norm
        normscores = edge_scores / vt.safe_max(edge_scores, nans=False)
        return normscores


class InfrFeedback2(object):

    def init_bookkeeping(infr):
        infr.recovery_cc = None
        infr.recover_prev_neg_nids = None
        # Set of PCCs that are positive redundant
        infr.pos_redun_nids = set([])
        # Represents the metagraph of negative edges between PCCs
        infr.neg_redun_nids = nx.Graph()

    def _add_review_edge(infr, edge, decision):
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edge(*edge)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

    @profile
    def add_feedback2(infr, edge, decision, tags=None, user_id=None,
                      confidence=None, verbose=None):
        if verbose is None:
            verbose = infr.verbose
        edge = e_(*edge)
        aid1, aid2 = edge
        if verbose >= 1:
            print(('[infr] add_feedback(%r, %r, decision=%r, tags=%r, '
                                        'user_id=%r, confidence=%r)') % (
                aid1, aid2, decision, tags, user_id, confidence))

        # Keep track of sequential reviews and set properties on global graph
        num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)
        feedback_item = {
            'decision': decision,
            'tags': tags,
            'timestamp': ut.get_timestamp('int', isutc=True),
            'confidence': confidence,
            'user_id': user_id,
            'num_reviews': num_reviews + 1,
        }
        infr.internal_feedback[edge].append(feedback_item)
        infr.refresh.add(decision, user_id)
        infr.set_edge_attr(edge, feedback_item)

        if infr.test_mode:
            if user_id.startswith('auto'):
                infr.test_state['n_auto'] += 1
            elif user_id == 'oracle':
                infr.test_state['n_manual'] += 1
            else:
                raise AssertionError('unknown user_id=%r' % (user_id,))

        if infr.enable_inference:
            # Update priority queue based on the new edge
            if infr.recovery_cc:
                infr.inconsistent_inference(edge, decision)
            else:
                infr.consistent_inference(edge, decision)
        else:
            infr._add_review_edge(edge, decision)

        metrics = infr.measure_metrics2()
        infr.metrics_list.append(metrics)

    @profile
    def consistent_inference(infr, edge, decision):
        # assuming we are in a consistent state
        # k_pos = infr.queue_params['pos_redundancy']
        aid1, aid2 = edge
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        nid1 = pos_graph.node_label(aid1)
        nid2 = pos_graph.node_label(aid2)
        cc1 = pos_graph.component_nodes(nid1)
        cc2 = pos_graph.component_nodes(nid2)
        if decision == 'match' and nid1 == nid2:
            # add positive redundancy
            infr._add_review_edge(edge, decision)
            if nid1 not in infr.pos_redun_nids:
                infr.update_pos_redun(nid1)
        elif decision == 'match' and nid1 != nid2:
            if any(bridges_cross(neg_graph, cc1, cc2)):
                # Added positive edge between PCCs with a negative edge
                infr._enter_recovery(edge, decision, nid1, nid2)
            else:
                # merge into single pcc
                infr._add_review_edge(edge=(aid1, aid2), decision='match')
                new_nid = pos_graph.node_label(aid1)
                prev_neg_nids = infr.purge_redun_flags(nid1, nid2)
                for other_nid in prev_neg_nids:
                    infr.update_neg_redun(new_nid, other_nid)
                infr.update_pos_redun(new_nid)
        elif decision == 'nomatch' and nid1 == nid2:
            # Added negative edge in positive PCC
            infr._enter_recovery(edge, decision, nid1)
        elif decision == 'nomatch' and nid1 != nid2:
            # add negative redundancy
            infr._add_review_edge(edge, decision)
            if not infr.neg_redun_nids.has_edge(nid1, nid2):
                infr.update_neg_redun(nid1, nid2)
        elif decision == 'notcomp' and nid1 == nid2:
            if pos_graph.has_edge(*edge):
                # changed an existing positive edge
                infr._add_review_edge(edge, decision)
                new_nid1, new_nid2 = pos_graph.node_labels(aid1, aid2)
                if new_nid1 != new_nid2:
                    # split case
                    old_nid = nid1
                    prev_neg_nids = infr.purge_redun_flags(old_nid)
                    for other_nid in prev_neg_nids:
                        infr.update_neg_redun(new_nid1, other_nid)
                        infr.update_neg_redun(new_nid2, other_nid)
                    infr.update_neg_redun(new_nid1, new_nid2)
                    infr.update_pos_redun(new_nid1)
                    infr.update_pos_redun(new_nid2)
                else:
                    infr.update_pos_redun(cc1)
            else:
                infr._add_review_edge(edge, decision)

        elif decision == 'notcomp' and nid1 != nid2:
            if neg_graph.has_edge(*edge):
                # changed and existing negative edge
                infr._add_review_edge(edge, decision)
                if not infr.neg_redun_nids.has_edge(nid1, nid2):
                    infr.update_neg_redun(nid1, nid2)
            else:
                infr._add_review_edge(edge, decision)
        else:
            raise AssertionError('impossible consistent state')

    def _enter_recovery(infr, edge, decision, *nids):
        ut.cprint('GRAPH HAS ENTERED AN INCONSISTENT STATE', 'red')
        aid1, aid2 = edge
        pos_graph = infr.pos_graph
        infr.recover_cc = set(ut.flatten([pos_graph.component_nodes(nid)
                                          for nid in nids]))
        infr.recover_prev_neg_nids = infr.purge_redun_flags(nids)
        infr.inconsistent_inference(edge, decision)

    def inconsistent_inference(infr, edge, decision):
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph

        # Add in the new edge
        infr._add_review_edge(edge, decision)

        # Check if there is any inconsistent edge between any pcc in
        # infr.recovery_cc
        neg_subgraph = neg_graph.subgraph(infr.recover_cc)
        inconsistent_edges = [
            (u, v) for u, v in neg_subgraph.edges()
            if pos_graph.node_label(u) == pos_graph.node_label(v)
        ]
        if inconsistent_edges:
            ut.cprint('graph is inconsistent. searching for errors', 'red')
            pos_subgraph = pos_graph.subgraph(infr.recover_cc).copy()
            error_edge_gen = infr.hypothesis_errors(pos_subgraph,
                                                    inconsistent_edges)
            # choose just one error edge and give it insanely high priority
            try:
                error_edge = next(error_edge_gen)
            except StopIteration:
                raise AssertionError('As least one edge must be generated')
            base = infr.graph.get_edge_data(*error_edge).get('prob_match')
            infr.queue[error_edge] = -(10 + base)
        else:
            ut.cprint('consistency has been restored', 'green')
            nid_to_cc = ut.group_items(
                infr.recover_cc,
                map(pos_graph.node_label, infr.recover_cc))

            # Update redundancies on the influenced subgraph
            # Force reinstatement
            for nid in nid_to_cc.keys():
                infr.update_pos_redun(nid, check_reinstate=True)

            for nid1, nid2 in it.combinations(nid_to_cc.keys(), 2):
                infr.update_neg_redun(nid1, nid2, check_reinstate=True)

            for nid1, nid2 in it.product(nid_to_cc.keys(),
                                         infr.recover_prev_neg_nids):
                infr.update_neg_redun(nid1, nid2, check_reinstate=True)

            # Remove recovery flags
            infr.recover_cc = None
            infr.recover_prev_neg_nids = None

    @profile
    def hypothesis_errors(infr, pos_subgraph, neg_edges):
        pos_edges = list(pos_subgraph.edges())

        # Generate weights for edges
        pos_prob = list(infr.gen_edge_values('prob_match', pos_edges))
        neg_prob = list(infr.gen_edge_values('prob_match', neg_edges))
        pos_n = list(infr.gen_edge_values('num_reviews', pos_edges))
        neg_n = list(infr.gen_edge_values('num_reviews', neg_edges))
        pos_weight = np.add(pos_prob, pos_n)
        neg_weight = np.add(neg_prob, neg_n)
        capacity = 'weight'
        nx.set_edge_attributes(pos_subgraph, capacity,
                               ut.dzip(pos_edges, pos_weight))

        # Solve a multicut problem for multiple pairs of terminal nodes.
        # Running multiple min-cuts produces a k-factor approximation
        maybe_error_edges = set([])
        for (s, t), join_weight in zip(neg_edges, neg_weight):
            _, parts = nx.minimum_cut(pos_subgraph, s, t, capacity=capacity)
            cut_edgeset = bridges_cross(pos_subgraph, *parts)
            join_edgeset = {e_(s, t)}
            cut_edgeset_weight = sum([
                pos_subgraph.get_edge_data(u, v)[capacity]
                for u, v in cut_edgeset])
            join_edgeset_weight = join_weight
            if join_edgeset_weight < cut_edgeset_weight:
                chosen = join_edgeset
            else:
                chosen = cut_edgeset
            for edge in chosen:
                edge = e_(*edge)
                if edge not in maybe_error_edges:
                    maybe_error_edges.add(edge)
                    yield edge

    def purge_redun_flags(infr, *nids):
        neighbs = (infr.neg_redun_nids.neighbors(nid) for nid in nids
                   if infr.neg_redun_nids.has_node(nid))
        prev_neg_nids = set(ut.iflatten(neighbs))
        prev_neg_nids -= set(nids)
        infr.neg_redun_nids.remove_nodes_from(nids)
        infr.pos_redun_nids.difference_update(set(nids))
        return prev_neg_nids

    @profile
    def update_neg_redun(infr, nid1, nid2, check_reinstate=False):
        cc1 = infr.pos_graph.component_nodes(nid1)
        cc2 = infr.pos_graph.component_nodes(nid2)
        if infr.is_neg_redundant(cc1, cc2):
            # Flag ourselves as negative redundant and remove priorities
            infr.neg_redun_nids.add_edge(nid1, nid2)
            infr.queue.delete_items(bridges_cross(infr.graph, cc1, cc2))
        else:
            # FIXME: we can make this faster with assumption flags
            # if we are not k negative redunant but we are flagged as such
            # then remove flag and reinstate priority
            if check_reinstate or infr.neg_redun_nids.has_edge(nid1, nid2):
                try:
                    infr.neg_redun_nids.remove_edge(nid1, nid2)
                except nx.exception.NetworkXError:
                    pass
                edges = bridges_cross(infr.graph, cc1, cc2)
                prob_match = np.array(list(infr.gen_edge_values(
                    'prob_match', edges)))
                priority = -prob_match
                infr.queue.update(ut.dzip(edges, priority))

    @profile
    def update_pos_redun(infr, nid, check_reinstate=False):
        cc = infr.pos_graph.component_nodes(nid)
        if infr.is_pos_redundant(cc):
            infr.pos_redun_nids.add(nid)
            infr.queue.delete_items(bridges_inside(infr.graph, cc))
        else:
            if check_reinstate or nid in infr.pos_redun_nids:
                infr.pos_redun_nids -= {nid}
                edges = bridges_inside(infr.graph, cc)
                prob_match = np.array(list(infr.gen_edge_values(
                    'prob_match', edges)))
                priority = -prob_match
                infr.queue.update(ut.dzip(edges, priority))

    @profile
    def is_neg_redundant(infr, cc1, cc2):
        k_neg = infr.queue_params['neg_redundancy']
        neg_graph = infr.neg_graph
        # from ibeis.algo.hots.graph_iden_utils import bridges_cross
        # num_neg = len(list(bridges_cross(neg_graph, cc1, cc2)))
        # return num_neg >= k_neg
        neg_edge_gen = (
            1 for u in cc1 for v in cc2.intersection(neg_graph.adj[u])
        )
        # do a lazy count of bridges
        for count in neg_edge_gen:
            if count >= k_neg:
                return True


class InfrReviewers(object):

    @profile
    def try_auto_review2(infr, edge):
        review = {
            'user_id': 'auto_clf',
            'confidence': 'pretty_sure',
            'decision': None,
            'tags': [],
        }
        if infr.recovery_cc:
            # Do not autoreview if we are in an inconsistent state
            if infr.verbose > 1:
                print('Must manually review inconsistent edge')
            return False, review
        # Determine if anything passes the match threshold
        primary_task = 'match_state'
        data = infr.get_edge_data(*edge)
        decision_probs = pd.Series(data['task_probs'][primary_task])
        a, b = decision_probs.align(infr.task_thresh[primary_task])
        decision_flags = a > b
        # decision_probs > infr.task_thresh[primary_task]
        auto_flag = False
        if sum(decision_flags) == 1:
            # Check to see if it might be confounded by a photobomb
            pb_probs = data['task_probs']['photobomb_state']
            pb_thresh = infr.task_thresh['photobomb_state']['pb']
            confounded = pb_probs['pb'] > pb_thresh
            if not confounded:
                review['decision'] = decision_flags.argmax()
                truth = infr.match_state_gt(edge).idxmax()
                if review['decision'] != truth:
                    ut.cprint('AUTOMATIC ERROR', 'purple')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            print('Automatic review success')

        return auto_flag, review

    def try_implicit_review2(infr, edge):
        review = {}
        # Check if edge is implicitly negative
        if not infr.recovery_cc:
            implicit_flag = (
                infr.check_prob_completeness(edge[0]) and
                infr.check_prob_completeness(edge[1])
            )
            if implicit_flag:
                review = {
                    'user_id': 'auto_implicit_complete',
                    'confidence': 'pretty_sure',
                    'decision': 'nomatch',
                    'tags': [],
                }
        else:
            implicit_flag = False
        return implicit_flag, review

    def request_user_review(infr, edge):
        if infr.test_mode:
            true_state = infr.match_state_gt(edge)
            truth = true_state.idxmax()
            feedback = infr.oracle.review(edge, truth)
        else:
            raise NotImplementedError('no user review')
        return feedback


class TestStuff2(object):
    def init_test_mode2(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, phis=None, complete_thresh=None,
                        match_state_thresh=None):
        ut.cprint('INIT TEST', 'yellow')
        infr.init_bookkeeping()

        infr.classifiers = classifiers
        infr.enable_inference = enable_inference
        infr.enable_autoreview = enable_autoreview

        infr.queue_params['pos_redundancy'] = k_redun
        infr.queue_params['neg_redundancy'] = k_redun
        infr.queue_params['complete_thresh'] = complete_thresh

        infr.queue = ut.PriorityQueue()

        rng = ut.ensure_rng(2315329092, impl='python')
        infr.oracle = UserOracle(oracle_accuracy, rng)
        infr.term = TerminationCriteria2(phis)

        infr.task_thresh = {
            'photobomb_state': pd.Series({
                'pb': .5,
                'notpb': .9,
            }),
            'match_state': pd.Series(match_state_thresh)
        }

        infr.test_mode = True
        infr.edge_truth = {}
        infr.metrics_list = []
        infr.test_state = {
            'n_auto': 0,
            'n_manual': 0,
        }
        infr.nid_to_gt_cc = ut.group_items(infr.aids, infr.orig_name_labels)
        infr.real_n_pcc_mst_edges = sum(
            len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        ut.cprint('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), 'red')

    def error_edges2(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('decision', 'unreviewed')
            if pred_state != 'unreviewed':
                if true_state != pred_state:
                    error = ut.odict([('real', true_state),
                                      ('pred', pred_state)])
                    yield edge, error

    def measure_metrics2(infr):
        real_pos_edges = []
        n_error_edges = 0
        pred_n_pcc_mst_edges = 0
        n_fn = 0
        n_fp = 0

        for edge, data in infr.edges(data=True):
            true_state = infr.edge_truth[edge]
            decision = data.get('decision', 'unreviewed')
            if true_state == decision and true_state == 'match':
                real_pos_edges.append(edge)
            elif decision != 'unreviewed':
                if true_state != decision:
                    n_error_edges += 1
                    if true_state == 'match':
                        n_fn += 1
                    elif true_state == 'nomatch':
                        n_fp += 1

        import networkx as nx
        for cc in nx.connected_components(nx.Graph(real_pos_edges)):
            pred_n_pcc_mst_edges += len(cc) - 1

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_manual': infr.test_state['n_manual'],
            'n_auto': infr.test_state['n_auto'],
            'pos_acc': pos_acc,
            'n_merge_remain': infr.real_n_pcc_mst_edges - pred_n_pcc_mst_edges,
            'merge_remain': 1 - pos_acc,
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
        }
        return metrics


class AnnotInfr2(InfrFeedback2, CandidateSearch2, InfrReviewers, TestStuff2):

    def inner_loop2(infr):
        """
        Executes reviews until the queue is empty or

        """
        ut.cprint('Start inner loop', 'blue')
        while True:
            if len(infr.queue) == 0:
                ut.cprint('No more edges, need refresh', 'blue')
                break
            edge, priority = infr.pop()
            if not infr.recovery_cc:
                if infr.refresh.check():
                    ut.cprint('Refresh criteria flags refresh', 'blue')
                    break
            else:
                ut.cprint('IN RECOVERY MODE priority=%r' % (priority,), 'red')

            flag = False
            if infr.enable_autoreview:
                flag, feedback = infr.try_auto_review2(edge)
            if not flag:
                # if infr.enable_inference:
                #     flag, feedback = infr.try_implicit_review2(edge)
                if not flag:
                    feedback = infr.request_user_review(edge)
            infr.add_feedback2(edge=edge, **feedback)
            if infr.recovery_cc:
                infr.recovery_review_loop()

    def recovery_review_loop(infr):
        while infr.recovery_cc:
            edge, priority = infr.pop()
            ut.cprint('IN RECOVERY LOOP priority=%r' % (priority,), 'red')
            feedback = infr.request_user_review(edge)
            infr.add_feedback2(edge=edge, **feedback)

    def priority_review_loop(infr, max_loops):
        infr.refresh = RefreshCriteria2()
        for count in it.count(0):
            if count >= max_loops:
                ut.cprint('Early stop', 'blue')
                break
            ut.cprint('Outer loop iter %d ' % (count,), 'blue')
            infr.refresh_candidate_edges2()
            if not len(infr.queue):
                ut.cprint('Queue is empty. Terminate.', 'blue')
                break
            infr.inner_loop2()
            if infr.enable_inference:
                print('HACK FIX REDUN')
                # Fix anything that is not positive/negative redundant
                real_queue = infr.queue
                # use temporary queue
                infr.queue = ut.PriorityQueue()
                infr.refresh_candidate_edges2(ranking=False)
                infr.inner_loop2()
                infr.queue = real_queue

    @profile
    def main_loop2(infr, max_loops=np.inf):
        import utool
        utool.embed()

        infr.reset(state='empty')
        infr.remove_feedback(apply=True)

        infr.priority_review_loop(max_loops)

        if infr.enable_inference:
            # Enforce that a user checks any PCC that was auto-reviewed
            # but was unable to achieve k-positive-consistency
            for pcc in list(infr.non_pos_redundant_pccs()):
                subgraph = infr.graph.subgraph(pcc)
                for u, v, data in subgraph.edges(data=True):
                    edge = infr.e_(u, v)
                    if data.get('user_id', '').startswith('auto'):
                        feedback = infr.request_user_review(edge)
                        infr.add_feedback2(edge=edge, **feedback)
            # Check for inconsistency recovery
            infr.recovery_review_loop()

        if infr.enable_inference:
            assert len(list(infr.inconsistent_components())) == 0
        # true_groups = list(map(set, infr.nid_to_gt_cc.values()))
        # pred_groups = list(infr.positive_connected_compoments())
        # from ibeis.algo.hots import sim_graph_iden
        # comparisons = sim_graph_iden.compare_groups(true_groups, pred_groups)
        # pred_merges = comparisons['pred_merges']
        # print(pred_merges)
        ut.cprint('FINISH ETE')
