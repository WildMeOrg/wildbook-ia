# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import pandas as pd
import itertools as it
import networkx as nx
import vtool as vt
from ibeis.algo.hots.graph_iden_utils import e_
from ibeis.algo.hots.graph_iden_utils import (
    edges_inside, edges_cross, edges_outgoing, ensure_multi_index)
print, rrr, profile = ut.inject2(__name__)


DEBUG_INCON = True


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
        if isinstance(rng, six.string_types):
            rng = sum(map(ord, rng))
        rng = ut.ensure_rng(rng, impl='python')

        oracle.accuracy = accuracy
        oracle.rng = rng
        oracle.states = {'match', 'nomatch', 'notcomp'}

    def review(oracle, edge, truth, force=False):
        feedback = {
            'user_id': 'oracle',
            'confidence': 'absolutely_sure',
            'decision': None,
            'tags': [],
        }
        error = oracle.accuracy < oracle.rng.random()
        if force:
            error = False
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
            ut.cprint('MADE MANUAL ERROR edge=%r, truth=%r, observed=%r' %
                      (edge, truth, observed), 'red')
        return feedback


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def inner_loop(infr):
        """
        Executes reviews until the queue is empty or needs refresh
        """

        infr.print('Start inner loop')
        for count in it.count(0):
            if len(infr.queue) == 0:
                infr.print('No more edges, need refresh')
                break
            edge, priority = infr.pop()
            if infr.is_recovering():
                infr.print('IN RECOVERY MODE priority=%r' % (priority,),
                           color='red')
            else:
                if infr.refresh.check():
                    infr.print('Refresh criteria flags refresh')
                    break

            flag = False
            if infr.enable_autoreview:
                flag, feedback = infr.try_auto_review(edge)
            if not flag:
                # if infr.enable_inference:
                #     flag, feedback = infr.try_implicit_review(edge)
                if not flag:
                    feedback = infr.request_user_review(edge)
            infr.add_feedback(edge=edge, **feedback)

            if infr.is_recovering():
                infr.recovery_review_loop()

    def recovery_review_loop(infr):
        while infr.is_recovering():
            edge, priority = infr.pop()
            num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)
            feedback = infr.request_user_review(edge)
            infr.print(
                'RECOVERY LOOP edge={}, decision={}, priority={}, '
                'n_reviews={}, len(recover_ccs)={}'.format(
                    edge, feedback['decision'], priority, num_reviews,
                    len(infr.recovery_ccs)),
                color='red'
            )
            infr.add_feedback(edge=edge, **feedback)

    def priority_review_loop(infr, max_loops):
        infr.refresh = RefreshCriteria2()
        for count in it.count(0):
            if count >= max_loops:
                infr.print('early stop')
                break
            infr.print('Outer loop iter %d ' % (count,))
            infr.refresh_candidate_edges()
            if not len(infr.queue):
                infr.print('Queue is empty. Terminate.')
                break
            infr.inner_loop()
            if infr.enable_inference:
                infr.assert_consistency_invariant()
                infr.print('HACK FIX REDUN', color='white')
                # Fix anything that is not positive/negative redundant
                real_queue = infr.queue
                # use temporary queue
                infr.queue = ut.PriorityQueue()
                infr.refresh_candidate_edges(ranking=False)
                infr.inner_loop()
                infr.queue = real_queue

    @profile
    def main_loop(infr, max_loops=np.inf):
        infr.print('Starting main loop', 1)
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
                        infr.add_feedback(edge=edge, **feedback)
            # Check for inconsistency recovery
            infr.recovery_review_loop()

        if infr.enable_inference and DEBUG_INCON:
            infr.assert_consistency_invariant()
        # true_groups = list(map(set, infr.nid_to_gt_cc.values()))
        # pred_groups = list(infr.positive_connected_compoments())
        # from ibeis.algo.hots import sim_graph_iden
        # comparisons = sim_graph_iden.compare_groups(true_groups, pred_groups)
        # pred_merges = comparisons['pred_merges']
        # print(pred_merges)
        infr.print('Exiting main loop')


class InfrReviewers(object):
    @profile
    def try_auto_review(infr, edge):
        review = {
            'user_id': 'auto_clf',
            'confidence': 'pretty_sure',
            'decision': None,
            'tags': [],
        }
        if infr.is_recovering():
            # Do not autoreview if we are in an inconsistent state
            infr.print('Must manually review inconsistent edge', 1)
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
                    infr.print('AUTOMATIC ERROR edge=%r, truth=%r, decision=%r' %
                               (edge, truth, review['decision']), color='purple')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        return auto_flag, review

    def try_implicit_review(infr, edge):
        review = {}
        # Check if edge is implicitly negative
        if not infr.is_recovering():
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
            feedback = infr.oracle.review(
                edge, truth, infr.is_recovering())
        else:
            raise NotImplementedError('no user review')
        return feedback


class SimulationHelpers(object):
    def init_simulation(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, phis=None, complete_thresh=None,
                        match_state_thresh=None, name=None):

        infr.name = name

        infr.print('INIT SIMULATION', color='yellow')

        infr.classifiers = classifiers
        infr.enable_inference = enable_inference
        infr.enable_autoreview = enable_autoreview

        infr.queue_params['pos_redundancy'] = k_redun
        infr.queue_params['neg_redundancy'] = k_redun
        infr.queue_params['complete_thresh'] = complete_thresh

        infr.queue = ut.PriorityQueue()

        infr.oracle = UserOracle(oracle_accuracy, infr.name)
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
        infr.print('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), color='red')

    def measure_error_edges(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('decision', 'unreviewed')
            if pred_state != 'unreviewed':
                if true_state != pred_state:
                    error = ut.odict([('real', true_state),
                                      ('pred', pred_state)])
                    yield edge, error

    @profile
    def measure_metrics(infr):
        real_pos_edges = []
        n_error_edges = 0
        pred_n_pcc_mst_edges = 0
        n_fn = 0
        n_fp = 0

        # TODO: dynamic measurement

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


class InfrInvariants(object):

    def assert_invariants(infr, msg=''):
        infr.assert_disjoint_invariant(msg)
        infr.assert_consistency_invariant(msg)
        infr.assert_recovery_invariant(msg)

    def assert_disjoint_invariant(infr, msg=''):
        edge_sets = [
            set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        ]
        for es1, es2 in it.combinations(edge_sets, 2):
            assert es1.isdisjoint(es2), 'edge sets must be disjoint'
        all_edges = set(it.starmap(e_, infr.graph.edges()))
        edge_union = set.union(*edge_sets)
        assert edge_union == all_edges, 'edge sets must have full union'

    def assert_consistency_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        if infr.enable_inference:
            incon_ccs = list(infr.inconsistent_components())
            with ut.embed_on_exception_context:
                if len(incon_ccs) > 0:
                    raise AssertionError('The graph is not consistent. ' +
                                         msg)

    def assert_recovery_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
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


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInfrMatching(object):
    """
    Methods for running matching algorithms
    """

    def exec_matching(infr, prog_hook=None, cfgdict=None):
        """
        Loads chip matches into the inference structure
        Uses graph name labeling and ignores ibeis labeling
        """
        infr.print('exec_matching', 1)
        #from ibeis.algo.hots import graph_iden
        ibs = infr.ibs
        aids = infr.aids
        if cfgdict is None:
            cfgdict = {
                # 'can_match_samename': False,
                'can_match_samename': True,
                'can_match_sameimg': True,
                # 'augment_queryside_hack': True,
                'K': 3,
                'Knorm': 3,
                'prescore_method': 'csum',
                'score_method': 'csum'
            }
        # hack for ulsing current nids
        custom_nid_lookup = ut.dzip(aids, infr.get_annot_attrs('name_label',
                                                               aids))
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=infr.verbose >= 2)

        cm_list = qreq_.execute(prog_hook=prog_hook)

        infr.vsmany_qreq_ = qreq_
        infr.vsmany_cm_list = cm_list
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def exec_vsone(infr, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> infr.ensure_full()
            >>> result = infr.exec_vsone()
            >>> print(result)
        """
        # Post process ranks_top and bottom vsmany queries with vsone
        # Execute vsone queries on the best vsmany results
        parent_rowids = list(infr.graph.edges())
        qaids = ut.take_column(parent_rowids, 0)
        daids = ut.take_column(parent_rowids, 1)

        config = {
            # 'sv_on': False,
            'ratio_thresh': .9,
        }

        result_list = infr.ibs.depc.get('vsone', (qaids, daids), config=config)
        # result_list = infr.ibs.depc.get('vsone', parent_rowids)
        # result_list = infr.ibs.depc.get('vsone', [list(zip(qaids)), list(zip(daids))])
        # hack copy the postprocess
        import ibeis
        unique_qaids, groupxs = ut.group_indices(qaids)
        grouped_daids = ut.apply_grouping(daids, groupxs)

        unique_qnids = infr.ibs.get_annot_nids(unique_qaids)
        single_cm_list = ut.take_column(result_list, 1)
        grouped_cms = ut.apply_grouping(single_cm_list, groupxs)

        _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_cms)
        cm_list = []
        for qaid, qnid, daids, cms in _iter:
            # Hacked in version of creating an annot match object
            chip_match = ibeis.ChipMatch.combine_cms(cms)
            # chip_match.score_maxcsum(request)
            cm_list.append(chip_match)

        # cm_list = qreq_.execute(parent_rowids)
        infr.vsone_qreq_ = infr.ibs.depc.new_request('vsone', qaids, daids, cfgdict=config)
        infr.vsone_cm_list_ = cm_list
        infr.qreq_  = infr.vsone_qreq_
        infr.cm_list = cm_list

    def _exec_pairwise_match(infr, edges, config={}, prog_hook=None):
        edges = ut.lmap(tuple, ut.aslist(edges))
        infr.print('exec_vsone_subset')
        qaids = ut.take_column(edges, 0)
        daids = ut.take_column(edges, 1)
        # TODO: ensure feat/chip configs are resepected
        match_list = infr.ibs.depc.get('pairwise_match', (qaids, daids),
                                       'match', config=config)
        # recompute=True)
        # Hack: Postprocess matches to re-add annotation info in lazy-dict format
        from ibeis import core_annots
        config = ut.hashdict(config)
        configured_lazy_annots = core_annots.make_configured_annots(
            infr.ibs, qaids, daids, config, config, preload=True)
        for qaid, daid, match in zip(qaids, daids, match_list):
            match.annot1 = configured_lazy_annots[config][qaid]
            match.annot2 = configured_lazy_annots[config][daid]
            match.config = config
        return match_list

    def _enrich_matches_lnbnn(infr, matches, inplace=False):
        """
        applies lnbnn scores to pairwise one-vs-one matches
        """
        from ibeis.algo.hots import nn_weights
        qreq_ = infr.qreq_
        qreq_.load_indexer()
        indexer = qreq_.indexer
        if not inplace:
            matches_ = [match.copy() for match in matches]
        else:
            matches_ = matches
        K = qreq_.qparams.K
        Knorm = qreq_.qparams.Knorm
        normalizer_rule  = qreq_.qparams.normalizer_rule

        infr.print('Stacking vecs for batch lnbnn matching')
        offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
        stacked_vecs = np.vstack([
            match_.matched_vecs2()
            for match_ in ut.ProgIter(matches_, label='stack matched vecs')
        ])

        vecs = stacked_vecs
        num = (K + Knorm)
        idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192,
                                        label='lnbnn scoring')

        idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
        dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
        iter_ = zip(matches_, idx_list, dist_list)
        prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
        for match_, neighb_idx, neighb_dist in prog:
            qaid = match_.annot2['aid']
            norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm,
                                          normalizer_rule)
            ndist = vt.take_col_per_row(neighb_dist, norm_k)
            vdist = match_.local_measures['match_dist']
            lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
            lnbnn_clip_dist = np.clip(lnbnn_dist, 0, np.inf)
            match_.local_measures['lnbnn_norm_dist'] = ndist
            match_.local_measures['lnbnn'] = lnbnn_dist
            match_.local_measures['lnbnn_clip'] = lnbnn_clip_dist
            match_.fs = lnbnn_dist
        return matches_

    def _enriched_pairwise_matches(infr, edges, config={}, global_keys=None,
                                   need_lnbnn=True, prog_hook=None):
        if global_keys is None:
            global_keys = ['yaw', 'qual', 'gps', 'time']
        matches = infr._exec_pairwise_match(edges, config=config,
                                            prog_hook=prog_hook)
        infr.print('enriching matches')
        if need_lnbnn:
            infr._enrich_matches_lnbnn(matches, inplace=True)
        # Ensure matches know about relavent metadata
        for match in matches:
            vt.matching.ensure_metadata_normxy(match.annot1)
            vt.matching.ensure_metadata_normxy(match.annot2)
        for match in ut.ProgIter(matches, label='setup globals'):
            match.add_global_measures(global_keys)
        for match in ut.ProgIter(matches, label='setup locals'):
            match.add_local_measures()
        return matches

    def _pblm_pairwise_features(infr, edges, data_key=None):
        from ibeis.scripts.script_vsone import AnnotPairFeatInfo
        infr.print('Requesting %d cached pairwise features' % len(edges))
        pblm = infr.classifiers
        if data_key is not None:
            data_key = pblm.default_data_key
        # Parse the data_key to build the appropriate feature
        featinfo = AnnotPairFeatInfo(pblm.samples.X_dict[data_key])
        # Find the kwargs to make the desired feature subset
        pairfeat_cfg, global_keys = featinfo.make_pairfeat_cfg()
        need_lnbnn = any('lnbnn' in key for key in pairfeat_cfg['local_keys'])

        config = pblm.hyper_params.vsone_assign

        def tmprepr(cfg):
            return ut.repr2(cfg, strvals=True, explicit=True,
                            nobr=True).replace(' ', '').replace('\'', '')

        ibs = infr.ibs
        edge_uuids = ibs.unflat_map(ibs.get_annot_visual_uuids, edges)
        edge_hashid = ut.hashstr_arr27(edge_uuids, 'edges', hashlen=32)

        feat_cfgstr = '_'.join([
            edge_hashid,
            config.get_cfgstr(),
            'need_lnbnn={}'.format(need_lnbnn),
            'local(' + tmprepr(pairfeat_cfg) + ')',
            'global(' + tmprepr(global_keys) + ')'
        ])
        feat_cacher = ut.Cacher('bulk_pairfeat_cache', feat_cfgstr,
                                appname=pblm.appname, verbose=20)
        data = feat_cacher.tryload()
        if data is None:
            data = infr._make_pairwise_features(edges, config, pairfeat_cfg,
                                                global_keys, need_lnbnn)
            feat_cacher.save(data)
        matches, feats = data
        assert np.all(featinfo.X.columns == feats.columns), (
            'inconsistent feature dimensions')
        return matches, feats

    def _make_pairwise_features(infr, edges, config={}, pairfeat_cfg={},
                                global_keys=None, need_lnbnn=True,
                                multi_index=True):
        """
        Construct matches and their pairwise features
        """
        import pandas as pd
        # TODO: ensure feat/chip configs are resepected
        edges = ut.lmap(tuple, ut.aslist(edges))
        matches = infr._enriched_pairwise_matches(edges, config=config,
                                                  global_keys=global_keys,
                                                  need_lnbnn=need_lnbnn)
        # ---------------
        # Try different feature constructions
        infr.print('building pairwise features')
        X = pd.DataFrame([
            m.make_feature_vector(**pairfeat_cfg)
            for m in ut.ProgIter(matches, label='making pairwise feats')
        ])
        if multi_index:
            # Index features by edges
            uv_index = ensure_multi_index(edges, ('aid1', 'aid2'))
            X.index = uv_index
        X[pd.isnull(X)] = np.nan
        # Re-order column names to ensure dimensions are consistent
        X = X.reindex_axis(sorted(X.columns), axis=1)
        return matches, X

    def exec_vsone_subset(infr, edges, config={}, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.hots.graph_iden exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> config = {}
            >>> infr.ensure_full()
            >>> edges = [(1, 2), (2, 3)]
            >>> result = infr.exec_vsone_subset(edges)
            >>> print(result)
        """
        match_list = infr._exec_pairwise_match(edges, config=config,
                                               prog_hook=prog_hook)
        vsone_matches = {e_(u, v): match
                         for (u, v), match in zip(edges, match_list)}
        infr.vsone_matches.update(vsone_matches)
        edge_to_score = {e: match.fs.sum() for e, match in
                         vsone_matches.items()}
        infr.graph.add_edges_from(edge_to_score.keys())
        infr.set_edge_attrs('score', edge_to_score)
        return match_list

    def lookup_cm(infr, aid1, aid2):
        """
        Get chipmatch object associated with an edge if one exists.
        """
        if infr.cm_list is None:
            return None, aid1, aid2
        # TODO: keep chip matches in dictionary by default?
        aid2_idx = ut.make_index_lookup(
            [cm.qaid for cm in infr.cm_list])
        switch_order = False

        if aid1 in aid2_idx:
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                switch_order = True
                # raise KeyError('switch order')
        else:
            switch_order = True

        if switch_order:
            # switch order
            aid1, aid2 = aid2, aid1
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                raise KeyError('No ChipMatch for edge (%r, %r)' % (aid1, aid2))
        return cm, aid1, aid2

    @profile
    def apply_match_edges(infr, review_cfg={}):
        """
        Adds results from one-vs-many rankings as edges in the graph
        """
        if infr.cm_list is None:
            infr.print('apply_match_edges - matching has not been run!')
            return
        infr.print('apply_match_edges', 1)
        edges = infr._cm_breaking(review_cfg)
        # Create match-based graph structure
        infr.remove_dummy_edges()
        infr.print('apply_match_edges adding %d edges' % len(edges), 1)
        infr.graph.add_edges_from(edges)
        infr.apply_match_scores()

    def _cm_breaking(infr, review_cfg={}):
        """
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> review_cfg = {}
        """
        cm_list = infr.cm_list
        ranks_top = review_cfg.get('ranks_top', None)
        ranks_bot = review_cfg.get('ranks_bot', None)

        # Construct K-broken graph
        edges = []

        if ranks_bot is None:
            ranks_bot = 0

        for count, cm in enumerate(cm_list):
            score_list = cm.annot_score_list
            rank_list = ut.argsort(score_list)[::-1]
            sortx = ut.argsort(rank_list)

            top_sortx = sortx[:ranks_top]
            bot_sortx = sortx[len(sortx) - ranks_bot:]
            short_sortx = ut.unique(top_sortx + bot_sortx)

            daid_list = ut.take(cm.daid_list, short_sortx)
            for daid in daid_list:
                u, v = (cm.qaid, daid)
                if v < u:
                    u, v = v, u
                edges.append((u, v))
        return edges

    def _cm_training_pairs(infr, top_gt=2, mid_gt=2, bot_gt=2, top_gf=2,
                           mid_gf=2, bot_gf=2, rand_gt=2, rand_gf=2, rng=None):
        """
        Constructs training data for a pairwise classifier

        CommandLine:
            python -m ibeis.algo.hots.graph_iden _cm_training_pairs

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching(cfgdict={
            >>>     'can_match_samename': True,
            >>>     'K': 4,
            >>>     'Knorm': 1,
            >>>     'prescore_method': 'csum',
            >>>     'score_method': 'csum'
            >>> })
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> exec(ut.execstr_funckw(infr._cm_training_pairs))
            >>> rng = np.random.RandomState(42)
            >>> aid_pairs = np.array(infr._cm_training_pairs(rng=rng))
            >>> print(len(aid_pairs))
            >>> assert np.sum(aid_pairs.T[0] == aid_pairs.T[1]) == 0
        """
        cm_list = infr.cm_list
        qreq_ = infr.qreq_
        ibs = infr.ibs
        aid_pairs = []
        dnids = qreq_.ibs.get_annot_nids(qreq_.daids)
        # dnids = qreq_.get_qreq_annot_nids(qreq_.daids)
        rng = ut.ensure_rng(rng)
        for cm in ut.ProgIter(cm_list, lbl='building pairs'):
            all_gt_aids = cm.get_top_gt_aids(ibs)
            all_gf_aids = cm.get_top_gf_aids(ibs)
            gt_aids = ut.take_percentile_parts(all_gt_aids, top_gt, mid_gt,
                                               bot_gt)
            gf_aids = ut.take_percentile_parts(all_gf_aids, top_gf, mid_gf,
                                               bot_gf)
            # get unscored examples
            unscored_gt_aids = [aid for aid in qreq_.daids[cm.qnid == dnids]
                                if aid not in cm.daid2_idx]
            rand_gt_aids = ut.random_sample(unscored_gt_aids, rand_gt, rng=rng)
            # gf_aids = cm.get_groundfalse_daids()
            _gf_aids = qreq_.daids[cm.qnid != dnids]
            _gf_aids = qreq_.daids.compress(cm.qnid != dnids)
            # gf_aids = ibs.get_annot_groundfalse(cm.qaid, daid_list=qreq_.daids)
            rand_gf_aids = ut.random_sample(_gf_aids, rand_gf, rng=rng).tolist()
            chosen_daids = ut.unique(gt_aids + gf_aids + rand_gf_aids +
                                     rand_gt_aids)
            aid_pairs.extend([(cm.qaid, aid) for aid in chosen_daids if cm.qaid != aid])

        return aid_pairs

    def _get_cm_agg_aid_ranking(infr, cc):
        aid_to_cm = {cm.qaid: cm for cm in infr.cm_list}
        # node_to_cm = {infr.aid_to_node[cm.qaid]:
        #               cm for cm in infr.cm_list}
        all_scores = ut.ddict(list)
        for qaid in cc:
            cm = aid_to_cm[qaid]
            # should we be doing nids?
            for daid, score in zip(cm.get_top_aids(), cm.get_top_scores()):
                all_scores[daid].append(score)

        max_scores = sorted((max(scores), aid)
                            for aid, scores in all_scores.items())[::-1]
        ranked_aids = ut.take_column(max_scores, 1)
        return ranked_aids
        # aid = infr.aid_to_node[node]
        # node_to

    def _get_cm_edge_data(infr, edges):
        symmetric = True

        # Find scores for the edges that exist in the graph
        edge_to_data = ut.ddict(dict)
        node_to_cm = {infr.aid_to_node[cm.qaid]:
                      cm for cm in infr.cm_list}
        for u, v in edges:
            if symmetric:
                u, v = e_(u, v)
            cm1 = node_to_cm.get(u, None)
            cm2 = node_to_cm.get(v, None)
            scores = []
            ranks = []
            for cm in ut.filter_Nones([cm1, cm2]):
                for node in [u, v]:
                    aid = infr.node_to_aid[node]
                    idx = cm.daid2_idx.get(aid, None)
                    if idx is None:
                        continue
                    score = cm.annot_score_list[idx]
                    rank = cm.get_annot_ranks([aid])[0]
                    scores.append(score)
                    ranks.append(rank)
            if len(scores) == 0:
                score = None
                rank = None
            else:
                rank = vt.safe_min(ranks)
                score = np.nanmean(scores)
            edge_to_data[(u, v)]['score'] = score
            edge_to_data[(u, v)]['rank'] = rank
        return edge_to_data

    @profile
    def apply_match_scores(infr):
        """

        Applies precomputed matching scores to edges that already exist in the
        graph. Typically you should run infr.apply_match_edges() before running
        this.

        CommandLine:
            python -m ibeis.algo.hots.graph_iden apply_match_scores --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.get_edge_attrs('score')
        """
        if infr.cm_list is None:
            infr.print('apply_match_scores - no scores to apply!')
            return
        infr.print('apply_match_scores', 1)
        edges = list(infr.graph.edges())
        edge_to_data = infr._get_cm_edge_data(edges)

        # Remove existing attrs
        ut.nx_delete_edge_attr(infr.graph, 'score')
        ut.nx_delete_edge_attr(infr.graph, 'rank')
        ut.nx_delete_edge_attr(infr.graph, 'normscore')

        edges = list(edge_to_data.keys())
        edge_scores = list(ut.take_column(edge_to_data.values(), 'score'))
        edge_scores = ut.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        edge_ranks = np.array(ut.take_column(edge_to_data.values(), 'rank'))
        # take the inf-norm
        normscores = edge_scores / vt.safe_max(edge_scores, nans=False)

        # Add new attrs
        infr.set_edge_attrs('score', ut.dzip(edges, edge_scores))
        infr.set_edge_attrs('rank', ut.dzip(edges, edge_ranks))

        # Hack away zero probabilites
        # probs = np.vstack([p_nomatch, p_match, p_notcomp]).T + 1e-9
        # probs = vt.normalize(probs, axis=1, ord=1, out=probs)
        # entropy = -(np.log2(probs) * probs).sum(axis=1)
        infr.set_edge_attrs('normscore', dict(zip(edges, normscores)))


class InfrLearning(object):
    def learn_evaluataion_clasifiers(infr):
        infr.print('learn_evaluataion_clasifiers')
        from ibeis.scripts.script_vsone import OneVsOneProblem
        pblm = OneVsOneProblem.from_aids(infr.ibs, aids=infr.aids, verbose=True)
        pblm.primary_task_key = 'match_state'
        pblm.default_clf_key = 'RF'
        pblm.default_data_key = 'learn(sum,glob,4)'
        pblm.load_features()
        pblm.load_samples()
        pblm.build_feature_subsets()

        # pblm.evaluate_simple_scores(task_keys)
        feat_cfgstr = ut.hashstr_arr27(
            pblm.samples.X_dict['learn(all)'].columns.values, 'matchfeat')
        cfg_prefix = (pblm.samples.make_sample_hashid() +
                      pblm.qreq_.get_cfgstr() + feat_cfgstr)
        pblm.learn_evaluation_classifiers(cfg_prefix=cfg_prefix)
        infr.classifiers = pblm
        pass


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
        # Refresh the name labels
        for nid, cc in infr.pos_graph._ccs.items():
            infr.set_node_attrs('name_label', ut.dzip(cc, [nid]))

        # do LNBNN query for new edges
        # Use one-vs-many to establish candidate edges to classify
        infr.exec_matching(cfgdict={
            'resize_dim': 'width',
            'dim_size': 700,
            'requery': True,
            'can_match_samename': False,
            'can_match_sameimg': False,
            # 'sv_on': False,
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

        infr.print('vsmany found %d/%d new edges' % (
            len(candidate_edges), len(candidate_edges) +
            len(already_reviewed)), 1)
        return candidate_edges

    @profile
    def find_pos_redun_candidate_edges(infr):
        # Add random edges between exisiting non-redundant PCCs
        candidate_edges = set([])
        for pcc in infr.non_pos_redundant_pccs(relax_size=True):
            sub = infr.graph.subgraph(pcc)

            # Get edges between biconnected (nodes) components
            sub_comp = nx.complement(sub)
            bicon = list(nx.biconnected_components(sub))
            check_edges = set([])
            for c1, c2 in it.combinations(bicon, 2):
                check_edges.update(edges_cross(sub_comp, c1, c2))
            # Very agressive, need to tone down
            check_edges = set(it.starmap(e_, check_edges))
            # check_edges = set(it.starmap(e_, nx.complement(sub).edges()))
            candidate_edges.update(check_edges)
        return candidate_edges

    @profile
    def find_neg_redun_candidate_edges(infr):
        candidate_edges = set([])
        for c1, c2, check_edges in infr.non_complete_pcc_pairs():
            candidate_edges.update(check_edges)
        return candidate_edges

    @profile
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
        new_edges = {
            edge for edge in candidate_edges if not infr.graph.has_edge(*edge)
        }
        return new_edges

    def apply_edge_truth(infr, edges):
        edge_truth_df = infr.match_state_df(edges)
        edge_truth = edge_truth_df.idxmax(axis=1).to_dict()
        infr.set_edge_attrs('truth', edge_truth)
        infr.edge_truth.update(edge_truth)

    @profile
    def add_new_candidate_edges(infr, new_edges):
        new_edges = list(new_edges)
        if len(new_edges) == 0:
            return

        infr.graph.add_edges_from(new_edges)
        infr.set_edge_attrs('num_reviews', ut.dzip(new_edges, [0]))

        if infr.test_mode:
            infr.apply_edge_truth(new_edges)

        if infr.classifiers:
            infr.print('Prioritizing edges with one-vs-one probabilities', 1)
            # Construct pairwise features on edges in infr
            # needs_probs = infr.get_edges_where_eq('task_probs', None,
            #                                       edges=new_edges,
            #                                       default=None)
            task_probs = infr._make_task_probs(new_edges)

            primary_task = 'match_state'
            primary_probs = task_probs[primary_task]
            primary_thresh = infr.task_thresh[primary_task]
            prob_match = primary_probs['match']

            default_priority = prob_match.copy()
            # Give negatives that pass automatic thresholds high priority
            if True:
                _probs = task_probs[primary_task]['nomatch']
                flags = _probs > primary_thresh['nomatch']
                default_priority[flags] = np.maximum(default_priority[flags],
                                                     _probs[flags])

            # Give not-comps that pass automatic thresholds high priority
            if True:
                _probs = task_probs[primary_task]['notcomp']
                flags = _probs > primary_thresh['notcomp']
                default_priority[flags] = np.maximum(default_priority[flags],
                                                     _probs[flags])

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
            infr.print('Prioritizing edges with one-vs-vsmany scores', 1)
            # Not given any deploy classifier, this is the best we can do
            infr.task_probs = None
            scores = infr._make_lnbnn_scores(new_edges)
            infr.set_edge_attrs('normscore', ut.dzip(new_edges, scores))
            infr.queue.update(ut.dzip(new_edges, -scores))

    @profile
    def refresh_candidate_edges(infr, ranking=True):
        """
        Search for candidate edges.
        Assign each edge a priority and add to queue.
        """
        infr.print('refresh_candidate_edges', 1)

        infr.assert_consistency_invariant()
        infr.refresh.reset()
        new_edges = infr.find_new_candidate_edges(ranking=ranking)
        infr.add_new_candidate_edges(new_edges)
        infr.assert_consistency_invariant()

    @profile
    def _make_task_probs(infr, edges):
        pblm = infr.classifiers
        data_key = pblm.default_data_key
        # TODO: find a good way to cache this
        cfgstr = infr.ibs.dbname + ut.hashstr27(repr(edges)) + data_key
        cacher = ut.Cacher('foobarclf_taskprobs', cfgstr=cfgstr,
                           appname=pblm.appname, enabled=1,
                           verbose=pblm.verbose)
        X = cacher.tryload()
        if X is None:
            X = pblm.make_deploy_features(infr, edges, data_key)
            cacher.save(X)
        task_keys = list(pblm.samples.subtasks.keys())
        task_probs = pblm.predict_proba_deploy(X, task_keys)
        return task_probs

    @profile
    def _make_lnbnn_scores(infr, edges):
        edge_to_data = infr._get_cm_edge_data(edges)
        edges = list(edge_to_data.keys())
        edge_scores = list(ut.take_column(edge_to_data.values(), 'score'))
        edge_scores = ut.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        # take the inf-norm
        normscores = edge_scores / vt.safe_max(edge_scores, nans=False)
        return normscores


class InfrDynamicSubroutines(object):
    """
    # 12 total possible states

    # details of these states.
    POSITIVE, WITHIN, CONSISTENT
        * pos-within never changes PCC status
        * never introduces inconsistency
        * might add pos-redun
    POSITIVE, WITHIN, INCONSISTENT
        * pos-within never changes PCC status
        * might fix inconsistent edge
    POSITIVE, BETWEEN, BOTH_CONSISTENT
        * pos-between edge always does merge
    POSITIVE, BETWEEN, ANY_INCONSISTENT
        * pos-between edge always does merge
        * pos-between never fixes inconsistency

    NEGATIVE, WITHIN, CONSISTENT
        * might split PCC, results will be consistent
        * might causes an inconsistency
    NEGATIVE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    NEGATIVE, BETWEEN, BOTH_CONSISTENT
        * might add neg-redun
    NEGATIVE, BETWEEN, ANY_INCONSISTENT
        * might add to incon-neg-external
        * neg-redun not tracked for incon.

    INCOMPARABLE, WITHIN, CONSISTENT
        * might remove pos-redun
        * might split PCC, results will be consistent
    INCOMPARABLE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    INCOMPARABLE, BETWEEN, BOTH_CONSISTENT
        * might remove neg-redun
    INCOMPARABLE, BETWEEN, ANY_INCONSISTENT
        * might remove incon-neg-external
    """

    def _new_inconsistency(infr, nid, from_):
        cc = infr.pos_graph.component(nid)
        pos_edges = infr.pos_graph.edges(cc)
        infr.recover_graph.add_edges_from(pos_edges)
        # num = len(list(nx.connected_components(infr.recover_graph)))
        num = infr.recover_graph.number_of_components()
        msg = 'New inconsistency from {}, {} total'.format(from_, num)
        infr.print(msg, color='red')
        infr._check_inconsistency(nid, cc=cc)

    def _check_inconsistency(infr, nid, cc=None):
        if cc is None:
            cc = infr.pos_graph.component(nid)
        # infr.print('Checking consistency of {}'.format(nid))
        pos_subgraph = infr.pos_graph.subgraph(cc)
        if not nx.is_connected(pos_subgraph):
            print('cc = %r' % (cc,))
            print('pos_subgraph = %r' % (pos_subgraph,))
            raise AssertionError('must be connected')
        neg_edges = list(infr.neg_graph.subgraph(cc).edges())

        old_error_edges = infr.error_edges2.pop(nid, [])

        # Remove priority from old error edges
        infr.set_edge_attrs('maybe_error', ut.dzip(old_error_edges, [None]))
        if infr.queue is not None:
            for error_edge in old_error_edges:
                infr.queue[error_edge] = 0

        if neg_edges:
            hypothesis = dict(infr.hypothesis_errors(
                pos_subgraph, neg_edges))
            assert len(hypothesis) > 0, 'must have at least one'
            new_error_edges = set(hypothesis.keys())

            # flag error edges
            infr.error_edges2[nid] = new_error_edges
            # infr.print('Add new error edges {}'.format(infr.error_edges))

            # choose one and give it insanely high priority
            infr.set_edge_attrs('maybe_error', ut.dzip(new_error_edges, [True]))
            if infr.queue is not None:
                for error_edge in new_error_edges:
                    data = infr.graph.get_edge_data(*error_edge)
                    base = data.get('prob_match', 1e-9)
                    infr.queue[error_edge] = -(10 + base)
        else:
            infr.recover_graph.remove_nodes_from(cc)
            num = infr.recover_graph.number_of_components()
            # num = len(list(nx.connected_components(infr.recover_graph)))
            msg = ('An inconsistent PCC recovered, '
                   '{} inconsistent PCC(s) remain').format(num)
            infr.print(msg, color='green')
            infr.update_pos_redun(nid, force=True)
            infr.update_extern_neg_redun(nid, force=True)

    def _positive_decision(infr, edge):
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1 = infr.recover_graph.has_node(edge[0])
        incon2 = infr.recover_graph.has_node(edge[1])
        any_inconsistent = (incon1 or incon2)
        all_consistent = not any_inconsistent
        was_within = nid1 == nid2

        if was_within:
            infr._add_review_edge(edge, 'match')
            if all_consistent:
                # infr.print('Internal consistent positive review')
                infr.print('pos-within-clean',)
                infr.update_pos_redun(nid1, may_remove=False)
            else:
                # infr.print('Internal inconsistent positive review')
                infr.print('pos-within-dirty',)
                infr._check_inconsistency(nid1)
        else:
            # infr.print('Merge case')
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)

            infr.purge_redun_flags(nid1, nid2)

            if any_inconsistent:
                # infr.print('Inconsistent merge',)
                infr.print('pos-between-dirty-merge',)
                if not incon1:
                    recover_edges = list(infr.pos_graph.subgraph(cc1).edges())
                else:
                    recover_edges = list(infr.pos_graph.subgraph(cc2).edges())
                infr.recover_graph.add_edges_from(recover_edges)
                infr._add_review_edge(edge, 'match')
                infr.recover_graph.add_edge(*edge)
            elif any(edges_cross(infr.neg_graph, cc1, cc2)):
                # infr.print('Merge creates inconsistency',)
                infr.print('pos-between-clean-merge-dirty',)
                infr._add_review_edge(edge, 'match')
                new_nid = infr.pos_graph.node_label(edge[0])
                infr._new_inconsistency(new_nid, 'positive')
            else:
                # infr.print('Consistent merge')
                infr.print('pos-between-clean-merge-clean',)
                infr._add_review_edge(edge, 'match')
                new_nid = infr.pos_graph.node_label(edge[0])
                infr.update_extern_neg_redun(new_nid, may_remove=False)
                infr.update_pos_redun(new_nid, may_remove=False)

    def _negative_decision(infr, edge):
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1 = infr.recover_graph.has_node(edge[0])
        incon2 = infr.recover_graph.has_node(edge[1])
        all_consistent = not (incon1 or incon2)

        infr._add_review_edge(edge, 'nomatch')
        new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)

        was_within = nid1 == nid2
        was_split = was_within and new_nid1 != new_nid2

        if was_within:
            if was_split:
                if all_consistent:
                    # infr.print('Consistent split from negative')
                    infr.print('neg-within-split-clean',)
                    infr.purge_redun_flags(nid1)
                    infr.update_pos_redun(new_nid1, may_remove=False)
                    infr.update_pos_redun(new_nid2, may_remove=False)
                    infr.update_extern_neg_redun(new_nid1, may_remove=False)
                    infr.update_extern_neg_redun(new_nid2, may_remove=False)
                else:
                    # infr.print('Inconsistent split from negative')
                    infr.print('neg-within-split-dirty',)
                    if infr.recover_graph.has_edge(*edge):
                        infr.recover_graph.remove_edge(*edge)
                    infr.purge_redun_flags(nid1)
                    infr._check_inconsistency(new_nid1)
                    infr._check_inconsistency(new_nid2)
            else:
                if all_consistent:
                    # infr.print('Negative added within clean PCC')
                    infr.print('neg-within-clean',)
                    infr.purge_redun_flags(nid1)
                    infr._new_inconsistency(nid1, 'negative')
                else:
                    # infr.print('Negative added within inconsistent PCC')
                    infr.print('neg-within-dirty',)
                    pass
        else:
            if all_consistent:
                # infr.print('Negative added between consistent PCCs')
                infr.print('neg-between-clean',)
                infr.update_neg_redun(nid1, nid2, may_remove=False)
            else:
                # infr.print('Negative added external to inconsistent PCC')
                infr.print('neg-between-dirty',)
                # nothing to do if a negative edge is added between two PCCs
                # where at least one is inconsistent
                pass

    def _incomp_decision(infr, edge):
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1 = infr.recover_graph.has_node(edge[0])
        incon2 = infr.recover_graph.has_node(edge[1])
        all_consistent = not (incon1 or incon2)

        was_within = nid1 == nid2

        overwrote_positive = infr.pos_graph.has_edge(*edge)
        overwrote_negative = infr.neg_graph.has_edge(*edge)

        if was_within:
            infr._add_review_edge(edge, 'notcomp')
            if overwrote_positive:
                # changed an existing positive edge
                if infr.recover_graph.has_edge(*edge):
                    infr.recover_graph.remove_edge(*edge)
                new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)
                was_split = new_nid1 != new_nid2
                if was_split:
                    old_nid = nid1
                    prev_neg_nids = infr.purge_redun_flags(old_nid)
                    if all_consistent:
                        # infr.print('Split CC from incomparable')
                        infr.print('incon-within-pos-split-clean',)
                        # split case
                        for other_nid in prev_neg_nids:
                            infr.update_neg_redun(new_nid1, other_nid)
                            infr.update_neg_redun(new_nid2, other_nid)
                        infr.update_neg_redun(new_nid1, new_nid2)
                        infr.update_pos_redun(new_nid1, may_remove=False)
                        infr.update_pos_redun(new_nid2, may_remove=False)
                    else:
                        # infr.print('Split inconsistent CC from incomparable')
                        infr.print('incon-within-pos-split-dirty',)
                        if infr.recover_graph.has_edge(*edge):
                            infr.recover_graph.remove_edge(*edge)
                        infr._check_inconsistency(new_nid1)
                        infr._check_inconsistency(new_nid2)
                elif all_consistent:
                    # infr.print('Overwrote pos in CC with incomp')
                    infr.print('incon-within-pos-clean',)
                    infr.update_pos_redun(new_nid1, may_add=False)
                else:
                    # infr.print('Overwrote pos in inconsistent CC with incomp')
                    infr.print('incon-within-pos-dirty',)
                    # Overwriting a positive edge that is not a split
                    # in an inconsistent component, means no inference.
                    pass
            elif overwrote_negative:
                # infr.print('Overwrite negative within CC')
                infr.print('incon-within-neg-dirty',)
                assert not all_consistent
                infr._check_inconsistency(nid1)
            else:
                if all_consistent:
                    infr.print('incon-within-clean',)
                    # infr.print('Incomp edge within consistent CC')
                else:
                    infr.print('incon-within-dirty',)
                    # infr.print('Incomp edge within inconsistent CC')
        else:
            infr._add_review_edge(edge, 'notcomp')
            if overwrote_negative:
                if all_consistent:
                    # changed and existing negative edge only influences
                    # consistent pairs of PCCs
                    # infr.print('Overwrote neg edge between CCs')
                    infr.print('incon-between-neg-clean',)
                    infr.update_neg_redun(nid1, nid2, may_add=False)
                else:
                    infr.print('incon-between-neg-dirty',)
                    # infr.print('Overwrote pos edge between incon CCs')
            else:
                infr.print('incon-between',)
                # infr.print('Incomp edge between CCs')

    @profile
    def dynamic_inference(infr, edge, decision):
        if decision == 'match':
            infr._positive_decision(edge)
        elif decision == 'nomatch':
            infr._negative_decision(edge)
        elif decision == 'incomp':
            infr._incomp_decision(edge)
        else:
            assert False
        # print('infr.recover_graph = %r' % (infr.recover_graph,))


class InfrRecovery2(object):
    """ recovery funcs """

    def is_recovering(infr):
        return len(infr.recover_graph) > 0

    @profile
    def hypothesis_errors(infr, pos_subgraph, neg_edges):
        if not nx.is_connected(pos_subgraph):
            raise AssertionError('Not connected' + repr(pos_subgraph))
        infr.print(
            'Find hypothesis errors in {} nodes with {} neg edges'.format(
                len(pos_subgraph), len(neg_edges)), 2)

        pos_edges = list(pos_subgraph.edges())

        # Generate weights for edges
        default = 0
        # default = 1e-6
        pos_gen = infr.gen_edge_values('prob_match', pos_edges, default=default)
        neg_gen = infr.gen_edge_values('prob_match', neg_edges, default=default)
        pos_prob = list(pos_gen)
        neg_prob = list(neg_gen)
        pos_n = list(infr.gen_edge_values('num_reviews', pos_edges, default=0))
        neg_n = list(infr.gen_edge_values('num_reviews', neg_edges, default=0))
        pos_weight = pos_n
        neg_weight = neg_n
        pos_weight = np.add(pos_prob, np.array(pos_n))
        neg_weight = np.add(neg_prob, np.array(neg_n))
        capacity = 'weight'
        nx.set_edge_attributes(pos_subgraph, capacity,
                               ut.dzip(pos_edges, pos_weight))

        # Solve a multicut problem for multiple pairs of terminal nodes.
        # Running multiple min-cuts produces a k-factor approximation
        maybe_error_edges = set([])
        for (s, t), join_weight in zip(neg_edges, neg_weight):
            cut_weight, parts = nx.minimum_cut(pos_subgraph, s, t,
                                               capacity=capacity)
            cut_edgeset = edges_cross(pos_subgraph, *parts)
            # cut_edgeset_weight = sum([
            #     pos_subgraph.get_edge_data(u, v)[capacity]
            #     for u, v in cut_edgeset])
            # infr.print('cut_weight = %r' % (cut_weight,), 3)
            # infr.print('join_weight = %r' % (join_weight,), 3)
            if join_weight < cut_weight:
                join_edgeset = {(s, t)}
                chosen = join_edgeset
                hypothesis = 'match'
            else:
                chosen = cut_edgeset
                hypothesis = 'nomatch'
            for edge in chosen:
                if edge not in maybe_error_edges:
                    maybe_error_edges.add(edge)
                    yield (edge, hypothesis)
        # return maybe_error_edges


class DynamicUpdate2(object):

    def _add_review_edges_from(infr, edges, decision):
        infr.print('add decision=%r from %d reviews' % (
            decision, len(edges)), 1)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edges_from(edges)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                # print('replaced edge from %r graph' % (k,))
                G.remove_edges_from(edges)

    def _add_review_edge(infr, edge, decision):
        infr.print('add review edge=%r, decision=%r' % (edge, decision), 20)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edge(*edge)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                if G.has_edge(*edge):
                    # print('replaced edge from %r graph' % (k,))
                    G.remove_edge(*edge)

    def purge_redun_flags(infr, *nids):
        """
        Removes positive and negative redundancy from nids and all other PCCs
        touching nids respectively. Return the external PCC nids.
        """
        neighbs = (infr.neg_redun_nids.neighbors(nid) for nid in nids
                   if infr.neg_redun_nids.has_node(nid))
        prev_neg_nids = set(ut.iflatten(neighbs))
        prev_neg_nids -= set(nids)
        infr.neg_redun_nids.remove_nodes_from(nids)
        infr.pos_redun_nids.difference_update(set(nids))
        return prev_neg_nids

    def update_extern_neg_redun(infr, nid, may_add=True, may_remove=True,
                                force=False):
        k_neg = infr.queue_params['neg_redundancy']
        cc1 = infr.pos_graph.component(nid)
        if force or True:
            # TODO: non-force versions
            freqs = infr.find_external_neg_nid_freq(cc1)
            for other_nid, freq in freqs.items():
                if freq >= k_neg:
                    cc2 = infr.pos_graph.component(other_nid)
                    infr.neg_redun_nids.add_edge(nid, other_nid)
                    if infr.queue is not None:
                        infr.remove_between_priority(cc1, cc2)
                elif may_remove:
                    try:
                        infr.neg_redun_nids.remove_edge(nid, other_nid)
                    except nx.exception.NetworkXError:
                        pass
                    else:
                        if infr.queue is not None:
                            infr.reinstate_between_priority(cc1, cc2)

    @profile
    def update_neg_redun(infr, nid1, nid2, may_add=True, may_remove=True,
                         force=False):
        """
        Checks if two PCCs are newly or no longer negative redundant.
        Edges are either removed or added to the queue appropriately.
        """
        need_add = False
        need_remove = False
        force = True
        if force:
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)
            need_add = infr.is_neg_redundant(cc1, cc2)
            need_remove = not need_add
        else:
            was_neg_redun = infr.neg_redun_nids.has_edge(nid1, nid2)
            if may_add and not was_neg_redun:
                cc1 = infr.pos_graph.component(nid1)
                cc2 = infr.pos_graph.component(nid2)
                need_add = infr.is_neg_redundant(cc1, cc2)
            elif may_remove and not was_neg_redun:
                need_remove = not infr.is_neg_redundant(cc1, cc2)
        if need_add:
            # Flag ourselves as negative redundant and remove priorities
            infr.neg_redun_nids.add_edge(nid1, nid2)
            if infr.queue is not None:
                infr.remove_between_priority(cc1, cc2)
        elif need_remove:
            try:
                infr.neg_redun_nids.remove_edge(nid1, nid2)
            except nx.exception.NetworkXError:
                pass
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)
            if infr.queue is not None:
                infr.reinstate_between_priority(cc1, cc2)

    def update_pos_redun(infr, nid, may_add=True, may_remove=True,
                         force=False):
        """
        Checks if a PCC is newly, or no longer positive redundant.
        Edges are either removed or added to the queue appropriately.
        """
        need_add = False
        need_remove = False
        if force:
            cc = infr.pos_graph.component(nid)
            need_add = infr.is_pos_redundant(cc)
            need_remove = not need_add
        else:
            was_pos_redun = nid in infr.pos_redun_nids
            if may_add and not was_pos_redun:
                cc = infr.pos_graph.component(nid)
                need_add = infr.is_pos_redundant(cc)
            elif may_remove and not was_pos_redun:
                need_remove = not infr.is_pos_redundant(cc)
        if need_add:
            # If checks pass, flag nid as pos-redun
            infr.pos_redun_nids.add(nid)
            if infr.queue is not None:
                infr.remove_internal_priority(cc)
        elif need_remove:
            # If the checks fails then, unflag the nid
            infr.pos_redun_nids -= {nid}
            if infr.queue is not None:
                infr.reinstate_internal_priority(cc)

    def remove_internal_priority(infr, cc):
        infr.queue.delete_items(edges_inside(infr.graph, cc))

    def remove_external_priority(infr, cc):
        infr.queue.delete_items(edges_outgoing(infr.graph, cc))

    def remove_between_priority(infr, cc1, cc2):
        infr.queue.delete_items(edges_cross(infr.graph, cc1, cc2))

    def reinstate_between_priority(infr, cc1, cc2):
        # Reinstate the appropriate edges into the queue
        edges = edges_cross(infr.unreviewed_graph, cc1, cc2)
        infr.reinstate_edge_priority(edges)

    def reinstate_internal_priority(infr, cc):
        # Reinstate the appropriate edges into the queue
        edges = edges_inside(infr.unreviewed_graph, cc)
        infr.reinstate_edge_priority(edges)

    def reinstate_external_priority(infr, cc):
        # Reinstate the appropriate edges into the queue
        edges = edges_outgoing(infr.unreviewed_graph, cc)
        infr.reinstate_edge_priority(edges)

    def reinstate_edge_priority(infr, edges):
        prob_match = np.array(list(infr.gen_edge_values(
            'prob_match', edges, default=1e-9)))
        priority = -prob_match
        infr.queue.update(ut.dzip(edges, priority))


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInfrRedundancy(object):
    """ methods for computing redundancy """

    @profile
    def rand_neg_check_edges(infr, c1_nodes, c2_nodes):
        """
        Find enough edges to between two pccs to make them k-negative complete
        """
        k = infr.queue_params['neg_redundancy']
        existing_edges = edges_cross(infr.graph, c1_nodes, c2_nodes)
        reviewed_edges = {
            edge: state
            for edge, state in infr.get_edge_attrs(
                'decision', existing_edges,
                default='unreviewed').items()
            if state != 'unreviewed'
        }
        n_neg = sum([state == 'nomatch'
                     for state in reviewed_edges.values()])
        if n_neg < k:
            # Find k random negative edges
            check_edges = existing_edges - set(reviewed_edges)
            if len(check_edges) < k:
                for edge in it.starmap(e_, it.product(c1_nodes, c2_nodes)):
                    if edge not in reviewed_edges:
                        check_edges.add(edge)
                        if len(check_edges) == k:
                            break
        else:
            check_edges = {}
        return check_edges

    def find_external_neg_nids(infr, cc):
        """
        Find the nids with at least one negative edge external
        to this cc.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        out_neg_nids = set([])
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                out_neg_nids.add(nid2)
        return out_neg_nids

    def find_external_neg_nid_freq(infr, cc):
        """
        Find the number of edges leaving `cc` and directed towards specific
        names.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        neg_nid_freq = ut.ddict(lambda: 0)
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                neg_nid_freq[nid2] += 1
        return neg_nid_freq

    @profile
    def negative_redundant_nids(infr, cc):
        """
        Get PCCs that are k-negative redundant with `cc`

            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> infr = testdata_infr2()
            >>> node = 20
            >>> cc = infr.pos_graph.connected_to(node)
            >>> infr.queue_params['neg_redundancy'] = 2
            >>> infr.negative_redundant_nids(cc)
        """
        neg_nid_freq = infr.find_external_neg_nid_freq(cc)
        # check for k-negative redundancy
        k_neg = infr.queue_params['neg_redundancy']
        pos_graph = infr.pos_graph
        neg_nids = [
            nid2 for nid2, freq in neg_nid_freq.items()
            if (
                freq >= k_neg or
                freq == len(cc) or
                freq == len(pos_graph.connected_to(nid2))
            )
        ]
        return neg_nids

    @profile
    def prob_complete(infr, cc):
        if infr.term is None:
            assert False
            return 0
        else:
            size = len(cc)
            # Choose most appropriate phi
            if size not in infr.term.phis:
                size = max(infr.term.phis.keys())
            phi = infr.term.phis[size]
            # pos_graph.node_label()
            num_ccs = len(infr.pos_graph._ccs)
            # We use annot scores because names could be different if
            # reviews have happened.
            ranked_aids = infr._get_cm_agg_aid_ranking(cc)
            # Map these aids onto current nid label
            ranked_nids = ut.unique(
                [infr.pos_graph.node_label(aid) for aid in ranked_aids])
            nid_to_rank = ut.make_index_lookup(ranked_nids)
            neg_nid_neighbors = set(infr.negative_redundant_nids(cc))
            # Get the ranks of known negative neighbors
            neg_ranks = [rank for nid, rank in nid_to_rank.items()
                         if nid in neg_nid_neighbors]
            neg_ranks = sorted(neg_ranks)
            slack = num_ccs - len(phi)
            if slack:
                phi = np.append(phi, [phi[-1]] * slack)
                phi = phi / phi.sum()
            # TODO: extend phi if needed for current dbsize
            p_complete = sum([phi[r] for r in neg_ranks])
            return p_complete

    @profile
    def check_prob_completeness(infr, node):
        """
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> infr = testdata_infr2()
            >>> infr.initialize_visual_node_attrs()
            >>> #ut.ensureqt()
            >>> #infr.show()
            >>> infr.refresh_candidate_edges()
            >>> node = 1
            >>> node = 20
            >>> infr.is_node_complete(node)
        """
        thresh = infr.queue_params['complete_thresh']
        cc = infr.pos_graph.connected_to(node)
        if thresh < 1.0:
            p_complete = infr.prob_complete(cc)
            if p_complete > thresh:
                return True
        return False

    @profile
    def non_complete_pcc_pairs(infr):
        """
        Get pairs of PCCs that are not complete.
        Finds edges that might complete them.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> infr = testdata_infr2()
            >>> categories = infr.categorize_edges(graph)
            >>> negative = categories['negative']
            >>> ne, edges = #list(categories['reviewed_negatives'].items())[0]
            >>> infr.graph.remove_edges_from(edges)
            >>> cc1, cc2, _edges = list(infr.non_complete_pcc_pairs())[0]
            >>> result = non_complete_pcc_pairs(infr)
            >>> print(result)

        """
        thresh = infr.queue_params['complete_thresh']
        pcc_set = list(infr.positive_connected_compoments())
        # Remove anything under the probabilistic threshold
        if thresh < 1.0:
            pcc_set = [
                c1 for c1 in pcc_set if
                infr.prob_complete(c1) < thresh
            ]
        else:
            assert False
        # Loop through all pairs
        for c1_nodes, c2_nodes in it.combinations(pcc_set, 2):
            check_edges = infr.rand_neg_check_edges(c1_nodes, c2_nodes)
            if len(check_edges) > 0:
                # no check edges means we can't do anything
                yield (c1_nodes, c2_nodes, check_edges)

    @profile
    def is_pos_redundant(infr, cc, relax_size=False):
        k = infr.queue_params['pos_redundancy']
        if k == 1:
            return True  # assumes cc is connected
        else:
            # if the nodes are not big enough for this amount of connectivity
            # then we relax the requirement
            if relax_size:
                required_k = min(len(cc) - 1, k)
            else:
                required_k = k
            assert isinstance(cc, set)
            if required_k <= 1:
                return True
            if required_k == 2:
                pos_subgraph = infr.pos_graph.subgraph(cc)
                return nx.is_biconnected(pos_subgraph)
            else:
                pos_subgraph = infr.pos_graph.subgraph(cc)
                pos_conn = nx.edge_connectivity(pos_subgraph)
                return pos_conn >= required_k

    @profile
    def is_neg_redundant(infr, cc1, cc2):
        k_neg = infr.queue_params['neg_redundancy']
        neg_graph = infr.neg_graph
        # from ibeis.algo.hots.graph_iden_utils import edges_cross
        # num_neg = len(list(edges_cross(neg_graph, cc1, cc2)))
        # return num_neg >= k_neg
        neg_edge_gen = (
            1 for u in cc1 for v in cc2.intersection(neg_graph.adj[u])
        )
        # do a lazy count of bridges
        for count in neg_edge_gen:
            if count >= k_neg:
                return True

    @profile
    def pos_redundant_pccs(infr, relax_size=False):
        for cc in infr.consistent_components():
            if len(cc) == 2:
                continue
            if infr.is_pos_redundant(cc, relax_size):
                yield cc

    @profile
    def non_pos_redundant_pccs(infr, relax_size=False):
        """
        Get PCCs that are not k-positive-redundant
        """
        for cc in infr.consistent_components():
            if not infr.is_pos_redundant(cc, relax_size):
                yield cc

    def find_pos_redun_nids(infr):
        """ recomputes infr.pos_redun_nids """
        for cc in infr.pos_redundant_pccs():
            node = next(iter(cc))
            nid = infr.pos_graph.node_label(node)
            yield nid

    def find_neg_redun_nids(infr):
        """ recomputes edges in infr.neg_redun_nids """
        for cc in infr.consistent_components():
            node = next(iter(cc))
            nid1 = infr.pos_graph.node_label(node)
            for nid2 in infr.negative_redundant_nids(cc):
                if nid1 < nid2:
                    yield nid1, nid2


class AnnotInfr2(InfrRecovery2, CandidateSearch2, InfrReviewers,
                 InfrLearning, AnnotInfrMatching, AnnotInfrRedundancy,
                 SimulationHelpers, DynamicUpdate2, InfrInvariants,
                 InfrDynamicSubroutines,
                 InfrLoops):
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.graph_iden_new
        python -m ibeis.algo.hots.graph_iden_new --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
