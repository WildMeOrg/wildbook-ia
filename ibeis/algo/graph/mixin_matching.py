# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import itertools as it
import networkx as nx
import vtool as vt
from ibeis.algo.graph import nx_utils
from ibeis.algo.graph.nx_utils import e_
from ibeis.algo.graph.nx_utils import (edges_cross, ensure_multi_index)  # NOQA
from ibeis.algo.graph.state import UNREV
print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInfrMatching(object):
    """
    Methods for running matching algorithms
    """

    def exec_matching(infr, aids=None, prog_hook=None, cfgdict=None):
        """
        Loads chip matches into the inference structure
        Uses graph name labeling and ignores ibeis labeling
        """
        infr.print('exec_matching', 1)
        #from ibeis.algo.graph import graph_iden
        ibs = infr.ibs
        if aids is None:
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
        custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=infr.verbose >= 2)

        cm_list = qreq_.execute(prog_hook=prog_hook)
        infr._set_vsmany_info(qreq_, cm_list)

    def _set_vsmany_info(infr, qreq_, cm_list):
        infr.vsmany_qreq_ = qreq_
        infr.vsmany_cm_list = cm_list
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def exec_vsone(infr, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.graph.core exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
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
        """
        Adds extra domain specific local and global properties that the match
        object doesnt directly provide.
        """
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

        aid_pairs_ = [(m.annot1['aid'], m.annot2['aid']) for m in matches]
        assert aid_pairs_ == edges, 'edge ordering changed'

        return matches, X

    def exec_vsone_subset(infr, edges, config={}, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.graph.core exec_vsone

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
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
        infr.print('apply_match_edges adding %d edges' % len(edges), 1)
        infr.graph.add_edges_from(edges)
        infr.apply_match_scores()

    def _cm_breaking(infr, review_cfg={}):
        """
            >>> from ibeis.algo.graph.core import *  # NOQA
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

    def _cm_training_pairs(infr, qreq_=None, cm_list=None,
                           top_gt=2, mid_gt=2, bot_gt=2, top_gf=2,
                           mid_gf=2, bot_gf=2, rand_gt=2, rand_gf=2, rng=None):
        """
        Constructs training data for a pairwise classifier

        CommandLine:
            python -m ibeis.algo.graph.core _cm_training_pairs

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching(cfgdict={
            >>>     'can_match_samename': True,
            >>>     'K': 4,
            >>>     'Knorm': 1,
            >>>     'prescore_method': 'csum',
            >>>     'score_method': 'csum'
            >>> })
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> exec(ut.execstr_funckw(infr._cm_training_pairs))
            >>> rng = np.random.RandomState(42)
            >>> aid_pairs = np.array(infr._cm_training_pairs(rng=rng))
            >>> print(len(aid_pairs))
            >>> assert np.sum(aid_pairs.T[0] == aid_pairs.T[1]) == 0
        """
        if qreq_ is None:
            cm_list = infr.cm_list
            qreq_ = infr.qreq_
        ibs = infr.ibs
        aid_pairs = []
        dnids = qreq_.get_qreq_annot_nids(qreq_.daids)
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

    def _get_cm_edge_data(infr, edges, cm_list=None):
        symmetric = True

        if cm_list is None:
            cm_list = infr.cm_list
        # Find scores for the edges that exist in the graph
        edge_to_data = ut.ddict(dict)
        aid_to_cm = {cm.qaid: cm for cm in cm_list}
        for u, v in edges:
            if symmetric:
                u, v = e_(u, v)
            cm1 = aid_to_cm.get(u, None)
            cm2 = aid_to_cm.get(v, None)
            scores = []
            ranks = []
            for cm in ut.filter_Nones([cm1, cm2]):
                for aid in [u, v]:
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
                # score = np.nanmean(scores)
                score = np.nanmax(scores)
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
            python -m ibeis.algo.graph.core apply_match_scores --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
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

    def photobomb_samples(infr):
        edges = list(infr.edges())
        tags_list = list(infr.gen_edge_values('tags', edges=edges, default=[]))
        flags = ut.filterflags_general_tags(tags_list, has_any=['photobomb'])
        pb_edges = ut.compress(edges, flags)
        return pb_edges


class CandidateSearch(object):
    """ Search for candidate edges """
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

    def find_pos_augment_edges(infr, pcc):
        pos_k = infr.queue_params['pos_redun']
        pos_sub = infr.pos_graph.subgraph(pcc)

        # First try to augment only with unreviewed existing edges
        avail = list(nx_utils.edges_inside(infr.unreviewed_graph, pcc))
        check_edges = nx_utils.edge_connected_augmentation(
            pos_sub, pos_k, avail=avail)
        if not check_edges:
            # Allow new edges to be introduced
            full_sub = infr.graph.subgraph(pcc)
            avail += list(nx.complement(full_sub).edges())
            n_max = (len(pos_sub) * (len(pos_sub) - 1)) // 2
            n_comp = n_max - pos_sub.number_of_edges()
            if len(avail) == n_comp:
                # can use the faster algorithm
                check_edges = nx_utils.edge_connected_augmentation(
                    pos_sub, pos_k)
            else:
                # have to use the slow approximate algo
                check_edges = nx_utils.edge_connected_augmentation(
                    pos_sub, pos_k, avail=avail)
        check_edges = set(it.starmap(e_, check_edges))
        return check_edges

    @profile
    def find_pos_redun_candidate_edges(infr, verbose=False):
        r"""
        CommandLine:
            python -m ibeis.algo.graph.mixin_matching find_pos_redun_candidate_edges

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_matching import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.make_demo_infr(ccs=[(1, 2, 3, 4, 5), (7, 8, 9, 10)])
            >>> infr.add_feedback((2, 5), decision='match')
            >>> infr.add_feedback((1, 5), decision='notcomp')
            >>> infr.queue_params['pos_redun'] = 2
            >>> candidate_edges = infr.find_pos_redun_candidate_edges()
            >>> result = ('candidate_edges = %s' % (ut.repr2(candidate_edges),))
            >>> print(result)
            candidate_edges = {(1, 3), (7, 10)}
        """
        # Add random edges between exisiting non-redundant PCCs
        candidate_edges = set([])
        pcc_gen = list(infr.non_pos_redundant_pccs(relax_size=True))
        pcc_gen = ut.ProgIter(pcc_gen, enabled=verbose, freq=1, adjust=False)
        for pcc in pcc_gen:
            check_edges = infr.find_pos_augment_edges(pcc)
            candidate_edges.update(check_edges)
            # kcon_ccs = list(nx_utils.edge_connected_components(sub, pos_k))
            # bicon = list(nx.biconnected_components(sub))
            # check_edges = set([])
            # Get edges between k-edge-connected components
            # sub_comp = nx.complement(sub)
            # for c1, c2 in it.combinations(kcon_ccs, 2):
            #     check_edges.update(edges_cross(sub_comp, c1, c2))
            # Very agressive, need to tone down
            # check_edges = set(it.starmap(e_, check_edges))
            # check_edges = set(it.starmap(e_, nx.complement(sub).edges()))
            # candidate_edges.update(check_edges)
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

    def ensure_edges(infr, edges):
        """
        Adds any new edges as unreviewed edges
        """
        missing_edges = ut.compress(edges, [not infr.has_edge(e) for e in edges])
        infr.graph.add_edges_from(missing_edges, decision=UNREV, num_reviews=0)
        infr._add_review_edges_from(missing_edges, decision=UNREV)

    @profile
    def add_new_candidate_edges(infr, new_edges):
        new_edges = list(new_edges)
        infr.print('Adding %d new candidate edges' % (len(new_edges)))
        new_edges = list(new_edges)
        if len(new_edges) == 0:
            return

        infr.graph.add_edges_from(new_edges, decision=UNREV, num_reviews=0)
        infr.unreviewed_graph.add_edges_from(new_edges)

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
        elif hasattr(infr, 'dummy_matcher'):
            prob_match = infr.dummy_matcher.predict(new_edges)
            infr.set_edge_attrs('prob_match', ut.dzip(new_edges, prob_match))
            infr.queue.update(ut.dzip(new_edges, -prob_match))
        elif infr.cm_list is not None:
            infr.print('Prioritizing edges with one-vs-vsmany scores', 1)
            # Not given any deploy classifier, this is the best we can do
            infr.task_probs = None
            scores = infr._make_lnbnn_scores(new_edges)
            infr.set_edge_attrs('normscore', ut.dzip(new_edges, scores))
            infr.queue.update(ut.dzip(new_edges, -scores))
        else:
            infr.print('No information to prioritize edges')
            scores = np.zeros(len(new_edges)) + 1e-6
            infr.queue.update(ut.dzip(new_edges, -scores))

        if hasattr(infr, 'on_new_candidate_edges'):
            # hack callback for demo
            infr.on_new_candidate_edges(infr, new_edges)

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
