# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import itertools as it
import networkx as nx
import vtool as vt
from os.path import join
from ibeis.algo.graph import nx_utils as nxu
from ibeis.algo.graph.nx_utils import e_
from ibeis.algo.graph.nx_utils import (edges_cross, ensure_multi_index)  # NOQA
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
print, rrr, profile = ut.inject2(__name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotInfrMatching(object):
    """
    Methods for running matching algorithms
    """

    @profile
    def exec_matching(infr, aids=None, prog_hook=None, cfgdict=None,
                      name_method='node'):
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
        # hack for using current nids
        if name_method == 'node':
            custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        elif name_method == 'edge':
            custom_nid_lookup = {
                aid: nid for nid, cc in infr.pos_graph._ccs.items() for aid in cc
            }
        elif name_method == 'ibeis':
            custom_nid_lookup = None
        else:
            raise KeyError('Unknown name_method={}'.format(name_method))

        qreq_ = ibs.new_query_request(aids, aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=infr.verbose >= 2)

        cacher = qreq_.get_big_cacher()
        if not cacher.exists():
            pass
            # import sys
            # sys.exit(1)

        cm_list = qreq_.execute(prog_hook=prog_hook)
        infr._set_vsmany_info(qreq_, cm_list)

    def _set_vsmany_info(infr, qreq_, cm_list):
        infr.vsmany_qreq_ = qreq_
        infr.vsmany_cm_list = cm_list
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def exec_vsone(infr, prog_hook=None):
        r"""
        DEPRICATE and use _exec_pairwise_match instead

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
            # chip_match.score_name_maxcsum(request)
            cm_list.append(chip_match)

        # cm_list = qreq_.execute(parent_rowids)
        infr.vsone_qreq_ = infr.ibs.depc.new_request('vsone', qaids, daids,
                                                     cfgdict=config)
        infr.vsone_cm_list_ = cm_list
        infr.qreq_  = infr.vsone_qreq_
        infr.cm_list = cm_list

    def _exec_pairwise_match(infr, edges, config={}, prog_hook=None):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph.core import *  # NOQA
            >>> infr = testdata_infr('testdb1')
            >>> config = {}
            >>> infr.ensure_full()
            >>> edges = [(1, 2), (2, 3)]
            >>> config = {}
            >>> prog_hook = None
            >>> match_list = infr._exec_pairwise_match(edges, config)
            >>> match1, match2 = match_list
            >>> assert match1.annot2 is match2.annot1
            >>> assert match1.annot1 is not match2.annot2
        """
        edges = ut.lmap(tuple, ut.aslist(edges))
        infr.print('exec_vsone_subset')
        qaids = ut.take_column(edges, 0)
        daids = ut.take_column(edges, 1)
        # TODO: ensure feat/chip configs are resepected
        match_list = infr.ibs.depc.get('pairwise_match', (qaids, daids),
                                       'match', config=config)

        # Hack: Postprocess matches to re-add annotation info in lazy-dict format
        from ibeis import core_annots
        config = ut.hashdict(config)
        ibs = infr.ibs
        qannot_cfg = dannot_cfg = config
        preload = True
        configured_lazy_annots = core_annots.make_configured_annots(
            ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=preload)
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

        config = {
            'K': 1,
            'Knorm': 3,
            'affine_invariance': True,
            'augment_orientation': True,
            'checks': 20,
            'ratio_thresh': 0.8,
            'refine_method': 'homog',
            'sv_on': True,
            'sver_xy_thresh': 0.01,
            'symmetric': True,
            'weight': 'fgweights'
        }
        need_lnbnn = False
        global_keys = ['gps', 'qual', 'time', 'yaw']
        prog_hook = None
        """
        if global_keys is None:
            raise ValueError('specify global keys')
            # global_keys = ['yaw', 'qual', 'gps', 'time']
            # global_keys = ['view', 'qual', 'gps', 'time']
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

    def _make_pairwise_features(infr, edges, config={}, pairfeat_cfg={},
                                global_keys=None, need_lnbnn=True,
                                multi_index=True):
        """
        Construct matches and their pairwise features

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_mtest_infr()
            >>> config = {
            >>>     'K': 1,
            >>>     'Knorm': 3,
            >>>     'affine_invariance': True,
            >>>     'augment_orientation': True,
            >>>     'checks': 20,
            >>>     'ratio_thresh': 0.8,
            >>>     'refine_method': 'homog',
            >>>     'sv_on': True,
            >>>     'sver_xy_thresh': 0.01,
            >>>     'symmetric': True,
            >>>     'weight': 'fgweights'
            >>> }
            >>> need_lnbnn = False
            >>> #global_keys = ['gps', 'qual', 'time', 'yaw']
            >>> global_keys = ['gps', 'qual', 'time', 'view']
            >>> local_keys =  [
            >>>     'fgweights', 'match_dist', 'norm_dist', 'norm_x1', 'norm_x2',
            >>>     'norm_y1', 'norm_y2', 'ratio_score', 'scale1', 'scale2',
            >>>     'sver_err_ori', 'sver_err_scale', 'sver_err_xy',
            >>>     'weighted_norm_dist', 'weighted_ratio']
            >>> pairfeat_cfg = {
            >>>     'bin_key': 'ratio',
            >>>     'bins': [0.6, 0.7, 0.8],
            >>>     'indices': [],
            >>>     'local_keys': local_keys,
            >>>     'sorters': [],
            >>>     'summary_ops': ['len', 'mean', 'sum']
            >>> }
            >>> multi_index = True
            >>> edges = [(1, 2), (2, 3)]
            >>> matches, X = infr._make_pairwise_features(
            >>>     edges, config, pairfeat_cfg, global_keys, need_lnbnn,
            >>>     multi_index)
            >>> featinfo = vt.AnnotPairFeatInfo(X.columns)
            >>> print(featinfo.get_infostr())
            >>> match = matches[0]
            >>> match._make_global_feature_vector(global_keys)
        """
        import pandas as pd
        # TODO: ensure feat/chip configs are resepected
        edges = ut.lmap(tuple, ut.aslist(edges))
        if len(edges) == 0:
            return [], []

        matches = infr._enriched_pairwise_matches(edges, config=config,
                                                  global_keys=global_keys,
                                                  need_lnbnn=need_lnbnn)
        # ---------------
        # Try different feature constructions
        infr.print('building pairwise features')
        pairfeat_cfg['summary_ops'] = set(pairfeat_cfg['summary_ops'])
        X = pd.DataFrame([
            m.make_feature_vector(**pairfeat_cfg)
            for m in ut.ProgIter(matches, label='making pairwise feats')
        ])
        if multi_index:
            # Index features by edges
            uv_index = nxu.ensure_multi_index(edges, ('aid1', 'aid2'))
            X.index = uv_index
        X[pd.isnull(X)] = np.nan
        X[np.isinf(X)] = np.nan
        # Re-order column names to ensure dimensions are consistent
        X = X.reindex_axis(sorted(X.columns), axis=1)

        # hack to fix feature validity
        if np.any(np.isinf(X['global(speed)'])):
            flags = np.isinf(X['global(speed)'])
            numer = X.loc[flags, 'global(gps_delta)']
            denom = X.loc[flags, 'global(time_delta)']
            newvals = np.full(len(numer), np.nan)
            newvals[(numer == 0) & (denom == 0)] = 0
            X.loc[flags, 'global(speed)'] = newvals

        aid_pairs_ = [(m.annot1['aid'], m.annot2['aid']) for m in matches]
        assert aid_pairs_ == edges, 'edge ordering changed'

        return matches, X

    def exec_vsone_subset(infr, edges, config={}, prog_hook=None):
        r"""
        Args:
            prog_hook (None): (default = None)

        CommandLine:
            python -m ibeis.algo.graph.core exec_vsone_subset

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
        from ibeis.algo.verif.vsone import OneVsOneProblem
        pblm = OneVsOneProblem(infr, verbose=True)
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
        infr.pblm = pblm
        # infr.classifiers = pblm

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
        """

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_mtest_infr()
            >>> cand_edges = infr.find_lnbnn_candidate_edges()
            >>> assert len(cand_edges) > 200
        """
        # Refresh the name labels

        # do LNBNN query for new edges
        # Use one-vs-many to establish candidate edges to classify
        infr.exec_matching(name_method='edge', cfgdict={
            'resize_dim': 'width',
            'dim_size': 700,
            'requery': True,
            'can_match_samename': False,
            'can_match_sameimg': False,
            # 'sv_on': False,
        })
        # infr.apply_match_edges(review_cfg={'ranks_top': 5})
        lnbnn_results = set(infr._cm_breaking(review_cfg={'ranks_top': 5}))
        already_reviewed = set(infr.get_edges_where_ne(
            'decision', 'unreviewed', edges=lnbnn_results,
            default='unreviewed', on_missing='filter'))
        candidate_edges = lnbnn_results - already_reviewed

        infr.print('ranking alg found {}/{} unreviewed edges'.format(
            len(candidate_edges), len(lnbnn_results)), 1)

        # if infr.enable_inference:
        #     orig_candidate_edges = candidate_edges
        #     candidate_edges = set(infr.filter_nonredun_edges(candidate_edges))
        #     infr.print('removed {} redundant candidates'.format(
        #         len(orig_candidate_edges) - len(candidate_edges)), 1)

        return candidate_edges

    @profile
    def find_neg_redun_candidate_edges(infr, k=None):
        """
        Get pairs of PCCs that are not complete.
        Finds edges that might complete them.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.algo.graph.mixin_dynamic import *  # NOQA
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_infr2()
            >>> categories = infr.categorize_edges(graph)
            >>> negative = categories[NEGTV]
            >>> ne, edges = #list(categories['reviewed_negatives'].items())[0]
            >>> infr.graph.remove_edges_from(edges)
            >>> cc1, cc2, _edges = list(infr.non_complete_pcc_pairs())[0]
            >>> result = non_complete_pcc_pairs(infr)
            >>> print(result)
        """
        if k is None:
            k = infr.queue_params['neg_redun']
        # Loop through all pairs
        for cc1, cc2 in infr.find_non_neg_redun_pccs(k=k):
            for u, v in infr.find_neg_augment_edges(cc1, cc2, k):
                yield e_(u, v)

    @profile
    def find_non_neg_redun_pccs(infr, k=None):
        """
        Get pairs of PCCs that are not complete.
        """
        if k is None:
            k = infr.queue_params['neg_redun']
        pccs = infr.positive_components()
        # Loop through all pairs
        for cc1, cc2 in it.combinations(pccs, 2):
            if not infr.is_neg_redundant(cc1, cc2):
                yield cc1, cc2

    def find_neg_augment_edges(infr, cc1, cc2, k=None):
        """
        Find enough edges to between two pccs to make them k-negative complete
        """
        if k is None:
            k = infr.queue_params['neg_redun']
        existing_edges = set(edges_cross(infr.graph, cc1, cc2))
        reviewed_edges = {
            edge: state
            for edge, state in infr.get_edge_attrs(
                'decision', existing_edges,
                default=UNREV).items()
            if state != UNREV
        }

        # Find how many negative edges we already have
        num = sum([state == NEGTV for state in reviewed_edges.values()])
        if num < k:
            # Find k random negative edges
            check_edges = existing_edges - set(reviewed_edges)
            # Check the existing but unreviewed edges first
            for edge in check_edges:
                num += 1
                yield edge
                if num >= k:
                    raise StopIteration()
            # Check non-existing edges next
            seed = 2827295125
            try:
                seed += sum(cc1) + sum(cc2)
            except Exception:
                pass
            rng = np.random.RandomState(seed)
            cc1 = ut.shuffle(list(cc1), rng=rng)
            cc2 = ut.shuffle(list(cc2), rng=rng)
            cc1 = ut.shuffle(list(cc1), rng=rng)
            for edge in it.starmap(nxu.e_, nxu.diag_product(cc1, cc2)):
                if edge not in existing_edges:
                    num += 1
                    yield edge
                    if num >= k:
                        raise StopIteration()

    def find_pos_augment_edges(infr, pcc, k=None):
        """
        # [[1, 0], [0, 2], [1, 2], [3, 1]]
        pos_sub = nx.Graph([[0, 1], [1, 2], [0, 2], [1, 3]])
        """
        if k is None:
            pos_k = infr.queue_params['pos_redun']
        else:
            pos_k = k
        pos_sub = infr.pos_graph.subgraph(pcc)

        # First try to augment only with unreviewed existing edges
        unrev_avail = list(nxu.edges_inside(infr.unreviewed_graph, pcc))
        try:
            check_edges = nxu.edge_connected_augmentation(
                pos_sub, pos_k, avail=unrev_avail, return_anyway=False)
        except ValueError:
            check_edges = None
        if not check_edges:
            # Allow new edges to be introduced
            full_sub = infr.graph.subgraph(pcc)
            new_avail = ut.estarmap(infr.e_, nx.complement(full_sub).edges())
            full_avail = unrev_avail + new_avail
            n_max = (len(pos_sub) * (len(pos_sub) - 1)) // 2
            n_complement = n_max - pos_sub.number_of_edges()
            if len(full_avail) == n_complement:
                # can use the faster algorithm
                check_edges = nxu.edge_connected_augmentation(
                    pos_sub, k=pos_k, return_anyway=True)
            else:
                # have to use the slow approximate algo
                check_edges = nxu.edge_connected_augmentation(
                    pos_sub, k=pos_k, avail=full_avail, return_anyway=True)
        check_edges = set(it.starmap(e_, check_edges))
        return check_edges

    @profile
    def find_pos_redun_candidate_edges(infr, k=None, verbose=False):
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
            >>> candidate_edges = list(infr.find_pos_redun_candidate_edges())
            >>> result = ('candidate_edges = %s' % (ut.repr2(candidate_edges),))
            >>> print(result)
            candidate_edges = {(1, 3), (7, 10)}
        """
        # Add random edges between exisiting non-redundant PCCs
        if k is None:
            k = infr.queue_params['pos_redun']
        # pcc_gen = list(infr.non_pos_redundant_pccs(relax=True))
        pcc_gen = list(infr.positive_components())
        prog = ut.ProgIter(pcc_gen, enabled=verbose, freq=1, adjust=False)
        for pcc in prog:
            if not infr.is_pos_redundant(pcc, k=k, relax=True,
                                         assume_connected=True):
                for edge in infr.find_pos_augment_edges(pcc, k=k):
                    yield nxu.e_(*edge)

    @profile
    def ensure_priority_scores(infr, priority_edges):
        if infr.classifiers:
            infr.print('Prioritizing {} edges with one-vs-one probs'.format(
                len(priority_edges)), 1)
            # Construct pairwise features on edges in infr
            primary_task = 'match_state'
            infr.task_probs[primary_task]

            match_task = infr.task_probs.get('match_state', {})
            need_flags = [e not in match_task for e in priority_edges]
            need_edges = ut.compress(priority_edges, need_flags)

            infr.print('There are {} edges without probabilities'.format(
                len(need_edges)), 1)

            # task_probs = infr._make_task_probs(priority_edges)
            if len(need_edges) > 0:
                # Only recompute for the needed edges
                task_probs = infr._make_task_probs(need_edges)
                # FIXME: this is slow
                for task, probs in task_probs.items():
                    if task not in infr.task_probs:
                        # infr.task_probs[task] = probs
                        infr.task_probs[task] = probs.to_dict(orient='index')
                    else:
                        # import pandas as pd
                        # old = infr.task_probs[task]
                        # new = pd.merge(old, probs, 'outer', probs.columns.tolist(),
                        #                left_index=True, right_index=True,
                        #                copy=False)
                        # infr.task_probs[task] = new
                        infr.task_probs[task].update(probs.to_dict(orient='index'))

            import pandas as pd
            primary_probs = pd.DataFrame(
                ut.take(infr.task_probs[primary_task], priority_edges),
                index=nxu.ensure_multi_index(priority_edges, ('aid1', 'aid2'))
            )

            primary_thresh = infr.task_thresh[primary_task]
            prob_match = primary_probs[POSTV]

            default_priority = prob_match.copy()
            # Give negatives that pass automatic thresholds high priority
            if infr.enable_auto_prioritize_nonpos:
                _probs = primary_probs[NEGTV]
                flags = _probs > primary_thresh[NEGTV]
                default_priority[flags] = np.maximum(default_priority[flags],
                                                     _probs[flags])

            # Give not-comps that pass automatic thresholds high priority
            if infr.enable_auto_prioritize_nonpos:
                _probs = primary_probs[INCMP]
                flags = _probs > primary_thresh[INCMP]
                default_priority[flags] = np.maximum(default_priority[flags],
                                                     _probs[flags])

            # Pack into edge attributes
            # edge_task_probs = {edge: {} for edge in priority_edges}
            # for task, probs in task_probs.items():
            #     for edge, val in probs.to_dict(orient='index').items():
            #         edge_task_probs[edge][task] = val

            infr.set_edge_attrs('prob_match', prob_match.to_dict())
            infr.set_edge_attrs('default_priority', default_priority.to_dict())

            metric = 'default_priority'
            priority = default_priority
        elif hasattr(infr, 'dummy_matcher'):
            prob_match = np.array(infr.dummy_matcher.predict_edges(priority_edges))
            infr.set_edge_attrs('prob_match', ut.dzip(priority_edges, prob_match))
            metric = 'prob_match'
            priority = prob_match
        elif infr.cm_list is not None:
            infr.print('Prioritizing edges with one-vs-vsmany scores', 1)
            # Not given any deploy classifier, this is the best we can do
            infr.task_probs = None
            scores = infr._make_lnbnn_scores(priority_edges)
            metric = 'normscore'
            priority = scores
        else:
            infr.print('No information to prioritize edges')
            metric = 'random'
            priority = np.zeros(len(priority_edges)) + 1e-6

        infr.set_edge_attrs(metric, ut.dzip(priority_edges, priority))
        return metric, priority

    @profile
    def add_candidate_edges(infr, candidate_edges):

        candidate_edges = list(candidate_edges)
        new_edges = infr.ensure_edges_from(candidate_edges)
        infr.print('There are {}/{} new candidate edges'.format(
            len(new_edges), len(candidate_edges)))

        if infr.test_mode:
            infr.apply_edge_truth(new_edges)

        if infr.enable_redundancy:
            priority_edges = list(infr.filter_nonredun_edges(candidate_edges))
            infr.print('using only {}/{} non-redun candidate edges'.format(
                len(priority_edges), len(candidate_edges)))
        else:
            priority_edges = candidate_edges

        if len(priority_edges) > 0:
            metric, priority = infr.ensure_priority_scores(priority_edges)

            infr.prioritize(metric, priority_edges, priority)

        if hasattr(infr, 'on_new_candidate_edges'):
            # hack callback for demo
            infr.on_new_candidate_edges(infr, new_edges)
        return len(priority_edges)

    @profile
    def refresh_candidate_edges(infr):
        """
        Search for candidate edges.
        Assign each edge a priority and add to queue.
        """
        infr.print('refresh_candidate_edges', 1)
        infr.assert_consistency_invariant()

        if infr.ibs is not None:
            candidate_edges = infr.find_lnbnn_candidate_edges()
        elif hasattr(infr, 'dummy_matcher'):
            infr.print('Searching for dummy candidates')
            infr.print('dummy vsone params =' + ut.repr4(
                infr.dummy_matcher.dummy_params, nl=1, si=True))
            candidate_edges = infr.dummy_matcher.find_candidate_edges()
        else:
            raise Exception(
                'No method available to search for candidate edges')
        infr.add_candidate_edges(candidate_edges)
        infr.assert_consistency_invariant()

    def _cached_pairwise_features(infr, edges, data_info):
        """
        Create pairwise features for annotations in a test inference object
        based on the features used to learn here

        TODO: need a more systematic way of specifying which feature dimensions
        need to be computed

        Notes:
            Given a edge (u, v), we need to:
            * Check which classifiers we have
            * Check which feat-cols the classifier needs,
               and construct a configuration that can acheive that.
                * Construct the chip/feat config
                * Construct the vsone config
                * Additional LNBNN enriching config
                * Pairwise feature construction config
            * Then we can apply the feature to the classifier

        edges = [(1, 2)]
        """
        import ubelt as ub
        import pandas as pd
        infr.print('Requesting %d cached pairwise features' % len(edges),
                   level=3)
        edges = list(edges)
        ibs = infr.ibs
        edge_uuids = ibs.unflat_map(ibs.get_annot_visual_uuids, edges)
        edge_hashid = ut.hashid_arr(edge_uuids, 'edges')

        feat_construct_config, feat_dims = data_info
        if len(edges) == 0:
            index = nxu.ensure_multi_index([], ('aid1', 'aid2'))
            feats = pd.DataFrame(columns=feat_dims, index=index)
            return feats

        _cfg_lbl = ut.partial(ut.repr2, stritems=True, itemsep='', kvsep=':')
        match_configclass = ibs.depc_annot.configclass_dict['pairwise_match']

        feat_cfgstr = '_'.join([
            edge_hashid,
            _cfg_lbl(feat_construct_config['match_config']),
            _cfg_lbl(feat_construct_config['pairfeat_cfg']),
            'global(' + _cfg_lbl(feat_construct_config['global_keys']) + ')',
            'pairwise_match_version=%r' % (match_configclass().version,)
        ])
        use_cache = not feat_construct_config['need_lnbnn']
        if len(edges) < 2:
            use_cache = False
        cache_dir = join(infr.ibs.get_cachedir(), 'infr_bulk_cache')
        feat_cacher = ub.Cacher('bulk_pairfeats_v3',
                                feat_cfgstr, enabled=use_cache,
                                dpath=cache_dir, verbose=infr.verbose > 3)
        if feat_cacher.enabled:
            ut.ensuredir(cache_dir)
        if feat_cacher.exists() and infr.verbose > 3:
            fpath = feat_cacher.get_fpath()
            print('Load match cache size: {}'.format(ut.get_file_nBytes_str(fpath)))
        data = feat_cacher.tryload()
        if data is None:
            config = feat_construct_config['match_config']
            pairfeat_cfg = feat_construct_config['pairfeat_cfg']
            global_keys = feat_construct_config['global_keys']
            need_lnbnn = feat_construct_config['need_lnbnn']

            multi_index = True
            data = infr._make_pairwise_features(edges, config, pairfeat_cfg,
                                                global_keys, need_lnbnn,
                                                multi_index)
            feat_cacher.save(data)
            if feat_cacher.enabled and infr.verbose > 3:
                fpath = feat_cacher.get_fpath()
                print('Save match cache size: {}'.format(ut.get_file_nBytes_str(fpath)))
        matches, feats = data

        # # Take the filtered subset of columns
        if feat_dims is not None:
            missing = set(feat_dims).difference(feats.columns)
            if any(missing):
                # print('We have: ' + ut.repr4(feats.columns))
                alt = feats.columns.difference(feat_dims)
                mis_msg = ('Missing feature dims: ' + ut.repr4(missing))
                alt_msg = ('Did you mean? ' + ut.repr4(alt))
                print(mis_msg)
                print(alt_msg)
                raise KeyError(mis_msg)
            feats = feats[feat_dims]
        return feats

    def find_pretrained_classifiers(infr, dpath):
        import glob
        import parse
        from os.path import join, basename
        fname_fmt = 'deploy_{task_key}_learn({data_key})_{n_dims}_{clf_key}_{hashid}.cPkl'
        task_clf_candidates = ut.ddict(list)
        for fpath in glob.iglob(join(dpath, 'deploy_*.cPkl')):
            fname = basename(fpath)
            result = parse.parse(fname_fmt, fname)
            if result:
                task_key = result.named['task_key']
                task_clf_candidates[task_key].append(fpath)
        return task_clf_candidates

    def load_latest_classifiers(infr, dpath):
        from os.path import getctime
        task_clf_candidates = infr.find_pretrained_classifiers(dpath)
        task_clf_fpaths = {}
        for task_key, fpaths in task_clf_candidates.items():
            # Find the classifier most recently created
            fpath = fpaths[ut.argmax(map(getctime, fpaths))]
            task_clf_fpaths[task_key] = fpath
        classifiers = {}
        for task_key, fpath in task_clf_fpaths.items():
            clf_info = ut.load_data(fpath)
            assert clf_info['metadata']['task_key'] == task_key, (
                'bad saved clf at fpath={}'.format(fpath))
            classifiers[task_key] = clf_info
        return classifiers

    @profile
    def _make_task_probs(infr, edges, data_key=None):
        """
        Predict edge probs for each pairwise classifier task
        GZ_Master1xwusomqdrrgoszenlearn(sum,glob)
        GZ_Master1xwusomqdrrgoszenlearn
        """
        # import ubelt as ub
        from ibeis.scripts import sklearn_utils
        if infr.classifiers is None:
            raise ValueError('no classifiers exist')

        if not isinstance(infr.classifiers, dict):
            raise NotImplementedError(
                'need to deploy or implement eval prediction')

        prev_data_info = None
        task_keys = list(infr.classifiers.keys())
        task_probs = {}
        infr.print('predict {} for {} edges'.format(
            ut.conj_phrase(task_keys, 'and'), len(edges)))
        for task_key in task_keys:
            deploy_info = infr.classifiers[task_key]
            data_info = deploy_info['metadata']['data_info']
            class_names = deploy_info['metadata']['class_names']
            clf = deploy_info['clf']
            if prev_data_info != data_info:
                X_df = infr._cached_pairwise_features(edges, data_info)
                prev_data_info = data_info
            probs_df = sklearn_utils.predict_proba_df(clf, X_df, class_names)
            task_probs[task_key] = probs_df
        # pblm = infr.classifiers
        # data_key = pblm.default_data_key
        # # TODO: find a good way to cache this
        # data_info = pblm.feat_construct_info[data_key]
        # X_df = infr._cached_pairwise_features(edges, data_info)

        # feat_construct_config, feat_dims = data_info
        # cfgstr = '_'.join([
        #     infr.ibs.dbname, data_key,
        #     ut.hashid_arr(edges, 'edges'),
        #     ut.hashid_arr(feat_dims, 'feat_dims'),
        #     ut.repr2(feat_construct_config, stritems=True, itemsep='',
        #              kvsep=':')
        # ])
        # cache_dir = ut.ensuredir(infr.ibs.get_cachedir(), 'infr_bulk_cache')
        # cacher = ub.Cacher('foobarclf_taskprobs_' + infr.ibs.dbname,
        #                    cfgstr=cfgstr, dpath=cache_dir, enabled=1,
        #                    meta=[edges, feat_dims,
        #                          ut.repr2(feat_construct_config)],
        #                    verbose=pblm.verbose)
        # X = cacher.tryload()
        # if cacher.exists():
        #     fpath = cacher.get_fpath()
        #     print('Bulk cache size: {}'.format(ut.get_file_nBytes_str(fpath)))
        # if X is None:
        #     X = infr._pblm_pairwise_features(edges, data_key)
        #     cacher.save(X)
        #     fpath = cacher.get_fpath()
        #     print('Saved bulk cache of size: {}'.format(
        #         ut.get_file_nBytes_str(fpath)))

        # task_keys = list(pblm.samples.subtasks.keys())
        # task_probs = pblm.predict_proba_deploy(X_df, task_keys)
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
