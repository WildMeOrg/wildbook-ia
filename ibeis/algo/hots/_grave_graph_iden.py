

    def hack_write_ibeis_staging_onetime(infr):
        """
        TODO: depricate

        CommandLine:
            python -m ibeis.algo.hots.graph_iden hack_write_ibeis_staging_onetime

        Example:
            >>> from ibeis.algo.hots.graph_iden import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(db='PZ_PB_RF_TRAIN')
            >>> infr = AnnotInference(ibs, ibs.get_valid_aids(), autoinit=True)
            >>> infr.verbose = 3
            >>> infr.reset_feedback('staging')
            >>> infr.apply_feedback_edges()
            >>> infr.review_dummy_edges()
            >>> #consistent_subgraphs = infr.consistent_compoments()
            >>> #consistent_aids = ut.flatten([g.nodes() for g in consistent_subgraphs])
            >>> #infr.remove_aids(consistent_aids)
            >>> infr.relabel_using_reviews()
            >>> infr.apply_review_inference()
            >>> infr.start_qt_interface()
        """
        # puts data from annotmatch into staging

        ibs = infr.ibs

        external_feedback = infr.external_feedback

        staged = infr.read_ibeis_staging_feedback()
        aid_1_list = []
        aid_2_list = []
        decision_list = []
        tags_list = []
        for (aid1, aid2), feedbacks in external_feedback.items():
            if (aid1, aid2) in staged:
                continue
            for feedback_item in feedbacks:
                decision_key = feedback_item['decision']
                decision_int = ibs.const.REVIEW_MATCH_CODE[decision_key]
                tags = feedback_item['tags']
                aid_1_list.append(aid1)
                aid_2_list.append(aid2)
                decision_list.append(decision_int)
                tags_list.append(tags)

        identity_list = [None] * len(aid_1_list)
        user_confidence_list = None
        r = ibs.add_review(aid_1_list, aid_2_list, decision_list,
                           identity_list=identity_list,
                           user_confidence_list=user_confidence_list,
                           tags_list=tags_list)
        assert len(ut.find_duplicate_items(r)) == 0

        #
        ibs.staging.delete_rowids('reviews', ibs.staging.get_all_rowids('reviews'))

        # ---
        # ipy hack
        infr.external_feedback = {k: [_ for _ in v if 'timestamp' not in _]
                                  for k, v in infr.user_feedback.items()}
        infr.internal_feedback = {k: [_ for _ in v if 'timestamp' in _]
                                  for k, v in infr.user_feedback.items()}
        infr.external_feedback = {k: v for k, v in infr.external_feedback.items() if v}
        infr.internal_feedback = {k: v for k, v in infr.internal_feedback.items() if v}



    @profile
    def rank_priority_edges(sim):
        """
        Find the order of reviews that would be selected by the priority queue
        for an oracle reviewer

        Differences between this and iterative review
            * Here we know what the minimum set of negative edges is in advance
              the iterative review may have to go through several more
              negatives
            * This positive reviews done here are minimal if the current
              decisions contain no split errors.
        """
        import networkx as nx
        import pandas as pd
        # auto_decisions = sim.auto_decisions
        primary_probs = sim.primary_probs
        primary_truth = sim.primary_truth
        infr = sim.infr

        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        assert n_inconsistent == 0, 'must be zero here'

        curr_decisions = infr.edge_attr_df('reviewed_state')

        # Choose weights proportional to the liklihood an edge will be reviewed
        # Give a negative weight to edges that are already reviewed.
        mwc_weights = primary_probs['match'].copy()
        # mwc_weights.loc[curr_decisions.index] = 2
        for k, sub in curr_decisions.groupby(curr_decisions):
            if k == 'match':
                mwc_weights[sub.index] += 1
            if k == 'notcomp':
                mwc_weights[sub.index] += 1
            if k == 'nomatch':
                mwc_weights[sub.index] += 2
        mwc_weights = 2 - mwc_weights

        undiscovered_errors = []
        gt_forests = []
        gt_clusters = ut.group_pairs(infr.gen_node_attrs('orig_name_label'))
        for nid, nodes in gt_clusters.items():
            if len(nodes) == 1:
                continue
            cc_cand_edges = list(ut.nx_edges_between(infr.graph, nodes))
            cc = ut.nx_from_node_edge(nodes, cc_cand_edges)
            cc_weights = mwc_weights.loc[cc_cand_edges]
            nx.set_edge_attributes(cc, 'weight', cc_weights.to_dict())
            # Minimum Spanning Compoment will contain the minimum possible
            # positive reviews and previously reviewed edges
            mwc = ut.nx_minimum_weight_component(cc)
            # Remove all no-matching edges to disconnect impossible matches
            mwc_decision = curr_decisions.loc[ut.lstarmap(infr.e_, mwc.edges())]
            remove_edges = list(mwc_decision[mwc_decision == 'nomatch'].index)
            if len(remove_edges) > 0:
                undiscovered_errors.append(remove_edges)
                mwc.remove_edges_from(remove_edges)
            ccs = list(nx.connected_component_subgraphs(mwc))
            if len(remove_edges) > 0:
                # print(len(remove_edges))
                if len(ccs) <= len(remove_edges):
                    print('negatives not breaking things in two')
                # print(len(ccs))
            gt_forests.append(ccs)
        # ut.dict_hist(ut.lmap(len, gt_forests))

        pos_edges_ = [[ut.lstarmap(infr.e_, t.edges()) for t in f]
                      for f in gt_forests]
        minimal_positive_edges = ut.flatten(ut.flatten(pos_edges_))
        minimal_positive_edges = list(set(minimal_positive_edges).difference(
            curr_decisions.index))

        priority_edges = minimal_positive_edges

        # need to also get all negative edges I think to know what the set of
        # negative edges is you need sequential information because it depends
        # on the current set of known positive edges. For this reason we
        # primarilly measure how many positive reviews we do.
        # WE hack negative edges and assume they are mostly skipped
        HACK_MIN_NEG_EDGES = False
        if HACK_MIN_NEG_EDGES:
            import vtool as vt
            node_to_nid = dict(infr.graph.nodes(data='orig_name_label'))
            is_nomatch = sim.primary_truth['nomatch']
            negative_edges = is_nomatch[is_nomatch].index.tolist()

            neg_nid_edges = ut.lstarmap(infr.e_, ut.unflat_take(node_to_nid,
                                                                negative_edges))
            # Group all negative edges between components
            groupxs = ut.group_indices(neg_nid_edges)[1]
            grouped_scores = vt.apply_grouping(
                primary_probs.loc[negative_edges]['match'].values,
                groupxs)
            max_xs = [xs[s.argmax()] for xs, s in zip(groupxs, grouped_scores)]

            # Take only negative edges that don't produce inconsistencies
            # given our current decisions
            split_flags = curr_decisions.loc[negative_edges] == 'match'

            split_flags = np.array([
                np.any(flags) for flags in
                vt.apply_grouping(split_flags.values, groupxs)
            ])
            max_xs2 = ut.compress(max_xs, ~split_flags)
            minimal_negative_edges = ut.take(negative_edges, max_xs2)
            priority_edges = priority_edges + minimal_negative_edges

        # priority_edges = minimal_positive_edges + minimal_negative_edges
        priorites = infr._get_priorites(priority_edges)
        priority_edges = ut.sortedby(priority_edges, priorites)[::-1]
        priorites = priorites[priorites.argsort()[::-1]]

        lowest_pos = infr._get_priorites(minimal_positive_edges).min()
        (priorites >= lowest_pos).sum()

        sim.results['n_pos_want'] = len(minimal_positive_edges)

        # primary_truth.loc[minimal_positive_edges]['match']
        # import numpy as np
        # true_ranks = np.where(primary_truth.loc[priority_edges]['match'])[0]

        # User will only review so many pairs for us
        max_user_reviews = 100
        user_edges = priority_edges[:max_user_reviews]

        infr.add_feedback_df(
            primary_truth.loc[user_edges].idxmax(axis=1),
            user_id='oracle', apply=True)

        # infr.relab()
        n_clusters, n_inconsistent = infr.relabel_using_reviews(rectify=False)
        assert n_inconsistent == 0, 'should not create any inconsistencies'

        if False:
            undiscovered_errors
            len(primary_truth.loc[user_edges].idxmax(axis=1))

        sim.results['n_user_clusters'] = n_clusters
        infr.apply_review_inference()

        curr_decisions = infr.edge_attr_df('reviewed_state')
        curr_truth = primary_truth.loc[curr_decisions.index].idxmax(axis=1)
        n_user_mistakes = curr_decisions != curr_truth
        sim.results['n_user_mistakes'] = sum(n_user_mistakes)

        if True:
            print("Post-User classification")
            from ibeis.scripts import clf_helpers
            user_pred = infr.infr_pred_df(primary_truth.index)
            # If we don't predict on a pair it is implicitly different
            user_pred[pd.isnull(user_pred)] = 'diff'
            # = user_pred.replace(None, 'diff')
            real_truth = pd.Series(infr.is_same(list(primary_truth.index)),
                                   index=primary_truth.index)
            real_truth = real_truth.replace(True, 'same').replace(False, 'diff')

            clf_helpers.classification_report2(real_truth, user_pred)
        return user_edges
        # pass
