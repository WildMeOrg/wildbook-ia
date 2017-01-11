

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
