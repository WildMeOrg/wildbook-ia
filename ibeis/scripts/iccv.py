import numpy as np
import utool as ut


def debug_expanded_aids(ibs, expanded_aids_list, verbose=1):
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    # print('len(expanded_aids_list) = %r' % (len(expanded_aids_list),))
    cfgargs = dict(per_vp=False, per_multiple=False, combo_dists=False,
                   per_name=True, per_enc=True, use_hist=False,
                   combo_enc_info=False)

    for qaids, daids in expanded_aids_list:
        stats = ibs.get_annotconfig_stats(qaids, daids, **cfgargs)
        hashids = (stats['qaid_stats']['qhashid'],
                   stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        if verbose > 1:
            print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def encounter_stuff(ibs, aids):
    from ibeis.init import main_helpers
    main_helpers.monkeypatch_encounters(ibs, aids, days=50)
    annots = ibs.annots(aids)

    from ibeis.init.filter_annots import encounter_crossval
    expanded_aids = encounter_crossval(ibs, annots.aids, qenc_per_name=1,
                                       denc_per_name=2)
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    # with warnings.catch_warnings():
    for qaids, daids in expanded_aids:
        stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False, combo_enc_info=False)
        hashids = (stats['qaid_stats']['qhashid'],
                   stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        # print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def end_to_end():
    import ibeis
    from ibeis.init import main_helpers
    ibs = ibeis.opendb('PZ_MTEST')
    # ibs = ibeis.opendb('GZ_Master1')
    # Specialized database params
    enc_kw = dict(minutes=30)
    filt_kw = dict(require_timestamp=True, require_gps=True, is_known=True)
    if ibs.dbname == 'PZ_MTEST':
        enc_kw = dict(days=50)
        filt_kw = dict(require_timestamp=True, is_known=True)
    expt_aids = ibs.filter_annots_general(**filt_kw)
    main_helpers.monkeypatch_encounters(ibs, aids=expt_aids, **enc_kw)

    annots = ibs.annots(expt_aids)
    names = list(annots.group_items(annots.nids).values())
    ut.shuffle(names, rng=321)
    train_aids = ut.flatten(names[0::2])
    test_aids = ut.flatten(names[1::2])

    # -----------
    # TRAINING

    # aids = train_aids
    # phis = learn_termination(ibs, aids=train_aids)

    from ibeis.scripts.script_vsone import OneVsOneProblem
    clf_key = 'RF'
    data_key = 'learn(sum,glob)'
    task_keys = ['match_state', 'photobomb_state']
    pblm = OneVsOneProblem.from_aids(ibs, aids=train_aids)
    pblm.load_features()
    pblm.load_samples()
    pblm.samples.print_info()
    pblm.build_feature_subsets()
    # task_keys = list(pblm.samples.subtasks.keys())

    pblm.learn_deploy_classifiers(task_keys, data_key=data_key,
                                  clf_key=clf_key)
    # match_clf = pblm.deploy_task_clfs['match_state']
    # pb_clf = pblm.deploy_task_clfs['photobomb_state']

    # Inspect feature importances if you want
    # pblm.report_classifier_importance2(match_clf, data_key=data_key)
    # pblm.report_classifier_importance2(pb_clf, data_key=data_key)

    # ------------
    # TESTING
    """
    Algorithm Alternatives:
        1. Ranking only
        2. ReRanking using 1v1 with automatic thresholds
        3. K=1
        4. K=2
    """
    if False:
        test_aids = ut.flatten(names[1::2][::2])

    import pandas as pd
    import plottool as pt  # NOQA
    # Create a new AnnotInference instance to go end-to-end
    # Dials
    dials1 = {
        'name': 'Ranking Only',
        'k_redun': np.inf,
        'cand_kw': dict(pblm=None),
        'priority_metric': 'normscore',
        'oracle_accuracy': 1.0,
    }
    dials2 = {
        'name': 'Graph K=1',
        'k_redun': 1,
        'cand_kw': dict(pblm=pblm),
        'priority_metric': 'priority',
        'oracle_accuracy': 1.0,
    }

    verbose = 0
    # infr.set_node_attrs('pin', True)
    # infr.show(show_candidate_edges=True)
    infr = ibeis.AnnotInference(ibs=ibs, aids=test_aids, autoinit=True,
                                verbose=verbose)
    metrics_df1 = run_expt(infr, dials=dials1)

    infr = ibeis.AnnotInference(ibs=ibs, aids=test_aids, autoinit=True,
                                verbose=verbose)
    metrics_df2 = run_expt(infr, dials=dials2)

    import plottool as pt
    pt.qtensure()
    pt.plot(metrics_df1['n_manual'].values, metrics_df1['merge_remain'].values, label=dials1['name'])
    pt.plot(metrics_df2['n_manual'].values, metrics_df2['merge_remain'].values, label=dials2['name'])
    pt.set_xlabel('# manual reviews')
    pt.set_ylabel('% merges remaining')
    pt.legend()

    # infr.show(show_candidate_edges=True)
    # TODO: use phi to check termination
    # TODO: recompute candidate edges


def run_expt(infr, dials):
    k_redun = dials['k_redun']
    oracle_accuracy = dials['oracle_accuracy']
    cand_kw = dials['cand_kw']
    priority_metric = dials['priority_metric']
    infr.PRIORITY_METRIC = priority_metric
    autoreview_enabled = infr.PRIORITY_METRIC == 'priority'
    infr.queue_params['pos_redundancy'] = k_redun
    infr.queue_params['neg_redundancy'] = k_redun
    infr.task_thresh = {
        'photobomb_state': pd.Series({
            'pb': .5,
            'notpb': .9,
        }),
        'match_state': pd.Series(ut.odict([
            ('nomatch', .99),
            ('match', .99),
            ('notcomp', .99),
        ]))
    }
    nid_to_gt_cc = ut.group_items(infr.aids, infr.orig_name_labels)
    real_n_pcc_mst_edges = sum([len(cc) - 1 for cc in nid_to_gt_cc.values()])
    def measure_metrics(infr, n_manual, n_auto):
        pred_n_pcc_mst_edges = sum([len(cc.node) - 1 for cc in
                                    infr.positive_connected_compoments()])
        pos_acc = pred_n_pcc_mst_edges / real_n_pcc_mst_edges
        metrics = {
            'n_manual': n_manual,
            'n_auto': n_auto,
            'pos_acc': pos_acc,
            'merge_remain': 1 - pos_acc,
        }
        return metrics

    n_manual = 0
    n_auto = 0
    metrics_list = []

    rng = ut.ensure_rng(10, impl='python')
    infr.reset(state='empty')
    infr.remove_feedback(apply=True)

    while True:
        infr.refresh_candidate_edges(**cand_kw)
        infr.init_refresh_criteria()
        infr._init_priority_queue()

        if not len(infr.queue):
            print('Queue is empty')
            break

        while not infr.refresh.check() and len(infr.queue):
            edge, priority = infr.pop()
            print('edge=%r, priority=%r' % (edge, priority))
            auto_flag = False
            if autoreview_enabled:
                auto_flag, review = infr.try_auto_review(edge, priority)
                n_auto += auto_flag
            if not auto_flag:
                error, review = infr.oracle_review(edge, oracle_accuracy, rng)
                n_manual += 1
            infr.add_feedback(edge=edge, apply=True, rectify=False, **review)
            metrics = measure_metrics(infr, n_manual, n_auto)
            metrics_list.append(metrics)

        print('need to refresh')
        # For now don't refresh
        break
    metrics_df = pd.DataFrame.from_dict(metrics_list)
    return metrics_df


def learn_termination(ibs, aids):
    """
    Example:
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> ranks = 20
        >>> ydatas = [phi.cumsum()[0:ranks] for phi in phis.values()]
        >>> pt.multi_plot(
        >>>     xdata=np.arange(1, ranks + 1),
        >>>     ydata_list=ydatas,
        >>>     num_xticks=ranks,
        >>>     label_list=['annots per query: %d' % d for d in phis.keys()],
        >>>     title='Learned Termination CDF',
        >>> )
    """
    from ibeis.init.filter_annots import encounter_crossval
    from ibeis.expt import test_result
    pipe_cfg = {
        'resize_dim': 'area',
        'dim_size': 450,
    }

    n_splits = 3
    crossval_splits = []
    avail_confusors = []
    import random
    rng = random.Random(0)
    for n_query_per_name in range(1, 5):
        reshaped_splits, nid_to_confusors = encounter_crossval(
            ibs, aids, qenc_per_name=n_query_per_name, denc_per_name=1,
            rng=rng, n_splits=n_splits, early=True)

        avail_confusors.append(nid_to_confusors)

        expanded_aids = []
        for qpart, dpart in reshaped_splits:
            # For each encounter choose just 1 annotation
            qaids = [rng.choice(enc) for enc in ut.flatten(qpart)]
            daids = [rng.choice(enc) for enc in ut.flatten(dpart)]
            assert len(set(qaids).intersection(daids)) == 0
            expanded_aids.append((sorted(qaids), sorted(daids)))

        crossval_splits.append((n_query_per_name, expanded_aids))

    n_daid_spread = [len(expanded_aids[0][1])
                     for _, expanded_aids in crossval_splits]

    # Check to see if we can pad confusors to make the database size equal
    max_size = max(n_daid_spread)
    afford = (min(map(len, avail_confusors)) - (max(n_daid_spread) -
                                                min(n_daid_spread)))
    max_size += afford

    crossval_splits2 = []
    _iter =  zip(crossval_splits, avail_confusors)
    for (n_query_per_name, expanded_aids), nid_to_confusors in _iter:
        crossval_splits2.append((n_query_per_name, []))
        for qaids, daids in expanded_aids:
            n_extra = max(max_size - len(daids), 0)
            if n_extra <= len(nid_to_confusors):
                extra2 = ut.take_column(nid_to_confusors.values(), 0)[:n_extra]
                extra = ut.flatten(extra2)
            else:
                extra2 = ut.flatten(nid_to_confusors.values())
                rng.shuffle(extra2)
                extra = extra2[:n_extra]
            crossval_splits2[-1][1].append((qaids, sorted(daids + extra)))

    for n_query_per_name, expanded_aids in crossval_splits2:
        debug_expanded_aids(ibs, expanded_aids, verbose=1)

    phis = {}
    for n_query_per_name, expanded_aids in crossval_splits2:
        accumulators = []
        for qaids, daids in expanded_aids:
            num_datab_pccs = len(np.unique(ibs.annots(daids).nids))
            # num_query_pccs = len(np.unique(ibs.annots(qaids).nids))
            qreq_ = ibs.new_query_request(qaids, daids, verbose=False,
                                          cfgdict=pipe_cfg)
            cm_list = qreq_.execute()
            testres = test_result.TestResult.from_cms(cm_list, qreq_)
            freqs, bins = testres.get_rank_histograms(
                key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs))
            freq = freqs[0]
            accumulators.append(freq)
        size = max(map(len, accumulators))
        accum = np.zeros(size)
        for freq in accumulators:
            accum[0:len(freq)] += freq

        # unsmoothed
        phi1 = accum / accum.sum()
        # kernel = cv2.getGaussianKernel(ksize=3, sigma=.9).T[0]
        # phi2 = np.convolve(phi1, kernel)
        # Smooth out everything after the sv rank to be uniform
        svrank = qreq_.qparams.nNameShortlistSVER
        phi = phi1.copy()
        phi[svrank:] = (phi[svrank:].sum()) / (len(phi) - svrank)
        # phi = accum
        phis[n_query_per_name] = phi
    return phis


def learn_phi():
    # from ibeis.init import main_helpers
    # dbname = 'GZ_Master1'
    # a = 'timectrl'
    # t = 'baseline'
    # ibs, testres = main_helpers.testdata_expts(dbname, a=a, t=t)
    import ibeis
    # from ibeis.init.filter_annots import annot_crossval
    from ibeis.init.filter_annots import encounter_crossval
    from ibeis.init import main_helpers
    from ibeis.expt import test_result
    import plottool as pt
    pt.qtensure()

    ibs = ibeis.opendb('GZ_Master1')
    # ibs = ibeis.opendb('PZ_MTEST')
    # ibs = ibeis.opendb('PZ_PB_RF_TRAIN')

    aids = ibs.filter_annots_general(require_timestamp=True, require_gps=True,
                                     is_known=True)
    # aids = ibs.filter_annots_general(is_known=True, require_timestamp=True)

    # annots = ibs.annots(aids=aids, asarray=True)
    # Take only annots with time and gps data
    # annots = annots.compress(~np.isnan(annots.image_unixtimes_asfloat))
    # annots = annots.compress(~np.isnan(np.array(annots.gps)).any(axis=1))

    main_helpers.monkeypatch_encounters(ibs, aids, minutes=30)

    # pt.draw_time_distribution(annots.image_unixtimes_asfloat, bw=1209600.0)

    pipe_cfg = {
        'resize_dim': 'area',
        'dim_size': 450,
    }
    # qreq_ = ibs.new_query_request(qaids, daids, verbose=False, query_cfg=pipe_cfg)
    # cm_list = qreq_.execute()
    # annots = ibs.annots(aids=aids)
    # nid_to_aids = ut.group_items(annots.aids, annots.nids)

    # TO FIX WE SHOULD GROUP ENCOUNTERS

    n_splits = 3
    # n_splits = 5
    crossval_splits = []
    avail_confusors = []
    import random
    rng = random.Random(0)
    for n_query_per_name in range(1, 5):
        reshaped_splits, nid_to_confusors = encounter_crossval(
            ibs, aids, qenc_per_name=n_query_per_name, denc_per_name=1,
            rng=rng, n_splits=n_splits, early=True)

        avail_confusors.append(nid_to_confusors)

        expanded_aids = []
        for qpart, dpart in reshaped_splits:
            # For each encounter choose just 1 annotation
            qaids = [rng.choice(enc) for enc in ut.flatten(qpart)]
            daids = [rng.choice(enc) for enc in ut.flatten(dpart)]
            assert len(set(qaids).intersection(daids)) == 0
            expanded_aids.append((sorted(qaids), sorted(daids)))

        # expanded_aids = annot_crossval(ibs, annots.aids,
        #                                n_qaids_per_name=n_query_per_name,
        #                                n_daids_per_name=1, n_splits=n_splits,
        #                                rng=rng, debug=False)
        crossval_splits.append((n_query_per_name, expanded_aids))

    n_daid_spread = [len(expanded_aids[0][1]) for _, expanded_aids in crossval_splits]
    # Check to see if we can pad confusors to make the database size equal

    max_size = max(n_daid_spread)

    afford = (min(map(len, avail_confusors)) - (max(n_daid_spread) -
                                                min(n_daid_spread)))
    max_size += afford

    crossval_splits2 = []

    for (n_query_per_name, expanded_aids), nid_to_confusors in zip(crossval_splits, avail_confusors):
        crossval_splits2.append((n_query_per_name, []))
        for qaids, daids in expanded_aids:
            n_extra = max(max_size - len(daids), 0)
            if n_extra <= len(nid_to_confusors):
                extra2 = ut.take_column(nid_to_confusors.values(), 0)[:n_extra]
                extra = ut.flatten(extra2)
            else:
                extra2 = ut.flatten(nid_to_confusors.values())
                rng.shuffle(extra2)
                extra = extra2[:n_extra]
            crossval_splits2[-1][1].append((qaids, sorted(daids + extra)))

    for n_query_per_name, expanded_aids in crossval_splits2:
        debug_expanded_aids(ibs, expanded_aids, verbose=1)

    phis = {}
    for n_query_per_name, expanded_aids in crossval_splits2:
        accumulators = []
        # with warnings.catch_warnings():
        for qaids, daids in expanded_aids:
            num_datab_pccs = len(np.unique(ibs.annots(daids).nids))
            # num_query_pccs = len(np.unique(ibs.annots(qaids).nids))
            qreq_ = ibs.new_query_request(qaids, daids, verbose=False,
                                          cfgdict=pipe_cfg)

            cm_list = qreq_.execute()
            testres = test_result.TestResult.from_cms(cm_list, qreq_)
            # nranks = testres.get_infoprop_list(key='qnx2_gt_name_rank')[0]
            # aranks = testres.get_infoprop_list(key='qx2_gt_annot_rank')[0]
            # freqs, bins = testres.get_rank_histograms(
            #     key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs))
            freqs, bins = testres.get_rank_histograms(
                key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs))
            freq = freqs[0]
            accumulators.append(freq)
        size = max(map(len, accumulators))
        accum = np.zeros(size)
        for freq in accumulators:
            accum[0:len(freq)] += freq

        # unsmoothed
        phi1 = accum / accum.sum()
        # kernel = cv2.getGaussianKernel(ksize=3, sigma=.9).T[0]
        # phi2 = np.convolve(phi1, kernel)
        # Smooth out everything after the sv rank to be uniform
        svrank = qreq_.qparams.nNameShortlistSVER
        phi = phi1.copy()
        phi[svrank:] = (phi[svrank:].sum()) / (len(phi) - svrank)
        # phi = accum
        phis[n_query_per_name] = phi

    # ydatas = [phi.cumsum() for phi in phis.values()]
    # label_list = list(map(str, phis.keys()))
    # pt.multi_plot(xdata=np.arange(len(phi)), ydata_list=ydatas, label_list=label_list)
    # pt.figure()
    ranks = 20
    # ydatas = [phi[0:ranks] for phi in phis.values()]
    ydatas = [phi.cumsum()[0:ranks] for phi in phis.values()]
    pt.multi_plot(
        xdata=np.arange(1, ranks + 1),
        ydata_list=ydatas,
        num_xticks=ranks,
        # kind='bar',
        label_list=['annots per query: %d' % d for d in phis.keys()],
        title='Learned Termination CDF',
    )

    #cmc = phi3.cumsum()
    # accum[20:].sum()
    # import cv2
    # accum

    # from sklearn.neighbors.kde import KernelDensity
    # kde = KernelDensity(kernel='gaussian', bandwidth=3)
    # X = ut.flatten([[rank] * (1 + int(freq)) for rank, freq in enumerate(accum)])
    # kde.fit(np.array(X)[:, None])
    # basis = np.linspace(0, len(accum), 1000)
    # density = np.exp(kde.score_samples(basis[:, None]))
    # pt.plot(basis, density)

    # bins, edges = testres.get_rank_percentage_cumhist()
    # import plottool as pt
    # pt.qtensure()
    # pt.multi_plot(edges, [bins[0]])

    # testres.get_infoprop_list('qx2_gt_name_rank')
    # [cm.extend_results(qreq_).get_name_ranks([cm.qnid])[0] for cm in cm_list]

    # if False:
    #     accumulator = np.zeros(num_datab_pccs)
    #     for cm in cm_list:
    #         cm = cm.extend_results(qreq_)
    #         rank = cm.get_name_ranks([cm.qnid])[0]
    #         # rank = min(cm.get_annot_ranks(cm.get_groundtruth_daids()))
    #         accumulator[rank] += 1
