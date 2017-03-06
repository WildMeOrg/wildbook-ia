import numpy as np
import utool as ut

def debug_expanded_aids(expanded_aids_list, verbose=1):
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    # print('len(expanded_aids_list) = %r' % (len(expanded_aids_list),))
    for qaids, daids in expanded_aids_list:
        stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
                                          combo_enc_info=False)
        hashids = (stats['qaid_stats']['qhashid'],
                   stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        if verbose > 1:
            print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def monkeypatch_encounters(ibs, aids, **kwargs):
    """
    50 days for PZ_MTEST
    kwargs = dict(days=50)

    if False:
        name_mindeltas = []
        for name in annots.group_items(annots.nids).values():
            times = name.image_unixtimes_asfloat
            deltas = [ut.unixtime_to_timedelta(np.abs(t1 - t2))
                      for t1, t2 in ut.combinations(times, 2)]
            if deltas:
                name_mindeltas.append(min(deltas))
        print(ut.repr3(ut.lmap(ut.get_timedelta_str,
                               sorted(name_mindeltas))))
    """
    from ibeis.algo.preproc.occurrence_blackbox import cluster_timespace_sec
    import datetime
    annots = ibs.annots(aids)
    thresh_sec = datetime.timedelta(**kwargs).total_seconds()
    # thresh_sec = datetime.timedelta(minutes=30).seconds

    # cfgstr = str(ut.combine_uuids(annots.visual_uuids))
    # cacher = ut.Cacher('occurrence_labels', cfgstr=cfgstr, enabled=0)
    # data = cacher.tryload()

    data = cluster_timespace_sec(
        annots.image_unixtimes_asfloat, annots.gps, thresh_sec=thresh_sec,
        km_per_sec=.002)

    # cacher.save(data)
    occurrence_labels = data

    ndec = int(np.ceil(np.log10(max(occurrence_labels))))
    suffmt = '-monkey-occur%0' + str(ndec) + 'd'
    encounter_labels = [n + suffmt % (o,)
                        for o, n in zip(occurrence_labels, annots.names)]
    enc_lookup = ut.dzip(annots.aids, encounter_labels)

    annots_per_enc = ut.dict_hist(encounter_labels, ordered=True)
    ut.get_stats(list(annots_per_enc.values()))

    encounters = ibs._annot_groups(annots.group(encounter_labels)[1])
    enc_names = ut.take_column(encounters.nids, 0)
    name_to_encounters = ut.group_items(encounters, enc_names)

    # print('name_to_encounters = %s' % (ut.repr3(name_to_encounters)),)
    # print('Names to num encounters')
    # name_to_num_enc = ut.dict_hist(
    #     ut.map_dict_vals(len, name_to_encounters).values())

    # monkey patch to override encounter info
    def _monkey_get_annot_encounter_text(ibs, aids):
        return ut.dict_take(enc_lookup, aids)
    ut.inject_func_as_method(ibs, _monkey_get_annot_encounter_text,
                             'get_annot_encounter_text', force=True)

def unmonkeypatch_encounters(ibs):
    from ibeis.other import ibsfuncs
    ut.inject_func_as_method(ibs, ibsfuncs.get_annot_encounter_text,
                             'get_annot_encounter_text', force=True)


def encounter_stuff():

    monkeypatch_encounters(ibs, aids, days=50)

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


def learn_phi():
    # from ibeis.init import main_helpers
    # dbname = 'GZ_Master1'
    # a = 'timectrl'
    # t = 'baseline'
    # ibs, testres = main_helpers.testdata_expts(dbname, a=a, t=t)

    import datetime
    import ibeis
    from ibeis.init.filter_annots import annot_crossval
    from ibeis.expt import test_result
    import plottool as pt
    pt.qtensure()

    ibs = ibeis.opendb('GZ_Master1')
    # ibs = ibeis.opendb('PZ_MTEST')
    # ibs = ibeis.opendb('PZ_PB_RF_TRAIN')

    aids = ibs.filter_annots_general(require_timestamp=True, require_gps=True,
                                     is_known=True)
    # aids = ibs.filter_annots_general(is_known=True, require_timestamp=True)

    annots = ibs.annots(aids=aids, asarray=True)
    # Take only annots with time and gps data
    # annots = annots.compress(~np.isnan(annots.image_unixtimes_asfloat))
    # annots = annots.compress(~np.isnan(np.array(annots.gps)).any(axis=1))

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
    n_splits = 5
    crossval_splits = []
    for n_query_per_name in range(1, 5):
        rng = np.random.RandomState(0)
        expanded_aids = annot_crossval(ibs, annots.aids,
                                       n_qaids_per_name=n_query_per_name,
                                       n_daids_per_name=1, n_splits=n_splits,
                                       rng=rng, debug=False)
        crossval_splits.append((n_query_per_name, expanded_aids))

    # for n_query_per_name, expanded_aids in crossval_splits:
    #     debug_expanded_aids(expanded_aids, verbose=2)

    phis = {}
    for n_query_per_name, expanded_aids in crossval_splits:
        accumulators = []
        # with warnings.catch_warnings():
        for qaids, daids in expanded_aids:
            num_datab_pccs = len(np.unique(ibs.annots(daids).nids))
            num_query_pccs = len(np.unique(ibs.annots(qaids).nids))
            qreq_ = ibs.new_query_request(qaids, daids, verbose=False,
                                          cfgdict=pipe_cfg)

            cm_list = qreq_.execute()
            testres = test_result.TestResult.from_cms(cm_list, qreq_)
            nranks = testres.get_infoprop_list(key='qnx2_gt_name_rank')[0]
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

    ydatas = [phi.cumsum() for phi in phis.values()]
    label_list = list(map(str, phis.keys()))
    pt.multi_plot(xdata=np.arange(len(phi)), ydata_list=ydatas, label_list=label_list)

    ranks = 10
    ydatas = [phi.cumsum()[0:ranks] for phi in phis.values()]
    pt.multi_plot(xdata=np.arange(ranks), ydata_list=ydatas, label_list=label_list)

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
