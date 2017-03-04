import numpy as np
import utool as ut


def stratified_annot_per_name_sample(ibs, aid_list, n_qaids_per_name=1,
                                     n_daids_per_name=1, rng=None, debug=True,
                                     n_folds=None):
    """ Stratified Sampling per name size """
    # Parameters
    rng = ut.ensure_rng(rng)
    n_need = n_qaids_per_name + n_daids_per_name

    # Group annotations by name
    annots = ibs.annots(aids=aid_list)
    nid_to_aids = ut.group_items(annots.aids, annots.nids)

    # Any name without enough data becomes a confusor
    # Otherwise we can use it in the sampling pool
    nid_to_confusors = {
        nid: aids for nid, aids in nid_to_aids.items()
        if len(aids) < n_need
    }
    nid_to_sample_pool = {
        nid: aids for nid, aids in nid_to_aids.items()
        if len(aids) >= n_need
    }

    if n_folds is None:
        # What is the maximum number of annotations in a name?
        maxsize_name = max(map(len, nid_to_sample_pool.values()))
        n_folds = maxsize_name

    raw_samples = [{} for _ in range(n_folds)]

    # Create several splits for each name
    for nid, aids in nid_to_sample_pool.items():
        # Randomly select combinations of appropriate size
        # idxs = ut.random_indexes(len(aids), rng=rng)
        combo_iter = ut.random_combinations(aids, n_need, n_folds, rng=rng)
        for count, aid_combo in enumerate(combo_iter):
            aid_combo = list(aid_combo)
            rng.shuffle(aid_combo)
            fold_qaids = aid_combo[:n_qaids_per_name]
            fold_daids = aid_combo[n_qaids_per_name:]
            fold_split = (fold_qaids, fold_daids)
            # Earlier samples will be biased towards names with more
            # annotations
            sample = raw_samples[count]
            sample[nid] = fold_split

    rebalance = True
    if not rebalance:
        # At this point we could create our folds like so:
        # But each fold would not have a good balance of qaids to daids

        unbalanced_samples = []
        for aid_split_ in raw_samples:
            qaids = []
            daids = []
            for qaids_, daids_ in aid_split_.values():
                qaids.extend(qaids_)
                daids.extend(daids_)
            unbalanced_samples.append((sorted(qaids), sorted(daids)))

        # if debug:
        #     # Notice how sets at the begining are larger and biased towards
        #     # annotations with more possible crossvalidation combinations
        #     print('len(unbalanced_samples) = %r' % (len(unbalanced_samples),))
        #     import warnings
        #     warnings.simplefilter('ignore', RuntimeWarning)
        #     totalq = 0
        #     for qaids, daids in unbalanced_samples:
        #         totalq += len(qaids)
        #         stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
        #                                           combo_enc_info=False)
        #         hashids = (stats['qaid_stats']['qhashid'],
        #                    stats['daid_stats']['dhashid'])
        #         print('hashids = %r' % (hashids,))
        #     print('totalq = %r' % (totalq,))
        annot_samples = unbalanced_samples
    elif rebalance:
        # We can rebalance these so each run is about the same size
        names = [nid for aid_split_ in raw_samples
                 for nid in aid_split_.keys()]
        aidsplits = [aids_ for aid_split_ in raw_samples
                     for aids_ in aid_split_.values()]
        group_to_idxs = ut.dzip(*ut.group_indices(names))
        freq = ut.dict_hist(names)
        g = list(freq.keys())[ut.argmax(list(freq.values()))]
        size = freq[g]
        new_splits = [[] for _ in range(size)]
        while True:
            try:
                g = list(freq.keys())[ut.argmax(list(freq.values()))]
                if freq[g] == 0:
                    raise StopIteration()
                group_idxs = group_to_idxs[g]
                group_to_idxs[g] = []
                freq[g] = 0
                priorityx = ut.argsort(list(map(len, new_splits)))
                for nextidx, splitx in zip(group_idxs, priorityx):
                    new_splits[splitx].append(nextidx)
            except StopIteration:
                break
        # name_splits = ut.unflat_take(groups, new_splits)
        aidsplits = ut.unflat_take(aidsplits, new_splits)

        rebalanced_samples = []
        for aidsplit in aidsplits:
            qaids = sorted(ut.flatten(ut.take_column(aidsplit, 0)))
            daids = sorted(ut.flatten(ut.take_column(aidsplit, 1)))
            rebalanced_samples.append((qaids, daids))

        # if debug:
        #     # Now we can see that the annotations are more evenly distributed
        #     # across the cross validation runs
        #     import warnings
        #     warnings.simplefilter('ignore', RuntimeWarning)
        #     print('len(rebalanced_samples) = %r' % (len(rebalanced_samples),))
        #     totalq = 0
        #     for qaids, daids in rebalanced_samples:
        #         totalq += len(qaids)
        #         stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
        #                                           combo_enc_info=False)
        #         hashids = (stats['qaid_stats']['qhashid'],
        #                    stats['daid_stats']['dhashid'])
        #         print('hashids = %r' % (hashids,))
        #     print('totalq = %r' % (totalq,))
        annot_samples = rebalanced_samples

    confusor_aids = ut.flatten(nid_to_confusors.values())
    expanded_aids_list = [(qaids, sorted(daids + confusor_aids))
                          for qaids, daids in annot_samples]

    if debug:
        # Now we can see that the annotations are more evenly distributed
        # across the cross validation runs
        import warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        print('len(expanded_aids_list) = %r' % (len(expanded_aids_list),))
        for qaids, daids in expanded_aids_list:
            stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
                                              combo_enc_info=False)
            hashids = (stats['qaid_stats']['qhashid'],
                       stats['daid_stats']['dhashid'])
            print('hashids = %r' % (hashids,))
    return expanded_aids_list


def learn_phi():
    # from ibeis.init import main_helpers
    # dbname = 'GZ_Master1'
    # a = 'timectrl'
    # t = 'baseline'
    # ibs, testres = main_helpers.testdata_expts(dbname, a=a, t=t)

    from ibeis.algo.preproc.occurrence_blackbox import cluster_timespace_sec
    import datetime
    import ibeis
    import plottool as pt
    pt.qtensure()

    ibs = ibeis.opendb('GZ_Master1')

    aids = ibs.filter_annots_general(require_timestamp=True, require_gps=True,
                                     is_known=True)

    annots = ibs.annots(aids=aids, asarray=True)
    # Take only annots with time and gps data
    # annots = annots.compress(~np.isnan(annots.image_unixtimes_asfloat))
    # annots = annots.compress(~np.isnan(np.array(annots.gps)).any(axis=1))

    # pt.draw_time_distribution(annots.image_unixtimes_asfloat, bw=1209600.0)

    cfgstr = str(ut.combine_uuids(annots.visual_uuids))

    class Encounter(ut.NiceRepr):
        def __init__(self, annots):
            self.annots = annots

        @property
        def name(self):
            return self.annots[0].name

        def __nice__(self):
            return self.name + ', ' + str(len(self.annots))

    cacher = ut.Cacher('occurrence_labels', cfgstr=cfgstr)
    data = cacher.tryload()
    if data is None:
        thresh_sec = datetime.timedelta(minutes=30).seconds
        print('Clustering occurrences')
        data = cluster_timespace_sec(
            annots.image_unixtimes_asfloat, annots.gps, thresh_sec=thresh_sec,
            km_per_sec=.002)
        cacher.save(data)
    occurrence_labels = data
    ndec = int(np.ceil(np.log10(max(occurrence_labels))))
    suffmt = '-occur%0' + str(ndec) + 'd'
    encounter_labels = [n + suffmt % (o,) for o, n in zip(occurrence_labels, annots.names)]
    enc_lookup = ut.dzip(annots.aids, encounter_labels)
    annots_per_enc = ut.dict_hist(encounter_labels, ordered=True)
    ut.get_stats(list(annots_per_enc.values()))

    encounters = ut.lmap(Encounter, annots.group(encounter_labels)[1])
    enc_names = [enc.name for enc in encounters]
    name_to_encounters = ut.group_items(encounters, enc_names)

    print('Names to num encounters')
    name_to_num_enc = ut.dict_hist(ut.map_dict_vals(len, name_to_encounters).values())

    # monkey patch to override encounter info
    def _monkey_get_annot_encounter_text(ibs, aids):
        return ut.dict_take(enc_lookup, aids)
    ut.inject_func_as_method(ibs, _monkey_get_annot_encounter_text,
                             'get_annot_encounter_text', force=True)

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

    pipe_cfg = {
        'resize_dim': 'area',
        'dim_size': 450,
    }
    # qreq_ = ibs.new_query_request(qaids, daids, verbose=False, query_cfg=pipe_cfg)
    # cm_list = qreq_.execute()
    # annots = ibs.annots(aids=aids)
    # nid_to_aids = ut.group_items(annots.aids, annots.nids)

    for n_query_per_name in range(1, 4):
        expanded_aids = stratified_annot_per_name_sample(ibs, annots.aids,
                                                         n_qaids_per_name=n_query_per_name,
                                                         n_daids_per_name=1,
                                                         n_folds=3,
                                                         rng=None, debug=False)
        accumulators = []
        # with warnings.catch_warnings():
        for qaids, daids in expanded_aids:

            qreq_ = ibs.new_query_request(qaids, daids, verbose=False, cfgdict=pipe_cfg)
            cm_list = qreq_.execute()
            accumulator = np.zeros(len(qreq_.daids))
            for cm in cm_list:
                cm = cm.extend_results(qreq_)
                rank = min(cm.get_annot_ranks(cm.get_groundtruth_daids()))
                accumulator[rank] += 1

            stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False, combo_enc_info=False)
            hashids = (stats['qaid_stats']['qhashid'],
                       stats['daid_stats']['dhashid'])
            print('hashids = %r' % (hashids,))
            print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))
            accumulators.append(accumulator)
