import numpy as np
import utool as ut

def debug_expanded_aids(expanded_aids_list, verbose=1):
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    print('len(expanded_aids_list) = %r' % (len(expanded_aids_list),))
    for qaids, daids in expanded_aids_list:
        stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
                                          combo_enc_info=False)
        hashids = (stats['qaid_stats']['qhashid'],
                   stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        if verbose > 1:
            print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def stratified_annot_per_name_sample(ibs, aid_list, n_qaids_per_name=1,
                                     n_daids_per_name=1, rng=None, debug=True,
                                     n_splits=None):
    """
    Stratified Sampling per name size

    Args:
        n_splits (int): number of query/database splits to create.
            note, some names may not be big enough to split this many times.

    CommandLine:
        python -m ibeis.scripts.iccv stratified_annot_per_name_sample

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.iccv import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> n_qaids_per_name = 1
        >>> n_daids_per_name = 1
        >>> rng = 0
        >>> debug = True
        >>> n_splits = None
        >>> expanded_aids_list = stratified_annot_per_name_sample(ibs, aid_list, n_qaids_per_name, n_daids_per_name, rng, debug, n_splits)
        >>> result = ('expanded_aids_list = %s' % (ut.repr2(expanded_aids_list),))
        >>> print(result)
    """
    # Parameters
    rng = ut.ensure_rng(rng)
    n_need = n_qaids_per_name + n_daids_per_name
    rebalance = True

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

    if n_splits is None:
        # What is the maximum number of annotations in a name?
        maxsize_name = max(map(len, nid_to_sample_pool.values()))
        n_splits = maxsize_name

    # This is a list of dictionaries that maps a name to one possible split
    split_samples = [{} for _ in range(n_splits)]

    # Create a mapping from each name to the possible split combos
    nid_to_splits = ut.ddict(list)

    # Create several splits for each name
    for nid, aids in nid_to_sample_pool.items():
        # Randomly select combinations of appropriate size
        combo_iter = ut.random_combinations(aids, n_need, n_splits, rng=rng)
        for count, aid_combo in enumerate(combo_iter):
            aid_combo = list(aid_combo)
            rng.shuffle(aid_combo)
            fold_split = (aid_combo[:n_qaids_per_name],
                          aid_combo[n_qaids_per_name:])
            nid_to_splits[nid].append(fold_split)
            # Earlier samples will be biased towards names with more annots
            split_samples[count][nid] = fold_split

    # Some names may have more splits than others
    nid_to_nsplits = ut.map_vals(len, nid_to_splits)
    # Find the name with the most splits
    max_nid = ut.argmax(nid_to_nsplits)
    max_size = nid_to_nsplits[max_nid]

    # if max_size < n_splits:
    #     warnings.warn('Splits will not be full')
    assert max_size <= n_splits, 'cycle assumption does not hold'
    new_splits = [[] for _ in range(n_splits)]
    if rebalance:
        # Rebalance by adding combos from each name in a cycle.
        # The difference between the largest and smallest split is at most one.
        for count, aid_combo in enumerate(ut.iflatten(nid_to_splits.values())):
            new_splits[count % len(new_splits)].append(aid_combo)
    else:
        # No rebalancing. The first split contains everything from the dataset
        # and subsequent splits contain less and less.
        for nid, aid_combos in nid_to_splits.items():
            for count, aid_combo in enumerate(aid_combos):
                new_splits[count].append(aid_combo)

    # Reshape into an expanded aids list
    expanded_aids_list = [
        [sorted(ut.flatten(qaids_)), sorted(ut.flatten(daids_))]
        for qaids_, daids_ in (ut.listT(splits) for splits in new_splits)
    ]

    if debug:
        debug_expanded_aids(expanded_aids_list)

    # Add confusors the the dataset
    confusor_aids = ut.flatten(nid_to_confusors.values())
    expanded_aids_list = [(qaids, sorted(daids + confusor_aids))
                          for qaids, daids in expanded_aids_list]

    if debug:
        debug_expanded_aids(expanded_aids_list)
    return expanded_aids_list


def encounter_stuff():
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

    ibs = ibeis.opendb('PZ_Master1')

    aids = ibs.filter_annots_general(require_timestamp=True, require_gps=True,
                                     is_known=True)

    annots = ibs.annots(aids=aids, asarray=Grue)
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

    phis = {}

    rng = np.random.RandomState(0)

    for n_query_per_name in range(1, 4):
        expanded_aids = stratified_annot_per_name_sample(ibs, annots.aids,
                                                         n_qaids_per_name=n_query_per_name,
                                                         n_daids_per_name=1,
                                                         n_splits=3,
                                                         rng=rng, debug=False)
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
        phis[n_query_per_name] = accumulators
