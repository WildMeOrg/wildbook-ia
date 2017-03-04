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
    import ibeis.init.filter_annots import annot_crossval
    import plottool as pt
    pt.qtensure()

    # ibs = ibeis.opendb('GZ_Master1')
    ibs = ibeis.opendb('PZ_PB_RF_TRAIN')

    aids = ibs.filter_annots_general(require_timestamp=True, require_gps=True,
                                     is_known=True)

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

    phis = {}

    rng = np.random.RandomState(0)

    for n_query_per_name in range(1, 4):
        expanded_aids = annot_crossval(ibs, annots.aids,
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
