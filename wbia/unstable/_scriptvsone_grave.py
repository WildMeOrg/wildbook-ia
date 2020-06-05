# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import print_function, division, absolute_import
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def load_multiclass_scores(self):
    # convert simple scores to multiclass scores
    import vtool as vt

    self.multiclass_scores = {}
    for key in self.samples.simple_scores.keys():
        scores = self.samples.simple_scores[key].values
        # Hack scores into the range 0 to 1
        normer = vt.ScoreNormalizer(adjust=8, monotonize=True)
        normer.fit(scores, y=self.samples.is_same())
        normed_scores = normer.normalize_scores(scores)
        # Create a dimension for each class
        # but only populate two of the dimensions
        class_idxs = ut.take(self.samples.text_to_class, ['nomatch', 'match'])
        pred = np.zeros((len(scores), len(self.samples.class_names)))
        pred[:, class_idxs[0]] = 1 - normed_scores
        pred[:, class_idxs[1]] = normed_scores
        self.multiclass_scores[key] = pred


def photobombing_subset():
    """
    CommandLine:
        python -m wbia.scripts.script_vsone photobombing_subset
    """
    import wbia

    # pair_sample = ut.odict([
    #     ('top_gt', 4), ('mid_gt', 2), ('bot_gt', 2), ('rand_gt', 2),
    #     ('top_gf', 3), ('mid_gf', 2), ('bot_gf', 1), ('rand_gf', 2),
    # ])
    qreq_ = wbia.testdata_qreq_(
        defaultdb='PZ_Master1',
        a=':mingt=2,species=primary',
        # t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum',
        t='default:K=4,Knorm=1,score_method=csum,prescore_method=csum,QRH=True',
    )
    ibs = qreq_.ibs
    # cm_list = qreq_.execute()
    # infr = wbia.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)
    # aid_pairs_ = infr._cm_training_pairs(rng=np.random.RandomState(42),
    #                                      **pair_sample)

    # # ut.dict_hist(ut.flatten(am_tags))
    # am_rowids = ibs._get_all_annotmatch_rowids()
    # am_tags = ibs.get_annotmatch_case_tags(am_rowids)
    # am_flags = ut.filterflags_general_tags(am_tags, has_any=['photobomb'])
    # am_rowids_ = ut.compress(am_rowids, am_flags)
    # aids1 = ibs.get_annotmatch_aid1(am_rowids_)
    # aids2 = ibs.get_annotmatch_aid2(am_rowids_)
    # pb_aids_pairs = list(zip(aids1, aids2))

    # # aids = unique_pb_aids = ut.unique(ut.flatten(pb_aids_pairs))
    # # ut.compress(unique_pb_aids, ibs.is_aid_unknown(unique_pb_aids))

    # assert len(pb_aids_pairs) > 0

    # # Keep only a random subset
    # subset_idxs = list(range(len(aid_pairs_)))
    # rng = np.random.RandomState(3104855634)
    # num_max = len(pb_aids_pairs)
    # if num_max < len(subset_idxs):
    #     subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
    #     subset_idxs = sorted(subset_idxs)
    # aid_pairs_ = ut.take(aid_pairs_, subset_idxs)

    # aid_pairs_ += pb_aids_pairs
    # unique_aids = ut.unique(ut.flatten(aid_pairs_))

    # a1 = ibs.filter_annots_general(unique_aids, is_known=True, verbose=True, min_pername=2, has_none=['photobomb'])
    # a2 = ibs.filter_annots_general(unique_aids, has_any=['photobomb'], verbose=True, is_known=True)
    # a = sorted(set(a1 + a2))
    # ibs.print_annot_stats(a)
    # len(a)

    a = [
        8,
        27,
        30,
        86,
        87,
        90,
        92,
        94,
        99,
        103,
        104,
        106,
        111,
        217,
        218,
        242,
        298,
        424,
        425,
        456,
        464,
        465,
        472,
        482,
        529,
        559,
        574,
        585,
        588,
        592,
        598,
        599,
        601,
        617,
        630,
        645,
        661,
        664,
        667,
        694,
        723,
        724,
        759,
        768,
        843,
        846,
        861,
        862,
        866,
        933,
        934,
        980,
        987,
        1000,
        1003,
        1005,
        1011,
        1017,
        1020,
        1027,
        1059,
        1074,
        1076,
        1080,
        1095,
        1096,
        1107,
        1108,
        1192,
        1203,
        1206,
        1208,
        1220,
        1222,
        1223,
        1224,
        1256,
        1278,
        1293,
        1294,
        1295,
        1296,
        1454,
        1456,
        1474,
        1484,
        1498,
        1520,
        1521,
        1548,
        1563,
        1576,
        1593,
        1669,
        1675,
        1680,
        1699,
        1748,
        1751,
        1811,
        1813,
        1821,
        1839,
        1927,
        1934,
        1938,
        1952,
        1992,
        2003,
        2038,
        2054,
        2066,
        2080,
        2103,
        2111,
        2170,
        2171,
        2175,
        2192,
        2216,
        2227,
        2240,
        2250,
        2253,
        2266,
        2272,
        2288,
        2292,
        2314,
        2329,
        2341,
        2344,
        2378,
        2397,
        2417,
        2429,
        2444,
        2451,
        2507,
        2551,
        2552,
        2553,
        2581,
        2628,
        2640,
        2642,
        2646,
        2654,
        2667,
        2686,
        2733,
        2743,
        2750,
        2759,
        2803,
        2927,
        3008,
        3054,
        3077,
        3082,
        3185,
        3205,
        3284,
        3306,
        3334,
        3370,
        3386,
        3390,
        3393,
        3401,
        3448,
        3508,
        3542,
        3597,
        3614,
        3680,
        3684,
        3695,
        3707,
        3727,
        3758,
        3765,
        3790,
        3812,
        3813,
        3818,
        3858,
        3860,
        3874,
        3875,
        3887,
        3892,
        3915,
        3918,
        3924,
        3927,
        3929,
        3933,
        3941,
        3952,
        3955,
        3956,
        3959,
        4004,
        4059,
        4073,
        4076,
        4089,
        4094,
        4124,
        4126,
        4128,
        4182,
        4189,
        4217,
        4222,
        4229,
        4257,
        4266,
        4268,
        4288,
        4289,
        4296,
        4306,
        4339,
        4353,
        4376,
        4403,
        4428,
        4455,
        4487,
        4494,
        4515,
        4517,
        4524,
        4541,
        4544,
        4556,
        4580,
        4585,
        4597,
        4604,
        4629,
        4639,
        4668,
        4671,
        4672,
        4675,
        4686,
        4688,
        4693,
        4716,
        4730,
        4731,
        4749,
        4772,
        4803,
        4820,
        4823,
        4832,
        4833,
        4836,
        4900,
        4902,
        4909,
        4924,
        4936,
        4938,
        4939,
        4944,
        5004,
        5006,
        5034,
        5043,
        5044,
        5055,
        5064,
        5072,
        5115,
        5131,
        5150,
        5159,
        5165,
        5167,
        5168,
        5174,
        5218,
        5235,
        5245,
        5249,
        5309,
        5319,
        5334,
        5339,
        5344,
        5347,
        5378,
        5379,
        5384,
        5430,
        5447,
        5466,
        5509,
        5546,
        5587,
        5588,
        5621,
        5640,
        5663,
        5676,
        5682,
        5685,
        5687,
        5690,
        5707,
        5717,
        5726,
        5732,
        5733,
        5791,
        5830,
        5863,
        5864,
        5869,
        5870,
        5877,
        5879,
        5905,
        5950,
        6008,
        6110,
        6134,
        6160,
        6167,
        6234,
        6238,
        6265,
        6344,
        6345,
        6367,
        6384,
        6386,
        6437,
        6495,
        6533,
        6538,
        6569,
        6587,
        6626,
        6634,
        6643,
        6659,
        6661,
        6689,
        6714,
        6725,
        6739,
        6754,
        6757,
        6759,
        6763,
        6781,
        6830,
        6841,
        6843,
        6893,
        6897,
        6913,
        6930,
        6932,
        6936,
        6944,
        6976,
        7003,
        7022,
        7037,
        7052,
        7058,
        7074,
        7103,
        7107,
        7108,
        7113,
        7143,
        7183,
        7185,
        7187,
        7198,
        7200,
        7202,
        7207,
        7222,
        7275,
        7285,
        7388,
        7413,
        7421,
        7425,
        7429,
        7445,
        7487,
        7507,
        7508,
        7528,
        7615,
        7655,
        7696,
        7762,
        7786,
        7787,
        7796,
        7797,
        7801,
        7807,
        7808,
        7809,
        7826,
        7834,
        7835,
        7852,
        7861,
        7874,
        7881,
        7901,
        7902,
        7905,
        7913,
        7918,
        7941,
        7945,
        7990,
        7999,
        8007,
        8009,
        8017,
        8018,
        8019,
        8034,
        8041,
        8057,
        8058,
        8079,
        8080,
        8086,
        8089,
        8092,
        8094,
        8100,
        8105,
        8109,
        8147,
        8149,
        8153,
        8221,
        8264,
        8302,
        8303,
        8331,
        8366,
        8367,
        8370,
        8376,
        8474,
        8501,
        8504,
        8506,
        8507,
        8514,
        8531,
        8532,
        8534,
        8538,
        8563,
        8564,
        8587,
        8604,
        8608,
        8751,
        8771,
        8792,
        9175,
        9204,
        9589,
        9726,
        9841,
        10674,
        12122,
        12305,
        12796,
        12944,
        12947,
        12963,
        12966,
        13098,
        13099,
        13101,
        13103,
        13109,
        13147,
        13157,
        13168,
        13194,
        13236,
        13253,
        13255,
        13410,
        13450,
        13474,
        13477,
        13481,
        13508,
        13630,
        13670,
        13727,
        13741,
        13819,
        13820,
        13908,
        13912,
        13968,
        13979,
        14007,
        14009,
        14010,
        14019,
        14066,
        14067,
        14072,
        14074,
        14148,
        14153,
        14224,
        14230,
        14237,
        14239,
        14241,
        14274,
        14277,
        14290,
        14293,
        14308,
        14309,
        14313,
        14319,
        14668,
        14670,
        14776,
        14918,
        14920,
        14924,
        15135,
        15157,
        15318,
        15319,
        15490,
        15518,
        15531,
        15777,
        15903,
        15913,
        16004,
        16012,
        16013,
        16014,
        16020,
        16215,
        16221,
        16235,
        16240,
        16259,
        16273,
        16279,
        16284,
        16289,
        16316,
        16322,
        16329,
        16336,
        16364,
        16389,
        16706,
        16897,
        16898,
        16903,
        16949,
        17094,
        17101,
        17137,
        17200,
        17222,
        17290,
        17327,
        17336,
    ]

    from wbia.dbio import export_subset

    export_subset.export_annots(ibs, a, 'PZ_PB_RF_TRAIN')

    # closed_aids = ibs.annots(unique_aids).get_name_image_closure()

    # annots = ibs.annots(unique_aids)
    # closed_gt_aids = ut.unique(ut.flatten(ibs.get_annot_groundtruth(unique_aids)))
    # closed_gt_aids = ut.unique(ut.flatten(ibs.get_annot_groundtruth(unique_aids)))
    # closed_img_aids = ut.unique(ut.flatten(ibs.get_annot_otherimage_aids(unique_aids)))

    # ibs.print_annot_stats(unique_aids)
    # all_annots = ibs.annots()


def bigcache_vsone(qreq_, hyper_params):
    """
    Cached output of one-vs-one matches

        >>> from wbia.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> qreq_ = self.qreq_
        >>> hyper_params = self.hyper_params
    """
    import vtool as vt
    import wbia

    # Get a set of training pairs
    ibs = qreq_.ibs
    cm_list = qreq_.execute()
    infr = wbia.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)

    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(
        rng=np.random.RandomState(42), **hyper_params.pair_sample
    )

    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_), directed=False).tolist()

    pb_aid_pairs_ = photobomb_samples(ibs)

    # TODO: try to add in more non-comparable samples
    aid_pairs_ = pb_aid_pairs_ + aid_pairs_
    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_))

    # ======================================
    # Compute one-vs-one scores and local_measures
    # ======================================

    # Prepare lazy attributes for annotations
    qreq_ = infr.qreq_
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    # Remove any pairs missing features
    if dannot_cfg == qannot_cfg:
        unique_annots = ibs.annots(np.unique(np.array(aid_pairs_)), config=dannot_cfg)
        bad_aids = unique_annots.compress(~np.array(unique_annots.num_feats) > 0).aids
        bad_aids = set(bad_aids)
    else:
        annots1_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 0)), config=qannot_cfg)
        annots2_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 1)), config=dannot_cfg)
        bad_aids1 = annots1_.compress(~np.array(annots1_.num_feats) > 0).aids
        bad_aids2 = annots2_.compress(~np.array(annots2_.num_feats) > 0).aids
        bad_aids = set(bad_aids1 + bad_aids2)
    subset_idxs = np.where(
        [not (a1 in bad_aids or a2 in bad_aids) for a1, a2 in aid_pairs_]
    )[0]
    # Keep only a random subset
    if hyper_params.subsample:
        rng = np.random.RandomState(3104855634)
        num_max = hyper_params.subsample
        if num_max < len(subset_idxs):
            subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
            subset_idxs = sorted(subset_idxs)

    # Take the current selection
    aid_pairs = ut.take(aid_pairs_, subset_idxs)

    if True:
        # NEW WAY
        config = hyper_params.vsone_assign
        # TODO: ensure annot probs like chips and features can be appropriately
        # set via qreq_ config or whatever
        matches = infr.exec_vsone_subset(aid_pairs, config=config)
    else:
        query_aids = ut.take_column(aid_pairs, 0)
        data_aids = ut.take_column(aid_pairs, 1)
        # OLD WAY
        # Determine a unique set of annots per config
        configured_aids = ut.ddict(set)
        configured_aids[qannot_cfg].update(query_aids)
        configured_aids[dannot_cfg].update(data_aids)

        # Make efficient annot-object representation
        configured_obj_annots = {}
        for config, aids in configured_aids.items():
            annots = ibs.annots(sorted(list(aids)), config=config)
            configured_obj_annots[config] = annots

        annots1 = configured_obj_annots[qannot_cfg].loc(query_aids)
        annots2 = configured_obj_annots[dannot_cfg].loc(data_aids)

        # Get hash based on visual annotation appearence of each pair
        # as well as algorithm configurations used to compute those properties
        qvuuids = annots1.visual_uuids
        dvuuids = annots2.visual_uuids
        qcfgstr = annots1._config.get_cfgstr()
        dcfgstr = annots2._config.get_cfgstr()
        annots_cfgstr = ut.hashstr27(qcfgstr) + ut.hashstr27(dcfgstr)
        vsone_uuids = [
            ut.combine_uuids(uuids, salt=annots_cfgstr)
            for uuids in ut.ProgIter(
                zip(qvuuids, dvuuids), length=len(qvuuids), label='hashing ids'
            )
        ]

        # Combine into a big cache for the entire 1-v-1 matching run
        big_uuid = ut.hashstr_arr27(vsone_uuids, '', pathsafe=True)
        cacher = ut.Cacher('vsone_v7', cfgstr=str(big_uuid), appname='vsone_rf_train')

        cached_data = cacher.tryload()
        if cached_data is not None:
            # Caching doesn't work 100% for PairwiseMatch object, so we need to do
            # some postprocessing
            configured_lazy_annots = ut.ddict(dict)
            for config, annots in configured_obj_annots.items():
                annot_dict = configured_lazy_annots[config]
                for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
                    annot_dict[_annot.aid] = _annot._make_lazy_dict()

            # Extract pairs of annot objects (with shared caches)
            lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
            lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

            # Create a set of PairwiseMatches with the correct annot properties
            matches = [
                vt.PairwiseMatch(annot1, annot2)
                for annot1, annot2 in zip(lazy_annots1, lazy_annots2)
            ]

            # Updating a new matches dictionary ensure the annot1/annot2 properties
            # are set correctly
            for key, cached_matches in list(cached_data.items()):
                fixed_matches = [match.copy() for match in matches]
                for fixed, internal in zip(fixed_matches, cached_matches):
                    dict_ = internal.__dict__
                    ut.delete_dict_keys(dict_, ['annot1', 'annot2'])
                    fixed.__dict__.update(dict_)
                cached_data[key] = fixed_matches
        else:
            cached_data = vsone_(
                qreq_,
                query_aids,
                data_aids,
                qannot_cfg,
                dannot_cfg,
                configured_obj_annots,
                hyper_params,
            )
            cacher.save(cached_data)
        # key_ = 'SV_LNBNN'
        key_ = 'RAT_SV'
        # for key in list(cached_data.keys()):
        #     if key != 'SV_LNBNN':
        #         del cached_data[key]
        matches = cached_data[key_]
    return matches, infr


def bigcache_vsone(qreq_, hyper_params):
    """
    Cached output of one-vs-one matches

        >>> from wbia.scripts.script_vsone import *  # NOQA
        >>> self = OneVsOneProblem()
        >>> qreq_ = self.qreq_
        >>> hyper_params = self.hyper_params
    """
    import vtool as vt
    import wbia

    # Get a set of training pairs
    ibs = qreq_.ibs
    cm_list = qreq_.execute()
    infr = wbia.AnnotInference.from_qreq_(qreq_, cm_list, autoinit=True)

    # Per query choose a set of correct, incorrect, and random training pairs
    aid_pairs_ = infr._cm_training_pairs(
        rng=np.random.RandomState(42), **hyper_params.pair_sample
    )

    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_), directed=False).tolist()

    pb_aid_pairs_ = photobomb_samples(ibs)

    # TODO: try to add in more non-comparable samples
    aid_pairs_ = pb_aid_pairs_ + aid_pairs_
    aid_pairs_ = vt.unique_rows(np.array(aid_pairs_))

    # ======================================
    # Compute one-vs-one scores and local_measures
    # ======================================

    # Prepare lazy attributes for annotations
    qreq_ = infr.qreq_
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    # Remove any pairs missing features
    if dannot_cfg == qannot_cfg:
        unique_annots = ibs.annots(np.unique(np.array(aid_pairs_)), config=dannot_cfg)
        bad_aids = unique_annots.compress(~np.array(unique_annots.num_feats) > 0).aids
        bad_aids = set(bad_aids)
    else:
        annots1_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 0)), config=qannot_cfg)
        annots2_ = ibs.annots(ut.unique(ut.take_column(aid_pairs_, 1)), config=dannot_cfg)
        bad_aids1 = annots1_.compress(~np.array(annots1_.num_feats) > 0).aids
        bad_aids2 = annots2_.compress(~np.array(annots2_.num_feats) > 0).aids
        bad_aids = set(bad_aids1 + bad_aids2)
    subset_idxs = np.where(
        [not (a1 in bad_aids or a2 in bad_aids) for a1, a2 in aid_pairs_]
    )[0]
    # Keep only a random subset
    if hyper_params.subsample:
        rng = np.random.RandomState(3104855634)
        num_max = hyper_params.subsample
        if num_max < len(subset_idxs):
            subset_idxs = rng.choice(subset_idxs, size=num_max, replace=False)
            subset_idxs = sorted(subset_idxs)

    # Take the current selection
    aid_pairs = ut.take(aid_pairs_, subset_idxs)

    if True:
        # NEW WAY
        config = hyper_params.vsone_assign
        # TODO: ensure annot probs like chips and features can be appropriately
        # set via qreq_ config or whatever
        matches = infr.exec_vsone_subset(aid_pairs, config=config)
    else:
        query_aids = ut.take_column(aid_pairs, 0)
        data_aids = ut.take_column(aid_pairs, 1)
        # OLD WAY
        # Determine a unique set of annots per config
        configured_aids = ut.ddict(set)
        configured_aids[qannot_cfg].update(query_aids)
        configured_aids[dannot_cfg].update(data_aids)

        # Make efficient annot-object representation
        configured_obj_annots = {}
        for config, aids in configured_aids.items():
            annots = ibs.annots(sorted(list(aids)), config=config)
            configured_obj_annots[config] = annots

        annots1 = configured_obj_annots[qannot_cfg].loc(query_aids)
        annots2 = configured_obj_annots[dannot_cfg].loc(data_aids)

        # Get hash based on visual annotation appearence of each pair
        # as well as algorithm configurations used to compute those properties
        qvuuids = annots1.visual_uuids
        dvuuids = annots2.visual_uuids
        qcfgstr = annots1._config.get_cfgstr()
        dcfgstr = annots2._config.get_cfgstr()
        annots_cfgstr = ut.hashstr27(qcfgstr) + ut.hashstr27(dcfgstr)
        vsone_uuids = [
            ut.combine_uuids(uuids, salt=annots_cfgstr)
            for uuids in ut.ProgIter(
                zip(qvuuids, dvuuids), length=len(qvuuids), label='hashing ids'
            )
        ]

        # Combine into a big cache for the entire 1-v-1 matching run
        big_uuid = ut.hashstr_arr27(vsone_uuids, '', pathsafe=True)
        cacher = ut.Cacher('vsone_v7', cfgstr=str(big_uuid), appname='vsone_rf_train')

        cached_data = cacher.tryload()
        if cached_data is not None:
            # Caching doesn't work 100% for PairwiseMatch object, so we need to do
            # some postprocessing
            configured_lazy_annots = ut.ddict(dict)
            for config, annots in configured_obj_annots.items():
                annot_dict = configured_lazy_annots[config]
                for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
                    annot_dict[_annot.aid] = _annot._make_lazy_dict()

            # Extract pairs of annot objects (with shared caches)
            lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
            lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

            # Create a set of PairwiseMatches with the correct annot properties
            matches = [
                vt.PairwiseMatch(annot1, annot2)
                for annot1, annot2 in zip(lazy_annots1, lazy_annots2)
            ]

            # Updating a new matches dictionary ensure the annot1/annot2 properties
            # are set correctly
            for key, cached_matches in list(cached_data.items()):
                fixed_matches = [match.copy() for match in matches]
                for fixed, internal in zip(fixed_matches, cached_matches):
                    dict_ = internal.__dict__
                    ut.delete_dict_keys(dict_, ['annot1', 'annot2'])
                    fixed.__dict__.update(dict_)
                cached_data[key] = fixed_matches
        else:
            cached_data = vsone_(
                qreq_,
                query_aids,
                data_aids,
                qannot_cfg,
                dannot_cfg,
                configured_obj_annots,
                hyper_params,
            )
            cacher.save(cached_data)
        # key_ = 'SV_LNBNN'
        key_ = 'RAT_SV'
        # for key in list(cached_data.keys()):
        #     if key != 'SV_LNBNN':
        #         del cached_data[key]
        matches = cached_data[key_]
    return matches, infr


def vsone_(
    qreq_,
    query_aids,
    data_aids,
    qannot_cfg,
    dannot_cfg,
    configured_obj_annots,
    hyper_params,
):
    # Do vectorized preload before constructing lazy dicts
    # Then make sure the lazy dicts point to this subset
    unique_obj_annots = list(configured_obj_annots.values())
    for annots in ut.ProgIter(unique_obj_annots, 'vectorized preload'):
        annots.set_caching(True)
        annots.chip_size
        annots.vecs
        annots.kpts
        annots.yaw
        annots.qual
        annots.gps
        annots.time
        if qreq_.qparams.featweight_enabled:
            annots.fgweights
    # annots._internal_attrs.clear()

    # Make convinient lazy dict representations (after loading pre info)
    configured_lazy_annots = ut.ddict(dict)
    for config, annots in configured_obj_annots.items():
        annot_dict = configured_lazy_annots[config]
        for _annot in ut.ProgIter(annots.scalars(), label='make lazy dict'):
            annot = _annot._make_lazy_dict()
            annot_dict[_annot.aid] = annot

    unique_lazy_annots = ut.flatten([x.values() for x in configured_lazy_annots.values()])

    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    for annot in ut.ProgIter(unique_lazy_annots, label='lazy flann'):
        vt.matching.ensure_metadata_flann(annot, flann_params)
        vt.matching.ensure_metadata_normxy(annot)

    for annot in ut.ProgIter(unique_lazy_annots, 'preload kpts'):
        annot['kpts']
    for annot in ut.ProgIter(unique_lazy_annots, 'preload normxy'):
        annot['norm_xys']
    for annot in ut.ProgIter(unique_lazy_annots, 'preload vecs'):
        annot['vecs']

    # Extract pairs of annot objects (with shared caches)
    lazy_annots1 = ut.take(configured_lazy_annots[qannot_cfg], query_aids)
    lazy_annots2 = ut.take(configured_lazy_annots[dannot_cfg], data_aids)

    # TODO: param search over grid
    #     'use_sv': [0, 1],
    #     'use_fg': [0, 1],
    #     'use_ratio_test': [0, 1],
    matches_RAT = [
        vt.PairwiseMatch(annot1, annot2)
        for annot1, annot2 in zip(lazy_annots1, lazy_annots2)
    ]

    # Construct global measurements
    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(matches_RAT, label='setup globals'):
        match.add_global_measures(global_keys)

    # Preload flann for only specific annots
    for match in ut.ProgIter(matches_RAT, label='preload FLANN'):
        match.annot1['flann']

    cfgdict = hyper_params.vsone_assign
    # Find one-vs-one matches
    # cfgdict = {'checks': 20, 'symmetric': False}
    for match in ut.ProgIter(matches_RAT, label='assign vsone'):
        match.assign(cfgdict=cfgdict)

    # gridsearch_ratio_thresh()
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_ratio_test', {
    #     'ratio_thresh': np.linspace(.6, .7, 50)
    # })
    for match in ut.ProgIter(matches_RAT, label='apply ratio thresh'):
        match.apply_ratio_test({'ratio_thresh': 0.638}, inplace=True)

    # TODO gridsearch over sv params
    # vt.matching.gridsearch_match_operation(matches_RAT, 'apply_sver', {
    #     'xy_thresh': np.linspace(0, 1, 3)
    # })
    matches_RAT_SV = [
        match.apply_sver(inplace=True) for match in ut.ProgIter(matches_RAT, label='sver')
    ]

    # Add keypoint spatial information to local features
    for match in matches_RAT_SV:
        match.add_local_measures()
        # key_ = 'norm_xys'
        # norm_xy1 = match.annot1[key_].take(match.fm.T[0], axis=1)
        # norm_xy2 = match.annot2[key_].take(match.fm.T[1], axis=1)
        # match.local_measures['norm_x1'] = norm_xy1[0]
        # match.local_measures['norm_y1'] = norm_xy1[1]
        # match.local_measures['norm_x2'] = norm_xy2[0]
        # match.local_measures['norm_y2'] = norm_xy2[1]

        # match.local_measures['scale1'] = vt.get_scales(
        #     match.annot1['kpts'].take(match.fm.T[0], axis=0))
        # match.local_measures['scale2'] = vt.get_scales(
        #     match.annot2['kpts'].take(match.fm.T[1], axis=0))

    # Create another version where we find global normalizers for the data
    # qreq_.load_indexer()
    # matches_SV_LNBNN = batch_apply_lnbnn(matches_RAT_SV, qreq_, inplace=True)

    # if 'weight' in cfgdict:
    #     for match in matches_SV_LNBNN[::-1]:
    #         lnbnn_dist = match.local_measures['lnbnn']
    #         ndist = match.local_measures['lnbnn_norm_dist']
    #         weights = match.local_measures[cfgdict['weight']]
    #         match.local_measures['weighted_lnbnn'] = weights * lnbnn_dist
    #         match.local_measures['weighted_lnbnn_norm_dist'] = weights * ndist
    #         match.fs = match.local_measures['weighted_lnbnn']

    cached_data = {
        # 'RAT': matches_RAT,
        'RAT_SV': matches_RAT_SV,
        # 'SV_LNBNN': matches_SV_LNBNN,
    }
    return cached_data

    from sklearn.metrics.classification import coo_matrix

    def quick_cm(y_true, y_pred, labels, sample_weight):
        n_labels = len(labels)
        C = coo_matrix(
            (sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels)
        ).toarray()
        return C

    def quick_mcc(C):
        """ assumes y_true and y_pred are in index/encoded format """
        t_sum = C.sum(axis=1)
        p_sum = C.sum(axis=0)
        n_correct = np.diag(C).sum()
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
        return mcc

    def mcc_hack():
        sample_weight = np.ones(len(self.samples), dtype=np.int)
        task_mccs = ut.ddict(dict)
        # Determine threshold levels per score type
        score_to_order = {}
        for scoretype in score_dict.keys():
            y_score = score_dict[scoretype].values
            sortx = np.argsort(y_score, kind='mergesort')[::-1]
            y_score = y_score[sortx]
            distinct_value_indices = np.where(np.diff(y_score))[0]
            threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]
            thresh = y_score[threshold_idxs]
            score_to_order[scoretype] = (sortx, y_score, thresh)

        classes_ = np.array([0, 1], dtype=np.int)
        for task in task_list:
            labels = self.samples.subtasks[task]
            for sublabels in labels.gen_one_vs_rest_labels():
                for scoretype in score_dict.keys():
                    sortx, y_score, thresh = score_to_order[scoretype]
                    y_true = sublabels.y_enc[sortx]
                    mcc = -np.inf
                    for t in thresh:
                        y_pred = (y_score > t).astype(np.int)
                        C1 = quick_cm(y_true, y_pred, classes_, sample_weight)
                        mcc1 = quick_mcc(C1)
                        if mcc1 < 0:
                            C2 = quick_cm(y_true, 1 - y_pred, classes_, sample_weight)
                            mcc1 = quick_mcc(C2)
                        mcc = max(mcc1, mcc)
                    # print('mcc = %r' % (mcc,))
                    task_mccs[sublabels.task_name][scoretype] = mcc
        return task_mccs

    if 0:
        with ut.Timer('mcc'):
            task_mccs = mcc_hack()
            print('\nMCC of simple scoring measures:')
            df = pd.DataFrame.from_dict(task_mccs, orient='index')
            from utool.experimental.pandas_highlight import to_string_monkey

            print(to_string_monkey(df, highlight_cols=np.arange(len(df.columns))))

        # _all_dfs.append(df_rf)
        # df_all = pd.concat(_all_dfs, axis=1)

        # # Add in the simple scores
        # from utool.experimental.pandas_highlight import to_string_monkey
        # print(to_string_monkey(df_all, highlight_cols=np.arange(len(df_all.columns))))

        # best_name = df_all.columns[df_all.values.argmax()]
        # pt.show_if_requested()
        # import utool
        # utool.embed()
        # print('rat_sver_rf_auc = %r' % (rat_sver_rf_auc,))
        # columns = ['Method', 'AUC']
        # data = [
        #     ['1vM-LNBNN',       vsmany_lnbnn_auc],
        #     ['1v1-LNBNN',       vsone_sver_lnbnn_auc],
        #     ['1v1-RAT',         rat_auc],
        #     ['1v1-RAT+SVER',    rat_sver_auc],
        #     ['1v1-RAT+SVER+RF', rat_sver_rf_auc],
        # ]
        # table = pd.DataFrame(data, columns=columns)
        # error = 1 - table['AUC']
        # orig = 1 - vsmany_lnbnn_auc
        # import tabulate
        # table = table.assign(percent_error_decrease=(orig - error) / orig * 100)
        # col_to_nice = {
        #     'percent_error_decrease': '% error decrease',
        # }
        # header = [col_to_nice.get(c, c) for c in table.columns]
        # print(tabulate.tabulate(table.values, header, tablefmt='orgtbl'))
