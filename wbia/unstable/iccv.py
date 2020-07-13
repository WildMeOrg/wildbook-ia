# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import utool as ut
import pathlib
from wbia.algo.graph.state import POSTV, NEGTV, INCMP

print, rrr, profile = ut.inject2(__name__)


def qt_review():
    """
    CommandLine:
        python -m wbia.scripts.iccv qt_review
    """
    import wbia.guitool as gt
    import wbia

    app = gt.ensure_qapp()[0]  # NOQA
    defaultdb = ut.get_argval('--db', default='GZ_Master1')
    ibs = wbia.opendb(defaultdb=defaultdb)
    infr = wbia.AnnotInference(ibs=ibs, aids='all', autoinit=True, verbose=True)
    infr.params['redun.pos'] = 2
    infr.params['redun.neg'] = 2

    infr.reset_feedback('staging', apply=True)
    if False:
        infr.relabel_using_reviews(rectify=False)
        infr.reset_labels_to_wbia()
        infr.relabel_using_reviews(rectify=True)
        infr.wbia_delta_info()
        infr.name_group_stats(verbose=1)
        infr.wbia_name_group_delta_info(verbose=1)

    # After we ensure that the annotmatch stuff is in staging, and all our
    # reviews are there we load them. Then we rectify the old name label
    # id-scheme by reviewing dummy edges. These edges should be saved to the
    # staging database as dummy edges, but they wont overwrite any review that
    # we've already made. This will likely cause inconsistency.
    if False:
        infr.review_dummy_edges(method=3)
        infr.write_wbia_staging_feedback()

    infr.prioritize()
    infr.verifiers = None
    if False:
        pccs = list(infr.non_pos_redundant_pccs())
        pccs = ut.sortedby(pccs, ut.lmap(len, pccs))
        pcc = pccs[-1]
        aug_edges = infr.find_pos_augment_edges(pcc)
        infr.add_new_candidate_edges(aug_edges)
    infr.qt_review_loop()
    gt.qtapp_loop(qwin=infr.manual_wgt, freq=10)
    return
    # annots = ibs.annots(infr.aids)
    # for key, val in ut.group_items(annots.nids, annots.names).items():
    #     if len(set(val)) > 1:
    #         print((key, val))
    # ut.get_stats(ut.lmap(len, ut.group_items(annots.aids, annots.nids).values()))
    # ut.get_stats(ut.lmap(len, ut.group_items(annots.aids, annots.names).values()))

    # infr.set_node_attrs(
    #     'name_label', ut.dzip(annots.aids, annots.nids),
    # )
    # infr.review_dummy_edges(method=1)
    # infr.apply_nondynamic_update()
    # for edge, vals in infr.read_wbia_staging_feedback().items():
    #     feedback = infr._rectify_feedback_item(vals)
    #     infr.add_feedback(edge, **feedback)

    # infr.reset_feedback('staging', apply=True)
    # infr.review_dummy_edges(method=2)
    # infr.reset_feedback('staging', apply=True)
    # infr.relabel_using_reviews()

    infr.recovery_review_loop()
    infr.write_wbia_staging_feedback()
    win = infr.start_qt_interface(loop=False)
    win.show()
    gt.qtapp_loop(qwin=win, freq=10)

    # USE THE RECOVERY LOOP

    # nodes = ut.flatten(infr.recovery_ccs)
    # graph = infr.graph.subgraph(nodes)
    # infr.show_graph(graph=graph)
    # infr = wbia.AnnotInference(ibs=ibs, aids=nodes, autoinit=True,
    #                             verbose=True)
    # infr.reset_feedback('staging', apply=True)
    # infr.relabel_using_reviews()
    # win = infr.start_qt_interface(loop=False)


def gt_review():
    r"""
    CommandLine:
        python -m wbia.scripts.iccv gt_review --db PZ_MTEST
        python -m wbia.scripts.iccv gt_review --db GZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> result = gt_review()
        >>> print(result)
    """
    import wbia
    import ubelt as ub

    defaultdb = ut.get_argval('--db', default='GZ_Master1')

    # defaultdb = 'PZ_MTEST'
    cacher = ub.Cacher('tmp_gz_review', defaultdb + 'v4')
    data = cacher.tryload()
    if data is None:
        ibs = wbia.opendb(defaultdb=defaultdb)
        infr = wbia.AnnotInference(
            ibs=ibs, aids=ibs.get_valid_aids(), autoinit=True, verbose=True
        )
        # TODO: ensure that staging has all data from annotmatch in it
        infr.reset_feedback('staging', apply=True)
        # infr.reset_staging_with_ensure()
        # infr.reset_feedback('annotmatch', apply=True)
        infr.review_dummy_edges(method=2)

        infr.learn_evaluataion_clasifiers()

        want_edges = list(infr.edges())
        pblm = infr.verifiers
        task_probs = pblm.predict_proba_evaluation(infr, want_edges, ['match_state'])
        match_probs = task_probs[pblm.primary_task_key]
        len(match_probs)
        len(want_edges)

        real_probs = infr.match_state_df(want_edges).astype(np.float)
        real_probs, pred_probs = real_probs.align(match_probs)

        pred_probs['hardness'] = np.nan

        real = real_probs.idxmax(axis=1)
        pred = pred_probs.idxmax(axis=1)
        failed = real != pred
        pred_probs['failed'] = failed
        pred_probs['pred'] = pred
        pred_probs['truth'] = real

        labels, groupxs = ut.group_indices(real.values)
        for label, groupx in zip(labels, groupxs):
            print('label = %r' % (label,))
            real_probs_ = real_probs.iloc[groupx]
            pred_probs_ = pred_probs.iloc[groupx]
            diff = real_probs_ - pred_probs_
            hardness = diff[label]
            pred_probs['hardness'].loc[hardness.index] = hardness
        pred_probs = pred_probs.sort_values('hardness')[::-1]
        data = infr, pred_probs
        infr.verifiers = None
        cacher.save(data)
    infr, pred_probs = data

    # TODO: add an inspect pair to infr
    print('[graph_widget] show_selected')
    import wbia.guitool as gt
    from wbia.viz import viz_graph2

    app = gt.ensure_qapp()[0]  # NOQA
    ibs = infr.ibs

    import pandas as pd

    ut.qtensure()
    infr.params['inference.enabled'] = False
    infr.reset_feedback('staging', apply=True)

    # from wbia.guitool.__PYQT__ import QtCore
    # Qt = QtCore.Qt

    # Move absolutely sure edges down so they arn't re-reviewed
    edge_to_conf = infr.get_edge_attrs('confidence', pred_probs.index)
    pred_probs = pred_probs.assign(
        conf=pd.DataFrame.from_dict(edge_to_conf, orient='index')
    )

    easiness = 1 - pred_probs['hardness']
    sureness = np.nan_to_num(pred_probs['conf'].map(ibs.const.CONFIDENCE.CODE_TO_INT))

    pred_probs = pred_probs[sureness <= 2]
    easiness = 1 - pred_probs['hardness']
    sureness = np.nan_to_num(pred_probs['conf'].map(ibs.const.CONFIDENCE.CODE_TO_INT))

    # Order by least sure first, and then least easy
    priorities = list(zip(sureness, easiness))
    sortx = ut.argsort(priorities)
    pred_probs = pred_probs.iloc[sortx]
    # group2 = group2.sort_values('hardness', ascending=False)

    pred_probs = pred_probs[pred_probs['failed']]

    # lambda x: 0 if x == 'absolutely_sure' else 1)

    def get_index_data(count):
        edge = pred_probs.index[count]
        info_text = str(pred_probs.loc[edge])
        return edge, info_text

    self = viz_graph2.AnnotPairDialog(
        infr=infr, get_index_data=get_index_data, total=len(pred_probs)
    )
    self.seek(0)
    self.show()
    self.activateWindow()
    self.raise_()

    if False:
        df = infr._feedback_df('staging')
        edge = (184, 227)
        df.loc[edge]
    gt.qtapp_loop(qwin=self, freq=10)

    # import wbia.plottool as pt
    # from wbia.viz import viz_chip
    # TODO: Next step is to hook up infr and let AnnotPairDialog set feedback
    # then we just need to iterate through results

    # for count in ut.InteractiveIter(list(range(0, len(pred_probs)))):
    #     edge = pred_probs.index[count]
    #     info_text = str(pred_probs.loc[edge])
    #     self.set_edge(edge, info_text=info_text)

    #     # # self = viz_graph2.AnnotPairDialog(ibs=ibs, aid1=aid1, aid2=aid2,
    #     # #                                   info_text=info_text)
    #     # self.show()
    #     # fig.show()
    #     # fig.canvas.draw()

    # from wbia.gui import inspect_gui
    # self = tuner = inspect_gui.make_vsone_tuner(ibs, (aid1, aid2))  # NOQA
    # tuner.show()

    # ut.check_debug_import_times()
    # import utool
    # utool.embed()

    # if False:
    #     fnum = pt.ensure_fnum(10)
    #     for count in ut.InteractiveIter(list(range(0, len(pred_probs)))):
    #         fig = pt.figure(fnum=fnum)  # NOQA
    #         edge = pred_probs.index[count]
    #         info = pred_probs.loc[edge]
    #         viz_chip.show_many_chips(infr.ibs, edge, fnum=fnum)
    #         pt.set_title(str(info))
    #         # fig.show()
    #         fig.canvas.draw()

    # print('%d/%d failed' % (failed.sum(), len(failed)))
    # real_probs[failed] - match_probs[failed]
    # real - match_probs
    # # .idxmax(axis=1)
    # pred = match_probs.idxmax(axis=1)
    # real, pred = real
    # (pred != real).sum()
    # res = pblm.task_combo_res['match_state']['RF']['learn(sum,glob,4)']
    # infr.review_dummy_edges(method=2)
    # infr.relabel_using_reviews()
    # infr.apply_review_inference()


def debug_expanded_aids(ibs, expanded_aids_list, verbose=1):
    import warnings

    warnings.simplefilter('ignore', RuntimeWarning)
    # print('len(expanded_aids_list) = %r' % (len(expanded_aids_list),))
    cfgargs = dict(
        per_vp=False,
        per_multiple=False,
        combo_dists=False,
        per_name=True,
        per_enc=True,
        use_hist=False,
        combo_enc_info=False,
    )

    for qaids, daids in expanded_aids_list:
        stats = ibs.get_annotconfig_stats(qaids, daids, **cfgargs)
        hashids = (stats['qaid_stats']['qhashid'], stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        if verbose > 1:
            print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def encounter_stuff(ibs, aids):
    from wbia.init import main_helpers

    main_helpers.monkeypatch_encounters(ibs, aids, days=50)
    annots = ibs.annots(aids)

    from wbia.init.filter_annots import encounter_crossval

    expanded_aids = encounter_crossval(ibs, annots.aids, qenc_per_name=1, denc_per_name=2)
    import warnings

    warnings.simplefilter('ignore', RuntimeWarning)
    # with warnings.catch_warnings():
    for qaids, daids in expanded_aids:
        stats = ibs.get_annotconfig_stats(
            qaids, daids, use_hist=False, combo_enc_info=False
        )
        hashids = (stats['qaid_stats']['qhashid'], stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        # print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))


def iccv_data(defaultdb=None):
    import wbia
    from wbia.init import main_helpers

    if defaultdb is None:
        defaultdb = 'PZ_MTEST'
    # defaultdb = 'PZ_Master1'
    # defaultdb = 'GZ_Master1'
    # defaultdb = 'PZ_PB_RF_TRAIN'
    ibs = wbia.opendb(defaultdb=defaultdb)
    # Specialized database params
    enc_kw = dict(minutes=30)
    filt_kw = dict(
        require_timestamp=True, require_gps=True, is_known=True, minqual='good'
    )
    if ibs.dbname == 'PZ_MTEST':
        enc_kw = dict(days=50)
        filt_kw = dict(require_timestamp=True, is_known=True)

    if ibs.dbname == 'PZ_Master1':
        filt_kw['min_pername'] = 3

    expt_aids = ibs.filter_annots_general(**filt_kw)
    print('len(expt_aids) = %r' % (len(expt_aids),))

    if ibs.dbname == 'PZ_Master1':
        mtest_aids = wbia.dbio.export_subset.find_overlap_annots(
            ibs, wbia.opendb('PZ_MTEST'), method='images'
        )
        pbtest_aids = wbia.dbio.export_subset.find_overlap_annots(
            ibs, wbia.opendb('PZ_PB_RF_TRAIN'), method='annots'
        )
        expt_aids.extend(mtest_aids)
        expt_aids.extend(pbtest_aids)
        expt_aids = sorted(set(expt_aids))
        print('Expanding dataset')
        print('len(expt_aids) = %r' % (len(expt_aids),))

    main_helpers.monkeypatch_encounters(ibs, aids=expt_aids, **enc_kw)

    annots = ibs.annots(expt_aids)
    names = list(annots.group_items(annots.nids).values())
    ut.shuffle(names, rng=321)
    train_aids = ut.flatten(names[0::2])
    test_aids = ut.flatten(names[1::2])

    print_cfg = dict(per_multiple=False, use_hist=False)
    ibs.print_annot_stats(expt_aids, prefix='EXPT_', **print_cfg)
    ibs.print_annot_stats(train_aids, prefix='TRAIN_', **print_cfg)
    ibs.print_annot_stats(test_aids, prefix='TEST_', **print_cfg)

    species_code = ibs.get_database_species(expt_aids)[0]
    if species_code == 'zebra_plains':
        species = 'Plains Zebras'
    if species_code == 'zebra_grevys':
        species = "Grévy's Zebras"
    # species = ibs.get_species_nice(
    #     ibs.get_species_rowids_from_text())[0]
    return ibs, expt_aids, train_aids, test_aids, species


def iccv_cmc(defaultdb=None):
    """
    Ranking Experiment

    CommandLine:
        python -m wbia.scripts.iccv iccv_cmc

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> defaultdb = ut.get_argval('--db', default='PZ_MTEST')
        >>> result = iccv_cmc(defaultdb)
        >>> print(result)
    """
    if defaultdb is None:
        defaultdb = 'PZ_Master1'
    ibs, expt_aids, train_aids, test_aids, species = iccv_data(defaultdb)
    max_per_query = 3
    phis = learn_termination(ibs, aids=expt_aids, max_per_query=max_per_query)
    phis = {str(k): v for k, v in phis.items() if k <= 3}
    expt_annots = ibs.annots(expt_aids)

    expt_cfgstr = ibs.get_annot_hashid_visual_uuid(expt_aids)

    info = {
        'phis': phis,
        'species': species,
        'dbname': ibs.dbname,
        'annot_uuids': expt_annots.uuids,
        'visual_uuids': expt_annots.visual_uuids,
    }
    suffix = 'k=%d,nAids=%r,nNids=%r' % (
        max_per_query,
        len(expt_annots),
        len(set(expt_annots.nids)),
    )
    import pathlib

    dbname = ibs.dbname
    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))
    timestamp = ut.timestamp()
    fname = '_'.join(['cmc', dbname, timestamp, suffix, expt_cfgstr])
    info_fname = fname + '.cPkl'
    info_fpath = fig_dpath.joinpath(info_fname)
    ut.save_cPkl(info_fpath, info)

    draw_cmcs(dbname)


def draw_cmcs(dbname):
    """
    rsync
    scp -r lev:code/wbia/cmc* ~/latex/crall-iccv-2017/figures
    rsync -r lev:code/wbia/cmc* ~/latex/crall-iccv-2017/figures
    rsync -r hyrule:cmc_expt* ~/latex/crall-iccv-2017/figures

    CommandLine:
        python -m wbia.scripts.iccv draw_cmcs --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
        >>> result = draw_cmcs(dbname)
        >>> print(result)
    """
    # DRAW RESULTS
    import vtool as vt
    import wbia.plottool as pt
    import pathlib

    if dbname is None:
        dbname = 'PZ_Master1'

    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))
    possible_paths = sorted(fig_dpath.glob('cmc_' + dbname + '_*.cPkl'))[::-1]
    info_fpath = possible_paths[-1]
    fig_fpath = info_fpath.splitext()[0] + '.png'

    info = ut.load_cPkl(info_fpath)
    phis = info['phis']
    species = info['species']

    pt.qtensure()
    tmprc = {
        'legend.fontsize': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'legend.facecolor': 'w',
        'font.family': 'DejaVu Sans',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    import matplotlib as mpl

    mpl.rcParams.update(tmprc)
    fnum = 12
    fnum = pt.ensure_fnum(fnum)
    fig = pt.figure(fnum=fnum)
    ax = pt.gca()
    ranks = 20
    xdata = np.arange(1, ranks + 1)
    for k, phi in sorted(phis.items()):
        ax.plot(xdata, np.cumsum(phi[:ranks]), label='annots per query: %s' % (k,))
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Rank CMC for %s' % (species,))
    ax.set_ylim(ax.get_ylim()[0], 1)
    ax.set_ylim(0.7, 1)
    ax.set_xlim(0.9, ranks)
    ax.set_xticks(xdata)
    ax.legend()
    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9, wspace=0.2, hspace=0.2)
    fig.set_size_inches([7.4375, 3.125])
    fig.savefig(str(fig_fpath))
    vt.clipwhite_ondisk(str(fig_fpath))

    ut.show_if_requested()


def iccv_roc(dbname):
    """
    Classifier Experiment

    CommandLine:
        python -m wbia.scripts.iccv iccv_roc --db PZ_MTEST --show
        python -m wbia.scripts.iccv iccv_roc --db PZ_Master1 --show
        python -m wbia.scripts.iccv iccv_roc --db GZ_Master1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
        >>> result = iccv_roc(dbname)
        >>> print(result)
    """
    import wbia.plottool as pt
    import vtool as vt
    from wbia.scripts.script_vsone import OneVsOneProblem

    ibs, expt_aids, train_aids, test_aids, species = iccv_data(dbname)
    pt.qtensure()
    clf_key = 'RF'
    data_key = 'learn(sum,glob,4)'
    task_keys = ['match_state']
    task_key = task_keys[0]
    pblm = OneVsOneProblem.from_aids(ibs, aids=expt_aids, verbose=1)
    pblm.default_clf_key = clf_key
    pblm.default_data_key = data_key
    pblm.load_features()
    pblm.load_samples()
    pblm.build_feature_subsets()

    pblm.evaluate_simple_scores(task_keys)
    feat_cfgstr = ut.hashstr_arr27(
        pblm.samples.X_dict['learn(all)'].columns.values, 'matchfeat'
    )
    cfg_prefix = pblm.samples.make_sample_hashid() + pblm.qreq_.get_cfgstr() + feat_cfgstr
    pblm.learn_evaluation_classifiers(['match_state'], ['RF'], [data_key], cfg_prefix)

    res = pblm.task_combo_res[task_key][clf_key][data_key]

    pblm.report_simple_scores(task_key)
    res.extended_clf_report()

    class_name = POSTV
    clf_labels = res.target_bin_df[class_name].values
    clf_probs = res.probs_df[class_name].values
    clf_conf = vt.ConfusionMetrics().fit(clf_probs, clf_labels)
    clf_fpr = clf_conf.fpr
    clf_tpr = clf_conf.tpr

    class_name = POSTV
    task = pblm.samples.subtasks['match_state']
    lnbnn_labels = task.indicator_df[class_name]
    lnbnn_scores = pblm.samples.simple_scores['score_lnbnn_1vM']
    lnbnn_conf = vt.ConfusionMetrics().fit(lnbnn_scores, lnbnn_labels)
    lnbnn_fpr = lnbnn_conf.fpr
    lnbnn_tpr = lnbnn_conf.tpr

    clf_bins = np.linspace(0, 1, 100)
    clf_pos_probs = clf_probs[clf_labels]
    clf_neg_probs = clf_probs[~clf_labels]
    clf_pos_freq, _ = np.histogram(clf_pos_probs, clf_bins)
    clf_neg_freq, _ = np.histogram(clf_neg_probs, clf_bins)
    clf_pos_freq = clf_pos_freq / clf_pos_freq.sum()
    clf_neg_freq = clf_neg_freq / clf_neg_freq.sum()

    lnbnn_bins = np.linspace(0, 10, 100)
    lnbnn_pos_probs = lnbnn_scores[lnbnn_labels].values
    lnbnn_neg_probs = lnbnn_scores[~lnbnn_labels].values
    lnbnn_pos_freq, _ = np.histogram(lnbnn_pos_probs, lnbnn_bins)
    lnbnn_neg_freq, _ = np.histogram(lnbnn_neg_probs, lnbnn_bins)
    lnbnn_pos_freq = lnbnn_pos_freq / lnbnn_pos_freq.sum()
    lnbnn_neg_freq = lnbnn_neg_freq / lnbnn_neg_freq.sum()

    # Dump the data required to recreate the ROC into a folder
    expt_annots = ibs.annots(expt_aids)
    info = {
        'species': species,
        'lnbnn_fpr': lnbnn_fpr,
        'lnbnn_tpr': lnbnn_tpr,
        'clf_fpr': clf_fpr,
        'clf_tpr': clf_tpr,
        'dbname': ibs.dbname,
        'annot_uuids': expt_annots.uuids,
        'visual_uuids': expt_annots.visual_uuids,
        'qreq_cfgstr': pblm.qreq_.get_cfgstr(),
        'lnbnn_bins': lnbnn_bins,
        'lnbnn_pos_freq': lnbnn_pos_freq,
        'lnbnn_neg_freq': lnbnn_neg_freq,
        'clf_bins': clf_bins,
        'clf_pos_freq': clf_pos_freq,
        'clf_neg_freq': clf_neg_freq,
        # 'pblm_hyperparams': getstate_todict_recursive(pblm.hyper_params),
        'pblm_hyperparams': pblm.hyper_params.getstate_todict_recursive(),
    }
    info['pblm_hyperparams']['pairwise_feats']['summary_ops'] = list(
        info['pblm_hyperparams']['pairwise_feats']['summary_ops']
    )
    suffix = 'nAids=%r,nNids=%r,nPairs=%r' % (
        len(expt_annots),
        len(set(expt_annots.nids)),
        len(pblm.samples),
    )
    hashid = ut.hashstr27(pblm.qreq_.get_cfgstr())

    dbname = ibs.dbname
    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))
    timestamp = ut.timestamp()
    fname = '_'.join(['roc', dbname, timestamp, suffix, hashid])
    info_fname = fname + '.cPkl'
    info_fpath = fig_dpath.joinpath(info_fname)
    ut.save_cPkl(str(info_fpath), info)

    draw_saved_roc(dbname)


def draw_saved_roc(dbname):
    """
    rsync
    scp -r lev:code/wbia/roc* ~/latex/crall-iccv-2017/figures
    rsync -r hyrule:roc* ~/latex/crall-iccv-2017/figures

    rsync lev:code/wbia/roc* ~/latex/crall-iccv-2017/figures
    rsync lev:code/wbia/cmc* ~/latex/crall-iccv-2017/figures

    rsync lev:latex/crall-iccv-2017/figures/roc* ~/latex/crall-iccv-2017/figures

    CommandLine:
        python -m wbia.scripts.iccv draw_saved_roc --db PZ_Master1 --show
        python -m wbia.scripts.iccv draw_saved_roc --db GZ_Master1 --show
        python -m wbia.scripts.iccv draw_saved_roc --db PZ_MTEST --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
        >>> result = draw_saved_roc(dbname)
        >>> print(result)

    """
    # Draw the ROC in another process for quick iterations to appearance
    # DRAW RESULTS
    import wbia.plottool as pt
    import sklearn.metrics
    import vtool as vt
    import matplotlib as mpl

    pt.qtensure()

    if dbname is None:
        dbname = 'PZ_Master1'

    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))

    possible_paths = sorted(fig_dpath.glob('roc_' + dbname + '_*.cPkl'))[::-1]
    info_fpath = possible_paths[0]
    fig_fpath = info_fpath.parent.joinpath(info_fpath.stem + '.png')

    info = ut.load_data(str(info_fpath))
    lnbnn_fpr = info['lnbnn_fpr']
    lnbnn_tpr = info['lnbnn_tpr']
    clf_fpr = info['clf_fpr']
    clf_tpr = info['clf_tpr']
    species = info['species']

    clf_auc = sklearn.metrics.auc(clf_fpr, clf_tpr)
    lnbnn_auc = sklearn.metrics.auc(lnbnn_fpr, lnbnn_tpr)

    tmprc = {
        'legend.fontsize': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'legend.facecolor': 'w',
        'font.family': 'DejaVu Sans',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    mpl.rcParams.update(tmprc)
    # with pt.plt.rc_context(tmprc):
    fnum = 11
    fnum = pt.ensure_fnum(fnum)
    fig = pt.figure(fnum=fnum)
    pt.plot(clf_fpr, clf_tpr, label='pairwise AUC=%.3f' % (clf_auc,))
    pt.plot(lnbnn_fpr, lnbnn_tpr, label='LNBNN AUC=%.3f' % (lnbnn_auc,))
    ax = pt.gca()
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    species = species.lower()
    if species[0] == 'g':
        species = 'G' + species[1:]
    ax.set_title('Positive match ROC for %s' % (species,))
    ax.legend()
    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9, wspace=0.2, hspace=0.2)
    fig.set_size_inches([7.4375, 3.125])
    fig.savefig(str(fig_fpath))
    clip_fpath = vt.clipwhite_ondisk(str(fig_fpath))
    print('clip_fpath = %r' % (clip_fpath,))

    clf_bins = info['clf_bins']
    clf_pos_freq = info['clf_pos_freq']
    clf_neg_freq = info['clf_neg_freq']
    lnbnn_bins = info['lnbnn_bins']
    lnbnn_pos_freq = info['lnbnn_pos_freq']
    lnbnn_neg_freq = info['lnbnn_neg_freq']

    fnum = 12
    pnum = (1, 1, 1)
    true_color = pt.TRUE_BLUE  # pt.TRUE_GREEN
    false_color = pt.FALSE_RED
    score_colors = (false_color, true_color)
    lnbnn_fig = pt.multi_plot(
        clf_bins,
        (clf_neg_freq, clf_pos_freq),
        label_list=('negative', 'positive'),
        fnum=fnum,
        color_list=score_colors,
        pnum=pnum,
        kind='bar',
        width=np.diff(clf_bins)[0],
        alpha=0.7,
        stacked=True,
        edgecolor='none',
        rcParams=tmprc,
        xlabel='positive probability',
        ylabel='frequency',
        title='pairwise probability separation',
    )
    lnbnn_fig_fpath = ut.augpath(str(fig_fpath), prefix='clf_scoresep_')
    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9, wspace=0.2, hspace=0.2)
    lnbnn_fig.set_size_inches([7.4375, 3.125])
    lnbnn_fig.savefig(str(lnbnn_fig_fpath), dpi=256)
    vt.clipwhite_ondisk(str(lnbnn_fig_fpath))

    fnum = 13
    score_colors = (false_color, true_color)
    clf_fig = pt.multi_plot(
        lnbnn_bins,
        (lnbnn_neg_freq, lnbnn_pos_freq),
        label_list=('negative', 'positive'),
        fnum=fnum,
        color_list=score_colors,
        pnum=pnum,
        kind='bar',
        width=np.diff(lnbnn_bins)[0],
        alpha=0.7,
        stacked=True,
        edgecolor='none',
        rcParams=tmprc,
        xlabel='LNBNN score',
        ylabel='frequency',
        title='LNBNN score separation',
    )
    pt.adjust_subplots(top=0.8, bottom=0.2, left=0.12, right=0.9, wspace=0.2, hspace=0.2)
    clf_fig.set_size_inches([7.4375, 3.125])
    clf_fig_fpath = ut.augpath(str(fig_fpath), prefix='lnbnn_scoresep_')
    clf_fig.savefig(str(clf_fig_fpath), dpi=256)
    vt.clipwhite_ondisk(str(clf_fig_fpath))

    ut.show_if_requested()


# @profile
def end_to_end():
    r"""
    CommandLine:
        python -m wbia.scripts.iccv end_to_end --show --db PZ_MTEST
        python -m wbia.scripts.iccv end_to_end --show --db PZ_Master1
        python -m wbia.scripts.iccv end_to_end --show --db GZ_Master1
        python -m wbia.scripts.iccv end_to_end --db PZ_Master1
        python -m wbia.scripts.iccv end_to_end --db GZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> result = end_to_end()
        >>> print(result)
    """
    import wbia
    from wbia.scripts.script_vsone import OneVsOneProblem

    ibs, expt_aids, train_aids, test_aids, species = iccv_data()
    # -----------
    # TRAINING

    clf_key = 'RF'
    # data_key = 'learn(sum,glob)'
    data_key = 'learn(sum,glob,4)'
    task_keys = ['match_state', 'photobomb_state']
    pblm = OneVsOneProblem.from_aids(ibs=ibs, aids=train_aids, verbose=1)
    pblm.default_clf_key = clf_key
    pblm.default_data_key = data_key
    pblm.load_features()
    pblm.load_samples()
    pblm.build_feature_subsets()

    train_cfgstr = ibs.get_annot_hashid_visual_uuid(train_aids)

    # Figure out what the thresholds should be
    thresh_cacher = ut.Cacher(
        'clf_thresh',
        cfgstr=train_cfgstr + data_key + 'v3',
        appname=pblm.appname,
        enabled=True,
    )
    fpr_thresholds = thresh_cacher.tryload()
    if fpr_thresholds is None:
        feat_cfgstr = ut.hashstr_arr27(
            pblm.samples.X_dict['learn(all)'].columns.values, 'matchfeat'
        )
        cfg_prefix = (
            pblm.samples.make_sample_hashid() + pblm.qreq_.get_cfgstr() + feat_cfgstr
        )
        pblm.learn_evaluation_classifiers(['match_state'], ['RF'], [data_key], cfg_prefix)
        task_key = 'match_state'
        clf_key = 'RF'
        res = pblm.task_combo_res[task_key][clf_key][data_key]
        res.extended_clf_report()
        fpr_thresholds = {
            fpr: res.get_pos_threshes('fpr', value=fpr) for fpr in [0, 0.001, 0.005, 0.01]
        }
        for fpr, thresh_df in fpr_thresholds.items():
            # disable notcomp thresholds due to training issues
            thresh_df[INCMP] = max(1.0, thresh_df[INCMP])
            # ensure thresholds are over .5
            thresh_df[POSTV] = max(0.51, thresh_df[POSTV])
            thresh_df[NEGTV] = max(0.51, thresh_df[NEGTV])
        print('fpr_thresholds = %s' % (ut.repr3(fpr_thresholds),))
        thresh_cacher.save(fpr_thresholds)

    clf_cacher = ut.Cacher(
        'deploy_clf_v2_', appname=pblm.appname, cfgstr=train_cfgstr + data_key
    )
    pblm.deploy_task_clfs = clf_cacher.tryload()
    if pblm.deploy_task_clfs is None:
        pblm.learn_deploy_classifiers(task_keys, data_key=data_key, clf_key=clf_key)
        clf_cacher.save(pblm.deploy_task_clfs)

    if False:
        pblm.print_featinfo(data_key)
        pblm.samples.print_info()
        task_keys = list(pblm.samples.subtasks.keys())
        match_clf = pblm.deploy_task_clfs['match_state']
        pb_clf = pblm.deploy_task_clfs['photobomb_state']
        # Inspect feature importances if you want
        pblm.report_classifier_importance2(match_clf, data_key=data_key)
        pblm.report_classifier_importance2(pb_clf, data_key=data_key)

    # aids = train_aids
    maxphi = 3
    phi_cacher = ut.Cacher(
        'term_phis', appname=pblm.appname, cfgstr=train_cfgstr + str(maxphi)
    )
    phis = phi_cacher.tryload()
    if phis is None:
        phis = learn_termination(ibs, train_aids, maxphi)
        phi_cacher.save(phis)
    # show_phis(phis)

    # ------------
    # TESTING
    complete_thresh = 0.95
    # graph_loops = np.inf
    ranking_loops = 2
    graph_loops = 2
    # np.inf
    expt_dials = [
        {
            'name': 'Ranking',
            'method': 'ranking',
            'k_redun': np.inf,
            'cand_kw': dict(pblm=None),
            'priority_metric': 'normscore',
            'oracle_accuracy': 1.0,
            'complete_thresh': 1.0,
            'match_state_thresh': None,
            'max_loops': ranking_loops,
        },
        {
            'name': 'Ranking+Classifier,fpr=0',
            'method': 'ranking',
            'k_redun': np.inf,
            'cand_kw': dict(pblm=pblm),
            'priority_metric': 'priority',
            'oracle_accuracy': 1.0,
            'complete_thresh': 1.0,
            'match_state_thresh': fpr_thresholds[0],
            'max_loops': ranking_loops,
        },
        {
            'name': 'Graph,K=2,fpr=.001',
            'method': 'graph',
            'k_redun': 2,
            'cand_kw': dict(pblm=pblm),
            'priority_metric': 'priority',
            'oracle_accuracy': 1.0,
            'complete_thresh': complete_thresh,
            'match_state_thresh': fpr_thresholds[0.001],
            'max_loops': graph_loops,
        },
    ]
    oracle_accuracy = 0.98
    expt_dials += [
        {
            'name': 'Ranking+Error',
            'method': 'ranking',
            'k_redun': np.inf,
            'cand_kw': dict(pblm=None),
            'priority_metric': 'normscore',
            'oracle_accuracy': oracle_accuracy,
            'complete_thresh': 1.0,
            'match_state_thresh': None,
            'max_loops': ranking_loops,
        },
        {
            'name': 'Ranking+Classifier,fpr=0+Error',
            'method': 'ranking',
            'k_redun': np.inf,
            'cand_kw': dict(pblm=pblm),
            'priority_metric': 'priority',
            'oracle_accuracy': oracle_accuracy,
            'complete_thresh': 1.0,
            'match_state_thresh': fpr_thresholds[0],
            'max_loops': ranking_loops,
        },
        {
            'name': 'Graph,K=2,fpr=.001+Error',
            'method': 'graph',
            'k_redun': 2,
            'cand_kw': dict(pblm=pblm),
            'priority_metric': 'priority',
            'oracle_accuracy': oracle_accuracy,
            'complete_thresh': complete_thresh,
            'match_state_thresh': fpr_thresholds[0.001],
            'max_loops': graph_loops,
        },
    ]

    # colors = pt.distinct_colors(len(expt_dials))_

    # hack, reduce size
    # test_annots = ibs.annots(test_aids)
    # test_aids = ut.flatten(test_annots.group(test_annots.nids)[1][0:200])

    dials = expt_dials[1]

    verbose = 0
    # verbose = 1
    expt_metrics = {}
    # idx_list = list(range(0, 3))
    # idx_list = list(range(0, 6))
    # idx_list = [3, 4, 5]
    idx_list = [3, 4, 5]
    # idx_list = [5]

    for idx in idx_list:
        dials = expt_dials[idx]
        infr = wbia.AnnotInference(
            ibs=ibs, aids=test_aids, autoinit=True, verbose=verbose
        )
        new_dials = dict(
            phis=phis,
            oracle_accuracy=dials['oracle_accuracy'],
            k_redun=dials['k_redun'],
            enable_autoreview=dials['priority_metric'] == 'priority',
            enable_inference=dials['method'] == 'graph',
            classifiers=dials['cand_kw']['pblm'],
            complete_thresh=dials['complete_thresh'],
            match_state_thresh=dials['match_state_thresh'],
            name=dials['name'],
        )
        infr.init_simulation(**new_dials)
        print('new_dials = %s' % (ut.repr4(new_dials),))
        infr.reset(state='empty')
        infr.main_loop(max_loops=dials['max_loops'])
        metrics_df = pd.DataFrame.from_dict(infr.metrics_list)
        # infr.non_complete_pcc_pairs().__next__()
        # Remove non-transferable attributes
        infr.ibs = None
        infr.qreq_ = None
        infr.vsmany_qreq_ = None
        infr.verifiers = None
        expt_metrics[idx] = (dials, metrics_df, infr)

    ut.cprint('SAVE ETE', 'green')

    expt_cfgstr = ibs.get_annot_hashid_visual_uuid(expt_aids)
    import pathlib

    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))
    # fig_dpath = pathlib.Path('~/latex/crall-iccv-2017/figures').expanduser()
    expt_dname = '_'.join(['ete_expt', ibs.dbname, ut.timestamp()])
    expt_dpath = fig_dpath.joinpath(expt_dname)
    ut.ensuredir(str(expt_dpath))
    # expt_dpath.mkdir(exist_ok=True)
    from six.moves import cPickle

    for count, (dials, metrics_df, infr) in expt_metrics.items():
        ete_info = {
            'expt_count': count,
            'dbname': ibs.dbname,
            'infr': infr,
            'species': species,
            'test_auuids': ibs.annots(test_aids).uuids,
            'train_auuids': ibs.annots(train_aids).uuids,
            'dials': ut.delete_keys(dials.copy(), ['cand_kw']),
            'metrics_df': metrics_df.to_dict('list'),
            # 'qreq_cfgstr': pblm.qreq_.get_cfgstr(),
            # 'pblm_hyperparams': getstate_todict_recursive(pblm.hyper_params),
            # 'pblm_hyperparams': pblm.hyper_params.getstate_todict_recursive(),
        }
        ete_info_fname = 'ETE_info_' + dials['name'] + expt_cfgstr
        ete_info_fpath = expt_dpath.joinpath(ete_info_fname + '.cPkl')
        try:
            ut.save_cPkl(str(ete_info_fpath), ete_info)
        except Exception:
            for k, v in vars(ete_info['infr']).items():
                print('k = %r' % (k,))
                cPickle.dumps(v)
            raise

    dbname = ibs.dbname
    draw_ete(dbname)


def draw_ete(dbname):
    """
    rsync -r hyrule:latex/crall-iccv-2017/figures/ete_expt* ~/latex/crall-iccv-2017/figures
    rsync -r    lev:latex/crall-iccv-2017/figures/ete_expt* ~/latex/crall-iccv-2017/figures

    CommandLine:
        python -m wbia.scripts.iccv draw_ete --db PZ_Master1
        python -m wbia.scripts.iccv draw_ete --db GZ_Master1 --show
        python -m wbia.scripts.iccv draw_ete --db PZ_MTEST --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.iccv import *  # NOQA
        >>> dbname = 'GZ_Master1'
        >>> dbname = 'PZ_MTEST'
        >>> dbname = 'PZ_Master1'
        >>> dbname = ut.get_argval('--db', default=dbname)
        >>> draw_ete(dbname)
    """
    ut.cprint('Draw ETE', 'green')
    # DRAW RESULTS
    import wbia.plottool as pt

    pt.qtensure()

    import pathlib

    # fig_dpath = pathlib.Path('~/latex/crall-iccv-2017/figures').expanduser()
    fig_dpath = pathlib.Path(ut.truepath('~/latex/crall-iccv-2017/figures'))
    possible_expts = sorted(fig_dpath.glob('ete_expt_' + dbname + '*'))[::-1]
    for dpath in possible_expts:
        if not any(p.is_file() for p in dpath.rglob('*')):
            dpath.rmdir()
    possible_expts = sorted(fig_dpath.glob('ete_expt_' + dbname + '*'))[::-1]
    assert len(possible_expts) > 0, 'no ete expts found'
    expt_dpath = possible_expts[0]

    infos_ = []
    for fpath in expt_dpath.glob('*.cPkl'):
        x = ut.load_cPkl(str(fpath))
        infr = x['infr']
        print('DIALS')
        print(ut.repr4(x['dials']))
        if getattr(infr, 'vsmany_qreq_', None) is not None:
            infr.vsmany_qreq_ = None
            ut.save_cPkl(str(fpath), x)
        infos_.append(x)
        if False:
            infr.show(groupby='orig_name_label')
        if 0:
            # from wbia.algo.graph.nx_utils import edges_inside, edges_cross

            groups_nid = ut.ddict(list)
            groups_type = ut.ddict(list)
            for edge, error in list(infr.measure_error_edges()):
                print('error = %s' % (ut.repr2(error),))
                data = infr.graph.get_edge_data(*edge)
                print('user_id = %r' % (data['user_id'],))
                aid1, aid2 = edge
                nid1 = infr.graph.nodes[aid1]['orig_name_label']
                nid2 = infr.graph.nodes[aid2]['orig_name_label']
                cc1 = infr.nid_to_gt_cc[nid1]
                cc2 = infr.nid_to_gt_cc[nid2]
                print('nid1, nid2 = %r, %r' % (nid1, nid2))
                print('len1, len2 = %r, %r' % (len(cc1), len(cc2)))
                list(ut.nx_edges_between(infr.graph, cc1, cc2))
                groups_type[(error['real'], error['pred'])].append(edge)
                groups_nid[(nid1, nid2)].append((edge, error))
            print('error breakdown: %r' % ut.map_dict_vals(len, groups_type))

            for (real, pred), edges in groups_type.items():
                for edge in edges:
                    nid1 = infr.graph.nodes[aid1]['orig_name_label']
                    nid2 = infr.graph.nodes[aid2]['orig_name_label']

    infos = {
        info['dials']['name']: info for info in infos_ if 'Error' in info['dials']['name']
    }

    # xmax = 2000
    alias = {
        'Ranking+Error': 'ranking',
        'Ranking+Classifier,fpr=0+Error': 'ranking+clf',
        'Graph,K=2,fpr=.001+Error': 'graph',
    }

    import matplotlib as mpl

    tmprc = {
        'legend.fontsize': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'legend.facecolor': 'w',
        'font.family': 'DejaVu Sans',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }

    keys = sorted(infos.keys())

    mpl.rcParams.update(tmprc)
    fnum = 13
    fnum = pt.ensure_fnum(fnum)
    # colors = pt.distinct_colors(4)
    fig = pt.figure(fnum=fnum, pnum=(1, 2, 1))
    ax = pt.gca()
    for key in keys:
        ete_info = infos[key]
        # count = ete_info['expt_count']
        metrics = ete_info['metrics_df']
        dials = ete_info['dials']
        # metrics.keys()
        pt.plt.plot(
            metrics['n_manual'],
            metrics['n_merge_remain'],
            '-',
            label=alias.get(dials['name'], dials['name']),
            # color=colors[count],
        )
    # ax.set_xlim(0,xmax)
    ax.set_xlabel('# manual reviews')
    ax.set_ylabel('# merges remaining')
    ax.legend()

    pt.figure(fnum=fnum, pnum=(1, 2, 2))
    ax = pt.gca()
    for key in keys:
        ete_info = infos[key]
        infr = ete_info['infr']
        species = ete_info['species']

        species = species.lower()
        if species[0] == 'g':
            species = 'G' + species[1:]

        # count = ete_info['expt_count']
        metrics = ete_info['metrics_df']
        dials = ete_info['dials']
        # metrics.keys()
        pt.plt.plot(
            metrics['n_manual'],
            metrics['n_errors'],
            '-',
            label=alias.get(dials['name'], dials['name']),
            # color=colors[count],
        )
    # ax.set_xlim(0,xmax)
    # ax.set_ylim(0,40)
    ax.set_xlabel('# manual reviews')
    ax.set_ylabel('# of errors')
    # ax.legend()
    pt.set_figtitle(
        'End-to-end accuracy and error for %s' % (species,),
        fontweight='normal',
        fontfamily='DejaVu Sans',
    )

    pt.adjust_subplots(
        top=0.85, bottom=0.2, left=0.1, right=0.98, wspace=0.2, hspace=0.35
    )
    fig.set_size_inches([1.3 * 7.4375, 3.125])
    # fig_fpath = splitext(info_fpath)[0] + '.png'
    plot_fpath = pathlib.Path(str(expt_dpath) + '.png')
    plot_fpath = plot_fpath.parent.joinpath('fig_' + plot_fpath.stem + plot_fpath.suffix)
    # .joinpath('ete_%s.png' % (dbname))
    fig.savefig(str(plot_fpath))
    import vtool as vt

    clip_fpath = vt.clipwhite_ondisk(str(plot_fpath))
    print('plot_fpath = %r' % (plot_fpath,))
    print('clip_fpath = %r' % (clip_fpath,))

    # infr.show(fnum=3, groupby='name_label')
    ut.show_if_requested()


def show_phis(phis):
    import wbia.plottool as pt

    pt.qtensure()
    ranks = 20
    ydatas = [phi.cumsum()[0:ranks] for phi in phis.values()]
    pt.multi_plot(
        xdata=np.arange(1, ranks + 1),
        ydata_list=ydatas,
        num_xticks=ranks,
        label_list=['annots per query: %d' % d for d in phis.keys()],
        title='Learned Termination CDF',
    )


def collect_termination_data(ibs, aids, max_per_query=3):
    from wbia.init.filter_annots import encounter_crossval

    n_splits = 3
    crossval_splits = []
    avail_confusors = []
    import random

    rng = random.Random(0)
    for n_query_per_name in range(1, max_per_query + 1):
        reshaped_splits, nid_to_confusors = encounter_crossval(
            ibs,
            aids,
            qenc_per_name=n_query_per_name,
            denc_per_name=1,
            rng=rng,
            n_splits=n_splits,
            early=True,
        )

        avail_confusors.append(nid_to_confusors)

        expanded_aids = []
        for qpart, dpart in reshaped_splits:
            # For each encounter choose just 1 annotation
            qaids = [rng.choice(enc) for enc in ut.flatten(qpart)]
            daids = [rng.choice(enc) for enc in ut.flatten(dpart)]
            assert len(set(qaids).intersection(daids)) == 0
            expanded_aids.append((sorted(qaids), sorted(daids)))

        crossval_splits.append((n_query_per_name, expanded_aids))

    n_daid_spread = [len(expanded_aids[0][1]) for _, expanded_aids in crossval_splits]

    # Check to see if we can pad confusors to make the database size equal
    max_size = max(n_daid_spread)
    afford = min(map(len, avail_confusors)) - (max(n_daid_spread) - min(n_daid_spread))
    max_size += afford

    crossval_splits2 = []
    _iter = zip(crossval_splits, avail_confusors)
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
    return crossval_splits2


def learn_termination(ibs, aids, max_per_query=3):
    """
    Example:
    """
    ut.cprint('Learning termination phi', 'white')
    from wbia.expt import test_result

    crossval_splits2 = collect_termination_data(ibs, aids, max_per_query)

    pipe_cfg = {
        # 'resize_dim': 'area',
        # 'dim_size': 450,
    }

    phis = {}
    for n_query_per_name, expanded_aids in crossval_splits2:
        accumulators = []
        for qaids, daids in expanded_aids:
            num_datab_pccs = len(np.unique(ibs.annots(daids).nids))
            # num_query_pccs = len(np.unique(ibs.annots(qaids).nids))
            qreq_ = ibs.new_query_request(qaids, daids, verbose=False, cfgdict=pipe_cfg)
            cm_list = qreq_.execute()
            testres = test_result.TestResult.from_cms(cm_list, qreq_)
            freqs, bins = testres.get_rank_histograms(
                key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs)
            )
            freq = freqs[0]
            accumulators.append(freq)
        size = max(map(len, accumulators))
        accum = np.zeros(size)
        for freq in accumulators:
            accum[0 : len(freq)] += freq

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
    # from wbia.init import main_helpers
    # dbname = 'GZ_Master1'
    # a = 'timectrl'
    # t = 'baseline'
    # ibs, testres = main_helpers.testdata_expts(dbname, a=a, t=t)
    import wbia

    # from wbia.init.filter_annots import annot_crossval
    from wbia.init.filter_annots import encounter_crossval
    from wbia.init import main_helpers
    from wbia.expt import test_result
    import wbia.plottool as pt

    pt.qtensure()

    ibs = wbia.opendb('GZ_Master1')
    # ibs = wbia.opendb('PZ_MTEST')
    # ibs = wbia.opendb('PZ_PB_RF_TRAIN')

    aids = ibs.filter_annots_general(
        require_timestamp=True, require_gps=True, is_known=True
    )
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
            ibs,
            aids,
            qenc_per_name=n_query_per_name,
            denc_per_name=1,
            rng=rng,
            n_splits=n_splits,
            early=True,
        )

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

    afford = min(map(len, avail_confusors)) - (max(n_daid_spread) - min(n_daid_spread))
    max_size += afford

    crossval_splits2 = []

    for (n_query_per_name, expanded_aids), nid_to_confusors in zip(
        crossval_splits, avail_confusors
    ):
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
            qreq_ = ibs.new_query_request(qaids, daids, verbose=False, cfgdict=pipe_cfg)

            cm_list = qreq_.execute()
            testres = test_result.TestResult.from_cms(cm_list, qreq_)
            # nranks = testres.get_infoprop_list(key='qnx2_gt_name_rank')[0]
            # aranks = testres.get_infoprop_list(key='qx2_gt_annot_rank')[0]
            # freqs, bins = testres.get_rank_histograms(
            #     key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs))
            freqs, bins = testres.get_rank_histograms(
                key='qnx2_gt_name_rank', bins=np.arange(num_datab_pccs)
            )
            freq = freqs[0]
            accumulators.append(freq)
        size = max(map(len, accumulators))
        accum = np.zeros(size)
        for freq in accumulators:
            accum[0 : len(freq)] += freq

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

    # cmc = phi3.cumsum()
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
    # import wbia.plottool as pt
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.iccv
        python -m wbia.scripts.iccv --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
