import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__)


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
        stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=False,
                                          combo_enc_info=False)
        hashids = (stats['qaid_stats']['qhashid'],
                   stats['daid_stats']['dhashid'])
        print('hashids = %r' % (hashids,))
        # print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))

def iccv_data(defaultdb='PZ_MTEST'):
    import ibeis
    from ibeis.init import main_helpers
    # defaultdb = 'PZ_Master1'
    # defaultdb = 'GZ_Master1'
    # defaultdb = 'PZ_PB_RF_TRAIN'
    # defaultdb = 'PZ_MTEST'
    ibs = ibeis.opendb(defaultdb=defaultdb)
    # Specialized database params
    enc_kw = dict(minutes=30)
    filt_kw = dict(require_timestamp=True, require_gps=True, is_known=True,
                   minqual='good')
    if ibs.dbname == 'PZ_MTEST':
        enc_kw = dict(days=50)
        filt_kw = dict(require_timestamp=True, is_known=True)

    if ibs.dbname == 'PZ_Master1':
        filt_kw['min_pername'] = 3

    expt_aids = ibs.filter_annots_general(**filt_kw)
    print('len(expt_aids) = %r' % (len(expt_aids),))

    if ibs.dbname == 'PZ_Master1':
        mtest_aids = ibeis.dbio.export_subset.find_overlap_annots(
            ibs, ibeis.opendb('PZ_MTEST'), method='images')
        pbtest_aids = ibeis.dbio.export_subset.find_overlap_annots(
            ibs, ibeis.opendb('PZ_PB_RF_TRAIN'), method='annots')
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

    # Hack
    # num_names = 75
    # train_aids = ut.flatten(names[0::2][0:num_names])
    # test_aids = ut.flatten(names[1::2][0:num_names])

    print_cfg = dict(per_multiple=False, use_hist=False)
    ibs.print_annot_stats(expt_aids, prefix='EXPT_', **print_cfg)
    ibs.print_annot_stats(train_aids, prefix='TRAIN_', **print_cfg)
    ibs.print_annot_stats(test_aids, prefix='TEST_', **print_cfg)

    species_code = ibs.get_database_species(expt_aids)[0]
    if species_code == 'zebra_plains':
        species = 'Plains Zebras'
    if species_code == 'zebra_grevys':
        species = 'Grévy\'s Zebras'
    # species = ibs.get_species_nice(
    #     ibs.get_species_rowids_from_text())[0]
    return ibs, expt_aids, train_aids, test_aids, species

def iccv_cmc():
    """ Ranking Experiment """
    ibs, expt_aids, train_aids, test_aids, species = iccv_data()
    # quick test
    # phi_aids = ibs.filter_annots_general(min_pername=2, **filt_kw)
    # phi_annots = ibs.annots(phi_aids)
    # phi_names = list(phi_annots.group_items(phi_annots.nids).values())
    # cmc_aids = ut.flatten(phi_names[:75])
    # ibs.print_annot_stats(cmc_aids, prefix='CMC_', **print_cfg)
    import plottool as pt
    pt.qtensure()
    phis = learn_termination(ibs, aids=expt_aids, max_per_query=4)
    # phis_ = phis
    phis = {str(k): v for k, v in phis.items() if k < 5}

    info = {
        'phis': phis,
        'species': species,
        'dbname': ibs.dbname,
        'annot_uuids': expt_annots.uuids,
        'visual_uuids': expt_annots.visual_uuids,
        # 'qreq_cfgstr': pblm.qreq_.get_cfgstr(),
        # 'pblm_hyperparams': getstate_todict_recursive(pblm.hyper_params),
        # 'pblm_hyperparams': pblm.hyper_params.getstate_todict_recursive(),
    }
    expt_annots = ibs.annots(expt_aids)
    suffix = 'nAids=%r,nNids=%r' % (len(expt_annots),
                                    len(set(expt_annots.nids)))
    hashid = ut.hashstr27(pblm.qreq_.get_cfgstr())
    fname = '_'.join(['cmc', species_code, suffix, hashid])
    fig_fname = fname  + '.png'
    info_fname = fname + '.json'
    dpath = 'cmc_expt_' + ut.timestamp()
    ut.ensuredir(dpath)
    info_fpath = join(dpath, info_fname)
    ut.save_json(info_fpath, info)

    def draw_cmcs():
        """
        rsync
        scp -r lev:code/ibeis/cmc* ~/latex/crall-iccv-2017/figures
        rsync -r lev:code/ibeis/cmc* ~/latex/crall-iccv-2017/figures
        rsync -r hyrule:cmc_expt* ~/latex/crall-iccv-2017/figures
        """
        # DRAW RESULTS
        import vtool as vt
        import plottool as pt
        from os.path import splitext
        pt.qtensure()
        fig_dpath = ut.truepath('~/latex/crall-iccv-2017/figures')
        dpath = sorted(ut.glob(fig_dpath, 'cmc_expt_*'))[-1]
        info_fpath = ut.glob(dpath, '*.json')[0]
        fig_fpath = splitext(info_fpath)[0] + '.png'

        info = ut.load_json(info_fpath)
        phis = info['phis']
        species = info['species']

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
            ax.plot(xdata, np.cumsum(phi[:ranks]),
                    label='annots per query: %s' % (k,))
        ax.set_xlabel('Rank')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Rank CMC for %s' % (species,))
        ax.set_ylim(ax.get_ylim()[0], 1)
        ax.set_ylim(.7, 1)
        ax.set_xlim(.9, ranks)
        ax.set_xticks(xdata)
        ax.legend()
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9, wspace=.2,
                           hspace=.2)
        fig.set_size_inches([7.4375,  3.125])
        fig.savefig(fig_fpath)
        vt.clipwhite_ondisk(fig_fpath)


def iccv_roc():
    """ Classifier Experiment """
    ibs, expt_aids, train_aids, test_aids, species = iccv_data()
    import plottool as pt
    pt.qtensure()
    from ibeis.scripts.script_vsone import OneVsOneProblem
    clf_key = 'RF'
    data_key = 'learn(sum,glob)'
    task_keys = ['match_state']
    pblm = OneVsOneProblem.from_aids(ibs, aids=expt_aids, verbose=1)
    pblm.load_features()
    pblm.load_samples()
    pblm.build_feature_subsets()

    pblm.evaluate_simple_scores(task_keys)
    feat_cfgstr = ut.hashstr_arr27(
        pblm.samples.X_dict['learn(all)'].columns.values, 'matchfeat')
    cfg_prefix = (pblm.samples.make_sample_hashid() +
                  pblm.qreq_.get_cfgstr() + feat_cfgstr)
    pblm.learn_evaluation_classifiers(['match_state'], ['RF'], [data_key],
                                      cfg_prefix)
    task_key = 'match_state'
    clf_key = 'RF'

    res = pblm.task_combo_res[task_key][clf_key][data_key]

    pblm.report_simple_scores(task_key)
    res.extended_clf_report()

    import vtool as vt
    class_name = 'match'
    labels = res.target_bin_df[class_name].values
    probs = res.probs_df[class_name].values
    clf_conf = vt.ConfusionMetrics.from_scores_and_labels(probs, labels)
    clf_fpr = clf_conf.fpr
    clf_tpr = clf_conf.tpr

    fnum = 1
    class_name = 'match'
    labels = pblm.samples.subtasks['match_state'].indicator_df[class_name]
    scores = pblm.samples.simple_scores['score_lnbnn_1vM']
    lnbnn_conf = vt.ConfusionMetrics.from_scores_and_labels(scores, labels)
    lnbnn_fpr = lnbnn_conf.fpr
    lnbnn_tpr = lnbnn_conf.tpr

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
        # 'pblm_hyperparams': getstate_todict_recursive(pblm.hyper_params),
        'pblm_hyperparams': pblm.hyper_params.getstate_todict_recursive(),
    }
    info['pblm_hyperparams']['pairwise_feats']['summary_ops'] = list(info['pblm_hyperparams']['pairwise_feats']['summary_ops'])
    suffix = 'nAids=%r,nNids=%r,nPairs=%r' % (len(expt_annots),
                                              len(set(expt_annots.nids)),
                                              len(pblm.samples))
    hashid = ut.hashstr27(pblm.qreq_.get_cfgstr())
    fname = '_'.join(['roc', species_code, suffix, hashid])
    fig_fname = fname  + '.png'
    info_fname = fname + '.json'
    dpath = 'roc_expt_' + ut.timestamp()
    ut.ensuredir(dpath)
    info_fpath = join(dpath, info_fname)
    ut.save_json(info_fpath, info)

    # Draw the ROC in another process for quick iterations to appearance
    def draw_saved_roc():
        """
        rsync
        scp -r lev:code/ibeis/roc* ~/latex/crall-iccv-2017/figures
        rsync -r hyrule:roc* ~/latex/crall-iccv-2017/figures

        rsync lev:code/ibeis/roc* ~/latex/crall-iccv-2017/figures
        rsync lev:code/ibeis/cmc* ~/latex/crall-iccv-2017/figures
        """
        # DRAW RESULTS
        import plottool as pt
        pt.qtensure()
        dpath = sorted(ut.glob('.', 'roc_expt_*'))[-1]
        from os.path import splitext
        info_fpath = ut.glob(dpath, '*.json')[0]
        fig_fpath = splitext(info_fpath)[0] + '.png'

        info = ut.load_json(info_fpath)
        lnbnn_fpr = info['lnbnn_fpr']
        lnbnn_tpr = info['lnbnn_tpr']
        clf_fpr = info['clf_fpr']
        clf_tpr = info['clf_tpr']
        species = info['species']


        import sklearn.metrics
        clf_auc = sklearn.metrics.auc(clf_fpr, clf_tpr)
        lnbnn_auc = sklearn.metrics.auc(lnbnn_fpr, lnbnn_tpr)
        import matplotlib as mpl

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
        if True:
            fnum = 11
            fnum = pt.ensure_fnum(fnum)
            fig = pt.figure(fnum=fnum)
            pt.plot(clf_fpr, clf_tpr, label='Pairwise AUC=%.3f' % (clf_auc,))
            pt.plot(lnbnn_fpr, lnbnn_tpr, label='LNBNN AUC=%.3f' % (lnbnn_auc,))
            ax = pt.gca()
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Positive Match ROC for %s' % (species,))
            ax.legend()
            pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9, wspace=.2,
                               hspace=.2)
            fig.set_size_inches([7.4375,  3.125])
            savekw = {
                # 'bbox_inches': pt.extract_axes_extents(fig)[0]
            }
            fig.savefig(fig_fpath, **savekw)
        vt.clipwhite_ondisk(fig_fpath)
    pass


# @profile
def end_to_end():
    r"""
    CommandLine:
        python -m ibeis.scripts.iccv end_to_end --show --db PZ_MTEST
        python -m ibeis.scripts.iccv end_to_end --show --db PZ_Master1
        python -m ibeis.scripts.iccv end_to_end --show --db GZ_Master1

    Example:
        >>> from ibeis.scripts.iccv import *  # NOQA
        >>> result = end_to_end()
        >>> print(result)
    """
    import ibeis
    from ibeis.scripts.script_vsone import OneVsOneProblem

    ibs, expt_aids, train_aids, test_aids, species = iccv_data()
    # -----------
    # TRAINING

    clf_key = 'RF'
    data_key = 'learn(sum,glob)'
    task_keys = ['match_state', 'photobomb_state']
    pblm = OneVsOneProblem.from_aids(ibs=ibs, aids=train_aids, verbose=1)
    pblm.load_features()
    pblm.load_samples()
    pblm.build_feature_subsets()

    train_cfgstr = ibs.get_annot_hashid_visual_uuid(train_aids)

    # Figure out what the thresholds should be
    thresh_cacher = ut.Cacher('clf_thresh', cfgstr=train_cfgstr + 'v3',
                              enabled=False)
    fpr_thresholds = thresh_cacher.tryload()
    if fpr_thresholds is None:
        feat_cfgstr = ut.hashstr_arr27(
            pblm.samples.X_dict['learn(all)'].columns.values, 'matchfeat')
        cfg_prefix = (pblm.samples.make_sample_hashid() +
                      pblm.qreq_.get_cfgstr() + feat_cfgstr)
        pblm.learn_evaluation_classifiers(['match_state'], ['RF'], [data_key],
                                          cfg_prefix)
        task_key = 'match_state'
        clf_key = 'RF'
        res = pblm.task_combo_res[task_key][clf_key][data_key]
        res.extended_clf_report()
        fpr_thresholds = {
            fpr: res.get_pos_threshes('fpr', value=fpr)
            for fpr in [0, .001, .005, .01]
        }
        for fpr, thresh_df in fpr_thresholds.items():
            pass
            # fpr_thresholds
        # import pprint
        print('fpr_thresholds = %s' % (ut.repr3(fpr_thresholds),))
        thresh_cacher.save(fpr_thresholds)

    clf_cacher = ut.Cacher('deploy_clf_', cfgstr=train_cfgstr)
    pblm.deploy_task_clfs = clf_cacher.tryload()
    if pblm.deploy_task_clfs is None:
        pblm.learn_deploy_classifiers(task_keys, data_key=data_key,
                                      clf_key=clf_key)
        clf_cacher.save(pblm.deploy_task_clfs)
    # pblm.samples.print_info()
    # task_keys = list(pblm.samples.subtasks.keys())
    # match_clf = pblm.deploy_task_clfs['match_state']
    # pb_clf = pblm.deploy_task_clfs['photobomb_state']

    # Inspect feature importances if you want
    # pblm.report_classifier_importance2(match_clf, data_key=data_key)
    # pblm.report_classifier_importance2(pb_clf, data_key=data_key)

    # aids = train_aids
    maxphi = 4
    phi_cacher = ut.Cacher('term_phis', cfgstr=train_cfgstr + str(maxphi))
    phis = phi_cacher.tryload()
    if phis is None:
        phis = learn_termination(ibs, train_aids, maxphi)
        phi_cacher.save(phis)
    # show_phis(phis)

    # ------------
    # TESTING
    import plottool as pt  # NOQA
    # test_aids = ut.flatten(names[1::2])
    # test_aids = ut.flatten(names[1::2][0:4])
    # test_aids = ut.flatten(names[1::2][::2])
    complete_thresh = .95
    ranking_loops = 2
    graph_loops = np.inf
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
            'match_state_thresh': fpr_thresholds[.001],
            'max_loops': graph_loops,
        },
    ]
    oracle_accuracy = .98
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
            'match_state_thresh': fpr_thresholds[.001],
            'max_loops': graph_loops,
        },
    ]

    colors = pt.distinct_colors(len(expt_dials))

    dials = expt_dials[1]
    import pandas as pd

    # verbose = 0
    verbose = 1
    expt_metrics = {}
    # idx_list = list(range(0, 3))
    # idx_list = list(range(0, 6))
    idx_list = [5]

    for idx in idx_list:
        dials = expt_dials[idx]
        infr = ibeis.AnnotInference(ibs=ibs, aids=test_aids, autoinit=True,
                                    verbose=verbose)
        infr.init_test_mode()
        infr.init_termination_criteria(phis)
        run_expt(infr, dials=dials)
        metrics_df = pd.DataFrame.from_dict(infr.metrics_list)
        # infr.non_complete_pcc_pairs().__next__()
        expt_metrics[idx] = (dials, metrics_df, infr)

    ut.cprint('SAVE ETE', 'green')

    expt_cfgstr = ibs.get_annot_hashid_visual_uuid(expt_aids)
    import pathlib
    fig_dpath = pathlib.Path('~/latex/crall-iccv-2017/figures').expanduser()
    expt_dname = '_'.join(['ete_expt', ibs.dbname, ut.timestamp()])
    expt_dpath = fig_dpath.joinpath(expt_dname)
    expt_dpath.mkdir(exist_ok=True)
    from six.moves import cPickle as pickle

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
        ut.save_cPkl(str(ete_info_fpath), ete_info)

    dbname = ibs.dbname
    draw_ete(dbname)

def draw_ete(dbname):
    """
    rsync -r hyrule:ete* ~/latex/crall-iccv-2017/figures
    rsync -r lev:code/ibeis/ete* ~/latex/crall-iccv-2017/figures
    rsync -r lev:ete* ~/latex/crall-iccv-2017/figures

    Args:
        dbname (?):

    CommandLine:
        python -m ibeis.scripts.iccv draw_ete --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.iccv import *  # NOQA
        >>> dbname = 'PZ_MTEST'
        >>> dbname = ut.get_argval('--db', default=dbname)
        >>> draw_ete(dbname)
    """
    ut.cprint('Draw ETE', 'green')
    # DRAW RESULTS
    import plottool as pt
    pt.qtensure()

    import pathlib
    fig_dpath = pathlib.Path('~/latex/crall-iccv-2017/figures').expanduser()
    possible_expts = sorted(fig_dpath.glob('ete_expt_' + dbname + '*'))[::-1]
    for dpath in possible_expts:
        if not any(p.is_file() for p in dpath.rglob('*')):
            dpath.rmdir()
    possible_expts = sorted(fig_dpath.glob('ete_expt_' + dbname + '*'))[::-1]
    assert len(possible_expts) > 0, 'no ete expts found'
    expt_dpath = possible_expts[0]
    infos_ = [
        ut.load_cPkl(str(fpath))
        for fpath in expt_dpath.glob('*.cPkl')
    ]

    infos = {info['dials']['name']: info
             for info in infos_ if 'Error' in info['dials']['name']}

    # xmax = 2000

    import matplotlib as mpl
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
    fnum = 13
    fnum = pt.ensure_fnum(fnum)
    # colors = pt.distinct_colors(4)
    fig = pt.figure(fnum=fnum, pnum=(1, 2, 1))
    ax = pt.gca()
    for ete_info in infos.values():
        count = ete_info['expt_count']
        metrics = ete_info['metrics_df']
        dials = ete_info['dials']
        # metrics.keys()
        pt.plt.plot(
            metrics['n_manual'],
            metrics['n_merge_remain'], '-',
            label=dials['name'],
            # color=colors[count],
        )
    # ax.set_xlim(0,xmax)
    ax.set_xlabel('# manual reviews')
    ax.set_ylabel('# merges remaining')
    # ax.legend()

    pt.figure(fnum=fnum, pnum=(1, 2, 2))
    ax = pt.gca()
    for ete_info in infos.values():
        infr = ete_info['infr']

        count = ete_info['expt_count']
        metrics = ete_info['metrics_df']
        dials = ete_info['dials']
        # metrics.keys()
        pt.plt.plot(
            metrics['n_manual'],
            metrics['n_errors'], '-',
            label=dials['name'],
            # color=colors[count],
        )
    # ax.set_xlim(0,xmax)
    # ax.set_ylim(0,40)
    ax.set_xlabel('# manual reviews')
    ax.set_ylabel('# of errors')
    ax.legend()
    pt.set_figtitle('Identification performance', fontweight='normal',
                    fontfamily='DejaVu Sans')

    pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9, wspace=.25,
                       hspace=.2)
    fig.set_size_inches([7.4375,  3.125])
    # fig_fpath = splitext(info_fpath)[0] + '.png'
    plot_fpath = expt_dpath.joinpath('ete_%s.png' % (dbname))
    fig.savefig(str(plot_fpath))
    import vtool as vt
    vt.clipwhite_ondisk(str(plot_fpath))

    # infr.show(fnum=3, groupby='name_label')
    ut.show_if_requested()


@profile
def run_expt(infr, dials):
    import pandas as pd
    import itertools as it

    ut.cprint('RUNING TEST', 'yellow')
    print('dials = %s' % (ut.repr4(dials),))
    k_redun = dials['k_redun']
    oracle_accuracy = dials['oracle_accuracy']
    cand_kw = dials['cand_kw']
    priority_metric = dials['priority_metric']
    infr.method = dials['method']
    infr.PRIORITY_METRIC = priority_metric
    autoreview_enabled = infr.PRIORITY_METRIC == 'priority'
    infr.queue_params['pos_redundancy'] = k_redun
    infr.queue_params['neg_redundancy'] = k_redun
    infr.queue_params['complete_thresh'] = dials['complete_thresh']
    infr.task_thresh = {
        'photobomb_state': pd.Series({
            'pb': .5,
            'notpb': .9,
        }),
        'match_state': pd.Series(dials['match_state_thresh'])
    }

    # aid_to_gt_nid = ut.dzip(infr.aids, infr.orig_name_labels)
    seed = sum(map(ord, dials['name']))

    rng = ut.ensure_rng(seed, impl='python')
    infr.reset(state='empty')
    infr.remove_feedback(apply=True)

    infr.init_refresh_criteria()

    for count in it.count(0):
        if count >= dials['max_loops']:
            # Just do a maximum of some number of runs for now
            ut.cprint('Early stop', 'blue')
            break
        ut.cprint('Outer loop iter %d ' % (count,), 'blue')
        infr.refresh_candidate_edges(**cand_kw)

        if not len(infr.queue):
            ut.cprint('Queue is empty. Terminate.', 'blue')
            break

        ut.cprint('Start inner loop', 'blue')
        while True:
            if len(infr.queue) == 0:
                ut.cprint('No more edges, need refresh', 'blue')
                break
            edge, priority = infr.pop()
            if priority <= 1:
                if infr.refresh.check():
                    ut.cprint('Refresh criteria flags refresh', 'blue')
                    break
            else:
                ut.cprint('IN RECOVERY MODE priority=%r' % (priority,), 'red')

            flag = False
            if autoreview_enabled:
                flag, review = infr.try_auto_review(edge, priority)
            if not flag:
                if infr.method == 'graph':
                    flag, review = infr.try_implicit_review(edge, priority)
                if not flag:
                    error, review = infr.oracle_review(edge, oracle_accuracy,
                                                       rng)
            infr.add_feedback(edge=edge, apply=True, rectify=False, **review)

    if infr.method == 'graph':
        # Enforce that a user checks any PCC that was auto-reviewed
        # but was unable to achieve k-positive-consistency
        for pcc in list(infr.non_pos_redundant_pccs()):
            subgraph = infr.graph.subgraph(pcc)
            for u, v, data in subgraph.edges(data=True):
                edge = infr.e_(u, v)
                if data.get('user_id', '').startswith('auto'):
                    error, review = infr.oracle_review(edge, oracle_accuracy, rng)
                    infr.add_feedback(edge=edge, apply=True, rectify=False,
                                      **review)
        # Check for inconsistency recovery
        while len(infr.queue):
            edge, priority = infr.pop()
            if priority <= 1:
                break
            error, review = infr.oracle_review(edge, oracle_accuracy, rng)
            infr.add_feedback(edge=edge, apply=True, rectify=False, **review)

    if infr.method == 'graph':
        assert len(list(infr.inconsistent_components())) == 0

    true_groups = list(map(set, infr.nid_to_gt_cc.values()))
    pred_groups = list(infr.positive_connected_compoments())
    from ibeis.algo.hots import sim_graph_iden
    comparisons = sim_graph_iden.compare_groups(true_groups, pred_groups)
    pred_merges = comparisons['pred_merges']
    # print(pred_merges)
    ut.cprint('FINISHE ETE')
    # print(ut.repr4(comparisons, nl=4))


def show_phis(phis):
    import plottool as pt
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


def learn_termination(ibs, aids, max_per_query=4):
    """
    Example:
    """
    ut.cprint('Learning termination phi', 'white')
    from ibeis.init.filter_annots import encounter_crossval
    from ibeis.expt import test_result
    pipe_cfg = {
        # 'resize_dim': 'area',
        # 'dim_size': 450,
    }

    n_splits = 3
    crossval_splits = []
    avail_confusors = []
    import random
    rng = random.Random(0)
    for n_query_per_name in range(1, max_per_query + 1):
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

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.iccv
        python -m ibeis.scripts.iccv --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
