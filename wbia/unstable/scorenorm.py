# -*- coding: utf-8 -*-
"""
GOALS:
    1) vsmany
       * works resaonable for very few and very many
       * stars with small k and then k becomes a percent or log percent
       * distinctiveness from different location

    2) 1-vs-1
       * uses distinctiveness and foreground when available
       * start with ratio test and ransac

    3) First N decision are interactive until we learn a good threshold

    4) Always show numbers between 0 and 1 spatial verification is based on
    single best exemplar

       x - build encoder
       x - test encoder
       x - monotonicity (both nondecreasing and strictly increasing)
       x - cache encoder
       x - cache maitainance (deleters and listers)
       o - Incemental learning
       o - Spceies sensitivity

    * Add ability for user to relearn encoder from labeled database.

TODO:
    * One class SVM http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import re
from wbia import dtool
import numpy as np
import utool as ut
import vtool as vt
import six  # NOQA
from functools import partial
from os.path import join
from wbia import constants as const
from wbia.init import sysres

print, rrr, profile = ut.inject2(__name__)


def compare_score_pdfs(testres):
    """
    CommandLine:
        python -m wbia.expt.test_result --exec-compare_score_pdfs --show --present
        python -m wbia.expt.test_result --exec-compare_score_pdfs --show --present --nocache
        python -m wbia.expt.test_result --exec-compare_score_pdfs --show --present -a timectrl:qindex=0:50

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.test_result import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> defaultdb = 'PZ_Master1'
        >>> ibs, testres = wbia.testdata_expts(
        >>>     defaultdb=defaultdb, a=['timectrl'], t=['best'])
        >>> testres.compare_score_pdfs()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    # from wbia.init import main_helpers
    import utool as ut

    # import wbia.plottool as pt
    ut.ensureqt()

    testres.draw_annot_scoresep(f='fail=False')
    # pt.adjust_subplots(bottom=.25, top=.8)
    encoder = testres.draw_feat_scoresep(f='fail=False', disttype=None)
    # pt.adjust_subplots(bottom=.25, top=.8)
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn'])
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['ratio'])
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['L2_sift'])
    encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn', 'fg'])
    # pt.adjust_subplots(bottom=.25, top=.8)

    # ibs, testres = main_helpers.testdata_expts(
    #    defaultdb=defaultdb, a=['timectrl'], t=['best:lnbnn_on=False,ratio_thresh=1.0'])
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['ratio'])
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn'])
    # encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['L2_sift'])
    # TODO:
    return encoder


def draw_annot_scoresep(testres, f=None):
    from wbia.expt import experiment_drawing

    experiment_drawing.draw_annot_scoresep(testres.ibs, testres, f=f)


def draw_feat_scoresep(testres, f=None, disttype=None):
    r"""
    SeeAlso:
        wbia.algo.hots.scorenorm.train_featscore_normalizer

    CommandLine:
        python -m wbia --tf TestResult.draw_feat_scoresep --show
        python -m wbia --tf TestResult.draw_feat_scoresep --show -t default:sv_on=[True,False]
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --disttype=L2_sift,fg
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --disttype=L2_sift
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True --namemode=True
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True --namemode=False

        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=L2_sift
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=L2_sift -t best:SV=False

        utprof.py -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1
        utprof.py -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --fsvx=1:2
        utprof.py -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --fsvx=0:1

        utprof.py -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 -t best:lnbnn_on=False,bar_l2_on=True  --fsvx=0:1

        # We want to query the oxford annots taged query
        # and we want the database to contain
        # K correct images per query, as well as the distractors

        python -m wbia --tf TestResult.draw_feat_scoresep  --show --db Oxford -a default:qhas_any=\(query,\),dpername=1,exclude_reference=True,minqual=ok
        python -m wbia --tf TestResult.draw_feat_scoresep  --show --db Oxford -a default:qhas_any=\(query,\),dpername=1,exclude_reference=True,minqual=good

        python -m wbia --tf get_annotcfg_list  --db PZ_Master1 -a timectrl --acfginfo --verbtd  --veryverbtd --nocache-aid

        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=ratio

    Example:
        >>> # SCRIPT
        >>> from wbia.expt.test_result import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> disttype = ut.get_argval('--disttype', type_=list, default=None)
        >>> ibs, testres = main_helpers.testdata_expts(
        >>>     defaultdb='PZ_MTEST', a=['timectrl'], t=['best'])
        >>> f = ut.get_argval(('--filt', '-f'), type_=list, default=[''])
        >>> testres.draw_feat_scoresep(f=f)
        >>> ut.show_if_requested()
    """
    print('[testres] draw_feat_scoresep')
    import wbia.plottool as pt

    def load_feat_scores(qreq_, qaids):
        import wbia  # NOQA
        from os.path import dirname, join  # NOQA

        # HACKY CACHE
        cfgstr = qreq_.get_cfgstr(with_input=True)
        cache_dir = join(dirname(dirname(wbia.__file__)), 'TMP_FEATSCORE_CACHE')
        namemode = ut.get_argval('--namemode', default=True)
        fsvx = ut.get_argval(
            '--fsvx', type_='fuzzy_subset', default=slice(None, None, None)
        )
        threshx = ut.get_argval('--threshx', type_=int, default=None)
        thresh = ut.get_argval('--thresh', type_=float, default=0.9)
        num = ut.get_argval('--num', type_=int, default=1)
        cfg_components = [cfgstr, disttype, namemode, fsvx, threshx, thresh, f, num]
        cache_cfgstr = ','.join(ut.lmap(six.text_type, cfg_components))
        cache_hashid = ut.hashstr27(cache_cfgstr + '_v1')
        cache_name = 'get_cfgx_feat_scores_' + cache_hashid

        @ut.cached_func(cache_name, cache_dir=cache_dir, key_argx=[], use_cache=True)
        def get_cfgx_feat_scores(qreq_, qaids):
            from wbia.algo.hots import scorenorm

            cm_list = qreq_.execute(qaids)
            # print('Done loading cached chipmatches')
            tup = scorenorm.get_training_featscores(
                qreq_, cm_list, disttype, namemode, fsvx, threshx, thresh, num=num
            )
            # print(ut.depth_profile(tup))
            tp_scores, tn_scores, scorecfg = tup
            return tp_scores, tn_scores, scorecfg

        tp_scores, tn_scores, scorecfg = get_cfgx_feat_scores(qreq_, qaids)
        return tp_scores, tn_scores, scorecfg

    valid_case_pos = testres.case_sample2(filt_cfg=f, return_mask=False)
    cfgx2_valid_qxs = ut.group_items(valid_case_pos.T[0], valid_case_pos.T[1])
    test_qaids = testres.get_test_qaids()
    cfgx2_valid_qaids = ut.map_dict_vals(ut.partial(ut.take, test_qaids), cfgx2_valid_qxs)

    join_acfgs = True

    # TODO: option to average over pipeline configurations
    if join_acfgs:
        groupxs = testres.get_cfgx_groupxs()
    else:
        groupxs = list(zip(range(len(testres.cfgx2_qreq_))))
    grouped_qreqs = ut.apply_grouping(testres.cfgx2_qreq_, groupxs)

    grouped_scores = []
    for cfgxs, qreq_group in zip(groupxs, grouped_qreqs):
        # testres.print_pcfg_info()
        score_group = []
        for cfgx, qreq_ in zip(cfgxs, testres.cfgx2_qreq_):
            print('Loading cached chipmatches')
            qaids = cfgx2_valid_qaids[cfgx]
            tp_scores, tn_scores, scorecfg = load_feat_scores(qreq_, qaids)
            score_group.append((tp_scores, tn_scores, scorecfg))
        grouped_scores.append(score_group)

    cfgx2_shortlbl = testres.get_short_cfglbls(join_acfgs=join_acfgs)
    for score_group, lbl in zip(grouped_scores, cfgx2_shortlbl):
        tp_scores = np.hstack(ut.take_column(score_group, 0))
        tn_scores = np.hstack(ut.take_column(score_group, 1))
        scorecfg = '+++'.join(ut.unique(ut.take_column(score_group, 2)))
        score_group
        # TODO: learn this score normalizer as a model
        # encoder = vt.ScoreNormalizer(adjust=4, monotonize=False)
        encoder = vt.ScoreNormalizer(adjust=2, monotonize=True)
        encoder.fit_partitioned(tp_scores, tn_scores, verbose=False)
        figtitle = 'Feature Scores: %s, %s' % (scorecfg, lbl)
        fnum = None

        vizkw = {}
        sephack = ut.get_argflag('--sephack')
        if not sephack:
            vizkw['target_tpr'] = 0.95
            vizkw['score_range'] = (0, 1.0)

        encoder.visualize(
            figtitle=figtitle,
            fnum=fnum,
            with_scores=False,
            # with_prebayes=True,
            with_prebayes=False,
            with_roc=True,
            with_postbayes=False,
            # with_postbayes=True,
            **vizkw,
        )
        icon = testres.ibs.get_database_icon()
        if icon is not None:
            pt.overlay_icon(icon, coords=(1, 0), bbox_alignment=(1, 0))

        if ut.get_argflag('--contextadjust'):
            pt.adjust_subplots(left=0.1, bottom=0.25, wspace=0.2, hspace=0.2)
            pt.adjust_subplots(use_argv=True)
    return encoder


def get_global_species_scorenorm_cachedir(ibs, species_text, ensure=True):
    """
    Args:
        species_text (str):
        ensure       (bool):

    Returns:
        str: species_cachedir

    CommandLine:
        python -m wbia.control.IBEISControl --test-get_global_species_scorenorm_cachedir

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> species_text = wbia.const.TEST_SPECIES.ZEB_GREVY
        >>> ensure = True
        >>> species_cachedir = ibs.get_global_species_scorenorm_cachedir(species_text, ensure)
        >>> resourcedir = ibs.get_wbia_resource_dir()
        >>> result = ut.relpath_unix(species_cachedir, resourcedir)
        >>> print(result)
        scorenorm/zebra_grevys
    """
    scorenorm_cachedir = join(ibs.get_wbia_resource_dir(), const.PATH_NAMES.scorenormdir)
    species_cachedir = join(scorenorm_cachedir, species_text)
    if ensure:
        ut.ensurepath(scorenorm_cachedir)
        ut.ensuredir(species_cachedir)
    return species_cachedir


def get_local_species_scorenorm_cachedir(ibs, species_text, ensure=True):
    """
    """
    scorenorm_cachedir = join(ibs.get_cachedir(), const.PATH_NAMES.scorenormdir)
    species_cachedir = join(scorenorm_cachedir, species_text)
    if ensure:
        ut.ensuredir(scorenorm_cachedir)
        ut.ensuredir(species_cachedir)
    return species_cachedir


def get_global_distinctiveness_modeldir(ibs, ensure=True):
    """
    Returns:
        global_distinctdir (str): ibs internal directory
    """
    global_distinctdir = sysres.get_global_distinctiveness_modeldir(ensure=ensure)
    return global_distinctdir


def get_local_distinctiveness_modeldir(ibs):
    """
    Returns:
        distinctdir (str): ibs internal directory
    """
    return ibs.distinctdir


class NormFeatScoreConfig(dtool.Config):
    _alias = 'nfscfg'
    _param_info_list = [
        ut.ParamInfo('disttype', None),
        ut.ParamInfo('namemode', True),
        ut.ParamInfo('fsvx', None, type_='fuzzy_subset', hideif=None),
        ut.ParamInfo('threshx', None, hideif=None),
        ut.ParamInfo('thresh', 0.9, hideif=lambda cfg: cfg['threshx'] is None),
        ut.ParamInfo('num', 5),
        # ut.ParamInfo('top_percent', None, hideif=None),
        ut.ParamInfo('top_percent', 0.5, hideif=None),
    ]


def compare_featscores():
    r"""
    CommandLine:

        wbia --tf compare_featscores  --db PZ_MTEST \
            --nfscfg :disttype=[L2_sift,lnbnn],top_percent=[None,.5,.1] -a timectrl \
            -p default:K=[1,2],normalizer_rule=name \
            --save featscore{db}.png --figsize=13,20 --diskshow

        wbia --tf compare_featscores  --db PZ_MTEST \
            --nfscfg :disttype=[L2_sift,normdist,lnbnn],top_percent=[None,.5] -a timectrl \
            -p default:K=[1],normalizer_rule=name,sv_on=[True,False] \
            --save featscore{db}.png --figsize=13,10 --diskshow

        wbia --tf compare_featscores --nfscfg :disttype=[L2_sift,normdist,lnbnn] \
            -a timectrl -p default:K=1,normalizer_rule=name --db PZ_Master1 \
            --save featscore{db}.png  --figsize=13,13 --diskshow

        wbia --tf compare_featscores --nfscfg :disttype=[L2_sift,normdist,lnbnn] \
            -a timectrl -p default:K=1,normalizer_rule=name --db GZ_ALL \
            --save featscore{db}.png  --figsize=13,13 --diskshow

        wbia --tf compare_featscores  --db GIRM_Master1 \
            --nfscfg ':disttype=fg,L2_sift,normdist,lnbnn' \
            -a timectrl -p default:K=1,normalizer_rule=name \
            --save featscore{db}.png  --figsize=13,13

        wbia --tf compare_featscores --nfscfg :disttype=[L2_sift,normdist,lnbnn] \
            -a timectrl -p default:K=[1,2,3],normalizer_rule=name,sv_on=False \
            --db PZ_Master1 --save featscore{db}.png  \
                --dpi=128 --figsize=15,20 --diskshow

        wbia --tf compare_featscores --show --nfscfg :disttype=[L2_sift,normdist] -a timectrl -p :K=1 --db PZ_MTEST
        wbia --tf compare_featscores --show --nfscfg :disttype=[L2_sift,normdist] -a timectrl -p :K=1 --db GZ_ALL
        wbia --tf compare_featscores --show --nfscfg :disttype=[L2_sift,normdist] -a timectrl -p :K=1 --db PZ_Master1
        wbia --tf compare_featscores --show --nfscfg :disttype=[L2_sift,normdist] -a timectrl -p :K=1 --db GIRM_Master1

        wbia --tf compare_featscores  --db PZ_MTEST \
            --nfscfg :disttype=[L2_sift,normdist,lnbnn],top_percent=[None,.5,.2] -a timectrl \
            -p default:K=[1],normalizer_rule=name \
            --save featscore{db}.png --figsize=13,20 --diskshow

        wbia --tf compare_featscores  --db PZ_MTEST \
            --nfscfg :disttype=[L2_sift,normdist,lnbnn],top_percent=[None,.5,.2] -a timectrl \
            -p default:K=[1],normalizer_rule=name \
            --save featscore{db}.png --figsize=13,20 --diskshow

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> result = compare_featscores()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt
    import wbia

    nfs_cfg_list = NormFeatScoreConfig.from_argv_cfgs()
    learnkw = {}
    ibs, testres = wbia.testdata_expts(
        defaultdb='PZ_MTEST', a=['default'], p=['default:K=1']
    )
    print('nfs_cfg_list = ' + ut.repr3(nfs_cfg_list))

    encoder_list = []
    lbl_list = []

    varied_nfs_lbls = ut.get_varied_cfg_lbls(nfs_cfg_list)
    varied_qreq_lbls = ut.get_varied_cfg_lbls(testres.cfgdict_list)
    # varies_qreq_lbls

    # func = ut.cached_func(cache_dir='.')(learn_featscore_normalizer)
    for datakw, nlbl in zip(nfs_cfg_list, varied_nfs_lbls):
        for qreq_, qlbl in zip(testres.cfgx2_qreq_, varied_qreq_lbls):
            lbl = qlbl + ' ' + nlbl
            cfgstr = '_'.join([datakw.get_cfgstr(), qreq_.get_full_cfgstr()])
            try:
                encoder = vt.ScoreNormalizer()
                encoder.load(cfgstr=cfgstr)
            except IOError:
                print('datakw = %r' % (datakw,))
                encoder = learn_featscore_normalizer(qreq_, datakw, learnkw)
                encoder.save(cfgstr=cfgstr)
            encoder_list.append(encoder)
            lbl_list.append(lbl)

    fnum = 1
    # next_pnum = pt.make_pnum_nextgen(nRows=len(encoder_list), nCols=3)
    next_pnum = pt.make_pnum_nextgen(nRows=len(encoder_list) + 1, nCols=3, start=3)

    iconsize = 94
    if len(encoder_list) > 3:
        iconsize = 64

    icon = qreq_.ibs.get_database_icon(max_dsize=(None, iconsize), aid=qreq_.qaids[0])
    score_range = (0, 0.6)
    for encoder, lbl in zip(encoder_list, lbl_list):
        # encoder.visualize(figtitle=encoder.get_cfgstr(), with_prebayes=False, with_postbayes=False)
        encoder._plot_score_support_hist(
            fnum, pnum=next_pnum(), titlesuf='\n' + lbl, score_range=score_range
        )
        encoder._plot_prebayes(fnum, pnum=next_pnum())
        encoder._plot_roc(fnum, pnum=next_pnum())
        if icon is not None:
            pt.overlay_icon(icon, coords=(1, 0), bbox_alignment=(1, 0))

    nonvaried_lbl = ut.get_nonvaried_cfg_lbls(nfs_cfg_list)[0]
    figtitle = qreq_.__str__() + '\n' + nonvaried_lbl

    pt.set_figtitle(figtitle)
    pt.adjust_subplots(hspace=0.5, top=0.92, bottom=0.08, left=0.1, right=0.9)
    pt.update_figsize()
    pt.plt.tight_layout()
    # pt.adjust_subplots(top=.95)


def learn_annotscore_normalizer(qreq_, learnkw={}):
    """
    Takes the result of queries and trains a score encoder

    Args:
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters

    Returns:
        vtool.ScoreNormalizer: encoder

    CommandLine:
        python -m wbia --tf learn_annotscore_normalizer --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(
        >>>     defaultdb='PZ_MTEST', a=['default'], p=['default'])
        >>> encoder = learn_annotscore_normalizer(qreq_)
        >>> ut.quit_if_noshow()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    cm_list = qreq_.execute()
    tup = get_training_annotscores(qreq_, cm_list)
    tp_scores, tn_scores, good_tn_aidnid_pairs, good_tp_aidnid_pairs = tup
    part_attrs = {
        0: {'aid_pairs': good_tn_aidnid_pairs},
        1: {'aid_pairs': good_tp_aidnid_pairs},
    }
    scores, labels, attrs = vt.flatten_scores(tp_scores, tn_scores, part_attrs)
    _learnkw = {'monotonize': True}
    _learnkw.update(learnkw)
    # timestamp = ut.get_timestamp()
    encoder = vt.ScoreNormalizer(**_learnkw)
    encoder.fit(scores, labels, attrs=attrs)
    encoder.cfgstr = 'annotscore'
    return encoder


def load_featscore_normalizer(normer_cfgstr):
    r"""
    Args:
        normer_cfgstr (?):

    CommandLine:
        python -m wbia.algo.hots.scorenorm --exec-load_featscore_normalizer --show
        python -m wbia.algo.hots.scorenorm --exec-load_featscore_normalizer --show --cfgstr=featscore
        python -m wbia.algo.hots.scorenorm --exec-load_featscore_normalizer --show --cfgstr=lovb

    Example:
        >>> # SCRIPT
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> normer_cfgstr = ut.get_argval('--cfgstr', default='featscore')
        >>> encoder = load_featscore_normalizer(normer_cfgstr)
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    encoder = vt.ScoreNormalizer()
    # qreq_.lnbnn_normer.load(cfgstr=config2_.lnbnn_normer)
    encoder.fuzzyload(partial_cfgstr=normer_cfgstr)
    return encoder


def train_featscore_normalizer():
    r"""
    CommandLine:
        python -m wbia --tf train_featscore_normalizer --show

        # Write Encoder
        python -m wbia --tf train_featscore_normalizer --db PZ_MTEST -t best -a default --fsvx=0 --threshx=1 --show

        # Visualize encoder score adjustment
        python -m wbia --tf TestResult.draw_feat_scoresep --db PZ_MTEST -a timectrl -t best:lnbnn_normer=lnbnn_fg_featscore --show --nocache --nocache-hs

        # Compare ranking with encoder vs without
        python -m wbia --tf draw_rank_cmc --db PZ_MTEST -a timectrl -t best:lnbnn_normer=[None,wulu] --show
        python -m wbia --tf draw_rank_cmc --db PZ_MTEST -a default  -t best:lnbnn_normer=[None,wulu] --show

        # Compare in ipynb
        python -m wbia --tf autogen_ipynb --ipynb --db PZ_MTEST -a default -t best:lnbnn_normer=[None,lnbnn_fg_0.9__featscore]

        # Big Test
        python -m wbia --tf draw_rank_cmc --db PZ_Master1 -a timectrl -t best:lnbnn_normer=[None,lovb],lnbnn_norm_thresh=.5 --show
        python -m wbia --tf draw_rank_cmc --db PZ_Master1 -a timectrl -t best:lnbnn_normer=[None,jypz],lnbnn_norm_thresh=.1 --show
        python -m wbia --tf draw_rank_cmc --db PZ_Master1 -a timectrl -t best:lnbnn_normer=[None,jypz],lnbnn_norm_thresh=0 --show


        # Big Train
        python -m wbia --tf learn_featscore_normalizer --db PZ_Master1 -a timectrl -t best:K=1 --fsvx=0 --threshx=1 --show
        python -m wbia --tf train_featscore_normalizer --db PZ_Master1 -a timectrl:has_none=photobomb -t best:K=1 --fsvx=0 --threshx=1 --show --ainfo
        python -m wbia --tf train_featscore_normalizer --db PZ_Master1 -a timectrl:has_none=photobomb -t best:K=1 --fsvx=0 --threshx=1 --show
        python -m wbia --tf train_featscore_normalizer --db PZ_Master1 -a timectrl:has_none=photobomb -t best:K=3 --fsvx=0 --threshx=1 --show

    Example:
        >>> # SCRIPT
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> encoder = train_featscore_normalizer()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    import wbia

    # TODO: training / loading / general external models
    qreq_ = wbia.testdata_qreq_(defaultdb='PZ_MTEST', a=['default'], p=['default'])
    datakw = NormFeatScoreConfig.from_argv_dict()
    # datakw = dict(
    #    disttype=None,
    #    namemode=ut.get_argval('--namemode', default=True),
    #    fsvx=ut.get_argval('--fsvx', type_='fuzzy_subset',
    #                         default=slice(None, None, None)),
    #    threshx=ut.get_argval('--threshx', type_=int, default=None),
    #    thresh=ut.get_argval('--thresh', type_=float, default=.9),
    # )
    encoder = learn_featscore_normalizer(qreq_, datakw=datakw)
    encoder.save()
    return encoder


def learn_featscore_normalizer(qreq_, datakw={}, learnkw={}):
    r"""
    Takes the result of queries and trains a score encoder

    Args:
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters

    Returns:
        vtool.ScoreNormalizer: encoder

    CommandLine:
        python -m wbia --tf learn_featscore_normalizer --show -t default:
        python -m wbia --tf learn_featscore_normalizer --show --fsvx=0 --threshx=1 --show
        python -m wbia --tf learn_featscore_normalizer --show -a default:size=40 -t default:fg_on=False,lnbnn_on=False,ratio_thresh=1.0,K=1,Knorm=6,sv_on=False,normalizer_rule=name --fsvx=0 --threshx=1 --show

        python -m wbia --tf learn_featscore_normalizer --show --disttype=ratio
        python -m wbia --tf learn_featscore_normalizer --show --disttype=lnbnn
        python -m wbia --tf learn_featscore_normalizer --show --disttype=L2_sift -t default:K=1

        python -m wbia --tf learn_featscore_normalizer --show --disttype=L2_sift -a timectrl -t default:K=1 --db PZ_Master1
        python -m wbia --tf learn_featscore_normalizer --show --disttype=ratio -a timectrl -t default:K=1 --db PZ_Master1
        python -m wbia --tf learn_featscore_normalizer --show --disttype=lnbnn -a timectrl -t default:K=1 --db PZ_Master1

        # LOOK AT THIS
        python -m wbia --tf learn_featscore_normalizer --show --disttype=normdist -a timectrl -t default:K=1 --db PZ_Master1
        #python -m wbia --tf learn_featscore_normalizer --show --disttype=parzen -a timectrl -t default:K=1 --db PZ_Master1
        #python -m wbia --tf learn_featscore_normalizer --show --disttype=norm_parzen -a timectrl -t default:K=1 --db PZ_Master1

        python -m wbia --tf learn_featscore_normalizer --show --disttype=lnbnn --db PZ_Master1 -a timectrl -t best

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> learnkw = {}
        >>> datakw = NormFeatScoreConfig.from_argv_dict()
        >>> qreq_ = wbia.testdata_qreq_(
        >>>     defaultdb='PZ_MTEST', a=['default'], p=['default'])
        >>> encoder = learn_featscore_normalizer(qreq_, datakw, learnkw)
        >>> ut.quit_if_noshow()
        >>> encoder.visualize(figtitle=encoder.get_cfgstr())
        >>> ut.show_if_requested()
    """
    cm_list = qreq_.execute()
    print('learning scorenorm')
    print('datakw = %s' % ut.repr3(datakw))
    tp_scores, tn_scores, scorecfg = get_training_featscores(qreq_, cm_list, **datakw)
    _learnkw = dict(monotonize=True, adjust=2)
    _learnkw.update(learnkw)
    encoder = vt.ScoreNormalizer(**_learnkw)
    encoder.fit_partitioned(tp_scores, tn_scores, verbose=False)
    # ut.hashstr27(qreq_.get_cfgstr())

    # Maintain regen command info: TODO: generalize and integrate
    encoder._regen_info = {
        'cmd': 'python -m wbia --tf learn_featscore_normalizer',
        'scorecfg': scorecfg,
        'learnkw': learnkw,
        'datakw': datakw,
        'qaids': qreq_.qaids,
        'daids': qreq_.daids,
        'qreq_cfg': qreq_.get_full_cfgstr(),
        'qreq_regen_info': getattr(qreq_, '_regen_info', {}),
    }
    # 'timestamp': ut.get_timestamp(),

    scorecfg_safe = scorecfg
    scorecfg_safe = re.sub('[' + re.escape('()= ') + ']', '', scorecfg_safe)
    scorecfg_safe = re.sub('[' + re.escape('+*<>[]') + ']', '_', scorecfg_safe)

    hashid = ut.hashstr27(ut.to_json(encoder._regen_info))
    naidinfo = 'q%s_d%s' % (len(qreq_.qaids), len(qreq_.daids))
    cfgstr = 'featscore_{}_{}_{}_{}'.format(
        scorecfg_safe, qreq_.ibs.get_dbname(), naidinfo, hashid
    )
    encoder.cfgstr = cfgstr
    return encoder


def get_training_annotscores(qreq_, cm_list):
    """
    Returns the annotation scores between each query and the correct groundtruth
    annotations as well as the top scoring false annotations.
    """
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []

    ibs = qreq_.ibs

    trainable = [
        ibs.get_annot_has_groundtruth(cm.qaid, daid_list=cm.daid_list) for cm in cm_list
    ]
    cm_list_ = ut.compress(cm_list, trainable)

    for cm in cm_list_:
        qaid = cm.qaid
        qnid = ibs.get_annot_name_rowids(cm.qaid)

        nscoretup = cm.get_ranked_nids_and_aids()
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores) = nscoretup

        sorted_ndiff = -np.diff(sorted_nscores.tolist())
        sorted_nids = np.array(sorted_nids)
        is_positive = sorted_nids == qnid
        is_negative = np.logical_and(~is_positive, sorted_nids > 0)
        # Only take data from results with positive and negative examples
        if not np.any(is_positive) or not np.any(is_negative):
            continue
        gt_rank = np.nonzero(is_positive)[0][0]
        gf_rank = np.nonzero(is_negative)[0][0]
        # Only take correct groundtruth scores
        if gt_rank == 0 and len(sorted_nscores) > gf_rank:
            if len(sorted_ndiff) > gf_rank:
                good_tp_nscores.append(sorted_nscores[gt_rank])
                good_tn_nscores.append(sorted_nscores[gf_rank])
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))
    tp_scores = np.array(good_tp_nscores)
    tn_scores = np.array(good_tn_nscores)
    return tp_scores, tn_scores, good_tn_aidnid_pairs, good_tp_aidnid_pairs


def get_training_featscores(
    qreq_,
    cm_list,
    disttype=None,
    namemode=True,
    fsvx=slice(None, None, None),
    threshx=None,
    thresh=0.9,
    num=None,
    top_percent=None,
):
    """
    Returns the flattened set of feature scores between each query and the
    correct groundtruth annotations as well as the top scoring false
    annotations.

    Args:
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters
        cm_list (list):
        disttype (None): (default = None)
        namemode (bool): (default = True)
        fsvx (slice): (default = slice(None, None, None))
        threshx (None): (default = None)
        thresh (float): only used if threshx is specified (default = 0.9)

    SeeAlso:
        TestResult.draw_feat_scoresep

    Returns:
        tuple: (tp_scores, tn_scores, scorecfg)

    CommandLine:
        python -m wbia.algo.hots.scorenorm --exec-get_training_featscores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> cm_list, qreq_ = wbia.testdata_cmlist(defaultdb='PZ_MTEST', a=['default:qsize=10'])
        >>> disttype = None
        >>> namemode = True
        >>> fsvx = None
        >>> threshx = 1
        >>> thresh = 0.5
        >>> (tp_scores, tn_scores, scorecfg) = get_training_featscores(
        >>>     qreq_, cm_list, disttype, namemode, fsvx, threshx, thresh)
        >>> result = scorecfg
        >>> print(result)
        (lnbnn*fg)[fg > 0.5]

        lnbnn*fg[fg > 0.5]
    """
    if fsvx is None:
        fsvx = slice(None, None, None)

    fsv_col_lbls = None
    tp_fsvs_list = []
    tn_fsvs_list = []

    # cm_list = [ cm_list[key] for key in sorted(cm_list.keys()) ]
    # Train on only positive examples
    trainable = [
        qreq_.ibs.get_annot_has_groundtruth(cm.qaid, daid_list=cm.daid_list)
        and cm.get_top_nids()[0] == cm.qnid
        for cm in cm_list
    ]
    cm_list_ = ut.compress(cm_list, trainable)
    print('training using %d chipmatches' % (len(cm_list)))

    if disttype is None:
        # fsv_col_lbls = cm.fsv_col_lbls
        train_getter = get_training_fsv
    else:
        # fsv_col_lbls = ut.ensure_iterable(disttype)
        # annots = {}  # Hack for cached vector lookups
        ibs = qreq_.ibs
        data_annots = ut.KeyedDefaultDict(
            ibs.get_annot_lazy_dict, config2_=qreq_.data_config2_
        )
        query_annots = ut.KeyedDefaultDict(
            ibs.get_annot_lazy_dict, config2_=qreq_.query_config2_
        )
        train_getter = partial(
            get_training_desc_dist,
            fsv_col_lbls=fsv_col_lbls,
            qreq_=qreq_,
            data_annots=data_annots,
            query_annots=query_annots,
        )

    for cm in ut.ProgIter(cm_list_, lbl='building train featscores', adjust=True, freq=1):
        try:
            tp_fsv, tn_fsv = train_getter(cm, namemode=namemode, top_percent=top_percent)
            tp_fsvs_list.extend(tp_fsv)
            tn_fsvs_list.extend(tn_fsv)
        except UnbalancedExampleException:
            continue
    fsv_tp = np.vstack(tp_fsvs_list)
    fsv_tn = np.vstack(tn_fsvs_list)

    fsv_col_lbls_ = ut.take(fsv_col_lbls, fsvx)
    fsv_tp_ = fsv_tp.T[fsvx].T
    fsv_tn_ = fsv_tn.T[fsvx].T

    if threshx is not None:
        tp_scores = fsv_tp_[fsv_tp.T[threshx] > thresh].prod(axis=1)
        tn_scores = fsv_tn_[fsv_tn.T[threshx] > thresh].prod(axis=1)
        threshpart = '[' + fsv_col_lbls[threshx] + ' > ' + str(thresh) + ']'
        scorecfg = '(%s)%s' % ('*'.join(fsv_col_lbls_), threshpart)
    else:
        tp_scores = fsv_tp_.prod(axis=1)
        tn_scores = fsv_tn_.prod(axis=1)
        scorecfg = '*'.join(fsv_col_lbls_)

    return tp_scores, tn_scores, scorecfg


class UnbalancedExampleException(Exception):
    pass


def get_topannot_training_idxs(cm, num=2):
    """ top annots version

    Args:
        cm (wbia.ChipMatch):  object of feature correspondences and scores
        num (int): number of top annots per TP/TN (default = 2)

    CommandLine:
        python -m wbia.algo.hots.scorenorm --exec-get_topannot_training_idxs --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(defaultdb='PZ_MTEST')
        >>> num = 2
        >>> cm.score_annot_csum(qreq_)
        >>> (tp_idxs, tn_idxs) = get_topannot_training_idxs(cm, num)
        >>> result = ('(tp_idxs, tn_idxs) = %s' % (ut.repr2((tp_idxs, tn_idxs), nl=1),))
        >>> print(result)

        (tp_idxs, tn_idxs) = (
            np.array([0, 1], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
        )
    """
    if num is None:
        num = 2
    sortx = cm.argsort()
    sorted_nids = cm.dnid_list.take(sortx, axis=0)
    mask = sorted_nids == cm.qnid
    tp_idxs_ = np.where(mask)[0]
    if len(tp_idxs_) == 0:
        # if ut.STRICT:
        #    raise Exception('tp_idxs_=0')
        # else:
        raise UnbalancedExampleException('tp_idxs_=0')
    tn_idxs_ = np.where(~mask)[0]
    if len(tn_idxs_) == 0:
        # if ut.STRICT:
        #    raise Exception('tn_idxs_=0')
        # else:
        raise UnbalancedExampleException('tn_idxs_=0')
    tp_idxs = tp_idxs_[0:num]
    tn_idxs = tn_idxs_[0:num]
    return tp_idxs, tn_idxs


def get_topname_training_idxs(cm, num=5):
    """
    gets the index of the annots in the top groundtrue name and the top
    groundfalse names.

    Args:
        cm (wbia.ChipMatch):  object of feature correspondences and scores
        num(int): number of false names (default = 5)

    Returns:
        tuple: (tp_idxs, tn_idxs)
            cm.daid_list[tp_idxs] are all of the
               annotations in the correct name.
            cm.daid_list[tn_idxs] are all of the
               annotations in the top `num_false` incorrect names.

    CommandLine:
        python -m wbia --tf get_topname_training_idxs --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm('PZ_MTEST', a='default:dindex=0:10,qindex=0:1', t='best')
        >>> num = 1
        >>> (tp_idxs, tn_idxs) = get_topname_training_idxs(cm, num)
        >>> result = ('(tp_idxs, tn_idxs) = %s' % (ut.repr2((tp_idxs, tn_idxs), nl=1),))
        >>> print(result)

        (tp_idxs, tn_idxs) = (
            np.array([0, 1, 2, 3], dtype=np.int64),
            [4, 5, 6, 7],
        )
    """
    if num is None:
        num = 5
    sortx = cm.name_argsort()
    sorted_nids = vt.take2(cm.unique_nids, sortx)
    sorted_groupxs = ut.take(cm.name_groupxs, sortx)
    # name ranks of the groundtrue name
    tp_ranks = np.where(sorted_nids == cm.qnid)[0]
    if len(tp_ranks) == 0:
        # if ut.STRICT:
        #    raise Exception('tp_ranks=0')
        # else:
        raise UnbalancedExampleException('tp_ranks=0')

    # name ranks of the top groundfalse names
    tp_rank = tp_ranks[0]
    tn_ranks = [
        rank for rank in range(num + 1) if rank != tp_rank and rank < len(sorted_groupxs)
    ]
    if len(tn_ranks) == 0:
        # if ut.STRICT:
        #    raise Exception('tn_ranks=0')
        # else:
        raise UnbalancedExampleException('tn_ranks=0')
    # annot idxs of the examples
    tp_idxs = sorted_groupxs[tp_rank]
    tn_idxs = ut.flatten(ut.take(sorted_groupxs, tn_ranks))
    return tp_idxs, tn_idxs


def get_training_fsv(cm, namemode=True, num=None, top_percent=None):
    """
    CommandLine:
        python -m wbia.algo.hots.scorenorm --exec-get_training_fsv --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> num = None
        >>> cm, qreq_ = wbia.testdata_cm('PZ_MTEST', a='default:dindex=0:10,qindex=0:1', t='best')
        >>> (tp_fsv, tn_fsv) = get_training_fsv(cm, namemode=False)
        >>> result = ('(tp_fsv, tn_fsv) = %s' % (ut.repr2((tp_fsv, tn_fsv), nl=1),))
        >>> print(result)
    """
    if namemode:
        tp_idxs, tn_idxs = get_topname_training_idxs(cm, num=num)
    else:
        tp_idxs, tn_idxs = get_topannot_training_idxs(cm, num=num)

    # Keep only the top scoring half of the feature matches
    # top_percent = None
    if top_percent is not None:
        cm_orig = cm
        # cm_orig.assert_self(qreq_)

        tophalf_indicies = [
            ut.take_percentile(fs.argsort()[::-1], top_percent)
            for fs in cm.get_fsv_prod_list()
        ]
        cm = cm_orig.take_feature_matches(tophalf_indicies, keepscores=True)

        assert np.all(cm_orig.daid_list.take(tp_idxs) == cm.daid_list.take(tp_idxs))
        assert np.all(cm_orig.daid_list.take(tn_idxs) == cm.daid_list.take(tn_idxs))
        # cm.assert_self(qreq_)

    tp_fsv = np.vstack(ut.take(cm.fsv_list, tp_idxs))
    tn_fsv = np.vstack(ut.take(cm.fsv_list, tn_idxs))
    return tp_fsv, tn_fsv


@profile
def get_training_desc_dist(
    cm,
    qreq_,
    fsv_col_lbls=[],
    namemode=True,
    top_percent=None,
    data_annots=None,
    query_annots=None,
    num=None,
):
    r"""
    computes custom distances on prematched descriptors

    SeeAlso:
        python -m wbia --tf learn_featscore_normalizer --show --disttype=ratio

        python -m wbia --tf learn_featscore_normalizer --show --disttype=normdist -a timectrl -t default:K=1 --db PZ_Master1 --save pzmaster_normdist.png
        python -m wbia --tf learn_featscore_normalizer --show --disttype=normdist -a timectrl -t default:K=1 --db PZ_MTEST --save pzmtest_normdist.png
        python -m wbia --tf learn_featscore_normalizer --show --disttype=normdist -a timectrl -t default:K=1 --db GZ_ALL

        python -m wbia --tf learn_featscore_normalizer --show --disttype=L2_sift -a timectrl -t default:K=1 --db PZ_MTEST
        python -m wbia --tf learn_featscore_normalizer --show --disttype=L2_sift -a timectrl -t default:K=1 --db PZ_Master1

        python -m wbia --tf compare_featscores --show --disttype=L2_sift,normdist -a timectrl -t default:K=1 --db GZ_ALL

    CommandLine:
        python -m wbia.algo.hots.scorenorm --exec-get_training_desc_dist
        python -m wbia.algo.hots.scorenorm --exec-get_training_desc_dist:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(defaultdb='PZ_MTEST')
        >>> fsv_col_lbls = ['ratio', 'lnbnn', 'L2_sift']
        >>> namemode = False
        >>> (tp_fsv, tn_fsv) = get_training_desc_dist(cm, qreq_, fsv_col_lbls,
        >>>                                           namemode=namemode)
        >>> result = ut.repr2((tp_fsv.T, tn_fsv.T), nl=1)
        >>> print(result)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scorenorm import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(defaultdb='PZ_MTEST')
        >>> fsv_col_lbls = cm.fsv_col_lbls
        >>> num = None
        >>> namemode = False
        >>> top_percent = None
        >>> data_annots = None
        >>> (tp_fsv1, tn_fsv1) = get_training_fsv(cm, namemode=namemode,
        >>>                                       top_percent=top_percent)
        >>> (tp_fsv, tn_fsv) = get_training_desc_dist(cm, qreq_, fsv_col_lbls,
        >>>                                           namemode=namemode,
        >>>                                           top_percent=top_percent)
        >>> vt.asserteq(tp_fsv1, tp_fsv)
        >>> vt.asserteq(tn_fsv1, tn_fsv)
    """
    if namemode:
        tp_idxs, tn_idxs = get_topname_training_idxs(cm, num=num)
    else:
        tp_idxs, tn_idxs = get_topannot_training_idxs(cm, num=num)

    if top_percent is not None:
        cm_orig = cm
        cm_orig.assert_self(qreq_, verbose=False)

        # Keep only the top scoring half of the feature matches
        tophalf_indicies = [
            ut.take_percentile(fs.argsort()[::-1], top_percent)
            for fs in cm.get_fsv_prod_list()
        ]
        cm = cm_orig.take_feature_matches(tophalf_indicies, keepscores=True)

        assert np.all(cm_orig.daid_list.take(tp_idxs) == cm.daid_list.take(tp_idxs))
        assert np.all(cm_orig.daid_list.take(tn_idxs) == cm.daid_list.take(tn_idxs))

        cm.assert_self(qreq_, verbose=False)

    ibs = qreq_.ibs
    query_config2_ = qreq_.extern_query_config2
    data_config2_ = qreq_.extern_data_config2
    special_xs, dist_xs = vt.index_partition(
        fsv_col_lbls, ['fg', 'ratio', 'lnbnn', 'normdist']
    )
    dist_lbls = ut.take(fsv_col_lbls, dist_xs)
    special_lbls = ut.take(fsv_col_lbls, special_xs)

    qaid = cm.qaid
    # cm.assert_self(qreq_=qreq_)

    fsv_list = []
    for idxs in [tp_idxs, tn_idxs]:
        daid_list = cm.daid_list.take(idxs)

        # Matching indices in query / databas images
        qfxs_list = ut.take(cm.qfxs_list, idxs)
        dfxs_list = ut.take(cm.dfxs_list, idxs)

        need_norm = (
            len(ut.setintersect_ordered(['ratio', 'lnbnn', 'normdist'], special_lbls)) > 0
        )
        # need_norm |= 'parzen' in special_lbls
        # need_norm |= 'norm_parzen' in special_lbls
        need_dists = len(dist_xs) > 0

        if need_dists or need_norm:
            qaid_list = [qaid] * len(qfxs_list)
            qvecs_flat_m = np.vstack(
                ibs.get_annot_vecs_subset(qaid_list, qfxs_list, config2_=query_config2_)
            )
            dvecs_flat_m = np.vstack(
                ibs.get_annot_vecs_subset(daid_list, dfxs_list, config2_=data_config2_)
            )

        if need_norm:
            assert any(x is not None for x in cm.filtnorm_aids), 'no normalizer known'
            naids_list = ut.take(cm.naids_list, idxs)
            nfxs_list = ut.take(cm.nfxs_list, idxs)
            nvecs_flat = ibs.lookup_annot_vecs_subset(
                naids_list, nfxs_list, config2_=data_config2_, annots=data_annots
            )
            # import utool
            # with utool.embed_on_exception_context:
            # nvecs_flat_m = np.vstack(ut.compress(nvecs_flat, nvecs_flat))
            _nvecs_flat_m = ut.compress(nvecs_flat, nvecs_flat)
            nvecs_flat_m = vt.safe_vstack(
                _nvecs_flat_m, qvecs_flat_m.shape, qvecs_flat_m.dtype
            )

            vdist = vt.L2_sift(qvecs_flat_m, dvecs_flat_m)
            ndist = vt.L2_sift(qvecs_flat_m, nvecs_flat_m)

            # assert np.all(vdist <= ndist)
            # import utool
            # utool.embed()

            # vdist = vt.L2_sift_sqrd(qvecs_flat_m, dvecs_flat_m)
            # ndist = vt.L2_sift_sqrd(qvecs_flat_m, nvecs_flat_m)

            # vdist = vt.L2_root_sift(qvecs_flat_m, dvecs_flat_m)
            # ndist = vt.L2_root_sift(qvecs_flat_m, nvecs_flat_m)

            # x = cm.fsv_list[0][0:5].T[0]
            # y = (ndist - vdist)[0:5]

        if len(special_xs) > 0:
            special_dist_list = []
            # assert special_lbls[0] == 'fg'
            if 'fg' in special_lbls:
                # hack for fgweights (could get them directly from fsv)
                qfgweights_flat_m = np.hstack(
                    ibs.get_annot_fgweights_subset(
                        [qaid] * len(qfxs_list), qfxs_list, config2_=query_config2_
                    )
                )
                dfgweights_flat_m = np.hstack(
                    ibs.get_annot_fgweights_subset(
                        daid_list, dfxs_list, config2_=data_config2_
                    )
                )
                fgweights = np.sqrt(qfgweights_flat_m * dfgweights_flat_m)
                special_dist_list.append(fgweights)

            if 'ratio' in special_lbls:
                # Integrating ratio test
                ratio_dist = vdist / ndist
                special_dist_list.append(ratio_dist)

            if 'lnbnn' in special_lbls:
                lnbnn_dist = ndist - vdist
                special_dist_list.append(lnbnn_dist)

            # if 'parzen' in special_lbls:
            #    parzen = vt.gauss_parzen_est(vdist, sigma=.38)
            #    special_dist_list.append(parzen)

            # if 'norm_parzen' in special_lbls:
            #    parzen = vt.gauss_parzen_est(ndist, sigma=.38)
            #    special_dist_list.append(parzen)

            if 'normdist' in special_lbls:
                special_dist_list.append(ndist)

            special_dists = np.vstack(special_dist_list).T
        else:
            special_dists = np.empty((0, 0))

        if len(dist_xs) > 0:
            # Get descriptors
            # Compute descriptor distnaces
            _dists = vt.compute_distances(qvecs_flat_m, dvecs_flat_m, dist_lbls)
            dists = np.vstack(_dists.values()).T
        else:
            dists = np.empty((0, 0))

        fsv = vt.rebuild_partition(special_dists.T, dists.T, special_xs, dist_xs)
        fsv = np.array(fsv).T
        fsv_list.append(fsv)
    tp_fsv, tn_fsv = fsv_list
    return tp_fsv, tn_fsv


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.scorenorm
        python -m wbia.algo.hots.scorenorm --allexamples
        python -m wbia.algo.hots.scorenorm --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
