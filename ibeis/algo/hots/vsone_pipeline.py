# -*- coding: utf-8 -*-
"""
special pipeline for vsone specific functions

Current Issues:
    * getting feature distinctiveness is too slow, we can either try a different
      model, or precompute feature distinctiveness.

      - we can reduce the size of the vsone shortlist

TODOLIST:
    * Unconstrained is a terrible name. It is constrianed by the ratio
    * Precompute distinctivness
    #* keep feature matches from vsmany (allow fm_B)
    #* Each keypoint gets
    #  - foregroundness
    #  - global distinctivness (databasewide) LNBNN
    #  - local distinctivness (imagewide) RATIO
    #  - regional match quality (descriptor based) COS
    * Asymetric weight scoring

    * FIX BUGS IN score_chipmatch_nsum FIRST THING TOMORROW.
     dict keys / vals are being messed up. very inoccuous

    Visualization to "prove" that vsone works


TestCases:
    PZ_Master0 - aids 1801, 4792 - near-miss


TestFuncs:
    >>> # VsMany Only
    python -m ibeis.algo.hots.vsone_pipeline --test-show_post_vsmany_vser --show
    >>> # VsOne Only
    python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --no-vsmany_coeff
    >>> # VsOne + VsMany
    python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show


    >>> # Rerank Vsone Test Harness
    python -c "import utool as ut; ut.write_modscript_alias('Tvs1RR.sh', 'dev.py', '--allgt  --db PZ_MTEST --index 1:40:2')"  # NOQA
    sh Tvs1RR.sh -t custom:rrvsone_on=True custom custom:rrvsone_on=True
    sh Tvs1RR.sh -t custom custom:rrvsone_on=True --print-scorediff-mat-stats
    sh Tvs1RR.sh -t custom:rrvsone_on=True custom:rrvsone_on=True, --print-confusion-stats --print-scorediff-mat-stats

    --print-scorediff-mat-stats --print-confusion-stats

"""
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
import six  # NOQA
import numpy as np
import vtool as vt
from ibeis.algo.hots import hstypes
from ibeis.algo.hots import chip_match
from ibeis.algo.hots import scoring
import functools
from vtool import matching
from ibeis.algo.hots import _pipeline_helpers as plh  # NOQA
import utool as ut
from six.moves import zip, range, reduce  # NOQA
print, rrr, profile = ut.inject2(__name__)

#from collections import namedtuple


def get_training_pairs():
    """
    Notes:
        Meausres are:

          Local:
            * LNBNN score
            * Foregroundness score
            * SIFT correspondence distance
            * SIFT normalizer distance
            * Correspondence neighbor rank
            * Nearest unique name distances
            * SVER Error

          Global:
            * Viewpoint labels
            * Quality Labels
            * Database Size
            * Number of correspondences
            % Total LNBNN Score

        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> from vtool.matching import *  # NOQA
    """
    import ibeis
    qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', a='default')
    cm_list = qreq_.execute()
    num_gt = 3
    num_hard_gf = 2
    num_rand_gf = 1
    aid_pairs = []

    # Extract pairs of annotation ids
    for cm in cm_list:
        cm.score_csum(qreq_)
        gt_aids = cm.get_top_gt_aids(qreq_.ibs)[0:num_gt].tolist()
        hard_gf_aids = cm.get_top_gf_aids(qreq_.ibs)[0:num_hard_gf].tolist()
        gf_aids = qreq_.ibs.get_annot_groundfalse(cm.qaid,
                                                  daid_list=qreq_.daids)
        rand_gf_aids = ut.random_sample(gf_aids, num_rand_gf)
        aid_pairs.extend([(cm.qaid, aid)
                          for aid in gt_aids + hard_gf_aids + rand_gf_aids])
    aid_pairs = vt.unique_rows(np.array(aid_pairs), directed=False).tolist()
    query_aids = ut.take_column(aid_pairs, 0)
    data_aids = ut.take_column(aid_pairs, 1)

    # Prepare lazy attributes for annotations
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)
    configured_annot_dict = ut.ddict(dict)
    config_aids_pairs = [(qannot_cfg, query_aids), (dannot_cfg, data_aids)]
    for config, aids in ut.ProgIter(config_aids_pairs, lbl='prepare annots'):
        annot_dict = configured_annot_dict[config]
        for aid in ut.unique(aids):
            if aid not in annot_dict:
                annot = ibs.get_annot_lazy_dict(aid, config)
                flann_params = {'algorithm': 'kdtree', 'trees': 4}
                vt.matching.ensure_metadata_flann(annot, flann_params)
                annot_dict[aid] = annot
                annot['yaw'] = ibs.get_annot_yaw_texts(aid)
                annot['qual'] = ibs.get_annot_qualities(aid)
                annot['gps'] = ibs.get_annot_image_gps2(aid)
                annot['time'] = ibs.get_annot_image_unixtimes_asfloat(aid)
                del annot['annot_context_options']

    # Extract pairs of annot objects (with shared caches)
    annot1_list = ut.take(configured_annot_dict[qannot_cfg], query_aids)
    annot2_list = ut.take(configured_annot_dict[dannot_cfg], data_aids)
    truth_list = np.array(qreq_.ibs.get_aidpair_truths(*zip(*aid_pairs)))

    verbose = True  # NOQA

    match_list = [vt.PairwiseMatch(annot1, annot2)
                  for annot1, annot2 in zip(annot1_list, annot2_list)]

    # Preload needed attributes
    for match in ut.ProgIter(match_list, lbl='preload'):
        match.annot1['flann']
        match.annot2['vecs']

    cfgdict = {'checks': 20}

    for match in ut.ProgIter(match_list, lbl='assign vsone'):
        match.assign(cfgdict)

    global_keys = ['yaw', 'qual', 'gps', 'time']
    for match in ut.ProgIter(match_list, lbl='setup globals'):
        match.global_measures = {}
        for key in global_keys:
            match.global_measures[key] = (match.annot1[key], match.annot2[key])

    import sklearn
    import sklearn.metrics
    if False:
        # Param search for vsone
        import plottool as pt
        pt.qt4ensure()

        import sklearn
        import sklearn.metrics
        skf = sklearn.model_selection.StratifiedKFold(n_splits=10,
                                                      random_state=119372)

        basis = {'ratio_thresh': np.linspace(.6, .7, 50).tolist()}
        grid = ut.all_dict_combinations(basis)
        xdata = np.array(ut.take_column(grid, 'ratio_thresh'))

        def _ratio_thresh(y_true, match_list):
            # Try and find optional ratio threshold
            auc_list = []
            for cfgdict in ut.ProgIter(grid, lbl='gridsearch'):
                y_score = [
                    match.fs.compress(match.ratio_test_flags(cfgdict)).sum()
                    for match in match_list
                ]
                auc = sklearn.metrics.roc_auc_score(y_true, y_score)
                auc_list.append(auc)
            auc_list = np.array(auc_list)
            return auc_list

        auc_list = _ratio_thresh(truth_list, match_list)
        pt.plot(xdata, auc_list)
        subx, suby = vt.argsubmaxima(auc_list, xdata)
        best_ratio_thresh = subx[suby.argmax()]

        skf_results = []
        y_true = truth_list
        for train_idx, test_idx in skf.split(match_list, truth_list):
            match_list_ = ut.take(match_list, train_idx)
            y_true = truth_list.take(train_idx)
            auc_list = _ratio_thresh(y_true, match_list_)
            subx, suby = vt.argsubmaxima(auc_list, xdata, maxima_thresh=.8)
            best_ratio_thresh = subx[suby.argmax()]
            skf_results.append(best_ratio_thresh)
        print('skf_results.append = %r' % (np.mean(skf_results),))

    for match in ut.ProgIter(match_list, lbl='assign vsone'):
        match.apply_ratio_test({'ratio_thresh': .638}, inplace=True)

    score_list = np.array([m.fs.sum() for m in match_list])
    encoder = vt.ScoreNormalizer()
    encoder.fit(score_list, truth_list, verbose=True)
    encoder.visualize()

    def matches_auc(match_list):
        score_list = np.array([m.fs.sum() for m in match_list])
        auc = sklearn.metrics.roc_auc_score(truth_list, score_list)
        print('auc = %r' % (auc,))
        return auc

    matchesORIG = match_list
    matches_auc(matchesORIG)

    matches_SV = [match.apply_sver(inplace=False)
                  for match in ut.ProgIter(matchesORIG, lbl='sver')]
    matches_auc(matches_SV)

    matches_RAT = [match.apply_ratio_test(inplace=False)
                   for match in ut.ProgIter(matchesORIG, lbl='ratio')]
    matches_auc(matches_RAT)

    matches_RAT_SV = [match.apply_sver(inplace=False)
                      for match in ut.ProgIter(matches_RAT, lbl='sver')]
    matches_auc(matches_RAT_SV)

    # =====================================
    # Attempt to train a simple classsifier
    # =====================================

    import pandas as pd
    import sklearn.model_selection
    from sklearn.ensemble import RandomForestClassifier

    allow_nan = True
    pairwise_feats = pd.DataFrame([m.make_pairwise_constlen_feature('ratio')
                                   for m in matches_RAT_SV])
    pairwise_feats[pd.isnull(pairwise_feats)] = np.nan

    if allow_nan:
        X_withnan = pairwise_feats.values.copy()
    valid_colx = np.where(np.all(pairwise_feats.notnull(), axis=0))[0]
    valid_cols = pairwise_feats.columns[valid_colx]
    X_nonan = pairwise_feats[valid_cols].values.copy()

    y = np.array([m.annot1['nid'] == m.annot2['nid'] for m in matches_RAT_SV])

    rng = np.random.RandomState(42)
    xvalkw = dict(n_splits=5, shuffle=True, random_state=rng)
    skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
    skf_iter = skf.split(X=X_nonan, y=y)
    df_results = pd.DataFrame(columns=['auc_naive', 'auc_learn_nonan', 'auc_learn_withnan'])

    rf_params = dict(n_estimators=20, bootstrap=True, verbose=1)
    for count, (train_idx, test_idx) in enumerate(skf_iter):
        print('\n========')
        y_test = y[test_idx]
        y_train = y[train_idx]
        if True:
            score_list = np.array([m.fs.sum() for m in ut.take(matches_RAT_SV, test_idx)])
            auc_naive = sklearn.metrics.roc_auc_score(y_test, score_list)

        if True:
            X_train = X_nonan[train_idx]
            X_test = X_nonan[test_idx]
            # Train uncalibrated random forest classifier on train data
            clf = RandomForestClassifier(**rf_params)
            clf.fit(X_train, y_train)

            # evaluate on test data
            clf_probs = clf.predict_proba(X_test)
            auc_learn_nonan = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])

        if allow_nan:
            X_train = X_withnan[train_idx]
            X_test = X_withnan[test_idx]
            # Train uncalibrated random forest classifier on train data
            clf = RandomForestClassifier(missing_values=np.nan, **rf_params)
            clf.fit(X_train, y_train)

            # evaluate on test data
            clf_probs = clf.predict_proba(X_test)
            # log_loss = sklearn.metrics.log_loss(y_test, clf_probs)
            auc_learn_withnan = sklearn.metrics.roc_auc_score(y_test, clf_probs.T[1])

        newrow = pd.DataFrame([[auc_naive, auc_learn_nonan, auc_learn_withnan]], columns=df_results.columns)
        print(newrow)
        df_results = df_results.append([newrow], ignore_index=True)

    print(df_results)

    # TODO: TSNE?
    # http://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
    # Perform t-distributed stochastic neighbor embedding.
    from sklearn import manifold
    import matplotlib.pyplot as plt
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(feats).T
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    def monkey_to_str_columns(self):
        frame = self.tr_frame
        highlight_func = 'max'
        highlight_func = ut.partial(np.argmax, axis=1)
        highlight_cols = [0, 1]

        perrow_colxs = highlight_func(frame[highlight_cols].values)
        n_rows = len(perrow_colxs)
        n_cols = len(highlight_cols)
        shape = (n_rows, n_cols)
        flat_idxs = np.ravel_multi_index((np.arange(n_rows), perrow_colxs), shape)
        flags2d = np.zeros(shape, dtype=np.int32)
        flags2d.ravel()[flat_idxs] = 1
        # np.unravel_index(flat_idxs, shape)

        def color_func(val, level):
            if level:
                return ut.color_text(val, 'red')
            else:
                return val

        _make_fixed_width = pd.formats.format._make_fixed_width
        frame = self.tr_frame
        str_index = self._get_formatted_index(frame)
        str_columns = self._get_formatted_column_labels(frame)
        if self.header:
            stringified = []
            for i, c in enumerate(frame):
                cheader = str_columns[i]
                max_colwidth = max(self.col_space or 0, *(self.adj.len(x)
                                                          for x in cheader))
                fmt_values = self._format_col(i)
                fmt_values = _make_fixed_width(fmt_values, self.justify,
                                               minimum=max_colwidth,
                                               adj=self.adj)
                max_len = max(np.max([self.adj.len(x) for x in fmt_values]),
                              max_colwidth)
                cheader = self.adj.justify(cheader, max_len, mode=self.justify)

                # Apply custom coloring
                # cflags = flags2d.T[i]
                # fmt_values = [color_func(val, level) for val, level in zip(fmt_values, cflags)]

                stringified.append(cheader + fmt_values)
        else:
            stringified = []
            for i, c in enumerate(frame):
                fmt_values = self._format_col(i)
                fmt_values = _make_fixed_width(fmt_values, self.justify,
                                               minimum=(self.col_space or 0),
                                               adj=self.adj)

                stringified.append(fmt_values)

        strcols = stringified
        if self.index:
            strcols.insert(0, str_index)

        # Add ... to signal truncated
        truncate_h = self.truncate_h
        truncate_v = self.truncate_v

        if truncate_h:
            col_num = self.tr_col_num
            # infer from column header
            col_width = self.adj.len(strcols[self.tr_size_col][0])
            strcols.insert(self.tr_col_num + 1, ['...'.center(col_width)] *
                           (len(str_index)))
        if truncate_v:
            n_header_rows = len(str_index) - len(frame)
            row_num = self.tr_row_num
            for ix, col in enumerate(strcols):
                # infer from above row
                cwidth = self.adj.len(strcols[ix][row_num])
                is_dot_col = False
                if truncate_h:
                    is_dot_col = ix == col_num + 1
                if cwidth > 3 or is_dot_col:
                    my_str = '...'
                else:
                    my_str = '..'

                if ix == 0:
                    dot_mode = 'left'
                elif is_dot_col:
                    cwidth = self.adj.len(strcols[self.tr_size_col][0])
                    dot_mode = 'center'
                else:
                    dot_mode = 'right'
                dot_str = self.adj.justify([my_str], cwidth, mode=dot_mode)[0]
                strcols[ix].insert(row_num + n_header_rows, dot_str)

        for cx_ in highlight_cols:
            cx = cx_ + bool(self.header)
            col = strcols[cx]
            for rx, val in enumerate(col[1:], start=1):
                strcols[cx][rx] = color_func(val, flags2d[rx - 1, cx - 1])

        return strcols

    def to_string_monkey(df):
        """  monkey patch to pandas to highlight the maximum value in specified
        cols of a row """
        kwds = dict(buf=None, columns=None, col_space=None, header=True,
                    index=True, na_rep='NaN', formatters=None,
                    float_format=None, sparsify=None, index_names=True,
                    justify=None, line_width=None, max_rows=None,
                    max_cols=None, show_dimensions=False)
        self = pd.formats.format.DataFrameFormatter(df, **kwds)
        ut.inject_func_as_method(self, monkey_to_str_columns, '_to_str_columns', override=True, force=True)
        def strip_ansi(text):
            import re
            ansi_escape = re.compile(r'\x1b[^m]*m')
            return ansi_escape.sub('', text)
        def justify_ansi(self, texts, max_len, mode='right'):
            if mode == 'left':
                return [x.ljust(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
            elif mode == 'center':
                return [x.center(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
            else:
                return [x.rjust(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
        def strlen_ansii(self, text):
            return pd.compat.strlen(strip_ansi(text), encoding=self.encoding)
        ut.inject_func_as_method(self.adj, strlen_ansii, 'len', override=True, force=True)
        ut.inject_func_as_method(self.adj, justify_ansi, 'justify', override=True, force=True)
        # strcols = monkey_to_str_columns(self)
        # texts = strcols[2]
        # str_ = self.adj.adjoin(1, *strcols)
        # print(str_)
        # print(strip_ansi(str_))
        self.to_string()
        result = self.buf.getvalue()
        return result

    df = df_results
    change = df[df.columns[1]] - df[df.columns[0]]
    percent_change = change / df[df.columns[0]] * 100
    df = df.assign(change=change)
    df = df.assign(percent_change=percent_change)

    print(to_string_monkey(df))
    print(df.to_string())


def build_vsone_metadata(qaid, daid, qreq_, use_ibscache=True):
    r"""
    Args:
        qaid (int):  query annotation id
        daid (?):
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters
        use_ibscache (bool): (default = True)

    Returns:
        tuple: (metadata, cfgdict)

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline build_vsone_metadata --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> cfgdict = {}
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> qaid, daid = ibs.get_name_aids([ibs.get_valid_nids()[3]])[0][0:2]
        >>> qaid, daid = 1, 10
        >>> qreq_ = ibs.new_query_request([qaid], [daid], cfgdict=cfgdict)
        >>> use_ibscache = not ut.get_argflag('--noibscache')
        >>> metadata, cfgdict = build_vsone_metadata(qaid, daid, qreq_, use_ibscache)
        >>> from vtool import inspect_matches
        >>> gt.ensure_qapp()
        >>> ut.qt4ensure()
        >>> self = inspect_matches.MatchInspector(metadata=metadata)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """
    ibs = qreq_.ibs
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_query_config2

    metadata = ut.LazyDict()
    annot1 = metadata['annot1'] = ut.LazyDict()
    annot2 = metadata['annot2'] = ut.LazyDict()
    annot1['rchip_fpath'] = ibs.get_annot_chip_fpath([qaid], config2_=qconfig2_)[0]
    annot2['rchip_fpath'] = ibs.get_annot_chip_fpath([daid], config2_=qconfig2_)[0]
    cfgdict = {}

    if use_ibscache:
        hack_multi_config = False
        if hack_multi_config:
            cfgdict = {'refine_method': 'affine'}
            data_config_list = query_config_list = [
                dict(affine_invariance=True),
                dict(affine_invariance=False),
            ]
            kpts1 = np.vstack([ibs.get_annot_kpts(qaid, config2_=config2_)
                               for config2_ in query_config_list])
            vecs1 = np.vstack([ibs.get_annot_vecs(qaid, config2_=config2_)
                               for config2_ in query_config_list])

            kpts2 = np.vstack([ibs.get_annot_kpts(daid, config2_=config2_)
                               for config2_ in data_config_list])
            vecs2 = np.vstack([ibs.get_annot_vecs(daid, config2_=config2_)
                               for config2_ in data_config_list])
            dlen_sqrd2 = ibs.get_annot_chip_dlensqrd([daid],
                                                     config2_=dconfig2_)[0]
            annot1['kpts'] = kpts1
            annot1['vecs'] = vecs1
            annot2['kpts'] = kpts2
            annot2['vecs'] = vecs2
            annot2['dlen_sqrd'] = dlen_sqrd2
        else:
            annot1['kpts'] = ibs.get_annot_kpts(qaid, config2_=qconfig2_)
            annot1['vecs'] = ibs.get_annot_vecs(qaid, config2_=qconfig2_)
            annot2['kpts'] = ibs.get_annot_kpts(daid, config2_=dconfig2_)
            annot2['vecs'] = ibs.get_annot_vecs(daid, config2_=dconfig2_)
            annot2['dlen_sqrd'] = ibs.get_annot_chip_dlensqrd([daid], config2_=dconfig2_)[0]
    return metadata, cfgdict


def vsone_single(qaid, daid, qreq_, use_ibscache=True, verbose=None):
    r"""
    Uses vtools internal method

    Args:
        qaid (int):  query annotation id
        daid (?):
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --exec-vsone_single --show

        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_single
        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_single --nocache
        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_single --nocache --show
        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_single --show -t default:AI=False

    SeeAlso:
        python -m ibeis.algo.hots.vsone_pipeline --exec-extract_aligned_parts:1 --show  -t default:AI=False  # see x 11

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> cfgdict = {}
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> qaid, daid = ibs.get_name_aids([ibs.get_valid_nids()[3]])[0][0:2]
        >>> qreq_ = ibs.new_query_request([qaid], [daid], cfgdict=cfgdict)
        >>> use_ibscache = not ut.get_argflag('--noibscache')
        >>> match = vsone_single(qaid, daid, qreq_, use_ibscache)
        >>> H1 = match.metadata['H_RAT']
        >>> ut.quit_if_noshow()
        >>> match.show(mode=1)
        >>> ut.show_if_requested()
    """
    metadata, cfgdict = build_vsone_metadata(qaid, daid, qreq_, use_ibscache=True)
    match = vt.vsone_matching(metadata, cfgdict=cfgdict, verbose=verbose)
    assert match.metadata is metadata
    return match


def vsone_name_independant_hack(ibs, nids, qreq_=None):
    r"""
    show grid of aids with matches inside and between names
    Args:
        ibs (IBEISController):  ibeis controller object
        nid (?):
        qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --exec-vsone_name_independant_hack --db PZ_MTEST --show
        python -m ibeis.algo.hots.vsone_pipeline --exec-vsone_name_independant_hack --db PZ_Master1 --show --nids=5099,5181

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> # TODO: testdata_qparams?
        >>> qreq_ = ibeis.testdata_qreq_(defaultdb='testdb1')
        >>> nids = ut.get_argval('--nids', type_=list, default=[1])
        >>> ibs = qreq_.ibs
        >>> result = vsone_name_independant_hack(qreq_.ibs, nids, qreq_)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    from plottool.interactions import ExpandableInteraction
    import plottool as pt
    import ibeis.viz
    fnum = pt.ensure_fnum(None)
    inter = ExpandableInteraction(fnum)

    aids = ut.flatten(ibs.get_name_aids(nids))
    print('len(aids) = %r' % (len(aids),))
    idxs = range(len(aids))

    def wrap_show_func(func, *args, **kwargs):
        def _show_func_wrap(fnum=None, pnum=None):
            return func(*args, fnum=fnum, pnum=pnum, **kwargs)
        return _show_func_wrap

    # TODO: append an optional alternate ishow function

    #for idx1, idx2 in ut.upper_diag_self_prodx(idxs):
    for idx1, idx2 in ut.self_prodx(idxs):
        aid1 = aids[idx1]
        aid2 = aids[idx2]
        #vsone_independant_pair_hack(ibs, aid1, aid2, qreq_)
        cfgdict = dict(codename='vsone', fg_on=False)
        cfgdict.update(**qreq_.extern_data_config2.hesaff_params)
        vsone_qreq_ = ibs.new_query_request([aid1], [aid2], cfgdict=cfgdict)
        cm_vsone = vsone_qreq_.execute()[0]
        cm_vsone = cm_vsone.extend_results(vsone_qreq_)
        #cm_vsone.ishow_single_annotmatch(vsone_qreq_, aid2=aid2, fnum=fnum, pnum=(len(aids), len(aids), (idx1 * len(aids) + idx2) + 1))
        #cm_vsone.show_single_annotmatch(vsone_qreq_, aid2=aid2, fnum=fnum, pnum=(len(aids), len(aids), (idx1 * len(aids) + idx2) + 1))
        pnum = (len(aids), len(aids), (idx1 * len(aids) + idx2) + 1)
        show_func = wrap_show_func(cm_vsone.show_single_annotmatch, vsone_qreq_, aid2=aid2, draw_lbl=False, show_name_score=True)
        ishow_func = wrap_show_func(cm_vsone.ishow_single_annotmatch, vsone_qreq_, aid2=aid2, draw_lbl=False, noupdate=True)
        inter.append_plot(show_func, pnum=pnum, ishow_func=ishow_func)
        #inter.append_plot(cm_vsone.ishow_single_annotmatch(vsone_qreq_, aid2=aid2, draw_lbl=False, noupdate=True), pnum=pnum)
        #,  pnum=pnum)
        #cm_vsone.show_single_annotmatch(vsone_qreq_, daid=aid2, draw_lbl=False,
        #                                fnum=fnum, pnum=pnum)

    for idx in idxs:
        aid = aids[idx]
        pnum = (len(aids), len(aids), (idx * len(aids) + idx) + 1)
        inter.append_plot(wrap_show_func(ibeis.viz.show_chip, ibs, aid, qreq_=qreq_, draw_lbl=False, nokpts=True), pnum=pnum)
        #ibeis.viz.show_chip(ibs, aid, qreq_=qreq_,
        #                    draw_lbl=False, nokpts=True,
        #                    fnum=fnum, pnum=pnum)
    inter.show_page()


def vsone_independant_pair_hack(ibs, aid1, aid2, qreq_=None):
    r""" simple hack convinience func
    Uses vsmany qreq to build a "similar" vsone qreq

    TODO:
        in the context menu let me change preferences for running vsone

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --exec-vsone_independant_pair_hack --show --db PZ_MTEST
        python -m ibeis.algo.hots.vsone_pipeline --exec-vsone_independant_pair_hack --show --qaid=1 --daid=4
        --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.testdata_qreq_(defaultdb='testdb1')
        >>> aid1 = ut.get_argval('--qaid', default=1)
        >>> aid2 = ut.get_argval('--daid', default=2)
        >>> result = vsone_independant_pair_hack(qreq_.ibs, aid1, aid2, qreq_)
        >>> print(result)
        >>> ibs = qreq_.ibs
        >>> ut.show_if_requested()
    """
    cfgdict = dict(codename='vsone', fg_on=False)
    # FIXME: update this cfgdict a little better
    if qreq_ is not None:
        cfgdict.update(**qreq_.extern_data_config2.hesaff_params)
    vsone_qreq_ = ibs.new_query_request([aid1], [aid2], cfgdict=cfgdict)
    cm_vsone = vsone_qreq_.execute()[0]
    cm_vsone = cm_vsone.extend_results(vsone_qreq_)
    #cm_vsone = chip_match.ChipMatch.from_qres(vsone_qres)
    #cm_vsone.ishow_analysis(vsone_qreq_)
    cm_vsone.ishow_single_annotmatch(vsone_qreq_, aid2=aid2)
    #qres_vsone.ishow_analysis(ibs=ibs)
    #rchip_fpath1, rchip_fpath2 = ibs.get_annot_chip_fpath([aid1, aid2])
    #matches, metadata = vt.matching.vsone_image_fpath_matching(rchip_fpath1, rchip_fpath2)
    #vt.matching.show_matching_dict(matches, metadata)


def vsone_independant(qreq_):
    r"""
    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        ./dev.py -t custom --db PZ_Master0 --allgt --species=zebra_plains

        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_independant --show

        python -m ibeis.control.manual_annot_funcs --test-get_annot_groundtruth:0 --db=PZ_Master0 --aids=117 --exec-mode  # NOQA

        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_independant --qaid_list=97 --daid_list=all --db PZ_Master0 --species=zebra_plains
        python -m ibeis.viz.viz_name --test-show_multiple_chips --show --db PZ_Master0 --aids=118,117

        python -m ibeis.algo.hots.pipeline --test-request_ibeis_query_L0:0 --show --db PZ_Master0 --qaid_list=97 --daid-list=4813,4815
        --daid_list=all

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots import _pipeline_helpers as plh
        >>> cfgdict = dict(pipeline_root='vsone', codename='vsone', fg_on=False)
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict)
        >>> ibs, qreq_ = ibeis.testdata_qreq_(p=p, qaid_override=[1], qaid_override=[2, 5])
        >>> result = vsone_independant(qreq_)
        >>> print(result)
    """
    # test script
    ibs = qreq_.ibs
    # ./dev.py --db lewa_grevys --cmd
    # Get matches that have some property (scenerymatch)
    # HACK TEST SCRIPT

    #def get_scenery_match_aid_pairs(ibs):
    #    import utool as ut
    #    annotmatch_rowids = ibs._get_all_annotmatch_rowids()
    #    flags = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowids)
    #    annotmatch_rowids_ = ut.compress(annotmatch_rowids, flags)
    #    aid1_list = ibs.get_annotmatch_aid1(annotmatch_rowids_)
    #    aid2_list = ibs.get_annotmatch_aid2(annotmatch_rowids_)
    #    aid_pair_list = list(zip(aid1_list, aid2_list))
    #    return aid_pair_list

    #aid_pair_list = get_scenery_match_aid_pairs(ibs)
    #aid1, aid2 = aid_pair_list[0]

    qaids = qreq_.get_internal_qaids()
    daids = qreq_.get_internal_daids()
    import vtool as vt

    info_list = []
    #for aid1, aid2 in ut.ProgressIter(aid_pair_list):
    for aid1 in qaids:
        for aid2 in daids:
            rchip_fpath1, rchip_fpath2 = ibs.get_annot_chip_fpath([aid1, aid2])
            matches, metadata = vt.matching.vsone_image_fpath_matching(rchip_fpath1, rchip_fpath2)
            info_list.append((matches, metadata))
            print('So Far')
            print(sum([matches['RAT+SV'][0].shape[0] for (matches, metadata) in info_list]))  # NOQA
    # nnp_master had 15621 background examples

    for matches, metadata in info_list:
        vt.matching.show_matching_dict(matches, metadata)


def show_post_vsmany_vser():
    r""" TESTFUNC just show the input data

    CommandLine:
        python -m ibeis show_post_vsmany_vser --show --homog
        python -m ibeis show_post_vsmany_vser --show --csum --homog

    Example:
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> show_post_vsmany_vser()
    """
    import plottool as pt
    ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
    # HACK TO PRESCORE
    prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    show_all_ranked_matches(qreq_, cm_list_SVER, figtitle='vsmany post sver')
    pt.show_if_requested()


def get_normalized_score_column(fsv, colx, min_, max_, power):
    fs = fsv.T[colx].T.copy()
    fs = fs if min_ == 0 and max_ == 1 else vt.clipnorm(fs, min_, max_, out=fs)
    fs = fs if power == 1 else np.power(fs, power, out=fs)
    return fs


def prepare_vsmany_chipmatch(qreq_, cm_list_SVER):
    """ gets normalized vsmany priors

    Example:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
        >>> prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    """
    # Hack: populate aid2 score field in cmtup_old using prescore
    # grab normalized lnbnn scores
    fs_lnbnn_min   = qreq_.qparams.fs_lnbnn_min
    fs_lnbnn_max   = qreq_.qparams.fs_lnbnn_max
    fs_lnbnn_power = qreq_.qparams.fs_lnbnn_power
    _args = (fs_lnbnn_min, fs_lnbnn_max, fs_lnbnn_power)
    for cm in cm_list_SVER:
        lnbnn_index    = cm.fsv_col_lbls.index('lnbnn')
        vsmany_fs_list = [
            get_normalized_score_column(vsmany_fsv, lnbnn_index, *_args)
            for vsmany_fsv in cm.fsv_list]
        cm.fsv_list = matching.ensure_fsv_list(vsmany_fs_list)
        cm.fs_list = vsmany_fs_list


#@profile
def vsone_reranking(qreq_, cm_list_SVER, verbose=False):
    r"""
    Driver function

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show

        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show

        python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking
        utprof.py -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking

    Example:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
        >>> print(qreq_.qparams.rrvsone_cfgstr)
        >>> # cm_list_SVER = ut.dict_subset(cm_list_SVER, [6])
        >>> cm_list_VSONE = vsone_reranking(qreq_, cm_list_SVER)
        >>> #cm_list = cm_list_VSONE
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> figtitle = 'FIXME USE SUBSET OF CFGDICT'
        >>> # ut.dict_str(rrvsone_cfgdict, newlines=False)
        >>> show_all_ranked_matches(qreq_, cm_list_VSONE, figtitle=figtitle)
        >>> pt.show_if_requested()
    """
    config = qreq_.qparams
    # Filter down to a shortlist
    nNameShortlist = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName  = qreq_.qparams.nAnnotPerNameVsone
    scoring.score_chipmatch_list(qreq_, cm_list_SVER, 'nsum')
    prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    cm_shortlist = scoring.make_chipmatch_shortlists(qreq_, cm_list_SVER,
                                                     nNameShortlist,
                                                     nAnnotPerName)
    # Execute vsone reranking
    _prog = functools.partial(ut.ProgressIter, nTotal=len(cm_shortlist),
                              lbl='VSONE RERANKING', freq=1)
    cm_list_VSONE = [
        single_vsone_rerank(qreq_, prior_cm, config)
        for prior_cm in _prog(cm_shortlist)
    ]
    return cm_list_VSONE


def extract_aligned_parts(ibs, qaid, daid, qreq_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        daid (?):
        H1 (?):
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m ibeis extract_aligned_parts:0 --show --db testdb1
        python -m ibeis extract_aligned_parts:1 --show
        python -m ibeis extract_aligned_parts:1 --show  -t default:AI=False

    Ipy:
        ibs.get_annot_chip_fpath([qaid, daid])

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_FlankHack')
        >>> #nid_list = ibs.get_valid_nids(min_pername=2)
        >>> #nid_list = nid_list[1:2]
        >>> #qaid, daid = ibs.get_name_aids(nid_list)[0][0:2]
        >>> qaid, daid = ibs.get_valid_aids()[0:2]
        >>> qreq_ = None
        >>> matches, metadata = extract_aligned_parts(ibs, qaid, daid, qreq_)
        >>> rchip1_crop, rchip2_crop = metadata['rchip1_crop'], metadata['rchip2_crop']
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> #pt.imshow(vt.stack_images(rchip1_, rchip2)[0])
        >>> pt.figure(doclf=True)
        >>> blend = vt.blend_images(rchip1_crop, rchip2_crop)
        >>> vt.show_matching_dict(matches, metadata, fnum=1, mode=1)
        >>> stack = vt.stack_images(rchip1_crop, rchip2_crop)[0]
        >>> pt.imshow(stack, pnum=(1, 2, 1), fnum=2)
        >>> pt.imshow(blend, pnum=(1, 2, 2), fnum=2)[0]
        >>> ut.show_if_requested()

    Example:
        >>> # SCRIPT
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_FlankHack')
        >>> nid_list = ibs.get_valid_nids(min_pername=2)
        >>> pcfgdict = ibeis.main_helpers.testdata_pipecfg()
        >>> import plottool as pt
        >>> custom_actions = [
        >>>     ('present', ['s'], 'present', pt.present),
        >>> ]
        >>> for nid in ut.InteractiveIter(nid_list, custom_actions=custom_actions):
        >>>     nid_list = nid_list[1:2]
        >>>     qaid, daid = ibs.get_name_aids(nid)[0:2]
        >>>     qreq_ = ibs.new_query_request([qaid], [daid], cfgdict=pcfgdict)
        >>>     matches, metadata = extract_aligned_parts(ibs, qaid, daid, qreq_)
        >>>     if matches['RAT+SV'][0].shape[0] < 4:
        >>>         print('Not enough matches')
        >>>         continue
        >>>     rchip1_crop, rchip2_crop = metadata['rchip1_crop'], metadata['rchip2_crop']
        >>>     vt.inspect_matches.show_matching_dict(matches, metadata, fnum=1, mode=1)
        >>>     blend = vt.blend_images(rchip1_crop, rchip2_crop)
        >>>     pt.imshow(vt.stack_images(rchip1_crop, rchip2_crop)[0], pnum=(1, 2, 1), fnum=2)
        >>>     pt.imshow(blend, pnum=(1, 2, 2), fnum=2)[0]
        >>>     pt.draw()
    """
    if qreq_ is None:
        qreq_ = ibs.new_query_request([qaid], [daid])
    matches, metadata = vsone_single(qaid, daid, qreq_)
    rchip1 = metadata['annot1']['rchip']
    rchip2 = metadata['annot2']['rchip']
    H1 = metadata['H_RAT']
    import vtool as vt
    wh2 = vt.get_size(rchip2)
    rchip1_ = vt.warpHomog(rchip1, H1, wh2) if H1 is not None else rchip1
    isfill = vt.get_pixel_dist(rchip1_, np.array([0, 0, 0])) == 0
    rowslice, colslice = vt.get_crop_slices(isfill)
    # remove parts of the image that cannot match
    rchip1_crop = rchip1_[rowslice, colslice]
    rchip2_crop = rchip2[rowslice, colslice]
    metadata['annot1']['rchip_crop'] = rchip1_crop
    metadata['annot2']['rchip_crop'] = rchip2_crop
    return matches, metadata


def unsupervised_similarity(ibs, aids):
    """
    http://repository.upenn.edu/cgi/viewcontent.cgi?article=1101&context=cis_papers

    Ignore:
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> #ibs, aids = ibeis.testdata_aids('wd_peter2', 'timectrl:pername=2,view=left,view_ext=1,exclude_reference=True')
        >>> ibs, aids = ibeis.testdata_aids('wd_peter2', 'timectrl:pername=2,view=left,view_ext=0,exclude_reference=True')
        >>> ibs, aids = ibeis.testdata_aids('PZ_MTEST', 'timectrl:pername=4,view=left,view_ext=0,exclude_reference=True')
    """
    ut.ensure_pylab_qt4()

    # qreq_ = ibs.new_query_request(aids, aids, cfgdict=dict(affine_invariance=False))
    # qreq_ = ibs.new_query_request(aids, aids, cfgdict=dict())
    qreq_ = ibs.new_query_request(aids, aids, cfgdict=dict(affine_invariance=True, augment_queryside_hack=True))

    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_query_config2
    qaid, daid = aids[2:4]

    # [0:8]
    aids = aids
    #aids = aids[0:6]
    nids = np.array(ibs.get_annot_nids(aids))

    pair_dict = ut.ddict(dict)

    #score_mat = np.full((len(aids), len(aids)), np.inf)
    idx_list = list(ut.self_prodx(range(len(aids))))
    #idx_list = list(ut.iprod(range(len(aids)), range(len(aids))))

    cfgdict = {'ratio_thresh': 1.0, 'sver_xy_thresh': .001, 'scale_thresh': 1.5}
    cfgdict = {'ratio_thresh': 1.0, 'sver_xy_thresh': .001, 'scale_thresh': 1.5, 'symmetric': False}
    verbose = None

    qx, dx = idx_list[0]

    for qx, dx in ut.ProgressIter(idx_list):
        qaid, daid = ut.take(aids, [qx, dx])
        metadata = ibs.get_annot_pair_lazy_dict(qaid, daid, qconfig2_, dconfig2_)
        flann_params = {'algorithm': 'kdtree', 'trees': 8}
        # truth = (nids[qx] == nids[dx])
        metadata['annot1']['flann'] = vt.flann_cache(metadata['annot1']['vecs'], flann_params=flann_params)
        # metadata['flann2'] = vt.flann_cache(metadata['vecs2'], flann_params=flann_params)
        match = vt.vsone_matching(metadata, cfgdict=cfgdict, verbose=verbose)
        pair_dict[(qx, dx)] = match

    from skimage.future import graph
    g = graph.rag.RAG()
    score_mat = np.zeros((len(aids), len(aids)))
    # Populate weight matrix off-diagonal
    for qx, dx in ut.ProgressIter(idx_list):
        qaid, daid = ut.take(aids, [qx, dx])
        match = pair_dict[(qx, dx)]
        print(match)
        score = match.matches['TOP+SV'].fs.sum()
        score_mat[qx, dx] = score
        g.add_edge(qx, dx, weight=score)
        (nids[qx] == nids[dx])
    # Populate weight matrix on-diagonal
    W = score_mat
    D = np.diag(W.sum(axis=0))  # NOQA

    from plottool.interactions import ExpandableInteraction
    import plottool as pt
    import networkx as nx
    if False:
        pos = nx.circular_layout(g)
        pt.figure()
        nx.draw(g, pos)
        nx.draw_networkx_edge_labels(g, pos, font_size=10)

    if True:
        # NORMALIZED CUT STUFF
        # https://github.com/IAS-ZHAW/machine_learning_scripts/blob/master/mlscripts/ml/normalized_min_cut.py

        #labels = np.arange(len(aids))
        #graph.cut_normalized(labels, g)
        m_adjacency = np.array(nx.to_numpy_matrix(g))
        D = np.diag(np.sum(m_adjacency, 0))
        D_half_inv = np.diag(1.0 / np.sqrt(np.sum(m_adjacency, 0)))
        M = np.dot(D_half_inv, np.dot((D - m_adjacency), D_half_inv))
        (w, v) = np.linalg.eig(M)
        #find index of second smallest eigenvalue
        index = np.argsort(w)[1]
        v_partition = v[:, index]
        v_partition = np.sign(v_partition)

        import ibeis
        flags = v_partition > 0
        cluster_aids = ut.compress(aids, flags)
        ibeis.viz.viz_chip.show_many_chips(ibs, cluster_aids)
        remain_aids = ut.compress(aids, ut.not_list(flags))

        for i in range(len(set(nids))):
            remain_m = m_adjacency.compress(~flags, axis=0).compress(~flags, axis=1)
            D = np.diag(np.sum(remain_m, 0))
            D_half_inv = np.diag(1.0 / np.sqrt(np.sum(remain_m, 0)))
            M = np.dot(D_half_inv, np.dot((D - remain_m), D_half_inv))
            (w, v) = np.linalg.eig(M)
            #find index of second smallest eigenvalue
            index = np.argsort(w)[1]
            v_partition = v[:, index]
            v_partition = np.sign(v_partition)

            flags = v_partition > 0
            cluster_aids = ut.compress(remain_aids, flags)
            ibeis.viz.viz_chip.show_many_chips(ibs, cluster_aids)
            remain_aids = ut.compress(remain_aids, ut.not_list(flags))
            pass

    #diag_idx = np.diag_indices_from(score_mat)
    #score_mat[diag_idx] = score_mat.sum(axis=0)

    pos = np.meshgrid(np.arange(len(aids)), np.arange(len(aids)))
    gt_scores = score_mat[nids[pos[1]] == nids[pos[0]]]
    min_gt_score = gt_scores.min()

    fnum = 1
    #_pnumiter = pt.make_pnum_nextgen(nSubplots=score_mat.size)
    nRows, nCols = pt.get_square_row_cols(score_mat.size, fix=True)
    ut.ensure_pylab_qt4()
    inter = ExpandableInteraction(fnum, nRows=nRows, nCols=nCols)

    # Show matches in off-diagonal
    for qx, dx in ut.ProgressIter(idx_list):
        match = pair_dict[(qx, dx)]
        score = score_mat[qx, dx]
        match_inter = match.make_interaction(mode=1, title='%.2f' % (score,),
                                             truth=(nids[qx] == nids[dx]))
        if score >= min_gt_score:
            inter.append_plot(match_inter, px=(qx, dx))
            #inter.append_plot(ut.partial(pt.imshow_null, msg=(('qx=%r, dx=%r') % (qx, dx))), px=(qx, dx))

    # Show chips in diagonal
    for qx in range(len(aids)):
        qaid = aids[qx]
        chip = ibs.get_annot_chips(qaid)
        inter.append_plot(ut.partial(pt.imshow, chip), px=(qx, qx))

    inter.start()
    #pt.plot_score_histograms([fx2_to_dist.T[0]])
    #import plottool as pt
    #pt.plot_score_histograms([m.matches['TOP+SV'][1]])

    sym_score_mat = np.maximum(score_mat, score_mat.T)
    #gt_scores = sym_score_mat[nids[pos[1]] == nids[pos[0]]]
    #gf_scores = sym_score_mat[nids[pos[1]] != nids[pos[0]]]

    cost_matrix = sym_score_mat - 50
    labels = vt.unsupervised_multicut_labeling(cost_matrix)
    print('labels = %r' % (labels,))

    import hdbscan
    alg = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=1, p=1, gen_min_span_tree=1, min_samples=2)
    alg.fit_predict((1 / score_mat))
    #import pandas
    #score_mat = pandas.DataFrame(pairs).as_matrix()

    #nids = np.array(nids)
    #[np.array(idx_list)]
    #score_mat[]

    #for qx, dx in ut.ProgressIter(idx_list):
    #    match = pair_dict[(qx, dx)]
    #    match.show()


def marge_matches_lists(fmfs_A, fmfs_B):
    if fmfs_A is None:
        return fmfs_B
    fm_A_list, fsv_A_list = fmfs_A
    fm_B_list, fsv_B_list = fmfs_B
    fm_merge_list = []
    fsv_merge_list = []
    fsv_A_list = matching.ensure_fsv_list(fsv_A_list)
    fsv_B_list = matching.ensure_fsv_list(fsv_B_list)
    for fm_A, fm_B, fsv_A, fsv_B in zip(fm_A_list, fm_B_list, fsv_A_list, fsv_B_list):
        fm_merged, fs_merged = matching.marge_matches(fm_A, fm_B, fsv_A, fsv_B)
        fm_merge_list.append(fm_merged)
        fsv_merge_list.append(fs_merged)
    fmfs_merge = fm_merge_list, fsv_merge_list
    return fmfs_merge


def get_selectivity_score_list(qreq_, qaid, daid_list, fm_list, cos_power):
    vecs1 = qreq_.ibs.get_annot_vecs(qaid, config2_=qreq_.extern_query_config2)
    vecs2_list = qreq_.ibs.get_annot_vecs(daid_list, config2_=qreq_.extern_data_config2)
    vecs1_m_iter = (vecs1.take(fm.T[0], axis=0) for fm in fm_list)
    vecs2_m_iter = (vecs2.take(fm.T[1], axis=0) for fm, vecs2 in zip(fm_list, vecs2_list))
    # Rescore constrained using selectivity function
    fs_list = [scoring.sift_selectivity_score(vecs1_m, vecs2_m, cos_power)
               for vecs1_m, vecs2_m in zip(vecs1_m_iter, vecs2_m_iter)]
    return fs_list


@profile
def sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config={}):
    from vtool import spatial_verification as sver
    # params
    # TODO: rectify with sver_single_chipmatch
    # TODO paramaterize better
    xy_thresh    = config.get('xy_thresh') * 1.5
    scale_thresh = config.get('scale_thresh') * 2
    ori_thresh   = config.get('ori_thresh') * 2
    min_nInliers = config.get('min_nInliers')
    # input data
    fm_list, fs_list = fmfs_merge
    fsv_list   = matching.ensure_fsv_list(fs_list)
    kpts1      = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.extern_query_config2)
    kpts2_list = qreq_.ibs.get_annot_kpts(daid_list, config2_=qreq_.extern_data_config2)
    chip2_dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlensqrd(
        daid_list, config2_=qreq_.extern_data_config2)
    # chip diagonal length
    res_list = []
    # homog_inliers
    for kpts2, chip2_dlen_sqrd, fm, fsv in zip(kpts2_list, chip2_dlen_sqrd_list, fm_list, fsv_list):
        match_weights = np.ones(len(fm))
        sv_tup = sver.spatially_verify_kpts(
            kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
            chip2_dlen_sqrd, min_nInliers,
            returnAff=False, match_weights=match_weights)
        if sv_tup is not None:
            (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = sv_tup
            fm_SV = fm.take(homog_inliers, axis=0)
            fsv_SV = fsv.take(homog_inliers, axis=0)
        else:
            fm_SV = np.empty((0, 2), dtype=hstypes.FM_DTYPE)
            fsv_SV = np.empty((0, fsv.shape[1]))
            H = np.eye(3)
        res_list.append((fm_SV, fsv_SV, H))

    fm_list_SV  = ut.get_list_column(res_list, 0)
    fsv_list_SV = ut.get_list_column(res_list, 1)
    H_list      = ut.get_list_column(res_list, 2)
    fmfs_merge_SV = (fm_list_SV, fsv_list_SV)
    return fmfs_merge_SV, H_list


@profile
def refine_matches(qreq_, prior_cm, config={}):
    r"""
    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-refine_matches --show
        python -m ibeis.algo.hots.vsone_pipeline --test-refine_matches --show --homog
        python -m ibeis.algo.hots.vsone_pipeline --test-refine_matches --show --homog --sver_unconstrained
        python -m ibeis.algo.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained&
        python -m ibeis.algo.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained --sver_unconstrained&

        # CONTROLLED EXAMPLES
        python -m ibeis.algo.hots.vsone_pipeline --exec-refine_matches --show --qaid 1801 --controlled_daids --db PZ_Master0 --sv_on=False --present

        # WITH DEV HARNESS
        python dev.py -t custom:rrvsone_on=True --allgt --index 0:40 --db PZ_MTEST --print-confusion-stats --print-scorediff-mat-stats
        python dev.py -t custom:rrvsone_on=True custom --allgt --index 0:40 --db PZ_MTEST --print-confusion-stats --print-scorediff-mat-stats

        python dev.py -t custom:rrvsone_on=True,constrained_coeff=0 custom --qaid 12 --db PZ_MTEST \
            --print-confusion-stats --print-scorediff-mat-stats --show --va

        python dev.py -t custom:rrvsone_on=True,constrained_coeff=0,maskscore_mode=kpts --qaid 12 --db PZ_MTEST  \
            --print-confusion-stats --print-scorediff-mat-stats --show --va

        python dev.py -t custom:rrvsone_on=True,maskscore_mode=kpts --qaid 12 --db PZ_MTEST \
                --print-confusion-stats --print-scorediff-mat-stats --show --va


        use_kptscov_scoring

    Example1:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching('PZ_MTEST')
        >>> config = qreq_.qparams
        >>> unscored_cm = refine_matches(qreq_, prior_cm, config)
        >>> unscored_cm.print_csv(ibs=ibs)
        >>> prior_cm.print_csv(ibs=ibs)
        >>> ut.quit_if_noshow()
        >>> prior_cm.show_ranked_matches(qreq_, figtitle=qreq_.qparams.query_cfgstr)
        >>> ut.show_if_requested()
    """
    # THIS CAUSES THE ISSUE
    #prior_cm.fs_list = prior_cm.fsv_list
    #return prior_cm
    if qreq_.ibs.get_annot_num_feats(prior_cm.qaid, config2_=qreq_.qparams) == 0:
        num_daids = len(prior_cm.daid_list)
        empty_unscored_cm = chip_match.ChipMatch.from_unscored(
            prior_cm, ut.alloc_lists(num_daids), ut.alloc_lists(num_daids),
            ut.alloc_lists(num_daids))
        return empty_unscored_cm

    prior_coeff         = config.get('prior_coeff')
    unconstrained_coeff = config.get('unconstrained_coeff')
    constrained_coeff   = config.get('constrained_coeff')
    sver_unconstrained  = config.get('sver_unconstrained')
    sver_constrained    = config.get('sver_constrained')
    # TODO: Param
    scr_cos_power      = 3.0
    #
    qaid           = prior_cm.qaid
    daid_list      = prior_cm.daid_list
    fm_prior_list  = prior_cm.fm_list
    fsv_prior_list = prior_cm.fsv_list
    H_prior_list   = prior_cm.H_list
    H_list         = H_prior_list

    assert unconstrained_coeff is not None, '%r' % (unconstrained_coeff,)

    col_coeff_list = []
    fmfs_merge = None

    if prior_coeff != 0:
        # Merge into result
        col_coeff_list.append(prior_coeff)
        fmfs_prior = (fm_prior_list, fsv_prior_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_prior)

    if unconstrained_coeff != 0:
        col_coeff_list.append(unconstrained_coeff)
        unc_match_results = compute_query_unconstrained_matches(qreq_, qaid, daid_list, config)
        fm_unc_list, fs_unc_list, fm_norm_unc_list = unc_match_results
        # Merge into result
        fmfs_unc = (fm_unc_list, fs_unc_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_unc)

        # We have the option of spatially verifying the merged results from the
        # prior and the new unconstrained matches.
        if sver_unconstrained:
            fmfs_merge, H_list = sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config)

    if constrained_coeff != 0:
        scr_match_results = compute_query_constrained_matches(qreq_, qaid,
                                                              daid_list,
                                                              H_list, config)
        fm_scr_list, fs_scr_list, fm_norm_scr_list = scr_match_results
        fs_scr_list = get_selectivity_score_list(qreq_, qaid, daid_list, fm_scr_list, scr_cos_power)
        # Merge into result
        fmfs_scr = (fm_scr_list, fs_scr_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_scr)
        col_coeff_list.append(constrained_coeff)

        # Another optional round of spatial verification
        if sver_constrained:
            fmfs_merge, H_list = sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config)

    coeffs = np.array(col_coeff_list)
    assert np.isclose(coeffs.sum(), 1.0), 'must sum to 1 coeffs = %r' % (coeffs)
    # merge different match types
    fm_list, fsv_list = fmfs_merge
    # apply linear combination
    fs_list = [(np.nan_to_num(fsv) * coeffs[None, :]).sum(axis=1) for fsv in fsv_list]

    unscored_cm = chip_match.ChipMatch.from_unscored(prior_cm, fm_list, fs_list, H_list)
    return unscored_cm


@profile
def single_vsone_rerank(qreq_, prior_cm, config={}):
    r"""
    Runs a single vsone-pair (query, daid_list)

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank --show
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank --show --qaid 18
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank --show --qaid 18
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank --show --qaid 1801 --db PZ_Master0 --controlled --verb-testdata
        python -m ibeis.algo.hots.vsone_pipeline --test-single_vsone_rerank --show --qaid 1801 --controlled_daids --db PZ_Master0 --verb-testdata

        python -m ibeis.algo.hots.vsone_pipeline --exec-single_vsone_rerank --show --qaid 1801 --controlled_daids --db PZ_Master0 --verb-testdata
        python -m ibeis.algo.hots.vsone_pipeline --exec-single_vsone_rerank --show --qaid 1801 --controlled_daids --db PZ_Master0 --verb-testdata --sv_on=False --present
        python -m ibeis.algo.hots.vsone_pipeline --exec-single_vsone_rerank --show --qaid 1801 --controlled_daids --db PZ_Master0 --verb-testdata --sv_on=False --present --affine-invariance=False
        python -m ibeis.algo.hots.vsone_pipeline --exec-single_vsone_rerank --show --qaid 1801 --controlled_daids --db PZ_Master0 --verb-testdata --sv_on=False --present --affine-invariance=False --rotation-invariant=True

    Example1:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> import plottool as pt
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching('PZ_MTEST')
        >>> config = qreq_.qparams
        >>> rerank_cm = single_vsone_rerank(qreq_, prior_cm, config)
        >>> #rerank_cm.print_rawinfostr()
        >>> rerank_cm.print_csv()
        >>> print(rerank_cm.score_list)
        >>> ut.quit_if_noshow()
        >>> prior_cm.score_nsum(qreq_)
        >>> prior_cm.show_ranked_matches(qreq_, fnum=1, figtitle='prior')
        >>> rerank_cm.show_ranked_matches(qreq_, fnum=2, figtitle='rerank')
        >>> pt.show_if_requested()
    """
    #print('==================')
    unscored_cm = refine_matches(qreq_, prior_cm, config)

    if qreq_.qparams.covscore_on:
        unscored_cm.score_coverage(qreq_)
    else:
        # Apply score weights
        data_baseline_weight_list = scoring.get_annot_kpts_baseline_weights(
            qreq_.ibs, unscored_cm.daid_list,
            config2_=qreq_.extern_data_config2, config=config)
        query_baseline_weight = scoring.get_annot_kpts_baseline_weights(
            qreq_.ibs, [unscored_cm.qaid],
            config2_=qreq_.extern_query_config2, config=config)[0]
        qfx_list = [fm.T[0] for fm in unscored_cm.fm_list]
        dfx_list = [fm.T[1] for fm in unscored_cm.fm_list]

        qfweight_list = [query_baseline_weight.take(qfx) for qfx in qfx_list]
        dfweight_list = [data_baseline_weight.take(dfx)
                         for dfx, data_baseline_weight in zip(dfx_list, data_baseline_weight_list)]
        fweight_list = [np.sqrt(qfweight * dfweight) for qfweight, dfweight in
                        zip(qfweight_list, dfweight_list)]
        # hack in the distinctivness and fgweights
        unscored_cm.fs_list = [
            fs * fweight
            for fs, fweight in zip(unscored_cm.fs_list, fweight_list)]
        unscored_cm.fsv_list = matching.ensure_fsv_list(unscored_cm.fs_list)

        #queryside_weights =
        #dfweights_list =
        # hack
        unscored_cm.score_nsum(qreq_)

    # Convert our one score to a score vector here
    rerank_cm = unscored_cm
    rerank_cm.fsv_list = matching.ensure_fsv_list(rerank_cm.fs_list)
    return rerank_cm


def quick_vsone_flann(flann_cachedir, qvecs):
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    use_cache = save = True
    flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params,
                           quiet=True, verbose=False, use_cache=use_cache, save=save)
    return flann


@profile
def compute_query_unconstrained_matches(qreq_, qaid, daid_list, config):
    r"""

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm --homog

    Example1:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching()
        >>> config = qreq_.qparams
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_unconstrained_matches(qreq_, qaid, daid_list, config)
        >>> fm_RAT_list, fs_RAT_list, fm_norm_RAT_list = match_results
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> idx = ut.listfind(ibs.get_annot_nids(daid_list).tolist(), ibs.get_annot_nids(qaid))
        >>> args = (ibs, qaid, daid_list, fm_RAT_list, fs_RAT_list, fm_norm_RAT_list, H_list)
        >>> show_single_match(*args, index=idx)
        >>> pt.set_title('unconstrained')
        >>> pt.show_if_requested()
    """
    unc_ratio_thresh = config['unc_ratio_thresh']
    #, .625)
    qvecs = qreq_.ibs.get_annot_vecs(qaid, config2_=qreq_.extern_query_config2)
    dvecs_list = qreq_.ibs.get_annot_vecs(daid_list, config2_=qreq_.extern_data_config2)
    #print(len(qvecs))
    flann = quick_vsone_flann(qreq_.ibs.get_flann_cachedir(), qvecs)
    rat_kwargs = {
        'unc_ratio_thresh' : unc_ratio_thresh,
        'fm_dtype'     : hstypes.FM_DTYPE,
        'fs_dtype'     : hstypes.FS_DTYPE,
    }
    #print('rat_kwargs = ' + ut.dict_str(rat_kwargs))
    scrtup_list = [
        matching.unconstrained_ratio_match(
            flann, vecs2, **rat_kwargs)
        for vecs2 in ut.ProgIter(dvecs_list, lbl='unconstrained matching', adjust=True, time_thresh=7)
    ]
    fm_RAT_list = ut.get_list_column(scrtup_list, 0)
    fs_RAT_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_RAT_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_RAT_list, fs_RAT_list, fm_norm_RAT_list
    return match_results


@profile
def compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, config):
    r"""

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm --homog
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show --homog
        python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show --homog --index 2

    Example1:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching()
        >>> config = qreq_.qparams
        >>> print(config.query_cfgstr)
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, config)
        >>> fm_SCR_list, fs_SCR_list, fm_norm_SCR_list = match_results
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> idx = ut.listfind(ibs.get_annot_nids(daid_list), ibs.get_annot_nids(qaid))
        >>> index = ut.get_argval('--index', int, idx)
        >>> args = (ibs, qaid, daid_list, fm_SCR_list, fs_SCR_list, fm_norm_SCR_list, H_list)
        >>> show_single_match(*args, index=index)
        >>> pt.set_title('constrained')
        >>> pt.show_if_requested()
    """
    scr_ratio_thresh     = config.get('scr_ratio_thresh', .1)
    scr_K                = config.get('scr_K', 7)
    scr_match_xy_thresh  = config.get('scr_match_xy_thresh', .05)
    scr_norm_xy_min      = config.get('scr_norm_xy_min', 0.1)
    scr_norm_xy_max      = config.get('scr_norm_xy_max', 1.0)
    scr_norm_xy_bounds = (scr_norm_xy_min, scr_norm_xy_max)
    vecs1 = qreq_.ibs.get_annot_vecs(qaid, config2_=qreq_.extern_query_config2)
    kpts1 = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.extern_query_config2)
    vecs2_list = qreq_.ibs.get_annot_vecs(daid_list, config2_=qreq_.extern_data_config2)
    kpts2_list = qreq_.ibs.get_annot_kpts(daid_list, config2_=qreq_.extern_data_config2)
    chip2_dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlensqrd(
        daid_list, config2_=qreq_.extern_data_config2)  # chip diagonal length
    # build flann for query vectors
    flann = quick_vsone_flann(qreq_.ibs.get_flann_cachedir(), vecs1)
    # match database chips to query chip
    scr_kwargs = {
        'scr_K'            : scr_K,
        'match_xy_thresh'  : scr_match_xy_thresh,
        'norm_xy_bounds'   : scr_norm_xy_bounds,
        'scr_ratio_thresh' : scr_ratio_thresh,
        'fm_dtype'         : hstypes.FM_DTYPE,
        'fs_dtype'         : hstypes.FS_DTYPE,
    }
    print('scr_kwargs = ' + ut.dict_str(scr_kwargs))
    # Homographys in H_list map image1 space into image2 space
    scrtup_list = [
        matching.spatially_constrained_ratio_match(
            flann, vecs2, kpts1, kpts2, H, chip2_dlen_sqrd, **scr_kwargs)
        for vecs2, kpts2, chip2_dlen_sqrd, H in
        zip(vecs2_list, kpts2_list, chip2_dlen_sqrd_list, H_list)]
    fm_SCR_list = ut.get_list_column(scrtup_list, 0)
    fs_SCR_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_SCR_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_SCR_list, fs_SCR_list, fm_norm_SCR_list
    return match_results


# -----------------------------
# GRIDSEARCH
# -----------------------------


COVKPTS_DEFAULT = vt.coverage_kpts.COVKPTS_DEFAULT
COVGRID_DEFAULT = vt.coverage_grid.COVGRID_DEFAULT

OTHER_RRVSONE_PARAMS = ut.ParamInfoList('OTHERRRVSONE', [
    #ut.ParamInfo('fs_lnbnn_min', .0001),
    #ut.ParamInfo('fs_lnbnn_max', .05),
    #ut.ParamInfo('fs_lnbnn_power', 1.0),
    ut.ParamInfo('fs_lnbnn_min', 0.0, hideif=0.0),
    ut.ParamInfo('fs_lnbnn_max', 1.0, hideif=1.0),
    ut.ParamInfo('fs_lnbnn_power', 1.0, hideif=1.0),
    ut.ParamInfoBool('covscore_on', False, hideif=lambda cfg: True),
    ut.ParamInfo('dcvs_on', False, hideif=False),
])


SHORTLIST_DEFAULTS = ut.ParamInfoList('SLIST', [
    ut.ParamInfo('nNameShortlistVsone', 20, 'nNm='),
    ut.ParamInfo('nAnnotPerNameVsone', 3, 'nApN='),
])

# matching types
COEFF_DEFAULTS = ut.ParamInfoList('COEFF', [
    ut.ParamInfo('prior_coeff', .6, 'prior_coeff='),
    ut.ParamInfo('unconstrained_coeff',    .4, 'unc_coeff='),
    ut.ParamInfo('constrained_coeff',     0.0, 'scr_coeff=', hideif=0.0),
    ut.ParamInfo('sver_unconstrained',   True, 'sver_unc=',
                 hideif=lambda cfg: cfg['unconstrained_coeff'] <= 0),
    ut.ParamInfo('sver_constrained',    False, 'sver_scr=',
                 hideif=lambda cfg: cfg['constrained_coeff'] <= 0),
    ut.ParamInfo('maskscore_mode', 'grid', 'cov=',
                 hideif=lambda cfg: not cfg['covscore_on']),
]
)

UNC_DEFAULTS = ut.ParamInfoList('UNC', [
    ut.ParamInfo('unc_ratio_thresh', .8, 'uncRat=', varyvals=[.625, .82, .9, 1.0, .8]),
])


def scr_constraint_func(cfg):
    if cfg['scr_norm_xy_min'] >= cfg['scr_norm_xy_max']:
        return False

SCR_DEFAULTS = ut.ParamInfoList('SCR', [
    ut.ParamInfo('scr_match_xy_thresh', .15, 'xy=',
                 varyvals=[.05, 1.0, .1], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_norm_xy_min', 0.1, '',
                 varyvals=[0, .1, .2], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_norm_xy_max', 1.0, '',
                 varyvals=[1, .3], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_ratio_thresh', .95, 'scrRat=',
                 varyvals=[.625, .3, .9, 0.0, 1.0], varyslice=slice(0, 1)),
    ut.ParamInfo('scr_K', 7, 'scK',
                 varyvals=[7, 2], varyslice=slice(0, 2)),
],
    scr_constraint_func,
    hideif=lambda cfg: cfg['constrained_coeff'] <= 0)


def gridsearch_single_vsone_rerank():
    r"""

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_single_vsone_rerank --show
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_single_vsone_rerank --show --testindex 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_single_vsone_rerank()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    ibs, qreq_, prior_cm = plh.testdata_matching()
    config = qreq_.qparams
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_ = dict(prescore_method='nsum', score_method='nsum', sver_output_weighting=True)
    cfgdict_['rrvsone_on'] = True
    # HACK TO GET THE DATA WE WANT WITHOUT UNNCESSARY COMPUTATION
    # Get pipeline testdata for this configuration

    p = 'default' + ut.get_cfg_lbl(cfgdict_)
    import ibeis
    ibs, qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', p=p, a=['default:qsize=1,mingt=2'])

    qaid_list = qreq_.qaids.tolist()
    qaid = qaid_list[0]
    daid_list = qreq_.ibs.get_annot_groundtruth(qaid)[0:1]
    #
    cfgdict_list, cfglbl_list = COEFF_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    assert len(cfgdict_list) > 0
    #config = qreq_.qparams
    cfgresult_list = [
        single_vsone_rerank(qreq_, prior_cm, config)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='rerank')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch(qaid=qaid, daid_list=daid_list,
                                   fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        inter = interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)
        inter.start()

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)
    pt.iup()


def gridsearch_constrained_matches():
    r"""
    Search spatially constrained matches

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --qaid 41
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --testindex 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> gridsearch_constrained_matches()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = SCR_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    #fname = None  # 'easy1.png'
    ibs, qreq_, prior_cm = plh.testdata_matching()
    qaid      = prior_cm.qaid
    daid_list = prior_cm.daid_list
    H_list    = prior_cm.H_list
    #config = qreq_.qparams
    cfgresult_list = [
        compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='scr match')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list,
                                 H_list=H_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch(qaid=qaid, daid_list=daid_list,
                                   fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        inter = interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)
        inter.start()

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)

    #if use_separate_norm:
    #    ut.interact_gridsearch_result_images(
    #        functools.partial(show_single_match, use_separate_norm=True),
    #        cfgdict_list, cfglbl_list,
    #        cfgresult_list, fnum=fnum + 1, figtitle='constrained ratio match', unpack=True,
    #        max_plots=25, scorelbl='sumscore')
    pt.iup()


def gridsearch_unconstrained_matches():
    r"""
    Search unconstrained ratio test vsone match

    This still works

    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 27
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 41 --daid_list 39
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 40 --daid_list 39
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --testindex 2


        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 117 --daid_list 118 --db PZ_Master0
        python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 117 --daid_list 118 --db PZ_Master0 --rotation_invariance

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_unconstrained_matches()
        >>> pt.show_if_requested()
    """
    import ibeis
    import plottool as pt
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_ = dict(prescore_method='nsum', score_method='nsum', sver_output_weighting=True)
    cfgdict_['rrvsone_on'] = True
    # HACK TO GET THE DATA WE WANT WITHOUT UNNCESSARY COMPUTATION
    # Get pipeline testdata for this configuration
    p = 'default' + ut.get_cfg_lbl(cfgdict_)
    ibs, qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', p=p, a=['default:qsize=1,mingt=2,dsize=1'])
    qaid_list = qreq_.qaids.tolist()
    qaid = qaid_list[0]
    daid_list = qreq_.get_external_query_groundtruth(qaid)[0:1]
    #
    cfgdict_list, cfglbl_list = UNC_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    assert len(cfgdict_list) > 0
    #config = qreq_.qparams
    cfgresult_list = [
        compute_query_unconstrained_matches(qreq_, qaid, daid_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='scr match')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch(qaid=qaid, daid_list=daid_list,
                                   fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        inter = interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)
        inter.start()

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)
    pt.iup()


# -----------------------------
# VISUALIZATIONS
# -----------------------------


def show_single_match(ibs, qaid, daid_list, fm_list, fs_list,
                      fm_norm_list=None, H_list=None, index=None, **kwargs):
    use_sameaxis_norm = ut.get_argflag('--shownorm')
    fs = fs_list[index]
    fm = fm_list[index]
    if use_sameaxis_norm:
        fm_norm = fm_norm_list[index]
    else:
        fm_norm = None
    kwargs['darken'] = .7
    daid = daid_list[index]
    if H_list is None:
        H1 = None
    else:
        H1 = H_list[index]
    #H1 = None  # uncomment to see warping
    show_matches(ibs, qaid, daid, fm, fs=fs, H1=H1, fm_norm=fm_norm, **kwargs)


def show_matches(ibs, qaid, daid, fm, fs=None, fm_norm=None,
                               H1=None, fnum=None, pnum=None, **kwargs):
    from ibeis.viz import viz_matches
    if not ut.get_argflag('--homog'):
        H1 = None

    viz_matches.show_matches2(ibs, qaid, daid, fm=fm, fs=fs, fm_norm=fm_norm, ori=True,
                              H1=H1, fnum=fnum, pnum=pnum, show_name=False, **kwargs)

    #pt.set_title('score = %.3f' % (score,))


def show_ranked_matches(ibs, cm, fnum=None):
    import plottool as pt
    qaid = cm.qaid
    if fnum is None:
        fnum = pt.next_fnum()
    CLIP_TOP = 6
    top_idx_list  = ut.listclip(cm.argsort(), CLIP_TOP)
    nRows, nCols  = pt.get_square_row_cols(len(top_idx_list), fix=True)
    next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
    for idx in top_idx_list:
        daid  = cm.daid_list[idx]
        fm    = cm.fm_list[idx]
        fsv   = cm.fsv_list[idx]
        fs    = fsv.prod(axis=1)
        H1 = cm.H_list[idx]
        pnum = next_pnum()
        #with ut.EmbedOnException():
        show_matches(ibs, qaid, daid, fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum)
        score = None if cm.score_list is None else cm.score_list[idx]
        if score is not None:
            pt.set_title('score = %.3f' % (score,))
        else:
            pt.set_title('score = %r' % (score,))


def show_all_ranked_matches(qreq_, cm_list, fnum_offset=0, figtitle=''):
    """ helper """
    import plottool as pt
    for fnum_, cm in enumerate(cm_list):
        #cm.foo()
        fnum = fnum_ + fnum_offset
        if True:
            #cm.show_ranked_matches(qreq_, fnum=fnum)
            cm.show_analysis(qreq_, fnum=fnum)
        else:
            show_ranked_matches(qreq_.ibs, cm, fnum)
            #pt.figure(fnum=fnum, doclf=True, docla=True)
            pt.set_figtitle('qaid=%r %s' % (cm.qaid, figtitle))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline
        python -m ibeis.algo.hots.vsone_pipeline --allexamples
        python -m ibeis.algo.hots.vsone_pipeline --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
