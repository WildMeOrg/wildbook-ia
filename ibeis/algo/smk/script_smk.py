from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
#from ibeis.algo.smk import match_chips5 as mc5
from ibeis.algo.smk import vocab_indexer
from ibeis.algo.smk import inverted_index
from ibeis.algo.smk import smk_funcs
from ibeis.algo.smk import smk_pipeline
#from ibeis.algo.smk import smk_funcs
#from ibeis.algo.smk import inverted_index
#from ibeis import core_annots
#from ibeis.algo import Config as old_config
(print, rrr, profile) = ut.inject2(__name__)


def load_internal_data():
    from ibeis.algo.smk.smk_pipeline import *  # NOQA
    import ibeis
    qreq_ = ibeis.testdata_qreq_(
        defaultdb='Oxford', a='oxford',
        p='smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True')
    cm_list = qreq_.execute()
    ave_precisions = [cm.get_annot_ave_precision() for cm in cm_list]
    mAP = np.mean(ave_precisions)
    print('mAP = %.3f' % (mAP,))
    cm = cm_list[-1]
    return qreq_, cm


def load_external_data1():
    """
    Such hacks for reading external oxford
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    from os.path import join, basename, splitext
    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath1 = join(dbdir, 'oxford_data1.pkl')
    if ut.checkpath(data_fpath1):
        oxford_data1 = ut.load_data(data_fpath1)
        return oxford_data1
    else:
        sift_fpath = join(dbdir, 'OxfordSIFTDescriptors', 'feat_oxc1_hesaff_sift.bin')
        readme_fpath = join(dbdir, 'README2.txt')
        word_dpath = join(dbdir, 'word_oxc1_hesaff_sift_16M_1M')
        word_fpath_list = ut.ls(word_dpath)
        import pandas as pd
        imgid_to_df = {}

        imgid_order = ut.readfrom(readme_fpath).split('\n')[20:-1]

        for word_fpath in ut.ProgIter(word_fpath_list, lbl='reading kpts'):
            imgid = splitext(basename(word_fpath))[0]
            #if imgid not in imgid_order[0:10]:
            #    continue
            row_gen = (map(float, line.strip('\n').split(' '))
                       for line in ut.read_lines_from(word_fpath)[2:])
            rows = [(int(word_id), x, y, a, c, d)
                    for (word_id, x, y, a, c, d) in row_gen]
            df = pd.DataFrame(rows, columns=['word_id', 'x', 'y', 'a', 'c', 'd'])
            imgid_to_df[imgid] = df

        df_list = ut.take(imgid_to_df, imgid_order)
        offset_list = [0] + ut.cumsum([len(df_) for df_ in df_list])
        try:
            shape = (16334970, 128)
            assert offset_list[-1] == shape[0]
            file_ = open(sift_fpath, 'rb')
            with ut.Timer('Reading file'):
                sifts = np.fromstring(file_.read(16334970 * 128), dtype=np.uint8)
            sifts = sifts.reshape(shape)
        finally:
            file_.close()

        vecs_list = [sifts[l:r] for l, r in ut.itertwo(offset_list)]
        kpts_list = [df_.loc[:, ('x', 'y', 'a', 'c', 'd')].values for df_ in df_list]
        wordid_list = [df_.loc[:, 'word_id'].values for df_ in df_list]

        import vtool as vt
        unique_word, groupxs = vt.group_indices(np.hstack(wordid_list))
        wx_to_sifts = vt.apply_grouping(sifts, groupxs, axis=0)

        wx_to_word = np.array([
            np.round(np.mean(sift_group, axis=0)).astype(np.uint8)
            for sift_group in ut.ProgIter(wx_to_sifts, lbl='compute words')
        ])
        vocab = vocab_indexer.VisualVocab(wx_to_word)
        vocab.build()

        oxford_data1 = {
            'imgid_order': imgid_order,
            'kpts_list': kpts_list,
            'vecs_list': vecs_list,
            'wordid_list': wordid_list,
            'vocab': vocab,
        }
        ut.save_data(data_fpath1, oxford_data1)
    return oxford_data1


def load_external_data2():
    """
    >>> from ibeis.algo.smk.script_smk import *  # NOQA
    """
    from os.path import join, basename, splitext
    import ibeis
    oxford_data1 = load_external_data1()
    imgid_order = oxford_data1['imgid_order']
    kpts_list = oxford_data1['kpts_list']
    vecs_list = oxford_data1['vecs_list']
    wordid_list = oxford_data1['wordid_list']
    vocab = oxford_data1['vocab']

    ibs = ibeis.opendb('Oxford')
    # Reorder internal data annots
    query_aids = ibs.filter_annots_general(has_any='query')
    data_aids = ibs.filter_annots_general(has_none='query')
    annots_ = ibs.annots(data_aids)
    intern_uris = [splitext(basename(uri))[0]
                   for uri in ibs.images(annots_.gids).uris_original]
    lookup = ut.make_index_lookup(intern_uris)
    uri_order = [x.replace('oxc1_', '') for x in imgid_order]
    sortx = ut.take(lookup, uri_order)
    annots = annots_.take(sortx)

    intern_uris = [splitext(basename(uri))[0]
                   for uri in ibs.images(annots.gids).uris_original]
    assert intern_uris == uri_order

    dbdir = ut.truepath('/raid/work/Oxford/')
    data_fpath2 = join(dbdir, 'oxford_data2.pkl')
    if ut.checkpath(data_fpath2):
        daids = annots.aids
        Y_list = []
        for aid, vecs in ut.ProgIter(zip(daids, vecs_list), nTotal=len(daids)):
            X = make_external_annot(aid, vecs, vocab)
            Y_list.append(X)
        external_data2 = {
            'Y_list': Y_list,
        }
        ut.save_data(data_fpath2, external_data2)
    else:
        external_data2 = ut.load_data(data_fpath2)
        Y_list = external_data2['Y_list']

    annots._internal_attrs['kpts'] = kpts_list
    annots._internal_attrs['vecs'] = vecs_list
    annots._internal_attrs['wordid'] = wordid_list
    annots._ibs = None


def make_external_annot(aid, vecs, vocab):
    nAssign = 1
    int_rvec = True
    # Compute assignments
    fx_to_vecs = vecs
    fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs, nAssign)
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    """
    z = np.array(ut.take_column(fx_to_wxs, 0)) + 1
    y = wordid_list[0]
    """
    # Build Aggregate Residual Vectors
    wx_list = sorted(wx_to_fxs.keys())
    word_list = ut.take(vocab.wx_to_word, wx_list)
    fxs_list = ut.take(wx_to_fxs, wx_list)
    maws_list = ut.take(wx_to_maws, wx_list)
    if int_rvec:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.int8)
    else:
        agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)
    agg_flags = np.empty((len(wx_list), 1), dtype=np.bool)
    for idx in range(len(wx_list)):
        word = word_list[idx]
        fxs = fxs_list[idx]
        maws = maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)
        _rvecs, _flags = smk_funcs.compute_rvec(vecs, word)
        _agg_rvec, _agg_flag = smk_funcs.aggregate_rvecs(_rvecs, maws, _flags)
        if int_rvec:
            _agg_rvec = smk_funcs.cast_residual_integer(_agg_rvec)
        agg_rvecs[idx] = _agg_rvec
        agg_flags[idx] = _agg_flag
    X = inverted_index.SingleAnnot()
    X.aid = aid
    X.wx_list = np.array(wx_list, dtype=np.int32)
    X.fxs_list = fxs_list
    X.maws_list = maws_list
    X.agg_rvecs = agg_rvecs
    X.agg_flags = agg_flags
    X.wx_to_idx = ut.make_index_lookup(X.wx_list)
    X.int_rvec = int_rvec
    X.wx_set = set(X.wx_list)

    # Ensure casting
    #for X in ut.ProgIter(X_list):
    #    X.agg_rvecs = smk_funcs.cast_residual_integer(X.agg_rvecs)
    #    X.wx_list = np.array(X.wx_list, dtype=np.int32)
    #    X.wx_to_idx = ut.map_dict_vals(np.int32, X.wx_to_idx)
    #    X.wx_to_idx = ut.map_dict_keys(np.int32, X.wx_to_idx)
    return X


def make_temporary_annot(aid, vocab, wx_to_weight, ibs, config):
    nAssign = config.get('nAssign', 1)
    alpha = config.get('smk_alpha', 3.0)
    thresh = config.get('smk_thresh', 3.0)
    # Compute assignments
    fx_to_vecs = ibs.get_annot_vecs(aid, config2_=config)
    fx_to_wxs, fx_to_maws = smk_funcs.assign_to_words(vocab, fx_to_vecs, nAssign)
    wx_to_fxs, wx_to_maws = smk_funcs.invert_assigns(fx_to_wxs, fx_to_maws)
    # Build Aggregate Residual Vectors
    wx_list = sorted(wx_to_fxs.keys())
    word_list = ut.take(vocab.wx_to_word, wx_list)
    fxs_list = ut.take(wx_to_fxs, wx_list)
    maws_list = ut.take(wx_to_maws, wx_list)
    agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)
    agg_flags = np.empty((len(wx_list), 1), dtype=np.bool)
    for idx in range(len(wx_list)):
        word = word_list[idx]
        fxs = fxs_list[idx]
        maws = maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)
        _rvecs, _flags = smk_funcs.compute_rvec(vecs, word)
        _agg_rvec, _agg_flag = smk_funcs.aggregate_rvecs(_rvecs, maws, _flags)
        agg_rvecs[idx] = _agg_rvec
        agg_flags[idx] = _agg_flag
    X = inverted_index.SingleAnnot()
    X.aid = aid
    X.wx_list = wx_list
    X.fxs_list = fxs_list
    X.maws_list = maws_list
    X.agg_rvecs = agg_rvecs
    X.agg_flags = agg_flags
    X.wx_to_idx = ut.make_index_lookup(X.wx_list)
    X.int_rvec = False
    X.wx_set = set(X.wx_list)

    weight_list = np.array(ut.take(wx_to_weight, wx_list))
    X.gamma = smk_funcs.gamma_agg(X.agg_rvecs, X.agg_flags, weight_list,
                                  alpha, thresh)
    return X


def verify_score():
    """
    Recompute all SMK things for two annotations and compare scores.

    >>> from ibeis.algo.smk.script_smk import *  # NOQA

    cm.print_inspect_str(qreq_)
    cm.show_single_annotmatch(qreq_, daid1)
    cm.show_single_annotmatch(qreq_, daid2)
    """
    qreq_, cm = load_internal_data()
    qreq_.ensure_data()

    ibs = qreq_.ibs
    qaid = cm.qaid
    daid1 = cm.get_top_truth_aids(ibs, ibs.const.TRUTH_MATCH)[0]
    daid2 = cm.get_top_truth_aids(ibs, ibs.const.TRUTH_MATCH, invert=True)[0]

    vocab = ibs.depc['vocab'].get_row_data([qreq_.dinva.vocab_rowid], 'words')[0]
    wx_to_weight = qreq_.dinva.wx_to_weight

    aid = qaid  # NOQA
    config = qreq_.qparams

    alpha = config.get('smk_alpha', 3.0)
    thresh = config.get('smk_thresh', 3.0)
    X = make_temporary_annot(qaid, vocab, wx_to_weight, ibs, config)
    assert np.isclose(smk_pipeline.match_kernel_agg(X, X, wx_to_weight, alpha, thresh)[0], 1.0)

    Y1 = make_temporary_annot(daid1, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y1, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.aid2_annot_score[daid1])
    assert np.isclose(smk_pipeline.match_kernel_agg(Y1, Y1, wx_to_weight, alpha, thresh)[0], 1.0)

    Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)
    item = smk_pipeline.match_kernel_agg(X, Y2, wx_to_weight, alpha, thresh)
    score = item[0]
    assert np.isclose(score, cm.aid2_annot_score[daid2])
    assert np.isclose(smk_pipeline.match_kernel_agg(Y2, Y2, wx_to_weight, alpha, thresh)[0], 1.0)
    #Y2 = make_temporary_annot(daid2, vocab, wx_to_weight, ibs, config)
