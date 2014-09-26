
#@profile
def build_daid2_chipmatch(invindex, common_wxs, wx2_qaids, wx2_qfxs,
                          scores_list, weight_list, daids_list, query_gamma,
                          daid2_gamma):
    """
    Total time: 13.2826 s
    this builds the structure that the rest of the pipeline plays nice with
    """
    if utool.VERBOSE:
        print('[smk_core] build chipmatch')
    #start_keys = set() set(locals().keys())
    wx2_dfxs   = invindex.wx2_fxs
    qfxs_list  = [qfxs for qfxs in wx2_qfxs[common_wxs]]
    dfxs_list  = [pdh.ensure_values(dfxs) for dfxs in wx2_dfxs[common_wxs].values]
    qaids_list = [pdh.ensure_values(qaids) for qaids in wx2_qaids[common_wxs].values]
    daid2_chipmatch_ = utool.ddict(list)
    _iter = list(zip(scores_list, qaids_list, daids_list, qfxs_list, dfxs_list,
                     weight_list))

    #def accumulate_chipmatch(
    # Accumulate all matching indicies with scores etc...
    for scores, qaids, daids, qfxs, dfxs, weight in _iter:
        _is, _js = np.meshgrid(np.arange(scores.shape[0]),
                               np.arange(scores.shape[1]), indexing='ij')  # 4.7%
        for i, j in zip(_is.flat, _js.flat):   # 4.7%
            try:
                score = scores.take(i, axis=0).take(j, axis=0)  # 9.3
                if score == 0:  # 4%
                    continue
                #qaid  = qaids[i]
                qfxs_ = qfxs[i]
                daid  = daids[j]
                dfxs_ = dfxs[j]
                # Cartesian product to list all matches that gave this score
                stackable = list(product(qfxs_, dfxs_))  # 4.9%
                # Distribute score over all words that contributed to it.
                # apply other normalizers as well so a sum will reconstruct the
                # total score
                norm  = weight * (query_gamma * daid2_gamma[daid]) / len(stackable)  # 15.0%
                _fm   = np.vstack(stackable)  # 16.6%
                _fs   = np.array([score] * _fm.shape[0]) * norm
                _fk   = np.ones(_fs.shape)
                #assert len(_fm) == len(_fs)
                #assert len(_fk) == len(_fs)
                chipmatch_ = (_fm, _fs, _fk)
                daid2_chipmatch_[daid].append(chipmatch_)
            except Exception as ex:
                local_keys = ['score', 'qfxs_', 'dfxs_', 'qaids', 'daids',
                              '_fm', '_fs', '_fm.shape', '_fs.shape', '_fk.shape']
                utool.printex(ex, keys=local_keys, separate=True)
                raise

    # Concatenate into full fmfsfk reprs
    daid2_cattup = {daid: concat_chipmatch(cmtup) for daid, cmtup in
                    six.iteritems(daid2_chipmatch_)}  # 12%
    #smk_debug.check_daid2_chipmatch(daid2_cattup)
    # Qreq needs unzipped chipmatch
    daid2_fm = {daid: cattup[0] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_fs = {daid: cattup[1] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_fk = {daid: cattup[2] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)
    return daid2_chipmatch


#@profile
def concat_chipmatch(cmtup):
    """
    Total time: 1.63271 s
    """

    fm_list = [_[0] for _ in cmtup]
    fs_list = [_[1] for _ in cmtup]
    fk_list = [_[2] for _ in cmtup]
    assert len(fm_list) == len(fs_list)
    assert len(fk_list) == len(fs_list)
    chipmatch = (np.vstack(fm_list), np.hstack(fs_list), np.hstack(fk_list))  # 88.9%
    assert len(chipmatch[0]) == len(chipmatch[1])
    assert len(chipmatch[2]) == len(chipmatch[1])
    return chipmatch



@profile
def featmatch_gen(scores_list, daids_list, qfxs_list, dfxs_list, weight_list,
                  query_gamma):
    """
    Total time: 2.25327 s
    """
    #shape_ranges = [(np.arange(w), np.arange(h)) for (w, h) in shapes_list]  # 960us
    #ijs_iter = [np.meshgrid(wrange, hrange, indexing='ij') for wrange, hrange in shape_ranges] # 13.6ms
    # Use caching to quickly create meshes
    shapes_list  = [scores.shape for scores in scores_list]  # 51us
    shape_ranges = [(mem_arange(w), mem_arange(h)) for (w, h) in shapes_list]  # 230us
    ijs_list = [mem_meshgrid(wrange, hrange) for wrange, hrange in shape_ranges]  # 278us
    _is_list = [ijs[0] for ijs in ijs_list]
    _js_list = [ijs[1] for ijs in ijs_list]
    shapenorm_list = [w * h for (w, h) in shapes_list]
    norm_list = np.multiply(np.divide(weight_list, shapenorm_list), query_gamma)

    ##with utool.Timer('fsd'):
    #gentype = lambda x: x
    #gentype = list
    #out_ijs    = [list(zip(_is.flat, _js.flat)) for (_is, _js) in ijs_list]
    #out_scores = gentype(([scores[ij] for ij in ijs]
    #                      for (scores, ijs) in zip(scores_list, out_ijs)))
    #out_qfxs   = gentype(([qfxs[i] for (i, j) in ijs]
    #                      for (qfxs, ijs) in zip(qfxs_list, out_ijs)))
    #out_dfxs   = gentype(([dfxs[j] for (i, j) in ijs]
    #                      for (dfxs, ijs) in zip(dfxs_list, out_ijs)))
    #out_daids  = gentype(([daids[j] for (i, j) in ijs]
    #                      for (daids, ijs) in zip(daids_list, out_ijs)))

    #all_qfxs = np.vstack(out_qfxs)
    #all_dfxs = np.vstack(out_dfxs)
    #all_scores = np.hstack(out_scores)
    #all_daids = np.hstack(out_daids)
    #from ibeis.model.hots.smk import smk_speed
    #daid_keys, groupxs = smk_speed.group_indicies(all_daids)
    #fs_list = smk_speed.apply_grouping(all_scores, groupxs)
    #fm1_list = smk_speed.apply_grouping(all_qfxs, groupxs)
    #fm2_list = smk_speed.apply_grouping(all_dfxs, groupxs)
    #fm_list = [np.hstack((fm1, fm2)) for fm1, fm2 in zip(fm1_list, fm2_list)]

    #aid_list = smk_speed.apply_grouping(all_daids, groupxs)

    #with utool.Timer('fds'):
    _iter = zip(scores_list, daids_list, qfxs_list, dfxs_list,
                norm_list, _is_list, _js_list)
    for scores, daids, qfxs, dfxs, norm, _is, _js in _iter:
        for i, j in zip(_is.flat, _js.flat):  # 4.7%
            score = scores.take(i, axis=0).take(j, axis=0)  # 9.3
            if score == 0:  # 4%
                continue
        yield score, norm, daids[j], qfxs[i], dfxs[j]


