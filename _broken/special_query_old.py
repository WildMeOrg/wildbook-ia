@profile
def get_new_qres_filter_scores(qres_vsone, qres_vsmany, top_aids, filtkey):
    """
    applies verified scores of type ``filtkey`` from qaid2_qres_vsmany to qaid2_qres_vsone

    Args:
        qres_vsone (QueryResult):  object of feature correspondences and scores
        qres_vsmany (QueryResult):  object of feature correspondences and scores
        top_aids (?):
        filtkey (?):

    Returns:
        tuple: (qaid2_qres, qreq_)

    CommandLine:
        python -m ibeis.model.hots.special_query --test-get_new_qres_filter_scores
    """
    newfsv_list = []
    newscore_aids = []
    for daid in top_aids:
        if (daid not in qres_vsone.aid2_fm or
             daid not in qres_vsmany.aid2_fm):
            # no matches to work with
            continue
        fm_vsone      = qres_vsone.aid2_fm[daid]
        fm_vsmany     = qres_vsmany.aid2_fm[daid]

        scorex_vsone  = ut.listfind(qres_vsone.filtkey_list, filtkey)
        scorex_vsmany = ut.listfind(qres_vsmany.filtkey_list, filtkey)
        if scorex_vsone is None:
            shape = (qres_vsone.aid2_fsv[daid].shape[0], 1)
            new_filtkey_list = qres_vsone.filtkey_list[:]
            #new_scores_vsone = np.full(shape, np.nan)
            new_scores_vsone = np.ones(shape)
            new_fsv_vsone = np.hstack((qres_vsone.aid2_fsv[daid], new_scores_vsone))
            new_filtkey_list.append(filtkey)
            assert len(new_filtkey_list) == len(new_fsv_vsone.T), 'filter length is not consistent'
            new_score_vsone = new_fsv_vsone.T[-1].T
        else:
            assert False, 'scorex_vsone should be None'
            new_score_vsone = qres_vsone.aid2_fsv[daid].T[scorex_vsone].T
        scores_vsmany = qres_vsmany.aid2_fsv[daid].T[scorex_vsmany].T

        # find intersecting matches
        # (should we just take the scores from the pre-spatial verification
        #  part of the pipeline?)
        common, fmx_vsone, fmx_vsmany = vt.intersect2d_numpy(fm_vsone, fm_vsmany, return_indicies=True)
        mutual_scores = scores_vsmany.take(fmx_vsmany)
        new_score_vsone[fmx_vsone] = mutual_scores

        newfsv_list.append(new_score_vsone)
        newscore_aids.append(daid)
    return newfsv_list, newscore_aids
