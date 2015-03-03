fsv
def get_sparse_matchinfo_agg(qreq_, qfx2_idx, nnfiltagg):
    # OLD WAY OF DOING THINGS
    K = qreq_.qparams.K
    # Build feature matches
    qfx2_nnidx = qfx2_idx.T[0:K].T
    qfx2_daid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    qfx2_dfx = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
    # Unpack filter scores and flags
    (qfx2_score_agg, qfx2_valid_agg) = nnfiltagg
    # We fill filter each relavant matrix by aggregate validity
    flat_validx = np.flatnonzero(qfx2_valid_agg)
    # Infer the valid internal query feature indexes and ranks
    valid_qfx   = np.floor_divide(flat_validx, K)
    valid_rank  = np.mod(flat_validx, K)
    # Then take the valid indices from internal database
    # annot_rowids, feature indexes, and all scores
    valid_daid  = qfx2_daid.take(flat_validx)
    valid_dfx   = qfx2_dfx.take(flat_validx)
    valid_score = qfx2_score_agg.take(flat_validx)
    valid_match_tup = (valid_daid, valid_qfx, valid_dfx, valid_score, valid_rank,)
    return valid_match_tup


def append_chipmatch_vsmany_agg(valid_match_tup):
    # OLD WAY OF DOING THINGS
    aid2_fm, aid2_fs, aid2_fk = new_fmfsfk()
    # TODO: Sorting the valid lists by aid might help the speed of this
    # code. Also, consolidating fm, fs, and fk into one vector will reduce
    # the amount of appends.
    (valid_daid, valid_qfx, valid_dfx, valid_score, valid_rank,) = valid_match_tup
    valid_fm = np.vstack((valid_qfx, valid_dfx)).T
    for daid, fm, fs, fk in zip(valid_daid, valid_fm, valid_score, valid_rank):
        # Note the difference in construction of fm
        aid2_fm[daid].append(fm)
        aid2_fs[daid].append(fs)
        aid2_fk[daid].append(fk)
    chipmatch = _fix_fmfsfk(aid2_fm, aid2_fs, aid2_fk)
    return chipmatch


def append_chipmatch_vsone_agg(valid_match_tup):
    # OLD WAY OF DOING THINGS
    (valid_daid, valid_qfx, valid_dfx, valid_score, valid_rank,) = valid_match_tup
    assert ut.list_allsame(valid_daid), 'internal daids should not have different daids for vsone'
    # Note the difference in construction of fm
    fm = np.vstack((valid_dfx, valid_qfx)).T
    fs = valid_score
    fk = valid_rank.tolist()
    return (fm, fs, fk)


@profile
def hack_fix_dupvote_weights(qreq_, qaid2_chipmatchSV):
    """
    If the one feature allowed to match to a name was removed by spatial
    verification then that feature never gets to vote.

    Maybe that is a good thing, but maybe we should try and reclaim it.

    CommandLine:
        python main.py --verbose --noqcache --cfg codename:vsmany_nsum  --query 1 --db PZ_MTEST

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-hack_fix_dupvote_weights

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> from ibeis.model.hots.pipeline import _spatial_verification
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', K=10)
        >>> ibs, qreq_ = get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = testrun_pipeline_upto(qreq_, 'spatial_verification')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_FILT']
        >>> qaid2_nnfilts = locals_['qaid2_nnfilts']
        >>> dupvotex = ut.listfind(qreq_.qparams.active_filter_list, hstypes.FiltKeys.DUPVOTE)
        >>> assert dupvotex is not None, 'dupvotex=%r' % dupvotex
        >>> qaid2_chipmatchSV = _spatial_verification(qreq_, qaid2_chipmatch)
        >>> before = [[fsv.T[dupvotex].sum()
        ...              for fsv in six.itervalues(cmtup_old[1])]
        ...                  for cmtup_old in six.itervalues(qaid2_chipmatchSV)]
        >>> before_sum = sum(ut.flatten(before))
        >>> print('before_sum=%r' % (before_sum))
        >>> ut.assert_inbounds(before_sum, 1940, 2200)
        >>> # execute test
        >>> total_reweighted = hack_fix_dupvote_weights(qreq_, qaid2_chipmatchSV)
        >>> print('total_reweighted=%r' % (total_reweighted))
        >>> after = [[fsv.T[dupvotex].sum()
        ...              for fsv in six.itervalues(cmtup_old[1])]
        ...                  for cmtup_old in six.itervalues(qaid2_chipmatchSV)]
        >>> after_sum = sum(ut.flatten(after))
        >>> print('after_sum=%r' % (after_sum))
        >>> diff = after_sum - before_sum
        >>> assert after_sum >= before_sum, 'sum should increase only'
        >>> ut.assert_inbounds(after_sum, 1950, 2210)
        >>> ut.assert_inbounds(total_reweighted, 5, 25)
        >>> ut.assert_inbounds(diff - total_reweighted, -1E-5, 1E-5)
        >>> total_reweighted2 = hack_fix_dupvote_weights(qreq_, qaid2_chipmatchSV)
        >>> print('total_reweighted2=%r' % (total_reweighted))
        >>> ut.assert_eq(total_reweighted2, 0, 'should be 0 reweighted')
    """
    #filtlist_list = [nnfilts[0] for nnfilts in six.itervalues(qaid2_nnfilts)]
    #assert ut.list_allsame(filtlist_list), 'different queries with differnt filts'
    #filtkey_list = filtlist_list[0]
    dupvotex = ut.listfind(qreq_.qparams.active_filter_list, hstypes.FiltKeys.DUPVOTE)
    if dupvotex is None:
        return
    dupvote_true = qreq_.qparams.filt2_stw[hstypes.FiltKeys.DUPVOTE][2]
    num_reweighted_list = []

    for qaid, cmtup_old in six.iteritems(qaid2_chipmatchSV):
        num_reweighted = 0
        daid2_fm, daid2_fsv, daid2_fk, aid2_score, daid2_H = cmtup_old
        daid_list = np.array(list(six.iterkeys(daid2_fsv)))
        fm_list = np.array(list(six.itervalues(daid2_fm)))
        fsv_list = np.array(list(six.itervalues(daid2_fsv)))
        # get dup weights in scores
        dw_list = np.array([fsv.T[dupvotex] for fsv in fsv_list])
        fk_list = np.array(list(six.itervalues(daid2_fk)))
        dnid_list = np.array(qreq_.ibs.get_annot_nids(list(daid_list)))
        unique_nids, nid_groupx = vt.group_indices(dnid_list)
        grouped_fm = vt.apply_grouping(fm_list, nid_groupx)
        grouped_dw = vt.apply_grouping(dw_list, nid_groupx)
        grouped_fk = vt.apply_grouping(fk_list, nid_groupx)
        grouped_daids = vt.apply_grouping(daid_list, nid_groupx)

        for daid_group, fm_group, dw_group, fk_group in zip(grouped_daids, grouped_fm, grouped_dw, grouped_fk):
            # all query features assigned to different annots in this name
            qfx_group = [fm.T[0] for fm in fm_group]
            flat_qfxs = np.hstack(qfx_group)
            #flat_dfws = np.hstack(dw_group)
            duplicate_qfxs = vt.find_duplicate_items(flat_qfxs)
            for qfx in duplicate_qfxs:
                idxs = [np.flatnonzero(qfxs == qfx) for qfxs in qfx_group]
                dupdws = [dws[idx] for idx, dws in zip(idxs, dw_group)]
                flat_dupdws = np.hstack(dupdws)
                # hack to find features where all votes are dupvote downweighted
                if np.all(flat_dupdws < .1):
                    dupdws
                    dupfks = [fks[idx] for idx, fks in zip(idxs, fk_group)]
                    flat_dupdws = np.hstack(dupfks)
                    # This feature needs its dupvote weight back
                    reweight_fk = np.min(flat_dupdws)
                    reweight_groupxs = np.nonzero([reweight_fk in fks for fks in dupfks])[0]
                    assert len(reweight_groupxs) == 1
                    reweight_groupx = reweight_groupxs[0]
                    reweight_daid = daid_group[reweight_groupx]
                    reweight_fsvxs = np.where(vt.and_lists(
                        daid2_fk[reweight_daid] == reweight_fk,
                        daid2_fm[reweight_daid].T[0] == qfx
                    ))[0]
                    assert len(reweight_fsvxs) == 1
                    reweight_fsvx = reweight_fsvxs[0]
                    # inplace modify
                    assert daid2_fsv[reweight_daid].T[dupvotex][reweight_fsvx] < .1, 'this was already reweighted'
                    daid2_fsv[reweight_daid].T[dupvotex][reweight_fsvx] = dupvote_true
                    num_reweighted += 1
                    #raise StopIteration('fds')

                #hasmatches = np.array(list(map(len, dupdws))) > 1
                #print(hasmatches)
                #if np.sum(hasmatches) > 1:
                #    raise StopIteration('fds')
                #    break
                #    pass
                #for idx in zip(fk_group, fsv_group, idxs
                #       pass
            #unique_indices = vt.group_indices(flat_qfxs)
            #idx2_groupid = flat_qfxs
            pass
        num_reweighted_list.append(num_reweighted)
    total_reweighted = sum(num_reweighted_list)
    #print('num_reweighted_list = %r' % (num_reweighted_list,))
    #print('total_reweighted = %r' % (total_reweighted,))
    return total_reweighted

