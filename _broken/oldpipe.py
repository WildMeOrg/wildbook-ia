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


