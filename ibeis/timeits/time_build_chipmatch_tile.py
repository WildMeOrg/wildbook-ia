    def tilemesh(nQKpts, K):
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        return qfx2_qfx, qfx2_k

    qfx2_fx, qfx2_k = tilemesh(nQKpts, K)

    qfx2_fx_, qfx2_k_ = np.meshgrid(np.arange(nQKpts), np.arange(K), indexing='ij')
    np.all(qfx2_fx == qfx2_fx_)
    np.all(qfx2_k == qfx2_k_)




def build_match_iterator(qfx2_idx, qfx2_score_agg, qfx2_valid_agg, qreq_):
    """
    builds sparse iterator that generates feature match pairs, scores, and ranks

    Args:
        qfx2_idx (ndarray):
        qfx2_fs (ndarray):
        qfx2_valid (ndarray):
        qreq_ (QueryRequest):  hyper-parameters

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-build_match_iterator

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *  # NOQA
        >>> from ibeis.model.hots import pipeline
        >>> verbose = True
        >>> cfgdict = dict(codename='vsmany')
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata('testdb1', cfgdict=cfgdict)
        >>> locals_ = pipeline.testrun_pipeline_upto(qreq_, 'build_chipmatches')
        >>> qaid2_nns, qaid2_nnfilts, qaid2_nnfiltagg = [
        ...     locals_[key] for key in ['qaid2_nns', 'qaid2_nnfilts', 'qaid2_nnfiltagg']]
        >>> qaid = qreq_.get_internal_qaids()[0]
        >>> qfx2_idx = qaid2_nns[qaid][0]
        >>> (qfx2_score_agg, qfx2_valid_agg) = qaid2_nnfiltagg[qaid]
        >>> (qfx2_score_list, qfx2_valid_list) = qaid2_nnfilts[qaid]
        >>>
    """
    K = qreq_.qparams.K
    nQKpts = qfx2_idx.shape[0]
    # Build feature matches
    qfx2_nnidx = qfx2_idx.T[0:K].T
    qfx2_aid  = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    qfx2_fx   = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
    # FIXME: Can probably get away without using tile here
    qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
    qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))

    def tilemesh(nQKpts, K):
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        return qfx2_qfx, qfx2_k

    def repeatmesh(nQKpts, K):
        qfx2_k = np.repeat(np.arange(K).reshape(K, 1).T, nQKpts, axis=0)
        qfx2_fx = np.repeat(np.arange(nQKpts).reshape(1, nQKpts).T, K, axis=1)
        return qfx2_fx, qfx2_k

    def repeatmesh2(nQKpts, K):
        qfx2_k = np.repeat(np.arange(K)[None], nQKpts, axis=0)
        qfx2_fx = np.repeat(np.arange(nQKpts)[:, None], K, axis=1)
        return qfx2_fx, qfx2_k

    def repeatmesh3(nQKpts, K):
        basek = np.arange(K)
        baseq = np.arange(nQKpts)
        qfx2_k = np.repeat(basek[None], nQKpts, axis=0)
        qfx2_fx = np.repeat(baseq[None].T, K, axis=1)
        return qfx2_fx, qfx2_k

    def repeatmesh4(nQKpts, K):
        return np.repeat(np.arange(K)[None], nQKpts, axis=0), np.repeat(np.arange(nQKpts)[None].T, K, axis=1)

    qfx2_fx1, qfx2_k1 = tilemesh(nQKpts, K)
    qfx2_fx5, qfx2_k5 = repeatmesh3(nQKpts, K)
    #qfx2_fx2, qfx2_k2 = np.meshgrid(np.arange(nQKpts), np.arange(K), indexing='ij')
    #qfx2_fx3, qfx2_k3 = repeatmesh(nQKpts, K)
    #qfx2_fx4, qfx2_k4 = repeatmesh2(nQKpts, K)

    assert np.all(qfx2_fx1 == qfx2_fx2)
    assert np.all(qfx2_k1 == qfx2_k2)

    assert np.all(qfx2_k1 == qfx2_k3)
    assert np.all(qfx2_fx1 == qfx2_fx3)

    assert np.all(qfx2_k1 == qfx2_k4)
    assert np.all(qfx2_fx1 == qfx2_fx4)

    assert np.all(qfx2_k1 == qfx2_k5)
    assert np.all(qfx2_fx1 == qfx2_fx5)

    %timeit tilemesh(nQKpts, K)
    %timeit np.meshgrid(np.arange(nQKpts), np.arange(K), indexing='ij')
    %timeit repeatmesh(nQKpts, K)
    %timeit repeatmesh2(nQKpts, K)
    %timeit repeatmesh3(nQKpts, K)
    %timeit repeatmesh4(nQKpts, K)

    # We fill filter each relavant matrix by aggregate validity
    tofiltertup = (qfx2_qfx, qfx2_aid, qfx2_fx, qfx2_score_agg, qfx2_k,)
    valid_lists = (qfx2[qfx2_valid_agg] for qfx2 in tofiltertup)
    match_iter = zip(*valid_lists)
    return match_iter


    K = qreq_.qparams.K
    nQKpts = qfx2_idx.shape[0]
    # Build feature matches
    qfx2_nnidx = qfx2_idx.T[0:K].T
    qfx2_aid  = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    qfx2_fx   = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
    # Tile is actually the fastest I've been able create the
    # grid, this is faster than the repeast and meshgrid methods
    qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
    qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
    # We fill filter each relavant matrix by aggregate validity
    flat_validx = qfx2_valid_agg.ravel().nonzero()[0]
    matchinfo = np.vstack((
        qfx2_qfx.take(flat_validx),
        qfx2_aid.take(flat_validx),
        qfx2_fx.take(flat_validx),
        qfx2_score_agg.take(flat_validx),
        qfx2_k.take(flat_validx),)).T

    # This is 700ms but produces correct datatypes
    #%timeit list(zip(qfx2_qfx.take(flat_validx), qfx2_aid.take(flat_validx), qfx2_fx.take(flat_validx), qfx2_score_agg.take(flat_validx), qfx2_k.take(flat_validx)))

    # This is 10x faster but produces wrong datatypes
    #%timeit np.vstack(( qfx2_qfx.take(flat_validx), qfx2_aid.take(flat_validx), qfx2_fx.take(flat_validx), qfx2_score_agg.take(flat_validx), qfx2_k.take(flat_validx),)).T

    valid_k = qfx2_k.take(flat_validx)
    valid_qfx = qfx2_qfx.take(flat_validx)
    assert np.all(valid_qfx == matchinfo.T[0])
    assert np.all(valid_k == matchinfo.T[4])
    %timeit np.mod(flat_validx, K)
    %timeit np.mod(flat_validx, nQKpts)

    np.all(valid_k == np.mod(flat_validx, K))
    np.all(valid_qfx == np.floor_divide(flat_validx, K))



    K = qreq_.qparams.K
    nQKpts = qfx2_idx.shape[0]
    # Build feature matches
    qfx2_nnidx = qfx2_idx.T[0:K].T
    qfx2_aid  = qreq_.indexer.get_nn_aids(qfx2_nnidx)
    qfx2_dfx   = qreq_.indexer.get_nn_featxs(qfx2_nnidx)
    # I can do without it by being smart
    # Tile is actually the fastest I've been able create the
    # grid, this is faster than the repeast and meshgrid methods
    #qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
    #qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
    # We fill filter each relavant matrix by aggregate validity

    flat_validx = qfx2_valid_agg.ravel().nonzero()[0]
    valid_qfx = np.floor_divide(flat_validx, K)
    valid_k = np.mod(flat_validx, K)
    matchinfo = np.vstack((
        valid_qfx,
        qfx2_aid.take(flat_validx),
        qfx2_dfx.take(flat_validx),
        qfx2_score_agg.take(flat_validx),
        valid_k,)).T

    # This is 700ms but produces correct datatypes
    #%timeit list(zip(qfx2_qfx.take(flat_validx), qfx2_aid.take(flat_validx), qfx2_fx.take(flat_validx), qfx2_score_agg.take(flat_validx), qfx2_k.take(flat_validx)))

    # This is 10x faster but produces wrong datatypes
    #%timeit np.vstack(( qfx2_qfx.take(flat_validx), qfx2_aid.take(flat_validx), qfx2_fx.take(flat_validx), qfx2_score_agg.take(flat_validx), qfx2_k.take(flat_validx),)).T

    valid_k = qfx2_k.take(flat_validx)
    valid_qfx = qfx2_qfx.take(flat_validx)
    assert np.all(valid_qfx == matchinfo.T[0])
    assert np.all(valid_k == matchinfo.T[4])
    %timeit np.mod(flat_validx, K)
    %timeit np.floor_divide(flat_validx, K)
    %timeit  qfx2_qfx.take(flat_validx)

    np.all(valid_k == np.mod(flat_validx, K))
    np.all(valid_qfx == np.floor_divide(flat_validx, K))

