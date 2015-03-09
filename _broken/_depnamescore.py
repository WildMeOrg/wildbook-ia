def name_scoring_dense_old(nns_list, nnvalid0_list, qreq_):
    """
    DEPRICATE

    dupvotes gives duplicate name votes a weight close to 0.

    Dense version of name weighting

    Each query feature is only allowed to vote for each name at most once.
    IE: a query feature can vote for multiple names, but it cannot vote
    for the same name twice.

    CommandLine:
        python dev.py --allgt -t best --db PZ_MTEST
        python dev.py --allgt -t nsum --db PZ_MTEST
        python dev.py --allgt -t dupvote --db PZ_MTEST

    CommandLine:
        # Compares with dupvote on and dupvote off
        ./dev.py -t custom:dupvote_weight=0.0 custom:dupvote_weight=1.0  --db GZ_ALL --show --va -w --qaid 1032

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #tup = nn_weights.testdata_nn_weights('testdb1', slice(0, 1), slice(0, 11))
        >>> dbname = 'testdb1'  # 'GZ_ALL'  # 'testdb1'
        >>> cfgdict = dict(K=10, Knorm=10)
        >>> ibs, qreq_ = plh.get_pipeline_testdata(dbname=dbname, qaid_list=[2], daid_list=[1, 2, 3], cfgdict=cfgdict)
        >>> print(print(qreq_.get_infostr()))
        >>> pipeline_locals_ = plh.testrun_pipeline_upto(qreq_, 'weight_neighbors')
        >>> nns_list, nnvalid0_list = ut.dict_take(pipeline_locals_, ['nns_list', 'nnvalid0_list'])
        >>> # Test Function Call
        >>> dupvote_weight_list = name_scoring_dense_old(nns_list, nnvalid0_list, qreq_)
        >>> # Check consistency
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> qfx2_dupvote_weight = dupvote_weight_list[0]
        >>> flags = qfx2_dupvote_weight  > .5
        >>> qfx2_topnid = ibs.get_annot_name_rowids(qreq_.indexer.get_nn_aids(nns_list[0][0]))
        >>> isunique_list = [ut.isunique(row[flag]) for row, flag in zip(qfx2_topnid, flags)]
        >>> assert all(isunique_list), 'dupvote should only allow one vote per name'

    """
    K = qreq_.qparams.K
    def find_dupvotes(nns, qfx2_invalid0):
        if len(qfx2_invalid0) == 0:
            # hack for empty query features (should never happen, but it
            # inevitably will)
            qfx2_dupvote_weight = np.empty((0, K), dtype=hstypes.FS_DTYPE)
        else:
            (qfx2_idx, qfx2_dist) = nns
            qfx2_topidx = qfx2_idx.T[0:K].T
            qfx2_topaid = qreq_.indexer.get_nn_aids(qfx2_topidx)
            qfx2_topnid = qreq_.ibs.get_annot_name_rowids(qfx2_topaid)
            qfx2_topnid[qfx2_invalid0] = 0
            # A duplicate vote is when any vote for a name after the first
            qfx2_isnondup = np.array([ut.flag_unique_items(topnids) for topnids in qfx2_topnid])
            # set invalids to be duplicates as well (for testing)
            qfx2_isnondup[qfx2_invalid0] = False
            # Database feature index to chip index
            qfx2_dupvote_weight = (qfx2_isnondup.astype(hstypes.FS_DTYPE) * (1 - 1E-7)) + 1E-7
        return qfx2_dupvote_weight

    # convert ouf of dict format
    nninvalid0_list = [np.bitwise_not(qfx2_valid0) for qfx2_valid0 in nnvalid0_list]
    dupvote_weight_list = [
        find_dupvotes(nns, qfx2_invalid0)
        for nns, qfx2_invalid0 in zip(nns_list, nninvalid0_list)
    ]
    # convert into dict format
    return dupvote_weight_list




@_register_nn_simple_weight_func
def dupvote_match_weighter(nns_list, nnvalid0_list, qreq_):
    """
    DEPCIRATE

    dupvotes gives duplicate name votes a weight close to 0.

    Densve version of name weighting
    TODO: move to name_scoring
    TODO: sparse version of name weighting

    Each query feature is only allowed to vote for each name at most once.
    IE: a query feature can vote for multiple names, but it cannot vote
    for the same name twice.

    CommandLine:
        python dev.py --allgt -t best --db PZ_MTEST
        python dev.py --allgt -t nsum --db PZ_MTEST
        python dev.py --allgt -t dupvote --db PZ_MTEST

    CommandLine:
        python -m ibeis.model.hots.nn_weights --test-dupvote_match_weighter

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = plh.testdata_pre_weight_neighbors('testdb1', cfgdict=dict(K=10, Knorm=10))
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> # Test Function Call
        >>> dupvote_weight_list = nn_weights.dupvote_match_weighter(nns_list, nnvalid0_list, qreq_)
        >>> print(ut.numpy_str(dupvote_weight_list[0], precision=1))
        >>> # Check consistency
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> qfx2_dupvote_weight = dupvote_weight_list[0]
        >>> flags = qfx2_dupvote_weight  > .5
        >>> qfx2_topnid = ibs.get_annot_name_rowids(qreq_.indexer.get_nn_aids(nns_list[0][0]))
        >>> isunique_list = [ut.isunique(row[flag]) for row, flag in zip(qfx2_topnid, flags)]
        >>> assert all(isunique_list), 'dupvote should only allow one vote per name'

    CommandLine:
        ./dev.py -t nsum --db GZ_ALL --show --va -w --qaid 1032
        ./dev.py -t nsum_nosv --db GZ_ALL --show --va -w --qaid 1032

    """
    dupvote_weight_list = name_scoring.name_scoring_dense_old(nns_list, nnvalid0_list, qreq_)
    return dupvote_weight_list


@profile
def baseline_neighbor_filter(qreq_, nns_list, verbose=VERB_PIPELINE):
    """
    Removes matches to self, the same image, or the same name.

    CommandLine:
        python -m ibeis.model.hots.pipeline --test-baseline_neighbor_filter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *   # NOQA
        >>> qreq_, nns_list = plh.testdata_pre_baselinefilter(qaid_list=[1, 2, 3, 4], codename='vsmany')
        >>> nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list)
        >>> assert len(nnvalid0_list) == len(qreq_.get_external_qaids())
        >>> assert qreq_.qparams.K == 4
        >>> assert nnvalid0_list[0].shape[1] == qreq_.qparams.K
        >>> assert not np.any(nnvalid0_list[0][:, 0]), (
        ...    'first col should be all invalid because of self match')
        >>> assert not np.all(nnvalid0_list[0][:, 1]), (
        ...    'second col should have some good matches')
        >>> ut.assert_inbounds(nnvalid0_list[0].sum(), 1900, 2000)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pipeline import *   # NOQA
        >>> qreq_, nns_list = plh.testdata_pre_baselinefilter(codename='vsone')
        >>> nnvalid0_list = baseline_neighbor_filter(qreq_, nns_list)
        >>> assert len(nnvalid0_list) == len(qreq_.get_external_daids())
        >>> assert qreq_.qparams.K == 1
        >>> assert nnvalid0_list[0].shape[1] == qreq_.qparams.K
        >>> ut.assert_eq(nnvalid0_list[0].sum(), 0, 'no self matches')
        >>> ut.assert_inbounds(nnvalid0_list[1].sum(), 800, 1100)
    """
    if verbose:
        print('[hs] Step 2) Baseline neighbor filter')

    def flag_impossible_votes(qreq_, qaid, qfx2_nnidx, cant_match_self,
                              cant_match_sameimg, cant_match_samename,
                              verbose=VERB_PIPELINE):
        """
        Flags matches to self or same image
        """
        # Baseline is all matches have score 1 and all matches are valid
        qfx2_valid0 = np.ones(qfx2_nnidx.shape, dtype=np.bool)

        # Get neighbor annotation information
        qfx2_aid = qreq_.indexer.get_nn_aids(qfx2_nnidx)
        # dont vote for yourself or another chip in the same image
        if cant_match_sameimg:
            qfx2_gid = qreq_.ibs.get_annot_gids(qfx2_aid)
            qgid     = qreq_.ibs.get_annot_gids(qaid)
            qfx2_notsameimg = qfx2_gid != qgid
            if DEBUG_PIPELINE:
                plh._sameimg_verbose_check(qfx2_notsameimg, qfx2_valid0)
            qfx2_valid0 = np.logical_and(qfx2_valid0, qfx2_notsameimg)
        elif cant_match_self:
            # dont need to run this if cant_match_sameimg was True
            qfx2_notsamechip = qfx2_aid != qaid
            if DEBUG_PIPELINE:
                plh._self_verbose_check(qfx2_notsamechip, qfx2_valid0)
            qfx2_valid0 = np.logical_and(qfx2_valid0, qfx2_notsamechip)
        if cant_match_samename:
            # This should probably be off
            qfx2_nid = qreq_.ibs.get_annot_name_rowids(qfx2_aid)
            qnid = qreq_.ibs.get_annot_name_rowids(qaid)
            qfx2_notsamename = qfx2_nid != qnid
            if DEBUG_PIPELINE:
                plh._samename_verbose_check(qfx2_notsamename, qfx2_valid0)
            qfx2_valid0 = np.logical_and(qfx2_valid0, qfx2_notsamename)
        return qfx2_valid0
    cant_match_sameimg  = not qreq_.qparams.can_match_sameimg
    cant_match_samename = not qreq_.qparams.can_match_samename
    cant_match_self     = True
    K = qreq_.qparams.K
    internal_qaids = qreq_.get_internal_qaids()

    # Look at impossibility of the first K nearest neighbors
    nnidx_iter = (qfx2_idx.T[0:K].T for (qfx2_idx, _) in nns_list)
    nnvalid0_list = [
        flag_impossible_votes(qreq_, qaid, qfx2_nnidx, cant_match_self,
                              cant_match_sameimg, cant_match_samename,
                              verbose=verbose)
        for qaid, qfx2_nnidx in zip(internal_qaids, nnidx_iter)
    ]
    if False:
