
SIMPLE_MERGE = True
if SIMPLE_MERGE:
else:
    # Find cases where vsone and prior disagree
    en_flags1 = np.in1d(fm_vsone.T[0], fm_prior.T[0])
    en_flags2 = np.in1d(fm_prior.T[0], fm_vsone.T[0])
    ne_flags1 = np.in1d(fm_vsone.T[1], fm_prior.T[1])
    ne_flags2 = np.in1d(fm_prior.T[1], fm_vsone.T[1])
    print(fm_vsone.compress(en_flags1, axis=0))
    print(fm_prior.compress(en_flags2, axis=0))
    print(fm_vsone.compress(ne_flags1, axis=0))
    print(fm_prior.compress(ne_flags2, axis=0))

    # Cases where matches are mutually exclusive
    # (the other method did not find a match or match to these indicies)
    mutex_flags1 = np.logical_and(~en_flags1, ~ne_flags1)
    mutex_flags2 = np.logical_and(~en_flags2, ~ne_flags2)
    fm_vsone_mutex = fm_vsone.compress(mutex_flags1, axis=0)
    fm_prior_mutex = fm_prior.compress(mutex_flags2, axis=0)
    print(fm_vsone_mutex)
    print(fm_prior_mutex)
    fm_both = np.vstack([ fm_both, fm_vsone_mutex, fm_prior_mutex ])
    print(fm_both)


# DEPRICATE CODE UNDERNEATH
# DONE: TODO split both vsone and vsmany queries into chunks


def execute_query(ibs, qreq_, verbose, save_qcache):
    # Execute and save cachemiss queries
    if qreq_.qparams.vsone:
        # break vsone queries into multiple queries - one for each external qaid
        qaid2_qres = execute_vsone_query(ibs, qreq_, verbose, save_qcache)
    else:
        qaid2_qres = execute_nonvsone_query(ibs, qreq_, verbose, save_qcache)
    return qaid2_qres


@profile
def execute_nonvsone_query(ibs, qreq_, verbose, save_qcache):
    # execute non-vsone queries
    all_qaids = qreq_.get_external_qaids()

    chunksize = 64
    if len(all_qaids) <= chunksize:
        # If less than the chunksize peform old non-chuncked queries
        # We will get to the point where chunking will cost no overhead
        # and be robust. At that point this code will be depricated.
        qaid2_qres = pipeline.request_ibeis_query_L0(ibs, qreq_, verbose=verbose)
        if save_qcache:
            pipeline.save_resdict(qreq_, qaid2_qres, verbose=verbose)
        else:
            if ut.VERBOSE:
                print('[mc4] not saving vsmany chunk')
    else:
        qaid2_qres = {}
        # Iterate over vsone queries in chunks. This ensures that we dont lose
        # too much time if a qreq_ crashes after the 2000th nn index.
        nTotalChunks    = ut.get_nTotalChunks(len(all_qaids), chunksize)
        qaid_chunk_iter = ut.ichunks(all_qaids, chunksize)
        _qreq_iter = (qreq_.shallowcopy(qaids=qaids) for qaids in qaid_chunk_iter)
        qreq_iter = ut.ProgressIter(_qreq_iter, nTotal=nTotalChunks, freq=1,
                                    lbl='vsmany query chunk: ', backspace=False)
        for __qreq in qreq_iter:
            if ut.VERBOSE:
                print('Generating vsmany chunk')
            __qaid2_qres = pipeline.request_ibeis_query_L0(ibs, __qreq, verbose=verbose)
            if save_qcache:
                pipeline.save_resdict(qreq_, __qaid2_qres, verbose=verbose)
            else:
                if ut.VERBOSE:
                    print('[mc4] not saving vsmany chunk')
            qaid2_qres.update(__qaid2_qres)
    return qaid2_qres


def execute_vsone_query(ibs, qreq_, verbose, save_qcache):
    qaid_list = qreq_.get_external_qaids()
    qaid2_qres = {}
    chunksize = 4
    qres_gen = generate_vsone_qreqs(ibs, qreq_, qaid_list, chunksize,
                                    verbose=verbose)
    qres_iter = ut.ProgressIter(qres_gen, nTotal=len(qaid_list), freq=1,
                                backspace=False, lbl='vsone query: ',
                                use_rate=True)
    qres_chunk_iter = ut.ichunks(qres_iter, chunksize)

    for qres_chunk in qres_chunk_iter:
        qaid2_qres_ = {qaid: qres for qaid, qres in qres_chunk}
        # Save chunk of vsone queries
        if save_qcache:
            if ut.VERBOSE:
                print('[mc4] saving vsone chunk')
            pipeline.save_resdict(qreq_, qaid2_qres_, verbose=verbose)
        else:
            if ut.VERBOSE:
                print('[mc4] not saving vsone chunk')
        # Add current chunk to results
        qaid2_qres.update(qaid2_qres_)
    return qaid2_qres


def generate_vsone_qreqs(ibs, qreq_, qaid_list, chunksize, verbose=True):
    """
    helper

    Generate vsone quries one at a time, but create shallow qreqs in chunks.
    """
    #qreq_shallow_iter = ((query_request.qreq_shallow_copy(qreq_, qx), qaid)
    #                     for qx, qaid in enumerate(qaid_list))
    # normalizers are the same for all vsone queries but indexers are not
    qreq_.lazy_preload(verbose=verbose)
    qreq_shallow_iter = ((qreq_.shallowcopy(qx=qx), qaid)
                         for qx, qaid in enumerate(qaid_list))
    qreq_chunk_iter = ut.ichunks(qreq_shallow_iter, chunksize)
    for qreq_chunk in qreq_chunk_iter:
        for __qreq, qaid in qreq_chunk:
            if ut.VERBOSE:
                print('Generating vsone for qaid=%d' % (qaid,))
            __qaid2_qres = pipeline.request_ibeis_query_L0(ibs, __qreq, verbose=verbose)
            qres = __qaid2_qres[qaid]
            yield (qaid, qres)

