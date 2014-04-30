

def bigcache_query(ibs, qreq, batch_size=10, use_bigcache=True,
                   limit_memory=False, verbose=True):
    qcids = qreq.qcids
    if use_bigcache and not params.args.nocache_query:
        try:
            qcid2_res = load_bigcache_query(ibs, qreq, verbose)
            return qcid2_res
        except IOError as ex:
            print(ex)
    # Perform checks
    #pre_cache_checks(ibs, qreq)
    pre_exec_checks(ibs, qreq)
    # Execute queries in batches
    qcid2_res = {}
    nBatches = int(np.ceil(len(qcids) / batch_size))
    batch_enum = enumerate(utool.ichunks(qcids, batch_size))
    for batchx, qcids_batch in batch_enum:
        print('[mc3] batch %d / %d' % (batchx, nBatches))
        qreq.qcids = qcids_batch
        print('qcids_batch=%r. quid=%r' % (qcids_batch, qreq.get_uid()))
        try:
            qcid2_res_ = process_query_request(ibs, qreq, safe=False)
            # Append current batch results if we have the memory
            if not limit_memory:
                qcid2_res.update(qcid2_res_)
        except mf.QueryException as ex:
            print('[mc3] ERROR !!!: %r' % ex)
            if params.args.strict:
                raise
            continue
    qreq.qcids = qcids
    # Need to reload all queries
    if limit_memory:
        qcid2_res = process_query_request(ibs, qreq, safe=False)
    save_bigcache_query(qcid2_res, ibs, qreq)
    return qcid2_res



#@profile
#@utool.indent_decor('[pre_cache]')
#def pre_cache_checks(ibs, qreq):
    #print(' --- pre cache checks --- ')
    ## Ensure ibs object is using the right config
    ##ibs.attatch_qreq(qreq)
    #feat_uid = qreq.cfg._feat_cfg.get_uid()
    ## Load any needed features or chips into memory
    #if ibs.feats.feat_uid != feat_uid:
        #print(' !! UNLOAD DATA !!')
        #print('[mc3] feat_uid = %r' % feat_uid)
        #print('[mc3] ibs.feats.feat_uid = %r' % ibs.feats.feat_uid)
        #ibs.unload_ciddata('all')
    #return qreq
