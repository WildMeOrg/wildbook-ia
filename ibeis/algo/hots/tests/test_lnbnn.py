def test_lnbnn():
    import ibeis
    ibs = ibeis.opendb('PZ_MTEST')
    annots = ibs.annots()
    qaids = daids = annots.aids
    qreq = ibs.new_query_request(qaids, daids)
    cm_list = qreq.execute(use_cache=False)
