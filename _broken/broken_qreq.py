    #def load_oris(qreq_, ibs):
    #    if qreq_.idx2_oris is not None:
    #        return
    #    from vtool import keypoint as ktool
    #    qreq_.load_kpts(ibs)
    #    idx2_oris = ktool.get_oris(qreq_.idx2_kpts)
    #    assert len(idx2_oris) == len(qreq_.num_indexed_vecs())
    #    qreq_.idx2_oris = idx2_oris

    #def load_kpts(qreq_, ibs):
    #    if qreq_.idx2_kpts is not None:
    #        return
    #    aid_list = qreq_.indexer.aid_list
    #    kpts_list = qreq_.ibs.get_annot_kpts(aid_list)
    #    idx2_kpts = np.vstack(kpts_list)
    #    qreq_.idx2_kpts = idx2_kpts

    #def load_query_queryx(qreq_):
    #    qaids = qreq_.get_internal_qaids()
    #    qaid2_queryx = {aid: queryx for queryx, aid in enumerate(qaids)}
    #    qreq_.qaid2_queryx = qaid2_queryx

    #def load_data_datax(qreq_):
    #    daids = qreq_.get_internal_daids()
    #    daid2_datax = {aid: datax for datax, aid in enumerate(daids)}
    #    qreq_.daid2_datax = daid2_datax

    #def load_query_gids(qreq_, ibs):
    #    if qreq_.internal_qgid_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    gid_list = ibs.get_annot_gids(aid_list)
    #    qreq_.internal_qgid_list = gid_list

    #def load_query_nids(qreq_, ibs):
    #    if qreq_.internal_qnid_list is not None:
    #        return False
    #    aid_list = qreq_.get_internal_qaids()
    #    nid_list = ibs.get_annot_name_rowids(aid_list)
    #    qreq_.internal_qnid_list = nid_list
