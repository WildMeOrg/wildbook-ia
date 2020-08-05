# -*- coding: utf-8 -*-
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def test_lnbnn():
    import wbia

    ibs = wbia.opendb('PZ_MTEST')
    annots = ibs.annots()
    qaids = daids = annots.aids
    qreq = ibs.new_query_request(qaids, daids)
    cm_list = qreq.execute(use_cache=False)  # NOQA
