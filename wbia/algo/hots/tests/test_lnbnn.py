# -*- coding: utf-8 -*-
import logging
import sys  # noqa

import pytest
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


@pytest.mark.skipif("'--slow' not in sys.argv")
def test_lnbnn():
    import wbia

    ibs = wbia.opendb('PZ_MTEST')
    annots = ibs.annots()
    qaids = daids = annots.aids
    qreq = ibs.new_query_request(qaids, daids)
    cm_list = qreq.execute(use_cache=False)  # NOQA
