#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.tests import __testing__
import multiprocessing
import utool
# IBEIST
from ibeis.model.preproc import preproc_feat
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_COMPUTE_FEATS]')


def TEST_COMPUTE_FEATS(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    print('get_valid_ROIS')
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.add_chips(rid_list)
    assert len(cid_list) > 0, 'database chips cannot be empty for TEST_COMPUTE_FEATS'
    print(' * len(cid_list) = %r' % len(cid_list))
    feat_list = list(preproc_feat.add_feat_params_gen(ibs, cid_list))
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0')
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = __testing__.run_test(TEST_COMPUTE_FEATS, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
