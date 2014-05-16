#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
# IBEIST
from ibeis.model.preproc import preproc_chip
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_COMPUTE_CHIPS]')


def TEST_COMPUTE_CHIPS(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    print('get_valid_ROIS')
    rid_list = ibs.get_valid_rids()
    assert len(rid_list) > 0, 'database rois cannot be empty for TEST_COMPUTE_CHIPS'
    print(' * len(rid_list) = %r' % len(rid_list))
    preproc_chip.compute_and_write_chips(ibs, rid_list)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_COMPUTE_CHIPS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
