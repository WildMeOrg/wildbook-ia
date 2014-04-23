#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
try:
    import __testing__
except ImportError:
    from tests import __testing__
import multiprocessing
import utool
# IBEIST
from ibeis.model.preproc import preproc_chip
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_COMPUTE_CHIPS]')
printTEST = __testing__.printTEST


def TEST_COMPUTE_CHIPS(ibs):
    # Create a HotSpotter API (hs) and GUI backend (back)
    printTEST('get_valid_ROIS')
    rid_list = ibs.get_valid_rids()
    assert len(rid_list) > 0, 'database rois cannot be empty for TEST_COMPUTE_CHIPS'
    print(' * len(rid_list) = %r' % len(rid_list))
    preproc_chip.compute_and_write_chips_lazy(ibs, rid_list)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0')
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = __testing__.run_test(TEST_COMPUTE_CHIPS, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
