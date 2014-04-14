#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#-----
TEST_NAME = 'TEST_COMPUTE_CHIPS'
#-----
try:
    import __testing__
except ImportError:
    from tests import __testing__
import multiprocessing
import utool
# IBEIST
from ibeis.model.preproc import preproc_chip
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext2(TEST_NAME)
def TEST_COMPUTE_CHIPS():
    # Create a HotSpotter API (hs) and GUI backend (back)
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    printTEST('get_valid_ROIS')
    rid_list = ibs.get_valid_rids()
    assert len(rid_list) > 0, 'database rois cannot be empty for ' + TEST_NAME
    print(' * len(rid_list) = %r' % len(rid_list))
    preproc_chip.compute_and_write_chips_lazy(ibs, rid_list)
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=RUNGUI)
    return main_locals


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = TEST_COMPUTE_CHIPS()
    exec(test_locals['execstr'])
