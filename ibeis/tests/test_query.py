#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'TEST_QUERY'
#------
try:
    import __testing__
    printTEST = __testing__.printTEST
except ImportError:
    printTEST = print
    pass
# Python
import sys
from os.path import join, exists  # NOQA
import multiprocessing
# Tools
import utool
from plottool import draw_func2 as df2
#IBEIS
from ibeis.dev import params  # NOQA
from ibeis import interact
from ibeis.model.hots import QueryRequest  # NOQA
from ibeis.model.hots import NNIndex  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


def TEST_QUERY(ibs=None, qrid_list=None):
    ibs._init_query_requestor()
    qreq = ibs.qreq
    #query_helpers.find_matchable_chips(ibs)
    rids = ibs.get_recognition_database_rois()
    qres_dict = ibs.query_database(qrid_list)

    for qrid in qrid_list:
        qres  = qres_dict[qrid]
        top_rids = qres.get_top_rids(ibs)
        top_rids = utool.safe_slice(top_rids, 3)
        rid2 = top_rids[0]
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        #viz_matches.show_chipres(ibs, qres, rid2, fnum=fnum, in_image=True)
        #viz.show_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        interact.interact_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        df2.set_figtitle('Query Result')
        df2.adjust_subplots_safe(top=.8)
    return locals()

try:
    TEST_QUERY = __testing__.testcontext2(TEST_NAME)(TEST_QUERY)
except Exception:
    pass


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    main_locals = __testing__.main(defaultdb='test_big_ibeis', allow_newdir=True, nogui=True)
    ibs = main_locals['ibs']
    qrid_list = [0]
    test_locals = TEST_QUERY(ibs, qrid_list)
    execstr = __testing__.main_loop(main_locals, rungui=False)
    df2.present()
    exec(execstr)
