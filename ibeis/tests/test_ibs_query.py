#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
try:
    import __testing__
except ImportError:
    pass
# Python
from os.path import join, exists  # NOQA
import multiprocessing
# Tools
import utool
from plottool import draw_func2 as df2
#IBEIS
from ibeis.dev import params  # NOQA
from ibeis.viz import interact
from ibeis.model.hots import QueryRequest  # NOQA
from ibeis.model.hots import NNIndex  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_QUERY]')


def TEST_QUERY(ibs, qrid_list=None):
    if qrid_list is None:
        qrid_list = ibs.get_valid_rids()[0:1]
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
        #viz_matches.show_matches(ibs, qres, rid2, fnum=fnum, in_image=True)
        #viz.show_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        interact.ishow_qres(ibs, qres, fnum=fnum, top_rids=top_rids,
                            ensure=False, annote_mode=1)
        df2.set_figtitle('Query Result')
        df2.adjust_subplots_safe(top=.8)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb_big')
    ibs = main_locals['ibs']
    test_locals = __testing__.run_test(TEST_QUERY, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
