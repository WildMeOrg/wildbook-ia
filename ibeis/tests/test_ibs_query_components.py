#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
# Python
from collections import OrderedDict
import multiprocessing
# Tools
import utool
from plottool import draw_func2 as df2
#IBEIS
from ibeis import viz
from ibeis.viz import interact
from ibeis.model.hots import query_helpers
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[TEST_QUERY_COMP]')


def TEST_QUERY_COMP(ibs, qrid_list=None):
    if qrid_list is None:
        qrid_list = ibs.get_valid_rids()[0:1]
    ibs._init_query_requestor()
    qreq = ibs.qreq

    #query_helpers.find_matchable_chips(ibs)

    rids = ibs.get_recognition_database_rois()
    #nn_index = NNIndex.NNIndex(ibs, rid_list)
    index = 0
    index = utool.get_arg('--index', type_=int, default=index)
    qrid_list = utool.safe_slice(rids, index, index + 1)

    comp_locals_ = query_helpers.get_query_components(ibs, qrid_list)
    qres_dict = OrderedDict([
        ('ORIG', comp_locals_['qres_ORIG']),
        ('FILT', comp_locals_['qres_FILT']),
        ('SVER', comp_locals_['qres_SVER']),
    ])

    top_rids = qres_dict['SVER'].get_top_rids(ibs)
    top_rids = utool.safe_slice(top_rids, 3)
    rid2 = top_rids[0]

    for px, (label, qres) in enumerate(qres_dict.iteritems()):
        print(label)
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        #viz_matches.show_matches(ibs, qres, rid2, fnum=fnum, in_image=True)
        #viz.show_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        interact.ishow_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        df2.set_figtitle(label)
        df2.adjust_subplots_safe(top=.8)

    fnum = df2.next_fnum()

    qrid2_svtups = comp_locals_['qrid2_svtups']
    qrid2_chipmatch_FILT = comp_locals_['qrid2_chipmatch_FILT']
    rid1 = qrid = comp_locals_['qrid']
    rid2_svtup  = qrid2_svtups[rid1]
    chipmatch_FILT = qrid2_chipmatch_FILT[rid1]
    viz.show_sver(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup, fnum=fnum)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb1')
    ibs = main_locals['ibs']
    test_locals = __testing__.run_test(TEST_QUERY_COMP, ibs)
    execstr     = __testing__.main_loop(test_locals)
    df2.present()
    exec(execstr)
