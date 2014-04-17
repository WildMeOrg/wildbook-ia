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
from collections import OrderedDict
import multiprocessing
# Tools
import utool
from drawtool import draw_func2 as df2
#IBEIS
from ibeis.dev import params  # NOQA
from ibeis.view import viz
from ibeis.model.hots import QueryRequest  # NOQA
from ibeis.model.hots import NNIndex  # NOQA
from . import query_helpers
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


def TEST_QUERY(ibs=None):
    if ibs is None:
        print('ibs is none')
        main_locals = __testing__.main(defaultdb='test_big_ibeis',
                                       allow_newdir=True, nogui=True)
        ibs = main_locals['ibs']    # IBEIS Control

    ibs._init_query_requestor()
    qreq = ibs.qreq

    #query_helpers.find_matchable_chips(ibs)

    rids = ibs.get_recognition_database_rois()
    #nn_index = NNIndex.NNIndex(ibs, rid_list)
    index = 1
    index = utool.get_arg('--index', type_=int, default=index)
    qrids = utool.safe_slice(rids, index, index + 1)

    comp_locals_ = query_helpers.get_query_components(ibs, qrids)
    qres_dict = OrderedDict([
        #('ORIG', comp_locals_['qres_ORIG']),
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
        #viz_matches.show_chipres(ibs, qres, rid2, fnum=fnum, in_image=True)
        viz.show_qres(ibs, qres, fnum=fnum, top_rids=top_rids, ensure=False)
        df2.set_figtitle(label)
        df2.adjust_subplots_safe(top=.8)

    fnum = df2.next_fnum()

    qrid2_svtups = comp_locals_['qrid2_svtups']
    qrid2_chipmatch_FILT = comp_locals_['qrid2_chipmatch_FILT']
    rid1 = qrid = comp_locals_['qrid']
    rid2_svtup  = qrid2_svtups[rid1]
    chipmatch_FILT = qrid2_chipmatch_FILT[rid1]
    viz.show_sv(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup, fnum=fnum)
    df2.present(wh=900)
    comp_locals_.update(locals())

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    if main_locals is None:
        main_locals.update(locals())
        __testing__.main_loop(main_locals, rungui=False)
    return locals()

try:
    TEST_QUERY = __testing__.testcontext2(TEST_NAME)(TEST_QUERY)
except Exception:
    pass


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_QUERY()
    exec(test_locals['execstr'])
