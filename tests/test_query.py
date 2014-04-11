#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'BUILDQUERY'
#------
import __testing__
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
from ibeis.view import viz, viz_matches
from ibeis.model.jon_recognition import QueryRequest  # NOQA
from ibeis.model.jon_recognition import NNIndex  # NOQA
import query_helpers

printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


@__testing__.testcontext
def BUILDQUERY():
    main_locals = __testing__.main(defaultdb='test_big_ibeis',
                                   allow_newdir=True, nogui=True)
    ibs = main_locals['ibs']    # IBEIS Control

    ibs._init_query_requestor()
    qreq = ibs.qreq

    cids = ibs.get_recognition_database_chips()
    #nn_index = NNIndex.NNIndex(ibs, cid_list)
    index = 4
    qcids = utool.safe_slice(cids, index, index + 1)

    comp_locals_ = query_helpers.get_query_components(ibs, qcids)
    qres_dict = OrderedDict([
        ('ORIG', comp_locals_['qres_ORIG']),
        ('FILT', comp_locals_['qres_FILT']),
        ('SVER', comp_locals_['qres_SVER']),
    ])

    top_cids = qres_dict['SVER'].get_top_cids(ibs)
    top_cids = utool.safe_slice(top_cids, 6)
    cid2 = top_cids[0]

    for px, (label, qres) in enumerate(qres_dict.iteritems()):
        continue
        print(label)
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        #viz_matches.show_chipres(ibs, qres, cid2, fnum=fnum, in_image=True)
        viz_matches.show_qres(ibs, qres, fnum=fnum, top_cids=top_cids, ensure=False)
        df2.set_figtitle(label)
        df2.adjust_subplots_safe(top=.8)

    fnum = df2.next_fnum()

    qcid2_svtups = comp_locals_['qcid2_svtups']
    qcid2_chipmatch_FILT = comp_locals_['qcid2_chipmatch_FILT']
    qcid = comp_locals_['qcid']
    cid2_svtup    = qcid2_svtups[qcid]
    chipmatch_FILT = qcid2_chipmatch_FILT[qcid]
    viz.viz_spatial_verification(ibs, qcid, cid2, chipmatch_FILT, cid2_svtup,
                                 fnum=fnum)
    df2.present(wh=900)
    comp_locals_.update(locals())

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=False)
    return main_locals
BUILDQUERY.func_name = TEST_NAME


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = BUILDQUERY()
    if utool.get_flag('--cmd2'):
        exec(utool.execstr_dict(test_locals, 'test_locals'))
        exec(utool.execstr_embed())
