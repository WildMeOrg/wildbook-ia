#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
#------
TEST_NAME = 'BUILDQUERY'
#------
try:
    import __testing__
except ImportError:
    import tests.__testing__ as __testing__
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
from ibeis.view import viz_matches
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
    qcids = cids[2:3]

    qres_ORIG, qres_FILT, qres_SVER, comp_locals_ = query_helpers.get_query_components(ibs, qcids)
    qres_dict = OrderedDict([
        ('ORIG', qres_ORIG),
        ('FILT', qres_FILT),
        ('SVER', qres_SVER),
    ])

    top_cids_ORIG = qres_ORIG.get_top_cids(ibs)
    top_cids_FILT = qres_FILT.get_top_cids(ibs)
    top_cids_SVER = qres_SVER.get_top_cids(ibs)
    cid2 = top_cids_ORIG[0]

    for px, (label, qres) in enumerate(qres_dict.iteritems()):
        print(label)
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        top_cids = top_cids_SVER[0:min(len(top_cids_SVER), 6)]
        #viz_matches.show_chipres(ibs, qres, cid2, fnum=fnum, in_image=True)
        viz_matches.show_qres(ibs, qres, fnum=fnum, top_cids=top_cids, ensure=False)
        df2.set_figtitle(label)
        df2.adjust_subplots_safe(top=.8)
    df2.present(wh=900)
    comp_locals_.update(locals())

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=False)
    return main_locals
BUILDQUERY.func_name = TEST_NAME


if __name__ == '__main__':
    from ibeis.model.jon_recognition.matching_functions import *  # NOQA
    # For windows
    multiprocessing.freeze_support()
    test_locals = BUILDQUERY()
    del test_locals['print']
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    #from drawtool import draw_func2 as df2
    #exec(df2.present())
