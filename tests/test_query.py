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
import sys
from os.path import join, exists  # NOQA
from ibeis.dev import params  # NOQA
import multiprocessing
import utool
from ibeis.model.jon_recognition import match_chips3 as mc3
from ibeis.model.jon_recognition import QueryRequest  # NOQA
from ibeis.model.jon_recognition import NNIndex  # NOQA

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
    qreq = mc3.prep_query_request(qreq=qreq, qcids=cids[0:3], dcids=cids)
    mc3.pre_exec_checks(ibs, qreq)
    qcid2_qres = mc3.process_query_request(ibs, qreq)

    qres = qcid2_qres[cids[0]]

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
