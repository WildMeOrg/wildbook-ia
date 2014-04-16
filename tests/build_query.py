#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'BUILDQUERY'
#------
import __testing__  # Should be imported before any ibeis stuff
import sys
from os.path import join, exists  # NOQA
from ibeis.dev import params  # NOQA
import multiprocessing
import utool
from ibeis.model.hots import matching_functions as mf
from ibeis.model.hots import match_chips3 as mc3

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

    rids = ibs.get_recognition_database_rois()
    assert len(rids) > 0, 'need chips'
    #nn_index = NNIndex.NNIndex(ibs, rid_list)
    qreq = mc3.prep_query_request(qreq=qreq, qrids=rids[0], drids=rids)
    mc3.pre_exec_checks(ibs, qreq)

    try:
        '''
        Driver logic of query pipeline
        Input:
            ibs  - IBEIS database object to be queried
            qreq - QueryRequest Object   # use prep_qreq to create one
        Output:
            qrid2_res - mapping from query indexes to QueryResult Objects
        '''
        # Query Chip Indexes
        # * vsone qrids/drids swapping occurs here
        qrids = qreq.get_internal_qrids()
        qreq = ibs.qreq
        #mf.rrr()
        qrids = qreq.get_internal_qrids()
        if isinstance(qrids, int):
            qrids = [qrids]
        qrid2_nns = mf.nearest_neighbors(
            ibs, qrids, qreq)
        filt2_weights, filt2_meta = mf.weight_neighbors(
            ibs, qrid2_nns, qreq)
        qrid2_nnfilt = mf.filter_neighbors(
            ibs, qrid2_nns, filt2_weights, qreq)
        qrid2_chipmatch_FILT = mf.build_chipmatches(
            qrid2_nns, qrid2_nnfilt, qreq)
        qrid2_chipmatch_SVER = mf.spatial_verification(
            ibs, qrid2_chipmatch_FILT, qreq, dbginfo=False)
        qrid2_res = mf.chipmatch_to_resdict(
            ibs, qrid2_chipmatch_SVER, filt2_meta, qreq)
    except Exception as ex:
        utool.print_exception(ex, '[!build_query]')
        raise

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    main_locals.update(globals())
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=False)
    return main_locals
BUILDQUERY.func_name = TEST_NAME


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = BUILDQUERY()
    del test_locals['print']
    if utool.get_flag('--cmd2'):
        exec(utool.execstr_dict(test_locals, 'test_locals'))
        exec(utool.execstr_embed())
    if utool.get_flag('--wait'):
        print('waiting')
        raw_input('press enter')
