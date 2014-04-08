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
from ibeis.model.jon_recognition import matching_functions as mf
from ibeis.model.jon_recognition import match_chips3 as mc3
from ibeis.model.jon_recognition import QueryRequest  # NOQA
from ibeis.model.jon_recognition import NNIndex  # NOQA

printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


def get_query_components(ibs, qcids):
    ibs._init_query_requestor()
    qreq = ibs.qreq
    dcids = ibs.get_recognition_database_chips()
    qreq = mc3.prep_query_request(qreq=qreq, qcids=qcids, dcids=dcids)
    mc3.pre_exec_checks(ibs, qreq)
    neighbs = mf.nearest_neighbors(ibs, qcids, qreq)
    qcid2_nns = neighbs
    weights, filt2_meta = mf.weight_neighbors(ibs, neighbs, qreq)
    filt2_weights, filt2_meta = (weights, filt2_meta)
    nnfiltFILT = mf.filter_neighbors(ibs, neighbs, weights, qreq)
    matchesFILT = mf.build_chipmatches(ibs, neighbs, nnfiltFILT, qreq)
    matchesSVER = mf.spatial_verification(ibs, matchesFILT, qreq)
    qcid2_chipmatch = matchesSVER
    qcid2_res = mf.chipmatch_to_resdict(ibs, matchesSVER, filt2_meta, qreq)
    return locals()


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

    try:
        '''
        Driver logic of query pipeline
        Input:
            ibs  - IBEIS database object to be queried
            qreq - QueryRequest Object   # use prep_qreq to create one
        Output:
            qcid2_res - mapping from query indexes to QueryResult Objects
        '''
        # Query Chip Indexes
        # * vsone qcids/dcids swapping occurs here
        qcids = qreq.get_internal_qcids()
        qreq = ibs.qreq
        mf.rrr()
        # Nearest neighbors (qcid2_nns)
        # * query descriptors assigned to database descriptors
        # * FLANN used here
        with utool.Timer('1) Nearest Neighbors'):
            neighbs = mf.nearest_neighbors(ibs, qcids, qreq)
            qcid2_nns = neighbs
        # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
        # * feature matches are weighted
        with utool.Timer('2) Weight Neighbors'):
            weights, filt2_meta = mf.weight_neighbors(ibs, neighbs, qreq)
            filt2_weights, filt2_meta = (weights, filt2_meta)
        # Thresholding and weighting (qcid2_nnfilter)
        # * feature matches are pruned
        #
        with utool.Timer('3) Filter Neighbors'):
            nnfiltFILT = mf.filter_neighbors(ibs, neighbs, weights, qreq)
        # Nearest neighbors to chip matches (qcid2_chipmatch)
        # * Inverted index used to create cid2_fmfsfk (TODO: ccid2_fmfv)
        # * Initial scoring occurs
        # * vsone inverse swapping occurs here
        with utool.Timer('4) Filter Neighbors'):
            matchesFILT = mf.build_chipmatches(ibs, neighbs, nnfiltFILT, qreq)
        # Spatial verification (qcid2_chipmatch) (TODO: cython)
        # * prunes chip results and feature matches
        with utool.Timer('5) Filter Neighbors'):
            matchesSVER = mf.spatial_verification(ibs, matchesFILT, qreq)
            qcid2_chipmatch = matchesSVER
        # Query results format (qcid2_res) (TODO: SQL / Json Encoding)
        # * Final Scoring. Prunes chip results.
        # * packs into a wrapped query result object
        with utool.Timer('6) Filter Neighbors'):
            qcid2_res = mf.chipmatch_to_resdict(ibs, matchesSVER, filt2_meta, qreq)
    except Exception as ex:
        print(ex)

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
