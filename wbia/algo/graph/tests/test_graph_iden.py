# -*- coding: utf-8 -*-
import logging
import utool as ut
from wbia.algo.graph import demo
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV


(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


# FIXME failing-test (22-Jul-2020) This test is failing and it's not clear how to fix it
def _test_incomp_inference():
    infr = demo.demodata_infr(num_pccs=0)
    # Make 2 consistent and 2 inconsistent CCs
    infr.add_feedback((1, 2), POSTV)
    infr.add_feedback((2, 3), POSTV)
    infr.add_feedback((3, 4), POSTV)
    infr.add_feedback((4, 1), POSTV)
    # -----
    infr.add_feedback((11, 12), POSTV)
    infr.add_feedback((12, 13), POSTV)
    infr.add_feedback((13, 14), POSTV)
    infr.add_feedback((14, 11), POSTV)
    infr.add_feedback((12, 14), NEGTV)
    # -----
    infr.add_feedback((21, 22), POSTV)
    infr.add_feedback((22, 23), POSTV)
    infr.add_feedback((23, 21), NEGTV)
    # -----
    infr.add_feedback((31, 32), POSTV)
    infr.add_feedback((32, 33), POSTV)
    infr.add_feedback((33, 31), POSTV)
    infr.add_feedback((2, 32), NEGTV)
    infr.add_feedback((3, 33), NEGTV)
    infr.add_feedback((12, 21), NEGTV)
    # -----
    # Incomparable within CCs
    logger.info('==========================')
    infr.add_feedback((1, 3), INCMP)
    infr.add_feedback((1, 4), INCMP)
    infr.add_feedback((1, 2), INCMP)
    infr.add_feedback((11, 13), INCMP)
    infr.add_feedback((11, 14), INCMP)
    infr.add_feedback((11, 12), INCMP)
    infr.add_feedback((1, 31), INCMP)
    infr.add_feedback((2, 32), INCMP)
    infr.add_feedback((12, 21), INCMP)
    infr.add_feedback((23, 21), INCMP)
    infr.add_feedback((12, 14), INCMP)
    logger.info('Final state:')
    logger.info(ut.repr4(sorted(infr.gen_edge_attrs('decision'))))


# FIXME failing-test (22-Jul-2020) This test is failing and it's not clear how to fix it
def _test_unrev_inference():
    infr = demo.demodata_infr(num_pccs=0)
    # Make 2 consistent and 2 inconsistent CCs
    infr.add_feedback((1, 2), POSTV)
    infr.add_feedback((2, 3), POSTV)
    infr.add_feedback((3, 4), POSTV)
    infr.add_feedback((4, 1), POSTV)
    # -----
    infr.add_feedback((11, 12), POSTV)
    infr.add_feedback((12, 13), POSTV)
    infr.add_feedback((13, 14), POSTV)
    infr.add_feedback((14, 11), POSTV)
    infr.add_feedback((12, 14), NEGTV)
    # -----
    infr.add_feedback((21, 22), POSTV)
    infr.add_feedback((22, 23), POSTV)
    infr.add_feedback((23, 21), NEGTV)
    # -----
    infr.add_feedback((31, 32), POSTV)
    infr.add_feedback((32, 33), POSTV)
    infr.add_feedback((33, 31), POSTV)
    infr.add_feedback((2, 32), NEGTV)
    infr.add_feedback((3, 33), NEGTV)
    infr.add_feedback((12, 21), NEGTV)
    # -----
    # Incomparable within CCs
    logger.info('==========================')
    infr.add_feedback((1, 3), UNREV)
    infr.add_feedback((1, 4), UNREV)
    infr.add_feedback((1, 2), UNREV)
    infr.add_feedback((11, 13), UNREV)
    infr.add_feedback((11, 14), UNREV)
    infr.add_feedback((11, 12), UNREV)
    infr.add_feedback((1, 31), UNREV)
    infr.add_feedback((2, 32), UNREV)
    infr.add_feedback((12, 21), UNREV)
    infr.add_feedback((23, 21), UNREV)
    infr.add_feedback((12, 14), UNREV)
    logger.info('Final state:')
    logger.info(ut.repr4(sorted(infr.gen_edge_attrs('decision'))))


# FIXME failing-test (22-Jul-2020) This test is failing and it's not clear how to fix it
def _test_pos_neg():
    infr = demo.demodata_infr(num_pccs=0)
    # Make 3 inconsistent CCs
    infr.add_feedback((1, 2), POSTV)
    infr.add_feedback((2, 3), POSTV)
    infr.add_feedback((3, 4), POSTV)
    infr.add_feedback((4, 1), POSTV)
    infr.add_feedback((1, 3), NEGTV)
    # -----
    infr.add_feedback((11, 12), POSTV)
    infr.add_feedback((12, 13), POSTV)
    infr.add_feedback((13, 11), NEGTV)
    # -----
    infr.add_feedback((21, 22), POSTV)
    infr.add_feedback((22, 23), POSTV)
    infr.add_feedback((23, 21), NEGTV)
    # -----
    # Fix inconsistency
    infr.add_feedback((23, 21), POSTV)
    # Merge inconsistent CCS
    infr.add_feedback((1, 11), POSTV)
    # Negative edge within an inconsistent CC
    infr.add_feedback((2, 13), NEGTV)
    # Negative edge external to an inconsistent CC
    infr.add_feedback((12, 21), NEGTV)
    # -----
    # Make inconsistency from positive
    infr.add_feedback((31, 32), POSTV)
    infr.add_feedback((33, 34), POSTV)
    infr.add_feedback((31, 33), NEGTV)
    infr.add_feedback((32, 34), NEGTV)
    infr.add_feedback((31, 34), POSTV)
    # Fix everything
    infr.add_feedback((1, 3), POSTV)
    infr.add_feedback((2, 4), POSTV)
    infr.add_feedback((32, 34), POSTV)
    infr.add_feedback((31, 33), POSTV)
    infr.add_feedback((13, 11), POSTV)
    infr.add_feedback((23, 21), POSTV)
    infr.add_feedback((1, 11), NEGTV)
    logger.info('Final state:')
    logger.info(ut.repr4(sorted(infr.gen_edge_attrs('decision'))))
