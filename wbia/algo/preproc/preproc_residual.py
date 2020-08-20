# -*- coding: utf-8 -*-
"""
"""
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[preproc_residual]')
logger = logging.getLogger('wbia')


def add_residual_params_gen(ibs, fid_list, qreq_=None):
    return None


def on_delete(ibs, featweight_rowid_list):
    logger.info('Warning: Not Implemented')
    logger.info('Probably nothing to do here')
