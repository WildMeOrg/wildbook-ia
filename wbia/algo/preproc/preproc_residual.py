# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[preproc_residual]')


def add_residual_params_gen(ibs, fid_list, qreq_=None):
    return None


def on_delete(ibs, featweight_rowid_list):
    print('Warning: Not Implemented')
    print('Probably nothing to do here')


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    testable_list = []
    ut.doctest_funcs(testable_list)
