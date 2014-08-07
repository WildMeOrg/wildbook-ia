#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function


def transfer_data(ibs_src, ibs_dst, gid_list1=None, aid_list1=None):
    """
    >>> from ibeis.all_imports import *
    >>> ibs1 = ibeis.opendb('testdb1')
    >>> ibs2 = ibeis.opendb('testdb_dst', allow_newdir=True)
    >>> gid_list1 = None
    >>> aid_list1 = None
    """
    ibs1 = ibs_src
    ibs2 = ibs_dst
    if gid_list1 is None:
        gid_list1 = ibs1.get_valid_gids()
