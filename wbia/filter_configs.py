# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

default = {
    'fail': None,
    'success': None,
    'min_gtrank': None,
    'max_gtrank': None,
    'min_gf_timedelta': None,
    'orderby': None,  # 'gt_timedelta'
    'reverse': None,  # 'gt_timedelta'
}
