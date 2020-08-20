# -*- coding: utf-8 -*-
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

default = {
    'fail': None,
    'success': None,
    'min_gtrank': None,
    'max_gtrank': None,
    'min_gf_timedelta': None,
    'orderby': None,  # 'gt_timedelta'
    'reverse': None,  # 'gt_timedelta'
}
