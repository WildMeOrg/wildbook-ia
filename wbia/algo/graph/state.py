# -*- coding: utf-8 -*-
import logging
import utool as ut
import wbia.constants as const

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

POSTV = const.EVIDENCE_DECISION.CODE.POSITIVE
NEGTV = const.EVIDENCE_DECISION.CODE.NEGATIVE
INCMP = const.EVIDENCE_DECISION.CODE.INCOMPARABLE
UNREV = const.EVIDENCE_DECISION.CODE.UNREVIEWED
UNKWN = const.EVIDENCE_DECISION.CODE.UNKNOWN

SAME = const.META_DECISION.CODE.SAME
DIFF = const.META_DECISION.CODE.DIFF
NULL = const.META_DECISION.CODE.NULL

UNINFERABLE = (INCMP, UNREV, UNKWN)
