# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# UTool
import utool

(print, rrr, profile) = utool.inject2(__name__)


# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_rvecs]', DEBUG=False
)


def add_rvecs_params_gen(ibs, nInput=None):
    pass


def generate_rvecs(vecs_list, words):
    pass
