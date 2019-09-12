# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


class NeedsUserInput(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class UserCancel(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class InvalidRequest(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)
