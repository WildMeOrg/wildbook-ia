# -*- coding: utf-8 -*-
import logging

import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


class NeedsUserInput(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class UserCancel(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class InvalidRequest(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)
