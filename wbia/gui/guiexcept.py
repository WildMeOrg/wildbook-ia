# -*- coding: utf-8 -*-
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
