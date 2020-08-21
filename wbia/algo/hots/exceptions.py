# -*- coding: utf-8 -*-

# import utool
# print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[hsexcept]', DEBUG=False)
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qaid):
    msg = ('QUERY ERROR IN %s: qaid=%r has no descriptors!' + 'Please delete it.') % (
        ibs.get_dbname(),
        qaid,
    )
    ex = QueryException(msg)
    return ex


class HotsCacheMissError(Exception):
    pass


class HotsNeedsRecomputeError(Exception):
    pass
