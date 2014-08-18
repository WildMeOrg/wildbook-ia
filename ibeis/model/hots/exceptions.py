
class QueryException(Exception):
    """ </CYTH> """
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qaid):
    """ </CYTH> """
    msg = ('QUERY ERROR IN %s: qaid=%r has no descriptors!' +
           'Please delete it.') % (ibs.get_dbname(), qaid)
    ex = QueryException(msg)
    return ex
