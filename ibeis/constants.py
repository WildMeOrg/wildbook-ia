class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb  = '_ibeis_database.sqlite3'
    _ibsdb = '_ibsdb'
    cache  = '_ibeis_cache'
    chips  = 'chips'
    flann  = 'flann'
    images = 'images'
    qres   = 'qres'
    bigcache = 'bigcache'
    detectimg = 'detectimg'

# Names normalized to the standard UNKNOWN_NAME
ACCEPTED_UNKNOWN_NAMES = set(['Unassigned'])


# Name used to denote that idkwtfthisis
UNKNOWN_NAME = '____'
