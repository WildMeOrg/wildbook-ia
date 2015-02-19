### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function

import utool as ut
ut.noinject(__name__, '[ibeis.model.hots.__init__]', DEBUG=False)

from ibeis.model.hots import automated_helpers
from ibeis.model.hots import automated_matcher
from ibeis.model.hots import exceptions
from ibeis.model.hots import hots_query_result
from ibeis.model.hots import hstypes
from ibeis.model.hots import match_chips4
from ibeis.model.hots import name_scoring
from ibeis.model.hots import neighbor_index
from ibeis.model.hots import multi_index
from ibeis.model.hots import nn_weights
from ibeis.model.hots import pipeline
from ibeis.model.hots import precision_recall
from ibeis.model.hots import query_helpers
from ibeis.model.hots import query_request
from ibeis.model.hots import _pipeline_helpers
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[ibeis.model.hots]')


def reload_subs(verbose=True):
    """ Reloads ibeis.model.hots and submodules """
    rrr(verbose=verbose)
    getattr(automated_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(automated_matcher, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(exceptions, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(hots_query_result, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(hstypes, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(match_chips4, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(name_scoring, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(neighbor_index, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(multi_index, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(nn_weights, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(pipeline, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(precision_recall, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(query_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(query_request, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('automated_helpers', None),
    ('automated_matcher', None),
    ('exceptions', None, False),
    ('hots_query_result', None, False),
    ('hstypes', None, False),
    ('match_chips4', None, False),
    ('name_scoring', None, False),
    ('neighbor_index', None, False),
    ('multi_index', None, False),
    ('nn_weights', None, False),
    ('pipeline', None, False),
    ('precision_recall', None, False),
    ('query_helpers', None, False),
    ('query_request', None, False),
    ('_pipeline_helpers', None, False),
]
"""
Regen Command:
    makeinit.py -x smk word_index --modname ibeis.model.hots
"""

## flake8: noqa
#from __future__ import absolute_import, division, print_function
#from . import hots_query_result
#from . import match_chips4
#from . import neighbor_index
#from . import nn_weights
#from . import pipeline
#from . import query_helpers
#from . import query_request


#import utool
#print, print_, printDBG, rrr, profile = utool.inject(
#    __name__, '[hots]')

#def reload_subs():
#    """ Reloads hots and submodules """
#    rrr()
#    getattr(hots_query_result, 'rrr', lambda: None)()
#    getattr(match_chips4, 'rrr', lambda: None)()
#    getattr(neighbor_index, 'rrr', lambda: None)()
#    getattr(nn_weights, 'rrr', lambda: None)()
#    getattr(pipeline, 'rrr', lambda: None)()
#    getattr(query_helpers, 'rrr', lambda: None)()
#    getattr(query_request, 'rrr', lambda: None)()
#    rrr()


## HotSpotter User Interface
## MAKE A WALL HERE (NOT YET IMPLEMENTED)

#__QUERY_REQUESTOR__ = None  # THERE IS ONLY ONE QUERY REQUESTOR

#def query(ibs, qaid_list, daid_list):
#    pass
