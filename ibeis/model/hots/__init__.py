# flake8: noqa
from __future__ import absolute_import, division, print_function
from . import hots_nn_index
from . import hots_query_request
from . import hots_query_result
#from . import coverage_image
#from . import encounter
from . import match_chips3
from . import matching_functions
from . import nn_filters
from . import query_helpers
from . import voting_rules2
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[hots]')

def reload_subs():
    """ Reloads hots and submodules """
    rrr()
    getattr(hots_nn_index, 'rrr', lambda: None)()
    getattr(hots_query_request, 'rrr', lambda: None)()
    getattr(hots_query_result, 'rrr', lambda: None)()
    #getattr(coverage_image, 'rrr', lambda: None)()
    #getattr(encounter, 'rrr', lambda: None)()
    getattr(match_chips3, 'rrr', lambda: None)()
    getattr(matching_functions, 'rrr', lambda: None)()
    getattr(nn_filters, 'rrr', lambda: None)()
    getattr(query_helpers, 'rrr', lambda: None)()
    getattr(voting_rules2, 'rrr', lambda: None)()
    rrr()
rrrr = reload_subs

# HotSpotter User Interface
# MAKE A WALL HERE (NOT YET IMPLEMENTED)

__QUERY_REQUESTOR__ = None  # THERE IS ONLY ONE QUERY REQUESTOR

def query(ibs, qaid_list, daid_list):
    pass
