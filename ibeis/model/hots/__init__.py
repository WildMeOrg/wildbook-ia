# flake8: noqa
from __future__ import absolute_import, division, print_function
from . import hots_query_result
from . import match_chips4
from . import neighbor_index
from . import nn_weights
from . import pipeline
from . import query_helpers
from . import query_request
from . import voting_rules2
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[hots]')

def reload_subs():
    """ Reloads hots and submodules """
    rrr()
    getattr(hots_query_result, 'rrr', lambda: None)()
    getattr(match_chips4, 'rrr', lambda: None)()
    getattr(neighbor_index, 'rrr', lambda: None)()
    getattr(nn_weights, 'rrr', lambda: None)()
    getattr(pipeline, 'rrr', lambda: None)()
    getattr(query_helpers, 'rrr', lambda: None)()
    getattr(query_request, 'rrr', lambda: None)()
    getattr(voting_rules2, 'rrr', lambda: None)()
    rrr()


# HotSpotter User Interface
# MAKE A WALL HERE (NOT YET IMPLEMENTED)

__QUERY_REQUESTOR__ = None  # THERE IS ONLY ONE QUERY REQUESTOR

def query(ibs, qaid_list, daid_list):
    pass
