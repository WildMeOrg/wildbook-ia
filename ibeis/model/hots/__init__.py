# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[hots]', DEBUG=False)

from . import match_chips3
from . import matching_functions
from . import nn_filters
from . import NNIndex
from . import query_helpers
from . import QueryRequest
from . import QueryResult
from . import voting_rules2


def reload_subs():
    rrr()
    match_chips3.rrr()
    matching_functions.rrr()
    nn_filters.rrr()
    NNIndex.rrr()
    query_helpers.rrr()
    QueryRequest.rrr()
    QueryResult.rrr()
    voting_rules2.rrr()

rrrr = reload_subs

# HotSpotter User Interface
# MAKE A WALL HERE (NOT YET IMPLEMENTED)

__QUERY_REQUESTOR__ = None  # THERE IS ONLY ONE QUERY REQUESTOR

def query(ibs, qrid_list, drid_list):
    pass
