# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

import utool as ut
ut.noinject(__name__, '[ibeis.algo.hots.__init__]')

#from ibeis.algo.hots import automated_helpers
#from ibeis.algo.hots import automated_matcher
from ibeis.algo.hots import exceptions
from ibeis.algo.hots import hstypes
from ibeis.algo.hots import match_chips4
from ibeis.algo.hots import name_scoring
from ibeis.algo.hots import neighbor_index
from ibeis.algo.hots import multi_index
from ibeis.algo.hots import nn_weights
from ibeis.algo.hots import pipeline
from ibeis.algo.hots import precision_recall
from ibeis.algo.hots import query_request
from ibeis.algo.hots import _pipeline_helpers
print, rrr, profile = ut.inject2(__name__, '[ibeis.algo.hots]')


def reload_subs(verbose=True):
    """ Reloads ibeis.algo.hots and submodules """
    rrr(verbose=verbose)
    #getattr(automated_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    #getattr(automated_matcher, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(exceptions, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(hstypes, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(match_chips4, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(name_scoring, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(neighbor_index, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(multi_index, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(nn_weights, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(pipeline, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(precision_recall, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(query_request, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
rrrr = reload_subs

IMPORT_TUPLES = [
    #('automated_helpers', None),
    #('automated_matcher', None),
    ('exceptions', None, False),
    ('hstypes', None, False),
    ('match_chips4', None, False),
    ('name_scoring', None, False),
    ('neighbor_index', None, False),
    ('multi_index', None, False),
    ('nn_weights', None, False),
    ('pipeline', None, False),
    ('precision_recall', None, False),
    ('query_request', None, False),
    ('_pipeline_helpers', None, False),
]
"""
Regen Command:
    makeinit.py -x smk word_index --modname ibeis.algo.hots
"""
