# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

import utool as ut
ut.noinject(__name__, '[ibeis.model.hots.__init__]')

from ibeis.model.hots import automated_helpers
from ibeis.model.hots import automated_matcher
from ibeis.model.hots import exceptions
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
print, rrr, profile = ut.inject2(__name__, '[ibeis.model.hots]')


def reload_subs(verbose=True):
    """ Reloads ibeis.model.hots and submodules """
    rrr(verbose=verbose)
    getattr(automated_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(automated_matcher, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(exceptions, 'rrr', lambda verbose: None)(verbose=verbose)
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
