# -*- coding: utf-8 -*-
"""
TODO: rewrite the hotspotter lnbnn algo to be a generator

Wrapper around LNBNN hotspotter algorithm
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

print, rrr, profile = ut.inject2(__name__)


class Ranker(object):
    def __init__(ranker, ibs=None, config={}):
        ranker.ibs = ibs
        ranker.config = config

        ranker._daids = None
        ranker._nids = None
        ranker.verbose = True

    def fit(ranker, daids, dnids=None):
        ranker._daids = daids
        ranker._nids = dnids
        pass

    def predict(ranker, qaids, qnids=None, prog_hook=None):
        custom_nid_lookup = dict(zip(ranker._daids, ranker._dnids))
        custom_nid_lookup.update(dict(zip(qaids, qnids)))

        cfgdict = ranker.config
        ibs = ranker.ibs
        qreq_ = ibs.new_query_request(
            qaids,
            ranker._daids,
            cfgdict=cfgdict,
            custom_nid_lookup=custom_nid_lookup,
            verbose=ranker.verbose,
        )
        cm_list = qreq_.execute(prog_hook=prog_hook)
        return cm_list

        # ibs = ranker.ibs
        # aids = sorted(set(ut.aslist(qaids) + ut.aslist(daids)))
        # custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        # for qaid in qaids:
        #     pass
