from __future__ import absolute_import, division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[results]', DEBUG=False)
from ibeis.dev import results_organizer
from ibeis.dev import results_analyzer


class AllResults(utool.DynStruct):
    """
    Data container for all compiled results
    """
    def __init__(allres):
        super(AllResults, allres).__init__(child_exclude_list=['qrid2_qres'])
        allres.ibs = None
        allres.qrid2_qres = None
        allres.allorg = None
        allres.cfgstr = None

    def get_orgtype(allres, orgtype):
        orgres = allres.allorg.get(orgtype)
        return orgres

    def get_qres(allres, qrid):
        return allres.qrid2_qres[qrid]

    def get_orgres_desc_match_dists(allres, orgtype_list):
        return results_analyzer.get_orgres_desc_match_dists(allres, orgtype_list)

    def get_orgres_roimatch_scores(allres, orgtype_list):
        return results_analyzer.get_orgres_roimatch_scores(allres, orgtype_list)


def init_allres(ibs, qrid2_qres):
    allres_cfgstr = ibs.qreq.get_cfgstr()
    print('Building allres')
    allres = AllResults()
    allres.qrid2_qres = qrid2_qres
    allres.allorg = results_organizer.organize_results(ibs, qrid2_qres)
    allres.cfgstr = allres_cfgstr
    allres.ibs = ibs
    return allres
