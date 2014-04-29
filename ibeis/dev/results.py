from __future__ import absolute_import, division, print_function
import utool
from utool import DynStruct
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[results]', DEBUG=False)
from ibeis.dev import result_organizer
from ibeis.dev import result_analyzer


class AllResults(DynStruct):
    """
    Data container for all compiled results
    """
    def __init__(allres, ibs, qrid2_qres):
        allres.ibs = ibs
        allres.qrid2_qres = qrid2_qres
        allres.allorg = None
        allres.uid = None

    def get_orgtype(allres, orgtype):
        orgres = allres.allorg.get(orgtype)
        return orgres

    def get_fm(allres, qrid, rid):
        return allres.qrid2_qres[qrid].rid2_fm[rid]

    def get_qres(allres, qrid):
        return allres.qrid2_qres[qrid]

    def get_matching_distances(allres, orgtype):
        result_analyzer.get_orgres_descriptor_matches(allres, orgtype_=orgtype)

__ALLRES_CACHE__ = {}


def init_allres(ibs, qrid2_qres):
    global __ALLRES_CACHE__
    allres_uid = ibs.qreq.get_uid()
    try:
        return __ALLRES_CACHE__[allres_uid]
    except KeyError:
        try:
            allres = utool.load_testdata('allres')
        except Exception:
            allres = AllResults(ibs, qrid2_qres)
            allres.allorg = result_organizer.organize_results(ibs, qrid2_qres)
            allres.uid = allres_uid
            utool.save_testdata('allres')
        __ALLRES_CACHE__[allres_uid] = allres
    return allres
