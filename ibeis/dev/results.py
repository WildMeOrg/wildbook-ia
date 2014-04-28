from __future__ import absolute_import, division, print_function
import utool
from utool import DynStruct
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[results]', DEBUG=False)
from ibeis.dev import result_organizer


class AllResults(DynStruct):
    """
    Data container for all compiled results
    """
    def __init__(self, ibs, qrid2_qres):
        self.ibs = ibs
        self.qrid2_qres = qrid2_qres
        self.allorg = None
        self.uid = None

    def get_orgtype(self, orgtype):
        orgres = self.allorg.get(orgtype)
        return orgres

    def get_fm(self, qrid, rid):
        return self.qrid2_qres[qrid].rid2_fm[rid]

    def get_qres(self, qrid):
        return self.qrid2_qres[qrid]


def init_allres(ibs, qrid2_qres):
    allres = AllResults(ibs, qrid2_qres)
    allres.allorg = result_organizer.organize_results(ibs, qrid2_qres)
    allres.uid = ibs.qreq.get_uid()
    return allres
