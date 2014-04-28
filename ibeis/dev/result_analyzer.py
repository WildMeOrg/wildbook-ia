from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from itertools import izip
from ibeis.dev import ibsfuncs
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[resorg]', DEBUG=False)


def _get_orgres2_descriptor_distances(allres, orgres_list=None):
    if orgres_list is None:
        orgres_list = ['true', 'false']
    dist_fn = lambda orgres: get_orgres_descriptor_matches(allres, orgres)
    orgres2_distance = {}
    for orgres in orgres_list:
        try:
            orgres2_distance[orgres] = dist_fn(orgres)
        except Exception as ex:
            utool.printex(ex, 'failed dist orgres=%r' % orgres)
    return orgres2_distance


def get_orgres_descriptor_matches(allres, orgtype_='false'):
    orgres = allres.get_orgtype(orgtype_)
    qrids = orgres.qrids
    rids  = orgres.rids
    printDBG('[rr2] getting orgtype_=%r distances between sifts' % orgtype_)
    adesc1, adesc2 = get_matching_descriptors(allres, qrids, rids)
    printDBG('[rr2]  * adesc1.shape = %r' % (adesc1.shape,))
    printDBG('[rr2]  * adesc2.shape = %r' % (adesc2.shape,))
    #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
    #dist_list = ['L1', 'L2', 'hist_isect']
    dist_list = ['L2', 'hist_isect']
    hist1 = np.asarray(adesc1, dtype=np.float64)
    hist2 = np.asarray(adesc2, dtype=np.float64)
    distances = utool.compute_distances(hist1, hist2, dist_list)
    return distances


def get_matching_descriptors(allres, qrids, rids):
    ibs = allres.ibs
    qdesc_cache = ibsfuncs.get_roi_desc_cache(ibs, qrids)
    rdesc_cache = ibsfuncs.get_roi_desc_cache(ibs, rids)
    desc1_list = []
    desc2_list = []
    for qrid, rid in izip(qrids, rids):
        fm = allres.get_fm(qrid, rid)
        if len(fm) == 0:
            continue
        desc1_m = qdesc_cache[qrid][fm.T[0]]
        desc2_m = rdesc_cache[rid][fm.T[1]]
        desc1_list.append(desc1_m)
        desc2_list.append(desc2_m)
    aggdesc1 = np.vstack(desc1_list)
    aggdesc2 = np.vstack(desc2_list)
    return aggdesc1, aggdesc2
