from __future__ import absolute_import, division, print_function
import utool
(print, print_,  rrr, profile,
 printDBG) = utool.inject(__name__, '[encounter]', DEBUG=False)
# Python
from itertools import izip
# Science
import networkx as netx
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
# HotSpotter
from ibeis.model.hots import match_chips3 as mc3

import utool


def compute_encounters(ibs, seconds_thresh=15):
    '''
    clusters encounters togethers (by time, not space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    '''
    if not 'seconds_thresh' in vars():
        seconds_thresh = 3

    # For each image
    gid_list = ibs.get_valid_gxs()

    # TODO: Get image GPS location
    #gps_info_list = ibs.gid2_exif(gid_list, tag='GPSInfo')
    #gps_lat_list = ibs.gid2_exif(gid_list, tag='GPSLatitude')
    #gps_lon_list = ibs.gid2_exif(gid_list, tag='GPSLongitude')
    #gps_latref_list = ibs.gid2_exif(gid_list, tag='GPSLatitudeRef')
    #gps_lonref_list = ibs.gid2_exif(gid_list, tag='GPSLongitudeRef')

    # Get image timestamps
    datetime_list = ibs.gid2_exif(gid_list, tag='DateTime')

    nImgs = len(datetime_list)
    valid_listx = [ix for ix, dt in enumerate(datetime_list) if dt is not None]
    nWithExif = len(valid_listx)
    nWithoutExif = nImgs - nWithExif
    print('[encounter] %d / %d images with exif data' % (nWithExif, nImgs))
    print('[encounter] %d / %d images without exif data' % (nWithoutExif, nImgs))

    # Convert datetime objects to unixtime scalars
    unixtime_list = [utool.exiftime_to_unixtime(datetime_str) for datetime_str in datetime_list]
    unixtime_list = np.array(unixtime_list)

    # Agglomerative clustering of unixtimes
    print('[encounter] clustering')
    X_data = np.vstack([unixtime_list, np.zeros(len(unixtime_list))]).T
    gid2_clusterid = fclusterdata(X_data, seconds_thresh, criterion='distance')

    # Reverse the image to cluster index mapping
    clusterx2_gxs = [[] for _ in xrange(gid2_clusterid.max())]
    for gid, clusterx in enumerate(gid2_clusterid):
        clusterx2_gxs[clusterx - 1].append(gid)  # IDS are 1 based

    # Print images per encouter statistics
    clusterx2_nGxs = np.array(map(len, clusterx2_gxs))
    print('[encounter] image per encounter stats:\n %s'
          % utool.pstats(clusterx2_nGxs, True))

    # Sort encounters by images per encounter
    ex2_clusterx = clusterx2_nGxs.argsort()
    gid2_ex  = [None] * len(gid2_clusterid)
    ex2_gxs = [None] * len(ex2_clusterx)
    for ex, clusterx in enumerate(ex2_clusterx):
        gids = clusterx2_gxs[clusterx]
        ex2_gxs[ex] = gids
        for gid in gids:
            gid2_ex[gid] = ex
    return gid2_ex, ex2_gxs


def build_encounter_ids(ex2_gxs, gid2_clusterid):
    USE_STRING_ID = True
    gid2_eid = [None] * len(gid2_clusterid)
    for ex, gids in enumerate(ex2_gxs):
        for gid in gids:
            nGx = len(gids)
            gid2_eid[gid] = ('ex=%r_nGxs=%d' % (ex, nGx)
                             if USE_STRING_ID else
                             ex + (nGx / 10 ** np.ceil(np.log(nGx) / np.log(10))))


def get_chip_encounters(ibs):
    gid2_ex, ex2_gxs = compute_encounters(ibs)
    # Build encounter to chips from encounter to images
    ex2_cxs = [None for _ in xrange(len(ex2_gxs))]
    for ex, gids in enumerate(ex2_gxs):
        ex2_cxs[ex] = utool.flatten(ibs.gid2_cxs(gids))
    # optional
    # resort encounters by number of chips
    ex2_nCxs = map(len, ex2_cxs)
    ex2_cxs = [y for (x, y) in sorted(zip(ex2_nCxs, ex2_cxs))]
    return ex2_cxs


def get_fmatch_iter(res):
    # USE res.get_fmatch_iter()
    fmfsfk_enum = enumerate(izip(res.rid2_fm, res.rid2_fs, res.rid2_fk))
    fmatch_iter = ((rid, fx_tup, score, rank)
                   for rid, (fm, fs, fk) in fmfsfk_enum
                   for (fx_tup, score, rank) in izip(fm, fs, fk))
    return fmatch_iter


def get_cxfx_enum(qreq):
    ax2_cxs = qreq._data_index.ax2_cx
    ax2_fxs = qreq._data_index.ax2_fx
    ridfx_enum = enumerate(izip(ax2_cxs, ax2_fxs))
    return ridfx_enum


def intra_query_cxs(ibs, rids):
    dcxs = qcxs = rids
    qreq = mc3.prep_query_request(qreq=ibs.qreq, qcxs=qcxs, dcxs=dcxs,
                                  query_cfg=ibs.prefs.query_cfg)
    qcx2_res = mc3.process_query_request(ibs, qreq)
    return qcx2_res


#def intra_encounter_match(ibs, rids, **kwargs):
    # Make a graph between the chips
    #qcx2_res = intra_query_cxs(rids)
    #graph = make_chip_graph(qcx2_res)
    # TODO: Make a super cool algorithm which does this correctly
    #graph.cutEdges(**kwargs)
    # Get a temporary name id
    # TODO: ensure these name indexes do not conflict with other encounters
    #rid2_nx, nid2_cxs = graph.getConnectedComponents()
    #return graph


def execute_all_intra_encounter_match(ibs, **kwargs):
    # Group images / chips into encounters
    ex2_cxs = get_chip_encounters(ibs)
    # For each encounter
    ex2_names = {}
    for ex, rids in enumerate(ex2_cxs):
        pass
        # Perform Intra-Encounter Matching
        #nid2_cxs = intra_encounter_match(ibs, rids)
        #ex2_names[ex] = nid2_cxs
    return ex2_names


def inter_encounter_match(ibs, eid2_names=None, **kwargs):
    # Perform Inter-Encounter Matching
    #if eid2_names is None:
        #eid2_names = intra_encounter_match(ibs, **kwargs)
    all_nxs = utool.flatten(eid2_names.values())
    for nid2_cxs in eid2_names:
        qnxs = nid2_cxs
        dnxs = all_nxs
        name_result = ibs.query(qnxs=qnxs, dnxs=dnxs)
    qcx2_res = name_result.chip_results()
    graph = netx.Graph()
    graph.add_nodes_from(range(len(qcx2_res)))
    graph.add_edges_from([res.rid2_fm for res in qcx2_res.itervalues()])
    graph.setWeights([(res.rid2_fs, res.rid2_fk) for res in qcx2_res.itervalues()])
    graph.cutEdges(**kwargs)
    rid2_nx, nid2_cxs = graph.getConnectedComponents()
    return rid2_nx


def print_encounter_stats(ex2_cxs):
    ex2_nCxs = map(len, ex2_cxs)
    ex_statstr = utool.printable_mystats(ex2_nCxs)
    print('num_encounters = %r' % len(ex2_nCxs))
    print('encounter_stats = %r' % (ex_statstr,))
