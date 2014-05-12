from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
(print, print_,  rrr, profile,
 printDBG) = utool.inject(__name__, '[preproc_encounter]', DEBUG=False)


def compute_encounters(ibs, seconds_thresh=15):
    '''
    clusters encounters togethers (by time, not space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    '''
    print('[encounter] clustering encounters')
    # Get image timestamps
    gid_list = np.array(ibs.get_valid_gids())
    unixtime_list = ibs.get_image_unixtime(gid_list)

    # Remove images without timestamps
    isvalid_list = unixtime_list != -1
    valid_gid_list = gid_list[isvalid_list]
    valid_unixtimes = unixtime_list[isvalid_list]

    if len(valid_gid_list) == 0:
        print('WARNING: No unixtime data to compute encounters with')
        return [], []

    # Agglomerative clustering of unixtimes
    # fclusterdata requires 2d input
    X_data = np.vstack([valid_unixtimes, np.zeros(valid_unixtimes.size)]).T
    gid2_cluster = fclusterdata(X_data, seconds_thresh, criterion='distance')

    # Reverse the image to cluster index mapping
    cluster2_gids = utool.build_reverse_mapping(valid_gid_list, gid2_cluster)

    # Sort encounters by images per encounter
    cluster_list, gids_in_cluster = utool.unpack_items_sorted_by_lenvalue(cluster2_gids)
    nGids_per_cluster = np.array(map(len, gids_in_cluster))
    cluster_list    = np.array(cluster_list)
    gids_in_cluster = np.array(gids_in_cluster)

    # Remove encounters with only one image
    iscluster_valid = nGids_per_cluster > 1
    valid_cluster_list    = cluster_list[iscluster_valid]
    valid_gids_in_cluster = gids_in_cluster[iscluster_valid]

    # Rebase ids so encounter0 has the most images
    encounter_ids = range(len(valid_cluster_list))
    gids_in_eid = valid_gids_in_cluster

    # Get flat indexes
    flat_eids, flat_gids = utool.flatten_membership_mapping(encounter_ids, gids_in_eid)
    return flat_eids, flat_gids
