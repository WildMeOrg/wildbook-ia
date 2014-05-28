from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from ibeis import constants
(print, print_,  rrr, profile,
 printDBG) = utool.inject(__name__, '[preproc_encounter]', DEBUG=False)


def _cluster_encounters_by_time(valid_gids, valid_unixtimes, seconds_thresh):
    # fclusterdata requires 2d input
    X_data = np.vstack([valid_unixtimes, np.zeros(valid_unixtimes.size)]).T
    gid2_cluster = fclusterdata(X_data, seconds_thresh, criterion='distance')
    # Reverse the image to cluster index mapping
    cluster2_gids = utool.build_reverse_mapping(valid_gids, gid2_cluster)
    # Sort encounters by images per encounter
    cluster_list, gids_in_cluster = utool.unpack_items_sorted_by_lenvalue(cluster2_gids)
    cluster_list    = np.array(cluster_list)
    gids_in_cluster = np.array(gids_in_cluster)
    return cluster_list, gids_in_cluster


def _get_images_with_unixtimes(ibs):
    """ Returns valid gids with unixtimes """
    # Get image ids and timestamps
    gid_list = np.array(ibs.get_valid_gids())
    unixtime_list = ibs.get_image_unixtime(gid_list)
    # Remove images without timestamps
    isvalid_list = unixtime_list != -1
    valid_gids = gid_list[isvalid_list]
    valid_unixtimes = unixtime_list[isvalid_list]
    return valid_gids, valid_unixtimes


def _filter_encounters(cluster_list, gids_in_cluster, min_imgs_per_encounter):
    """ Removes cluster with too few images """
    nGids_per_cluster = np.array(map(len, gids_in_cluster))
    iscluster_valid = nGids_per_cluster >= min_imgs_per_encounter
    valid_cluster_list    = cluster_list[iscluster_valid]
    valid_gids_in_cluster = gids_in_cluster[iscluster_valid]
    # Rebase ids so encounter0 has the most images
    encounter_ids = range(len(valid_cluster_list))
    gids_in_eid = valid_gids_in_cluster
    return encounter_ids, gids_in_eid


def ibeis_compute_encounters(ibs):
    """
    clusters encounters togethers (by time, not yet space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    """
    print('[encounter] clustering encounters')
    # Config info
    seconds_thresh = ibs.cfg.enc_cfg.seconds_thresh
    min_imgs_per_encounter = ibs.cfg.enc_cfg.min_imgs_per_encounter
    # Data to cluster
    valid_gids, valid_unixtimes = _get_images_with_unixtimes(ibs)
    if len(valid_gids) == 0:
        print('WARNING: No unixtime data to compute encounters with')
        return [], []
    # Agglomerative clustering of unixtimes
    cluster_list, gids_in_cluster  = _cluster_encounters_by_time(valid_gids, valid_unixtimes, seconds_thresh)
    # Remove encounters less than the threshold
    encounter_ids, gids_in_eid = _filter_encounters(cluster_list, gids_in_cluster, min_imgs_per_encounter)
    # Flatten gids list by enounter
    flat_eids, flat_gids = utool.flatten_membership_mapping(encounter_ids, gids_in_eid)
    # Create enctext for each image
    enctext_list = [constants.ENCTEXT_PREFIX + repr(eid) for eid in flat_eids]
    print('[encounter] finished clustering')
    return enctext_list, flat_gids
