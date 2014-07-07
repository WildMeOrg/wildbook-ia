from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from ibeis.dev import ibsfuncs
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import pdist
#from ibeis import constants
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_encounter]', DEBUG=False)


@utool.indent_func('[encounter]')
def ibeis_compute_encounters(ibs, gid_list):
    """
    clusters encounters togethers (by time, not yet space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    """
    # Config info
    enc_cfgstr      = ibs.cfg.enc_cfg.get_cfgstr()
    seconds_thresh   = ibs.cfg.enc_cfg.seconds_thresh
    min_imgs_per_enc = ibs.cfg.enc_cfg.min_imgs_per_encounter
    cluster_algo     = ibs.cfg.enc_cfg.cluster_algo
    quantile         = ibs.cfg.enc_cfg.quantile
    print('Computing %r encounters on %r images.' % (cluster_algo, len(gid_list)))
    print('enc_cfgstr = %r' % enc_cfgstr)
    if len(gid_list) == 0:
        print('WARNING: No unixtime data to compute encounters with')
        return [], []
    X_data, gid_arr = _prepare_X_data(ibs, gid_list, use_gps=False)
    # Agglomerative clustering of unixtimes
    if cluster_algo == 'agglomerative':
        label_arr = _agglomerative_cluster_encounters(X_data, seconds_thresh)
    elif cluster_algo == 'meanshift':
        label_arr = _meanshift_cluster_encounters(X_data, quantile)
    else:
        raise AssertionError('Uknown clustering algorithm: %r' % cluster_algo)
    # Group images by unique label
    labels, label_gids = _group_images_by_label(label_arr, gid_arr)
    # Remove encounters less than the threshold
    enc_labels, enc_gids = _filter_and_relabel(labels, label_gids, min_imgs_per_enc)
    # Flatten gids list by enounter
    flat_eids, flat_gids = utool.flatten_membership_mapping(enc_labels, enc_gids)
    # Create enctext for each image
    #enctext_list = [constants.ENCTEXT_PREFIX + repr(eid) for eid in flat_eids]
    enctext_list = ibsfuncs.make_enctext_list(flat_eids, enc_cfgstr)
    print('Found %d clusters.' % len(labels))
    return enctext_list, flat_gids


def haversine(lon1, lat1, lon2, lat2):
    """
    #http://gis.stackexchange.com/questions/81551/matching-gps-tracks
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    http://en.wikipedia.org/wiki/Haversine_formula
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    EARTH_RADIUS = 6367
    km = EARTH_RADIUS * c
    return km


def timespace_distance(pt1, pt2):
    (sec1, lat1, lon1) = pt1
    (sec2, lat2, lon2) = pt2
    gps_dist = haversine(lon1, lat1, lon2, lat2)
    sec_dist = (sec1 - sec2) ** 2
    timespace_dist = gps_dist + sec_dist
    return timespace_dist


def timespace_pdist(X_data):
    if X_data.shape[1] == 3:
        return pdist(X_data, timespace_distance)
    if X_data.shape[1] == 3:
        return pdist(X_data, 'euclidian')


def _prepare_X_data(ibs, gid_list, use_gps=False):
    """
    FIXME: use haversine formula on gps dimensions
    fix weighting between seconds and gps
    """
    # Data to cluster
    unixtime_list = ibs.get_image_unixtime(gid_list)
    gid_arr       = np.array(gid_list)
    unixtime_arr  = np.array(unixtime_list)

    if use_gps:
        lat_list = ibs.get_image_lat(gid_list)
        lon_list = ibs.get_image_lon(gid_list)
        lat_arr = np.array(lat_list)
        lon_arr = np.array(lon_list)
        X_data = np.vstack([unixtime_arr, lat_arr, lon_arr]).T
    else:
        # scipy clustering requires 2d input
        X_data = np.vstack([unixtime_arr, np.zeros(unixtime_arr.size)]).T
    return X_data, gid_arr


def _agglomerative_cluster_encounters(X_data, seconds_thresh):
    """ Agglomerative encounter clustering algorithm
    Input:  Length N array of data to cluster
    Output: Length N array of cluster indexes
    """
    label_arr = fclusterdata(X_data, seconds_thresh, criterion='distance')
    return label_arr


def _meanshift_cluster_encounters(X_data, quantile):
    """ Meanshift encounter clustering algorithm
    Input: Length N array of data to cluster
    Output: Length N array of labels
    """
    # quantile should be between [0, 1]
    # e.g: quantile=.5 represents the median of all pairwise distances
    try:
        bandwidth = estimate_bandwidth(X_data, quantile=quantile, n_samples=500)
        if bandwidth == 0:
            raise AssertionError('[WARNING!] bandwidth is 0. Cannot cluster')
        # bandwidth is with respect to the RBF used in clustering
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms.fit(X_data)
        label_arr = ms.labels_
    except Exception as ex:
        utool.printex(ex, 'error computing meanshift',
                      key_list=['X_data', 'quantile'],
                      iswarning=True)
        # Fallback to all from same encounter
        label_arr = np.zeros(X_data.size)
    return label_arr


def _group_images_by_label(labels_arr, gid_arr):
    """
    Input: Length N list of labels and ids
    Output: Length M list of unique labels, and lenth M list of lists of ids
    """
    # Reverse the image to cluster index mapping
    label2_gids = utool.build_reverse_mapping(gid_arr, labels_arr)
    # Unpack dict, sort encounters by images-per-encounter
    labels, label_gids = utool.unpack_items_sorted_by_lenvalue(label2_gids)
    labels     = np.array(labels)
    label_gids = np.array(label_gids)
    return labels, label_gids


def _filter_and_relabel(labels, label_gids, min_imgs_per_enc):
    """
    Removes clusters with too few members.
    Relabels clusters-labels such that label 0 has the most members
    """
    label_nGids = np.array(map(len, label_gids))
    label_isvalid = label_nGids >= min_imgs_per_enc
    # Rebase ids so encounter0 has the most images
    enc_ids  = range(label_isvalid.sum())
    enc_gids = label_gids[label_isvalid]
    return enc_ids, enc_gids
