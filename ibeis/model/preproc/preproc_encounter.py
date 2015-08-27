# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from six.moves import zip, map  # NOQA
from scipy.spatial import distance
import scipy.cluster.hierarchy as hier
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import pdist
(print, rrr, profile) = utool.inject2(__name__, '[preproc_encounter]')


#@utool.indent_func('[encounter]')
def ibeis_compute_encounters(ibs, gid_list):
    """
    clusters encounters togethers (by time, not yet space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    """
    # Config info
    enc_cfgstr       = ibs.cfg.enc_cfg.get_cfgstr()
    seconds_thresh   = ibs.cfg.enc_cfg.seconds_thresh
    min_imgs_per_enc = ibs.cfg.enc_cfg.min_imgs_per_encounter
    cluster_algo     = ibs.cfg.enc_cfg.cluster_algo
    quantile         = ibs.cfg.enc_cfg.quantile
    print('[encounter] Computing %r encounters on %r images.' % (cluster_algo, len(gid_list)))
    print('[encounter] enc_cfgstr = %r' % enc_cfgstr)
    if len(gid_list) == 0:
        print('[encounter] WARNING: len(gid_list) == 0. No images to compute encounters with')
        return [], []
    elif len(gid_list) == 1:
        print('[encounter] WARNING: custering 1 image into its own encounter')
        gid_arr = np.array(gid_list)
        label_arr = np.zeros(gid_arr.shape)
    else:
        X_data, gid_arr = _prepare_X_data(ibs, gid_list, use_gps=False)
        # Agglomerative clustering of unixtimes
        if cluster_algo == 'agglomerative':
            label_arr = _agglomerative_cluster_encounters(X_data, seconds_thresh)
        elif cluster_algo == 'meanshift':
            label_arr = _meanshift_cluster_encounters(X_data, quantile)
        else:
            raise AssertionError('[encounter] Uknown clustering algorithm: %r' % cluster_algo)
    # Group images by unique label
    labels, label_gids = _group_images_by_label(label_arr, gid_arr)
    # Remove encounters less than the threshold
    enc_labels    = labels
    enc_gids      = label_gids
    enc_unixtimes = _compute_encounter_unixtime(ibs, enc_gids)
    enc_labels, enc_gids = _filter_and_relabel(labels, label_gids, min_imgs_per_enc, enc_unixtimes)
    # Flatten gids list by enounter
    flat_eids, flat_gids = utool.flatten_membership_mapping(enc_labels, enc_gids)
    # Create enctext for each image
    # Moved to ibeis.control.IBEISControl.compute_encounters so that it does not overwrite into existing encounters
    #enctext_list = ['Encounter ' + str(eid) for eid in flat_eids]
    # enctext_list = ['E' + str(eid) + enc_cfgstr for eid in flat_eids]
    #enctext_list = [dt + '_E' + str(num) + enc_cfgstr for num, dt in zip(enc_labels, enc_datetimes)]
    #enctext_list = ibsfuncs.make_enctext_list(flat_eids, enc_cfgstr)
    print('[encounter] Found %d clusters.' % len(labels))
    return flat_eids, flat_gids


def _compute_encounter_unixtime(ibs, enc_gids):
    #assert isinstance(ibs, IBEISController)
    from ibeis import ibsfuncs
    unixtimes = ibsfuncs.unflat_map(ibs.get_image_unixtime, enc_gids)
    time_arrs = list(map(np.array, unixtimes))
    enc_unixtimes = list(map(np.mean, time_arrs))
    return enc_unixtimes


def _compute_encounter_datetime(ibs, enc_gids):
    #assert isinstance(ibs, IBEISController)
    #from ibeis import ibsfuncs
    enc_unixtimes = _compute_encounter_unixtime(ibs, enc_gids)
    enc_datetimes = list(map(utool.unixtime_to_datetimestr, enc_unixtimes))
    return enc_datetimes


def _prepare_X_data(ibs, gid_list, use_gps=False):
    """
    FIXME: use ut.haversine formula on gps dimensions
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
    label_arr = hier.fclusterdata(X_data, seconds_thresh, criterion='distance')
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


def _filter_and_relabel(labels, label_gids, min_imgs_per_enc, enc_unixtimes=None):
    """
    Removes clusters with too few members.
    Relabels clusters-labels such that label 0 has the most members
    """
    label_nGids = np.array(list(map(len, label_gids)))
    label_isvalid = label_nGids >= min_imgs_per_enc
    if enc_unixtimes is None:
        # Rebase ids so encounter0 has the most images
        enc_ids  = range(label_isvalid.sum())
        enc_gids = label_gids[label_isvalid]
    else:
        # sort by time instead
        unixtime_arr = np.array(enc_unixtimes)
        # Reorder encounters so the oldest has the lowest number
        enc_gids = label_gids[unixtime_arr.argsort()]
        enc_ids = range(len(enc_gids))
    return enc_ids, enc_gids


def timespace_distance(pt1, pt2):
    (sec1, lat1, lon1) = pt1
    (sec2, lat2, lon2) = pt2
    km_dist = ut.haversine((lat1, lon1), (lat1, lon2))
    km_per_sec = .002  # conversion ratio for reasonable animal walking speed
    sec_dist = (((sec1 - sec2) * km_per_sec) ** 2)
    timespace_dist = km_dist + sec_dist
    return timespace_dist


def timespace_pdist(X_data):
    if X_data.shape[1] == 3:
        return pdist(X_data, timespace_distance)
    if X_data.shape[1] == 3:
        return pdist(X_data, 'euclidian')


def cluster_timespace(X_data, thresh):
    condenced_dist_mat = distance.pdist(X_data, timespace_distance)
    linkage_mat        = hier.linkage(condenced_dist_mat, method='centroid')
    X_labels           = hier.fcluster(linkage_mat, thresh, criterion='inconsistent',
                                        depth=2, R=None, monocrit=None)
    return X_labels


def testdata_gps():
    lon = np.array([4.54, 104.0, -14.9, 56.26, 103.46, 103.37, 54.22, 23.3,
                    25.53, 23.31, 118.0, 103.53, 54.40, 103.48, 6.14, 7.25,
                    2.38, 18.18, 103.54, 103.40, 28.59, 25.21, 29.35, 25.20, ])

    lat = np.array([52.22, 1.14, 27.34, 25.16, 1.16, 1.11, 24.30, 37.54, 37.26,
                    38.1, 24.25, 1.13, 24.49, 1.13, 42.33, 43.44, 39.34, 70.30,
                    1.16, 1.10, 40.58, 37.34, 41.18, 38.35, ])

    time = np.zeros(len(lon))

    X_data = np.vstack((time, lat, lon)).T

    X_name = np.array([0, 1, 2, 2, 2, 2, 3, 3, 3])  # NOQA
    X_data = np.array([
        (0, 42.727985, -73.683994),  # MRC
        (0, 42.657872, -73.764148),  # Home
        (0, 42.657414, -73.774448),  # Park1
        (0, 42.658333, -73.770993),  # Park2
        (0, 42.654384, -73.768919),  # Park3
        (0, 42.655039, -73.769048),  # Park4
        (0, 42.876974, -73.819311),  # CP1
        (0, 42.862946, -73.804977),  # CP2
        (0, 42.849809, -73.758486),  # CP3
    ])

    timespace_distance(X_data[1], X_data[0])

    from scipy.cluster.hierarchy import fcluster
    #import numpy as np
    #np.set_printoptions(precision=8, threshold=1000, linewidth=200)

    condenced_dist_mat = distance.pdist(X_data, timespace_distance)
    #linkage_methods = [
    #    'single',
    #    'complete',
    #    'average',
    #    'weighted',
    #    'centroid',
    #    'median',
    #    'ward',
    #]
    # Linkage matrixes are interpeted incrementally starting from the first row
    # They are unintuitive, but not that difficult to grasp
    linkage_mat = hier.linkage(condenced_dist_mat, method='single')
    #print(linkage_mat)
    #hier.leaves_list(linkage_mat)
    # FCluster forms flat clusters from the heirarchical linkage matrix
    #fcluster_criterions = [
    #    'inconsistent',  # use a threshold
    #    'distance',  # cophentic distance greter than t
    #    'maxclust',
    #    'monogrit',
    #    'maxclust_monocrit',
    #]
    # depth has no meaning outside inconsistent criterion
    #R = hier.inconsistent(linkage_mat) # calcualted automagically in fcluster
    thresh = .8
    depth = 2
    R = None  # calculated automatically for 'inconsistent' criterion
    monocrit = None
    X_labels = fcluster(linkage_mat, thresh, criterion='inconsistent',
                        depth=depth, R=R, monocrit=monocrit)

    # plot
    from plottool import draw_func2 as df2
    fig = df2.figure(fnum=1, doclf=True, docla=True)
    hier.dendrogram(linkage_mat, orientation='top')
    fig.show()

    print(X_labels)


def plot_annotaiton_gps(X_Data):
    """ Plots gps coordinates on a map projection """
    import plottool as pt
    from mpl_toolkits.basemap import Basemap
    #lat = X_data[1:5, 1]
    #lon = X_data[1:5, 2]
    lat = X_data[:, 1]  # NOQA
    lon = X_data[:, 2]  # NOQA
    fig = pt.figure(fnum=1, doclf=True, docla=True)  # NOQA
    pt.close_figure(fig)
    fig = pt.figure(fnum=1, doclf=True, docla=True)
    # setup Lambert Conformal basemap.
    m = Basemap(llcrnrlon=lon.min(),
                urcrnrlon=lon.max(),
                llcrnrlat=lat.min(),
                urcrnrlat=lat.max(),
                projection='cea',
                resolution='h')
    # draw coastlines.
    #m.drawcoastlines()
    #m.drawstates()
    # draw a boundary around the map, fill the background.
    # this background will end up being the ocean color, since
    # the continents will be drawn on top.
    #m.bluemarble()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='aqua')
    # Convert GPS to projected coordinates
    x1, y1 = m(lon, lat)  # convert to meters # lon==X, lat==Y
    m.plot(x1, y1, 'o')
    fig.show()


if __name__ == '__main__':
    """
    python -m ibeis.model.preproc.preproc_encounter
    python -m ibeis.model.preproc.preproc_encounter --allexamples
    """
    import utool as ut
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
