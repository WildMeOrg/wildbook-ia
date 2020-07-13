# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import vtool as vt
from six.moves import zip, map, range
from scipy.spatial import distance
import scipy.cluster.hierarchy
import sklearn.cluster

(print, rrr, profile) = ut.inject2(__name__, '[preproc_occurrence]')


def wbia_compute_occurrences(ibs, gid_list, config=None, verbose=None):
    """
    clusters occurrences togethers (by time, not yet space)
    An occurrence is a meeting, localized in time and space between a camera
    and a group of animals.
    Animals are identified within each occurrence.

    Does not modify database state, just returns cluster ids

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m wbia --tf wbia_compute_occurrences:0 --show
        TODO: FIXME: good example of autogen doctest return failure
    """
    if config is None:
        config = {'use_gps': False, 'seconds_thresh': 600}
        # from wbia.algo import Config
        # config = Config.OccurrenceConfig().asdict()
    occur_labels, occur_gids = compute_occurrence_groups(
        ibs, gid_list, config, verbose=verbose
    )
    if True:
        gid2_label = {
            gid: label for label, gids in zip(occur_labels, occur_gids) for gid in gids
        }
        # Assert that each gid only belongs to one occurrence
        flat_imgsetids = ut.dict_take(gid2_label, gid_list)
        flat_gids = gid_list
    else:
        # Flatten gids list by enounter
        flat_imgsetids, flat_gids = ut.flatten_membership_mapping(
            occur_labels, occur_gids
        )
    return flat_imgsetids, flat_gids


def compute_occurrence_groups(ibs, gid_list, config={}, use_gps=False, verbose=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m wbia compute_occurrence_groups

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> verbose = True
        >>> images = ibs.images()
        >>> gid_list = images.gids
        >>> config = {}  # wbia.algo.Config.OccurrenceConfig().asdict()
        >>> tup = wbia_compute_occurrences(ibs, gid_list)
        >>> (flat_imgsetids, flat_gids)
        >>> aids_list = list(ut.group_items(aid_list_, flat_imgsetids).values())
        >>> metric = list(map(len, aids_list))
        >>> sortx = ut.list_argsort(metric)[::-1]
        >>> index = sortx[1]
        >>> aids = aids_list[index]
        >>> gids = list(set(ibs.get_annot_gids(aids)))
    """
    if verbose is None:
        verbose = ut.NOT_QUIET
    # Config info
    gid_list = np.unique(gid_list)
    if verbose:
        print('[occur] Computing occurrences on %r images.' % (len(gid_list)))
        print('[occur] config = ' + ut.repr3(config))

    use_gps = config['use_gps']
    datas = prepare_X_data(ibs, gid_list, use_gps=use_gps)

    from wbia.algo.preproc import occurrence_blackbox

    cluster_algo = config.get('cluster_algo', 'agglomerative')
    km_per_sec = config.get('km_per_sec', occurrence_blackbox.KM_PER_SEC)
    thresh_sec = config.get('seconds_thresh', 30 * 60.0)
    min_imgs_per_occurence = config.get('min_imgs_per_occurence', 1)
    # 30 minutes = 3.6 kilometers
    # 5 minutes = 0.6 kilometers

    assert cluster_algo == 'agglomerative', 'only agglomerative is supported'

    # Group datas with different values separately
    all_gids = []
    all_labels = []
    for key in datas.keys():
        val = datas[key]
        gids, latlons, posixtimes = val
        labels = occurrence_blackbox.cluster_timespace_sec(
            latlons, posixtimes, thresh_sec, km_per_sec=km_per_sec
        )
        if labels is None:
            labels = np.zeros(len(gids), dtype=np.int)
        all_gids.append(gids)
        all_labels.append(labels)

    # Combine labels across different groups
    pads = [vt.safe_max(ys, fill=0) + 1 for ys in all_labels]
    offsets = np.array([0] + pads[:-1]).cumsum()
    all_labels_ = [ys + offset for ys, offset in zip(all_labels, offsets)]
    label_arr = np.array(ut.flatten(all_labels_))
    gid_arr = np.array(ut.flatten(all_gids))

    # Group images by unique label
    labels, label_gids = group_images_by_label(label_arr, gid_arr)
    # Remove occurrences less than the threshold
    occur_labels = labels
    occur_gids = label_gids
    occur_unixtimes = compute_occurrence_unixtime(ibs, occur_gids)
    occur_labels, occur_gids = filter_and_relabel(
        labels, label_gids, min_imgs_per_occurence, occur_unixtimes
    )
    if verbose:
        print('[occur] Found %d clusters.' % len(occur_labels))
    if len(label_gids) > 0 and verbose:
        print('[occur] Cluster image size stats:')
        ut.print_dict(
            ut.get_stats(list(map(len, occur_gids)), use_median=True, use_sum=True),
            'occur image stats',
        )
    return occur_labels, occur_gids


def compute_occurrence_unixtime(ibs, occur_gids):
    # assert isinstance(ibs, IBEISController)
    # TODO: account for -1
    from wbia.other import ibsfuncs

    unixtimes = ibsfuncs.unflat_map(ibs.get_image_unixtime, occur_gids)
    time_arrs = list(map(np.array, unixtimes))
    occur_unixtimes = list(map(np.mean, time_arrs))
    return occur_unixtimes


def _compute_occurrence_datetime(ibs, occur_gids):
    # assert isinstance(ibs, IBEISController)
    # from wbia.other import ibsfuncs
    occur_unixtimes = compute_occurrence_unixtime(ibs, occur_gids)
    occur_datetimes = list(map(ut.unixtime_to_datetimestr, occur_unixtimes))
    return occur_datetimes


def prepare_X_data(ibs, gid_list, use_gps=True):
    """
    Splits data into groups with/without gps and time

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> images = ibs.images()
        >>> # wbia.control.accessor_decors.DEBUG_GETTERS = True
        >>> use_gps = True
        >>> gid_list = images.gids
        >>> datas = prepare_X_data(ibs, gid_list, use_gps)
        >>> print(ut.repr2(datas, nl=2, precision=2))
        >>> assert len(datas['both'][0]) == 12
        >>> assert len(datas['neither'][0]) == 0
    """
    images = ibs.images(gid_list, caching=True)
    gps_list_ = images.gps2
    unixtime_list_ = images.unixtime2
    gps_list_ = vt.ensure_shape(gps_list_, (None, 2))
    has_gps = np.all(np.logical_not(np.isnan(gps_list_)), axis=1)
    has_time = np.logical_not(np.isnan(unixtime_list_))

    if not use_gps:
        has_gps[:] = False

    has_both = np.logical_and(has_time, has_gps)
    has_either = np.logical_or(has_time, has_gps)
    has_gps_only = np.logical_and(has_gps, np.logical_not(has_both))
    has_time_only = np.logical_and(has_time, np.logical_not(has_both))
    has_neither = np.logical_not(has_either)

    both = images.compress(has_both)
    xgps = images.compress(has_gps_only)
    xtime = images.compress(has_time_only)
    neither = images.compress(has_neither)

    # Group imagse with different attributes separately
    datas = {
        'both': (both.gids, both.unixtime2, both.gps2),
        'gps_only': (xgps.gids, None, xgps.gps2),
        'time_only': (xtime.gids, xtime.unixtime2, None),
        'neither': (neither.gids, None, None),
    }
    return datas


def agglomerative_cluster_occurrences(X_data, thresh_sec):
    """
    Agglomerative occurrence clustering algorithm

    Args:
        X_data (ndarray):  Length N array of data to cluster
        thresh_sec (float):

    Returns:
        ndarray: (label_arr) - Length N array of cluster indexes

    CommandLine:
        python -m wbia.algo.preproc.preproc_occurrence --exec-agglomerative_cluster_occurrences

    References:
        https://docs.scipy.org/doc/scipy-0.9.0/reference/generated/scipy.cluster.hierarchy.fclusterdata.html#scipy.cluster.hierarchy.fclusterdata
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.fcluster.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data = '?'
        >>> thresh_sec = '?'
        >>> (occur_ids, occur_gids) = agglomerative_cluster_occurrences(X_data, thresh_sec)
        >>> result = ('(occur_ids, occur_gids) = %s' % (str((occur_ids, occur_gids)),))
        >>> print(result)
    """
    label_arr = scipy.cluster.hierarchy.fclusterdata(
        X_data, thresh_sec, criterion='distance'
    )
    return label_arr


def meanshift_cluster_occurrences(X_data, quantile):
    """ Meanshift occurrence clustering algorithm

    Args:
        X_data (ndarray):  Length N array of data to cluster
        quantile (float): quantile should be between [0, 1].
            eg: quantile=.5 represents the median of all pairwise distances

    Returns:
        ndarray : Length N array of labels

    CommandLine:
        python -m wbia.algo.preproc.preproc_occurrence --exec-meanshift_cluster_occurrences

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data = '?'
        >>> quantile = '?'
        >>> result = meanshift_cluster_occurrences(X_data, quantile)
        >>> print(result)
    """
    try:
        bandwidth = sklearn.cluster.estimate_bandwidth(
            X_data, quantile=quantile, n_samples=500
        )
        assert bandwidth != 0, '[occur] bandwidth is 0. Cannot cluster'
        # bandwidth is with respect to the RBF used in clustering
        # ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms = sklearn.cluster.MeanShift(
            bandwidth=bandwidth, bin_seeding=True, cluster_all=False
        )
        ms.fit(X_data)
        label_arr = ms.labels_

        unique_labels = np.unique(label_arr)
        max_label = max(0, unique_labels.max())
        num_orphans = (label_arr == -1).sum()
        label_arr[label_arr == -1] = np.arange(max_label + 1, max_label + 1 + num_orphans)
    except Exception as ex:
        ut.printex(
            ex,
            'error computing meanshift',
            key_list=['X_data', 'quantile'],
            iswarning=True,
        )
        # Fallback to all from same occurrence
        label_arr = np.zeros(X_data.size)
    return label_arr


def group_images_by_label(label_arr, gid_arr):
    """
    Input: Length N list of labels and ids
    Output: Length M list of unique labels, and lenth M list of lists of ids
    """
    # Reverse the image to cluster index mapping
    import vtool as vt

    labels_, groupxs_ = vt.group_indices(label_arr)
    sortx = np.array(list(map(len, groupxs_))).argsort()[::-1]
    labels = labels_.take(sortx, axis=0)
    groupxs = ut.take(groupxs_, sortx)
    label_gids = vt.apply_grouping(gid_arr, groupxs)
    return labels, label_gids


def filter_and_relabel(labels, label_gids, min_imgs_per_occurence, occur_unixtimes=None):
    """
    Removes clusters with too few members.
    Relabels clusters-labels such that label 0 has the most members
    """
    label_nGids = np.array(list(map(len, label_gids)))
    label_isvalid = label_nGids >= min_imgs_per_occurence
    occur_gids = ut.compress(label_gids, label_isvalid)
    if occur_unixtimes is not None:
        occur_unixtimes = ut.compress(occur_unixtimes, label_isvalid)
        # Rebase ids so occurrence0 has the most images
        # occur_ids  = list(range(label_isvalid.sum()))
        # else:
        # sort by time instead
        unixtime_arr = np.array(occur_unixtimes)
        # Reorder occurrences so the oldest has the lowest number
        occur_gids = ut.take(label_gids, unixtime_arr.argsort())
    occur_ids = list(range(len(occur_gids)))
    return occur_ids, occur_gids


def timespace_distance(pt1, pt2):
    (sec1, lat1, lon1) = pt1
    (sec2, lat2, lon2) = pt2
    km_dist = vt.haversine((lat1, lon1), (lat2, lon2))
    km_per_sec = 0.002  # conversion ratio for reasonable animal walking speed
    # sec_dist = (((sec1 - sec2) * km_per_sec) ** 2)
    sec_dist = np.abs(sec1 - sec2) * km_per_sec
    timespace_dist = km_dist + sec_dist
    return timespace_dist


def timespace_pdist(X_data):
    if X_data.shape[1] == 3:
        return distance.pdist(X_data, timespace_distance)
    if X_data.shape[1] == 1:
        return distance.pdist(X_data, 'euclidian')


def cluster_timespace(X_data, thresh):
    """
    References:
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
            scipy.cluster.hierarchy.linkage.html

    CommandLine:
        python -m wbia.algo.preproc.preproc_occurrence cluster_timespace --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data = testdata_gps()
        >>> thresh = 10
        >>> X_labels = cluster_timespace(X_data, thresh)
        >>> fnum = pt.ensure_fnum(None)
        >>> fig = pt.figure(fnum=fnum, doclf=True, docla=True)
        >>> hier.dendrogram(linkage_mat, orientation='top')
        >>> plot_annotaiton_gps(X_data)
        >>> ut.show_if_requested()

    """
    condenced_dist_mat = distance.pdist(X_data, timespace_distance)
    # Compute heirarchical linkages
    linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat, method='centroid')
    # Cluster linkages
    X_labels = scipy.cluster.hierarchy.fcluster(
        linkage_mat, thresh, criterion='inconsistent', depth=2, R=None, monocrit=None
    )
    return X_labels


def testdata_gps():
    r"""
    Simple data to test GPS algorithm.

    Returns:
        X_name (ndarray): Nx1 matrix denoting groundtruth locations
        X_data (ndarray): Nx3 matrix where each columns are (time, lat, lon)
    """
    lon = np.array(
        [
            4.54,
            104.0,
            -14.9,
            56.26,
            103.46,
            103.37,
            54.22,
            23.3,
            25.53,
            23.31,
            118.0,
            103.53,
            54.40,
            103.48,
            6.14,
            7.25,
            2.38,
            18.18,
            103.54,
            103.40,
            28.59,
            25.21,
            29.35,
            25.20,
        ]
    )

    lat = np.array(
        [
            52.22,
            1.14,
            27.34,
            25.16,
            1.16,
            1.11,
            24.30,
            37.54,
            37.26,
            38.1,
            24.25,
            1.13,
            24.49,
            1.13,
            42.33,
            43.44,
            39.34,
            70.30,
            1.16,
            1.10,
            40.58,
            37.34,
            41.18,
            38.35,
        ]
    )

    time = np.zeros(len(lon))

    X_data = np.vstack((time, lat, lon)).T

    X_name = np.array([0, 1, 2, 2, 2, 2, 3, 3, 3])  # NOQA
    X_data = np.array(
        [
            (0, 42.727985, -73.683994),  # MRC
            (0, 42.657872, -73.764148),  # Home
            (0, 42.657414, -73.774448),  # Park1
            (0, 42.658333, -73.770993),  # Park2
            (0, 42.654384, -73.768919),  # Park3
            (0, 42.655039, -73.769048),  # Park4
            (0, 42.876974, -73.819311),  # CP1
            (0, 42.862946, -73.804977),  # CP2
            (0, 42.849809, -73.758486),  # CP3
        ]
    )
    return X_name, X_data


def plot_gps_html(gps_list):
    """ Plots gps coordinates on a map projection

    InstallBasemap:
        sudo apt-get install libgeos-dev
        pip install git+https://github.com/matplotlib/basemap
        http://matplotlib.org/basemap/users/examples.html

        pip install gmplot

        sudo apt-get install netcdf-bin
        sudo apt-get install libnetcdf-dev
        pip install netCDF4

    Ignore:
        pip install git+git://github.com/myuser/foo.git@v123

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.preproc.preproc_occurrence import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> images = ibs.images()
        >>> # Setup GPS points to draw
        >>> print('Setup GPS points')
        >>> gps_list_ = np.array(images.gps2)
        >>> unixtime_list_ = np.array(images.unixtime2)
        >>> has_gps = np.all(np.logical_not(np.isnan(gps_list_)), axis=1)
        >>> has_unixtime = np.logical_not(np.isnan(unixtime_list_))
        >>> isvalid = np.logical_and(has_gps, has_unixtime)
        >>> gps_list = gps_list_.compress(isvalid, axis=0)
        >>> unixtime_list = unixtime_list_.compress(isvalid)  # NOQA
        >>> plot_image_gps(gps_list)
    """
    import wbia.plottool as pt
    import gmplot
    import matplotlib as mpl
    import vtool as vt

    pt.qt4ensure()

    lat = gps_list.T[0]
    lon = gps_list.T[1]

    # Get extent of
    bbox = vt.bbox_from_verts(gps_list)
    centerx, centery = vt.bbox_center(bbox)

    gmap = gmplot.GoogleMapPlotter(centerx, centery, 13)
    color = mpl.colors.rgb2hex(pt.ORANGE)
    gmap.scatter(lat, lon, color=color, size=100, marker=False)
    gmap.draw('mymap.html')
    ut.startfile('mymap.html')

    # # Scale
    # bbox = vt.scale_bbox(bbox, 10.0)
    # extent = vt.extent_from_bbox(bbox)
    # basemap_extent = dict(llcrnrlon=extent[2], urcrnrlon=extent[3],
    #                      llcrnrlat=extent[0], urcrnrlat=extent[1])
    # # Whole globe
    # #basemap_extent = dict(llcrnrlon=0, llcrnrlat=-80,
    # #                      urcrnrlon=360, urcrnrlat=80)

    # from mpl_toolkits.basemap import Basemap
    # from matplotlib.colors import LightSource  # NOQA
    # from mpl_toolkits.basemap import shiftgrid, cm  # NOQA
    # from netCDF4 import Dataset
    # # Read information to make background pretty
    # print('Grab topo information')
    # etopodata = Dataset('http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc')
    # print('Read topo information')
    # topoin = etopodata.variables['ROSE'][:]
    # lons = etopodata.variables['ETOPO05_X'][:]
    # lats = etopodata.variables['ETOPO05_Y'][:]
    # # shift data so lons go from -180 to 180 instead of 20 to 380.
    # print('Shift data')
    # topoin, lons = shiftgrid(180., topoin, lons, start=False)

    # print('Make figure')
    # fnum = pt.ensure_fnum(None)
    # fig = pt.figure(fnum=fnum, doclf=True, docla=True)  # NOQA
    # print('Draw projection')
    # m = Basemap(projection='mill', **basemap_extent)
    # # setup Lambert Conformal basemap.
    # #m = Basemap(projection='cea',resolution='h', **basemap_extent)

    # # transform to nx x ny regularly spaced 5km native projection grid
    # print('projection grid')
    # nx = int((m.xmax - m.xmin) / 5000.) + 1
    # ny = int((m.ymax - m.ymin) / 5000.) + 1
    # topodat = m.transform_scalar(topoin, lons, lats, nx, ny)

    # # plot image over map with imshow.
    # im = m.imshow(topodat, cm.GMT_haxby)  # NOQA
    # # draw coastlines and political boundaries.
    # m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()

    # transform to nx x ny regularly spaced 5km native projection grid
    # ls = LightSource(azdeg=90, altdeg=20)
    # rgb = ls.shade(topodat, cm.GMT_haxby)
    # im = m.imshow(rgb)
    # draw coastlines and political boundaries.

    # m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()

    # draw a boundary around the map, fill the background.
    # this background will end up being the ocean color, since
    # the continents will be drawn on top.
    # m.bluemarble()
    # m.drawmapboundary(fill_color='aqua')
    # m.fillcontinents(color='coral', lake_color='aqua')
    # Convert GPS to projected coordinates
    # x1, y1 = m(lon, lat)  # convert to meters # lon==X, lat==Y
    # m.plot(x1, y1, '*', markersize=10)
    # fig.zoom_fac = pt.zoom_factory()
    # fig.pan_fac = pt.pan_factory()
    # fig.show()


if __name__ == '__main__':
    """
    python -m wbia.algo.preproc.preproc_occurrence
    python -m wbia.algo.preproc.preproc_occurrence --allexamples
    """
    import utool as ut
    import multiprocessing

    multiprocessing.freeze_support()
    ut.doctest_funcs()
