# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import vtool as vt
from six.moves import zip, map, range  # NOQA
from scipy.spatial import distance
import scipy.cluster.hierarchy
import sklearn.cluster
#from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import pdist
(print, rrr, profile) = ut.inject2(__name__, '[preproc_occurrence]')


#@ut.indent_func('[occurrence]')
def ibeis_compute_occurrences(ibs, gid_list):
    """
    clusters occurrences togethers (by time, not yet space) An occurrence is a
    meeting, localized in time and space between a camera and a group of
    animals.  Animals are identified within each occurrence.

    Does not modify database state, just returns cluster ids

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m ibeis --tf ibeis_compute_occurrences:0 --show

        TODO: FIXME: good example of autogen doctest return failure

    #Ignore:
    #    >>> import ibeis
    #    >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
    #    >>> ibs = ibeis.opendb(defaultdb='lynx')
    #    >>> aid_list = ibs.get_valid_aids()
    #    >>> filter_kw = {}
    #    >>> filter_kw['been_adjusted'] = True
    #    >>> aid_list_ = ibs.filter_annots_general(aid_list, filter_kw)
    #    >>> gid_list = ibs.get_annot_gids(aid_list_)
    #    >>> flat_imgsetids, flat_gids = ibeis_compute_occurrences(ibs, gid_list)
    #    >>> aids_list = list(ut.group_items(aid_list_, flat_imgsetids).values())
    #    >>> metric = list(map(len, aids_list))
    #    >>> sortx = ut.list_argsort(metric)[::-1]
    #    >>> index = sortx[1]
    #    >>> #gids = occur_gids[index]
    #    >>> aids = aids_list[index]
    #    >>> gids = list(set(ibs.get_annot_gids(aids)))
    #    >>> print('len(aids) = %r' % (len(aids),))
    #    >>> ut.quit_if_noshow()
    #    >>> from ibeis.viz import viz_graph
    #    >>> import plottool as pt
    #    >>> #pt.imshow(bigimg)
    #    >>> #bigimg = vt.stack_image_recurse(img_list)
    #    >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids, with_all=False)
    #    >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> aid_list_ = ibs.filter_annots_general(aid_list, filter_kw)
        >>> (flat_imgsetids, flat_gids) = ibeis_compute_occurrences(ibs, gid_list)
        >>> aids_list = list(ut.group_items(aid_list_, flat_imgsetids).values())
        >>> metric = list(map(len, aids_list))
        >>> sortx = ut.list_argsort(metric)[::-1]
        >>> index = sortx[1]
        >>> #gids = occur_gids[index]
        >>> aids = aids_list[index]
        >>> gids = list(set(ibs.get_annot_gids(aids)))
        >>> print('len(aids) = %r' % (len(aids),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> from ibeis.viz import viz_graph
        >>> import plottool as pt
        >>> #pt.imshow(bigimg)
        >>> #bigimg = vt.stack_image_recurse(img_list)
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids, with_all=False, prog='neato')
        >>> ut.show_if_requested()
    """
    occur_cfgstr = ibs.cfg.occur_cfg.get_cfgstr()
    print('[occur] occur_cfgstr = %r' % occur_cfgstr)
    cluster_algo  = ibs.cfg.occur_cfg.cluster_algo
    cfgdict = dict(
        min_imgs_per_occurence=ibs.cfg.occur_cfg.min_imgs_per_occurrence,
        seconds_thresh=ibs.cfg.occur_cfg.seconds_thresh,
        quantile=ibs.cfg.occur_cfg.quantile,
    )
    # TODO: use gps
    occur_labels, occur_gids = compute_occurrence_groups(ibs, gid_list, cluster_algo, cfgdict=cfgdict)
    if True:
        gid2_label = {gid: label for label, gids in zip(occur_labels, occur_gids) for gid in gids}
        # Assert that each gid only belongs to one occurrence
        flat_imgsetids = ut.dict_take(gid2_label, gid_list)
        flat_gids = gid_list
    else:
        # Flatten gids list by enounter
        flat_imgsetids, flat_gids = ut.flatten_membership_mapping(occur_labels, occur_gids)
    return flat_imgsetids, flat_gids


def compute_occurrence_groups(ibs, gid_list, cluster_algo, cfgdict={}, use_gps=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m ibeis --tf compute_occurrence_groups
        python -m ibeis --tf compute_occurrence_groups --show --zoom=.3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
        >>> import ibeis
        >>> import vtool as vt
        >>> #ibs = ibeis.opendb(defaultdb='testdb1')
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> gid_list = ibs.get_valid_gids(require_unixtime=True, require_gps=True)
        >>> use_gps = True
        >>> cluster_algo = 'meanshift'
        >>> cfgdict = dict(quantile=.005, min_imgs_per_occurence=2)
        >>> (occur_labels, occur_gids) = compute_occurrence_groups(ibs, gid_list, cluster_algo, cfgdict, use_gps=use_gps)
        >>> aidsgroups_list = ibs.unflat_map(ibs.get_image_aids, occur_gids)
        >>> aids_list = list(map(ut.flatten, aidsgroups_list))
        >>> nids_list = list(map(np.array, ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)))
        >>> metric = [len(np.unique(nids[nids > -1])) for nids in nids_list]
        >>> metric = [vt.safe_max(np.array(ut.dict_hist(nids).values())) for nids in nids_list]
        >>> #metric = list(map(len, aids_list))
        >>> sortx = ut.list_argsort(metric)[::-1]
        >>> index = sortx[20]
        >>> #gids = occur_gids[index]
        >>> aids = aids_list[index]
        >>> aids = ibs.filter_annots_general(aids, min_qual='ok', is_known=True)
        >>> gids = list(set(ibs.get_annot_gids(aids)))
        >>> print('len(aids) = %r' % (len(aids),))
        >>> img_list = ibs.get_images(gids)
        >>> ut.quit_if_noshow()
        >>> from ibeis.viz import viz_graph
        >>> import plottool as pt
        >>> #pt.imshow(bigimg)
        >>> #aids = ibs.group_annots_by_name(aids)[0][0]
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids, with_all=False, prog='neato')
        >>> ut.show_if_requested()

        ibs.unflat_map(ibs.get_annot_case_tags, aids_list)
        ibs.filter_aidpairs_by_tags(has_any='photobomb')

        photobomb_aids = ibs.filter_aidpairs_by_tags(has_any='photobomb')
        aids = photobomb_aids[0:10].flatten()
        _gt_aids = ibs.get_annot_groundtruth(aids)
        gt_aids = ut.get_list_column_slice(_gt_aids, slice(0, 3))
        aid_set = np.unique(np.append(aids.flatten(), ut.flatten(gt_aids)))
        aid_set = ibs.filter_annots_general(aid_set, minqual='ok')

        # This is the set of annotations used for testing intraoccurrence photobombs
        #print(ut.repr3(ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=aid_set), strvals=True, nl=1))
        print(ut.repr3(ibs.get_annot_stats_dict(aid_set, forceall=True), strvals=True, nl=1))

    """
    # Config info
    gid_list = np.unique(gid_list)

    print('[occur] Computing %r occurrences on %r images.' % (
        cluster_algo, len(gid_list)))
    if len(gid_list) == 0:
        print('[occur] WARNING: len(gid_list) == 0. '
              'No images to compute occurrences with')
        occur_labels, occur_gids = [], []
    else:
        if len(gid_list) == 1:
            print('[occur] WARNING: custering 1 image into its own occurrence')
            gid_arr = np.array(gid_list)
            label_arr = np.zeros(gid_arr.shape)
        else:
            X_data, gid_arr = prepare_X_data(ibs, gid_list, use_gps=use_gps)
            # Agglomerative clustering of unixtimes
            if cluster_algo == 'agglomerative':
                seconds_thresh = cfgdict.get('seconds_thresh', 60.0)
                label_arr = agglomerative_cluster_occurrences(X_data,
                                                              seconds_thresh)
            elif cluster_algo == 'meanshift':
                quantile = cfgdict.get('quantile', 0.01)
                label_arr = meanshift_cluster_occurrences(X_data, quantile)
            else:
                raise AssertionError(
                    '[occurrence] Uknown clustering algorithm: %r' % cluster_algo)
        # Group images by unique label
        labels, label_gids = group_images_by_label(label_arr, gid_arr)
        # Remove occurrences less than the threshold
        occur_labels    = labels
        occur_gids      = label_gids
        occur_unixtimes = compute_occurrence_unixtime(ibs, occur_gids)
        min_imgs_per_occurence = cfgdict.get('min_imgs_per_occurence', 1)
        occur_labels, occur_gids = filter_and_relabel(
            labels, label_gids, min_imgs_per_occurence, occur_unixtimes)
        print('[occur] Found %d clusters.' % len(occur_labels))
        if len(label_gids) > 0:
            print('Cluster size stats:')
            ut.print_dict(
                ut.get_stats(list(map(len, occur_gids)), use_median=True,
                             use_sum=True),
                'occur stats')
    return occur_labels, occur_gids


def compute_occurrence_unixtime(ibs, occur_gids):
    #assert isinstance(ibs, IBEISController)
    # TODO: account for -1
    from ibeis import ibsfuncs
    unixtimes = ibsfuncs.unflat_map(ibs.get_image_unixtime, occur_gids)
    time_arrs = list(map(np.array, unixtimes))
    occur_unixtimes = list(map(np.mean, time_arrs))
    return occur_unixtimes


def _compute_occurrence_datetime(ibs, occur_gids):
    #assert isinstance(ibs, IBEISController)
    #from ibeis import ibsfuncs
    occur_unixtimes = compute_occurrence_unixtime(ibs, occur_gids)
    occur_datetimes = list(map(ut.unixtime_to_datetimestr, occur_unixtimes))
    return occur_datetimes


def prepare_X_data(ibs, gid_list, use_gps=False):
    """
    FIXME: use vt.haversine formula on gps dimensions
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


def agglomerative_cluster_occurrences(X_data, seconds_thresh):
    """
    Agglomerative occurrence clustering algorithm

    Args:
        X_data (ndarray):  Length N array of data to cluster
        seconds_thresh (float):

    Returns:
        ndarray: (label_arr) - Length N array of cluster indexes

    CommandLine:
        python -m ibeis.algo.preproc.preproc_occurrence --exec-agglomerative_cluster_occurrences

    References:
        https://docs.scipy.org/doc/scipy-0.9.0/reference/generated/scipy.cluster.hierarchy.fclusterdata.html#scipy.cluster.hierarchy.fclusterdata
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.fcluster.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data = '?'
        >>> seconds_thresh = '?'
        >>> (occur_ids, occur_gids) = agglomerative_cluster_occurrences(X_data, seconds_thresh)
        >>> result = ('(occur_ids, occur_gids) = %s' % (str((occur_ids, occur_gids)),))
        >>> print(result)
    """
    label_arr = scipy.cluster.hierarchy.fclusterdata(
        X_data, seconds_thresh, criterion='distance')
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
        python -m ibeis.algo.preproc.preproc_occurrence --exec-meanshift_cluster_occurrences

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data = '?'
        >>> quantile = '?'
        >>> result = meanshift_cluster_occurrences(X_data, quantile)
        >>> print(result)
    """
    try:
        bandwidth = sklearn.cluster.estimate_bandwidth(X_data, quantile=quantile, n_samples=500)
        assert bandwidth != 0, ('[occur] bandwidth is 0. Cannot cluster')
        # bandwidth is with respect to the RBF used in clustering
        #ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
        ms.fit(X_data)
        label_arr = ms.labels_

        unique_labels = np.unique(label_arr)
        max_label = max(0, unique_labels.max())
        num_orphans = (label_arr == -1).sum()
        label_arr[label_arr == -1] = np.arange(max_label + 1, max_label + 1 + num_orphans)
    except Exception as ex:
        ut.printex(ex, 'error computing meanshift',
                      key_list=['X_data', 'quantile'],
                      iswarning=True)
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
    labels  = labels_.take(sortx, axis=0)
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
        #occur_ids  = list(range(label_isvalid.sum()))
        #else:
        # sort by time instead
        unixtime_arr = np.array(occur_unixtimes)
        # Reorder occurrences so the oldest has the lowest number
        occur_gids = ut.take(label_gids, unixtime_arr.argsort())
    occur_ids = list(range(len(occur_gids)))
    return occur_ids, occur_gids


def timespace_distance(pt1, pt2):
    (sec1, lat1, lon1) = pt1
    (sec2, lat2, lon2) = pt2
    km_dist = vt.haversine((lat1, lon1), (lat1, lon2))
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
    """
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    condenced_dist_mat = distance.pdist(X_data, timespace_distance)
    linkage_mat        = scipy.cluster.hierarchy.linkage(
        condenced_dist_mat, method='centroid')
    X_labels           = scipy.cluster.hierarchy.fcluster(
        linkage_mat, thresh, criterion='inconsistent',
        depth=2, R=None, monocrit=None)
    return X_labels


def testdata_gps():
    r"""
    CommandLine:
        python -m ibeis.algo.preproc.preproc_occurrence --exec-testdata_gps --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_occurrence import *  # NOQA
        >>> X_data, linkage_mat = testdata_gps()
        >>> ut.quit_if_noshow()
        >>> # plot
        >>> import plottool as pt
        >>> fnum = pt.ensure_fnum(None)
        >>> fig = pt.figure(fnum=fnum, doclf=True, docla=True)
        >>> hier.dendrogram(linkage_mat, orientation='top')
        >>> #fig.show()
        >>> plot_annotaiton_gps(X_data)
        >>> ut.show_if_requested()

    """
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
    linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat, method='single')
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

    print(X_labels)
    return X_data, linkage_mat


def plot_annotaiton_gps(X_data):
    """ Plots gps coordinates on a map projection

    InstallBasemap:
        sudo apt-get install libgeos-dev
        pip install git+https://github.com/matplotlib/basemap

    Ignore:
        pip install git+git://github.com/myuser/foo.git@v123

    """
    import plottool as pt
    from mpl_toolkits.basemap import Basemap
    #lat = X_data[1:5, 1]
    #lon = X_data[1:5, 2]
    lat = X_data[:, 1]  # NOQA
    lon = X_data[:, 2]  # NOQA
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True, docla=True)  # NOQA
    pt.close_figure(fig)
    fig = pt.figure(fnum=fnum, doclf=True, docla=True)
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
    python -m ibeis.algo.preproc.preproc_occurrence
    python -m ibeis.algo.preproc.preproc_occurrence --allexamples
    """
    import utool as ut
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
