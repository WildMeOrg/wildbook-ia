# -*- coding: utf-8 -*-
"""
animal_walking_speeds

ZEBRA_SPEED_MAX  = 64  # km/h
ZEBRA_SPEED_RUN  = 50  # km/h
ZEBRA_SPEED_SLOW_RUN  = 20  # km/h
ZEBRA_SPEED_FAST_WALK = 10  # km/h
ZEBRA_SPEED_WALK = 7  # km/h


km_per_sec = .02
km_per_sec = .002
mph = km_per_sec / ut.KM_PER_MILE * 60 * 60
print('mph = %r' % (mph,))

1 / km_per_sec

import datetime
thresh_sec = datetime.timedelta(minutes=5).seconds
thresh_km = thresh_sec * km_per_sec
print('thresh_sec = %r' % (thresh_sec,))
print('thresh_km = %r' % (thresh_km,))
thresh_sec = thresh_km / km_per_sec
print('thresh_sec = %r' % (thresh_sec,))
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import utool as ut
import scipy.cluster.hierarchy
from scipy.spatial import distance

(print, rrr, profile) = ut.inject2(__name__)


KM_PER_SEC = 0.002


def haversine(latlon1, latlon2):
    r"""
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        latlon1 (tuple): (lat, lon)
        latlon2 (tuple): (lat, lon)

    Returns:
        float : distance in kilometers

    References:
        en.wikipedia.org/wiki/Haversine_formula
        gis.stackexchange.com/questions/81551/matching-gps-tracks
        stackoverflow.com/questions/4913349/haversine-distance-gps-points

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import functools
        >>> latlon1 = [-80.21895315, -158.81099213]
        >>> latlon2 = [  9.77816711,  -17.27471498]
        >>> kilometers = haversine(latlon1, latlon2)
        >>> result = ('kilometers = %s' % (kilometers,))
        >>> print(result)
        kilometers = 11930.909364189827
    """
    # convert decimal degrees to radians
    lat1, lon1 = np.radians(latlon1)
    lat2, lon2 = np.radians(latlon2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    # convert to kilometers
    EARTH_RADIUS_KM = 6367
    kilometers = EARTH_RADIUS_KM * c
    return kilometers


def haversine_rad(lat1, lon1, lat2, lon2):
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    # convert to kilometers
    EARTH_RADIUS_KM = 6367
    kilometers = EARTH_RADIUS_KM * c
    return kilometers


def timespace_distance_km(pt1, pt2, km_per_sec=KM_PER_SEC):
    """
    Computes distance between two points in space and time.
    Time is converted into spatial units using km_per_sec

    Args:
        pt1 (tuple) : (seconds, lat, lon)
        pt2 (tuple) : (seconds, lat, lon)
        km_per_sec (float): reasonable animal walking speed

    Returns:
        float: distance in kilometers

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import functools
        >>> km_per_sec = .02
        >>> latlon1 = [40.779299,-73.9719498] # museum of natural history
        >>> latlon2 = [37.7336402,-122.5050342] # san fransisco zoo
        >>> pt1 = [0.0] + latlon1
        >>> pt2 = [0.0] + latlon2
        >>> # google measures about 4138.88 kilometers
        >>> dist_km1 = timespace_distance_km(pt1, pt2)
        >>> print('dist_km1 = {!r}'.format(dist_km1))
        >>> # Now add a time component
        >>> pt1 = [360.0] + latlon1
        >>> pt2 = [0.0] + latlon2
        >>> dist_km2 = timespace_distance_km(pt1, pt2)
        >>> print('dist_km2 = {!r}'.format(dist_km2))
        >>> assert np.isclose(dist_km1, 4136.4568647922624)
        >>> assert np.isclose(dist_km2, 4137.1768647922627)
    """
    sec1, latlon1 = pt1[0], pt1[1:]
    sec2, latlon2 = pt2[0], pt2[1:]
    # Get pure gps distance
    km_dist = haversine(latlon1, latlon2)
    # Get distance in seconds and convert to km
    sec_dist = np.abs(sec1 - sec2) * km_per_sec
    # Add distances
    # (return nan if points are not comparable, otherwise nansum)
    parts = np.array([km_dist, sec_dist])
    timespace_dist = np.nan if np.all(np.isnan(parts)) else np.nansum(parts)
    return timespace_dist


def timespace_distance_sec(pt1, pt2, km_per_sec=KM_PER_SEC):
    # Return in seconds
    sec1, latlon1 = pt1[0], pt1[1:]
    sec2, latlon2 = pt2[0], pt2[1:]
    # Get pure gps distance and convert to seconds
    km_dist = haversine(latlon1, latlon2)
    km_dist = km_dist / km_per_sec
    # Get distance in seconds
    sec_dist = np.abs(sec1 - sec2)
    # Add distances
    # (return nan if points are not comparable, otherwise nansum)
    parts = np.array([km_dist, sec_dist])
    timespace_dist = np.nan if np.all(np.isnan(parts)) else np.nansum(parts)
    return timespace_dist


def space_distance_sec(pt1, pt2, km_per_sec=KM_PER_SEC):
    # Return in seconds
    latlon1, latlon2 = pt1, pt2
    # Get pure gps distance and convert to seconds
    km_dist = haversine(latlon1, latlon2)
    space_dist = km_dist / km_per_sec
    return space_dist


def space_distance_km(pt1, pt2):
    # Return in seconds
    latlon1, latlon2 = pt1, pt2
    # Get pure gps distance and convert to seconds
    km_dist = haversine(latlon1, latlon2)
    return km_dist


def time_dist_sec(sec1, sec2):
    sec_dist = np.abs(sec1 - sec2)
    return sec_dist


def time_dist_km(sec1, sec2, km_per_sec=KM_PER_SEC):
    sec_dist = np.abs(sec1 - sec2)
    sec_dist *= km_per_sec
    return sec_dist


def prepare_data(posixtimes, latlons, km_per_sec=KM_PER_SEC, thresh_units='seconds'):
    r"""
    Package datas and picks distance function

    Args:
        posixtimes (ndarray):
        latlons (ndarray):
        km_per_sec (float): (default = 0.002)
        thresh_units (str): (default = 'seconds')

    Returns:
        ndarray: arr_ -

    CommandLine:
        python -m wbia.algo.preproc.occurrence_blackbox prepare_data

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> posixtimes = np.array([10, 50, np.nan, np.nan, 5, 80, np.nan, np.nan])
        >>> latlons = np.array([
        >>>     (42.727985, -73.683994),
        >>>     (np.nan, np.nan),
        >>>     (np.nan, np.nan),
        >>>     (42.658333, -73.770993),
        >>>     (42.227985, -73.083994),
        >>>     (np.nan, np.nan),
        >>>     (np.nan, np.nan),
        >>>     (42.258333, -73.470993),
        >>> ])
        >>> km_per_sec = 0.002
        >>> thresh_units = 'seconds'
        >>> X_data, dist_func, columns = prepare_data(posixtimes, latlons, km_per_sec, thresh_units)
        >>> result = ('arr_ = %s' % (ut.repr2(X_data),))
        >>> [dist_func(a, b) for a, b in ut.combinations(X_data, 2)]
        >>> print(result)
    """

    def atleast_nd(arr, n, tofront=False):
        r""" ut.static_func_source(vt.atleast_nd) """
        arr_ = np.asanyarray(arr)
        ndims = len(arr_.shape)
        if n is not None and ndims < n:
            # append the required number of dimensions to the end
            if tofront:
                expander = (None,) * (n - ndims) + (Ellipsis,)
            else:
                expander = (Ellipsis,) + (None,) * (n - ndims)
            arr_ = arr_[expander]
        return arr_

    def ensure_column_shape(arr, num_cols):
        r""" ut.static_func_source(vt.ensure_shape) """
        arr_ = np.asanyarray(arr)
        if len(arr_.shape) == 0:
            pass
        elif len(arr_.shape) == 1:
            arr_.shape = (arr_.size, num_cols)
        else:
            assert arr_.shape[1] == num_cols, 'bad number of cols'
        return arr_

    have_gps = latlons is not None and not np.all(np.isnan(latlons))
    have_times = posixtimes is not None and not np.all(np.isnan(posixtimes))

    if not have_gps and not have_times:
        # There is no data, so there is nothing to do
        dist_func = None
        X_data = None
        columns = tuple()
    elif not have_gps and have_times:
        # We have gps but no timestamps
        X_data = atleast_nd(posixtimes, 2)
        if thresh_units == 'seconds':
            dist_func = time_dist_sec
        elif thresh_units == 'km':
            dist_func = time_dist_km
        columns = ('time',)
    elif have_gps and not have_times:
        # We have timesamps but no gps
        X_data = np.array(latlons)
        if thresh_units == 'seconds':
            dist_func = functools.partial(space_distance_sec, km_per_sec=km_per_sec)
        elif thresh_units == 'km':
            dist_func = space_distance_km
        columns = ('lat', 'lon')
    elif have_gps and have_times:
        # We have some combination of gps and timestamps
        posixtimes = atleast_nd(posixtimes, 2)
        latlons = ensure_column_shape(latlons, 2)
        # latlons = np.array(latlons, ndmin=2)
        X_data = np.hstack([posixtimes, latlons])
        if thresh_units == 'seconds':
            dist_func = functools.partial(timespace_distance_sec, km_per_sec=km_per_sec)
        elif thresh_units == 'km':
            dist_func = functools.partial(timespace_distance_km, km_per_sec=km_per_sec)
        columns = ('time', 'lat', 'lon')
    else:
        raise AssertionError('impossible state')
    return X_data, dist_func, columns


def cluster_timespace_km(posixtimes, latlons, thresh_km, km_per_sec=KM_PER_SEC):
    """
    Agglometerative clustering of time/space data

    Args:
        X_data (ndarray) : Nx3 array where columns are (seconds, lat, lon)
        thresh_km (float) : threshold in kilometers

    References:
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
            scipy.cluster.hierarchy.linkage.html
            scipy.cluster.hierarchy.fcluster.html

    Notes:
        # Visualize spots
        http://www.darrinward.com/lat-long/?id=2009879

    CommandLine:
        python -m wbia.algo.preproc.occurrence_blackbox cluster_timespace_km

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> # Nx1 matrix denoting groundtruth locations (for testing)
        >>> X_name = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
        >>> # Nx3 matrix where each columns are (time, lat, lon)
        >>> X_data = np.array([
        >>>     (0, 42.727985, -73.683994),  # MRC
        >>>     (0, 42.657414, -73.774448),  # Park1
        >>>     (0, 42.658333, -73.770993),  # Park2
        >>>     (0, 42.654384, -73.768919),  # Park3
        >>>     (0, 42.655039, -73.769048),  # Park4
        >>>     (0, 42.657872, -73.764148),  # Park5
        >>>     (0, 42.876974, -73.819311),  # CP1
        >>>     (0, 42.862946, -73.804977),  # CP2
        >>>     (0, 42.849809, -73.758486),  # CP3
        >>> ])
        >>> thresh_km = 5.0  # kilometers
        >>> posixtimes = X_data.T[0]
        >>> latlons = X_data.T[1:3].T
        >>> km_per_sec = KM_PER_SEC
        >>> X_labels = cluster_timespace_km(posixtimes, latlons, thresh_km)
        >>> result = 'X_labels = {}'.format(ut.repr2(X_labels))
        >>> print(result)
        X_labels = np.array([3, 2, 2, 2, 2, 2, 1, 1, 1])
    """
    X_data, dist_func, columns = prepare_data(posixtimes, latlons, km_per_sec, 'km')

    if X_data is None:
        return None

    # Compute pairwise distances between all inputs
    dist_func = functools.partial(dist_func, km_per_sec=km_per_sec)
    condenced_dist_mat = distance.pdist(X_data, dist_func)
    # Compute heirarchical linkages
    linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat, method='single')
    # Cluster linkages
    X_labels = scipy.cluster.hierarchy.fcluster(
        linkage_mat, thresh_km, criterion='distance'
    )
    return X_labels


def cluster_timespace_sec(posixtimes, latlons, thresh_sec=5, km_per_sec=KM_PER_SEC):
    """
    Args:
        X_data (ndarray) : Nx3 array where columns are (seconds, lat, lon)
        thresh_sec (float) : threshold in seconds

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> # Nx1 matrix denoting groundtruth locations (for testing)
        >>> X_name = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
        >>> # Nx3 matrix where each columns are (time, lat, lon)
        >>> X_data = np.array([
        >>>     (0, 42.727985, -73.683994),  # MRC
        >>>     (0, 42.657414, -73.774448),  # Park1
        >>>     (0, 42.658333, -73.770993),  # Park2
        >>>     (0, 42.654384, -73.768919),  # Park3
        >>>     (0, 42.655039, -73.769048),  # Park4
        >>>     (0, 42.657872, -73.764148),  # Park5
        >>>     (0, 42.876974, -73.819311),  # CP1
        >>>     (0, 42.862946, -73.804977),  # CP2
        >>>     (0, 42.849809, -73.758486),  # CP3
        >>> ])
        >>> posixtimes = X_data.T[0]
        >>> latlons = X_data.T[1:3].T
        >>> thresh_sec = 250  # seconds
        >>> X_labels = cluster_timespace_sec(posixtimes, latlons, thresh_sec)
        >>> result = ('X_labels = %r' % (X_labels,))
        >>> print(result)
        X_labels = array([6, 4, 4, 4, 4, 5, 1, 2, 3])

    Doctest:
        >>> from wbia.algo.preproc.occurrence_blackbox import *  # NOQA
        >>> # Nx1 matrix denoting groundtruth locations (for testing)
        >>> X_name = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
        >>> # Nx3 matrix where each columns are (time, lat, lon)
        >>> X_data = np.array([
        >>>     (np.nan, 42.657414, -73.774448),  # Park1
        >>>     (0, 42.658333, -73.770993),  # Park2
        >>>     (np.nan, np.nan, np.nan),  # Park3
        >>>     (np.nan, np.nan, np.nan),  # Park3.5
        >>>     (0, 42.655039, -73.769048),  # Park4
        >>>     (0, 42.657872, -73.764148),  # Park5
        >>> ])
        >>> posixtimes = X_data.T[0]
        >>> latlons = X_data.T[1:3].T
        >>> thresh_sec = 250  # seconds
        >>> km_per_sec = KM_PER_SEC
        >>> X_labels = cluster_timespace_sec(posixtimes, latlons, thresh_sec)
        >>> result = 'X_labels = {}'.format(ut.repr2(X_labels))
        >>> print(result)
        X_labels = np.array([3, 4, 1, 2, 4, 5])
    """
    X_data, dist_func, columns = prepare_data(posixtimes, latlons, km_per_sec, 'seconds')
    if X_data is None:
        return None

    # Cluster nan distributions differently
    X_bools = ~np.isnan(X_data)
    group_id = (X_bools * np.power(2, [2, 1, 0])).sum(axis=1)
    import vtool as vt

    unique_ids, groupxs = vt.group_indices(group_id)
    grouped_labels = []
    for xs in groupxs:
        X_part = X_data.take(xs, axis=0)
        labels = _cluster_part(X_part, dist_func, columns, thresh_sec, km_per_sec)
        grouped_labels.append((labels, xs))
    # Undo grouping and rectify overlaps
    X_labels = _recombine_labels(grouped_labels)
    # Do clustering
    return X_labels


def _recombine_labels(chunk_labels):
    """
    Ensure each group has different indices

    chunk_labels = grouped_labels
    """
    import utool as ut

    labels = ut.take_column(chunk_labels, 0)
    idxs = ut.take_column(chunk_labels, 1)
    # nunique_list = [len(np.unique(a)) for a in labels]
    chunksizes = ut.lmap(len, idxs)
    cumsum = np.cumsum(chunksizes).tolist()
    combined_idxs = np.hstack(idxs)
    combined_labels = np.hstack(labels)
    offset = 0
    # Ensure each chunk has unique labels
    for start, stop in zip([0] + cumsum, cumsum):
        combined_labels[start:stop] += offset
        offset += len(np.unique(combined_labels[start:stop]))
    # Ungroup
    X_labels = np.empty(combined_idxs.max() + 1, dtype=np.int)
    # new_labels[:] = -1
    X_labels[combined_idxs] = combined_labels
    return X_labels


def _cluster_part(X_part, dist_func, columns, thresh_sec, km_per_sec):
    if len(X_part) > 500 and 'time' in columns and ~np.isnan(X_part[0, 0]):
        # Try and break problem up into smaller chunks by finding feasible
        # one-dimensional breakpoints (is this a cutting plane?)
        chunk_labels = []
        chunk_idxs = list(_chunk_time(X_part, thresh_sec))
        for idxs in chunk_idxs:
            # print('Doing occurrence chunk {}'.format(len(idxs)))
            X_chunk = X_part.take(idxs, axis=0)
            labels = _cluster_chunk(X_chunk, dist_func, thresh_sec)
            chunk_labels.append((labels, idxs))
        X_labels = _recombine_labels(chunk_labels)
    else:
        # Compute the whole problem
        X_labels = _cluster_chunk(X_part, dist_func, thresh_sec)
    return X_labels


def _cluster_chunk(X_data, dist_func, thresh_sec):
    if len(X_data) == 0:
        X_labels = np.empty(len(X_data), dtype=np.int)
    elif len(X_data) == 1:
        X_labels = np.ones(len(X_data), dtype=np.int)
    elif np.all(np.isnan(X_data)):
        X_labels = np.arange(1, len(X_data) + 1, dtype=np.int)
    else:
        # Compute pairwise distances between all inputs
        condenced_dist_mat = distance.pdist(X_data, dist_func)
        # Compute heirarchical linkages
        linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat, method='single')
        # Cluster linkages
        X_labels = scipy.cluster.hierarchy.fcluster(
            linkage_mat, thresh_sec, criterion='distance'
        )
    return X_labels


def _chunk_time(X_part, thresh_sec):
    X_time = X_part.T[0]
    time_sortx = X_time.argsort()
    timedata = X_time[time_sortx]
    # Look for points that are beyond the thresh in one dimension
    consec_delta = np.diff(timedata)
    consec_delta[consec_delta > thresh_sec]
    breakpoint = (consec_delta > thresh_sec) | np.isnan(consec_delta)
    idxs = np.hstack([[0], np.where(breakpoint)[0] + 1, [len(X_part)]])
    iter_window = list(zip(idxs, idxs[1:]))
    for start, stop in iter_window:
        idxs = time_sortx[start:stop]
        # chunk = X_time[idxs]
        # print((np.diff(chunk[chunk.argsort()]) > thresh_sec).sum())
        yield idxs


# def _chunk_lat(X_chunk, thresh_sec, km_per_sec):
#     # X_time = X_chunk.T[0]
#     X_lats = X_chunk.T[-2]
#     X_lons = X_chunk.T[-1]

#     # approximates 1 dimensional distance
#     ave_lon = np.mean(X_lons)
#     lat_sortx = X_lats.argsort()
#     latdata = X_lats[lat_sortx]

#     latdata_rad = np.radians(latdata)
#     avelon_rad  = np.radians(ave_lon)

#     lat1 = latdata_rad[:-1]
#     lon1 = avelon_rad

#     lat2 = latdata_rad[1:]
#     lon2 = avelon_rad

#     consec_kmdelta = haversine_rad(lat1, lon1, lat2, lon2)
#     consec_delta = consec_kmdelta / km_per_sec

#     consec_delta[consec_delta > thresh_sec]
#     breakpoint = (consec_delta > thresh_sec) | np.isnan(consec_delta)
#     idxs = np.hstack([[0], np.where(breakpoint)[0] + 1, [len(X_chunk)]])
#     iter_window = list(zip(idxs, idxs[1:]))
#     for start, stop in iter_window:
#         idxs = lat_sortx[start:stop]
#         # chunk = X_time[idxs]
#         # print((np.diff(chunk[chunk.argsort()]) > thresh_sec).sum())
#         yield idxs


# def _chunk_lon(X_chunk, thresh_sec, km_per_sec):
#     # X_time = X_chunk.T[0]
#     X_lats = X_chunk.T[-2]
#     X_lons = X_chunk.T[-1]

#     # approximates 1 dimensional distance (assuming lons are not too different)
#     ave_lat = np.mean(X_lats)
#     lon_sortx = X_lons.argsort()
#     londata = X_lons[lon_sortx]

#     londata_rad = np.radians(londata)
#     avelat_rad  = np.radians(ave_lat)

#     lat1 = avelat_rad
#     lon1 = londata_rad[:-1]

#     lat2 = avelat_rad
#     lon2 = londata_rad[1:]

#     consec_kmdelta = haversine_rad(lat1, lon1, lat2, lon2)
#     consec_delta = consec_kmdelta / km_per_sec

#     consec_delta[consec_delta > thresh_sec]
#     breakpoint = (consec_delta > thresh_sec) | np.isnan(consec_delta)
#     idxs = np.hstack([[0], np.where(breakpoint)[0] + 1, [len(X_chunk)]])
#     iter_window = list(zip(idxs, idxs[1:]))
#     for start, stop in iter_window:
#         idxs = lon_sortx[start:stop]
#         # chunk = X_time[idxs]
#         # print((np.diff(chunk[chunk.argsort()]) > thresh_sec).sum())
#         yield idxs


def main():
    """
    CommandLine:
        ib
        cd ~/code/wbia/wbia/algo/preproc
        python occurrence_blackbox.py --lat 42.727985 42.657414 42.658333 42.654384 --lon -73.683994 -73.774448 -73.770993 -73.768919 --sec 0 0 0 0
        # Should return
        X_labels = [2, 1, 1, 1]
    """
    import argparse

    parser = argparse.ArgumentParser(description='Compute agglomerative cluster')
    parser.add_argument('--lat', type=float, nargs='*', help='list of latitude coords')
    parser.add_argument('--lon', type=float, nargs='*', help='list of longitude coords')
    parser.add_argument(
        '--sec', type=float, nargs='*', help='list of POSIX_TIMEs in seconds'
    )
    parser.add_argument(
        '--thresh', type=float, nargs=1, default=1.0, help='threshold in kilometers'
    )
    parser.add_argument(
        '--km_per_sec',
        type=float,
        nargs=1,
        default=KM_PER_SEC,
        help='reasonable animal speed in km/s',
    )
    args = parser.parse_args()
    sec = [0] * len(args.lat) if args.sec is None else args.sec
    latlons = np.vstack([args.lat, args.lon]).T
    X_labels = cluster_timespace_km(sec, latlons, args.thresh, km_per_sec=args.km_per_sec)
    print('X_labels = %r' % (X_labels.tolist(),))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.preproc.occurrence_blackbox
        python -m wbia.algo.preproc.occurrence_blackbox --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    if not ut.doctest_funcs():
        main()
