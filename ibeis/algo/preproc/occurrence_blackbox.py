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
import scipy.cluster.hierarchy
from scipy.spatial import distance


KM_PER_SEC = .002


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

    Example:
        >>> from occurrence_blackbox import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import functools
        >>> latlon1 = [-80.21895315, -158.81099213]
        >>> latlon2 = [  9.77816711,  -17.27471498]
        >>> kilometers = haversine(latlon1, latlon2)
        >>> result = ('kilometers = %s' % (kilometers,))
        >>> print(result)
        kilometers = 11930.9093642
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

    Example:
        >>> from occurrence_blackbox import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import functools
        >>> km_per_sec = .02
        >>> latlon1 = [-80.21895315, -158.81099213]
        >>> latlon2 = [  9.77816711,  -17.27471498]
        >>> pt1 = [360.0] + latlon1
        >>> pt2 = [0.0] + latlon2
        >>> kilometers = timespace_distance_km(pt1, pt2)
        >>> result = ('kilometers = %s' % (kilometers,))
        >>> print(result)
        kilometers = 2058.6323187
    """
    sec1, latlon1 = pt1[0], pt1[1:]
    sec2, latlon2 = pt2[0], pt2[1:]
    # Get pure gps distance
    km_dist = haversine(latlon1, latlon2)
    # Get distance in seconds and convert to km
    sec_dist = np.abs(sec1 - sec2) * km_per_sec
    # Add distances
    timespace_dist = km_dist + sec_dist
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
    timespace_dist = km_dist + sec_dist
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
    # Package data and pick distance function

    def atleast_nd(arr, n, tofront=False):
        r""" ut.static_func_source(vt.atleast_nd) """
        arr_ = np.asanyarray(arr)
        ndims = len(arr_.shape)
        if n is not None and ndims <  n:
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

    if latlons is None and posixtimes is None:
        dist_func = None
        X_data = None
    elif latlons is None and posixtimes is not None:
        X_data = atleast_nd(posixtimes, 2)
        if thresh_units == 'seconds':
            dist_func = time_dist_sec
        elif thresh_units == 'km':
            dist_func = time_dist_km
    elif latlons is not None and posixtimes is None:
        X_data = np.array(latlons)
        if thresh_units == 'seconds':
            dist_func = functools.partial(space_distance_sec,
                                          km_per_sec=km_per_sec)
        elif thresh_units == 'km':
            dist_func = space_distance_km
    else:
        posixtimes = atleast_nd(posixtimes, 2)
        latlons = ensure_column_shape(latlons, 2)
        #latlons = np.array(latlons, ndmin=2)
        X_data = np.hstack([posixtimes, latlons])
        if thresh_units == 'seconds':
            dist_func = functools.partial(timespace_distance_sec,
                                          km_per_sec=km_per_sec)
        elif thresh_units == 'km':
            dist_func = functools.partial(timespace_distance_km,
                                          km_per_sec=km_per_sec)
    return X_data, dist_func


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

    Example:
        >>> # DISABLE_DOCTEST
        >>> from occurrence_blackbox import *  # NOQA
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
        >>> X_labels = cluster_timespace_sec(posixtimes, latlons, thresh_km)
        >>> result = ('X_labels = %r' % (X_labels,))
        >>> print(result)
        X_labels = array([3, 2, 2, 2, 2, 2, 1, 1, 1], dtype=int32)
    """
    X_data, dist_func = prepare_data(posixtimes, latlons, km_per_sec, 'km')

    # Compute pairwise distances between all inputs
    dist_func = functools.partial(dist_func, km_per_sec=km_per_sec)
    condenced_dist_mat = distance.pdist(X_data, dist_func)
    # Compute heirarchical linkages
    linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat,
                                                  method='single')
    # Cluster linkages
    X_labels = scipy.cluster.hierarchy.fcluster(linkage_mat, thresh_km,
                                                criterion='distance')
    return X_labels


def cluster_timespace_sec(posixtimes, latlons, thresh_sec=5, km_per_sec=KM_PER_SEC):
    """
    Args:
        X_data (ndarray) : Nx3 array where columns are (seconds, lat, lon)
        thresh_sec (float) : threshold in seconds

    Example:
        >>> # DISABLE_DOCTEST
        >>> from occurrence_blackbox import *  # NOQA
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
        X_labels = array([3, 2, 2, 2, 2, 2, 1, 1, 1], dtype=int32)
    """
    X_data, dist_func = prepare_data(posixtimes, latlons, km_per_sec, 'seconds')
    # Do clustering
    if X_data is None:
        X_labels = None
    elif len(X_data) == 0:
        X_labels = np.empty(0, dtype=np.int)
    elif len(X_data) == 1:
        X_labels = np.zeros(1, dtype=np.int)
    else:
        # Compute pairwise distances between all inputs
        condenced_dist_mat = distance.pdist(X_data, dist_func)
        # Compute heirarchical linkages
        linkage_mat = scipy.cluster.hierarchy.linkage(condenced_dist_mat,
                                                      method='single')
        # Cluster linkages
        X_labels = scipy.cluster.hierarchy.fcluster(linkage_mat, thresh_sec,
                                                    criterion='distance')
    return X_labels


def main():
    """
    CommandLine:
        ib
        cd ~/code/ibeis/ibeis/algo/preproc
        python occurrence_blackbox.py --lat 42.727985 42.657414 42.658333 42.654384 --lon -73.683994 -73.774448 -73.770993 -73.768919 --sec 0 0 0 0
        # Should return
        X_labels = [2, 1, 1, 1]
    """
    import argparse
    parser = argparse.ArgumentParser(description='Compute agglomerative cluster')
    parser.add_argument('--lat', type=float, nargs='*', help='list of latitude coords')
    parser.add_argument('--lon', type=float, nargs='*', help='list of longitude coords')
    parser.add_argument('--sec', type=float, nargs='*', help='list of POSIX_TIMEs in seconds')
    parser.add_argument('--thresh', type=float, nargs=1, default=1., help='threshold in kilometers')
    parser.add_argument('--km_per_sec', type=float, nargs=1, default=KM_PER_SEC, help='reasonable animal speed in km/s')
    args = parser.parse_args()
    sec = [0] * len(args.lat) if args.sec is None else args.sec
    latlons = np.vstack([args.lat, args.lon]).T
    X_labels = cluster_timespace_km(sec, latlons, args.thresh, km_per_sec=args.km_per_sec)
    print('X_labels = %r' % (X_labels.tolist(),))


if __name__ == '__main__':
    main()
