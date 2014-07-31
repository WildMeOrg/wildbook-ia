# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import zip, range
import six
from PIL.ExifTags import TAGS
from PIL import Image
#from utool import util_progress
import utool
from . import image as gtool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[exif]', DEBUG=False)


# Inverse of PIL.ExifTags.TAGS
EXIF_TAG_TO_TAGID = {val: key for (key, val) in six.iteritems(TAGS)}


@profile
def read_exif_tags(pil_img, exif_tagid_list, default_list=None):
    if default_list is None:
        default_list = [None for _ in range(len(exif_tagid_list))]
    exif_dict = get_exif_dict(pil_img)
    exif_val_list = [exif_dict.get(key, default) for key, default in
                     zip(exif_tagid_list, default_list)]
    return exif_val_list


@profile
def get_exif_dict(pil_img):
    """ Returns exif dictionary by TAGID """
    try:
        exif_dict = pil_img._getexif()
        if exif_dict is None:
            raise AttributeError
        assert isinstance(exif_dict, dict), 'type(exif_dict)=%r' % type(exif_dict)
    except IndexError:
        exif_dict = {}
    except AttributeError:
        exif_dict = {}
    except OverflowError:
        exif_dict = {}
    except Exception as ex:
        utool.printex(ex, 'get_exif_dict failed in an unexpected way')
        raise
    return exif_dict


@profile
def get_exif_dict2(pil_img):
    """ Returns exif dictionary by TAG (less efficient)"""
    try:
        exif_dict = pil_img._getexif()
        if exif_dict is None:
            raise AttributeError
        assert isinstance(exif_dict, dict), 'type(exif_dict)=%r' % type(exif_dict)
        exif_dict2 = {TAGS.get(key, key): val for (key, val) in six.iteritems(exif_dict)}
    except AttributeError:
        exif_dict2 = {}
    except OverflowError:
        exif_dict2 = {}
    return exif_dict2


@profile
def check_exif_keys(pil_img):
    info_ = pil_img._getexif()
    valid_keys = []
    invalid_keys = []
    for key, val in six.iteritems(info_):
        try:
            exif_keyval = TAGS[key]
            valid_keys.append((key, exif_keyval))
        except KeyError:
            invalid_keys.append(key)
    print('[io] valid_keys = ' + '\n'.join(valid_keys))
    print('-----------')
    #import draw_func2 as df2
    #exec(df2.present())


@profile
def read_all_exif_tags(pil_img):
    info_ = pil_img._getexif()
    info_iter = six.iteritems(info_)
    tag_ = lambda key: TAGS.get(key, key)
    exif = {} if info_ is None else {tag_(k): v for k, v in info_iter}
    return exif


@profile
def get_exif_tagids(tag_list):
    tagid_list = [EXIF_TAG_TO_TAGID[tag] for tag in tag_list]
    return tagid_list


@profile
def read_one_exif_tag(pil_img, tag):
    try:
        exif_key = TAGS.keys()[TAGS.values().index(tag)]
    except ValueError:
        return 'Invalid EXIF Tag'
    info_ = pil_img._getexif()
    if info_ is None:
        return None
    else:
        invalid_str = 'Invalid EXIF Key: exif_key=%r, tag=%r' % (exif_key, tag)
        exif_val = info_.get(exif_key, invalid_str)
    return exif_val


@profile
def read_exif(fpath, tag=None):
    try:
        pil_img = Image.open(fpath)
        if not hasattr(pil_img, '_getexif'):
            return 'No EXIF Data'
    except IOError as ex:
        print('Caught IOError: %r' % (ex,))
        gtool.print_image_checks(fpath)
        raise
        return {} if tag is None else None
    if tag is None:
        exif = read_all_exif_tags(pil_img)
    else:
        exif = read_one_exif_tag(pil_img, tag)
    del pil_img
    return exif


@profile
def get_exist(data, key):
    if key in data:
        return data[key]
    return None


def convert_degrees(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    return d + (m / 60.0) + (s / 3600.0)


@profile
def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif above)"""
    lat = -1.0
    lon = -1.0

    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']

        gps_latitude      = gps_info.get('GPSLatitude', None)
        gps_latitude_ref  = gps_info.get('GPSLatitudeRef', None)
        gps_longitude     = gps_info.get('GPSLongitude', None)
        gps_longitude_ref = gps_info.get('GPSLongitudeRef', None)

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_degrees(gps_latitude)
            if gps_latitude_ref != 'N':
                lat = 0 - lat

            lon = convert_degrees(gps_longitude)
            if gps_longitude_ref != 'E':
                lon = 0 - lon
    return lat, lon
