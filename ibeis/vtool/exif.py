# -*- coding: utf-8 -*-
# LICENCE
"""
References:
    http://www.exiv2.org/tags.html
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range
import six
from PIL.ExifTags import TAGS, GPSTAGS
import PIL.ExifTags  # NOQA
from PIL import Image
#from utool import util_progress
import utool as ut
from utool import util_time
from vtool import image_shared
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[exif]', DEBUG=False)


# Inverse of PIL.ExifTags.TAGS
EXIF_TAG_TO_TAGID = {val: key for (key, val) in six.iteritems(TAGS)}
GPS_TAG_TO_GPSID  = {val: key for (key, val) in six.iteritems(GPSTAGS)}

# Relevant EXIF Tags
#'GPSInfo': 34853
#'SensitivityType': 34864  # UNSUPPORTED

GPSINFO_CODE = EXIF_TAG_TO_TAGID['GPSInfo']
DATETIMEORIGINAL_TAGID = EXIF_TAG_TO_TAGID['DateTimeOriginal']
SENSITIVITYTYPE_CODE = 34864  # UNSUPPORTED BY PIL

ORIENTATION_CODE = EXIF_TAG_TO_TAGID['Orientation']
ORIENTATION_UNDEFINED = 'UNDEFINED'
ORIENTATION_000 = 'Normal'
ORIENTATION_090 = '90 Clockwise'
ORIENTATION_180 = 'Upside-Down'
ORIENTATION_270 = '90 Counter-Clockwise'

ORIENTATION_DICT = {
    0: ORIENTATION_UNDEFINED,
    1: ORIENTATION_000,
    2: None,  # Flip Left-to-Right
    3: ORIENTATION_180,
    4: None,  # Flip Top-to-Bottom
    5: None,  # Flip Left-to-Right then Rotate 90
    6: ORIENTATION_090,
    7: None,  # Flip Left-to-Right then Rotate 270
    8: ORIENTATION_270,
}


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
    except (IndexError, AttributeError, OverflowError):
        exif_dict = {}
    except Exception as ex:
        ut.printex(ex, 'get_exif_dict failed in an unexpected way')
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
        exif_dict2 = make_exif_dict_human_readable(exif_dict)
    except AttributeError:
        exif_dict2 = {}
    except OverflowError:
        exif_dict2 = {}
    return exif_dict2


def make_exif_dict_human_readable(exif_dict):
    exif_dict2 = {TAGS.get(key, key): val
                  for (key, val) in six.iteritems(exif_dict)}
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
    exif = {} if info_ is None else {
        TAGS.get(key, key): val
        for key, val in six.iteritems(info_)
    }
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
        image_shared.print_image_checks(fpath)
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
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format

    References:
        http://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    """
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    degrees_float = d + (m / 60.0) + (s / 3600.0)
    return degrees_float


GPSLATITUDE_CODE     = GPS_TAG_TO_GPSID['GPSLatitude']
GPSLATITUDEREF_CODE  = GPS_TAG_TO_GPSID['GPSLatitudeRef']
GPSLONGITUDE_CODE    = GPS_TAG_TO_GPSID['GPSLongitude']
GPSLONGITUDEREF_CODE = GPS_TAG_TO_GPSID['GPSLongitudeRef']


@profile
def get_lat_lon(exif_dict, default=(-1, -1)):
    r"""
    Returns the latitude and longitude, if available, from the provided
    exif_data2 (obtained through exif_data2 above)

    Notes:
        Might need to downgrade to Pillow 2.9.0 to solve a bug with getting GPS
        https://github.com/python-pillow/Pillow/issues/1477

        python -c "from PIL import Image; print(Image.PILLOW_VERSION)"

        pip uninstall Pillow
        pip install Pillow==2.9

    CommandLine:
        python -m vtool.exif --test-get_lat_lon

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.exif import *  # NOQA
        >>> import numpy as np
        >>> image_fpath = ut.grab_file_url('http://images.summitpost.org/original/769474.JPG')
        >>> pil_img = Image.open(image_fpath)
        >>> exif_dict = get_exif_dict(pil_img)
        >>> latlon = get_lat_lon(exif_dict)
        >>> result = np.array_str(np.array(latlon), precision=3)
        >>> print(result)
        [ 41.89   12.486]

    Ignore:
        ut.dict_take(PIL.ExifTags.TAGS, exif_dict.keys(), None)
        exif_dict[GPSINFO_CODE]
        PIL.ExifTags.TAGS[GPSINFO_CODE]
    """
    if GPSINFO_CODE in exif_dict:
        gps_info = exif_dict[GPSINFO_CODE]

        if (GPSLATITUDE_CODE in gps_info and
             GPSLATITUDEREF_CODE in gps_info and
             GPSLONGITUDE_CODE in gps_info and
             GPSLONGITUDEREF_CODE in gps_info):
            gps_latitude      = gps_info[GPSLATITUDE_CODE]
            gps_latitude_ref  = gps_info[GPSLATITUDEREF_CODE]
            gps_longitude     = gps_info[GPSLONGITUDE_CODE]
            gps_longitude_ref = gps_info[GPSLONGITUDEREF_CODE]
            try:
                lat = convert_degrees(gps_latitude)
                if gps_latitude_ref != 'N':
                    lat = 0 - lat

                lon = convert_degrees(gps_longitude)
                if gps_longitude_ref != 'E':
                    lon = 0 - lon
                return lat, lon
            except ZeroDivisionError:
                # FIXME: -1, -1 is not a good invalid GPS
                # Find out what the divide by zero really means
                # currently we think it just is bad gps data
                pass
    return default


def get_orientation(exif_dict, default=0):
    r"""
    Returns the image orientation, if available, from the provided
    exif_data2 (obtained through exif_data2 above)

    CommandLine:
        python -m vtool.exif --test-get_orientation

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.exif import *  # NOQA
        >>> from os.path import join
        >>> import numpy as np
        >>> url = 'https://lev.cs.rpi.edu/public/models/orientation.zip'
        >>> images_path = ut.grab_zipped_url(url)
        >>> result = []
        >>> for index in range(3):
        >>>     image_filename = 'orientation_%05d.JPG' % (index + 1, )
        >>>     pil_img = Image.open(join(images_path, image_filename))
        >>>     exif_dict = get_exif_dict(pil_img)
        >>>     orient = get_orientation(exif_dict)
        >>>     result.append(orient)
        >>> print(result)
        [1, 6, 8]

    Ignore:
        ut.dict_take(PIL.ExifTags.TAGS, exif_dict.keys(), None)
        exif_dict[ORIENTATION_CODE]
        PIL.ExifTags.TAGS[ORIENTATION_CODE]
    """
    if ORIENTATION_CODE in exif_dict:
        orient = exif_dict[ORIENTATION_CODE]
        if orient in ORIENTATION_DICT:
            if ORIENTATION_DICT[orient] is None:
                raise NotImplementedError('Orientation not defined')
            default = orient
        else:
            raise AssertionError('Unrecognized orientation in Exif')
    return default


def get_orientation_str(exif_dict):
    r"""
    Returns the image orientation strings, if available, from the provided
    exif_data2 (obtained through exif_data2 above)

    CommandLine:
        python -m vtool.exif --test-get_orientation_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.exif import *  # NOQA
        >>> from os.path import join
        >>> import numpy as np
        >>> url = 'https://lev.cs.rpi.edu/public/models/orientation.zip'
        >>> images_path = ut.grab_zipped_url(url)
        >>> result = []
        >>> for index in range(3):
        >>>     image_filename = 'orientation_%05d.JPG' % (index + 1, )
        >>>     pil_img = Image.open(join(images_path, image_filename))
        >>>     exif_dict = get_exif_dict(pil_img)
        >>>     orient_str = get_orientation_str(exif_dict)
        >>>     result.append(orient_str)
        >>> print(result)
        ['Normal', '90 Clockwise', '90 Counter-Clockwise']

    Ignore:
        ut.dict_take(PIL.ExifTags.TAGS, exif_dict.keys(), None)
        exif_dict[ORIENTATION_CODE]
        PIL.ExifTags.TAGS[ORIENTATION_CODE]
    """
    orient = get_orientation(exif_dict)
    orient_str = ORIENTATION_DICT[orient]
    return orient_str


def get_unixtime(exif_dict, default=-1):
    """
    TODO: Exif.Image.TimeZoneOffset

    Ignore:
        gpaths = ut.list_images('/home/joncrall/work/humpbacks_fb/_ibsdb/images', full=1)
        gpaths = ut.list_images('/home/joncrall/work/humpbacks/_ibsdb/images', full=1)
        exifs = list(ut.generate(vt.read_exif, gpaths))
        times = ut.dict_take_column(exifs, 'DateTimeOriginal', '!!!!!!!!!!!!!!!!!!!')
        idxs = ut.where([y[-2] == ' ' for y in times])

        gpath = gpaths[idxs[0]]
        exif_dict = vt.get_exif_dict(Image.open(gpath))
        ut.take(times, idxs)
        ut.take(exifs, idxs)
        ut.take(gpaths, idxs)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.exif import *  # NOQA
        >>> image_fpath = ut.grab_file_url('http://images.summitpost.org/original/769474.JPG')
        >>> pil_img = Image.open(image_fpath)
        >>> exif_dict = get_exif_dict(pil_img)
    """
    exiftime  = exif_dict.get(DATETIMEORIGINAL_TAGID, default)
    if isinstance(exiftime, tuple) and len(exiftime) == 1:
        # hack, idk why
        exiftime = exiftime[0]
    if exiftime != -1 and len(exiftime) == 19 and exiftime[-1] != ' ' and exiftime[-3:-1] == ': ':
        # Hack for weird fluke exif times '2009:10:01 11:52: 1'
        exiftime = list(exiftime)
        exiftime[-2] = '0'
        exiftime = ''.join(exiftime)
    unixtime = util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    timezone_offset = exif_dict.get('TimeZoneOffset', None)
    if timezone_offset is not None:
        pass
    return unixtime


def parse_exif_unixtime(image_fpath):
    r"""
    Args:
        image_fpath (str):

    Returns:
        float: unixtime

    CommandLine:
        python -m vtool.exif --test-parse_exif_unixtime

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.exif import *  # NOQA
        >>> image_fpath = ut.grab_file_url('http://images.summitpost.org/original/769474.JPG')
        >>> unixtime = parse_exif_unixtime(image_fpath)
        >>> result = str(unixtime)
        >>> print(result)
        1325351249

        1325369249.0
    """
    pil_img = image_shared.open_pil_image(image_fpath)
    exif_dict = get_exif_dict(pil_img)
    unixtime = get_unixtime(exif_dict)
    return unixtime


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.exif
        python -m vtool.exif --allexamples
        python -m vtool.exif --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
