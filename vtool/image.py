# LICENCE
from __future__ import print_function, division
# Python
from itertools import izip
from os.path import exists
# Science
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from utool import util_path
from utool import util_str
#from utool import util_progress


CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

CV2_WARP_KWARGS = {
    'flags': CV2_INTERPOLATION_TYPES['lanczos'],
    'borderMode': cv2.BORDER_CONSTANT
}


EXIF_TAG_GPS      = 'GPSInfo'
EXIF_TAG_DATETIME =  'DateTimeOriginal'


def imread(img_fpath):
    # opencv always reads in BGR mode (fastest load time)
    imgBGR = cv2.imread(img_fpath, flags=cv2.CV_LOAD_IMAGE_COLOR)
    if imgBGR is None:
        if not exists(img_fpath):
            raise IOError('cannot read img_fpath=%r does not exist' % img_fpath)
        else:
            raise IOError('cannot read img_fpath=%r seems corrupted.' % img_fpath)
    return imgBGR


def imwrite(img_fpath, imgBGR):
    try:
        cv2.imwrite(img_fpath, imgBGR)
    except Exception as ex:
        print('[gtool] Caught Exception: %r' % ex)
        print('[gtool] ERROR reading: %r' % (img_fpath,))
        raise


def get_num_channels(img):
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise Exception('Cannot determine number of channels')
    return nChannels


def open_pil_image(image_fpath):
    pil_img = Image.open(image_fpath)
    return pil_img


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def warpAffine(img, M, dsize):
    warped_img = cv2.warpAffine(img, M[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img


def check_exif_keys(pil_img):
    info_ = pil_img._getexif()
    valid_keys = []
    invalid_keys = []
    for key, val in info_.iteritems():
        try:
            exif_keyval = TAGS[key]
            valid_keys.append((key, exif_keyval))
        except KeyError:
            invalid_keys.append(key)
    print('[io] valid_keys = ' + '\n'.join(valid_keys))
    print('-----------')
    #import draw_func2 as df2
    #exec(df2.present())


def read_all_exif_tags(pil_img):
    info_ = pil_img._getexif()
    info_iter = info_.iteritems()
    tag_ = lambda key: TAGS.get(key, key)
    exif = {} if info_ is None else {tag_(k): v for k, v in info_iter}
    return exif


def get_exif_tagids(tag_list):
    exif_keys  = TAGS.keys()
    exif_vals  = TAGS.values()
    tagid_list = [exif_keys[exif_vals.index(tag)] for tag in tag_list]
    return tagid_list


def get_exif_dict(pil_img):
    try:
        exif_dict = pil_img._getexif()
        if exif_dict is None:
            raise AttributeError
        assert isinstance(exif_dict, dict), 'type(exif_dict)=%r' % type(exif_dict)
    except AttributeError:
        exif_dict = {}
    return exif_dict


def read_exif_tags(pil_img, exif_tagid_list, default_list=None):
    if default_list is None:
        default_list = [None for _ in xrange(len(exif_tagid_list))]
    exif_dict = get_exif_dict(pil_img)
    exif_val_list = [exif_dict.get(key, default) for key, default in
                     izip(exif_tagid_list, default_list)]
    return exif_val_list


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
    #try:
        #exif_val = info_[exif_key]
    #except KeyError:
        #exif_val = 'Invalid EXIF Key: exif_key=%r, tag=%r' % (exif_key, tag)
        #print('')
        #print(exif_val)
        #check_exif_keys(pil_img)


def read_exif(fpath, tag=None):
    try:
        pil_img = Image.open(fpath)
        if not hasattr(pil_img, '_getexif'):
            return 'No EXIF Data'
    except IOError as ex:
        import argparse2
        print('Caught IOError: %r' % (ex,))
        print_image_checks(fpath)
        if argparse2.ARGS_.strict:
            raise
        return {} if tag is None else None
    if tag is None:
        exif = read_all_exif_tags(pil_img)
    else:
        exif = read_one_exif_tag(pil_img, tag)
    del pil_img
    return exif


def print_image_checks(img_fpath):
    hasimg = util_path.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, util_str.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


#def get_exif(image, ext):
    #"""Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    #exif_data = {}
    #if ext in EXIF_EXTENSIONS:
        #info = image._getexif()
        #if info:
            #for tag, value in info.items():
                #decoded = TAGS.get(tag, tag)
                #if decoded == "GPSInfo":
                    #gps_data = {}
                    #for t in value:
                        #sub_decoded = GPSTAGS.get(t, t)
                        #gps_data[sub_decoded] = value[t]

                    #exif_data[decoded] = gps_data
                #elif decoded == "DateTimeOriginal":
                    #exif_data[decoded] = calendar.timegm(tuple(map(int, value.replace(" ", ":").split(":")) + [0,0,0]))
                #else:
                    #exif_data[decoded] = value
    #if EXIF_TAG_GPS not in exif_data:
        #exif_data[EXIF_TAG_GPS] = [-1.0, -1.0]
    #if EXIF_TAG_DATETIME not in exif_data:
        #exif_data[EXIF_TAG_DATETIME] = -1.0

    #return [exif_data[EXIF_TAG_DATETIME], exif_data[EXIF_TAG_GPS][0], exif_data[EXIF_TAG_GPS][1]]


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


def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif above)"""
    lat = -1.0
    lon = -1.0

    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']

        gps_latitude = get_exist(gps_info, 'GPSLatitude')
        gps_latitude_ref = get_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = get_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = get_exist(gps_info, 'GPSLongitudeRef')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_degrees(gps_latitude)
            if gps_latitude_ref != 'N':
                lat = 0 - lat

            lon = convert_degrees(gps_longitude)
            if gps_longitude_ref != 'E':
                lon = 0 - lon

    return lat, lon
