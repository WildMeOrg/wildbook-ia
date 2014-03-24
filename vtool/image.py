# LICENCE
from __future__ import print_function, division
# Science
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from utool import util_path
from utool import util_str
from utool import util_progress


CV2_WARP_KWARGS = {'flags': cv2.INTER_LANCZOS4,
                   'borderMode': cv2.BORDER_CONSTANT}


def imread(img_fpath):
    try:
        # opencv always reads in BGR mode (fastest load time)
        imgBGR = cv2.imread(img_fpath, flags=cv2.CV_LOAD_IMAGE_COLOR)
        return imgBGR
    except Exception as ex:
        print('[gtool] Caught Exception: %r' % ex)
        print('[gtool] ERROR reading: %r' % (img_fpath,))
        raise


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def warpAffine(img, M, dsize):
    warped_img = cv2.warpAffine(img, M[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img


def check_exif_keys(pil_image):
    info_ = pil_image._getexif()
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


def read_all_exif_tags(pil_image):
    info_ = pil_image._getexif()
    info_iter = info_.iteritems()
    tag_ = lambda key: TAGS.get(key, key)
    exif = {} if info_ is None else {tag_(k): v for k, v in info_iter}
    return exif


def read_one_exif_tag(pil_image, tag):
    try:
        exif_key = TAGS.keys()[TAGS.values().index(tag)]
    except ValueError:
        return 'Invalid EXIF Tag'
    info_ = pil_image._getexif()
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
        #check_exif_keys(pil_image)


def read_exif(fpath, tag=None):
    try:
        pil_image = Image.open(fpath)
        if not hasattr(pil_image, '_getexif'):
            return 'No EXIF Data'
    except IOError as ex:
        import argparse2
        print('Caught IOError: %r' % (ex,))
        print_image_checks(fpath)
        if argparse2.ARGS_.strict:
            raise
        return {} if tag is None else None
    if tag is None:
        exif = read_all_exif_tags(pil_image)
    else:
        exif = read_one_exif_tag(pil_image, tag)
    del pil_image
    return exif


def print_image_checks(img_fpath):
    hasimg = util_path.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, util_str.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


def get_exif(image, ext):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}

    if ext in EXIF_EXTENSIONS:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
     
                    exif_data[decoded] = gps_data
                elif decoded == "DateTimeOriginal":
                    exif_data[decoded] = calendar.timegm(tuple(map(int, value.replace(" ", ":").split(":")) + [0,0,0]))
                else:
                    exif_data[decoded] = value
     
    if "GPSInfo" not in exif_data:
        exif_data["GPSInfo"] = [-1.0, -1.0]
    if "DateTimeOriginal" not in exif_data:
        exif_data["DateTimeOriginal"] = -1.0

    return [ exif_data["DateTimeOriginal"], exif_data["GPSInfo"][0], exif_data["GPSInfo"][1] ]
 

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
 
    if "GPSInfo" in exif_data:      
        gps_info = exif_data["GPSInfo"]
 
        gps_latitude = get_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = get_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = get_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = get_exist(gps_info, 'GPSLongitudeRef')
 
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_degrees(gps_latitude)
            if gps_latitude_ref != "N":                  
                lat = 0 - lat
 
            lon = convert_degrees(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon
 
    return lat, lon
