from __future__ import absolute_import, division, print_function
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)
import vtool.exif as exif
from PIL import Image
import numpy as np
import hashlib
import uuid
from utool import util_time


GPSInfo_TAGID          = exif.EXIF_TAG_TO_TAGID['GPSInfo']
DateTimeOriginal_TAGID = exif.EXIF_TAG_TO_TAGID['DateTimeOriginal']


@profile
def get_exif(pil_img):
    """ Image EXIF helper """
    exif_dict = exif.get_exif_dict(pil_img)
    exiftime = exif_dict.get(DateTimeOriginal_TAGID, -1)
    # TODO: Fixme
    #latlon = exif_dict.get(GPSInfo_TAGID, (-1, -1))
    latlon = (-1, -1)
    time = util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    lat, lon = latlon
    return time, lat, lon


@profile
def get_image_uuid(img_bytes_):
    # Read PIL image data
    #img_bytes_ = pil_img.tobytes()
    # hash the bytes using sha1
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def preprocess_image(gpath):
    """
    This function should be called in parallel
    """
    try:
        pil_img = Image.open(gpath, 'r')      # Open PIL Image
    except IOError as ex:
        print('FAILED TO READ: %r' % gpath)
        if str(ex).startswith('cannot identify image file'):
            param_tup = None
            return param_tup
        else:
            raise
    width, height  = pil_img.size         # Read width, height
    time, lat, lon = get_exif(pil_img)    # Read exif tags
    img_bytes_ = np.asarray(pil_img).ravel()[::64].tostring()
    image_uuid = get_image_uuid(img_bytes_)  # Read pixels ]-hash-> guid = gid
    param_tup  = (image_uuid, gpath, width, height, time, lat, lon)
    return param_tup


@profile
def add_images_params_gen(gpath_list):
    """ generates values for add_images sqlcommands """
    mark_prog, end_prog = utool.progress_func(len(gpath_list), lbl='imgs: ',
                                              mark_start=True, flush_after=4)
    for count, gpath in enumerate(gpath_list):
        param_tup = preprocess_image(gpath)
        mark_prog(count)
        yield param_tup
    end_prog()
