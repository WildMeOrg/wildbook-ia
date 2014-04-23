from __future__ import absolute_import, division, print_function
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)
import vtool.exif as exif
from PIL import Image
import hashlib
import uuid
from utool import util_time


GPSInfo_TAGID          = exif.EXIF_TAG_TO_TAGID['GPSInfo']
DateTimeOriginal_TAGID = exif.EXIF_TAG_TO_TAGID['DateTimeOriginal']


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


def get_image_uuid(pil_img):
    # Read PIL image data
    img_bytes_ = pil_img.tobytes()
    # hash the bytes using sha1
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def add_images_paramters_gen(gpath_list):
    """ generates values for add_images sqlcommands """
    mark_prog, end_prog = utool.progress_func(len(gpath_list), lbl='imgs: ')
    for count, gpath in enumerate(gpath_list):
        mark_prog(count)
        try:
            pil_img = Image.open(gpath, 'r')      # Open PIL Image
        except IOError as ex:
            print(ex)
            print('FAILED TO READ: %r' % gpath)
            if str(ex) == 'cannot identify image file':
                continue
            else:
                raise
        width, height  = pil_img.size         # Read width, height
        time, lat, lon = get_exif(pil_img)    # Read exif tags
        image_uuid = get_image_uuid(pil_img)  # Read pixels ]-hash-> guid = gid
        yield (image_uuid, gpath, width, height, time, lat, lon)
    end_prog()
