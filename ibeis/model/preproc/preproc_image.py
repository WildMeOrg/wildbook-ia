from __future__ import absolute_import, division, print_function
# UTool
import utool
import vtool.exif as exif
from PIL import Image
from os.path import splitext, basename
import numpy as np
import hashlib
import uuid
from utool import util_time
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)


GPSInfo_TAGID          = exif.EXIF_TAG_TO_TAGID['GPSInfo']
DateTimeOriginal_TAGID = exif.EXIF_TAG_TO_TAGID['DateTimeOriginal']


@profile
def parse_exif(pil_img):
    """ Image EXIF helper """
    exif_dict = exif.get_exif_dict(pil_img)
    exiftime  = exif_dict.get(DateTimeOriginal_TAGID, -1)
    # TODO: Fixme
    #latlon = exif_dict.get(GPSInfo_TAGID, (-1, -1))
    latlon = (-1, -1)
    time = util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    lat, lon = latlon
    return time, lat, lon


@profile
def get_image_uuid(pil_img):
    # DEPRICATE
    # Read PIL image data (every 64th byte)
    img_bytes_ = np.asarray(pil_img).ravel()[::64].tostring()
    #print('[ginfo] npimg.sum() = %r' % npimg.sum())
    #img_bytes_ = np.asarray(pil_img).ravel().tostring()
    # hash the bytes using sha1
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    #uuid_ = uuid.uuid4()
    #print('[ginfo] hashbytes_16 = %r' % (hashbytes_16,))
    #print('[ginfo] uuid_ = %r' % (uuid_,))
    return uuid_


def get_standard_ext(gpath):
    """ Returns standardized image extension """
    ext = splitext(gpath)[1].lower()
    if ext == '.jpeg':
        ext = '.jpg'
    return ext


@profile
def parse_imageinfo(tup):
    """ Worker function: gpath must be in UNIX-PATH format!
    Input:
        a tuple of arguments (so the function can be parallelized easily)
    Output:
        if successful: returns a tuple of image parameters which are values for SQL columns on
        else: returns None
    """
    # Parse arguments from tuple
    gpath = tup
    #print('[ginfo] gpath=%r' % gpath)
    # Try to open the image
    try:
        pil_img = Image.open(gpath, 'r')  # Open PIL Image
    except IOError as ex:
        print('[preproc] IOError: %s' % (str(ex),))
        return None
    # Parse out the data
    width, height  = pil_img.size         # Read width, height
    time, lat, lon = parse_exif(pil_img)  # Read exif tags
    # We cannot use pixel data as libjpeg is not determenistic (even for reads!)
    image_uuid = utool.get_file_uuid(gpath)  # Read file ]-hash-> guid = gid
    #orig_gpath = gpath
    orig_gname = basename(gpath)
    ext = get_standard_ext(gpath)
    notes = ''
    # Build parameters tuple
    param_tup = (
        image_uuid,
        gpath,
        orig_gname,
        #orig_gpath,
        ext,
        width,
        height,
        time,
        lat,
        lon,
        notes
    )
    #print('[ginfo] %r %r' % (image_uuid, orig_gname))
    return param_tup


def imgparams_worker2(tup):
    #gpath, kwargs = tup
    #cache_dir       = kwargs.get('cache_dir', None)
    #max_width       = kwargs.get('max_image_width', None)
    #max_height      = kwargs.get('max_image_height', None)
    #localize_images = kwargs.get('localize_images', False)

    # Move images to the cache dir
    #if localize_images:
    #    gname = image_uuid + ext
    #    gpath = '/'.join((cache_dir, gname))

    #if width > max_width or height > max_height:
    #    pass
    # TODO: Resize, Filter, and localize Image
    pass


@profile
def add_images_params_gen(gpath_list, **kwargs):
    """ generates values for add_images sqlcommands asychronously

    TEST CODE:
        from ibeis.dev.all_imports import *
        gpath_list = grabdata.get_test_gpaths(ndata=3) + ['doesnotexist.jpg']
        params_list = list(preproc_image.add_images_params_gen(gpath_list))

    """
    #preproc_args = [(gpath, kwargs) for gpath in gpath_list]
    #print('[about to parse]: gpath_list=%r' % (gpath_list,))
    params_gen = utool.generate(parse_imageinfo, gpath_list, **kwargs)
    return params_gen
