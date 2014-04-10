from __future__ import absolute_import, division, print_function
# UTool
import utool
# VTool
import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)


tag_list = ['DateTimeOriginal', 'GPSInfo']
_TAGKEYS = gtool.get_exif_tagids(tag_list)
_TAGDEFAULTS = (-1, (-1, -1))


def get_exif(pil_img):
    """ Image EXIF helper """
    (exiftime, (lat, lon)) = gtool.read_exif_tags(pil_img, _TAGKEYS, _TAGDEFAULTS)
    time = utool.util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    return time, lat, lon


def add_images_paramters_gen(gpath_list):
    """ generates values for add_images sqlcommands """
    for gpath in gpath_list:
        pil_img = gtool.open_pil_image(gpath)      # Open PIL Image
        width, height  = pil_img.size              # Read width, height
        time, lat, lon = get_exif(pil_img)         # Read exif tags
        gid = utool.util_hash.image_uuid(pil_img)  # Read pixels ]-hash-> guid = gid
        yield (gid, gpath, width, height, time, lat, lon)
