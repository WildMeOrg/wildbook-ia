from __future__ import absolute_import, division, print_function
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)
# VTool
import vtool.image as gtool
import vtool.exif as exif


GPSInfo_TAGID          = exif.EXIF_TAG_TO_TAGID['GPSInfo']
DateTimeOriginal_TAGID = exif.EXIF_TAG_TO_TAGID['DateTimeOriginal']


def get_exif(pil_img):
    """ Image EXIF helper """
    exif_dict = exif.get_exif_dict(pil_img)
    exiftime = exif_dict.get(DateTimeOriginal_TAGID, -1)
    # TODO: Fixme
    #latlon = exif_dict.get(GPSInfo_TAGID, (-1, -1))
    latlon = (-1, -1)
    time = utool.util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    lat, lon = latlon
    return time, lat, lon


def add_images_paramters_gen(gpath_list):
    """ generates values for add_images sqlcommands """
    mark_prog, end_prog = utool.progress_func(len(gpath_list), lbl='imgs: ')
    for count, gpath in enumerate(gpath_list):
        mark_prog(count)
        pil_img = gtool.open_pil_image(gpath)      # Open PIL Image
        width, height  = pil_img.size              # Read width, height
        time, lat, lon = get_exif(pil_img)         # Read exif tags
        gid = utool.util_hash.image_uuid(pil_img)  # Read pixels ]-hash-> guid = gid
        yield (gid, gpath, width, height, time, lat, lon)
    end_prog()
