# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from os.path import exists
# Science
import cv2
from PIL import Image
from utool import util_path, util_str
#from utool import util_progress
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(
    __name__, '[img]', DEBUG=False)


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


def get_size(img):
    """ Returns the image size in (width, height) """
    (h, w) = img.shape[0:2]
    return (w, h)


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


def print_image_checks(img_fpath):
    hasimg = util_path.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, util_str.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg
