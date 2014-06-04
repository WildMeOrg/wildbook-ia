# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from os.path import exists
# Science
import cv2
import numpy as np
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


def dummy_img(w, h):
    """ Creates a dummy test image """
    img = np.zeros((h, w), dtype=np.uint8) + 200
    return img


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
    """ Returns the number of color channels """
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


@profile
def subpixel_values(img, pts):
    """
    adapted from
    stackoverflow.com/uestions/12729228/simple-efficient-binlinear-
    interpolation-of-images-in-numpy-and-python
    """
    # Image info
    nChannels = get_num_channels(img)
    height, width = img.shape[0:2]
    # Subpixel locations to sample
    ptsT = pts.T
    x = ptsT[0]
    y = ptsT[1]
    # Get quantized pixel locations near subpixel pts
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    # Make sure the values do not go past the boundary
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    # Find bilinear weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    if  nChannels != 1:
        wa = np.array([wa] *  nChannels).T
        wb = np.array([wb] *  nChannels).T
        wc = np.array([wc] *  nChannels).T
        wd = np.array([wd] *  nChannels).T
    # Sample values
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    # Perform the bilinear interpolation
    subpxl_vals = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    return subpxl_vals


def open_pil_image(image_fpath):
    pil_img = Image.open(image_fpath)
    return pil_img


def open_image_size(image_fpath):
    pil_img = Image.open(image_fpath)
    size = pil_img.size
    return size


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def cvt_BGR2RGB(imgBGR):
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    return imgRGB


@profile
def warpAffine(img, Aff, dsize):
    """
    dsize = (width, height) of return image
    """
    warped_img = cv2.warpAffine(img, Aff[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img


@profile
def warpHomog(img, Homog, dsize):
    """
    dsize = (width, height) of return image
    """
    warped_img = cv2.warpPerspective(img, Homog, tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img


def blend_images(img1, img2):
    assert img1.shape == img2.shape, 'chips must be same shape to blend'
    chip_blend = np.zeros(img2.shape, dtype=img2.dtype)
    chip_blend = img1 / 2 + img2 / 2
    return chip_blend


def print_image_checks(img_fpath):
    hasimg = util_path.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, util_str.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


def resize(img, dsize):
    return cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)


def resize_thumb(img, max_dsize=(64, 64)):
    max_width, max_height = max_dsize
    height, width = img.shape[0:2]
    ratio = min(max_width / width, max_height / height)
    if ratio > 1:
        return cvt_BGR2RGB(img)
    else:
        dsize = (int(round(width * ratio)), int(round(height * ratio)))
        return cvt_BGR2RGB(resize(img, dsize))
