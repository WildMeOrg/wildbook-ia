# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from os.path import exists, join
from itertools import izip
# Science
import cv2
import numpy as np
from PIL import Image
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
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
EXIF_TAG_DATETIME = 'DateTimeOriginal'


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


def get_gpathlist_sizes(gpath_list):
    """ reads the size of each image in gpath_list """
    gsize_list = [open_image_size(gpath) for gpath in gpath_list]
    return gsize_list


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
    hasimg = utool.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, utool.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


def resize(img, dsize):
    return cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)


def resize_thumb(img, max_dsize=(64, 64)):
    """ Resize an image such that its max width or height is: """
    max_width, max_height = max_dsize
    height, width = img.shape[0:2]
    ratio = min(max_width / width, max_height / height)
    if ratio > 1:
        return cvt_BGR2RGB(img)
    else:
        dsize = (int(round(width * ratio)), int(round(height * ratio)))
        return resize(img, dsize)


def _trimread(gpath):
    """ Try an imread """
    try:
        return imread(gpath)
    except Exception:
        return None


def get_scale_factor(src_img, dst_img):
    """ returns scale factor from one image to the next """
    src_h, src_w = src_img.shape[0:2]
    dst_h, dst_w = dst_img.shape[0:2]
    sx = dst_w / src_w
    sy = dst_h / src_h
    return (sx, sy)


def cvt_bbox_xywh_to_pt1pt2(xywh, sx=1.0, sy=1.0, round_=True):
    """ Converts bbox to thumb format with a scale factor"""
    (x1, y1, _w, _h) = xywh
    x2 = (x1 + _w)
    y2 = (y1 + _h)
    if round_:
        pt1 = (utool.iround(x1 * sx), utool.iround(y1 * sy))
        pt2 = (utool.iround(x2 * sx), utool.iround(y2 * sy))
    else:
        pt1 = ((x1 * sx), (y1 * sy))
        pt2 = ((x2 * sx), (y2 * sy))
    return (pt1, pt2)


class ThumbnailCacheContext(object):
    """ Lazy computation of of images as thumbnails.

    Just pass a list of uuids corresponding to the images. Then compute images
    flagged as dirty and give them back to the context.  thumbs_list will be
    populated on contex exit
    """
    def __init__(self, uuid_list, asrgb=True, thumb_size=64, thumb_dpath=None, appname='vtool'):
        if thumb_dpath is None:
            # Get default thumb path
            thumb_dpath = utool.get_app_resource_dir(appname, 'thumbs')
        utool.ensuredir(thumb_dpath)
        self.thumb_gpaths = [join(thumb_dpath, str(uuid) + 'thumb.png') for uuid in uuid_list]
        self.asrgb = asrgb
        self.thumb_size = thumb_size
        self.thumb_list = None
        self.dirty_list = None
        self.dirty_gpaths = None

    def __enter__(self):
        # These items need to be computed
        self.dirty_list = [not exists(gpath) for gpath in self.thumb_gpaths]
        self.dirty_gpaths = utool.filter_items(self.thumb_gpaths, self.dirty_list)
        #print('[gtool.thumb] len(dirty_gpaths): %r' % len(self.dirty_gpaths))
        self.needs_compute = len(self.dirty_gpaths) > 0
        return self

    def save_dirty_thumbs_from_images(self, img_list):
        """ Pass in any images marked by the context as dirty here """
        # Remove any non images
        isvalid_list = [img is not None for img in img_list]
        valid_images  = utool.filter_items(img_list, isvalid_list)
        valid_fpath = utool.filter_items(self.thumb_gpaths, isvalid_list)
        # Resize to thumbnails
        max_dsize = (self.thumb_size, self.thumb_size)
        valid_thumbs = [resize_thumb(img, max_dsize) for img in valid_images]
        # Write thumbs to disk
        for gpath, thumb in izip(valid_fpath, valid_thumbs):
            imwrite(gpath, thumb)

    def filter_dirty_items(self, list_):
        """ Returns only items marked by the context as dirty """
        return utool.filter_items(list_, self.dirty_list)

    def __exit__(self, exc_type, exc_value, traceback):
        if traceback is not None:
            print('[gtool.thumb] Error while in thumbnail context')
            return
        # Try to read thumbnails on disk
        self.thumb_list = [_trimread(gpath) for gpath in self.thumb_gpaths]
        if self.asrgb:
            self.thumb_list = [None if thumb is None else cvt_BGR2RGB(thumb)
                               for thumb in self.thumb_list]


# Parallel code for resizing many images
def resize_worker(tup):
    """ worker function for parallel generator """
    gfpath, new_gfpath, new_size = tup
    #print('[preproc] writing detectimg: %r' % new_gfpath)
    img = imread(gfpath)
    new_img = resize(img, new_size)
    imwrite(new_gfpath, new_img)
    return new_gfpath


def resize_imagelist_generator(gpath_list, new_gpath_list, newsize_list):
    """ Resizes images and yeilds results asynchronously  """
    # Compute and write detectimg in asychronous process
    arg_iter = izip(gpath_list, new_gpath_list, newsize_list)
    arg_list = list(arg_iter)
    return utool.util_parallel.generate(resize_worker, arg_list)


def resize_imagelist_to_sqrtarea(gpath_list, new_gpath_list=None,
                                 sqrt_area=800, output_dir=None):
    """ Resizes images and yeilds results asynchronously  """
    from .chip import get_scaled_sizes_with_area
    target_area = sqrt_area ** 2
    # Read image sizes
    gsize_list = get_gpathlist_sizes(gpath_list)
    # Compute new sizes which preserve aspect ratio
    newsize_list = get_scaled_sizes_with_area(target_area, gsize_list)
    if new_gpath_list is None:
        # Compute names for the new images if not given
        if output_dir is None:
            # Create an output directory if not specified
            output_dir      = 'resized_sqrtarea%r' % sqrt_area
            utool.ensuredir(output_dir)
        #basepath_list   = utool.get_basepath_list(gpath_list)
        gnamenoext_list  = utool.get_basename_noext_list(gpath_list)
        ext_list         = utool.get_ext_list(gpath_list)
        size_suffix_list = ['_' + repr(newsize).replace(' ', '')
                            for newsize in newsize_list]
        new_gname_list   = [gname + suffix + ext for gname, suffix, ext in
                            izip(gnamenoext_list, size_suffix_list, ext_list)]
        new_gpath_list   = [join(output_dir, gname) for gname in new_gname_list]
        new_gpath_list   = map(utool.unixpath, new_gpath_list)
    assert len(new_gpath_list) == len(gpath_list), 'unequal len'
    assert len(newsize_list) == len(gpath_list), 'unequal len'
    # Evaluate generator
    generator = resize_imagelist_generator(gpath_list, new_gpath_list, newsize_list)
    return [res for res in generator]
