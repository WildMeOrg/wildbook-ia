# LICENCE
from __future__ import absolute_import, division, print_function
# Python
from os.path import exists, join
from six.moves import zip, map
# Science
#import sys
#sys.exit(1)
import cv2
import numpy as np
from PIL import Image
from vtool import linalg
from vtool import geometry
import utool as ut
#from vtool.dummy import dummy_img  # NOQA
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[img]', DEBUG=False)


TAU = np.pi * 2


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


IMREAD_COLOR = cv2.IMREAD_COLOR if cv2.__version__[0] == '3' else cv2.CV_LOAD_IMAGE_COLOR


# References: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
#cv2.IMREAD_COLOR
#cv2.IMREAD_GRAYSCALE
#cv2.IMREAD_UNCHANGED


def imread(img_fpath, delete_if_corrupted=False, grayscale=False):
    r"""
    Args:
        img_fpath (?):
        delete_if_corrupted (bool):
        grayscale (bool):

    Returns:
        ndarray: imgBGR

    CommandLine:
        python -m vtool.image --test-imread


    References:
        http://docs.opencv.org/modules/core/doc/utility_and_system_functions_and_macros.html#error
        http://stackoverflow.com/questions/23572241/cv2-threshold-error-210

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> # build test data
        >>> img_fpath = '?'
        >>> delete_if_corrupted = False
        >>> grayscale = False
        >>> # execute function
        >>> imgBGR = imread(img_fpath, delete_if_corrupted, grayscale)
        >>> # verify results
        >>> result = str(imgBGR)
        >>> print(result)
    """
    try:
        if grayscale:
            imgBGR = cv2.imread(img_fpath, flags=cv2.IMREAD_GRAYSCALE)
        else:
            # opencv always reads in BGR mode (fastest load time?)
            imgBGR = cv2.imread(img_fpath, flags=IMREAD_COLOR)
    except cv2.error as cv2ex:
        ut.printex(cv2ex, iswarning=True)
        #print('cv2error dict = ' + ut.dict_str(cv2ex.__dict__))
        #print('cv2error dirlist = ' + ut.list_str(dir(cv2ex)))
        #print('cv2error args = ' + repr(cv2ex.args))
        #print('cv2error message = ' + repr(cv2ex.message))
        #cv2error args = ('c:/Users/joncrall/code/opencv/modules/core/src/alloc.cpp:52: error: (-4) Failed to allocate 22311168 bytes in function OutOfMemoryError\n',)
#cv2error message = 'c:/Users/joncrall/code/opencv/modules/core/src/alloc.cpp:52: error: (-4) Failed to allocate 22311168 bytes in function OutOfMemoryError\n'
        imgBGR = None
        #ismem_error = cv2ex.message.find('error: (-4)') > -1
        ismem_error = cv2ex.message.find('OutOfMemoryError') > -1
        if ismem_error:
            raise MemoryError('Memory Error while reading img_fpath=%s' % img_fpath)
    except Exception as ex:
        ut.printex(ex, iswarning=True)
        imgBGR = None
    if imgBGR is None:
        #if not exists(img_fpath):
        if not ut.checkpath(img_fpath, verbose=True):
            raise IOError('cannot read img_fpath=%s does not exist' % img_fpath)
        else:
            msg = 'cannot read img_fpath=%s seems corrupted or memory error.' % img_fpath
            print('[gtool] ' + msg)
            if delete_if_corrupted:
                print('[gtool] deleting corrupted image')
                ut.delete(img_fpath)
            raise IOError(msg)
    return imgBGR


def imwrite_fallback(img_fpath, imgBGR):
    try:
        import matplotlib.image as mpl_image
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        mpl_image.imsave(img_fpath, imgRGB)
        return None
    except Exception as ex:
        msg = '[gtool] FALLBACK ERROR writing: %s' % (img_fpath,)
        ut.printex(ex, msg, keys=['imgBGR.shape'])
        raise


def imwrite(img_fpath, imgBGR, fallback=False):
    """
    References:
        http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    """
    try:
        cv2.imwrite(img_fpath, imgBGR)
    except Exception as ex:
        if fallback:
            try:
                imwrite_fallback(img_fpath, imgBGR)
            except Exception as ex:
                pass
        msg = '[gtool] ERROR writing: %s' % (img_fpath,)
        ut.printex(ex, msg, keys=['imgBGR.shape'])
        raise


def get_size(img):
    """ Returns the image size in (width, height) """
    wh = img.shape[0:2][::-1]
    return wh


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
    References:
        stackoverflow.com/uestions/12729228/simple-efficient-binlinear-interpolation-of-images-in-numpy-and-python

    SeeAlso:
        cv2.getRectSubPix(image, patchSize, center[, patch[, patchType]])
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
    hasimg = ut.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, ut.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


def resize(img, dsize, interpolation=cv2.INTER_LANCZOS4):
    return cv2.resize(img, dsize, interpolation=interpolation)


def pad_image_on_disk(img_fpath, pad_, out_fpath=None, value=0, borderType=cv2.BORDER_CONSTANT, **kwargs):
    imgBGR = imread(img_fpath)
    imgBGR2 = cv2.copyMakeBorder(imgBGR, pad_, pad_, pad_, pad_, borderType=cv2.BORDER_CONSTANT, value=value)
    imgBGR2[:pad_, :] = value
    imgBGR2[-pad_:, :] = value
    imgBGR2[:, :pad_] = value
    imgBGR2[:, -pad_:] = value
    out_fpath_ = ut.augpath(img_fpath, '_pad=%r' % (pad_)) if out_fpath is None else out_fpath
    imwrite(out_fpath_, imgBGR2)
    return out_fpath_


def rotate_image_on_disk(img_fpath, theta, out_fpath=None, **kwargs):
    r"""
    Args:
        img_fpath (?):
        theta (?):
        out_fpath (None):

    CommandLine:
        python -m vtool.image --test-rotate_image_on_disk

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath('star.png')
        >>> theta = TAU * 3 / 8
        >>> # execute function
        >>> out_fpath = None
        >>> out_fpath_ = rotate_image_on_disk(img_fpath, theta, out_fpath)
        >>> print(out_fpath_)
        >>> if ut.get_argflag('--show') or ut.inIPython():
        >>>     import plottool as pt
        >>>     pt.imshow(out_fpath_,  pnum=(1, 1, 1))
        >>>     pt.show_if_requested()

    """
    img = imread(img_fpath)
    imgR = rotate_image(img, theta, **kwargs)
    out_fpath_ = ut.augpath(img_fpath, augsuf='_theta=%r' % (theta)) if out_fpath is None else out_fpath
    imwrite(out_fpath_, imgR)
    return out_fpath_


def rotate_image(img, theta, **kwargs):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        theta (?):

    CommandLine:
        python -m vtool.image --test-rotate_image

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> img = vt.get_test_patch('star2')
        >>> theta = TAU / 16.0
        >>> # execute function
        >>> imgR = rotate_image(img, theta)
        >>> if ut.get_argflag('--show') or ut.inIPython():
        >>>     import plottool as pt
        >>>     pt.imshow(img  * 255,  pnum=(1, 2, 1))
        >>>     pt.imshow(imgR * 255, pnum=(1, 2, 2))
        >>>     pt.show_if_requested()
    """
    from vtool import linalg as ltool
    dsize = [img.shape[1], img.shape[0]]
    bbox = [0, 0, img.shape[1], img.shape[0]]
    R = ltool.rotation_around_bbox_mat3x3(theta, bbox)
    warp_kwargs = CV2_WARP_KWARGS.copy()
    warp_kwargs.update(kwargs)
    imgR = cv2.warpAffine(img, R[0:2], tuple(dsize), **warp_kwargs)
    return imgR


def resize_image_by_scale(img, scale, interpolation=cv2.INTER_LANCZOS4):
    dsize, tonew_sf = get_round_scaled_dsize(get_size(img), scale)
    new_img = cv2.resize(img, dsize, interpolation=interpolation)
    return new_img


def get_round_scaled_dsize(dsize_old, scale):
    w, h = dsize_old
    dsize = int(round(w * scale)), int(round(h * scale))
    tonew_sf = dsize[0] / w, dsize[1] / h
    return dsize, tonew_sf


def resized_dims_and_ratio(img_size, max_dsize):
    max_width, max_height = max_dsize
    width, height = img_size
    ratio = min(max_width / width, max_height / height)
    dsize = (int(round(width * ratio)), int(round(height * ratio)))
    return dsize, ratio


def resized_clamped_thumb_dims(img_size, max_dsize):
    dsize_, ratio = resized_dims_and_ratio(img_size, max_dsize)
    dsize = img_size if ratio > 1 else dsize_
    sx = dsize[0] / img_size[0]
    sy = dsize[1] / img_size[1]
    return dsize, sx, sy


def padded_resize(img, target_size=(64, 64)):
    r"""
    makes the image resize to the target size and pads the rest of the area with a fill value

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        target_size (tuple):

    CommandLine:
        python -m vtool.image --test-padded_resize --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> import vtool as vt
        >>> imgA = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> imgB = vt.imread(ut.grab_test_imgpath('ada.jpg'))
        >>> imgC = vt.imread(ut.grab_test_imgpath('carl.jpg'), grayscale=True)
        >>> target_size = (64, 64)
        >>> img3_list = [padded_resize(img, target_size) for img in [imgA, imgB, imgC]]
        >>> # verify results
        >>> assert ut.list_allsame([vt.get_size(img3) for img3 in img3_list])
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pnum_ = pt.make_pnum_nextgen(1, 3)
        >>> pt.imshow(img3_list[0], pnum=pnum_())
        >>> pt.imshow(img3_list[1], pnum=pnum_())
        >>> pt.imshow(img3_list[2], pnum=pnum_())
        >>> ut.show_if_requested()
    """
    img2 = resize_thumb(img, target_size)
    dsize2 = get_size(img2)
    if dsize2 != target_size:
        target_shape = target_size[::-1] if get_num_channels(img2) == 1 else target_size[::-1] + (3,)
        rc_diff = (np.array(target_shape[0:2]) - np.array(img2.shape[0:2]))
        rc_start = np.floor(rc_diff / 2)
        rc_end  =  [None if e == 0 else e for e in (rc_start - rc_diff)]
        rc_slice = [slice(b, e) for (b, e) in zip(rc_start, rc_end)]
        img3 = np.zeros(target_shape, dtype=img2.dtype)
        img3[rc_slice[0], rc_slice[1]] = img2
    else:
        img3 = img2
    return img3


def resize_thumb(img, max_dsize=(64, 64)):
    """
    Resize an image such that its max width or height is:

    CommandLine:
        python -m vtool.image --test-resize_thumb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> max_dsize = (64, 64)
        >>> # execute function
        >>> img2 = resize_thumb(img, max_dsize)
        >>> print('img.shape = %r' % (img.shape,))
        >>> print('img2.shape = %r' % (img2.shape,))
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img2)
        >>> ut.show_if_requested()
    """
    height, width = img.shape[0:2]
    img_size = (width, height)
    dsize, ratio = resized_dims_and_ratio(img_size, max_dsize)
    if ratio > 1:
        return cvt_BGR2RGB(img)
    else:
        return resize(img, dsize)


def scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy):
    r"""
    Helps with drawing scaled bbounding boxes on thumbnails

    Args:
        bbox_list (list): bboxes in x,y,w,h format
        theta_list (list): rotation of bounding boxes
        sx (float): x scale factor
        sy (float): y scale factor

    Yeilds:
        new_verts - vertices of scaled bounding box for every input

    CommandLine:
        python -m vtool.image --test-scaled_verts_from_bbox_gen

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.image import *  # NOQA
        >>> # build test data
        >>> bbox_list = [(10, 10, 100, 100)]
        >>> theta_list = [0]
        >>> sx = .5
        >>> sy = .5
        >>> # execute function
        >>> new_verts_list = list(scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
        >>> result = str(new_verts_list)
        >>> # verify results
        >>> print(result)
        [[[5, 5], [55, 5], [55, 55], [5, 55], [5, 5]]]
    """
    # TODO: input verts support and better name
    for bbox, theta in zip(bbox_list, theta_list):
        new_verts = scaled_verts_from_bbox(bbox, theta, sx, sy)
        yield new_verts


def scaled_verts_from_bbox(bbox, theta, sx, sy):
    """ Helps with drawing scaled bbounding boxes on thumbnails """
    # Transformation matrixes
    R = linalg.rotation_around_bbox_mat3x3(theta, bbox)
    S = linalg.scale_mat3x3(sx, sy)
    # Get verticies of the annotation polygon
    verts = geometry.verts_from_bbox(bbox, close=True)
    # Rotate and transform to thumbnail space
    xyz_pts = linalg.add_homogenous_coordinate(np.array(verts).T)
    trans_pts = linalg.remove_homogenous_coordinate(S.dot(R).dot(xyz_pts))
    new_verts = np.round(trans_pts).astype(np.int32).T.tolist()
    return new_verts


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
    import vtool as vt
    (x1, y1, _w, _h) = xywh
    x2 = (x1 + _w)
    y2 = (y1 + _h)
    if round_:
        pt1 = (vt.iround(x1 * sx), vt.iround(y1 * sy))
        pt2 = (vt.iround(x2 * sx), vt.iround(y2 * sy))
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
            thumb_dpath = ut.get_app_resource_dir(appname, 'thumbs')
        ut.ensuredir(thumb_dpath)
        self.thumb_gpaths = [join(thumb_dpath, str(uuid) + 'thumb.png') for uuid in uuid_list]
        self.asrgb = asrgb
        self.thumb_size = thumb_size
        self.thumb_list = None
        self.dirty_list = None
        self.dirty_gpaths = None

    def __enter__(self):
        # These items need to be computed
        self.dirty_list = [not exists(gpath) for gpath in self.thumb_gpaths]
        self.dirty_gpaths = ut.filter_items(self.thumb_gpaths, self.dirty_list)
        #print('[gtool.thumb] len(dirty_gpaths): %r' % len(self.dirty_gpaths))
        self.needs_compute = len(self.dirty_gpaths) > 0
        return self

    def save_dirty_thumbs_from_images(self, img_list):
        """ Pass in any images marked by the context as dirty here """
        # Remove any non images
        isvalid_list = [img is not None for img in img_list]
        valid_images  = ut.filter_items(img_list, isvalid_list)
        valid_fpath = ut.filter_items(self.thumb_gpaths, isvalid_list)
        # Resize to thumbnails
        max_dsize = (self.thumb_size, self.thumb_size)
        valid_thumbs = [resize_thumb(img, max_dsize) for img in valid_images]
        # Write thumbs to disk
        for gpath, thumb in zip(valid_fpath, valid_thumbs):
            imwrite(gpath, thumb)

    def filter_dirty_items(self, list_):
        """ Returns only items marked by the context as dirty """
        return ut.filter_items(list_, self.dirty_list)

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('[gtool.thumb] Error while in thumbnail context')
            print('[gtool.thumb] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
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
    #if not exists(new_gfpath):
    #    return new_gfpath
    img = imread(gfpath)
    new_img = resize(img, new_size)
    imwrite(new_gfpath, new_img)
    return new_gfpath


def resize_imagelist_generator(gpath_list, new_gpath_list, newsize_list, **kwargs):
    """ Resizes images and yeilds results asynchronously  """
    # Compute and write detectimg in asychronous process
    kwargs['force_serial'] = kwargs.get('force_serial', True)
    kwargs['ordered']      = kwargs.get('ordered', True)
    arg_iter = zip(gpath_list, new_gpath_list, newsize_list)
    arg_list = list(arg_iter)
    return ut.util_parallel.generate(resize_worker, arg_list, **kwargs)


def resize_imagelist_to_sqrtarea(gpath_list, new_gpath_list=None,
                                 sqrt_area=800, output_dir=None,
                                 checkexists=True,
                                 **kwargs):
    """ Resizes images and yeilds results asynchronously  """
    from vtool.chip import get_scaled_sizes_with_area
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
        ut.ensuredir(output_dir)
        size_suffix_list = ['_' + repr(newsize).replace(' ', '') for newsize in newsize_list]
        new_gname_list = ut.append_suffixlist_to_namelist(gpath_list, size_suffix_list)
        new_gpath_list = [join(output_dir, gname) for gname in new_gname_list]
        new_gpath_list = list(map(ut.unixpath, new_gpath_list))
    assert len(new_gpath_list) == len(gpath_list), 'unequal len'
    assert len(newsize_list) == len(gpath_list), 'unequal len'
    # Evaluate generator
    if checkexists:
        exists_list = list(map(exists, new_gpath_list))
        gpath_list_ = ut.filterfalse_items(gpath_list, exists_list)
        new_gpath_list_ = ut.filterfalse_items(new_gpath_list, exists_list)
        newsize_list_ = ut.filterfalse_items(newsize_list, exists_list)
    else:
        gpath_list_ = gpath_list
        new_gpath_list_ = new_gpath_list
        newsize_list_ = newsize_list
    generator = resize_imagelist_generator(gpath_list_, new_gpath_list_,
                                           newsize_list_, **kwargs)
    for res in generator:
        pass
    #return [res for res in generator]
    return new_gpath_list


def find_pixel_value(img, pixel):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        pixel (?):

    CommandLine:
        python -m vtool.math --test-find_pixel_value

    References:
        http://stackoverflow.com/questions/21407815/get-column-row-index-from-numpy-array-that-meets-a-boolean-condition

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.math import *  # NOQA
        >>> # build test data
        >>> img = np.random.rand(10, 10, 3) + 1.0
        >>> pixel = np.array([0, 0, 0])
        >>> img[5, 5, :] = pixel
        >>> img[2, 3, :] = pixel
        >>> img[1, 1, :] = pixel
        >>> img[0, 0, :] = pixel
        >>> img[2, 0, :] = pixel
        >>> # execute function
        >>> result = find_pixel_value(img, pixel)
        >>> # verify results
        >>> print(result)
        [[0 0]
         [1 1]
         [2 0]
         [2 3]
         [5 5]]
    """
    mask2d = np.all(img == pixel[None, None, :], axis=2)
    pixel_locs = np.column_stack(np.where(mask2d))
    return pixel_locs


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.image
        python -m vtool.image --allexamples
        python -m vtool.image --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
