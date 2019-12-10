# -*- coding: utf-8 -*-
# LICENCE
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import os
from os.path import exists, join  # NOQA
from os.path import splitext
from six.moves import zip, map, range  # NOQA
import numpy as np
from PIL import Image
import cv2
from .util_math import TAU
from vtool_ibeis import exif
import utool as ut
import ubelt as ub


if cv2 is not None:

    CV2_INTERPOLATION_TYPES = {
        'nearest': cv2.INTER_NEAREST,
        'linear':  cv2.INTER_LINEAR,
        'area':    cv2.INTER_AREA,
        'cubic':   cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    CV2_BORDER_TYPES = {
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
        'reflect':    cv2.BORDER_REFLECT,
        'wrap': cv2.BORDER_WRAP,
        'reflect101': cv2.BORDER_REFLECT101,
        'tranparent':   cv2.BORDER_TRANSPARENT,
        'isolated':  cv2.BORDER_ISOLATED,
    }

    CV2_WARP_KWARGS = {
        'flags': CV2_INTERPOLATION_TYPES['lanczos'],
        'borderMode': cv2.BORDER_CONSTANT
    }

    try:
        IMREAD_COLOR = cv2.IMREAD_COLOR
    except AttributeError:
        from distutils.version import LooseVersion
        cv2_version = LooseVersion(cv2.__version__)
        print('UNKNOWN cv2_version = {!r}'.format(cv2_version))
        assert cv2_version.version[0] <= 2
        IMREAD_COLOR = cv2.CV_LOAD_IMAGE_COLOR
else:
    # Hacks
    cv2 = ut.DynStruct()
    cv2.BORDER_CONSTANT = None
    cv2.INTER_LANCZOS4 = None
    cv2.CV_AA = None
    cv2.FONT_HERSHEY_SIMPLEX = None
    cv2.INTER_NEAREST = None


try:
    LINE_AA = cv2.LINE_AA
except AttributeError:
    LINE_AA = cv2.CV_AA

#cv2.BORDER_CONSTANT     cv2.BORDER_REFLECT      cv2.BORDER_REPLICATE
#cv2.BORDER_DEFAULT      cv2.BORDER_REFLECT101   cv2.BORDER_TRANSPARENT
#cv2.BORDER_ISOLATED     cv2.BORDER_REFLECT_101  cv2.BORDER_WRAP


EXIF_TAG_GPS      = 'GPSInfo'
EXIF_TAG_DATETIME = 'DateTimeOriginal'


# References:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
#cv2.IMREAD_COLOR
#cv2.IMREAD_GRAYSCALE
#cv2.IMREAD_UNCHANGED

def _rectify_border_mode(border_mode, default=cv2.BORDER_CONSTANT):
    """ Converts argument to cv2 style """
    if border_mode is None:
        return default
    elif isinstance(border_mode, six.text_type):
        return CV2_BORDER_TYPES[border_mode]
    else:
        return border_mode


def _rectify_interpolation(interp, default=cv2.INTER_LANCZOS4):
    """
    Converts interpolation into flags suitable cv2 functions

    Args:
        interp (int or str): string or cv2-style interpolation type
        default (int): cv2 interpolation flag to use if `interp` is None

    Returns:
        int: flag specifying interpolation type that can be passed to
           functions like cv2.resize, cv2.warpAffine, etc...
    """
    if interp is None:
        return default
    elif isinstance(interp, six.text_type):
        try:
            return CV2_INTERPOLATION_TYPES[interp]
        except KeyError:
            print('Valid values for interpolation are {}'.format(
                list(CV2_INTERPOLATION_TYPES.keys())))
            raise
    else:
        return interp


def montage(img_list, dsize, rng=np.random, method='random', return_debug=False):
    """
    Creates a montage / collage from a set of images

    CommandLine:
        python -m vtool_ibeis.image --exec-montage:0 --show
        python -m vtool_ibeis.image --exec-montage:1

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_list0 = testdata_imglist()
        >>> img_list1 = [resize_to_maxdims(img, (256, 256)) for img in img_list0]
        >>> num = 4
        >>> img_list = list(ub.flatten([img_list1] * num))
        >>> dsize = (700, 700)
        >>> rng = np.random.RandomState(42)
        >>> method = 'unused'
        >>> #method = 'random'
        >>> dst, debug_info = montage(img_list, dsize, rng, method=method,
        >>>                           return_debug=True)
        >>> place_img = debug_info.get('place_img_', np.ones((2, 2)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(dst, pnum=(1, 2, 1))
        >>> pt.imshow(place_img / place_img.max(), pnum=(1, 2, 2))
        >>> ut.show_if_requested()

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> import ibeis
        >>> import random
        >>> from os.path import join, expanduser, abspath
        >>> from vtool_ibeis.image import *  # NOQA
        >>> ibs = ibeis.opendb('GZC')
        >>> gid_list0 = ibs.get_valid_gids()
        >>> img_list = []
        >>> for i in range(6000):
        >>>     print(i)
        >>>     try:
        >>>         gid = random.choice(gid_list0)
        >>>         image = ibs.get_images(gid)
        >>>         image = resize_to_maxdims(image, (512, 512))
        >>>         img_list.append(image)
        >>>     except Exception:
        >>>         pass
        >>> dsize = (19200, 10800)
        >>> rng = np.random.RandomState(42)
        >>> dst = montage(img_list, dsize, rng)
        >>> filepath = abspath(expanduser(join('~', 'Desktop', 'montage.jpg')))
        >>> print('Writing to: %r' % (filepath, ))
        >>> imwrite(filepath, dst)
    """
    import vtool_ibeis as vt
    channels = 3
    shape = tuple(dsize[::-1]) + (channels,)
    dst = np.zeros(shape, dtype=np.uint8)

    use_placement_prob = method == 'unused'

    if use_placement_prob:
        # TODO: place images in places that other images have not been placed yet
        place_img = np.ones(shape[0:2], dtype=np.float)
        #place_img[
        #place_img = vt.gaussian_patch(shape[0:2], np.array(shape[0:2]) * .1)
        #place_img = vt.gaussian_patch(shape[0:2], np.array(shape[0:2]) * .3)
        temp_img = np.ones(shape[0:2], dtype=np.float)
        # Enumerate valid 2d locations
        xy_locs_ = np.meshgrid(np.arange(place_img.shape[1]),
                               np.arange(place_img.shape[0]))
        xy_locs = np.vstack((xy_locs_[0].flatten(), xy_locs_[1].flatten())).T

    for img in img_list:
        #np.ones(img.shape, dtype=np.uint8) * 255
        #vt.warp_patch_onto_kpts()
        w, h = vt.get_size(img)
        qw = (w / 3.0)
        qh = (h / 3.0)
        tx_pdf = (-qw, dsize[0] + qw)
        ty_pdf = (-qh, dsize[1] + qh)
        #txy_pdf = []

        if use_placement_prob:
            # Enumerate probability of valid 2d locations
            # Renomralize place image
            np.divide(place_img, place_img.sum(), out=place_img)
            #import utool
            #utool.embed()
            if True:
                window_frac = .125
                w = h = int(round(min(dst.shape[0:2]) * window_frac))
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (w, h))
                place_img_ = place_img
                place_img_ = cv2.erode(place_img_, element)
                np.divide(place_img_, place_img_.sum(), out=place_img_)
                xy_prob = place_img_.flatten()
            else:
                xy_prob = place_img.flatten()
            xy_loc_idxs = np.arange(len(xy_locs))
            #np.ones(img.shape, dtype=np.uint8) * 255
            # overwrite tx_pdf with a better value
            idx = rng.choice(xy_loc_idxs, size=1, replace=True, p=xy_prob)
            tx, ty = xy_locs[idx][0]
            tx_pdf = (tx, tx)
            ty_pdf = (ty, ty)

        Aff = vt.random_affine_transform(
            tx_pdf=tx_pdf, ty_pdf=ty_pdf,
            theta_pdf=(-TAU / 32, TAU / 32),
            rng=rng)

        cv2.warpAffine(img, Aff[0:2], dsize, dst=dst,
                       flags=cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_TRANSPARENT)

        if use_placement_prob:
            # Denote that an image has been placed here.
            patch = vt.gaussian_patch(img.shape[0:2], np.array(img.shape[0:2]) * .3)
            #np.add(patch, min(0, patch.min()), out=patch)
            np.divide(patch, patch.max(), out=patch)
            np.subtract(1, patch, out=patch)
            #patch[:] = 0
            # Reset temp image
            temp_img[:] = 1
            # Align patch with placement image
            cv2.warpAffine(patch, Aff[0:2], dsize, dst=temp_img,
                           flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_TRANSPARENT)
            # Renormalize image
            #np.add(temp_img, min(0, temp_img.min()), out=temp_img)
            #np.divide(temp_img, temp_img.max(), out=temp_img)
            np.clip(temp_img, 0, 1, out=temp_img)
            # Blend with placement probability image
            np.multiply(temp_img, place_img, out=place_img)

        #(255 - get_pixel_dist(dst, 0)) / 255
    if return_debug:
        debug_info = {}
        if use_placement_prob:
            debug_info['place_img'] = place_img
            debug_info['place_img_'] = place_img_
        return dst, debug_info

    return dst


def imread(img_fpath, grayscale=False, orient=False, flags=None,
           force_pil=None, delete_if_corrupted=False, **kwargs):
    r"""
    Wrapper around the opencv imread function. Handles remote uris.

    Args:
        img_fpath (str):  file path string
        grayscale (bool): (default = False)
        orient (bool): (default = False)
        flags (None): opencv flags (default = None)
        force_pil (bool): (default = None)
        delete_if_corrupted (bool): (default = False)

    Returns:
        ndarray: imgBGR

    CommandLine:
        python -m vtool_ibeis.image --test-imread
        python -m vtool_ibeis.image --test-imread:1
        python -m vtool_ibeis.image --test-imread:2

    References:
        http://docs.opencv.org/modules/core/doc/utility_and_system_functions_and_macros.html#error
        http://stackoverflow.com/questions/23572241/cv2-threshold-error-210

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> imgBGR1 = imread(img_fpath, grayscale=False)
        >>> imgBGR2 = imread(img_fpath, grayscale=True)
        >>> imgBGR3 = imread(img_fpath, orient=True)
        >>> assert imgBGR1.shape == (250, 300, 3)
        >>> assert imgBGR2.shape == (250, 300)
        >>> assert np.all(imgBGR1 == imgBGR3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgBGR1, pnum=(2, 2, 1))
        >>> pt.imshow(imgBGR2, pnum=(2, 2, 2))
        >>> pt.imshow(imgBGR3, pnum=(2, 2, 3))
        >>> ut.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_url = 'http://images.summitpost.org/original/769474.JPG'
        >>> img_fpath = ut.grab_file_url(img_url)
        >>> imgBGR1 = imread(img_url)
        >>> imgBGR2 = imread(img_fpath)
        >>> #imgBGR2 = imread(img_fpath, force_pil=False, flags=cv2.IMREAD_UNCHANGED)
        >>> print('imgBGR.shape = %r' % (imgBGR1.shape,))
        >>> print('imgBGR2.shape = %r' % (imgBGR2.shape,))
        >>> result = str(imgBGR1.shape)
        >>> diff_pxls = imgBGR1 != imgBGR2
        >>> num_diff_pxls = diff_pxls.sum()
        >>> print(result)
        >>> print('num_diff_pxls=%r/%r' % (num_diff_pxls, diff_pxls.size))
        >>> assert num_diff_pxls == 0
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> diffMag = np.linalg.norm(imgBGR2 / 255. - imgBGR1 / 255., axis=2)
        >>> pt.imshow(imgBGR1, pnum=(1, 3, 1))
        >>> pt.imshow(diffMag / diffMag.max(), pnum=(1, 3, 2))
        >>> pt.imshow(imgBGR2, pnum=(1, 3, 3))
        >>> ut.show_if_requested()
        (2736, 3648, 3)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> url = 'http://www.sherv.net/cm/emo/funny/2/big-dancing-banana-smiley-emoticon.gif'
        >>> img_fpath = ut.grab_file_url(url)
        >>> delete_if_corrupted = False
        >>> grayscale = False
        >>> imgBGR = imread(img_fpath, grayscale=grayscale)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgBGR)
        >>> ut.show_if_requested()
    """
    path, ext = splitext(img_fpath)
    orient_ = 'auto' if orient in ['auto', 'on', True] else False
    use_pil = (orient_ or ext.lower() == '.gif' or force_pil is True)
    if img_fpath.startswith('http://') or img_fpath.startswith('https://'):
        imgBGR = imread_remote_url(img_fpath, grayscale=grayscale, orient=orient, use_pil=use_pil, flags=flags)
    elif img_fpath.startswith('s3://'):
        imgBGR = imread_remote_s3(img_fpath, grayscale=grayscale, orient=orient, use_pil=use_pil, flags=flags)
    else:
        try:
            if use_pil:
                # If we want to open with auto orient, only open once with PIL
                # Otherwise, open with OpenCV (faster) and reorient if given
                # the known orientation of the image
                #pil_img = Image.open(img_fpath)
                #print("USE PIL")
                with Image.open(img_fpath) as pil_img:
                    imgBGR = _fix_orient_pil_img(pil_img, grayscale=grayscale,
                                                 orient=orient_, **kwargs)
                #with Image.open(img_fpath) as pil_img: # breaks?
                #pil_img.close()  # breaks?
            else:
                #print("USE OPENCV")
                if flags is None:
                    flags = cv2.IMREAD_GRAYSCALE if grayscale else IMREAD_COLOR
                # TODO cv2.IMREAD_UNCHANGED
                imgBGR = cv2.imread(img_fpath, flags=flags)

        except cv2.error as cv2ex:
            ut.printex(cv2ex, 'opencv error', iswarning=True)
            #print('cv2error dict = ' + ub.repr2(cv2ex.__dict__))
            #print('cv2error dirlist = ' + ub.repr2(dir(cv2ex)))
            #print('cv2error args = ' + repr(cv2ex.args))
            #print('cv2error message = ' + repr(cv2ex.message))
            #cv2error args =
            #('c:/Users/joncrall/code/opencv/modules/core/src/alloc.cpp:52: error:
            # (-4) Failed to allocate 22311168 bytes in function
            # OutOfMemoryError\n',)
            #cv2error message =
            #'c:/Users/joncrall/code/opencv/modules/core/src/alloc.cpp:52: error:
            #(-4) #Failed to allocate 22311168 bytes in function
            #OutOfMemoryError\n'
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
                raise IOError('cannot read img_fpath=%s does not exist.' % img_fpath)
            else:
                if not os.access(img_fpath, os.R_OK):
                    raise PermissionError(
                        'cannot read img_fpath={} access denied.'.format(img_fpath))
                if delete_if_corrupted:
                    # Probably should depricate this. A bit out of scope
                    msg = (
                        'Cannot read corrupted img_fpath=%s, requires deletion.'
                        % img_fpath
                    )
                    print('[vt.imread] deleting corrupted image')
                    ut.delete(img_fpath)
                else:
                    msg = (
                        'Cannot read img_fpath=%s, '
                        'seems corrupted or memory error.'
                        % img_fpath
                    )
                print('[vt.imread] ' + msg)
                raise IOError(msg)
        if not isinstance(orient, bool) and orient in exif.ORIENTATION_DICT:
            if False:
                print('[vt.imread] Applying orientation %r' % (orient, ))
            imgBGR = _fix_orientation(imgBGR, orient)
    return imgBGR


def imread_remote_s3(img_fpath, **kwargs):
    import io
    try:
        s3_dict = ut.s3_str_decode_to_dict(img_fpath)
        contents = ut.read_s3_contents(**s3_dict)
        # btyedata = np.asarray(bytearray(contents), dtype=np.uint8)
        # imgBGR = cv2.imdecode(btyedata, -1)
        with io.BytesIO(contents) as image_stream:
            imgBGR = _imread_bytesio(image_stream, **kwargs)
            #with Image.open(image_stream) as pil_img:
            #    imgBGR = _fix_orient_pil_img(pil_img, **kwargs)
    except AttributeError:
        pass
    return imgBGR


def imread_remote_url(img_url, **kwargs):
    from six.moves import urllib
    import io
    print("USE PIL REMOTE")
    addinfourl = urllib.request.urlopen(img_url)
    try:
        # image_file = io.BytesIO(addinfourl.read())
        # pil_img =  Image.open(image_file)
        with io.BytesIO(addinfourl.read()) as image_stream:
            imgBGR = _imread_bytesio(image_stream, **kwargs)
            #nparr = np.fromstring(image_stream.getvalue(), np.uint8)
            #imgBGR = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
            #imgBGR = cv2.imread(image_file)
            #with Image.open(image_file) as pil_img:
            #    imgBGR = _fix_orient_pil_img(pil_img, **kwargs)
    except IOError:
        pass
    finally:
        addinfourl.close()
    return imgBGR


def _imread_bytesio(image_stream, use_pil=False, flags=None, **kwargs):
    if use_pil:
        with Image.open(image_stream) as pil_img:
            imgBGR = _fix_orient_pil_img(pil_img, **kwargs)
    else:
        if flags is None:
            grayscale = kwargs.get('grayscale', False)
            flags = cv2.IMREAD_GRAYSCALE if grayscale else IMREAD_COLOR
        nparr = np.fromstring(image_stream.getvalue(), np.uint8)
        imgBGR = cv2.imdecode(nparr, flags=flags)  # cv2.IMREAD_COLOR in OpenCV 3.1
    return imgBGR


def _fix_orient_pil_img(pil_img, grayscale=False, orient=False, **kwargs):
    if orient == 'auto':
        exif_dict = exif.get_exif_dict(pil_img)
        orient = exif.get_orientation(exif_dict)
    #if pil_img.format in ['MPO', 'GIF']:
    np_img = np.array(pil_img.convert('RGB'))
    #else:
    #    np_img = np.array(pil_img)
    if grayscale:
        imgBGR = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        imgBGR = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    if not isinstance(orient, bool) and orient in exif.ORIENTATION_DICT:
        imgBGR = _fix_orientation(imgBGR, orient, **kwargs)
    return imgBGR


def _fix_orientation(imgBGR, orient, fallback=True):
    assert not isinstance(orient, bool) and orient in exif.ORIENTATION_DICT
    orient_ = exif.ORIENTATION_DICT[orient]
    if orient_ == exif.ORIENTATION_000:
        return imgBGR

    dsize_h, dsize_w = imgBGR.shape[:2]
    if orient_ in [exif.ORIENTATION_090, exif.ORIENTATION_270]:
        dsize = (dsize_h, dsize_w, )
    else:
        dsize = (dsize_w, dsize_h, )

    # FIXME; rotation changes the shape of the images
    # rotate_image does not do this, it must incorrectly clip areas.
    # TODO 90 degree optimizations
    if orient_ == exif.ORIENTATION_090:
        #return np.rot90(imgBGR, k=1)
        return rotate_image(imgBGR, TAU * 0.25, dsize=dsize)
    elif orient_ == exif.ORIENTATION_180:
        #return np.rot90(imgBGR, k=2)
        return rotate_image(imgBGR, TAU * 0.50, dsize=dsize)
    elif orient_ == exif.ORIENTATION_270:
        #return np.rot90(imgBGR, k=3)
        return rotate_image(imgBGR, TAU * 0.75, dsize=dsize)
    elif fallback:
        return imgBGR
    else:
        raise IOError('Could not fix the image orientation')


def imwrite(img_fpath, imgBGR, fallback=False):
    """
    References:
        http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html

    Args:
        img_fpath (str):  file path string
        imgBGR (ndarray[uint8_t, ndim=2]):  image data in opencv format (blue, green, red)
        fallback (bool): (default = False)

    CommandLine:
        python -m vtool_ibeis.image --exec-imwrite

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> import utool as ut
        >>> img_fpath1 = ut.grab_test_imgpath('zebra.png')
        >>> imgBGR = vt.imread(img_fpath1)
        >>> img_dpath = ub.ensure_app_cache_dir('vtool_ibeis', 'testwrite')
        >>> img_fpath2 = ut.unixjoin(img_dpath, 'zebra.png')
        >>> fallback = False
        >>> imwrite(img_fpath2, imgBGR, fallback=fallback)
        >>> imgBGR2 = vt.imread(img_fpath2)
        >>> assert np.all(imgBGR2 == imgBGR)
    """
    try:
        cv2.imwrite(img_fpath, imgBGR)
    except Exception as ex:
        if fallback:
            try:
                imwrite_fallback(img_fpath, imgBGR)
            except Exception as ex:
                pass
        msg = '[vt.image] ERROR writing: %s' % (img_fpath,)
        ut.printex(ex, msg, keys=['imgBGR.shape'])
        raise


def imwrite_fallback(img_fpath, imgBGR):
    try:
        import matplotlib.image as mpl_image
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        mpl_image.imsave(img_fpath, imgRGB)
        return None
    except Exception as ex:
        msg = '[vt.image] FALLBACK ERROR writing: %s' % (img_fpath,)
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
    elif ndims == 3 and img.shape[2] == 4:
        nChannels = 4
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return nChannels


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


def open_image_size(image_fpath):
    """
    Gets image size from an image on disk

    Args:
        image_fpath (str):

    Returns:
        tuple: size (width, height)

    CommandLine:
        python -m vtool_ibeis.image --test-open_image_size

    Doctest:
        >>> from vtool_ibeis.image import *  # NOQA
        >>> image_fpath = ut.grab_test_imgpath('patsy.jpg')
        >>> size = open_image_size(image_fpath)
        >>> result = ('size = %s' % (str(size),))
        >>> print(result)
        size = (800, 441)

    Ignore:
        # Confirm that Image.open is a lazy load
        import vtool_ibeis as vt
        import utool as ut
        import timeit
        setup = ut.codeblock(
            '''
            from PIL import Image
            import utool as ut
            import vtool_ibeis as vt
            image_fpath = ut.grab_test_imgpath('patsy.jpg')
            '''
        )
        t1 = timeit.timeit('Image.open(image_fpath)', setup, number=100)
        t2 = timeit.timeit('Image.open(image_fpath).size', setup, number=100)
        t3 = timeit.timeit('vt.open_image_size(image_fpath)', setup, number=100)
        t4 = timeit.timeit('vt.imread(image_fpath).shape', setup, number=100)
        t5 = timeit.timeit('Image.open(image_fpath).getdata()', setup, number=100)
        print('t1 = %r' % (t1,))
        print('t2 = %r' % (t2,))
        print('t3 = %r' % (t3,))
        print('t4 = %r' % (t4,))
        print('t5 = %r' % (t5,))
        assert t2 < t5
        assert t3 < t4
    """
    try:
        pil_img = Image.open(image_fpath)
        size = pil_img.size
    except IOError as ex:
        print('ERROR: Failed open image size')
        ut.checkpath(image_fpath, verbose=True)
        ut.printex(ex, 'ERROR: Failed open image size')
        raise
    return size


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def cvt_BGR2RGB(imgBGR):
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    return imgRGB


def warpAffine(img, Aff, dsize, assume_float01=True):
    """
    dsize = (width, height) of return image

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        Aff (ndarray): affine matrix
        dsize (tuple):  width, height

    Returns:
        ndarray: warped_img

    CommandLine:
        python -m vtool_ibeis.image --test-warpAffine --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> Aff = vt.rotation_mat3x3(TAU / 8)
        >>> dsize = vt.get_size(img)
        >>> warped_img = warpAffine(img, Aff, dsize)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(warped_img)
        >>> ut.show_if_requested()

    Timeit:
        import skimage.transform
        %timeit cv2.warpAffine(img, Aff[0:2], tuple(dsize), **CV2_WARP_KWARGS)
        100 loops, best of 3: 7.95 ms per loop
        skimage.transform.AffineTransform
        tf = skimage.transform.AffineTransform(rotation=TAU / 8)
        Aff_ = tf.params
        out = skimage.transform._warps_cy._warp_fast(img[:, :, 0], Aff_, output_shape=dsize, mode='constant', order=1)
        %timeit skimage.transform._warps_cy._warp_fast(img[:, :, 0], Aff_, output_shape=dsize, mode='constant', order=1)
        100 loops, best of 3: 5.74 ms per loop
        %timeit cv2.warpAffine(img[:, :, 0], Aff[0:2], tuple(dsize), **CV2_WARP_KWARGS)
        100 loops, best of 3: 5.13 ms per loop

        CONCLUSION, cv2 transforms are better

    """
    warped_img = cv2.warpAffine(img, Aff[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    if assume_float01 and img.dtype.kind == 'f':
        # Ensure that image intensity doesnt go out of range
        warped_img = warped_img.clip(0, 1)
    return warped_img


def warpHomog(img, Homog, dsize, assume_float01=True):
    """
    dsize = (width, height) of return image

    Example:
        >>> img = np.random.rand(224, 224)
        >>> Homog = np.random.rand(3, 3)
        >>> dsize = (128, 128)
        >>> warped_img = warpHomog(img, Homog, dsize)
    """
    warped_img = cv2.warpPerspective(img, Homog, tuple(dsize), **CV2_WARP_KWARGS)
    if assume_float01 and img.dtype.kind == 'f':
        # Ensure that image intensity doesnt go out of range
        warped_img = warped_img.clip(0, 1)
    return warped_img


def resize(img, dsize, interpolation=None):
    interpolation = _rectify_interpolation(interpolation)
    return cv2.resize(img, dsize, interpolation=interpolation)


def resize_mask(mask, chip, interpolation=None):
    dsize = get_size(chip)
    return resize(mask, dsize, interpolation)


def resize_image_by_scale(img, scale, interpolation=None):
    interpolation = _rectify_interpolation(interpolation)
    dsize, tonew_sf = get_round_scaled_dsize(get_size(img), scale)
    new_img = cv2.resize(img, dsize, interpolation=interpolation)
    return new_img


def resized_dims_and_ratio(img_size, max_dsize):
    """
    returns resized dimensions to get ``img_size`` to fit into ``max_dsize``

    FIXME:
        Should specifying a None force the use of the original dim?

    Args:
        img_size (tuple):
        max_dsize (tuple):

    Returns:
        tuple: (dsize, ratio)

    CommandLine:
        python -m vtool_ibeis.image resized_dims_and_ratio --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_size = (200, 100)
        >>> max_dsize = (150, 150)
        >>> (dsize, ratio) = resized_dims_and_ratio(img_size, max_dsize)
        >>> result = ('(dsize, ratio) = %s' % (ub.repr2((dsize, ratio), nl=0),))
        >>> print(result)
        (dsize, ratio) = ((150, 75), 0.75)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_size = (200, 100)
        >>> max_dsize = (5000, 1000)
        >>> (dsize, ratio) = resized_dims_and_ratio(img_size, max_dsize)
        >>> result = ('(dsize, ratio) = %s' % (ub.repr2((dsize, ratio), nl=0),))
        >>> print(result)
        (dsize, ratio) = ((2000, 1000), 10.0)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_size = (200, 100)
        >>> max_dsize = (5000, None)
        >>> (dsize, ratio) = resized_dims_and_ratio(img_size, max_dsize)
        >>> result = ('(dsize, ratio) = %s' % (ub.repr2((dsize, ratio), nl=0),))
        >>> print(result)
        (dsize, ratio) = ((200, 100), 1.0)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_size = (200, 100)
        >>> max_dsize = (None, None)
        >>> (dsize, ratio) = resized_dims_and_ratio(img_size, max_dsize)
        >>> result = ('(dsize, ratio) = %s' % (ub.repr2((dsize, ratio), nl=0),))
        >>> print(result)
        (dsize, ratio) = ((200, 100), 1.0)
    """
    #if isinstance(max_dsize, (tuple, list, np.ndarray)):
    max_width, max_height = max_dsize
    width, height = img_size
    if False:
        if max_width is not None and max_height is not None:
            ratio = min(max_width / width, max_height / height)
        elif max_width is not None:
            ratio = max_width / width
        elif max_width is not None:
            ratio = max_height / height
        else:
            ratio = 1.0
    else:
        if max_width is None:
            max_width = width
        if max_height is None:
            max_height = height
        ratio = min(max_width / width, max_height / height)
    dsize = (int(round(width * ratio)), int(round(height * ratio)))
    return dsize, ratio


def resized_clamped_thumb_dims(img_size, max_dsize):
    dsize_, ratio = resized_dims_and_ratio(img_size, max_dsize)
    dsize = img_size if ratio > 1 else dsize_
    sx = dsize[0] / img_size[0]
    sy = dsize[1] / img_size[1]
    return dsize, sx, sy


def pad_image_ondisk(img_fpath, pad_, out_fpath=None, value=0,
                      borderType=cv2.BORDER_CONSTANT, **kwargs):
    r"""
    Returns:
        str: out_fpath -  file path string

    CommandLine:
        python -m vtool_ibeis.image pad_image_ondisk

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_fpath = ut.get_argval('--fpath', type_=str)
        >>> pad_ = '?'
        >>> out_fpath = None
        >>> value = 0
        >>> borderType = 0
        >>> out_fpath = pad_image_ondisk(img_fpath, pad_, out_fpath, value, borderType)
        >>> result = ('out_fpath = %s' % (ub.repr2(out_fpath),))
        >>> print(result)
    """
    imgBGR = imread(img_fpath)
    imgBGR2 = cv2.copyMakeBorder(imgBGR, pad_, pad_, pad_, pad_,
                                 borderType=cv2.BORDER_CONSTANT, value=value)
    imgBGR2[:pad_, :] = value
    imgBGR2[-pad_:, :] = value
    imgBGR2[:, :pad_] = value
    imgBGR2[:, -pad_:] = value
    if out_fpath is None:
        out_fpath = ut.augpath(img_fpath, '_pad=%r' % (pad_))
    imwrite(out_fpath, imgBGR2)
    return out_fpath


def pad_image(imgBGR, pad_, value=0, borderType=cv2.BORDER_CONSTANT):
    imgBGR2 = cv2.copyMakeBorder(imgBGR, pad_, pad_, pad_, pad_,
                                 borderType=cv2.BORDER_CONSTANT, value=value)
    return imgBGR2


def get_pixel_dist(img, pixel, channel=None):
    """
    pixel = fillval
    isfill = mask2d
    """
    if not isinstance(pixel, np.ndarray):
        if isinstance(pixel, (list, tuple)):
            pixel = np.array(pixel)
        else:
            pixel = np.array([pixel])
    mask2d = np.abs(img - pixel[None, None, :])
    if len(img.shape) > 2:
        if channel is None:
            mask2d = np.sum(mask2d, axis=2)
        else:
            mask2d = mask2d[:, :, channel]
    return mask2d


def make_white_transparent(imgBGR):
    r"""
    Args:
        imgBGR (ndarray[uint8_t, ndim=2]):  image data (blue, green, red)

    Returns:
        ndarray: imgBGRA

    CommandLine:
        python -m vtool_ibeis.image make_white_transparent --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> imgBGR = imread(ut.get_argval('--fpath', type_=str))
        >>> imgBGRA = make_white_transparent(imgBGR)
        >>> result = ('imgBGRA = %s' % (ub.repr2(imgBGRA),))
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> ut.show_if_requested()
    """
    import vtool_ibeis as vt
    dist = vt.get_pixel_dist(imgBGR, [255, 255, 255])
    # grayflags = np.logical_and(imgBGR[:, :, 0] == imgBGR[:, :, 1], imgBGR[:, :, 1] == imgBGR[:, :, 2])
    # dist = dist / dist.max()
    imgBGRA = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2BGRA)
    # imgBGRA[:, :, 3] = np.maximum(np.round(dist * 255), ~grayflags * 255)
    # imgBGRA[:, :, 3] = np.round(dist * 255)
    imgBGRA[:, :, 3] = np.round(dist / 3)
    return imgBGRA


def crop_out_imgfill(img, fillval=None, thresh=0, channel=None):
    r"""
    Crops image to remove fillval

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        fillval (None): (default = None)
        thresh (int): (default = 0)

    Returns:
        ndarray: cropped_img

    CommandLine:
        python -m vtool_ibeis.image --exec-crop_out_imgfill

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img = vt.get_stripe_patch()
        >>> img = (img * 255).astype(np.uint8)
        >>> print(img)
        >>> img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        >>> fillval = np.array([25, 25, 25])
        >>> thresh = 0
        >>> cropped_img = crop_out_imgfill(img, fillval, thresh)
        >>> cropped_img2 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        >>> result = ('cropped_img2 = \n%s' % (str(cropped_img2),))
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img = vt.get_stripe_patch()
        >>> img = (img * 255).astype(np.uint8)
        >>> print(img)
        >>> fillval = 25
        >>> thresh = 0
        >>> cropped_img = crop_out_imgfill(img, fillval, thresh)
        >>> result = ('cropped_img = \n%s' % (str(cropped_img),))
        >>> print(result)
    """
    import vtool_ibeis as vt
    if fillval is None:
        fillval = np.array([255] * get_num_channels(img))
    # for colored images
    #with ut.embed_on_exception_context:
    pixel = fillval
    dist = get_pixel_dist(img, pixel, channel=channel)
    isfill = dist <= thresh
    # isfill should just be 2D
    # Fix shape that comes back as (1, W, H)
    if len(isfill.shape) == 3 and isfill.shape[0] == 1:
        if np.all(np.greater(isfill.shape[1:2], [4, 4])):
            isfill = isfill[0]
    rowslice, colslice = vt.get_crop_slices(isfill)
    cropped_img = img[rowslice, colslice]
    return cropped_img


def clipwhite_ondisk(fpath_in, fpath_out=None, verbose=ut.NOT_QUIET):
    r"""
    Strips white borders off an image on disk

    Args:
        fpath_in (str):
        fpath_out (None): (default = None)
        verbose (bool):  verbosity flag(default = True)

    Returns:
        str: fpath_out

    CommandLine:
        python -m vtool_ibeis.image clipwhite_ondisk
    """
    import vtool_ibeis as vt
    if fpath_out is None:
        fpath_out = ut.augpath(fpath_in, '_clipwhite')
    img = vt.imread(fpath_in, flags=cv2.IMREAD_UNCHANGED)
    if verbose:
        print('[clipwhite] img.shape = %r' % (img.shape,))
    cropped_img = clipwhite(img)
    if verbose:
        print('[clipwhite] cropped_img.shape = %r' % (cropped_img.shape,))
    vt.imwrite(fpath_out, cropped_img)
    return fpath_out


def clipwhite(img):
    """
    Strips white borders off an image
    """
    nChannels = get_num_channels(img)
    # thresh = 128
    thresh = 64
    if nChannels == 4:
        # alpha
        thresh = 12
        fillval = 0
        channel = 3
        # imgBGRA
        # Clip alpha first
        cropped_img = crop_out_imgfill(img, fillval, thresh=thresh, channel=channel)
        # Clip white next
        fillval = np.array([255] * nChannels)
        cropped_img = crop_out_imgfill(cropped_img, fillval=fillval, thresh=thresh)
        # cropped_img = crop_out_imgfill(img, fillval=fillval, thresh=thresh)
    else:
        fillval = np.array([255] * nChannels)
        cropped_img = crop_out_imgfill(img, fillval=fillval, thresh=thresh)
    return cropped_img


def rotate_image_ondisk(img_fpath, theta, out_fpath=None, **kwargs):
    r"""
    Rotates an image on disk

    Args:
        img_fpath (?):
        theta (?):
        out_fpath (None):

    CommandLine:
        python -m vtool_ibeis.image --test-rotate_image_ondisk

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath('star.png')
        >>> theta = TAU * 3 / 8
        >>> # execute function
        >>> out_fpath = None
        >>> out_fpath_ = rotate_image_ondisk(img_fpath, theta, out_fpath)
        >>> print(out_fpath_)
        >>> if ut.get_argflag('--show') or ut.inIPython():
        >>>     import plottool_ibeis as pt
        >>>     pt.imshow(out_fpath_,  pnum=(1, 1, 1))
        >>>     pt.show_if_requested()

    """
    img = imread(img_fpath)
    imgR = rotate_image(img, theta, **kwargs)
    out_fpath_ = (
        ut.augpath(img_fpath, augsuf='_theta=%r' % (theta))
        if out_fpath is None else out_fpath)
    imwrite(out_fpath_, imgR)
    return out_fpath_


def rotate_image(img, theta, border_mode=None, interpolation=None, dsize=None):
    r"""
    Rotates an image around its center

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        theta (?):

    CommandLine:
        python -m vtool_ibeis.image --test-rotate_image

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # build test data
        >>> img = vt.get_test_patch('star2')
        >>> theta = TAU / 16.0
        >>> # execute function
        >>> imgR = rotate_image(img, theta)
        >>> if ut.get_argflag('--show') or ut.inIPython():
        >>>     import plottool_ibeis as pt
        >>>     pt.imshow(img  * 255,  pnum=(1, 2, 1))
        >>>     pt.imshow(imgR * 255, pnum=(1, 2, 2))
        >>>     pt.show_if_requested()
    """
    from vtool_ibeis import linalg as ltool
    border_mode = _rectify_border_mode(border_mode)
    interpolation = _rectify_interpolation(interpolation)
    bbox0 = [0, 0, img.shape[1], img.shape[0]]
    if dsize is None:
        dsize = [img.shape[1], img.shape[0]]
        bbox1 = bbox0
    else:
        bbox1 = [0, 0, dsize[0], dsize[1]]
    R = ltool.rotation_around_bbox_mat3x3(theta, bbox0, bbox1=bbox1)
    imgR = cv2.warpAffine(img, R[0:2], tuple(dsize), borderMode=border_mode,
                          flags=interpolation)
    return imgR


def shear(img, x_shear, y_shear, dsize=None, **kwargs):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        x_shear (?):
        y_shear (?):
        dsize (tuple):  width, height

    CommandLine:
        python -m vtool_ibeis.image --test-shear --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> x_shear = 0.05
        >>> y_shear = -0.05
        >>> dsize = None
        >>> imgSh = shear(img, x_shear, y_shear, dsize)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgSh)
        >>> ut.show_if_requested()
    """
    from vtool_ibeis import linalg as ltool
    if dsize is None:
        dsize = get_size(img)
    shear_mat3x3 = ltool.shear_mat3x3(x_shear, y_shear)
    warp_kwargs = CV2_WARP_KWARGS.copy()
    warp_kwargs.update(kwargs)
    imgSh = cv2.warpAffine(img, shear_mat3x3[0:2], tuple(dsize), **warp_kwargs)
    return imgSh


def affine_warp_around_center(img, sx=1, sy=1, theta=0, shear=0, tx=0, ty=0,
                              dsize=None, borderMode=cv2.BORDER_CONSTANT,
                              flags=cv2.INTER_LANCZOS4, out=None, **kwargs):
    r"""

    CommandLine:
        python -m vtool_ibeis.image --test-affine_warp_around_center --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath) / 255.0
        >>> img = img.astype(np.float32)
        >>> dsize = (1000, 1000)
        >>> shear = .2
        >>> theta = np.pi / 4
        >>> tx = 0
        >>> ty = 100
        >>> sx = 1.5
        >>> sy = 1.0
        >>> borderMode = cv2.BORDER_CONSTANT
        >>> flags = cv2.INTER_LANCZOS4
        >>> img_warped = affine_warp_around_center(img, sx=sx, sy=sy,
        ...     theta=theta, shear=shear, tx=tx, ty=ty, dsize=dsize,
        ...     borderMode=borderMode, flags=flags, borderValue=(.5, .5, .5))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow((img_warped * 255.0).astype(np.uint8))
        >>> ut.show_if_requested()
    """
    from vtool_ibeis import linalg as ltool
    if dsize is None:
        dsize = (img.shape[1], img.shape[0])
    else:
        dsize = tuple(dsize)
    w2, h2 = dsize
    h1, w1 = img.shape[0:2]
    y1, x1 = h1 / 2.0, w1 / 2.0
    y2, x2 = h2 / 2.0, w2 / 2.0
    # MOVE AFFINE AROUND w.r.t new dsize
    Aff = ltool.affine_around_mat3x3(x1, y1, sx, sy, theta, shear, tx, ty, x2, y2)
    img_warped = cv2.warpAffine(img, Aff[0:2], dsize, dst=out,
                                borderMode=borderMode, flags=flags, **kwargs)
    # Fix grayscale channel issues
    if len(img.shape) == 3 and len(img_warped.shape) == 2:
        img_warped.shape = img_warped.shape + (1,)
    return img_warped


def get_round_scaled_dsize(dsize, scale):
    """
    Returns an integer size and scale that best approximates
    the floating point scale on the original size

    Args:
        dsize (tuple): original width height
        scale (float or tuple): desired floating point scale factor
    """
    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(round(w * sx))
    new_h = int(round(h * sy))
    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)
    return new_dsize, new_scale


def rectify_to_square(img, extreme='max'):
    h, w = img.shape[0:2]
    if w == h:
        return img
    else:
        extreme_fn = {'max': max, 'min': min}[extreme]
        d = extreme_fn(w, h)
        return resize(img, (d, d))


def rectify_to_float01(img, dtype=np.float32):
    """ Ensure that an image is encoded using a float properly """
    if img.dtype.kind in ('i', 'u'):
        assert img.max() <= 255
        img_ = img.astype(dtype) / 255.0
    else:
        img_ = img.astype(dtype)
    return img_


def rectify_to_uint8(img):
    """ Ensure that an image is encoded in uint8 properly """
    if img.dtype.kind in ('f'):
        if img.max() <= 1.0 or img.min() >= 0.0:
            raise ValueError('Bad input image. Stats={}'.format(
                ub.repr2(ut.get_stats(img.ravel()), precision=2)))
        img_ = (img * 255.0).astype(np.uint8)
    else:
        img_ = img
    return img_


def make_channels_comparable(img1, img2):
    """
    Broadcasts image arrays so they can have elementwise operations applied

    CommandLine:
        python -m vtool_ibeis.image make_channels_comparable

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> wh_basis = [(5, 5), (3, 5), (5, 3), (1, 1), (1, 3), (3, 1)]
        >>> for w, h in wh_basis:
        >>>     shape_basis = [(w, h), (w, h, 1), (w, h, 3)]
        >>>     # Test all permutations of shap inputs
        >>>     for shape1, shape2 in ut.product(shape_basis, shape_basis):
        >>>         print('*    input shapes: %r, %r' % (shape1, shape2))
        >>>         img1 = np.empty(shape1)
        >>>         img2 = np.empty(shape2)
        >>>         img1, img2 = make_channels_comparable(img1, img2)
        >>>         print('... output shapes: %r, %r' % (img1.shape, img2.shape))
        >>>         elem = (img1 + img2)
        >>>         print('... elem(+) shape: %r' % (elem.shape,))
        >>>         assert elem.size == img1.size, 'outputs should have same size'
        >>>         assert img1.size == img2.size, 'new imgs should have same size'
        >>>         print('--------')
    """
    import vtool_ibeis as vt
    # w1, h1 = get_size(img1)
    # w2, h2 = get_size(img2)
    # if not (w1 == w2 and h1 == h2):
    #     raise AssertionError(
    #         'Images must have same size, %r, %r' % ((w1, h1), (w2, h2)))
    if img1.shape != img2.shape:
        c1 = vt.get_num_channels(img1)
        c2 = vt.get_num_channels(img2)
        if len(img1.shape) == 2 and len(img2.shape) == 2:
            raise AssertionError('UNREACHABLE: Both are 2-grayscale')
        elif len(img1.shape) == 3 and len(img2.shape) == 2:
            # Image 2 is grayscale
            if c1 == 3:
                img2 = np.tile(img2[..., None], 3)
            else:
                img2 = img2[..., None]
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Image 1 is grayscale
            if c2 == 3:
                img1 = np.tile(img1[..., None], 3)
            else:
                img1 = img1[..., None]
        elif len(img1.shape) == 3 and len(img2.shape) == 3:
            # Both images have 3 dims.
            # Check if either have color, then check for alpha
            if c1 == 1 and c2 == 1:
                raise AssertionError('UNREACHABLE: Both are 3-grayscale')
            elif c1 == 3 and c2 == 3:
                raise AssertionError('UNREACHABLE: Both are 3-color')
            elif c1 == 1 and c2 == 3:
                img1 = np.tile(img1, 3)
            elif c1 == 3 and c2 == 1:
                img2 = np.tile(img2, 3)
            elif c1 == 3 and c2  == 4:
                # raise NotImplementedError('alpha not handled yet')
                # assumes img1 is in 0:1 format
                img1 = np.dstack((img1, np.ones(img1.shape[0:2])))
            elif c1 == 4 and c2  == 3:
                # assumes img1 is in 0:1 format
                img2 = np.dstack((img2, np.ones(img2.shape[0:2])))
                # raise NotImplementedError('alpha not handled yet')
            elif c1 == 1 and c2  == 4:
                img1 = np.dstack((np.tile(img1, 3), np.ones(img1.shape[0:2])))
            elif c1 == 4 and c2  == 1:
                img2 = np.dstack((np.tile(img2, 3), np.ones(img2.shape[0:2])))
            else:
                raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
        else:
            raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
    return img1, img2


def _lookup_colorspace_code(colorspace, src_colorspace='BGR'):
    src_colorspace = src_colorspace.upper()
    colorspace = colorspace.upper()
    prefix = 'COLOR_' + src_colorspace + '2'
    valid_dst_colorspaces = [
        key.replace(prefix, '')
        for key in cv2.__dict__.keys() if key.startswith(prefix)]
    if colorspace not in valid_dst_colorspaces:
        raise NotImplementedError('unknown colorspace = %r' % (colorspace,))
    else:
        key = prefix + colorspace
        code = cv2.__dict__[key]
    return code


def convert_colorspace(img, colorspace, src_colorspace='BGR'):
    r"""
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        colorspace (str): RGB, LAB, etc
        src_colorspace (unicode): (default = u'BGR')

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    CommandLine:
        python -m vtool_ibeis.image convert_colorspace --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('zebra.png')
        >>> img_fpath = ut.grab_file_url('http://itsnasb.com/wp-content/uploads/2013/03/lisa-frank-logo1.jpg')
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> img_float = vt.rectify_to_float01(img, np.float32)
        >>> colorspace = 'LAB'
        >>> src_colorspace = 'BGR'
        >>> imgLAB = convert_colorspace(img, colorspace, src_colorspace)
        >>> imgL = imgLAB[:, :, 0]
        >>> fillL = imgL.mean()
        >>> fillAB = 0 if ut.is_float(img) else 128
        >>> imgAB_LAB = vt.embed_channels(imgLAB[:, :, 1:3], (1, 2), fill=fillL)
        >>> imgA_LAB  = vt.embed_channels(imgLAB[:, :, 1], (1,), fill=(fillL, fillAB))
        >>> imgB_LAB  = vt.embed_channels(imgLAB[:, :, 2], (2,), fill=(fillL, fillAB))
        >>> imgAB_BGR = convert_colorspace(imgAB_LAB, src_colorspace, colorspace)
        >>> imgA_BGR = convert_colorspace(imgA_LAB, src_colorspace, colorspace)
        >>> imgB_BGR = convert_colorspace(imgB_LAB, src_colorspace, colorspace)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> #imgAB_HSV = convert_colorspace(convert_colorspace(imgAB_LAB, 'LAB', 'BGR'), 'BGR', 'HSV')
        >>> imgAB_HSV = convert_colorspace(img, 'HSV', 'BGR')
        >>> imgAB_HSV[:, :, 1:3] = .6 if ut.is_float(img) else 128
        >>> imgCOLOR_BRG = convert_colorspace(imgAB_HSV, 'BGR', 'HSV')
        >>> pt.imshow(img, pnum=(3, 4, 1), title='input')
        >>> pt.imshow(imgL, pnum=(3, 4, 2), title='L (lightness)')
        >>> pt.imshow((imgLAB[:, :, 1]), pnum=(3, 4, 3), title='A (grayscale)')
        >>> pt.imshow((imgLAB[:, :, 2]), pnum=(3, 4, 4), title='B (grayscale)')
        >>> pt.imshow(imgCOLOR_BRG, pnum=(3, 4, 5), title='Hue')
        >>> pt.imshow(imgAB_BGR, pnum=(3, 4, 6), title='A+B (color overlay)')
        >>> pt.imshow(imgA_BGR, pnum=(3, 4, 7), title='A (Red-Green)')
        >>> pt.imshow(imgB_BGR, pnum=(3, 4, 8), title='B (Blue-Yellow)')
        >>> rgblind_LAB = vt.embed_channels(imgLAB[:, :, (0, 2)], (0, 2), fill=fillAB)
        >>> rgblind_BRG = convert_colorspace(rgblind_LAB, src_colorspace, colorspace)
        >>> byblind_LAB = vt.embed_channels(imgLAB[:, :, (0, 1)], (0, 1), fill=fillAB)
        >>> byblind_BGR = convert_colorspace(byblind_LAB, src_colorspace, colorspace)
        >>> pt.imshow(byblind_BGR, title='colorblind B-Y', pnum=(3, 4, 11))
        >>> pt.imshow(rgblind_BRG, title='colorblind R-G', pnum=(3, 4, 12))
        >>> ut.show_if_requested()
    """
    src_colorspace = src_colorspace.upper()
    colorspace = colorspace.upper()
    if colorspace == src_colorspace:
        return img  # FIXME copy?
    code = _lookup_colorspace_code(colorspace, src_colorspace)
    img2 = cv2.cvtColor(img, code)
    return img2


def convert_image_list_colorspace(image_list, colorspace, src_colorspace='BGR'):
    """
    converts a list of images from <src_colorspace> to <colorspace>
    """
    src_colorspace = src_colorspace.upper()
    colorspace = colorspace.upper()
    if colorspace == src_colorspace:
        return image_list
    code = _lookup_colorspace_code(colorspace, src_colorspace)
    if isinstance(image_list, np.ndarray):
        # Be more efficient if using numpy arrays
        if colorspace == 'GRAY' and src_colorspace != 'GRAY':
            image_list2 = np.empty(image_list.shape[0:3], dtype=image_list.dtype)
        else:
            image_list2 = np.empty(image_list.shape, dtype=image_list.dtype)
        for index in range(len(image_list2)):
            cv2.cvtColor(image_list[index], code, dst=image_list2[index])
    else:
        # If python list use comprehension
        image_list2 = [cv2.cvtColor(img, code) for img in image_list]
    return image_list2


def padded_resize(img, target_size=(64, 64), interpolation=None):
    r"""
    makes the image resize to the target size and pads the rest of the area with a fill value

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        target_size (tuple):

    CommandLine:
        python -m vtool_ibeis.image --test-padded_resize --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> imgA = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> imgB = vt.imread(ut.grab_test_imgpath('ada.jpg'))
        >>> imgC = vt.imread(ut.grab_test_imgpath('carl.jpg'), grayscale=True)
        >>> #target_size = (64, 64)
        >>> target_size = (1024, 1024)
        >>> img3_list = [padded_resize(img, target_size) for img in [imgA, imgB, imgC]]
        >>> # verify results
        >>> assert ut.allsame([vt.get_size(img3) for img3 in img3_list])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pnum_ = pt.make_pnum_nextgen(1, 3)
        >>> pt.imshow(img3_list[0], pnum=pnum_())
        >>> pt.imshow(img3_list[1], pnum=pnum_())
        >>> pt.imshow(img3_list[2], pnum=pnum_())
        >>> ut.show_if_requested()
    """
    interpolation = _rectify_interpolation(interpolation)
    img2 = resize_to_maxdims(img, target_size, interpolation=interpolation)
    dsize2 = get_size(img2)
    if dsize2 != target_size:
        img3 = embed_in_square_image(img2, target_size)
    else:
        img3 = img2
    return img3


def embed_in_square_image(img, target_size, img_origin=(.5, .5),
                          target_origin=(.5, .5)):
    r"""
    Embeds an image in the center of an empty image

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        target_size (tuple):
        offset (tuple): position of

    Returns:
        ndarray: img_sqare

    CommandLine:
        python -m vtool_ibeis.image embed_in_square_image --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> target_size = tuple(np.array(vt.get_size(img)) * 3)
        >>> img_origin = (.5, .5)
        >>> target_origin = (.5, .5)
        >>> img_square = embed_in_square_image(img, target_size, img_origin, target_origin)
        >>> assert img_square.sum() == img.sum()
        >>> assert vt.get_size(img_square) == target_size
        >>> img_origin = (0, 0)
        >>> target_origin = (0, 0)
        >>> img_square2 = embed_in_square_image(img, target_size, img_origin, target_origin)
        >>> assert img_square.sum() == img.sum()
        >>> assert vt.get_size(img_square) == target_size
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img_square, pnum=(1, 2, 1))
        >>> pt.imshow(img_square2, pnum=(1, 2, 2))
        >>> ut.show_if_requested()
    """
    # Allocate large image
    num_channels = get_num_channels(img)
    target_shape = (target_size[::-1] if num_channels == 1 else
                    tuple(target_size)[::-1] + (num_channels,))
    img_sqare = np.zeros(target_shape, dtype=img.dtype)
    # Determine slice of target shape that places img_origin at target_origin
    target_rc = np.array(target_shape[0:2])
    img_rc = np.array(img.shape[0:2])

    img_origin_abs = np.array(img_origin)[::-1] * img_rc
    target_origin_abs = np.array(target_origin)[::-1] * target_rc

    #img_left_rc = img_rc - img_origin_abs
    #img_right_rc = img_origin_abs
    # TODO: allow image to hang off edge

    #print('img_rc = %r' % (img_rc,))
    #print('img_origin = %r' % (img_origin,))
    #print('img_origin_abs = %r' % (img_origin_abs,))

    #print('target_rc = %r' % (target_rc,))
    #print('target_origin = %r' % (target_origin,))
    #print('target_origin_abs = %r' % (target_origin_abs,))

    ## Find start slice in the target image
    target_diff = np.floor(target_origin_abs - img_origin_abs)
    target_rc_start = np.maximum(target_diff, 0).astype(np.int)

    img_rc_start = (-(target_diff - target_rc_start)).astype(np.int)
    img_clip_rc_low = img_rc - img_rc_start

    end_hang = np.maximum((target_rc_start + img_clip_rc_low) - target_rc, 0)
    img_clip_rc = img_clip_rc_low - end_hang

    img_rc_end = (img_rc_start + img_clip_rc).astype(np.int)
    target_rc_end = (target_rc_start + img_clip_rc).astype(np.int)

    img_rc_slice = [slice(b, e) for (b, e) in zip(img_rc_start, img_rc_end)]
    target_rc_slice = [slice(b, e) for (b, e) in zip(target_rc_start, target_rc_end)]

    # embed image at position
    img_sqare[target_rc_slice[0], target_rc_slice[1]] = img[img_rc_slice[0], img_rc_slice[1]]

    #target_rc_overhang = target_rc_start - target_rc_hangstart

    ##-np.minimum(np.floor(img_origin_abs - target_origin_abs), 0)
    ##img_origin_abs - target_rc_overhang

    ## Find start slice in the given image
    #img_rc_start = img_rc - target_rc_overhang
    #cliped_img_rc = img_rc - img_rc_start

    #(target_rc_start + cliped_img_rc) - target_rc

    #image_rc_start = np.maximum(target_rc_overhang, 0)
    #image_rc_end = img_rc - image_rc_start
    #image_rc_start = np.maximum(-target_rc_start, 0)
    #image_rc_end = img_rc - image_rc_start

    if False:
        rc_diff = target_rc - img_rc  # amount of extra space in target
        rc_start = np.floor(rc_diff / 2)
        rc_end  =  [None if e == 0 else e for e in (rc_start - rc_diff)]
        rc_slice = [slice(b, e) for (b, e) in zip(rc_start, rc_end)]
        # embed image at center
        img_sqare[rc_slice[0], rc_slice[1]] = img
    return img_sqare


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


def resize_to_maxdims_ondisk(img_fpath, max_dsize, out_fpath=None):
    r"""
    Args:
        img_fpath (str):  file path string
        max_dsize (?):
        out_fpath (str):  file path string(default = None)

    CommandLine:
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormA.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormB.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormC.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormD.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormE.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormF.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormG.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormH.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormI.png --dsize=417,None
        python -m vtool_ibeis.image resize_to_maxdims_ondisk --fpath ~/latex/crall-candidacy-2015/figures3/knormJ.png --dsize=417,None

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_fpath = ut.get_argval('--fpath')
        >>> max_dsize = ut.get_argval('--dsize', type_=list)
        >>> out_fpath = None
        >>> resize_to_maxdims_ondisk(img_fpath, max_dsize, out_fpath)
    """
    img = imread(img_fpath, flags=cv2.IMREAD_UNCHANGED)
    img2 = resize_to_maxdims(img, max_dsize)
    out_fpath_ = ut.augpath(img_fpath, '_max_dsize=%r' % (max_dsize)) if out_fpath is None else out_fpath
    imwrite(out_fpath_, img2)


def resize_to_maxdims(img, max_dsize=(64, 64),
                      interpolation=None):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        max_dsize (tuple):
        interpolation (long):

    CommandLine:
        python -m vtool_ibeis.image --test-resize_to_maxdims --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> max_dsize = (1024, 1024)
        >>> img2 = resize_to_maxdims(img, max_dsize)
        >>> print('img.shape = %r' % (img.shape,))
        >>> print('img2.shape = %r' % (img2.shape,))
        >>> # verify results
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img2)
        >>> ut.show_if_requested()
    """
    img_size = get_size(img)
    dsize, ratio = resized_dims_and_ratio(img_size, max_dsize)
    interpolation = _rectify_interpolation(interpolation)
    return cv2.resize(img, dsize, interpolation=interpolation)


def resize_thumb(img, max_dsize=(64, 64), interpolation=None):
    """
    Resize an image such that its max width or height is:

    CommandLine:
        python -m vtool_ibeis.image --test-resize_thumb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> max_dsize = (64, 64)
        >>> # execute function
        >>> img2 = resize_thumb(img, max_dsize)
        >>> print('img.shape = %r' % (img.shape,))
        >>> print('img2.shape = %r' % (img2.shape,))
        >>> # verify results
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img2)
        >>> ut.show_if_requested()
    """
    height, width = img.shape[0:2]
    img_size = (width, height)
    dsize, ratio = resized_dims_and_ratio(img_size, max_dsize)
    interpolation = _rectify_interpolation(interpolation)
    if ratio > 1:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # return cvt_BGR2RGB(img)
    else:
        return cv2.resize(img, dsize, interpolation=interpolation)


def find_pixel_value_index(img, pixel):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        pixel (ndarray or scalar):

    CommandLine:
        python -m vtool_ibeis.util_math --test-find_pixel_value_index

    References:
        http://stackoverflow.com/questions/21407815/get-column-row-index-from-numpy-array-that-meets-a-boolean-condition

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> # build test data
        >>> img = np.random.rand(10, 10, 3) + 1.0
        >>> pixel = np.array([0, 0, 0])
        >>> img[5, 5, :] = pixel
        >>> img[2, 3, :] = pixel
        >>> img[1, 1, :] = pixel
        >>> img[0, 0, :] = pixel
        >>> img[2, 0, :] = pixel
        >>> # execute function
        >>> result = find_pixel_value_index(img, pixel)
        >>> # verify results
        >>> print(result)
        [[0 0]
         [1 1]
         [2 0]
         [2 3]
         [5 5]]
    """
    mask2d = get_pixel_dist(img, pixel) == 0
    pixel_locs = np.column_stack(np.where(mask2d))
    return pixel_locs


def draw_text(img, text, org, textcolor_rgb=[0, 0, 0], fontScale=1,
              thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=LINE_AA,
              bottomLeftOrigin=False):
    """

    CommandLine:
        python -m vtool_ibeis.image --test-draw_text:0 --show
        python -m vtool_ibeis.image --test-draw_text:1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> font_names = sorted([key for key in cv2.__dict__.keys() if key.startswith('FONT_H')])
        >>> text = 'opencv'
        >>> img = np.zeros((400, 1024), dtype=np.uint8)
        >>> thickness = 2
        >>> fontScale = 1.0
        >>> lineType = 4
        >>> lineType = 8
        >>> lineType = cv2.CV_AA
        >>> for count, font_name in enumerate(font_names, start=1):
        >>>     print(font_name)
        >>>     fontFace = cv2.__dict__[font_name]
        >>>     org = (10, count * 45)
        >>>     text = 'opencv - ' + font_name
        >>>     vt.draw_text(img, text, org,
        ...                  fontFace=fontFace, textcolor_rgb=[255, 255, 255],
        ...                  fontScale=fontScale, thickness=thickness)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img)
        >>> ut.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> font_names = sorted([key for key in cv2.__dict__.keys() if key.startswith('FONT_H')])
        >>> text = 'opencv'
        >>> img = np.zeros((400, 1024, 3), dtype=np.uint8)
        >>> img[:200, :512, 0] = 255
        >>> img[200:, 512:, 2] = 255
        >>> thickness = 2
        >>> fontScale = 1.0
        >>> lineType = 4
        >>> lineType = 8
        >>> lineType = cv2.CV_AA
        >>> for count, font_name in enumerate(font_names, start=1):
        >>>     print(font_name)
        >>>     fontFace = cv2.__dict__[font_name]
        >>>     org = (10, count * 45)
        >>>     text = 'opencv - ' + font_name
        >>>     vt.draw_text(img, text, org,
        ...                  fontFace=fontFace, textcolor_rgb=[255, 255, 255],
        ...                  fontScale=fontScale, thickness=thickness)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img)
        >>> ut.show_if_requested()

    where each of the font IDs can be combined with FONT_ITALIC to get the slanted letters.
    """
    if len(textcolor_rgb) == 4:
        # remove alpha
        textcolor_rgb = textcolor_rgb[:3]
    textcolor_bgr = textcolor_rgb[::-1]
    text_pt, text_sz = cv2.getTextSize(text, fontFace, fontScale, thickness)
    text_w, text_h = text_pt
    out = cv2.putText(img, text, org, fontFace, fontScale, textcolor_bgr,
                      thickness, lineType, bottomLeftOrigin)
    return out


#def testing(img):
#    r"""
#    Args:
#        img (ndarray[uint8_t, ndim=2]):  image data
#
#    CommandLine:
#        python -m vtool_ibeis.image --test-testing --show
#
#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from vtool_ibeis.image import *  # NOQA
#        >>> import vtool_ibeis as vt
#        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
#        >>> img = vt.imread(img_fpath)
#        >>> img2 = testing(img)
#        >>> # xdoctest: +REQUIRES(--show)
#        >>> import plottool_ibeis as pt
#        >>> pt.imshow(img, pnum=(1, 2, 1))
#        >>> pt.imshow(img2, pnum=(1, 2, 2))
#        >>> ut.show_if_requested()
#    """
#    img = img.astype(np.float32)
#    mask = np.ones(img.shape, dtype=np.uint8)
#    img2 = img.copy()
#    alpha = 2.2
#    beta = 50.0
#    cv2.illuminationChange(img, mask, img2, alpha, beta)
#    #ut.embed()
#    return img2


def perlin_noise(size, scale=32.0, rng=np.random):
    """
    References:
        http://www.siafoo.net/snippet/229

    CommandLine:
        python -m vtool_ibeis.image perlin_noise --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> #size = (64, 64)
        >>> size = (256, 256)
        >>> #scale = 32.0
        >>> scale = 64.0
        >>> img = perlin_noise(size, scale)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(img, pnum=(1, 1, 1))
        >>> ut.show_if_requested()
    """
    #from PIL import Image

    class PerlinNoiseGenerator(object):

        def __init__(self, size=None, n=None):

            n = n if n is not None else  256
            self.size = size if size is not None else (256, 256)

            self.order = len(self.size)

            # Generate WAY more numbers than we need
            # because we are throwing out all the numbers not inside a unit
            # sphere.  Something of a hack but statistically speaking
            # it should work fine... or crash.
            G = (rng.uniform(size=2 * self.order * n) * 2 - 1).reshape(-1, self.order)

            # GAH! How do I generalize this?!
            #length = hypot(G[:,i] for i in range(self.order))

            if self.order == 1:
                length = G[:, 0]
            elif self.order == 2:
                length = np.hypot(G[:, 0], G[:, 1])
            elif self.order == 3:
                length = np.hypot(G[:, 0], G[:, 1], G[:, 2])

            self.G = (G[length < 1] / (length[length < 1])[:, np.newaxis])[:n, ]
            self.P = np.arange(n, dtype=np.int32)

            rng.shuffle(self.P)

            self.idx_ar = np.indices(2 * np.ones(self.order, dtype=np.int8),
                                     dtype=np.int8).reshape(self.order, -1).T
            self.drop = np.poly1d((-6, 15, -10, 0, 0, 1.0))

        def noise(self, coords):

            ijk = (np.floor(coords) + self.idx_ar).astype(np.int8)

            uvw = coords - ijk

            indexes = self.P[ijk[:, :, self.order - 1]]

            for i in range(self.order - 1):
                indexes = self.P[(ijk[:, :, i] + indexes) % len(self.P)]

            gradiens = self.G[indexes % len(self.G)]
            #gradiens = self.G[(ijk[:,:, 0] + indexes) % len(self.G)]

            res = (self.drop(np.abs(uvw)).prod(axis=2) *
                   np.prod([gradiens, uvw], axis=0).sum(axis=2)).sum(axis=1)

            res[res > 1.0] = 1.0
            res[res < -1.0] = -1.0

            return ((res + 1) * 128).astype(np.int8)

        def getData(self, scale=32.0):
            return self.noise(np.indices(self.size).reshape(self.order, 1, -1).T / scale)

        def getImage(self, scale=32.0):
            return Image.frombuffer('L', self.size[:2],
                                    self.getData(scale)[ : self.size[0] * self.size[1]],
                                    'raw', 'L', 0, 1)

        def saveImage(self, fileName, scale=32.0):
            im = self.getImage(scale)
            im.save(fileName)

    self = PerlinNoiseGenerator(size=size)
    pil_img = self.getImage(scale=scale)
    img = np.array(pil_img.getdata()).reshape(size)
    return img


# STACK IMAGES STUFF


def testdata_imglist():
    # build test data
    import vtool_ibeis as vt
    img1 = vt.imread(ut.grab_test_imgpath('carl.jpg'))
    img2 = vt.imread(ut.grab_test_imgpath('astro.png'))
    img3 = vt.imread(ut.grab_test_imgpath('ada.jpg'))
    img4 = vt.imread(ut.grab_test_imgpath('jeff.png'))
    img5 = vt.imread(ut.grab_test_imgpath('star.png'))
    img_list = [img1, img2, img3, img4, img5]
    return img_list


def stack_image_list_special(img1, img_list, num=1, vert=True, use_larger=True,
                             initial_sf=None, interpolation=None):
    r"""
    # TODO: add initial scale down factor?

    CommandLine:
        python -m vtool_ibeis.image --test-stack_image_list_special --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_list_ = testdata_imglist()
        >>> img1 = img_list_[0]
        >>> img_list = img_list_[1:]
        >>> vert = True
        >>> return_offset = True
        >>> use_larger = False
        >>> num_bot = 1
        >>> initial_sf = None
        >>> initial_sf = .5
        >>> imgB, offset_list, sf_list = stack_image_list_special(img1, img_list, num_bot, vert, use_larger, initial_sf)
        >>> # xdoctest: +REQUIRES(--show)
        >>> wh_list = np.array([vt.get_size(img1)] + list(map(vt.get_size, img_list)))
        >>> wh_list_ = wh_list * sf_list
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgB)
        >>> print('imgB.shape = %r' % (imgB.shape,))
        >>> for offset, wh, color in zip(offset_list, wh_list_, pt.distinct_colors(len(offset_list))):
        ...    pt.draw_bbox((offset[0], offset[1], wh[0], wh[1]), bbox_color=color)
        >>> ut.show_if_requested()
    """
    import vtool_ibeis as vt
    interpolation = _rectify_interpolation(interpolation)
    #img2 = img_list[0]
    img_list2 = img_list[:num]
    img_list3 = img_list[num:]
    #img_list_ = img_list[1:]

    #interpolation = cv2.INTER_NEAREST

    stack_kw = dict(
        modifysize=True,
        return_sf=True,
        use_larger=use_larger,
        interpolation=interpolation
    )

    if vert is None:
        vert_, _ = ut.get_argval('--vert', return_was_specified=True)
        if _:
            vert = not vert_
        else:
            if len(img_list) > 0:
                vert = not infer_vert(img1, img_list[0], vert)[0]
            else:
                # HACK FIXME move flag setting to viz_matches or experiment drawing
                vert = True

    offset_list1 = [(0, 0)]
    if initial_sf is None:
        initial_sf = 1.0
        sf_list1     = [(initial_sf, initial_sf)]
        img1_ = img1
    else:
        dsize, initial_sf_ = vt.get_round_scaled_dsize(vt.get_size(img1), initial_sf)
        sf_list1 = [initial_sf_]
        img1_ = cv2.resize(img1, dsize, interpolation=interpolation)

    # stack the bottom images
    img_stack2, offset_list2, sf_list2 = stack_image_list(img_list2, vert=not
                                                          vert,
                                                          return_offset=True,
                                                          **stack_kw)
    # stack the top images
    img_stack3, offset_list3, sf_list3 = stack_image_list(img_list3, vert=vert,
                                                          return_offset=True,
                                                          **stack_kw)

    # stack img1_ and the first stack
    imgL, offset_listL, sf_listL = stack_multi_images(
        img1_, img_stack2, offset_list1, sf_list1, offset_list2, sf_list2,
        vert=vert, interpolation=interpolation)
    # stack the output and the second stack
    img, offset_list, sf_list = stack_multi_images(imgL, img_stack3,
                                                   offset_listL, sf_listL,
                                                   offset_list3, sf_list3,
                                                   vert=not vert)

    return img, offset_list, sf_list


# Combine the stacks
def stack_multi_images(img1, img2, offset_list1, sf_list1, offset_list2,
                       sf_list2, vert=True, use_larger=False, modifysize=True,
                       interpolation=None):
    """ combines images that are already stacked """
    interpolation = _rectify_interpolation(interpolation, default=cv2.INTER_NEAREST)
    if img1 is None:
        return img2, offset_list2, sf_list2
    if img2 is None:
        return img1, offset_list1, sf_list1
    # combine with the main image
    imgB, offset_tup, sf_tup = stack_images(img1, img2, vert=vert,
                                            use_larger=use_larger,
                                            modifysize=modifysize,
                                            return_sf=True,
                                            interpolation=interpolation)
    # combine the offsets
    def mult_tuplelist(tuple_list, scale_xy):
        return [(tup[0] * scale_xy[0], tup[1] * scale_xy[1]) for tup in tuple_list]
    def add_tuplelist(tuple_list, offset_xy):
        return [(tup[0] + offset_xy[0], tup[1] + offset_xy[1]) for tup in tuple_list]
    offset_list1_ = add_tuplelist(mult_tuplelist(offset_list1, sf_tup[0]), offset_tup[0])
    offset_list2_ = add_tuplelist(mult_tuplelist(offset_list2, sf_tup[1]), offset_tup[1])
    sf_list1_     = mult_tuplelist(sf_list1, sf_tup[0])
    sf_list2_     = mult_tuplelist(sf_list2, sf_tup[1])

    offset_listB = offset_list1_ + offset_list2_
    sf_listB     = sf_list1_ + sf_list2_
    #offset_listB, sf_listB = combine_offset_lists([offset_list1,
    #offset_list2], [sf_list1, sf_list2], offset_tup, sf_tup)
    return imgB, offset_listB, sf_listB


def stack_multi_images2(multiimg_list, offsets_list, sfs_list, vert=True, modifysize=True):
    r"""
    Args:
        multiimg_list (list):
        offset_lists (?):
        sfs_list (?):
        vert (bool):

    Returns:
        tuple: (stacked_img, stacked_img, stacked_sfs)

    CommandLine:
        python -m vtool_ibeis.image --test-stack_multi_images2 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> img_list = testdata_imglist()
        >>> img_stack1, offset_list1, sf_list1 = stack_image_list(img_list[::-1], vert=True, return_info=True, modifysize=True)
        >>> img_stack2, offset_list2, sf_list2 = stack_image_list(img_list, vert=True, return_info=True, modifysize=True)
        >>> img_stack3, offset_list3, sf_list3 = stack_image_list(img_list, vert=True, return_info=True, modifysize=False)
        >>> multiimg_list = [img_stack1, img_stack2, img_stack3]
        >>> offsets_list  = [offset_list1, offset_list2, offset_list3]
        >>> sfs_list      = [sf_list1, sf_list2, sf_list3]
        >>> vert = False
        >>> tup = stack_multi_images2(multiimg_list, offsets_list, sfs_list, vert)
        >>> (stacked_img, stacked_offsets, stacked_sfs) = tup
        >>> result = ut.remove_doublspaces(ub.repr2(np.array(stacked_offsets).T, precision=2, with_dtype=True, linewidth=10000)).replace(' ,', ',')
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(stacked_img)
        >>> wh_list = np.array([vt.get_size(img) for img in img_list[::-1] + img_list + img_list])
        >>> wh_list_ = wh_list * stacked_sfs
        >>> for offset, wh, color in zip(stacked_offsets, wh_list_, pt.distinct_colors(len(stacked_offsets))):
        ...    pt.draw_bbox((offset[0], offset[1], wh[0], wh[1]), bbox_color=color)
        >>> ut.show_if_requested()
        np.array([[ 0., 0., 0., 0., 0., 512., 512., 512., 512., 512., 1024., 1024., 1024., 1024., 1024. ],
         [ 0., 512.12, 1024.25, 1827., 2339., 0., 427., 939., 1742., 2254., 0., 373.18, 1137.45, 2073.38, 2670.47]], dtype=np.float64)


    """
    stacked_img, offset_tups, sf_tups = stack_image_list(multiimg_list,
                                                         return_sf=True,
                                                         return_offset=True,
                                                         vert=vert,
                                                         modifysize=modifysize)
    stacked_offsets, stacked_sfs = combine_offset_lists(offsets_list, sfs_list,
                                                        offset_tups, sf_tups)
    return stacked_img, stacked_offsets, stacked_sfs


def combine_offset_lists(offsets_list, sfs_list, offset_tups, sf_tups):
    """ Helper for stacking """
    # combine the offsets
    import operator
    from six.moves import reduce

    assert len(offsets_list) == len(offset_tups)
    assert len(sfs_list) == len(sf_tups)
    assert len(sfs_list) == len(offsets_list)

    def mult_tuplelist(tuple_list, scale_xy):
        return [(tup[0] * scale_xy[0], tup[1] * scale_xy[1]) for tup in tuple_list]

    def add_tuplelist(tuple_list, offset_xy):
        return [(tup[0] + offset_xy[0], tup[1] + offset_xy[1]) for tup in tuple_list]

    offset_lists_ = [
        add_tuplelist(mult_tuplelist(offsets, sf_tups[ix]), offset_tups[ix])
        for ix, offsets in enumerate(offsets_list)
    ]
    sf_lists_ = [
        mult_tuplelist(sfs, sf_tups[ix])
        for ix, sfs in enumerate(sfs_list)
    ]

    offset_listB = reduce(operator.add, offset_lists_)
    sf_listB = reduce(operator.add, sf_lists_)
    return offset_listB, sf_listB


def stack_square_images(img_list, return_info=False, **kwargs):
    r"""
    Args:
        img_list (list):

    Returns:
        ndarray:

    CommandLine:
        python -m vtool_ibeis.image --test-stack_square_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> img_list = '?'
        >>> result = stack_square_images(img_list)
        >>> print(result)
    """
    if len(img_list) == 0:
        raise IndexError('no images to stack')
    if len(img_list) == 1:
        return img_list[0]
    num_vert = int(np.ceil(np.sqrt(len(img_list))))
    num_horiz = int(np.ceil(len(img_list) / float(num_vert)))
    stacked_info_list = [
        stack_image_list(imgs, vert=True, return_offset=True, return_sf=True, **kwargs)
        for imgs in list(ut.ichunks(img_list, num_horiz))
    ]
    vert_patches = ut.get_list_column(stacked_info_list, 0)
    bigpatch, bigoffsets, bigsfs = stack_image_list(vert_patches, vert=False,
                                                    return_offset=True,
                                                    return_sf=True, **kwargs)
    if return_info:
        sfs_list = ut.get_list_column(stacked_info_list, 2)
        offsets_list = ut.get_list_column(stacked_info_list, 1)
        offset_listB, sf_listB = combine_offset_lists(offsets_list, sfs_list, bigoffsets, bigsfs)
        return bigpatch, offset_listB, sf_listB
    else:
        return bigpatch


def stack_image_list(img_list, return_offset=False, return_sf=False, return_info=False, **kwargs):
    r"""

    CommandLine:
        python -m vtool_ibeis.image --test-stack_image_list --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # build test data
        >>> img_list = testdata_imglist()
        >>> vert = False
        >>> return_offset = True
        >>> modifysize = True
        >>> return_sf=True
        >>> kwargs = dict(modifysize=modifysize, vert=vert, use_larger=False)
        >>> # execute function
        >>> imgB, offset_list, sf_list = stack_image_list(img_list, return_offset=return_offset, return_sf=return_sf, **kwargs)
        >>> # verify results
        >>> result = ub.repr2(np.array(offset_list).T, precision=2, with_dtype=True)
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgB)
        >>> wh_list = np.array([vt.get_size(img) for img in img_list])
        >>> wh_list_ = wh_list * sf_list
        >>> for offset, wh, color in zip(offset_list, wh_list_, pt.distinct_colors(len(offset_list))):
        ...    pt.draw_bbox((offset[0], offset[1], wh[0], wh[1]), bbox_color=color)
        >>> pt.show_if_requested()
        >>> #wh1 = img1.shape[0:2][::-1]
        >>> #wh2 = img2.shape[0:2][::-1]
        >>> #pt.draw_bbox((0, 0) + wh1, bbox_color=(1, 0, 0))
        >>> #pt.draw_bbox((woff, hoff) + wh2, bbox_color=(0, 1, 0))
        np.array([[   0.  ,   76.96,  141.08,  181.87,  246.  ],
                  [   0.  ,    0.  ,    0.  ,    0.  ,    0.  ]], dtype=np.float64)
    """
    if return_info:
        return_sf = True
        return_offset = True
    if len(img_list) == 0:
        imgB = None
        offset_list = []
        sf_list = []
    else:
        imgB = img_list[0]
        offset_list = [(0, 0)]
        sf_list = np.full((len(img_list), 2), np.nan)
        #sf_list = [(1., 1.)]
        sf_list[0, :] = (1, 1)
        #sf_list = np
        for count, img2 in enumerate(img_list[1:], start=1):
            out_ = stack_images(imgB, img2, return_sf=return_sf, **kwargs)
            if return_sf:
                imgB, offset_tup, sf_tup = out_
                offset2 = offset_tup[1]
                # need to modify scales of previous images
                sf1, sf2 = sf_tup
                #sf_list = [np.multiply(sf, sf1) for sf in sf_list]
                offset_list = [(sf1[0] * offset[0], sf1[1] * offset[1]) for offset in offset_list]
                sf_list[:count, :] *= sf1
                sf_list[count, :] = sf2
            else:
                imgB, woff, hoff = out_
                offset2 = (woff, hoff)
            offset_list.append(offset2)
    if return_offset:
        if return_sf:
            return imgB, offset_list, sf_list
        else:
            return imgB, offset_list
    else:
        return imgB


def embed_channels(img, input_channels=(0,), nchannels=3, fill=0):
    r"""

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        input_channels (tuple): (default = (0,))
        nchannels (int): (default = 3)

    CommandLine:
        python -m vtool_ibeis.image embed_channels --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # Embed a (N,M,2) image into an (N,M,3) image
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath).T[1:3].T
        >>> input_channels = (1, 2)
        >>> nchannels = 3
        >>> newimg = embed_channels(img, input_channels, nchannels)
        >>> assert newimg.shape[-1] == 3
        >>> assert np.all(newimg[:, :, input_channels] == img)
    """
    import vtool_ibeis as vt
    new_shape = img.shape[0:2] + (nchannels,)
    if not isinstance(fill, tuple):
        fill = (fill,)
    newimg = np.empty(new_shape, dtype=img.dtype)
    fill_dims = np.setdiff1d(np.arange(nchannels), input_channels)
    for dim, val in zip(fill_dims, fill):
        newimg[:, :, dim] = val
    # newimg[:, :, tuple(fill_dims.tolist())] = vt.atleast_nd([fill], 3, True)
    newimg[:, :, input_channels] = vt.atleast_nd(img, 3)
    return newimg


def ensure_4channel(img):
    assert len(img.shape) == 3

    if len(img.shape) == 3 and img.shape[2] == 4:
        img_alpha = img
        return img_alpha

    h, w = img.shape[0:2]
    if img.dtype.kind in {'i', 'u'}:
        alpha = np.full((h, w, 1), fill_value=255, dtype=img.dtype)
    elif img.dtype.kind == 'f':
        alpha = np.full((h, w, 1), fill_value=1, dtype=img.dtype)
    else:
        raise NotImplementedError('kind={}'.format(img.dtype.kind))

    if img.shape[2] == 1:
        img_alpha = np.dstack([img, img, img, alpha])
    elif img.shape[2] == 3:
        img_alpha = np.dstack([img, alpha])
    else:
        raise NotImplementedError('shape={}'.format(img.shape))
    # print('img_alpha.shape = {!r}'.format(img_alpha.shape))

    return img_alpha


def ensure_3channel(patch):
    r"""
    DEPRICATE IN FAVOR OF atleast_3channels?

    Ensures that there are 3 channels in the image

    Args:
        patch (ndarray[N, M, ...]): the image

    Returns:
        ndarray: [N, M, 3]

    CommandLine:
        python -m vtool_ibeis.image --exec-ensure_3channel --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> patch1 = vt.imread(ut.grab_test_imgpath('astro.png'))[0:512, 0:500, :]
        >>> patch2 = vt.imread(ut.grab_test_imgpath('ada.jpg'))[:, :, 0:1]
        >>> patch3 = vt.imread(ut.grab_test_imgpath('jeff.png'))[0:390, 0:400, 0]
        >>> res1 = ensure_3channel(patch1)
        >>> res2 = ensure_3channel(patch2)
        >>> res3 = ensure_3channel(patch3)
        >>> assert res1.shape[0:2] == patch1.shape[0:2], 'failed test1'
        >>> assert res2.shape[0:2] == patch2.shape[0:2], 'failed test2'
        >>> assert res3.shape[0:2] == patch3.shape[0:2], 'failed test3'
        >>> assert res1.shape[-1] == 3
        >>> assert res2.shape[-1] == 3
        >>> assert res3.shape[-1] == 3
    """
    # TODO: should this use atleast_nd as a subroutine?
    # res = vt.atleast_nd(patch, 3)
    # if res.shape[-1] == 1:
    #     res = np.tile(res, 3)
    # import utool
    # utool.embed()
    if len(patch.shape) == 2:
        res = np.tile(patch[:, :, None], 3)
    elif len(patch.shape) == 3 and patch.shape[-1] == 1:
        res = np.tile(patch, 3)
    else:
        res = patch.copy()
    return res


def infer_vert(img1, img2, vert):
    """ which is the better stack dimension """
    (h1, w1) = img1.shape[0: 2]  # get chip dimensions
    (h2, w2) = img2.shape[0: 2]
    woff, hoff = 0, 0
    vert_wh  = max(w1, w2), h1 + h2
    horiz_wh = w1 + w2, max(h1, h2)
    if vert is None:
        # Display the orientation with the better (closer to 1) aspect ratio
        vert_ar  = max(vert_wh) / min(vert_wh)
        horiz_ar = max(horiz_wh) / min(horiz_wh)
        vert = vert_ar < horiz_ar
    if vert:
        wB, hB = vert_wh
        hoff = h1
    else:
        wB, hB = horiz_wh
        woff = w1
    return vert, h1, h2, w1, w2, wB, hB, woff, hoff


def stack_images(img1, img2, vert=None, modifysize=False, return_sf=False,
                 use_larger=True, interpolation=None, white_background=False,
                 overlap=0):
    r"""

    Args:
        img1 (ndarray[uint8_t, ndim=2]):  image data
        img2 (ndarray[uint8_t, ndim=2]):  image data

    CommandLine:
        python -m vtool_ibeis.image --test-stack_images --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # build test data
        >>> img1 = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> img2 = vt.imread(ut.grab_test_imgpath('astro.png'))
        >>> vert = True
        >>> modifysize = False
        >>> # execute function
        >>> return_sf = True
        >>> #(imgB, woff, hoff) = stack_images(img1, img2, vert, modifysize, return_sf=return_sf)
        >>> overlap = 100
        >>> imgB, offset2, sf_tup = stack_images(img1, img2, vert, modifysize,
        >>>                                      return_sf=return_sf,
        >>>                                      overlap=overlap)
        >>> woff, hoff = offset2
        >>> # verify results
        >>> result = str((imgB.shape, woff, hoff))
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(imgB)
        >>> wh1 = np.multiply(vt.get_size(img1), sf_tup[0])
        >>> wh2 = np.multiply(vt.get_size(img2), sf_tup[1])
        >>> pt.draw_bbox((0, 0, wh1[0], wh1[1]), bbox_color=(1, 0, 0))
        >>> pt.draw_bbox((woff[1], hoff[1], wh2[0], wh2[0]), bbox_color=(0, 1, 0))
        >>> pt.show_if_requested()
        ((662, 512, 3), (0.0, 0.0), (0, 150))
    """
    import operator
    import vtool_ibeis as vt
    interpolation = _rectify_interpolation(interpolation, default=cv2.INTER_NEAREST)
    # TODO: move this to the same place I'm doing the color gradient
    nChannels1 = vt.get_num_channels(img1)
    nChannels2 = vt.get_num_channels(img2)
    if nChannels1 == 1 and nChannels2 == 3:
        img1 = vt.atleast_3channels(img1, copy=False)
    if nChannels1 == 3 and nChannels2 == 1:
        img2 = vt.atleast_3channels(img2, copy=False)
    nChannels1 = vt.get_num_channels(img1)
    nChannels2 = vt.get_num_channels(img2)
    assert nChannels1 == nChannels2
    # TODO: allow for some overlap / blending of the images
    vert, h1, h2, w1, w2, wB, hB, woff, hoff = infer_vert(img1, img2, vert)
    if overlap:
        if vert:
            hB -= overlap
        else:
            wB -= overlap
    # Rectify both images to they are the same dimension
    if modifysize:
        side_index = 1 if vert else 0
        # Compre the lengths of the width and height
        (length1, length2) = (img1.shape[side_index], img2.shape[side_index])
        comp_ = (operator.lt if use_larger else operator.gt)
        if comp_(length1, length2):
            tonew_sf2 = (1., 1.)
            scale = length2 / length1
            dsize, tonew_sf1 = vt.get_round_scaled_dsize(vt.get_size(img1), scale)
            img1 = cv2.resize(img1, dsize, interpolation=interpolation)
        elif comp_(length2, length1):
            tonew_sf1 = (1., 1.)
            scale = length1 / length2
            dsize, tonew_sf2 = vt.get_round_scaled_dsize(vt.get_size(img2), scale)
            img2 = cv2.resize(img2, dsize, interpolation=interpolation)
        else:
            tonew_sf1 = (1., 1.)
            tonew_sf2 = (1., 1.)
        vert, h1, h2, w1, w2, wB, hB, woff, hoff = infer_vert(img1, img2, vert)
    else:
        tonew_sf1 = (1., 1.)
        tonew_sf2 = (1., 1.)
    # Do image concatentation
    dtype = img1.dtype
    assert img1.dtype == img2.dtype, (
        'img1.dtype=%r, img2.dtype=%r' % (img1.dtype, img2.dtype))

    # if False:
    #     if nChannels1 == 3 or len(img1.shape) > 2:
    #         # Allocate new image for both
    #         imgB = np.zeros((hB, wB, nChannels1), dtype)
    #         # Insert the images
    #         imgB[0:h1, 0:w1, :] = img1
    #         imgB[hoff:(hoff + h2), woff:(woff + w2), :] = img2
    #     elif nChannels1 == 1:
    #         # Allocate new image for both
    #         imgB = np.zeros((hB, wB), dtype)
    #         # Insert the images
    #         imgB[0:h1, 0:w1] = img1
    #         imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
    # else:
    if nChannels1 == 3 or len(img1.shape) > 2:
        newshape = (hB, wB, nChannels1)
    else:
        newshape = (hB, wB)
    # Allocate new image for both
    imgB = np.zeros(newshape, dtype=dtype)

    if white_background:
        if dtype == np.uint8:
            imgB += 255
        else:
            imgB += 1.0

    if overlap:
        if vert:
            hoff -= overlap
        else:
            woff -= overlap
        # Insert the images
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
        # Blend the overlapping part
        if vert:
            part1 = img1[-overlap:, :]
            part2 = imgB[hoff:(hoff + overlap), 0:w1]
            alpha = vt.gradient_fill(part1.shape[0:2], vert=vert)
            imgB[hoff:(hoff + overlap), 0:w1] = vt.blend_images_average(part1, part2, alpha=alpha)
        else:
            part1 = img1[:, -overlap:]
            part2 = imgB[0:h1, woff:(woff + overlap)]
            alpha = vt.gradient_fill(part1.shape[0:2], vert=vert)
            imgB[0:h1, woff:(woff + overlap)] = vt.blend_images_average(part1, part2, alpha=alpha)
    else:
        # Insert the images
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2

    # return
    if return_sf:
        offset1 = (0.0, 0.0)
        offset2 = (woff, hoff)
        offset_tup = (offset1, offset2)
        sf_tup = (tonew_sf1, tonew_sf2)
        return imgB, offset_tup, sf_tup
    else:
        return imgB, woff, hoff


def stack_image_recurse(img_list1, img_list2=None, vert=True, modifysize=False,
                        return_offsets=False, interpolation=None):
    r"""
    TODO: return offsets as well

    Args:
        img_list1 (list):
        img_list2 (list):
        vert (bool):

    Returns:
        ndarray: None

    CommandLine:
        python -m vtool_ibeis.image --test-stack_image_recurse --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> # build test data
        >>> img1 = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> img2 = vt.imread(ut.grab_test_imgpath('astro.png'))
        >>> img3 = vt.imread(ut.grab_test_imgpath('ada.jpg'))
        >>> img4 = vt.imread(ut.grab_test_imgpath('jeff.png'))
        >>> img5 = vt.imread(ut.grab_test_imgpath('star.png'))
        >>> img_list1 = [img1, img2, img3, img4, img5]
        >>> img_list2 = None
        >>> vert = True
        >>> # execute function
        >>> imgB = stack_image_recurse(img_list1, img_list2, vert)
        >>> # verify results
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> imshow(imgB)
        >>> #wh1 = img1.shape[0:2][::-1]
        >>> #wh2 = img2.shape[0:2][::-1]
        >>> #pt.draw_bbox((0, 0) + wh1, bbox_color=(1, 0, 0))
        >>> #pt.draw_bbox((woff, hoff) + wh2, bbox_color=(0, 1, 0))
        >>> pt.show_if_requested()
    """
    interpolation = _rectify_interpolation(interpolation, default=cv2.INTER_NEAREST)
    if img_list2 is None:
        # Initialization and error checking
        if len(img_list1) == 0:
            return None
        if len(img_list1) == 1:
            return img_list1[0]
        return stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=vert,
                                   modifysize=modifysize,
                                   interpolation=interpolation)
    if len(img_list1) == 1:
        # Left base case
        img1 = img_list1[0]
    else:
        # Left recurse
        img1 = stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=not
                                   vert, modifysize=modifysize,
                                   interpolation=interpolation)
    if len(img_list2) == 1:
        # Right base case
        img2 = img_list2[0]
    else:
        # Right Recurse
        img2 = stack_image_recurse(img_list2[0::2], img_list2[1::2], vert=not
                                   vert, modifysize=modifysize,
                                   interpolation=interpolation)
    if return_offsets:
        raise NotImplementedError('finishme')
        #imgB, offset_list, sf_list = stack_multi_images(img1, img2,
        #offset_list1, sf_list1, offset_list2, sf_list2, vert=vert)
    else:
        imgB, offset_tup, sf_tup = stack_images(img1, img2, vert=vert,
                                                return_sf=True,
                                                modifysize=modifysize,
                                                interpolation=interpolation)
        (woff, hoff) = offset_tup[1]
    return imgB


# /STACK IMAGES STUFF


def filterflags_valid_images(gpaths, valid_formats=None,
                             invalid_formats=None, verbose=True):
    r"""
    Flags images with a format that disagrees with its extension

    Args:
        gpaths (list): list of image paths
        valid_formats (None): (default = None)
        invalid_formats (None): (default = None)
        verbose (bool):  verbosity flag(default = True)

    Returns:
        list: isvalid_flags

    CommandLine:
        python -m vtool_ibeis.image filterflags_valid_images --show

    Notes:
        An MPO (Multi Picture Object) file is a stereoscopic image and contains
        two JPG images side-by-side, and allows them to be viewed as a single
        3D image.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.image import *  # NOQA
        >>> gpaths = [ut.grab_test_imgpath('carl.jpg'),
        >>>           ut.grab_test_imgpath('astro.png')]
        >>> flags = filterflags_valid_images(gpaths)
        >>> assert all(flags)
    """
    from PIL import Image
    from os.path import splitext
    #import operator
    #import itertools as it
    # These are exact aliases
    img_format_alias_dict = {
        'JPG': 'JPEG',
        'TIF': 'TIFF',
    }
    # These aliases are not exact but generally fine
    #acceptable_alias = {
    #    'MPO': 'JPEG'
    #}
    def get_image_format_from_extension(gpath):
        gname, ext = splitext(gpath)
        ext_format = ext[1:].upper()
        ext_format = img_format_alias_dict.get(ext_format, ext_format)
        return ext_format

    def get_image_format_from_pil(gpath):
        try:
            pil_image = Image.open(gpath)
            pil_format = pil_image.format
        except IOError:
            pil_format = None
        #if pil_format == 'MPO':
        #    print(pil_image.n_frames)
        return pil_format

    #def read_frames(gpath):
    #    from PIL import Image, ImageSequence
    #    import vtool_ibeis as vt
    #    import cv2
    #    #pil_image.n_frames
    #    pil_image = Image.open(gpath)
    #    sequence = []
    #    for frame in ImageSequence.Iterator(pil_image):
    #        print('frame = %r' % (frame,))
    #        #img = np.asarray(frame)
    #        rgb_pil = frame.convert('RGB')
    #        img = np.array(rgb_pil)
    #        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #        sequence.append(img)
    #    stack = vt.stack_square_images(sequence)
    #    import plottool_ibeis as pt
    #    pt.qt4ensure()
    #    pt.imshow(stack)
    #    ## btyedata = np.asarray(bytearray(contents), dtype=np.uint8)
    #    #print('frame = %r' % (frame,))
    #    #frame.save("frame%d.png" % index)
    #    #index = index + 1
    #    pass

    def check_agrees(ext_format, pil_format):
        #pil_format_ = acceptable_alias.get(pil_format, pil_format)
        return pil_format == ext_format

    pil_foramt_list = [
        get_image_format_from_pil(gpath)
        for gpath in ut.ProgIter(gpaths, lbl='check image pil-format',
                                 enabled=verbose)
    ]
    ext_format_list = [
        get_image_format_from_extension(gpath)
        for gpath in ut.ProgIter(gpaths, lbl='check image ext-format',
                                 enabled=verbose)
    ]
    #agree_flags = list(it.starmap(operator.eq, zip(ext_format_list,
    #                                                pil_foramt_list)))
    agree_flags = [check_agrees(e, p) for e, p, in zip(ext_format_list,
                                                       pil_foramt_list)]
    valid_flags = agree_flags
    if valid_formats is not None:
        # explicitly mark valids
        valid_flags = ut.and_lists(
            valid_flags,
            [format_ in valid_formats for format_ in ext_format_list],
            [format_ in valid_formats for format_ in pil_foramt_list],
        )
    if invalid_formats is not None:
        invalid_fmt_flags = ut.or_lists(
            [format_ in invalid_formats for format_ in ext_format_list],
            [format_ in invalid_formats for format_ in pil_foramt_list],
        )
        valid_flags = ut.and_lists(
            valid_flags,
            ut.not_list(invalid_fmt_flags))
    if verbose > 0:
        # Inspect invalid items
        invalid_flags = ut.not_list(valid_flags)

        fmt_list = list(zip(ext_format_list, pil_foramt_list))
        invalid_fmt_list = ut.compress(fmt_list, invalid_flags)
        invalid_fmt_hist = ut.dict_hist(invalid_fmt_list)
        print('The following {(ext,pil): count} formats are invalid')
        print(ut.repr3(invalid_fmt_hist))
        print('Total Invalid Files %r' % (sum(invalid_fmt_hist.values()),))

        # Inspect valid items

        valid_fmt_list = ut.compress(fmt_list, valid_flags)
        valid_fmt_hist = ut.dict_hist(valid_fmt_list)
        print('The following {(ext,pil): count} formats are valid')
        print(ut.repr3(valid_fmt_hist))
        print('Total Valid Files %r' % (sum(valid_fmt_hist.values()),))

        if invalid_formats is not None:
            invalid_fmt_flags

        if verbose > 1:
            num_examples = 3
            print('Examples of invalid files:')
            invalid_gpaths = ut.compress(gpaths, invalid_flags)
            grouped_invalids = ut.group_items(invalid_gpaths, invalid_fmt_list)
            for key in invalid_fmt_hist.keys():
                val = grouped_invalids[key]
                print(key)
                print(ut.indentjoin(val[0:num_examples])[1:])
            print('\nExamples of valid files:')
            valid_gpaths = ut.compress(gpaths, valid_flags)
            grouped_valids = ut.group_items(valid_gpaths, valid_fmt_list)
            for key in valid_fmt_hist.keys():
                val = grouped_valids[key]
                print(key)
                print(ut.indentjoin(val[0:num_examples])[1:])
    return valid_flags


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool_ibeis.image
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
