# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from PIL import Image
from os.path import splitext, basename
#import numpy as np  # NOQA
import warnings  # NOQA
#import hashlib
#import uuid
import vtool.exif as vtexif
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[preproc_img]', DEBUG=False)


#@profile
def parse_exif(pil_img):
    """ Image EXIF helper

    Cyth::
        cdef:
            Image pil_img
            dict exif_dict
            long lat
            long lon
            long exiftime
    """
    exif_dict = vtexif.get_exif_dict(pil_img)
    # TODO: More tags
    lat, lon = vtexif.get_lat_lon(exif_dict)
    time = vtexif.get_unixtime(exif_dict)
    return time, lat, lon


def get_standard_ext(gpath):
    """ Returns standardized image extension

    Cyth::
        cdef:
            str gpath
            str ext

    """
    ext = splitext(gpath)[1].lower()
    return '.jpg' if ext == '.jpeg' else ext


@profile
def parse_imageinfo(tup):
    """ Worker function: gpath must be in UNIX-PATH format!

    Input:
        a tuple of arguments (so the function can be parallelized easily)

    Returns:
        if successful returns a tuple of image parameters which are values for
        SQL columns on else returns None

    Cyth::

        cdef:
            str gpath
            Image pil_img
            str orig_gname
            str ext
            long width
            long height
            long time
            long lat
            long lon
            str notes

    """
    # Parse arguments from tuple
    gpath = tup
    #print('[ginfo] gpath=%r' % gpath)
    # Try to open the image
    with warnings.catch_warnings(record=True) as w:
        try:
            pil_img = Image.open(gpath, 'r')  # Open PIL Image
        except IOError as ex:
            print('[preproc] IOError: %s' % (str(ex),))
            return None
        if len(w) > 0:
            for warn in w:
                warnings.showwarning(warn.message, warn.category, warn.filename, warn.lineno, warn.file, warn.line)
                #warnstr = warnings.formatwarning(warn.message, warn.category, warn.filename, warn.lineno, warn.line)
                #print(warnstr)
            print('Warnings issued by %r' % (gpath,))
    # Parse out the data
    width, height  = pil_img.size         # Read width, height
    time, lat, lon = parse_exif(pil_img)  # Read exif tags
    # We cannot use pixel data as libjpeg is not determenistic (even for reads!)
    image_uuid = ut.get_file_uuid(gpath)  # Read file ]-hash-> guid = gid
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


@profile
def add_images_params_gen(gpath_list, **kwargs):
    """
    generates values for add_images sqlcommands asychronously

    CommandLine:
        python -m ibeis.model.preproc.preproc_image --test-add_images_params_gen

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_image import *   # NOQA
        >>> from vtool.tests import grabdata
        >>> gpath_list = grabdata.get_test_gpaths(ndata=3) + ['doesnotexist.jpg']
        >>> params_list = list(add_images_params_gen(gpath_list))
        >>> assert str(params_list[0][0]) == '66ec193a-1619-b3b6-216d-1784b4833b61', 'UUID gen method changed'
        >>> assert str(params_list[0][2]) == 'easy1.JPG', 'orig name is different'
        >>> assert params_list[3] is None

    Cyth::
        cdef:
            list gpath_list
            dict kwargs
    """
    #preproc_args = [(gpath, kwargs) for gpath in gpath_list]
    #print('[about to parse]: gpath_list=%r' % (gpath_list,))
    params_gen = ut.generate(parse_imageinfo, gpath_list, **kwargs)
    return params_gen


def on_delete(ibs, featweight_rowid_list, qreq_=None):
    print('Warning: Not Implemented')


if __name__ == '__main__':
    """
    python -m ibeis.model.preproc.preproc_image
    python -m ibeis.model.preproc.preproc_image --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
