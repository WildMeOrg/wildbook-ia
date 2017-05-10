# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from os.path import splitext, basename
import warnings  # NOQA
import vtool.exif as vtexif
import utool as ut
#import numpy as np  # NOQA
#import hashlib
#import uuid
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
    # (mainly the orientation tag)
    lat, lon = vtexif.get_lat_lon(exif_dict)
    orient = vtexif.get_orientation(exif_dict, on_error='warn')
    time = vtexif.get_unixtime(exif_dict)
    return time, lat, lon, orient


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
def parse_imageinfo(gpath):
    """ Worker function: gpath must be in UNIX-PATH format!

    Args:
        tup (tuple): a tuple or one argument
            (so the function can be parallelized easily)
            (here it is just gpath, no tuple, sorry for confusion)

    Returns:
        tuple: param_tup -
            if successful returns a tuple of image parameters which are values
            for SQL columns on else returns None

    CommandLine:
        python -m ibeis.algo.preproc.preproc_image --exec-parse_imageinfo

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_image import *  # NOQA
        >>> gpath = ('/media/raid/work/lynx/_ibsdb/images/f6c84c6d-55ca-fd02-d0b4-1c7c9c27c894.jpg')
        >>> param_tup = parse_imageinfo(tup)
        >>> result = ('param_tup = %s' % (str(param_tup),))
        >>> print(result)
    """
    # Parse arguments from tuple
    #print('[ginfo] gpath=%r' % gpath)
    # Try to open the image
    from PIL import Image  # NOQA
    from os.path import isabs
    import six
    import tempfile
    if six.PY2:
        import urlparse
        urlsplit = urlparse.urlsplit
    else:
        import urllib
        urlsplit = urllib.parse.urlsplit

    url_protos = ['https://', 'http://']
    s3_proto = ['s3://']
    valid_protos = s3_proto + url_protos

    def isproto(gpath, valid_protos):
        return any(gpath.startswith(proto) for proto in valid_protos)

    def islocal(gpath):
        return not (isabs(gpath) and isproto(gpath, valid_protos))

    with warnings.catch_warnings(record=True) as w:
        try:
            temp_filepath = None
            if isproto(gpath, valid_protos):
                # Ensure that the Unicode string is properly encoded for web requests
                gpath_ = urlsplit(gpath)
                gpath_path = six.moves.urllib.parse.quote(gpath_.path.encode('utf8'))
                gpath_ = gpath_._replace(path=gpath_path)
                gpath = gpath_.geturl()
                suffix = '.%s' % (basename(gpath), )
                temp_file, temp_filepath = tempfile.mkstemp(suffix=suffix)
                print('[preproc] Caching remote file to temporary file %r' % (temp_filepath, ))

                if isproto(gpath, s3_proto):
                    s3_dict = ut.s3_str_decode_to_dict(gpath)
                    ut.grab_s3_contents(temp_filepath, **s3_dict)
                if isproto(gpath, url_protos):
                    six.moves.urllib.request.urlretrieve(gpath, filename=temp_filepath)
                gpath = temp_filepath

            # Open image with Exif support
            pil_img = Image.open(gpath, 'r')  # NOQA
        except IOError as ex:
            # ut.embed()
            print('[preproc] IOError: %s' % (str(ex),))
            return None
        if len(w) > 0:
            for warn in w:
                warnings.showwarning(warn.message, warn.category,
                                     warn.filename, warn.lineno, warn.file,
                                     warn.line)
                #warnstr = warnings.formatwarning
                #print(warnstr)
            print('Warnings issued by %r' % (gpath,))
    # Parse out the data
    width, height  = pil_img.size         # Read width, height
    time, lat, lon, orient = parse_exif(pil_img)  # Read exif tags
    if orient in [6, 8]:
        width, height = height, width
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
        gpath,
        orig_gname,
        #orig_gpath,
        ext,
        width,
        height,
        time,
        lat,
        lon,
        orient,
        notes
    )

    if temp_filepath is not None:
        os.unlink(temp_filepath)
    #print('[ginfo] %r %r' % (image_uuid, orig_gname))
    return param_tup


@profile
def add_images_params_gen(gpath_list, **kwargs):
    """
    generates values for add_images sqlcommands asychronously

    Args:
        gpath_list (list):

    Kwargs:
        ordered, force_serial, chunksize, prog, verbose, quiet, nTasks, freq,
        adjust

    Returns:
        generator: params_gen

    CommandLine:
        python -m ibeis.algo.preproc.preproc_image --exec-add_images_params_gen

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_image import *   # NOQA
        >>> from vtool.tests import grabdata
        >>> gpath_list = grabdata.get_test_gpaths(ndata=3) + ['doesnotexist.jpg']
        >>> params_list = list(add_images_params_gen(gpath_list))
        >>> assert str(params_list[0][0]) == '66ec193a-1619-b3b6-216d-1784b4833b61', 'UUID gen method changed'
        >>> assert str(params_list[0][3]) == 'easy1.JPG', 'orig name is different'
        >>> assert params_list[3] is None
    """
    #preproc_args = [(gpath, kwargs) for gpath in gpath_list]
    #print('[about to parse]: gpath_list=%r' % (gpath_list,))
    params_gen = ut.generate(parse_imageinfo, gpath_list, adjust=True,
                             force_serial=True, **kwargs)
    return params_gen


def on_delete(ibs, featweight_rowid_list, qreq_=None):
    print('Warning: Not Implemented')


if __name__ == '__main__':
    """
    python -m ibeis.algo.preproc.preproc_image
    python -m ibeis.algo.preproc.preproc_image --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
