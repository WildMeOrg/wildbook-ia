# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from os.path import join, exists
import zipfile
import time
from PIL import Image
import cStringIO as StringIO
from flask import request, current_app, send_file
from ibeis.control import controller_inject
import utool as ut
from ibeis.web import appfuncs as appf
print, rrr, profile = ut.inject2(__name__, '[apis]')


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


# Special function that is a route only to ignore the JSON response, but is
# actually (and should be) an API call
@register_route('/api/image/src/<gid>/', methods=['GET'], __api_prefix_check__=False)
def image_src_api(gid=None, thumbnail=False, fresh=False, **kwargs):
    r"""
    Returns the image file of image <gid>

    RESTful:
        Method: GET
        URL:    /api/image/src/<gid>/
    """
    thumbnail = thumbnail or 'thumbnail' in request.args or 'thumbnail' in request.form
    ibs = current_app.ibs
    if thumbnail:
        gpath = ibs.get_image_thumbpath(gid, ensure_paths=True)
        fresh = fresh or 'fresh' in request.args or 'fresh' in request.form
        if fresh:
            import os
            os.remove(gpath)
            gpath = ibs.get_image_thumbpath(gid, ensure_paths=True)
    else:
        gpath = ibs.get_image_paths(gid)

    image = appf.open_oriented_image(gpath)
    image_pil = Image.fromarray(image)
    img_io = StringIO.StringIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    # return send_file(gpath, mimetype='application/unknown')


@register_api('/api/upload/image/', methods=['POST'])
def image_upload(cleanup=True, **kwargs):
    r"""
    Returns the gid for an uploaded image.

    Args:
        image (image binary): the POST variable containing the binary
            (multi-form) image data
        **kwargs: Arbitrary keyword arguments; the kwargs are passed down to
            the add_images function

    Returns:
        gid (rowids): gid corresponding to the image submitted.
            lexigraphical order.

    RESTful:
        Method: POST
        URL:    /api/image/
    """
    ibs = current_app.ibs
    print('request.files = %s' % (request.files,))

    filestore = request.files.get('image', None)
    if filestore is None:
        raise IOError('Image not given')

    uploads_path = ibs.get_uploadsdir()
    ut.ensuredir(uploads_path)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')

    modifier = 1
    upload_filename = 'upload_%s.png' % (current_time)
    while exists(upload_filename):
        upload_filename = 'upload_%s_%04d.png' % (current_time, modifier)
        modifier += 1

    upload_filepath = join(uploads_path, upload_filename)
    filestore.save(upload_filepath)

    gid_list = ibs.add_images([upload_filepath], **kwargs)
    gid = gid_list[0]

    if cleanup:
        ut.remove_dirs(upload_filepath)

    return gid


@register_api('/api/upload/zip/', methods=['POST'])
def image_upload_zip(**kwargs):
    r"""
    Returns the gid_list for image files submitted in a ZIP archive.  The image
    archive should be flat (no folders will be scanned for images) and must be smaller
    than 100 MB.  The archive can submit multiple images, ideally in JPEG format to save
    space.  Duplicate image uploads will result in the duplicate images receiving
    the same gid based on the hashed pixel values.

    Args:
        image_zip_archive (binary): the POST variable containing the binary
            (multi-form) image archive data
        **kwargs: Arbitrary keyword arguments; the kwargs are passed down to
            the add_images function

    Returns:
        gid_list (list if rowids): the list of gids corresponding to the images
            submitted.  The gids correspond to the image names sorted in
            lexigraphical order.

    RESTful:
        Method: POST
        URL:    /api/image/zip
    """
    ibs = current_app.ibs
    # Get image archive
    image_archive = request.files.get('image_zip_archive', None)
    if image_archive is None:
        raise IOError('Image archive not given')

    # If the directory already exists, delete it
    uploads_path = ibs.get_uploadsdir()
    ut.ensuredir(uploads_path)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')

    modifier = 1
    upload_path = '%s' % (current_time)
    while exists(upload_path):
        upload_path = '%s_%04d' % (current_time, modifier)
        modifier += 1

    upload_path = join(uploads_path, upload_path)
    ut.ensuredir(upload_path)

    # Extract the content
    try:
        with zipfile.ZipFile(image_archive, 'r') as zfile:
            zfile.extractall(upload_path)
    except Exception:
        ut.remove_dirs(upload_path)
        raise IOError('Image archive extracton failed')

    """
    test to ensure Directory and utool do the same thing

    from detecttools.directory import Directory
    upload_path = ut.truepath('~/Pictures')
    gpath_list1 = sorted(ut.list_images(upload_path, recursive=False, full=True))

    direct = Directory(upload_path, include_file_extensions='images', recursive=False)
    gpath_list = direct.files()
    gpath_list = sorted(gpath_list)

    assert gpath_list1 == gpath_list
    """

    gpath_list = sorted(ut.list_images(upload_path, recursive=False, full=True))
    #direct = Directory(upload_path, include_file_extensions='images', recursive=False)
    #gpath_list = direct.files()
    #gpath_list = sorted(gpath_list)
    gid_list = ibs.add_images(gpath_list, **kwargs)
    return gid_list


@register_api('/api/test/helloworld/', methods=['GET', 'POST', 'DELETE', 'PUT'])
def hello_world(*args, **kwargs):
    """
    CommandLine:
        python -m ibeis.web.app --exec-hello_world

    Example:
        >>> # SCRIPT
        >>> from ibeis.web.app import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web(browser=True, start_job_queue=False, url_suffix='/api/test/helloworld/')
    """
    print('------------------ HELLO WORLD ------------------')
    print('Args: %r' % (args,))
    print('Kwargs: %r' % (kwargs,))
    print('request.args: %r' % (request.args,))
    print('request.form: %r' % (request.form,))
    print('request.url; %r' % (request.url,))
    print('request.environ: %s' % (ut.repr3(request.environ),))
    print('request: %s' % (ut.repr3(request.__dict__),))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
