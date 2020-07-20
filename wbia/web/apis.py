# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from os.path import join, exists
import zipfile
import time
from io import BytesIO
from six.moves import cStringIO as StringIO
from flask import request, current_app, send_file
from wbia.control import controller_inject
from wbia.web import appfuncs as appf
import utool as ut
import vtool as vt
import uuid as uuid_module
import six
from wbia.web.app import PROMETHEUS

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


@register_api('/api/embed/', methods=['GET'])
def web_embed(*args, **kwargs):
    ibs = current_app.ibs  # NOQA

    if False:
        from wbia.algo.graph.state import POSTV

        payload = {
            'action': 'update_task_thresh',
            'task': 'match_state',
            'decision': POSTV,
            'value': 0.95,
        }

        for graph_uuid in current_app.GRAPH_CLIENT_DICT:
            graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_uuid, None)
            if graph_client is None:
                continue
            if len(graph_client.futures) > 0:
                continue
            future = graph_client.post(payload)  # NOQA
            # future.result()  # Guarantee that this has happened before calling refresh

    ut.embed()


# Special function that is a route only to ignore the JSON response, but is
# actually (and should be) an API call
@register_route(
    '/api/image/src/<rowid>/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def image_src_api(rowid=None, thumbnail=False, fresh=False, **kwargs):
    r"""
    Returns the image file of image <gid>

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
        ...     web_ibs.send_wbia_request('/api/image/src/1/', type_='get')
        >>> print(resp)

    RESTful:
        Method: GET
        URL:    /api/image/src/<rowid>/
    """
    from PIL import Image  # NOQA

    thumbnail = thumbnail or 'thumbnail' in request.args or 'thumbnail' in request.form
    ibs = current_app.ibs
    if thumbnail:
        gpath = ibs.get_image_thumbpath(rowid, ensure_paths=True)
        fresh = fresh or 'fresh' in request.args or 'fresh' in request.form
        if fresh:
            # import os
            # os.remove(gpath)
            ut.delete(gpath)
            gpath = ibs.get_image_thumbpath(rowid, ensure_paths=True)
    else:
        gpath = ibs.get_image_paths(rowid)

    # Load image
    assert gpath is not None, 'image path should not be None'
    image = vt.imread(gpath, orient='auto')
    image = appf.resize_via_web_parameters(image)
    image = image[:, :, ::-1]

    # Encode image
    image_pil = Image.fromarray(image)
    if six.PY2:
        img_io = StringIO()
    else:
        img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    # return send_file(gpath, mimetype='application/unknown')


# Special function that is a route only to ignore the JSON response, but is
# actually (and should be) an API call
@register_route(
    '/api/annot/src/<rowid>/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def annot_src_api(rowid=None, fresh=False, **kwargs):
    r"""
    Returns the image file of annot <aid>

    Example:
        >>> # WEB_DOCTEST
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
        ...     web_ibs.send_wbia_request('/api/annot/src/1/', type_='get')
        >>> print(resp)

    RESTful:
        Method: GET
        URL:    /api/annot/src/<rowid>/
    """
    from PIL import Image  # NOQA

    ibs = current_app.ibs
    gpath = ibs.get_annot_chip_fpath(rowid, ensure=True)

    # Load image
    assert gpath is not None, 'image path should not be None'
    image = vt.imread(gpath, orient='auto')
    image = appf.resize_via_web_parameters(image)
    image = image[:, :, ::-1]

    # Encode image
    image_pil = Image.fromarray(image)
    if six.PY2:
        img_io = StringIO()
    else:
        img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    # return send_file(gpath, mimetype='application/unknown')


# Special function that is a route only to ignore the JSON response, but is
# actually (and should be) an API call
@register_route(
    '/api/background/src/<rowid>/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def background_src_api(rowid=None, fresh=False, **kwargs):
    r"""
    Returns the image file of annot <aid>

    Example:
        >>> # WEB_DOCTEST
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
        ...     web_ibs.send_wbia_request('/api/background/src/1/', type_='get')
        >>> print(resp)

    RESTful:
        Method: GET
        URL:    /api/annot/src/<rowid>/
    """
    from PIL import Image  # NOQA

    ibs = current_app.ibs
    gpath = ibs.get_annot_probchip_fpath(rowid)

    # Load image
    assert gpath is not None, 'image path should not be None'
    image = vt.imread(gpath, orient='auto')
    image = appf.resize_via_web_parameters(image)
    image = image[:, :, ::-1]

    # Encode image
    image_pil = Image.fromarray(image)
    if six.PY2:
        img_io = StringIO()
    else:
        img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    # return send_file(gpath, mimetype='application/unknown')


# Special function that is a route only to ignore the JSON response, but is
# actually (and should be) an API call
@register_route(
    '/api/image/src/json/<uuid>/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def image_src_api_json(uuid=None, **kwargs):
    r"""
    Returns the image file of image <gid>

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
        ...     web_ibs.send_wbia_request('/api/image/src/json/1/', type_='get')
        >>> print(resp)

    RESTful:
        Method: GET
        URL:    /api/image/src/<gid>/
    """
    ibs = current_app.ibs
    try:
        if isinstance(uuid, six.string_types):
            uuid = uuid_module.UUID(uuid)
    except Exception:
        from wbia.control.controller_inject import translate_wbia_webreturn

        return translate_wbia_webreturn(
            None, success=False, code=500, message='Invalid image UUID'
        )
    gid = ibs.get_image_gids_from_uuid(uuid)
    return image_src_api(gid, **kwargs)


def _image_conv_feature(ibs, gid, model):
    model = model.lower()
    model_list = ['vgg16', 'vgg19', 'resnet50', 'inception_v3']
    assert model in model_list, 'model must be one of %s' % (model_list,)
    config = {'algo': model}

    gid_list = [gid]
    feature_list = ibs.depc_image.get_property(
        'features', gid_list, 'vector', config=config
    )
    feature = feature_list[0]
    byte_str = feature.tobytes()
    return byte_str


@register_api('/api/image/feature/<rowid>/', methods=['GET'])
def image_conv_feature_api(rowid=None, model='resnet50', **kwargs):
    r"""
    RESTful:
        Method: GET
        URL:    /api/image/feature/json/<uuid>/
    """
    ibs = current_app.ibs

    gid = rowid
    assert gid is not None
    return _image_conv_feature(ibs, gid, model)


@register_api('/api/image/feature/json/<uuid>/', methods=['GET'])
def image_conv_feature_api_json(uuid=None, model='resnet50', **kwargs):
    r"""
    RESTful:
        Method: GET
        URL:    /api/image/feature/json/<uuid>/
    """
    ibs = current_app.ibs

    try:
        if isinstance(uuid, six.string_types):
            uuid = uuid_module.UUID(uuid)
        assert uuid is not None
    except Exception:
        from wbia.control.controller_inject import translate_wbia_webreturn

        return translate_wbia_webreturn(
            None, success=False, code=500, message='Invalid image UUID'
        )
    gid = ibs.get_image_gids_from_uuid(uuid)
    return _image_conv_feature(ibs, gid, model)


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
        URL:    /api/upload/image/
    """
    ibs = current_app.ibs
    print('request.files = %s' % (request.files,))

    filestore = request.files.get('image', None)
    if filestore is None:
        raise controller_inject.WebMissingInput(
            'Missing required image parameter', 'image'
        )
        # raise IOError('Image not given')

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

    if cleanup and exists(upload_filepath):
        ut.delete(upload_filepath)

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

    from wbia.detecttools.directory import Directory
    upload_path = ut.truepath('~/Pictures')
    gpath_list1 = sorted(ut.list_images(upload_path, recursive=False, full=True))

    direct = Directory(upload_path, include_file_extensions='images', recursive=False)
    gpath_list = direct.files()
    gpath_list = sorted(gpath_list)

    assert gpath_list1 == gpath_list
    """

    gpath_list = sorted(ut.list_images(upload_path, recursive=False, full=True))
    # direct = Directory(upload_path, include_file_extensions='images', recursive=False)
    # gpath_list = direct.files()
    # gpath_list = sorted(gpath_list)
    gid_list = ibs.add_images(gpath_list, **kwargs)
    return gid_list


@register_api('/api/test/helloworld/', methods=['GET', 'POST', 'DELETE', 'PUT'])
def hello_world(*args, **kwargs):
    """
    CommandLine:
        python -m wbia.web.apis --exec-hello_world:0
        python -m wbia.web.apis --exec-hello_world:1

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> web_ibs = wbia.opendb_bg_web(browser=True, start_job_queue=False, url_suffix='/api/test/helloworld/?test0=0')  # start_job_queue=False)
        >>> print('web_ibs = %r' % (web_ibs,))
        >>> print('Server will run until control c')
        >>> web_ibs.terminate2()

    Example1:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.app import *  # NOQA
        >>> import wbia
        >>> import requests
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', start_job_queue=False, managed=True) as web_ibs:
        ...     web_port = ibs.get_web_port_via_scan()
        ...     if web_port is None:
        ...         raise ValueError('IA web server is not running on any expected port')
        ...     domain = 'http://127.0.0.1:%s' % (web_port, )
        ...     url = domain + '/api/test/helloworld/?test0=0'
        ...     payload = {
        ...         'test1' : 'test1',
        ...         'test2' : None,  # NOTICE test2 DOES NOT SHOW UP
        ...     }
        ...     resp = requests.post(url, data=payload)
        ...     print(resp)
    """
    print('+------------ HELLO WORLD ------------')
    print('Args: %r' % (args,))
    print('Kwargs: %r' % (kwargs,))
    print('request.args: %r' % (request.args,))
    print('request.form: %r' % (request.form,))
    print('request.url; %r' % (request.url,))
    print('request.environ: %s' % (ut.repr3(request.environ),))
    print('request: %s' % (ut.repr3(request.__dict__),))
    print('L____________ HELLO WORLD ____________')


@register_ibs_method
@register_api('/api/test/heartbeat/', methods=['GET', 'POST', 'DELETE', 'PUT'])
def heartbeat(ibs, *args, **kwargs):
    """
    """
    # ut.embed()

    if PROMETHEUS:
        ibs.prometheus_update()

    return True


@register_ibs_method
@register_api('/api/test/dataset/id/', methods=['GET', 'POST', 'DELETE', 'PUT'])
def api_test_datasets_id(ibs, dataset, *args, **kwargs):
    assert dataset in ['zebra', 'dolphin', 'humpback']

    if dataset in ['zebra']:
        qtext = "Grevy's Zebra Query"
        dtext = "Grevy's Zebra Database"
    elif dataset in ['dolphin']:
        qtext = 'Dorsal Query'
        dtext = 'Dorsal Database'
    elif dataset in ['humpback']:
        qtext = 'Fluke Query'
        dtext = 'Fluke Database'

    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text([qtext, dtext])
    qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)

    response = {
        'query': qaid_list,
        'database': daid_list,
    }

    return response


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
