# -*- coding: utf-8 -*-
"""
Functions for images and encoutners that will be injected into an
IBEISController instance.


CommandLine:
    # Autogenerate ImageSet Functions
    # key should be the table name
    # the write flag makes a file, but dont use that
    python -m wbia.templates.template_generator --key image --onlyfn
    python -m wbia.templates.template_generator --key image --fnfilt timedelta_posix --modfname manual_image_funcs  # NOQA
    python -m wbia.templates.template_generator --key image --fnfilt location --modfname manual_image_funcs  # NOQA
    python -m wbia.templates.template_generator --key image --fnfilt set_.*time --modfname manual_image_funcs  # NOQA

    image_timedelta_posix

"""
from __future__ import absolute_import, division, print_function
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject
from wbia.control.controller_inject import make_ibs_register_decorator

# from os.path import join, exists, abspath, normpath, isabs
from os.path import join, exists, isabs
import numpy as np
import utool as ut
import vtool as vt
from wbia.web import routes_ajax
import six

print, rrr, profile = ut.inject2(__name__)


DEBUG_THUMB = False

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


IMAGE_TIME_POSIX = 'image_time_posix'
IMAGE_LOCATION_CODE = 'image_location_code'
IMAGE_TIMEDELTA_POSIX = 'image_timedelta_posix'
PARTY_ROWID = 'party_rowid'
CONTRIBUTOR_ROWID = 'contributor_rowid'

ANNOT_ROWID = 'annot_rowid'
ANNOT_ROWIDS = 'annot_rowids'
IMAGE_ROWID = 'image_rowid'

IMAGE_COLNAMES = (
    'image_uuid',
    'image_uri',
    'image_uri_original',
    'image_original_name',
    'image_ext',
    'image_width',
    'image_height',
    'image_time_posix',
    'image_gps_lat',
    'image_gps_lon',
    'image_orientation',
    'image_note',
)


@register_ibs_method
@accessor_decors.ider
def _get_all_gids(ibs):
    r"""
    alias

    Returns:
        list_ (list):  all unfiltered gids (image rowids)
    """
    all_gids = ibs._get_all_image_rowids()
    return all_gids


@register_ibs_method
def _get_all_image_rowids(ibs):
    r"""
    all_image_rowids <- image.get_all_rowids()

    Returns:
        list_ (list): unfiltered image_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = image

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-_get_all_image_rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> all_image_rowids = ibs._get_all_image_rowids()
        >>> result = str(all_image_rowids)
        >>> print(result)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    all_image_rowids = ibs.db.get_all_rowids(const.IMAGE_TABLE)
    return all_image_rowids


@register_ibs_method
@accessor_decors.ider
@register_api('/api/image/', methods=['GET'])
def get_valid_gids(
    ibs, imgsetid=None, require_unixtime=False, require_gps=None, reviewed=None, **kwargs
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        imgsetid (None):
        require_unixtime (bool):
        reviewed (None):

    Returns:
        list: gid_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_valid_gids

    RESTful:
        Method: GET
        URL:    /api/image/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> imgsetid = None
        >>> require_unixtime = False
        >>> reviewed = None
        >>> # execute function
        >>> gid_list = get_valid_gids(ibs, imgsetid, require_unixtime, reviewed)
        >>> # verify results
        >>> result = str(gid_list)
        >>> print(result)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    if imgsetid is None:
        gid_list = ibs._get_all_gids()
    else:
        assert not ut.isiterable(imgsetid)
        gid_list = ibs.get_imageset_gids(imgsetid)
    if require_unixtime:
        # Remove images without timestamps
        unixtime_list = ibs.get_image_unixtime(gid_list, **kwargs)
        isvalid_list = [unixtime != -1 for unixtime in unixtime_list]
        gid_list = ut.compress(gid_list, isvalid_list)
    if require_gps:
        isvalid_gps = [
            lat != -1 and lon != -1 for lat, lon in ibs.get_image_gps(gid_list)
        ]
        gid_list = ut.compress(gid_list, isvalid_gps)
    if reviewed is not None:
        reviewed_list = ibs.get_image_reviewed(gid_list)
        isvalid_list = [reviewed == flag for flag in reviewed_list]
        gid_list = ut.compress(gid_list, isvalid_list)
    return gid_list


@register_ibs_method
@register_api('/api/image/<rowid>/', methods=['GET'])
def image_base64_api(rowid=None, thumbnail=False, fresh=False, **kwargs):
    r"""
    Returns the base64 encoded image of image <rowid>

    RESTful:
        Method: GET
        URL:    /api/image/<rowid>/
    """
    return routes_ajax.image_src(rowid, thumbnail=thumbnail, fresh=fresh, **kwargs)


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_gid(ibs, gid_list, eager=True, nInput=None):
    """ self verifier

    CommandLine:
        python -m wbia.control.manual_image_funcs --exec-get_image_gid

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids() + [None, -1, 10434320432]
        >>> gid_list_ = ibs.get_image_gid(gid_list)
        >>> assert [r is None for r in gid_list_[-3:]]
        >>> assert [r is not None for r in gid_list_[0:-3]]
        >>> print('gid_list_ = %r' % (gid_list_,))
    """
    id_iter = gid_list
    colnames = (IMAGE_ROWID,)
    gid_list = ibs.db.get(
        const.IMAGE_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return gid_list


@register_ibs_method
@register_api('/api/image/dict/', methods=['GET'])
def get_image_gids_with_aids(ibs, gid_list=None):
    if gid_list is None:
        gid_list = sorted(ibs.get_valid_gids())
    aids_list = ibs.get_image_aids(gid_list)
    zipped = zip(gid_list, aids_list)
    combined_dict = {gid: aid_list for gid, aid_list in zipped}
    return combined_dict


@register_ibs_method
@accessor_decors.ider
# @register_api('/api/image/rowid/valid/', methods=['GET'])
def get_valid_image_rowids(ibs, imgsetid=None, require_unixtime=False, reviewed=None):
    r"""
    alias
    """
    return get_valid_gids(ibs, imgsetid, require_unixtime, reviewed)


@register_ibs_method
def get_num_images(ibs, **kwargs):
    r"""
    Number of valid images
    """
    gid_list = ibs.get_valid_gids(**kwargs)
    return len(gid_list)


@register_ibs_method
def _compute_image_uuids(ibs, gpath_list, sanitize=True, ensure=True, **kwargs):
    from wbia.algo.preproc import preproc_image
    from wbia.other import ibsfuncs

    # print('[ibs] gpath_list = %r' % (gpath_list,))
    # Processing an image might fail, yeilding a None instead of a tup
    if sanitize:
        gpath_list = ibsfuncs.ensure_unix_gpaths(gpath_list)

    # Create param_iter
    # params_list = list(preproc_image.add_images_params_gen(gpath_list))
    force_serial = ibs.force_serial or ibs.production
    params_list = list(
        ut.generate2(
            preproc_image.parse_imageinfo,
            list(zip(gpath_list)),
            nTasks=len(gpath_list),
            ordered=True,
            force_serial=force_serial,
            futures_threaded=True,
        )
    )

    # Error reporting
    failed_list = [
        gpath for (gpath, params_) in zip(gpath_list, params_list) if not params_
    ]

    print('\n'.join([' ! Failed reading gpath=%r' % (gpath,) for gpath in failed_list]))

    if ensure and len(failed_list) > 0:
        print('Importing %d files failed: %r' % (len(failed_list), failed_list,))

    return params_list


@register_ibs_method
@register_api('/api/image/uuid/', methods=['POST'])
def compute_image_uuids(ibs, gpath_list, **kwargs):
    params_list = _compute_image_uuids(ibs, gpath_list, **kwargs)

    uuid_colx = IMAGE_COLNAMES.index('image_uuid')
    uuid_list = [
        None if params_ is None else params_[uuid_colx] for params_ in params_list
    ]

    return uuid_list


@register_ibs_method
@accessor_decors.adder
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/', methods=['POST'])
def add_images(
    ibs,
    gpath_list,
    params_list=None,
    as_annots=False,
    auto_localize=None,
    location_for_names=None,
    ensure_unique=False,
    ensure_loadable=True,
    ensure_exif=True,
    **kwargs
):
    r"""
    Adds a list of image paths to the database.

    Initially we set the image_uri to exactely the given gpath.
    Later we change the uri, but keeping it the same here lets
    us process images asychronously.

    Args:
        gpath_list (list): list of image paths to add
        params_list (list): metadata list for corresponding images that can either be
            specified outright or can be parsed from the image data directly if None
        as_annots (bool): if True, an annotation is automatically added for the entire
            image
        auto_localize (bool): if None uses the default specified in ibs.cfg
        ensure (bool): check to see if the images exist on a \*NIX system.  Defaults to
            True

    Returns:
        gid_list (list of rowids): gids are image rowids

    RESTful:
        Method: POST
        URL:    /api/image/

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-add_images

    Doctest:
        >>> # Test returns None on fail to add
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gpath_list = ['doesnotexist.jpg']
        >>> assert not ut.checkpath(gpath_list[0])
        >>> gid_list = ibs.add_images(gpath_list)
        >>> assert len(gid_list) == len(gpath_list)
        >>> assert gid_list[0] is None

    Doctest:
        >>> # test double add
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> new_gpath_list = [ut.grab_test_imgpath('carl.jpg')]
        >>> new_gids1 = ibs.add_images(new_gpath_list, auto_localize=False)
        >>> new_gids2 = ibs.add_images(new_gpath_list, auto_localize=False)
        >>> #new_gids2 = ibs.add_images(new_gpath_list, auto_localize=True)
        >>> assert new_gids1 == new_gids2, 'should be the same'
        >>> new_gpath_list2 = ibs.get_image_paths(new_gids1)
        >>> assert new_gpath_list == new_gpath_list2, 'should not move when autolocalize is False'
        >>> # Clean things up
        >>> ibs.delete_images(new_gids1)
    """
    print('[ibs] add_images')
    print('[ibs] len(gpath_list) = %d' % len(gpath_list))
    if auto_localize is None:
        # grab value from config
        auto_localize = ibs.cfg.other_cfg.auto_localize

    location_for_names = None
    if location_for_names is None:
        location_for_names = ibs.cfg.other_cfg.location_for_names

    compute_params = params_list is None
    if compute_params:
        params_list = ibs._compute_image_uuids(gpath_list, **kwargs)

    # <DEBUG>
    debug = False
    if debug:
        uuid_colx = IMAGE_COLNAMES.index('image_uuid')
        uuid_list = [
            None if params_ is None else params_[uuid_colx] for params_ in params_list
        ]
        gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
        valid_gids = ibs.get_valid_gids()
        valid_uuids = ibs.get_image_uuids(valid_gids)
        print('[preadd] uuid / gid_ = ' + ut.indentjoin(zip(uuid_list, gid_list_)))
        print(
            '[preadd] valid uuid / gid = ' + ut.indentjoin(zip(valid_uuids, valid_gids))
        )
    # </DEBUG>

    # Execute SQL Add
    from distutils.version import LooseVersion

    if LooseVersion(ibs.db.get_db_version()) >= LooseVersion('1.3.4'):
        colnames = IMAGE_COLNAMES + ('image_original_path', 'image_location_code')
        params_list = [
            tuple(params) + (gpath, location_for_names) if params is not None else None
            for params, gpath in zip(params_list, gpath_list)
        ]

    all_gid_list = ibs.db.add_cleanly(
        const.IMAGE_TABLE, colnames, params_list, ibs.get_image_gids_from_uuid
    )

    # Filter for valid images and de-duplicate
    none_set = set([None])
    all_gid_set = set(all_gid_list)
    all_valid_gid_set = all_gid_set - none_set
    all_valid_gid_list = list(all_valid_gid_set)

    if auto_localize:
        # Move to wbia database local cache
        ibs.localize_images(all_valid_gid_list)

    # Check for duplicates
    has_duplicates = ut.duplicates_exist(all_gid_list)
    if ensure_unique and has_duplicates:
        debug_gpath_list = ibs.get_image_paths(all_gid_list)
        debug_guuid_list = ibs.get_image_uuids(all_gid_list)
        debug_gext_list = ibs.get_image_exts(all_gid_list)
        ut.debug_duplicate_items(
            all_gid_list, debug_gpath_list, debug_guuid_list, debug_gext_list
        )

    # Check loadable
    if ensure_loadable or ensure_exif:
        valid_gpath_list = ibs.get_image_paths(all_valid_gid_list)
        bad_load_list, bad_exif_list = ibs.check_image_loadable(all_valid_gid_list)
        bad_load_set = set(bad_load_list)
        bad_exif_set = set(bad_exif_list)

        delete_gid_set = set([])
        for valid_gid, valid_gpath in zip(all_valid_gid_list, valid_gpath_list):
            if ensure_loadable and valid_gid in bad_load_set:
                print('Loadable Image Validation: Failed to load %r' % (valid_gpath,))
                delete_gid_set.add(valid_gid)
            if ensure_exif and valid_gid in bad_exif_set:
                print('Loadable EXIF Validation:  Failed to load %r' % (valid_gpath,))
                delete_gid_set.add(valid_gid)

        delete_gid_list = list(delete_gid_set)
        ibs.delete_images(delete_gid_list, trash_images=False)

        all_valid_gid_set = all_gid_set - delete_gid_set - none_set
        all_valid_gid_list = list(all_valid_gid_set)

    if not compute_params:
        # We need to double check that the UUIDs are valid, considering we received the UUIDs
        guuid_list = ibs.get_image_uuids(all_gid_list)
        guuid_list_ = ibs.compute_image_uuids(gpath_list)
        assert guuid_list == guuid_list_

    if as_annots:
        # Add succesfull imports as annotations
        aid_list = ibs.use_images_as_annotations(all_valid_gid_list)
        print('[ibs] added %d annotations' % (len(aid_list),))

    # None out any gids that didn't pass the validity check
    assert None not in all_valid_gid_set
    all_gid_list = [aid if aid in all_valid_gid_set else None for aid in all_gid_list]
    assert len(gpath_list) == len(all_gid_list)
    return all_gid_list


@register_ibs_method
def get_image_exif_original(ibs, gid_list):
    import vtool.exif as vtexif
    from PIL import Image

    gpath_list = ibs.get_image_paths(gid_list)

    exif_dict_list = []
    for gpath in gpath_list:
        pil_img = Image.open(gpath, 'r')
        exif_dict = vtexif.get_exif_dict(pil_img)
        exif_dict_list.append(exif_dict)

    return exif_dict_list


@register_ibs_method
def localize_images(ibs, gid_list_=None):
    r"""
    Moves the images into the wbia image cache.
    Images are renamed to img_uuid.ext

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list_ (list):

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-localize_images

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gpath_list  = [ut.unixpath(ut.grab_test_imgpath('carl.jpg'))]
        >>> gid_list_   = ibs.add_images(gpath_list, auto_localize=False)
        >>> gpath_list2 = ibs.get_image_paths(gid_list_)
        >>> ut.assert_eq(gpath_list, gpath_list2, 'should not move when autolocalize is False')
        >>> # execute function
        >>> result = localize_images(ibs, gid_list_)
        >>> gpath_list3 = ibs.get_image_paths(gid_list_)
        >>> assert gpath_list3 != gpath_list2, 'should now be different gpath_list3=%r' % (gpath_list3,)
        >>> gpath3 = gpath_list3[0]
        >>> rel_gpath3 = ut.relpath_unix(gpath3, ibs.get_workdir())
        >>> result = rel_gpath3
        >>> print(result)
        >>> # Clean things up
        >>> ibs.delete_images(gid_list_)
        testdb1/_ibsdb/images/f498fa6f-6b24-b4fa-7932-2612144fedd5.jpg

    Ignore:
        ibs.vd()

    """
    # from os.path import isabs
    import six
    import requests

    if six.PY2:
        import urllib
        import urlparse

        urlsplit = urlparse.urlsplit
        urlquote = urllib.quote
    else:
        import urllib

        urlsplit = urllib.parse.urlsplit
        urlquote = urllib.parse.quote
        urlunquote = urllib.parse.unquote

    if gid_list_ is None:
        print('WARNING: you are localizing all gids')
        gid_list_ = ibs.get_valid_gids()
    isvalid_list = [gid is not None for gid in gid_list_]
    gid_list = ut.unique(ut.compress(gid_list_, isvalid_list))

    # gpath_list = ibs.get_image_paths(gid_list)
    uri_list = ibs.get_image_uris(gid_list)

    url_protos = ['https://', 'http://']
    s3_proto = ['s3://']
    valid_protos = s3_proto + url_protos

    def isproto(uri, valid_protos):
        return any(uri.startswith(proto) for proto in valid_protos)

    def islocal(uri):
        return not (isabs(uri) and isproto(uri, valid_protos))

    guuid_list = ibs.get_image_uuids(gid_list)
    gext_list = ibs.get_image_exts(gid_list)
    # Build list of image names based on uuid in the wbia imgdir
    guuid_strs = (str(guuid) for guuid in guuid_list)
    loc_gname_list = [guuid + ext for (guuid, ext) in zip(guuid_strs, gext_list)]
    loc_gpath_list = [join(ibs.imgdir, gname) for gname in loc_gname_list]

    # Copy any s3/http images first
    for uri, loc_gpath in zip(uri_list, loc_gpath_list):
        print('Localizing %r -> %r' % (uri, loc_gpath,))
        if isproto(uri, valid_protos):
            if isproto(uri, s3_proto):
                print('\tAWS S3 Fetch')
                s3_dict = ut.s3_str_decode_to_dict(uri)
                ut.grab_s3_contents(loc_gpath, **s3_dict)
            elif isproto(uri, url_protos):
                print('\tURL Download')
                # Ensure that the Unicode string is properly encoded for web requests
                uri_ = urlunquote(uri)
                uri_ = urlsplit(uri_, allow_fragments=False)
                uri_path = urlquote(uri_.path.encode('utf8'))
                uri_ = uri_._replace(path=uri_path)
                uri_ = uri_.geturl()
                try:
                    # six.moves.urllib.request.urlretrieve(uri_, filename=temp_filepath)
                    response = requests.get(uri_, stream=True, allow_redirects=True)
                    assert (
                        response.status_code == 200
                    ), '200 code not received on download'
                except Exception:
                    scheme = urlsplit(uri_, allow_fragments=False).scheme
                    uri_ = uri_.strip('%s://' % (scheme,))
                    uri_path = urlquote(uri_.encode('utf8'))
                    uri_ = '%s://%s' % (scheme, uri_path,)
                    # six.moves.urllib.request.urlretrieve(uri_, filename=temp_filepath)
                    response = requests.get(uri_, stream=True, allow_redirects=True)
                    assert (
                        response.status_code == 200
                    ), '200 code not received on download'
                # Save
                with open(loc_gpath, 'wb') as temp_file_:
                    for chunk in response.iter_content(1024):
                        temp_file_.write(chunk)
            else:
                raise ValueError('Sanity check failed')
        else:
            if not exists(loc_gpath):
                print('\tIO Copy')
                # Copy images to local directory
                uri if islocal(uri) else join(ibs.imgdir, uri)
                ut.copy_list([uri], [loc_gpath])
            else:
                print('\tSkipping (already localized)')
    # Update database uris
    ibs.set_image_uris(gid_list, loc_gname_list)
    assert all(map(exists, loc_gpath_list)), 'not all images copied'


# SETTERS::IMAGE


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/uri/', methods=['PUT'])
def set_image_uris(ibs, gid_list, new_gpath_list):
    r"""
    Sets the image URIs to a new local path.
    This is used when localizing or unlocalizing images.
    An absolute path can either be on this machine or on the cloud
    A relative path is relative to the wbia image cache on this machine.

    RESTful:
        Method: PUT
        URL:    /api/image/uri/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((new_gpath,) for new_gpath in new_gpath_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_uri',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/uri/original/', methods=['PUT'])
def set_image_uris_original(ibs, gid_list, new_gpath_list, overwrite=False):
    r"""
    Sets the (original) image URIs to a new local path.

    Args:
        overwrite (bool): If overwrite, replace the information in the database.
            This ensures that original uris cannot be accidentally overwritten.
            Defaults to False.

    RESTful:
        Method: PUT
        URL:    /api/image/uri/original/
    """
    if overwrite:
        gid_list_ = gid_list
        new_gpath_list_ = new_gpath_list
    else:
        current_uri_original_list = ibs.get_image_uris_original(gid_list)
        valid_flags = [
            current is None or len(current) == 0 for current in current_uri_original_list
        ]
        invalid_flags = ut.not_list(valid_flags)
        nInvalid = sum(invalid_flags)
        if nInvalid > 0:
            print('[ibs] WARNING: Preventing overwrite of %d original uris' % (nInvalid,))
        new_gpath_list_ = ut.compress(new_gpath_list, valid_flags)
        gid_list_ = ut.compress(gid_list, valid_flags)
    # new_gpath_list_ = [
    #    new if _invalid(current) or overwrite else current
    #    for current, new in zip(current_uri_original_list, new_gpath_list)
    # ]
    id_iter = ((gid,) for gid in gid_list_)
    val_list = ((new_gpath,) for new_gpath in new_gpath_list_)
    ibs.db.set(const.IMAGE_TABLE, ('image_uri_original',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/contributor/rowid/', methods=['PUT'])
def set_image_contributor_rowid(ibs, gid_list, contributor_rowid_list, **kwargs):
    r"""
    Sets the image contributor rowid
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((contributor_rowid,) for contributor_rowid in contributor_rowid_list)
    ibs.db.set(const.IMAGE_TABLE, ('contributor_rowid',), val_list, id_iter, **kwargs)


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/reviewed/', methods=['PUT'])
def set_image_reviewed(ibs, gid_list, reviewed_list):
    r"""
    Sets the image all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/image/reviewed/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((reviewed,) for reviewed in reviewed_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_reviewed',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/image/enabled/', methods=['PUT'])
def set_image_enabled(ibs, gid_list, enabled_list):
    r"""
    Sets the image all instances found bit
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((enabled,) for enabled in enabled_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_enabled',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/image/enabled/', methods=['PUT'])
def set_image_cameratrap(ibs, gid_list, cameratrap_list):
    r"""
    Sets the image all instances found bit
    """
    id_iter = ((gid,) for gid in gid_list)
    valid_set = set([False, True, None])
    valid_list = [cameratrap in valid_set for cameratrap in cameratrap_list]
    assert False not in valid_list
    val_list = ((cameratrap,) for cameratrap in cameratrap_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_cameratrap',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/note/', methods=['PUT'])
def set_image_notes(ibs, gid_list, notes_list):
    r"""
    Sets the image all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/image/note/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((notes,) for notes in notes_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_note',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/metadata/', methods=['PUT'])
def set_image_metadata(ibs, gid_list, metadata_dict_list):
    r"""
    Sets the image's metadata using a metadata dictionary

    RESTful:
        Method: PUT
        URL:    /api/image/metadata/

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-set_image_metadata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> metadata_dict_list = [
        >>>     {'test': random.uniform(0.0, 1.0)},
        >>> ]
        >>> print(ut.repr2(metadata_dict_list))
        >>> ibs.set_image_metadata(gid_list, metadata_dict_list)
        >>> # verify results
        >>> metadata_dict_list_ = ibs.get_image_metadata(gid_list)
        >>> print(ut.repr2(metadata_dict_list_))
        >>> assert metadata_dict_list == metadata_dict_list_
        >>> metadata_str_list = [ut.to_json(metadata_dict) for metadata_dict in metadata_dict_list]
        >>> print(ut.repr2(metadata_str_list))
        >>> metadata_str_list_ = ibs.get_image_metadata(gid_list, return_raw=True)
        >>> print(ut.repr2(metadata_str_list_))
        >>> assert metadata_str_list == metadata_str_list_
    """
    id_iter = ((gid,) for gid in gid_list)
    metadata_str_list = []
    for metadata_dict in metadata_dict_list:
        metadata_str = ut.to_json(metadata_dict)
        metadata_str_list.append(metadata_str)
    val_list = ((metadata_str,) for metadata_str in metadata_str_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_metadata_json',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/unixtime/', methods=['PUT'])
def set_image_unixtime(ibs, gid_list, unixtime_list, duplicate_behavior='error'):
    r"""
    Sets the image unixtime (does not modify exif yet)
        alias for set_image_time_posix

    RESTful:
        Method: PUT
        URL:    /api/image/unixtime/

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-set_image_unixtime

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> import time
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:5]
        >>> unixtime_list = [
        >>>     random.randint(1, int(time.time()))
        >>>     for _ in gid_list
        >>> ]
        >>> print(ut.repr2(unixtime_list))
        >>> ibs.set_image_unixtime(gid_list, unixtime_list)
        >>> # verify results
        >>> unixtime_list_ = ibs.get_image_unixtime(gid_list)
        >>> print(ut.repr2(unixtime_list_))
        >>> assert unixtime_list == unixtime_list_

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> import time
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:5]
        >>> gid_list = gid_list + gid_list
        >>> unixtime_list = [
        >>>     random.randint(1, int(time.time()))
        >>>     for _ in gid_list
        >>> ]
        >>> try:
        >>>     print(ut.repr2(unixtime_list))
        >>>     ibs.set_image_unixtime(gid_list, unixtime_list)
        >>> except AssertionError:
        >>>     pass

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> import time
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:5]
        >>> unixtime_list = [
        >>>     random.randint(1, int(time.time()))
        >>>     for _ in gid_list
        >>> ]
        >>> gid_list = gid_list + gid_list
        >>> unixtime_list = unixtime_list + unixtime_list
        >>> print(ut.repr2(unixtime_list))
        >>> ibs.set_image_unixtime(gid_list, unixtime_list)
        >>> # verify results
        >>> unixtime_list_ = ibs.get_image_unixtime(gid_list)
        >>> print(ut.repr2(unixtime_list_))
        >>> assert unixtime_list == unixtime_list_
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((unixtime,) for unixtime in unixtime_list)
    ibs.db.set(
        const.IMAGE_TABLE,
        (IMAGE_TIME_POSIX,),
        val_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@register_api('/api/image/time/posix/', methods=['PUT'])
def set_image_time_posix(
    ibs, gid_list, image_time_posix_list, duplicate_behavior='error'
):
    r"""
    image_time_posix_list -> image.image_time_posix[gid_list]

    SeeAlso:
        set_image_unixtime

    Args:
        gid_list
        image_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_time_posix

    RESTful:
        Method: PUT
        URL:    /api/image/time/posix/
    """
    id_iter = gid_list
    colnames = (IMAGE_TIME_POSIX,)
    ibs.db.set(
        const.IMAGE_TABLE,
        colnames,
        image_time_posix_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/imageset/text/', methods=['PUT'])
def set_image_imagesettext(ibs, gid_list, imagesettext_list):
    r"""
    Sets the encoutertext of each image

    RESTful:
        Method: PUT
        URL:    /api/image/imageset/text/
    """
    # FIXME: Slow and weird
    if ut.VERBOSE:
        print('[ibs] setting %r image imageset ids (from text)' % len(gid_list))
    imgsetid_list = ibs.add_imagesets(imagesettext_list)
    ibs.set_image_imgsetids(gid_list, imgsetid_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/imageset/rowid/', methods=['PUT'])
def set_image_imgsetids(ibs, gid_list, imgsetid_list):
    r"""
    Sets the encoutertext of each image

    RESTful:
        Method: PUT
        URL:    /api/image/imageset/rowid/
    """
    if ut.VERBOSE:
        print('[ibs] setting %r image imageset ids' % len(gid_list))
    ibs.add_image_relationship(gid_list, imgsetid_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/gps/', methods=['PUT'], __api_plural_check__=False)
def set_image_gps(ibs, gid_list, gps_list=None, lat_list=None, lon_list=None):
    r"""
    see get_image_gps for how the gps_list should look.
        lat and lon should be given in degrees

    RESTful:
        Method: PUT
        URL:    /api/image/gps/
    """
    if gps_list is not None:
        assert lat_list is None
        assert lon_list is None
        lat_list = [tup[0] for tup in gps_list]
        lon_list = [tup[1] for tup in gps_list]
    colnames = (
        'image_gps_lat',
        'image_gps_lon',
    )
    val_list = zip(lat_list, lon_list)
    id_iter = ((gid,) for gid in gid_list)
    ibs.db.set(const.IMAGE_TABLE, colnames, val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/gps/str/', methods=['PUT'], __api_plural_check__=False)
def set_image_gps_str(ibs, gid_list, gps_str_list):
    r"""
    see get_image_gps for how the gps_list should look.
        lat and lon should be given in degrees

    RESTful:
        Method: PUT
        URL:    /api/image/gps/
    """
    lat_list = []
    lon_list = []
    for gps_str in gps_str_list:
        # Strip any tuple () and spaces
        gps_str = gps_str.strip()
        gps_str = gps_str.strip('(').strip(')')
        # Replace any spaces with commas
        gps_str = gps_str.replace(' ', ',')
        # Split by commas and strip each component of spaces
        gps_str_ = gps_str.split(',')
        gps_str_ = [_.strip() for _ in gps_str_]
        # Filter out any values that are empty
        gps_str_ = [_ for _ in gps_str_ if len(_) > 0]
        # Make sure that there are only 2
        assert len(gps_str_) == 2
        # Cast to floats
        gps_str_ = [float(_) for _ in gps_str_]
        lat = gps_str_[0]
        lon = gps_str_[1]
        assert -90.0 <= lat and lat <= 90.0
        assert -180.0 <= lon and lon <= 180.0
        lat_list.append(lat)
        lon_list.append(lon)
    colnames = (
        'image_gps_lat',
        'image_gps_lon',
    )
    val_list = zip(lat_list, lon_list)
    id_iter = ((gid,) for gid in gid_list)
    ibs.db.set(const.IMAGE_TABLE, colnames, val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
def _set_image_sizes(ibs, gid_list, width_list, height_list):
    colnames = (
        'image_width',
        'image_height',
    )
    val_list = zip(width_list, height_list)
    id_iter = ((gid,) for gid in gid_list)
    ibs.db.set(const.IMAGE_TABLE, colnames, val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/orientation/', methods=['PUT'])
def _set_image_orientation(ibs, gid_list, orientation_list):
    r"""
    RESTful:
        Method: PUT
        URL:    /api/image/orientation/
    """
    colnames = ('image_orientation',)
    val_list = ((orientation,) for orientation in orientation_list)
    id_iter = ((gid,) for gid in gid_list)
    ibs.db.set(const.IMAGE_TABLE, colnames, val_list, id_iter)
    ibs.depc_image.notify_root_changed(gid_list, 'image_orientation')
    ibs.delete_image_thumbs(gid_list)


def update_image_rotate_90(ibs, gid_list, direction):
    from vtool.exif import (
        ORIENTATION_DICT_INVERSE,
        ORIENTATION_ORDER_LIST,
        ORIENTATION_UNDEFINED,
        ORIENTATION_000,
    )

    def _update_bounding_boxes(gid, val):
        full_w, full_h = ibs.get_image_sizes(gid)
        aid_list = ibs.get_image_aids(gid, is_staged=None)
        if len(aid_list) == 0:
            return
        bbox_list = ibs.get_annot_bboxes(aid_list)
        bbox_list_ = []
        for bbox in bbox_list:
            (xtl, ytl, width, height) = bbox
            if val > 0:
                xtl, ytl = full_w - ytl - height, xtl
            else:
                xtl, ytl = ytl, full_h - xtl - width
            width, height = height, width
            bbox_ = (xtl, ytl, width, height)
            bbox_list_.append(bbox_)
        ibs.set_annot_bboxes(aid_list, bbox_list_)

    if isinstance(direction, six.string_types):
        direction = direction.lower()

    if direction in ['left', 'l', -1]:
        val = -1
    elif direction in ['right', 'r', 1]:
        val = 1
    else:
        raise ValueError('Invalid direction supplied')

    new_orient_list = []
    orient_list = ibs.get_image_orientation(gid_list)
    for orient in orient_list:
        if orient == ORIENTATION_DICT_INVERSE[ORIENTATION_UNDEFINED]:
            orient = ORIENTATION_DICT_INVERSE[ORIENTATION_000]

        assert orient in ORIENTATION_ORDER_LIST, 'Unrecognized orientation = %r in %r' % (
            orient,
            ORIENTATION_ORDER_LIST,
        )

        current_index = ORIENTATION_ORDER_LIST.index(orient)
        new_index = int((current_index + val)) % len(ORIENTATION_ORDER_LIST)

        new_orient = ORIENTATION_ORDER_LIST[new_index]
        new_orient_list.append(new_orient)

    print('Rotating images %r -> %r' % (orient_list, new_orient_list,))
    ibs._set_image_orientation(gid_list, new_orient_list)

    # We've just rotated, invert the width, height values in the database for each image
    # IMPORTANT: DO THIS AFTER FIXING THE BBOXES
    image_list = ibs.get_images(gid_list)
    shape_list = [image.shape[:2] for image in image_list]
    height_list = [shape[0] for shape in shape_list]
    width_list = [shape[1] for shape in shape_list]
    ibs._set_image_sizes(gid_list, width_list, height_list)

    # Update the bounding box locations
    for gid in gid_list:
        _update_bounding_boxes(gid, val)

    # Update the annotation bounding box thetas
    aids_list = ibs.get_image_aids(gid_list, is_staged=None)
    for aid_list in aids_list:
        if len(aid_list) == 0:
            continue
        if val > 0:
            ibs.update_annot_rotate_left_90(aid_list)
        else:
            ibs.update_annot_rotate_right_90(aid_list)

    # Update the part bounding box thetas
    aid_list = ut.flatten(aids_list)
    part_rowids_list = ibs.get_annot_part_rowids(aid_list)
    for part_rowid_list in part_rowids_list:
        if len(part_rowid_list) == 0:
            continue
        if val > 0:
            ibs.update_part_rotate_left_90(part_rowid_list)
        else:
            ibs.update_part_rotate_right_90(part_rowid_list)

    ibs.delete_image_thumbs(gid_list)


@register_ibs_method
@register_api('/api/image/rotate/left/', methods=['POST'])
def update_image_rotate_left_90(ibs, gid_list):
    update_image_rotate_90(ibs, gid_list, 'left')


@register_ibs_method
@register_api('/api/image/rotate/right/', methods=['POST'])
def update_image_rotate_right_90(ibs, gid_list):
    update_image_rotate_90(ibs, gid_list, 'right')


@register_ibs_method
@register_api('/api/image/rotate/180/', methods=['POST'])
def update_image_rotate_180(ibs, gid_list):
    update_image_rotate_90(ibs, gid_list, 'right')
    update_image_rotate_90(ibs, gid_list, 'right')


#
# GETTERS::IMAGE_TABLE


@register_ibs_method
@accessor_decors.getter_1to1
def get_images(ibs, gid_list, force_orient=True, **kwargs):
    r"""
    Returns:
        list_ (list): a list of images in numpy matrix form by gid

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: image_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_images

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> # execute function
        >>> image_list = get_images(ibs, gid_list)
        >>> # verify results
        >>> result = str(image_list[0].shape)
        >>> print(result)
        (715, 1047, 3)
    """
    orient_list = ibs.get_image_orientation(gid_list)
    orient_list = [orient if force_orient else False for orient in orient_list]
    gpath_list = ibs.get_image_paths(gid_list)
    zipped = zip(gpath_list, orient_list)
    image_list = [vt.imread(gpath, orient=orient) for gpath, orient in zipped]
    return image_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_imgdata(ibs, gid_list, force_orient=False, **kwargs):
    """ alias for get_images with standardized name """
    return get_images(ibs, gid_list, force_orient=force_orient, **kwargs)


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/thumbtup/', methods=['GET'])
def get_image_thumbtup(ibs, gid_list, **kwargs):
    r"""
    Returns:
        list: thumbtup_list - [(thumb_path, img_path, imgsize, bboxes, thetas)]
    """
    if DEBUG_THUMB:
        print('{TUPPLE} get thumbtup kwargs = %r' % (kwargs,))
    # print('gid_list = %r' % (gid_list,))
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = ibs.unflat_map(ibs.get_annot_bboxes, aids_list)
    thetas_list = ibs.unflat_map(ibs.get_annot_thetas, aids_list)
    interests_list = ibs.unflat_map(ibs.get_annot_interest, aids_list)
    thumb_gpaths = ibs.get_image_thumbpath(gid_list, **kwargs)
    image_paths = ibs.get_image_paths(gid_list)
    gsize_list = ibs.get_image_sizes(gid_list)
    thumbtup_list = [
        (thumb_path, img_path, img_size, bboxes, thetas, interests)
        for thumb_path, img_path, img_size, bboxes, thetas, interests in zip(
            thumb_gpaths,
            image_paths,
            gsize_list,
            bboxes_list,
            thetas_list,
            interests_list,
        )
    ]
    # if DEBUG_THUMB:
    #     print('{TUPPLE} get thumbtup_list = %r' % (thumbtup_list,))
    return thumbtup_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/thumbpath/', methods=['GET'])
def get_image_thumbpath(ibs, gid_list, ensure_paths=False, **config):
    r"""
    Returns:
        list_ (list): the thumbnail path of each gid
    """
    if DEBUG_THUMB:
        print('[GET} get_image_thumbpath for %d gids' % (len(gid_list)))
        print('[GET} get thumbtup config = %r' % (config,))
        print('[GET} get thumbtup ensure_paths = %r' % (ensure_paths,))
    # raise Exception("FOOBAR")
    depc = ibs.depc_image
    # Do not force computation just ask where the thumbs will go
    # This will allow the frontend to know where to read the images when they
    # are ready. They should be computed in the background.
    # HACK: this is hacked into the depcache to force it to work
    # It is not gaurneteed that it will ever work
    # FIXME: I think Qt will end up computing these thumbnails and writing them
    # to where the depcache expects them. I think the depcache will then
    # override them but this may cause unexpected results.
    # FIXME: Thumbnails may have annotations drawn on them! This is not
    # represented anywhere in the depcache.
    thumbpath_list = depc.get(
        'thumbnails',
        gid_list,
        'img',
        config=config,
        read_extern=False,
        ensure=ensure_paths,
        hack_paths=not ensure_paths,
    )
    # except dtool.ExternalStorageException:
    #    # TODO; this check might go in dtool itself
    #    thumbpath_list = depc.get('thumbnails', gid_list, 'img', config=config,
    #                               read_extern=False)
    if DEBUG_THUMB:
        print('[GET} thumbpath_list = %r' % (thumbpath_list,))
    return thumbpath_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/thumbpath/', methods=['GET'])
def get_image_thumbnail(ibs, gid_list, **config):
    r"""
    Returns:
        list_ (list): the thumbnail path of each gid
    """
    depc = ibs.depc_image
    thumbpath_list = depc.get('thumbnails', gid_list, 'img', config=config)
    return thumbpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/uuid/', methods=['GET'])
def get_image_uuids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uuids by gid

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: image_uuid_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_uuids

    RESTful:
        Method: GET
        URL:    /api/image/uuid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> image_uuid_list = ibs.get_image_uuids(gid_list)
        >>> # verify results
        >>> result = ut.repr2(image_uuid_list, nl=1)
        >>> print(result)
        [
            UUID('66ec193a-1619-b3b6-216d-1784b4833b61'),
            UUID('d8903434-942f-e0f5-d6c2-0dcbe3137bf7'),
            UUID('b73b72f4-4acb-c445-e72c-05ce02719d3d'),
            UUID('0cd05978-3d83-b2ee-2ac9-798dd571c3b3'),
            UUID('0a9bc03d-a75e-8d14-0153-e2949502aba7'),
            UUID('2deeff06-5546-c752-15dc-2bd0fdb1198a'),
            UUID('a9b70278-a936-c1dd-8a3b-bc1e9a998bf0'),
            UUID('42fdad98-369a-2cbc-67b1-983d6d6a3a60'),
            UUID('c459d381-fd74-1d99-6215-e42e3f432ea9'),
            UUID('33fd9813-3a2b-774b-3fcc-4360d1ae151b'),
            UUID('97e8ea74-873f-2092-b372-f928a7be30fa'),
            UUID('588bc218-83a5-d400-21aa-d499832632b0'),
            UUID('163a890c-36f2-981e-3529-c552b6d668a3'),
        ]
    """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_uuid',), gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/uuid/valid/', methods=['GET'])
def get_valid_image_uuids(ibs):
    r"""
    Returns:
        list_ (list): a list of image uuids for all valid gids

    Args:
        ibs (IBEISController):  wbia controller object

    Returns:
        list: image_uuid_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_uuids
    """
    gid_list = ibs.get_valid_gids()
    image_uuid_list = ibs.get_image_uuids(gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/contributor/rowid/', methods=['GET'])
def get_image_contributor_rowid(ibs, gid_list, eager=True, nInput=None):
    r"""
    contributor_rowid_list <- image.contributor_rowid[gid_list]

    gets data from the "native" column "contributor_rowid" in the "image" table

    Args:
        gid_list (list):

    Returns:
        list: contributor_rowid_list - list of image contributor rowids by gid

    TemplateInfo:
        Tgetter_table_column
        col = contributor_rowid
        tbl = image

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> contributor_rowid_list = ibs.get_image_contributor_rowid(gid_list, eager=eager)
        >>> assert len(gid_list) == len(contributor_rowid_list)
    """
    id_iter = gid_list
    colnames = (CONTRIBUTOR_ROWID,)
    contributor_rowid_list = ibs.db.get(
        const.IMAGE_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return contributor_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/ext/', methods=['GET'])
def get_image_exts(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uuids by gid
    """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_ext',), gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/uri/', methods=['GET'])
def get_image_uris(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uris relative to the image dir by gid

    RESTful:
        Method: GET
        URL:    /api/image/uri/
    """
    uri_list = ibs.db.get(const.IMAGE_TABLE, ('image_uri',), gid_list)
    return uri_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/uri/original/', methods=['GET'])
def get_image_uris_original(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (original) image uris relative to the image dir by gid

    RESTful:
        Method: GET
        URL:    /api/image/uri/original/
    """
    uri_list = ibs.db.get(const.IMAGE_TABLE, ('image_uri_original',), gid_list)
    return uri_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/rowid/uuid/', methods=['GET'])
def get_image_gids_from_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): a list of original image names

    RESTful:
        Method: GET
        URL:    /api/image/rowid/uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    gid_list = ibs.db.get(
        const.IMAGE_TABLE, ('image_rowid',), uuid_list, id_colname='image_uuid'
    )
    return gid_list


# get_image_rowid_from_uuid = get_image_gids_from_uuid


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/uuid/missing/', methods=['GET'])
def get_image_missing_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): a list of missing image uuids
    """
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)
    zipped = zip(gid_list, uuid_list)
    missing_uuid_list = [uuid for gid, uuid in zipped if gid is None]
    return missing_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/file/path/', methods=['GET'])
def get_image_paths(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list): a list of image absolute paths to img_dir

    Returns:
        list: gpath_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_paths

    RESTful:
        Method: GET
        URL:    /api/image/file/path/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> #gid_list = ibs.get_valid_gids()
        >>> #gpath_list = get_image_paths(ibs, gid_list)
        >>> new_gpath = ut.unixpath(ut.grab_test_imgpath('carl.jpg'))
        >>> gid_list = ibs.add_images([new_gpath], auto_localize=False)
        >>> new_gpath_list = get_image_paths(ibs, gid_list)
        >>> ut.assert_eq(new_gpath, new_gpath_list[0])
        >>> result = str(new_gpath_list)
        >>> ibs.delete_images(gid_list)
        >>> print(result)
    """
    # ut.assert_all_not_None(gid_list, 'gid_list', key_list=['gid_list'])
    uri_list = ibs.get_image_uris(gid_list)

    url_protos = ['https://', 'http://']
    s3_proto = ['s3://']
    valid_protos = s3_proto + url_protos

    def isproto(uri, valid_protos):
        return any(uri.startswith(proto) for proto in valid_protos)

    def islocal(uri):
        return not (isabs(uri) and isproto(uri, valid_protos))

    gpath_list = []
    for uri in uri_list:
        if uri is None:
            gpath = None
        elif isproto(uri, valid_protos):
            gpath = uri
        elif isabs(uri):
            gpath = uri
        else:
            assert islocal(uri)
            gpath = join(ibs.imgdir, uri)
        gpath_list.append(gpath)

    return gpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/file/hash/', methods=['GET'])
def get_image_hash(ibs, gid_list=None, algo='md5'):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list): a list of image absolute paths to img_dir

    Returns:
        list: hash_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_hash

    RESTful:
        Method: GET
        URL:    /api/image/file/hash/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[:1]
        >>> image_path = ibs.get_image_paths(gid_list)
        >>> print('Hashing: %r' % (image_path, ))
        >>> hash_list = ibs.get_image_hash(gid_list, algo='md5')
        >>> assert hash_list == ['ab31dc5e1355247a0ea5ec940802a468']
        >>> hash_list = ibs.get_image_hash(gid_list, algo='sha1')
        >>> assert hash_list == ['66ec193a1619b3b6216d1784b4833b6194b13384']
        >>> hash_list = ibs.get_image_hash(gid_list, algo='sha256')
        >>> assert hash_list == ['fd09d22ec18c32d9db2cd026a9511ab228aadf0e5f7271760413448ddd16d483']
        >>> hash_list = ibs.get_image_hash(gid_list, algo='sha512')
        >>> assert hash_list == ['81d1d8ee4c8640b9aad26e4cc03536ed30a43b69e166748ec940a8f00e4776be93f4ac6367a06d92b772a9a60dc104c6f999e7197c2584fdc4cffcac2da71506']
    """
    import hashlib

    assert isinstance(algo, six.string_types)
    algo = algo.lower()
    assert algo in ['md5', 'sha1', 'sha256', 'sha512']

    image_path_list = ibs.get_image_paths(gid_list)

    if algo == 'md5':
        hash_func = hashlib.md5
    elif algo == 'sha1':
        hash_func = hashlib.sha1
    elif algo == 'sha256':
        hash_func = hashlib.sha256
    elif algo == 'sha512':
        hash_func = hashlib.sha512
    else:
        raise ValueError('algo must be in %r' % (algo,))

    hash_list = []
    for image_path in image_path_list:
        if not exists(image_path):
            hash_ = None
        else:
            hash_ = hash_func(open(image_path, 'rb').read()).hexdigest()
        hash_list.append(hash_)

    return hash_list


# TODO make this actually return a uri format
# get_image_absolute_uri = get_image_paths


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/detectpath/', methods=['GET'])
def get_image_detectpaths(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image paths resized to a constant area for detection
    """
    from wbia import dtool

    depc = ibs.depc_image
    config = {
        'thumbsize': ibs.cfg.detect_cfg.detectimg_sqrt_area,
        'force_serial': True,
    }
    try:
        thumbpath_list = depc.get(
            'thumbnails', gid_list, 'img', config=config, read_extern=False
        )
    except dtool.ExternalStorageException:
        # TODO; this check might go in dtool itself
        thumbpath_list = depc.get(
            'thumbnails', gid_list, 'img', config=config, read_extern=False
        )
    # print(thumbpath_list)
    return thumbpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/file/name/', methods=['GET'])
def get_image_gnames(ibs, gid_list):
    r"""
    Args:
        gid_list (list):

    Returns:
        list: gname_list - a list of original image names

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_gnames

    RESTful:
        Method: GET
        URL:    /api/image/file/name/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> gname_list = get_image_gnames(ibs, gid_list)
        >>> # verify results
        >>> result = ut.repr2(gname_list, nl=1)
        >>> print(result)
        [
            'easy1.JPG',
            'easy2.JPG',
            'easy3.JPG',
            'hard1.JPG',
            'hard2.JPG',
            'hard3.JPG',
            'jeff.png',
            'lena.jpg',
            'occl1.JPG',
            'occl2.JPG',
            'polar1.jpg',
            'polar2.jpg',
            'zebra.jpg',
        ]
    """
    gname_list = ibs.db.get(const.IMAGE_TABLE, ('image_original_name',), gid_list)
    return gname_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/size/', methods=['GET'])
def get_image_sizes(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/size/
    """
    gsize_list = ibs.db.get(const.IMAGE_TABLE, ('image_width', 'image_height'), gid_list)
    return gsize_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/width/', methods=['GET'])
def get_image_widths(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/width/
    """
    gwidth_list = ibs.db.get(const.IMAGE_TABLE, ('image_width',), gid_list)
    return gwidth_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/height/', methods=['GET'])
def get_image_heights(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/height/
    """
    gheight_list = ibs.db.get(const.IMAGE_TABLE, ('image_height',), gid_list)
    return gheight_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@register_api('/api/image/unixtime/', methods=['GET'])
def get_image_unixtime(ibs, gid_list, timedelta_correction=True):
    r"""
    Returns:
        list_ (list): a list of times that the images were taken by gid.

    Returns:
        list_ (list): -1 if no timedata exists for a given gid

    RESTful:
        Method: GET
        URL:    /api/image/unixtime/
    """
    unixtime_list = ibs.db.get(const.IMAGE_TABLE, ('image_time_posix',), gid_list)
    unixtime_list = [-1 if unixtime is None else unixtime for unixtime in unixtime_list]

    if timedelta_correction:
        timedelta_list = ibs.get_image_timedelta_posix(gid_list)
        timedelta_list = [
            0 if timedelta is None else timedelta for timedelta in timedelta_list
        ]
        unixtime_list = [
            unixtime + timedelta
            for unixtime, timedelta in zip(unixtime_list, timedelta_list)
        ]
    return unixtime_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
def get_image_unixtime_asfloat(ibs, gid_list, **kwargs):
    r"""
    Returns:
        list_ (list): a list of times that the images were taken by gid.

    Returns:
        list_ (list): np.nan if no timedata exists for a given gid
    """
    unixtime_list = ibs.get_image_unixtime(gid_list, **kwargs)
    unixtime_list = np.array(unixtime_list, dtype=np.float)
    # Fix problem in sql and make -1 be nans or nulls
    unixtime_list[unixtime_list == -1] = np.nan
    return unixtime_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@register_api('/api/image/unixtime2/', methods=['GET'])
def get_image_unixtime2(ibs, gid_list, **kwargs):
    """ alias for get_image_unixtime_asfloat """
    return ibs.get_image_unixtime_asfloat(gid_list, **kwargs)


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_datetime_str(ibs, gid_list, **kwargs):
    unixtime_list = ibs.get_image_unixtime(gid_list, **kwargs)
    datestr_list = list(map(ut.unixtime_to_datetimestr, unixtime_list))
    return datestr_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_datetime(ibs, gid_list, **kwargs):
    import datetime

    unixtime_list = ibs.get_image_unixtime(gid_list, **kwargs)
    datetime_list = [
        None if ts is None or ts == -1 else datetime.datetime.fromtimestamp(ts)
        for ts in unixtime_list
    ]
    return datetime_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/gps/', methods=['GET'], __api_plural_check__=False)
def get_image_gps(ibs, gid_list):
    r"""
    Returns:
        gps_list (list): -1 if no timedata exists for a given gid

    RESTful:
        Method: GET
        URL:    /api/image/gps/
    """
    gps_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat', 'image_gps_lon'), gid_list)
    # REPLACE -1 with np.nan FIXME in SQL
    # gps_list = [(np.nan if lat == -1 else lat, np.nan if lon == -1 else lon) for (lat, lon) in gps_list]
    return gps_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/gps2/', methods=['GET'], __api_plural_check__=False)
def get_image_gps2(ibs, gid_list):
    r"""
    Like get_image_gps, but fixes the SQL problem where -1 indicates a nan value.

    Returns:
        gps_list (list): -1 if no timedata exists for a given gid

    RESTful:
        Method: GET
        URL:    /api/image/gps/
    """
    gps_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat', 'image_gps_lon'), gid_list)
    gps_list = [
        (np.nan if lat == -1 else lat, np.nan if lon == -1 else lon)
        for (lat, lon) in gps_list
    ]
    # gps_list = [
    #    (np.nan, np.nan) if (lat == -1 and lon == -1) else (lat, lon)
    #    for (lat, lon) in gps_list
    # ]
    return gps_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/lat/', methods=['GET'])
def get_image_lat(ibs, gid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/image/lat/
    """
    lat_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat',), gid_list)
    # REPLACE -1 with np.nan FIXME in SQL
    # lat_list = [np.nan if lat == -1 else lat for lat in lat_list]
    return lat_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/lon/', methods=['GET'])
def get_image_lon(ibs, gid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/image/lon/
    """
    lon_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lon',), gid_list)
    # REPLACE -1 with np.nan FIXME in SQL
    # lon_list = [np.nan if lon == -1 else lon for lon in lon_list]
    return lon_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/orientation/', methods=['GET'])
def get_image_orientation(ibs, gid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/image/orientation/
    """
    orient_list = ibs.db.get(const.IMAGE_TABLE, ('image_orientation',), gid_list)
    return orient_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/orientation/str/', methods=['GET'])
def get_image_orientation_str(ibs, gid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/image/orientation/str/
    """
    from vtool.exif import ORIENTATION_DICT

    orient_list = ibs.get_image_orientation(gid_list)
    orient_str = [ORIENTATION_DICT[orient] for orient in orient_list]
    return orient_str


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/enabled/', methods=['GET'])
def get_image_enabled(ibs, gid_list):
    r"""
    Returns:
        list_ (list): "Image Enabled" flag, true if the image is enabled
    """
    enabled_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_enabled',), gid_list)
    return enabled_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_cameratrap(ibs, gid_list):
    cameratrap_list = ibs.db.get(
        const.IMAGE_TABLE, ('image_toggle_cameratrap',), gid_list
    )
    return cameratrap_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/reviewed/', methods=['GET'])
def get_image_reviewed(ibs, gid_list):
    r"""
    Returns:
        list_ (list): "All Instances Found" flag, true if all objects of interest
    (animals) have an ANNOTATION in the image

    RESTful:
        Method: GET
        URL:    /api/image/reviewed/
    """
    reviewed_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_reviewed',), gid_list)
    return reviewed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/detect/confidence/', methods=['GET'])
def get_image_detect_confidence(ibs, gid_list):
    r"""
    Returns:
        list_ (list): image detection confidence as the max of ANNOTATION confidences

    RESTful:
        Method: GET
        URL:    /api/image/detect/confidence/
    """
    aids_list = ibs.get_image_aids(gid_list)
    confs_list = ibs.unflat_map(ibs.get_annot_detect_confidence, aids_list)
    maxconf_list = [max(confs) if len(confs) > 0 else -1 for confs in confs_list]
    return maxconf_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/note/', methods=['GET'])
def get_image_notes(ibs, gid_list):
    r"""
    Returns:
        list_ (list): image notes

    RESTful:
        Method: GET
        URL:    /api/image/note/
    """
    notes_list = ibs.db.get(const.IMAGE_TABLE, ('image_note',), gid_list)
    return notes_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/metadata/', methods=['GET'])
def get_image_metadata(ibs, gid_list, return_raw=False):
    r"""
    Returns:
        list_ (list): image metadata dictionary

    RESTful:
        Method: GET
        URL:    /api/image/metadata/
    """
    metadata_str_list = ibs.db.get(const.IMAGE_TABLE, ('image_metadata_json',), gid_list)
    metadata_list = []
    for metadata_str in metadata_str_list:
        if metadata_str in [None, '']:
            metadata_dict = {}
        else:
            metadata_dict = metadata_str if return_raw else ut.from_json(metadata_str)
        metadata_list.append(metadata_dict)
    return metadata_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/name/rowid/', methods=['GET'])
def get_image_nids(ibs, gid_list):
    r"""

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: nids_list - the name ids associated with an image id

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_nids

    RESTful:
        Method: GET
        URL:    /api/image/name/rowid/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> nids_list = ibs.get_image_nids(gid_list)
        >>> # verify results
        >>> result = str(nids_list)
        >>> print(result)

    """
    aids_list = ibs.get_image_aids(gid_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    return nids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/name/uuid/', methods=['GET'])
def get_image_name_uuids(ibs, gid_list):
    r"""

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: name_uuids_list - the name uuids associated with an image id

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_nids

    RESTful:
        Method: GET
        URL:    /api/image/name/uuid/
    """
    nids_list = ibs.get_image_nids(gid_list)
    name_uuids_list = [ibs.get_name_uuids(nid_list) for nid_list in nids_list]
    return name_uuids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/species/rowid/', methods=['GET'], __api_plural_check__=False)
def get_image_species_rowids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): the name ids associated with an image id

    RESTful:
        Method: GET
        URL:    /api/image/species/rowid/
    """
    aids_list = ibs.get_image_aids(gid_list)
    species_rowid_list = ibs.get_annot_species_rowids(aids_list)
    return species_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/species/uuid/', methods=['GET'], __api_plural_check__=False)
def get_image_species_uuids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): the name ids associated with an image id

    RESTful:
        Method: GET
        URL:    /api/image/species/uuid/
    """
    species_rowids_list = ibs.get_image_species_rowids(gid_list)
    species_uuids_list = [
        ibs.get_species_uuids(species_rowid_list)
        for species_rowid_list in species_rowids_list
    ]
    return species_uuids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/imageset/rowid/', methods=['GET'])
@profile
def get_image_imgsetids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of imageset ids for each image by gid

    RESTful:
        Method: GET
        URL:    /api/image/imageset/rowid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    NEW_INDEX_HACK = True
    if NEW_INDEX_HACK:
        # FIXME: This index should when the database is defined.
        # Ensure that an index exists on the image column of the annotation table
        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS gs_to_gids ON {GSG_RELATION_TABLE} ({IMAGE_ROWID});
            """.format(
                GSG_RELATION_TABLE=const.GSG_RELATION_TABLE, IMAGE_ROWID=IMAGE_ROWID
            )
        ).fetchall()
    colnames = ('imageset_rowid',)
    imgsetids_list = ibs.db.get(
        const.GSG_RELATION_TABLE,
        colnames,
        gid_list,
        id_colname='image_rowid',
        unpack_scalars=False,
    )
    return imgsetids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/imageset/uuid/', methods=['GET'])
def get_image_imgset_uuids(ibs, gid_list):
    imgsetids_list = ibs.get_image_imgsetids(gid_list)
    imgset_uuids_list = [
        ibs.get_imageset_uuids(imgsetid_list) for imgsetid_list in imgsetids_list
    ]
    return imgset_uuids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/imageset/text/', methods=['GET'])
def get_image_imagesettext(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of imagesettexts for each image by gid

    RESTful:
        Method: GET
        URL:    /api/image/imageset/text/
    """
    imgsetids_list = ibs.get_image_imgsetids(gid_list)
    imagesettext_list = ibs.unflat_map(ibs.get_imageset_text, imgsetids_list)
    return imagesettext_list


@register_ibs_method
@accessor_decors.getter_1toM
@accessor_decors.cache_getter(const.IMAGE_TABLE, ANNOT_ROWIDS)
@register_api('/api/image/annot/rowid/', methods=['GET'])
def get_image_aids(ibs, gid_list, is_staged=False, __check_staged__=True):
    r"""
    Returns:
        list_ (list): a list of aids for each image by gid

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: aids_list

    CommandLine:
        python -m wbia.control.manual_image_funcs --test-get_image_aids

    RESTful:
        Method: GET
        URL:    /api/image/annot/rowid/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_annot_gids(ibs.get_valid_aids())
        >>> gid_list = gid_list + gid_list[::5]
        >>> # execute function
        >>> aids_list = get_image_aids(ibs, gid_list)
        >>> # verify results
        >>> result = str(aids_list)
        >>> print(result)
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [1], [6], [11]]


    Ignore:
        print('len(gid_list) = %r' % (len(gid_list),))
        print('len(input_list) = %r' % (len(input_list),))
        print('len(pair_list) = %r' % (len(pair_list),))
        print('len(aidscol) = %r' % (len(aidscol),))
        print('len(gidscol) = %r' % (len(gidscol),))
        print('len(unique_gids) = %r' % (len(unique_gids),))
    """
    from wbia.control.manual_annot_funcs import ANNOT_STAGED_FLAG

    # FIXME: SLOW JUST LIKE GET_NAME_AIDS
    # print('gid_list = %r' % (gid_list,))
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    NEW_INDEX_HACK = True
    USE_GROUPING_HACK = False
    if NEW_INDEX_HACK:
        # FIXME: This index should when the database is defined.
        # Ensure that an index exists on the image column of the annotation table

        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS gid_to_aids ON annotations (image_rowid);
            """
        ).fetchall()

        # The index maxes the following query very efficient
        if __check_staged__:
            params_iter = ((gid, is_staged) for gid in gid_list)
            where_colnames = (
                IMAGE_ROWID,
                ANNOT_STAGED_FLAG,
            )
        else:
            params_iter = ((gid,) for gid in gid_list)
            where_colnames = (IMAGE_ROWID,)
        aids_list = ibs.db.get_where_eq(
            ibs.const.ANNOTATION_TABLE,
            (ANNOT_ROWID,),
            params_iter,
            where_colnames,
            unpack_scalars=False,
        )
        # aids_list = [[wrapped_aids[0] for wrapped_aids in ibs.db.connection.execute(
        #    '''
        #    SELECT annot_rowid
        #    FROM annotations
        #    WHERE image_rowid = ?''', (gid,)).fetchall()
        # ]
        #    for gid in gid_list]

    elif USE_GROUPING_HACK:
        input_list, inverse_unique = np.unique(gid_list, return_inverse=True)
        # This code doesn't work because it doesn't respect empty names
        input_str = ', '.join(list(map(str, input_list)))
        opstr = """
        SELECT annot_rowid, image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE image_rowid IN
            ({input_str})
            ORDER BY image_rowid ASC, annot_rowid ASC
        """.format(
            input_str=input_str, ANNOTATION_TABLE=const.ANNOTATION_TABLE
        )
        pair_list = ibs.db.connection.execute(opstr).fetchall()
        aidscol = np.array(ut.get_list_column(pair_list, 0))
        gidscol = np.array(ut.get_list_column(pair_list, 1))
        unique_gids, groupx = vt.group_indices(gidscol)
        grouped_aids_ = vt.apply_grouping(aidscol, groupx)
        # aids_list = [sorted(arr.tolist()) for arr in grouped_aids_]
        structured_aids_list = [arr.tolist() for arr in grouped_aids_]
        with ut.EmbedOnException():
            aids_list = np.array(structured_aids_list)[inverse_unique].tolist()
    else:
        USE_NUMPY_IMPL = True
        # Use qt if getting one at a time otherwise perform bulk operation
        USE_NUMPY_IMPL = len(gid_list) > 1
        if USE_NUMPY_IMPL:
            # This seems to be 30x faster for bigger inputs
            valid_aids = np.array(ibs._get_all_aids())
            valid_gids = np.array(
                ibs.db.get_all_col_rows(const.ANNOTATION_TABLE, IMAGE_ROWID)
            )
            # np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))
            aids_list = [
                valid_aids.take(np.flatnonzero(np.equal(valid_gids, gid))).tolist()
                for gid in gid_list
            ]
        else:
            # SQL IMPL
            aids_list = ibs.db.get(
                ibs.const.ANNOTATION_TABLE,
                (ANNOT_ROWID,),
                gid_list,
                id_colname=IMAGE_ROWID,
                unpack_scalars=False,
            )
            # %timeit ibs.db.get(ibs.const.ANNOTATION_TABLE, (ANNOT_ROWID,), gid_list, id_colname=IMAGE_ROWID, unpack_scalars=False)

    if False:
        # cur = ibs.db.connection.execute(' .indices annotations;')
        # cur.fetchall()
        # aid_list3 = ibs.db.connection.execute('''
        #                               SELECT annot_rowid
        #                               FROM annotations
        #                               WHERE image_rowid IN
        #                               ({input_str})
        #                               GROUP BY image_rowid
        #                               '''.format(input_str=', '.join(list(map(str, gid_list))))
        #                              ).fetchall()
        # %timeit ibs.db.connection.execute('''SELECT annot_rowid FROM annotations WHERE image_rowid IN ({input_str}) GROUP BY image_rowid'''.format(input_str=', '.join(list(map(str, gid_list))))).fetchall()
        # aids_list3 = []
        """
        cur = ibs.db.connection.execute(
            '''
            SELECT * FROM sqlite_master WHERE type = 'index'
            ''')
        cur.fetchall()

        cur = ibs.db.connection.execute(
            '''
            CREATE INDEX IF NOT EXISTS gid_to_aids ON annotations (image_rowid);
            ''').fetchall()

        gid_list = ibs.get_valid_gids()
        gid_list_ = gid_list[0:15]
        gid_list_ = gid_list
        aids_list1 = ibs.get_image_aids(gid_list_)
        aids_list2 = [[wrapped_aids[0] for wrapped_aids in ibs.db.connection.execute(
            '''
            SELECT annot_rowid
            FROM annotations
            WHERE image_rowid = ?''', (gid,)).fetchall()] for gid in gid_list_]

        %timeit ibs.get_image_aids(gid_list_)

        %timeit [[wrapped_aids[0] for wrapped_aids in ibs.db.connection.execute('''SELECT annot_rowid FROM annotations WHERE image_rowid = ?''', (gid,)).fetchall()] for gid in gid_list_]
        """
    # print('aids_list = %r' % (aids_list,))

    # aids_list = [
    #     ibs.filter_annotation_set(aid_list_, is_staged=is_staged)
    #     for aid_list_ in aids_list
    # ]

    return aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/annot/uuid/', methods=['GET'])
def get_image_annot_uuids(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    annot_uuid_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuid_list


@register_ibs_method
@accessor_decors.getter_1toM
# @cache_getter(const.IMAGE_TABLE)
@register_api(
    '/api/image/annot/rowid/species/', methods=['GET'], __api_plural_check__=False
)
@profile
def get_image_aids_of_species(ibs, gid_list, species=None):
    r"""
    Returns:
        list_ (list): a list of aids for each image by gid filtered by species

    RESTful:
        Method: GET
        URL:    /api/image/annot/rowid/species/
    """

    def _filter(aid_list):
        species_list = ibs.get_annot_species(aid_list)
        isvalid_list = [species_ == species for species_ in species_list]
        aid_list = ut.compress(aid_list, isvalid_list)
        return aid_list

    # Get and filter aids_list
    aids_list = ibs.get_image_aids(gid_list)
    if species is None:
        # We do this so that the species flag behaves nicely with the getter_1toM
        print('[get_image_aids_of_species] WARNING! Use get_image_aids() instead.')
        return aids_list
    aids_list = [_filter(aid_list) for aid_list in aids_list]
    return aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api(
    '/api/image/annot/uuid/species/', methods=['GET'], __api_plural_check__=False
)
def get_image_annot_uuids_of_species(ibs, gid_list, **kwargs):
    aids_list = ibs.get_image_aids_of_species(gid_list, **kwargs)
    annot_uuid_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @profile
@register_api('/api/image/num/annot/', methods=['GET'])
def get_image_num_annotations(ibs, gid_list):
    r"""
    Returns:
        list_ (list): the number of chips in each image

    RESTful:
        Method: GET
        URL:    /api/image/num/annot/
    """
    return list(map(len, ibs.get_image_aids(gid_list)))


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/', methods=['DELETE'])
def delete_images(ibs, gid_list, trash_images=True):
    r"""
    deletes images from the database that belong to gids

    RESTful:
        Method: DELETE
        URL:    /api/image/

    Ignore:
        >>> # UNPORTED_DOCTEST
        >>> gpath_list = ut.get_test_gpaths(ndata=None)[0:4]
        >>> gid_list = ibs.add_images(gpath_list)
        >>> bbox_list = [(0, 0, 100, 100)] * len(gid_list)
        >>> name_list = ['a', 'b', 'a', 'd']
        >>> aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list, name_list=name_list)
        >>> gid = gid_list[0]
        >>> assert gid is not None, "gid is None"
        >>> aid_list = ibs.get_image_aids(gid)
        >>> assert len(aid_list) == 1, "Length of aid_list=%r" % (len(aid_list),)
        >>> aid = aid_list[0]
        >>> assert aid is not None, "aid is None"
        >>> cid = ibs.get_annot_chip_rowids(aid, ensure=False)
        >>> fid = ibs.get_annot_feat_rowids(aid, ensure=False)
        >>> assert cid is None, "cid=%r should be None" % (cid,)
        >>> assert fid is None, "fid=%r should be None" % (fid,)
        >>> cid = ibs.get_annot_chip_rowids(aid, ensure=True)
        >>> fid = ibs.get_annot_feat_rowids(aid, ensure=True)
        >>> assert cid is not None, "cid should be computed"
        >>> assert fid is not None, "fid should be computed"
        >>> gthumbpath = ibs.get_image_thumbpath(gid)
        >>> athumbpath = ibs.get_annot_chip_thumbpath(aid)
        >>> ibs.delete_images(gid)
        >>> all_gids = ibs.get_valid_gids()
        >>> all_aids = ibs.get_valid_aids()
        >>> all_cids = ibs.get_valid_cids()
        >>> all_fids = ibs.get_valid_fids()
        >>> assert gid not in all_gids, "gid still exists"
        >>> assert aid not in all_aids, "rid %r still exists" % aid
        >>> assert fid not in all_fids, "fid %r still exists" % fid
        >>> assert cid not in all_cids, "cid %r still exists" % cid
        >>> assert not utool.checkpath(gthumbpath), "Thumbnail still exists"
        >>> assert not utool.checkpath(athumbpath), "ANNOTATION Thumbnail still exists"
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d images' % len(gid_list))
    # Move images to trash before deleting them. #
    # TODO: only move localized images
    # TODO: ensure there are no name conflicts when using the original names
    gpath_list = ibs.get_image_paths(gid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    ext_list = ibs.get_image_exts(gid_list)
    if trash_images:
        trash_dir = ibs.get_trashdir()
        ut.ensuredir(trash_dir)
        gpath_list2 = [
            join(trash_dir, gname + ext) for (gname, ext) in zip(gname_list, ext_list)
        ]
        ut.copy_list(
            gpath_list, gpath_list2, ioerr_ok=True, oserror_ok=True, lbl='Trashing Images'
        )
    for gpath in gpath_list:
        ut.delete(gpath)
        # raise NotImplementedError('must trash images for now')
    # ut.view_directory(trash_dir)

    # Delete annotations first
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    ibs.delete_annots(aid_list)
    # delete thumbs in case an annot doesnt delete them
    # TODO: pass flag to not delete them in delete_annots
    gid_list = list(set(gid_list))
    ibs.delete_image_thumbs(gid_list)
    ibs.depc_image.delete_root(gid_list)
    ibs.db.delete_rowids(const.IMAGE_TABLE, gid_list)
    # gsgrid_list = ut.flatten(ibs.get_image_gsgrids(gid_list))
    # ibs.db.delete_rowids(const.GSG_RELATION_TABLE, gsgrid_list)
    ibs.db.delete(const.GSG_RELATION_TABLE, gid_list, id_colname='image_rowid')


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/image/thumb/', methods=['DELETE'])
def delete_image_thumbs(ibs, gid_list, **config2_):
    r"""
    Removes image thumbnails from disk

    RESTful:
        Method: DELETE
        URL:    /api/image/thumb/

    Ignore:
        >>> # UNPORTED_DOCTEST
        >>> gpath_list = ut.get_test_gpaths(ndata=None)[0:4]
        >>> gid_list = ibs.add_images(gpath_list)
        >>> bbox_list = [(0, 0, 100, 100)] * len(gid_list)
        >>> name_list = ['a', 'b', 'a', 'd']
        >>> aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list,
        >>>                           name_list=name_list)
        >>> assert len(aid_list) != 0, "No annotations added"
        >>> thumbpath_list = ibs.get_image_thumbpath(gid_list)
        >>> gpath_list = ibs.get_image_paths(gid_list)
        >>> ibs.delete_image_thumbs(gid_list)
        >>> assert utool.is_list(thumbpath_list), "thumbpath_list is not a list"
        >>> assert utool.is_list(gpath_list), "gpath_list is not a list"
        >>> for path in thumbpath_list:
        >>>     assert not utool.checkpath(path), "Thumbnail not deleted"
        >>> for path in gpath_list:
        >>>     utool.assertpath(path)
    """
    ibs.depc_image.delete_property_all('thumbnails', gid_list)
    ibs.depc_image.delete_property_all('web_src', gid_list)

    # if ut.VERBOSE:
    #     print('[ibs] deleting %d image thumbnails' % len(gid_list))
    #     if DEBUG_THUMB:
    #         print('{THUMB DELETE} config2_ = %r' % (config2_,))

    # # TODO: delete all configs?
    # gid_list = list(set(gid_list))
    # num_deleted = ibs.depc_image.delete_property('thumbnails', gid_list,
    #                                              config=config2_)

    # # HACK: Remove paths computed by QT and not the depcache.
    # thumbpath_list = ibs.get_image_thumbpath(gid_list, **config2_)
    # #print('thumbpath_list = %r' % (thumbpath_list,))
    # #ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='image_thumbs')
    # ut.remove_existing_fpaths(thumbpath_list, quiet=True,
    #                           lbl='image_thumbs')

    # if DEBUG_THUMB:
    #     print('num_deleted = %r' % (num_deleted,))
    #     print('{THUMB DELETE} DONE DELETE')


@register_ibs_method
# @accessor_decors.cache_getter(const.IMAGE_TABLE, IMAGE_TIMEDELTA_POSIX)
@register_api('/api/image/timedelta/posix/', methods=['GET'])
def get_image_timedelta_posix(ibs, gid_list, eager=True):
    r"""
    image_timedelta_posix_list <- image.image_timedelta_posix[gid_list]

    # TODO: INTEGRATE THIS FUNCTION. CURRENTLY OFFSETS ARE ENCODIED DIRECTLY IN UNIXTIME

    gets data from the "native" column "image_timedelta_posix" in the "image" table

    Args:
        gid_list (list):

    Returns:
        list: image_timedelta_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = image_timedelta_posix
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/timedelta/posix/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> image_timedelta_posix_list = ibs.get_image_timedelta_posix(gid_list, eager=eager)
        >>> assert len(gid_list) == len(image_timedelta_posix_list)
    """
    id_iter = gid_list
    colnames = (IMAGE_TIMEDELTA_POSIX,)
    image_timedelta_posix_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager
    )
    return image_timedelta_posix_list


@register_ibs_method
@register_api('/api/image/timedelta/posix/', methods=['PUT'])
def set_image_timedelta_posix(
    ibs, gid_list, image_timedelta_posix_list, duplicate_behavior='error'
):
    r"""
    image_timedelta_posix_list -> image.image_timedelta_posix[gid_list]

    Args:
        gid_list
        image_timedelta_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_timedelta_posix

    RESTful:
        Method: PUT
        URL:    /api/image/timedelta/posix/
    """
    id_iter = gid_list
    colnames = (IMAGE_TIMEDELTA_POSIX,)
    ibs.db.set(
        const.IMAGE_TABLE,
        colnames,
        image_timedelta_posix_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
# @accessor_decors.cache_getter(const.IMAGE_TABLE, IMAGE_LOCATION_CODE)
@register_api('/api/image/location/code/', methods=['GET'])
def get_image_location_codes(ibs, gid_list, eager=True):
    r"""
    image_location_code_list <- image.image_location_code[gid_list]

    gets data from the "native" column "image_location_code" in the "image" table

    Args:
        gid_list (list):

    Returns:
        list: image_location_code_list

    TemplateInfo:
        Tgetter_table_column
        col = image_location_code
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/location/code/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> image_location_code_list = ibs.get_image_location_codes(gid_list, eager=eager)
        >>> assert len(gid_list) == len(image_location_code_list)
    """
    id_iter = gid_list
    colnames = (IMAGE_LOCATION_CODE,)
    image_location_code_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager
    )
    return image_location_code_list


@register_ibs_method
@register_api('/api/image/location/code/', methods=['PUT'])
def set_image_location_codes(
    ibs, gid_list, image_location_code_list, duplicate_behavior='error'
):
    r"""
    image_location_code_list -> image.image_location_code[gid_list]

    Args:
        gid_list
        image_location_code_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_location_code

    RESTful:
        Method: PUT
        URL:    /api/image/location/code/
    """
    id_iter = gid_list
    colnames = (IMAGE_LOCATION_CODE,)
    ibs.db.set(
        const.IMAGE_TABLE,
        colnames,
        image_location_code_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/party/rowid/', methods=['GET'])
def get_image_party_rowids(ibs, gid_list, eager=True, nInput=None):
    r"""
    party_rowid_list <- image.party_rowid[gid_list]

    gets data from the "native" column "party_rowid" in the "image" table

    Args:
        gid_list (list):

    Returns:
        list: party_rowid_list

    TemplateInfo:
        Tgetter_table_column
        col = party_rowid
        tbl = image

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> party_rowid_list = ibs.get_image_party_rowids(gid_list, eager=eager)
        >>> assert len(gid_list) == len(party_rowid_list)
    """
    id_iter = gid_list
    colnames = (PARTY_ROWID,)
    party_rowid_list = ibs.db.get(
        const.IMAGE_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return party_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/image/party/tag/', methods=['GET'])
def get_image_party_tag(ibs, gid_list, eager=True, nInput=None):
    r"""
    party_tag_list <- image.party_tag[gid_list]

    Args:
        gid_list (list):

    Returns:
        list: party_tag_list

    TemplateInfo:
        Tgetter_extern
        tbl = image
        externtbl = party
        externcol = party_tag

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> party_tag_list = ibs.get_image_party_tag(gid_list, eager=eager)
        >>> assert len(gid_list) == len(party_tag_list)
    """
    party_rowid_list = ibs.get_image_party_rowids(gid_list, eager=eager, nInput=nInput)
    party_tag_list = ibs.get_party_tag(party_rowid_list, eager=eager, nInput=nInput)
    return party_tag_list


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/image/party/rowid/', methods=['PUT'])
def set_image_party_rowids(ibs, gid_list, party_rowid_list, duplicate_behavior='error'):
    r"""
    party_rowid_list -> image.party_rowid[gid_list]

    Args:
        gid_list
        party_rowid_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = party_rowid
    """
    id_iter = gid_list
    colnames = (PARTY_ROWID,)
    ibs.db.set(
        const.IMAGE_TABLE,
        colnames,
        party_rowid_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/contributor/tag/', methods=['GET'])
def get_image_contributor_tag(ibs, gid_list, eager=True, nInput=None):
    r"""
    contributor_tag_list <- image.contributor_tag[gid_list]

    Args:
        gid_list (list):

    Returns:
        list: contributor_tag_list

    TemplateInfo:
        Tgetter_extern
        tbl = image
        externtbl = contributor
        externcol = contributor_tag

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> gid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> contributor_tag_list = ibs.get_image_contributor_tag(gid_list, eager=eager)
        >>> assert len(gid_list) == len(contributor_tag_list)
    """
    contributor_rowid_list = ibs.get_image_contributor_rowid(
        gid_list, eager=eager, nInput=nInput
    )
    contributor_tag_list = ibs.get_contributor_tag(
        contributor_rowid_list, eager=eager, nInput=nInput
    )
    return contributor_tag_list


def testdata_ibs():
    r"""
    """
    import wbia

    ibs = wbia.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_image_funcs
        python -m wbia.control.manual_image_funcs --allexamples
        python -m wbia.control.manual_image_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
