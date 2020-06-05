# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function

# from os.path import splitext, basename
import uuid
import six
from wbia.web.routes_ajax import image_src
from wbia.control import controller_inject
import utool as ut
import wbia.constants as const

(print, rrr, profile) = ut.inject2(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)


@register_api('/api/imageset/json/', methods=['POST'])
def add_imagesets_json(
    ibs,
    imageset_text_list,
    imageset_uuid_list=None,
    config_rowid_list=None,
    imageset_notes_list=None,
    imageset_occurence_flag_list=None,
):
    r"""
    Adds a list of imagesets.

    Args:
        imagesettext_list (list):
        imageset_uuid_list (list):
        config_rowid_list (list):
        notes_list (list):

    Returns:
        imageset_uuid_list (list): added imageset uuids

    RESTful:
        Method: POST
        URL:    /api/imageset/json/
    """
    imageset_rowid_list = ibs.add_imagesets(
        imageset_text_list,
        imageset_uuid_list=imageset_uuid_list,
        occurence_flag_list=imageset_occurence_flag_list,
        config_rowid_list=config_rowid_list,
        notes_list=imageset_notes_list,
    )
    imageset_uuid_list = ibs.get_imageset_uuid(imageset_rowid_list)
    return imageset_uuid_list


# @register_api('/api/image/json/', methods=['POST'])
# def add_images_json(ibs, image_uri_list, image_uuid_list, image_width_list,
#                     image_height_list, image_orig_name_list=None, image_ext_list=None,
#                     image_time_posix_list=None, image_gps_lat_list=None,
#                     image_gps_lon_list=None, image_orientation_list=None,
#                     image_notes_list=None, **kwargs):
#     """
#     REST:
#         Method: POST
#         URL: /api/image/json/

#     Ignore:
#         sudo pip install boto

#     Args:
#         image_uri_list (list) : list of string image uris, most likely HTTP(S) or S3
#             encoded URLs.  Alternatively, this can be a list of dictionaries (JSON
#             objects) that specify AWS S3 stored assets.  An example below:

#                 image_uri_list = [
#                     'http://domain.com/example/asset1.png',
#                     '/home/example/Desktop/example/asset2.jpg',
#                     's3://s3.amazon.com/example-bucket-2/asset1-in-bucket-2.tif',
#                     {
#                         'bucket'          : 'example-bucket-1',
#                         'key'             : 'example/asset1.png',
#                         'auth_domain'     : None,  # Uses 127.0.0.1
#                         'auth_access_id'  : None,  # Uses system default
#                         'auth_secret_key' : None,  # Uses system default
#                     },
#                     {
#                         'bucket' : 'example-bucket-1',
#                         'key'    : 'example/asset2.jpg',
#                         # if unspecified, auth uses 127.0.0.1 and system defaults
#                     },
#                     {
#                         'bucket'          : 'example-bucket-2',
#                         'key'             : 'example/asset1-in-bucket-2.tif',
#                         'auth_domain'     : 's3.amazon.com',
#                         'auth_access_id'  : '____________________',
#                         'auth_secret_key' : '________________________________________',
#                     },
#                 ]

#             Note that you cannot specify AWS authentication access ids or secret keys
#             using string uri's.  For specific authentication methods, please use the
#             latter list of dictionaries.

#         image_uuid_list (list of str) : list of image UUIDs to be used in IBEIS IA
#         image_width_list (list of int) : list of image widths
#         image_height_list (list of int) : list of image heights
#         image_orig_name_list (list of str): list of original image names
#         image_ext_list (list of str): list of original image names
#         image_time_posix_list (list of int): list of image's POSIX timestamps
#         image_gps_lat_list (list of float): list of image's GPS latitude values
#         image_gps_lon_list (list of float): list of image's GPS longitude values
#         image_orientation_list (list of int): list of image's orientation flags
#         image_notes_list (list of str) : optional list of any related notes with
#             the images
#         **kwargs : key-value pairs passed to the ibs.add_images() function.

#     CommandLine:
#         python -m wbia.web.apis_json --test-add_images_json


#         ,"bucket":"flukebook-prod-asset-store","key":""

#     Example:
#         >>> # WEB_DOCTEST
#         >>> from wbia.control.IBEISControl import *  # NOQA
#         >>> import wbia
#         >>> import uuid
#         >>> web_instance = wbia.opendb(db='testdb1')
#         >>> _payload = {
#         >>>     'image_uri_list': [
#         >>>         'https://upload.wikimedia.org/wikipedia/commons/4/49/Zebra_running_Ngorongoro.jpg',
#         >>>         {
#         >>>             'bucket'          : 'test-asset-store',
#         >>>             'key'             : 'caribwhale/20130903-JAC-0002.JPG',
#         >>>         },
#         >>>         {
#         >>>             'bucket'          : 'flukebook-prod-asset-store',
#         >>>             'key'             : '3/a/3a76b0e8-1c64-403d-ace1-679cf2f081c0/f2.jpg',
#         >>>         },
#         >>>     ],
#         >>>     'image_uuid_list': [
#         >>>         uuid.uuid4(),
#         >>>         uuid.uuid4(),
#         >>>         uuid.uuid4(),
#         >>>     ],
#         >>>     'image_width_list': [
#         >>>         1992,
#         >>>         1194,
#         >>>         500,
#         >>>     ],
#         >>>     'image_height_list': [
#         >>>         1328,
#         >>>         401,
#         >>>         500,
#         >>>     ],
#         >>> }
#         >>> gid_list = wbia.web.apis_json.add_images_json(web_instance, **_payload)
#         >>> print(gid_list)
#         >>> print(web_instance.get_image_uuids(gid_list))
#         >>> print(web_instance.get_image_uris(gid_list))
#         >>> print(web_instance.get_image_paths(gid_list))
#         >>> print(web_instance.get_image_uris_original(gid_list))
#     """
#     def _rectify(list_, default, length, func=None):
#         if list_ is None:
#             list_ = [None] * length

#         ret_list = []
#         for item in list_:

#             if item is None:
#                 item = default

#             if None not in [func, item]:
#                 item = func(item)

#             ret_list.append(item)

#         return ret_list

#     def _rectify_uri(list_, default, length, func=None):
#         list_ = _rectify(list_, default, length, func=None)

#         ret_list = []
#         for item in list_:

#             if isinstance(item, dict):
#                 item = ut.s3_dict_encode_to_str(item)

#             if ibs.containerized and item is not None:
#                 item = item.replace('://localhost/', '://nginx:80/')

#             ret_list.append(item)

#         return ret_list

#     def _verify(list_, tag, length, allow_none=False):
#         length_ = len(list_)
#         if length_ != length:
#             message = 'The input list %s has the wrong length. Received: %d. Expected %d'
#             args = (tag, length_, length, )
#             raise ValueError(message % args)

#         error_list = []
#         for value in enumerate(list_):
#             index, item = value
#             if item is None:
#                 error_list.append(value)

#         if len(error_list) > 0:
#             message = 'The input list %s has invalid values (index, value): %r'
#             args = (tag, error_list, )
#             raise ValueError(message % args)

#     def _uuid(value):
#         import uuid
#         import six

#         if value is None:
#             return None

#         if isinstance(value, six.string_types):
#             value = uuid.UUID(value)

#         return value

#     def _base(value):
#         if value is None:
#             return None

#         return basename(value)

#     def _ext(value):
#         if value is None:
#             return None

#         value = splitext(value)[1].lower()
#         value = '.jpg' if value == '.jpeg' else value
#         return value

#     # TODO: FIX ME SO THAT WE DON'T HAVE TO LOCALIZE EVERYTHING
#     kwargs['auto_localize'] = kwargs.get('auto_localize', True)
#     kwargs['sanitize'] = kwargs.get('sanitize', False)

#     expected_length = len(image_uri_list)

#     # Rectify values
#     image_uri_list         = _rectify_uri(image_uri_list    , None, expected_length,   str)
#     image_uuid_list        = _rectify(image_uuid_list       , None, expected_length, _uuid)
#     image_width_list       = _rectify(image_width_list      , None, expected_length,   int)
#     image_height_list      = _rectify(image_height_list     , None, expected_length,   int)
#     image_orig_name_list   = _rectify(image_uri_list        , None, expected_length, _base)
#     image_ext_list         = _rectify(image_uri_list        , None, expected_length,  _ext)
#     image_time_posix_list  = _rectify(image_time_posix_list ,   -1, expected_length, float)
#     image_gps_lat_list     = _rectify(image_gps_lat_list    , -1.0, expected_length, float)
#     image_gps_lon_list     = _rectify(image_gps_lon_list    , -1.0, expected_length, float)
#     image_orientation_list = _rectify(image_orientation_list,  0.0, expected_length,   int)
#     image_notes_list       = _rectify(image_notes_list      ,   '', expected_length,   str)

#     # Verify values
#     image_uri_list         = _verify(image_uri_list        , 'image_uri_list'        , expected_length)
#     image_uuid_list        = _verify(image_uuid_list       , 'image_uuid_list'       , expected_length)
#     image_width_list       = _verify(image_width_list      , 'image_width_list'      , expected_length)
#     image_height_list      = _verify(image_height_list     , 'image_height_list'     , expected_length)
#     image_orig_name_list   = _verify(image_orig_name_list  , 'image_orig_name_list'  , expected_length)
#     image_ext_list         = _verify(image_ext_list        , 'image_ext_list'        , expected_length)
#     image_time_posix_list  = _verify(image_time_posix_list , 'image_time_posix_list' , expected_length)
#     image_gps_lat_list     = _verify(image_gps_lat_list    , 'image_gps_lat_list'    , expected_length)
#     image_gps_lon_list     = _verify(image_gps_lon_list    , 'image_gps_lon_list'    , expected_length)
#     image_orientation_list = _verify(image_orientation_list, 'image_orientation_list', expected_length)
#     image_notes_list       = _verify(image_notes_list      , 'image_notes_list'      , expected_length)

#     params_gen = zip(
#         image_uuid_list,
#         image_uri_list,
#         image_uri_list,
#         image_orig_name_list,
#         image_ext_list,
#         image_width_list,
#         image_height_list,
#         image_time_posix_list,
#         image_gps_lat_list,
#         image_gps_lon_list,
#         image_orientation_list,
#         image_notes_list
#     )

#     gid_list = ibs.add_images(image_uri_list, params_list=params_gen, **kwargs)  # NOQA
#     image_uuid_list = ibs.get_image_uuids(gid_list)
#     return image_uuid_list


@register_api('/api/image/json/', methods=['POST'])
def add_images_json(
    ibs,
    image_uri_list,
    image_unixtime_list=None,
    image_gps_lat_list=None,
    image_gps_lon_list=None,
    **kwargs,
):
    """
    REST:
        Method: POST
        URL: /api/image/json/

    Ignore:
        sudo pip install boto

    Args:
        image_uri_list (list) : list of string image uris, most likely HTTP(S) or S3
            encoded URLs.  Alternatively, this can be a list of dictionaries (JSON
            objects) that specify AWS S3 stored assets.  An example below:

                image_uri_list = [
                    'http://domain.com/example/asset1.png',
                    '/home/example/Desktop/example/asset2.jpg',
                    's3://s3.amazon.com/example-bucket-2/asset1-in-bucket-2.tif',
                    {
                        'bucket'          : 'example-bucket-1',
                        'key'             : 'example/asset1.png',
                        'auth_domain'     : None,  # Uses 127.0.0.1
                        'auth_access_id'  : None,  # Uses system default
                        'auth_secret_key' : None,  # Uses system default
                    },
                    {
                        'bucket' : 'example-bucket-1',
                        'key'    : 'example/asset2.jpg',
                        # if unspecified, auth uses 127.0.0.1 and system defaults
                    },
                    {
                        'bucket'          : 'example-bucket-2',
                        'key'             : 'example/asset1-in-bucket-2.tif',
                        'auth_domain'     : 's3.amazon.com',
                        'auth_access_id'  : '____________________',
                        'auth_secret_key' : '________________________________________',
                    },
                ]

            Note that you cannot specify AWS authentication access ids or secret keys
            using string uri's.  For specific authentication methods, please use the
            latter list of dictionaries.

        image_time_posix_list (list of int): list of image's POSIX timestamps
        image_gps_lat_list (list of float): list of image's GPS latitude values
        image_gps_lon_list (list of float): list of image's GPS longitude values
        **kwargs : key-value pairs passed to the ibs.add_images() function.

    CommandLine:
        python -m wbia.web.apis_json --test-add_images_json


        ,"bucket":"flukebook-prod-asset-store","key":""

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia
        >>> import uuid
        >>> web_instance = wbia.opendb(db='testdb1')
        >>> _payload = {
        >>>     'image_uri_list': [
        >>>         'https://upload.wikimedia.org/wikipedia/commons/4/49/Zebra_running_Ngorongoro.jpg',
        >>>         {
        >>>             'bucket'          : 'test-asset-store',
        >>>             'key'             : 'caribwhale/20130903-JAC-0002.JPG',
        >>>         },
        >>>         {
        >>>             'bucket'          : 'flukebook-prod-asset-store',
        >>>             'key'             : '3/a/3a76b0e8-1c64-403d-ace1-679cf2f081c0/f2.jpg',
        >>>         },
        >>>     ],
        >>>     'image_uuid_list': [
        >>>         uuid.uuid4(),
        >>>         uuid.uuid4(),
        >>>         uuid.uuid4(),
        >>>     ],
        >>>     'image_width_list': [
        >>>         1992,
        >>>         1194,
        >>>         500,
        >>>     ],
        >>>     'image_height_list': [
        >>>         1328,
        >>>         401,
        >>>         500,
        >>>     ],
        >>> }
        >>> gid_list = wbia.web.apis_json.add_images_json(web_instance, **_payload)
        >>> print(gid_list)
        >>> print(web_instance.get_image_uuids(gid_list))
        >>> print(web_instance.get_image_uris(gid_list))
        >>> print(web_instance.get_image_paths(gid_list))
        >>> print(web_instance.get_image_uris_original(gid_list))
    """

    def _rectify(list_, default, length, func=None):
        if list_ is None:
            list_ = [None] * length

        ret_list = []
        for item in list_:

            if item is None:
                item = default

            if None not in [func, item]:
                item = func(item)

            ret_list.append(item)

        return ret_list

    def _rectify_uri(list_, default, length, func=str):
        list_ = _rectify(list_, default, length, func=None)

        ret_list = []
        for item in list_:

            if isinstance(item, dict):
                item = ut.s3_dict_encode_to_str(item)

            if ibs.containerized and item is not None:
                item = item.replace('://localhost/', '://nginx:80/')

            ret_list.append(item)

        return ret_list

    def _verify(list_, tag, length, allow_none=False):
        length_ = len(list_)
        if length_ != length:
            message = 'The input list %s has the wrong length. Received: %d. Expected %d'
            args = (
                tag,
                length_,
                length,
            )
            raise ValueError(message % args)

        error_list = []
        for value in enumerate(list_):
            index, item = value
            if item is None and not allow_none:
                error_list.append(value)

        if len(error_list) > 0:
            message = 'The input list %s has invalid values (index, value): %r'
            args = (
                tag,
                error_list,
            )
            raise ValueError(message % args)

        return list_

    kwargs['auto_localize'] = kwargs.get('auto_localize', True)
    kwargs['sanitize'] = kwargs.get('sanitize', False)

    depricated_list = [
        'image_uuid_list',
        'image_width_list',
        'image_height_list',
        'image_orig_name_list',
        'image_ext_list',
        'image_time_posix_list',
        'image_orientation_list',
        'image_notes_list',
    ]

    bad_list = []
    for depricated_value in depricated_list:
        if depricated_value in kwargs:
            bad_list.append(depricated_value)

    if len(bad_list) > 0:
        raise ValueError(
            'This API signature has changed, the following parameters have been deprecated: %r.  Please remove them and try again.'
            % (bad_list,)
        )

    expected_length = len(image_uri_list)

    # Rectify values
    image_uri_list = _rectify_uri(image_uri_list, None, expected_length, str)
    image_uri_list = _verify(image_uri_list, 'image_uri_list', expected_length)
    gid_list = ibs.add_images(image_uri_list, **kwargs)  # NOQA

    if image_unixtime_list is not None:
        image_unixtime_list = _rectify(image_unixtime_list, -1, expected_length, float)
        image_unixtime_list = _verify(
            image_unixtime_list, 'image_unixtime_list', expected_length, allow_none=True
        )

        flag_list = [
            None not in [gid, image_unixtime]
            for gid, image_unixtime in zip(gid_list, image_unixtime_list)
        ]
        gid_list_ = ut.filter_items(gid_list, flag_list)
        image_unixtime_list_ = ut.filter_items(image_unixtime_list, flag_list)

        print('Setting times: %r -> %r' % (gid_list_, image_unixtime_list_,))
        ibs.set_image_unixtime(gid_list_, image_unixtime_list_)

    if image_gps_lat_list is not None and image_gps_lon_list is not None:
        image_gps_lat_list = _rectify(image_gps_lat_list, -1.0, expected_length, float)
        image_gps_lon_list = _rectify(image_gps_lon_list, -1.0, expected_length, float)
        image_gps_lat_list = _verify(
            image_gps_lat_list, 'image_gps_lat_list', expected_length, allow_none=True
        )
        image_gps_lon_list = _verify(
            image_gps_lon_list, 'image_gps_lon_list', expected_length, allow_none=True
        )

        for index, value in enumerate(zip(image_gps_lat_list, image_gps_lon_list)):
            image_gps_lat, image_gps_lon = value
            if image_gps_lat is not None:
                assert (
                    image_gps_lon is not None
                ), 'Cannot specify a longitude without a latitude, index %d' % (index,)
            if image_gps_lon is not None:
                assert (
                    image_gps_lat is not None
                ), 'Cannot specify a longitude without a latitude, index %d' % (index,)

        flag_list = [
            None not in [gid, image_gps_lat_, image_gps_lon_]
            for gid, image_gps_lat_, image_gps_lon_ in zip(
                gid_list, image_gps_lat_list, image_gps_lon_list
            )
        ]
        gid_list_ = ut.filter_items(gid_list, flag_list)
        image_gps_lat_list_ = ut.filter_items(image_gps_lat_list, flag_list)
        image_gps_lon_list_ = ut.filter_items(image_gps_lon_list, flag_list)

        print(
            'Setting gps: %r -> %r, %r'
            % (gid_list_, image_gps_lat_list_, image_gps_lon_list_,)
        )
        ibs.set_image_gps(
            gid_list_, lat_list=image_gps_lat_list_, lon_list=image_gps_lon_list_
        )

    image_uuid_list = ibs.get_image_uuids(gid_list)
    return image_uuid_list


class ParseError(object):
    def __init__(self, value):
        self.value = value


@register_api('/api/annot/json/', methods=['POST'])
def add_annots_json(
    ibs,
    image_uuid_list,
    annot_bbox_list,
    annot_theta_list,
    annot_viewpoint_list=None,
    annot_quality_list=None,
    annot_species_list=None,
    annot_multiple_list=None,
    annot_interest_list=None,
    annot_name_list=None,
    **kwargs,
):
    """
    REST:
        Method: POST
        URL: /api/annot/json/

    Ignore:
        sudo pip install boto

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be used in IBEIS IA
        annot_bbox_list (list of 4-tuple) : list of bounding box coordinates encoded as
            a 4-tuple of the values (xtl, ytl, width, height) where xtl is the
            'top left corner, x value' and ytl is the 'top left corner, y value'.
        annot_theta_list (list of float) : list of radian rotation around center.
            Defaults to 0.0 (no rotation).
        annot_species_list (list of str) : list of species for the annotation, if known.
            If the list is partially known, use None (null in JSON) for unknown entries.
        annot_name_list (list of str) : list of names for the annotation, if known.
            If the list is partially known, use None (null in JSON) for unknown entries.
        **kwargs : key-value pairs passed to the ibs.add_annots() function.

    CommandLine:
        python -m wbia.web.app --test-add_annots_json

    Example:
        >>> # DISABLE_DOCTEST
        >>> import wbia
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> web_instance = wbia.opendb(db='testdb1')
        >>> _payload = {
        >>>     'image_uuid_list': [
        >>>         uuid.UUID('7fea8101-7dec-44e3-bf5d-b8287fd231e2'),
        >>>         uuid.UUID('c081119a-e08e-4863-a710-3210171d27d6'),
        >>>     ],
        >>>     'annot_uuid_list': [
        >>>         uuid.UUID('fe1547c5-1425-4757-9b8f-b2b4a47f552d'),
        >>>         uuid.UUID('86d3959f-7167-4822-b99f-42d453a50745'),
        >>>     ],
        >>>     'annot_bbox_list': [
        >>>         [0, 0, 1992, 1328],
        >>>         [0, 0, 1194, 401],
        >>>     ],
        >>> }
        >>> aid_list = wbia.web.app.add_annots_json(web_instance, **_payload)
        >>> print(aid_list)
        >>> print(web_instance.get_annot_image_uuids(aid_list))
        >>> print(web_instance.get_annot_uuids(aid_list))
        >>> print(web_instance.get_annot_bboxes(aid_list))
    """

    def _rectify(list_, default, length, func=None):
        if list_ is None:
            list_ = [None] * length

        ret_list = []
        for item in list_:

            if item is None:
                item = default

            if None not in [func, item]:
                item = func(item)

            ret_list.append(item)

        return ret_list

    def _verify(list_, tag, length, allow_none=False):
        length_ = len(list_)
        if length_ != length:
            message = 'The input list %s has the wrong length. Received: %d. Expected %d'
            args = (
                tag,
                length_,
                length,
            )
            raise ValueError(message % args)

        error_list = []
        for value in enumerate(list_):
            index, item = value
            if item is None and not allow_none:
                error_list.append(value)
            if isinstance(item, ParseError):
                value = (index, item.value)
                error_list.append(value)

        if len(error_list) > 0:
            message = 'The input list %s has invalid values (index, value): %r'
            args = (
                tag,
                error_list,
            )
            raise ValueError(message % args)

        return list_

    def _uuid(value):
        if value is None:
            return ParseError(value)

        if isinstance(value, six.string_types):
            value = uuid.UUID(value)

        return value

    def _bbox(value):
        if len(value) != 4:
            return ParseError(value)

        value = tuple(map(float, value))

        return value

    depricated_list = [
        'annot_uuid_list',
        'annot_notes_list',
    ]

    bad_list = []
    for depricated_value in depricated_list:
        if depricated_value in kwargs:
            bad_list.append(depricated_value)

    if len(bad_list) > 0:
        raise ValueError(
            'This API signature has changed, the following parameters have been deprecated: %r.  Please remove them and try again.'
            % (bad_list,)
        )

    expected_length = len(image_uuid_list)

    image_uuid_list = _rectify(image_uuid_list, None, expected_length, _uuid)
    annot_bbox_list = _rectify(annot_bbox_list, None, expected_length, _bbox)
    annot_theta_list = _rectify(annot_theta_list, None, expected_length, float)

    image_uuid_list = _verify(image_uuid_list, 'image_uuid_list', expected_length)
    annot_bbox_list = _verify(annot_bbox_list, 'annot_bbox_list', expected_length)
    annot_theta_list = _verify(annot_theta_list, 'annot_theta_list', expected_length)

    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    gid_list = _verify(gid_list, 'image_uuid_list', expected_length)

    aid_list = ibs.add_annots(
        gid_list, bbox_list=annot_bbox_list, theta_list=annot_theta_list
    )

    if annot_viewpoint_list is not None:
        annot_viewpoint_list = _rectify(
            annot_viewpoint_list, const.VIEW.UNKNOWN, expected_length, str
        )
        annot_viewpoint_list = _verify(
            annot_viewpoint_list,
            'annot_viewpoint_list',
            expected_length,
            allow_none=True,
        )
        flag_list = [
            annot_viewpoint is not None for annot_viewpoint in annot_viewpoint_list
        ]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_viewpoint_list_ = ut.filter_items(annot_viewpoint_list, flag_list)
        ibs.set_annot_viewpoints(aid_list_, annot_viewpoint_list_)

    if annot_quality_list is not None:
        annot_quality_list = _rectify(
            annot_quality_list, const.QUAL_UNKNOWN, expected_length, str
        )
        annot_quality_list = _verify(
            annot_quality_list, 'annot_quality_list', expected_length, allow_none=True
        )
        flag_list = [annot_quality is not None for annot_quality in annot_quality_list]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_quality_list_ = ut.filter_items(annot_quality_list, flag_list)
        ibs.set_annot_quality_texts(aid_list_, annot_quality_list_)

    if annot_species_list is not None:
        annot_species_list = _rectify(
            annot_species_list, const.UNKNOWN, expected_length, str
        )
        annot_species_list = _verify(
            annot_species_list, 'annot_species_list', expected_length, allow_none=True
        )
        flag_list = [annot_species is not None for annot_species in annot_species_list]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_species_list_ = ut.filter_items(annot_species_list, flag_list)
        ibs.set_annot_species(aid_list_, annot_species_list_)

    if annot_multiple_list is not None:
        annot_multiple_list = _rectify(annot_multiple_list, False, expected_length, bool)
        annot_multiple_list = _verify(
            annot_multiple_list, 'annot_multiple_list', expected_length, allow_none=True
        )
        flag_list = [annot_multiple is not None for annot_multiple in annot_multiple_list]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_multiple_list_ = ut.filter_items(annot_multiple_list, flag_list)
        ibs.set_annot_multiple(aid_list_, annot_multiple_list_)

    if annot_interest_list is not None:
        annot_interest_list = _rectify(annot_interest_list, False, expected_length, bool)
        annot_interest_list = _verify(
            annot_interest_list, 'annot_interest_list', expected_length, allow_none=True
        )
        flag_list = [annot_interest is not None for annot_interest in annot_interest_list]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_interest_list_ = ut.filter_items(annot_interest_list, flag_list)
        ibs.set_annot_interest(aid_list_, annot_interest_list_)

    if annot_name_list is not None:
        annot_name_list = _rectify(annot_name_list, const.UNKNOWN, expected_length, str)
        annot_name_list = _verify(
            annot_name_list, 'annot_name_list', expected_length, allow_none=True
        )
        flag_list = [annot_name is not None for annot_name in annot_name_list]
        aid_list_ = ut.filter_items(aid_list, flag_list)
        annot_name_list_ = ut.filter_items(annot_name_list, flag_list)
        ibs.set_annot_names(aid_list_, annot_name_list_)

    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    return annot_uuid_list


@register_api('/api/part/json/', methods=['POST'])
def add_parts_json(ibs, annot_uuid_list, part_bbox_list, part_theta_list, **kwargs):
    """
    REST:
        Method: POST
        URL: /api/part/json/

    Ignore:
        sudo pip install boto

    Args:
        annot_uuid_list (list of str) : list of annot UUIDs to be used in IBEIS IA
        part_uuid_list (list of str) : list of annotations UUIDs to be used in IBEIS IA
        part_bbox_list (list of 4-tuple) : list of bounding box coordinates encoded as
            a 4-tuple of the values (xtl, ytl, width, height) where xtl is the
            'top left corner, x value' and ytl is the 'top left corner, y value'.
        part_theta_list (list of float) : list of radian rotation around center.
            Defaults to 0.0 (no rotation).
        **kwargs : key-value pairs passed to the ibs.add_annots() function.
    """

    def _rectify(list_, default, length, func=None):
        if list_ is None:
            list_ = [None] * length

        ret_list = []
        for item in list_:

            if item is None:
                item = default

            if None not in [func, item]:
                item = func(item)

            ret_list.append(item)

        return ret_list

    def _verify(list_, tag, length, allow_none=False):
        length_ = len(list_)
        if length_ != length:
            message = 'The input list %s has the wrong length. Received: %d. Expected %d'
            args = (
                tag,
                length_,
                length,
            )
            raise ValueError(message % args)

        error_list = []
        for value in enumerate(list_):
            index, item = value
            if item is None and not allow_none:
                error_list.append(value)
            if isinstance(item, ParseError):
                value = (index, item.value)
                error_list.append(value)

        if len(error_list) > 0:
            message = 'The input list %s has invalid values (index, value): %r'
            args = (
                tag,
                error_list,
            )
            raise ValueError(message % args)

        return list_

    def _uuid(value):
        if value is None:
            return ParseError(value)

        if isinstance(value, six.string_types):
            value = uuid.UUID(value)

        return value

    def _bbox(value):
        if len(value) != 4:
            return ParseError(value)

        value = tuple(map(float, value))

        return value

    expected_length = len(annot_uuid_list)

    annot_uuid_list = _rectify(annot_uuid_list, None, expected_length, _uuid)
    part_bbox_list = _rectify(part_bbox_list, None, expected_length, _bbox)
    part_theta_list = _rectify(part_theta_list, None, expected_length, float)

    annot_uuid_list = _verify(annot_uuid_list, 'image_uuid_list', expected_length)
    part_bbox_list = _verify(part_bbox_list, 'part_bbox_list', expected_length)
    part_theta_list = _verify(part_theta_list, 'part_theta_list', expected_length)

    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid_list = _verify(aid_list, 'annot_uuid_list', expected_length)

    part_rowid_list = ibs.add_parts(
        aid_list, bbox_list=part_bbox_list, theta_list=part_theta_list
    )
    part_uuid_list = ibs.get_part_uuids(part_rowid_list)
    return part_uuid_list


@register_api('/api/name/json/', methods=['POST'])
def add_names_json(ibs, name_text_list, name_uuid_list=None, name_note_list=None):
    nid_list = ibs.add_names_json(
        name_text_list, name_uuid_list=name_uuid_list, name_note_list=name_note_list
    )
    return ibs.get_name_uuids(nid_list)


@register_api('/api/species/json/', methods=['POST'], __api_plural_check__=False)
def add_species_json(
    ibs,
    species_nice_list,
    species_text_list=None,
    species_code_list=None,
    species_uuid_list=None,
    species_note_list=None,
    skip_cleaning=False,
):
    species_rowid_list = ibs.add_species(
        species_nice_list,
        species_text_list=species_text_list,
        species_code_list=species_code_list,
        species_uuid_list=species_uuid_list,
        species_note_list=species_note_list,
        skip_cleaning=skip_cleaning,
    )
    return ibs.get_species_uuids(species_rowid_list)


@register_api('/api/match/json/', methods=['POST'])
def add_annotmatch_json(
    ibs,
    match_annot_uuid1_list,
    match_annot_uuid2_list,
    match_evidence_decision_list=None,
    match_meta_decision_list=None,
    match_confidence_list=None,
    match_user_list=None,
    match_tag_list=None,
    match_modified_list=None,
    match_count_list=None,
):
    aids1 = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aids2 = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)

    am_rowids = ibs.add_annotmatch_undirected(aids1, aids2)
    if match_evidence_decision_list:
        ibs.set_annotmatch_evidence_decision(am_rowids, match_evidence_decision_list)
    if match_meta_decision_list:
        ibs.set_annotmatch_meta_decision(am_rowids, match_meta_decision_list)
    if match_confidence_list:
        ibs.set_annotmatch_confidence(am_rowids, match_confidence_list)
    if match_user_list:
        ibs.set_annotmatch_reviewer(am_rowids, match_user_list)
    if match_tag_list:
        ibs.set_annotmatch_tag_text(am_rowids, match_tag_list)
    if match_modified_list:
        ibs.set_annotmatch_posixtime_modified(am_rowids, match_modified_list)
    if match_count_list:
        ibs.set_annotmatch_count(am_rowids, match_count_list)
    return list(zip(match_annot_uuid1_list, match_annot_uuid2_list))


@register_api('/api/review/json/', methods=['POST'])
def add_review_json(
    ibs,
    review_annot_uuid1_list,
    review_annot_uuid2_list,
    review_evidence_decision_list,
    review_meta_decision_list=None,
    review_uuid_list=None,
    review_user_list=None,
    review_user_confidence_list=None,
    review_tags_list=None,
    review_client_start_time_posix=None,
    review_client_end_time_posix=None,
    review_server_start_time_posix=None,
    review_server_end_time_posix=None,
):

    aids1 = ibs.get_annot_aids_from_uuid(review_annot_uuid1_list)
    aids2 = ibs.get_annot_aids_from_uuid(review_annot_uuid2_list)

    ibs.add_review(
        aids1,
        aids2,
        evidence_decision_list=review_evidence_decision_list,
        meta_decision_list=review_meta_decision_list,
        review_uuid_list=review_uuid_list,
        identity_list=review_user_list,
        user_confidence_list=review_user_confidence_list,
        tags_list=review_tags_list,
        review_client_start_time_posix=review_client_start_time_posix,
        review_client_end_time_posix=review_client_end_time_posix,
        review_server_start_time_posix=review_server_start_time_posix,
        review_server_end_time_posix=review_server_end_time_posix,
    )
    return list(zip(review_annot_uuid1_list, review_annot_uuid2_list))


@register_api('/api/imageset/json/', methods=['GET'])
def get_valid_imageset_uuids_json(ibs, **kwargs):
    imgsetid_list = ibs.get_valid_imgsetids(**kwargs)
    return ibs.get_imageset_uuid(imgsetid_list)


@register_api('/api/imageset/annot/uuid/json/', methods=['GET'])
def get_imageset_annot_uuids_json(ibs, imageset_uuid_list):
    imgsetid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    aids_list = ibs.get_imageset_aids(imgsetid_list)
    annot_uuids_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuids_list


@register_api('/api/imageset/num/annot/reviewed/json/', methods=['GET'])
def get_imageset_num_annots_reviewed_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_annots_reviewed(imageset_rowid_list)


@register_api('/api/imageset/num/image/reviewed/json/', methods=['GET'])
def get_imageset_num_imgs_reviewed_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_imgs_reviewed(imageset_rowid_list)


@register_api('/api/imageset/num/name/exemplar/json/', methods=['GET'])
def get_imageset_num_names_with_exemplar_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_names_with_exemplar(imageset_rowid_list)


@register_api('/api/imageset/num/image/json/', methods=['GET'])
def get_imageset_num_gids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_gids(imageset_rowid_list)


@register_api('/api/imageset/num/annot/json/', methods=['GET'])
def get_imageset_num_aids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_aids(imageset_rowid_list)


@register_api('/api/imageset/annot/rowid/json/', methods=['GET'])
def get_imageset_aids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_aids(imageset_rowid_list)


@register_api('/api/imageset/image/rowid/json/', methods=['GET'])
def get_imageset_gids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gids(imageset_rowid_list)


@register_api('/api/imageset/image/uuid/json/', methods=['GET'])
def get_imageset_image_uuids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_image_uuids(imageset_rowid_list)


@register_api('/api/imageset/name/rowid/json/', methods=['GET'])
def get_imageset_nids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_nids(imageset_rowid_list)


@register_api('/api/imageset/name/uuid/json/', methods=['GET'])
def get_imageset_name_uuids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_name_uuids(imageset_rowid_list)


@register_api('/api/imageset/text/json/', methods=['GET'])
def get_imageset_text_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_text(imageset_rowid_list)


@register_api('/api/imageset/rowid/uuid/json/', methods=['GET'])
def get_imageset_imgsetids_from_uuid_json(ibs, imageset_uuid_list):
    return ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)


@register_api('/api/imageset/rowid/text/json/', methods=['GET'])
def get_imageset_imgsetids_from_text_json(ibs, imageset_text_list, **kwargs):
    return ibs.get_imageset_imgsetids_from_text(imageset_text_list, **kwargs)


@register_api('/api/imageset/note/json/', methods=['GET'])
def get_imageset_note_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_note(imageset_rowid_list)


@register_api('/api/imageset/time/posix/end/json/', methods=['GET'])
def get_imageset_end_time_posix_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_end_time_posix(imageset_rowid_list)


@register_api('/api/imageset/gps/lat/json/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lats_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gps_lats(imageset_rowid_list)


@register_api('/api/imageset/gps/lon/json/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lons_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gps_lons(imageset_rowid_list)


@register_api('/api/imageset/occurrence/json/', methods=['GET'])
def get_imageset_occurrence_flags_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_occurrence_flags(imageset_rowid_list)


@register_api('/api/imageset/processed/json/', methods=['GET'])
def get_imageset_processed_flags_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_processed_flags(imageset_rowid_list)


@register_api('/api/imageset/shipped/json/', methods=['GET'])
def get_imageset_shipped_flags_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_shipped_flags(imageset_rowid_list)


@register_api('/api/imageset/time/posix/start/json/', methods=['GET'])
def get_imageset_start_time_posix_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_start_time_posix(imageset_rowid_list)


@register_api('/api/imageset/duration/json/', methods=['GET'])
def get_imageset_duration_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_duration(imageset_rowid_list)


@register_api('/api/imageset/smart/waypoint/json/', methods=['GET'])
def get_imageset_smart_waypoint_ids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_waypoint_ids(imageset_rowid_list)


@register_api('/api/imageset/smart/xml/file/name/json/', methods=['GET'])
def get_imageset_smart_xml_fnames_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_xml_fnames(imageset_rowid_list)


@register_api('/api/imageset/smart/xml/file/content/json/', methods=['GET'])
def get_imageset_smart_xml_contents_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_xml_contents(imageset_rowid_list)


@register_api('/api/image/json/<uuid>/', methods=['GET'])
def image_base64_api_json(ibs, uuid=None, thumbnail=False, fresh=False, **kwargs):
    rowid = ibs.get_image_rowid_from_uuid(uuid)
    return image_src(rowid, thumbnail=thumbnail, fresh=fresh, **kwargs)


@register_api('/api/image/json/', methods=['GET'])
def get_valid_image_uuids_json(ibs, **kwargs):
    gid_list = ibs.get_valid_gids(**kwargs)
    return ibs.get_image_uuids(gid_list)


@register_api('/api/image/dict/json/', methods=['GET'])
def get_image_uuids_with_annot_uuids(ibs, gid_list=None):
    if gid_list is None:
        gid_list = sorted(ibs.get_valid_gids())
    aids_list = ibs.get_image_aids(gid_list)
    zipped = list(zip(gid_list, aids_list))
    combined_dict = {
        str(ibs.get_image_uiids(gid)): ibs.get_annot_uuids(aid_list)
        for gid, aid_list in zipped
    }
    return combined_dict


@register_api('/api/image/rowid/uuid/json/', methods=['GET'])
def get_image_gids_from_uuid_json(ibs, image_uuid_list):
    return ibs.get_image_gids_from_uuid(image_uuid_list)


@register_api('/api/image/uri/json/', methods=['GET'])
def get_image_uris_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_uris(gid_list)


@register_api('/api/image/uri/original/json/', methods=['GET'])
def get_image_uris_original_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_uris_original(gid_list)


@register_api('/api/image/file/path/json/', methods=['GET'])
def get_image_paths_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_paths(gid_list)


@register_api('/api/image/file/hash/json/', methods=['GET'])
def get_image_hash_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_hash(gid_list, **kwargs)


@register_api('/api/image/file/name/json/', methods=['GET'])
def get_image_gnames_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_gnames(gid_list)


@register_api('/api/image/size/json/', methods=['GET'])
def get_image_sizes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_sizes(gid_list)


@register_api('/api/image/width/json/', methods=['GET'])
def get_image_widths_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_widths(gid_list)


@register_api('/api/image/height/json/', methods=['GET'])
def get_image_heights_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_heights(gid_list)


@register_api('/api/image/gps/json/', methods=['GET'], __api_plural_check__=False)
def get_image_gps_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_gps(gid_list)


@register_api('/api/image/lat/json/', methods=['GET'])
def get_image_lat_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_lat(gid_list)


@register_api('/api/image/lon/json/', methods=['GET'])
def get_image_lon_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_lon(gid_list)


@register_api('/api/image/orientation/json/', methods=['GET'])
def get_image_orientation_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_orientation(gid_list)


@register_api('/api/image/orientation/str/json/', methods=['GET'])
def get_image_orientation_str_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_orientation_str(gid_list)


@register_api('/api/image/reviewed/json/', methods=['GET'])
def get_image_reviewed_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_reviewed(gid_list)


@register_api('/api/image/detect/confidence/json/', methods=['GET'])
def get_image_detect_confidence_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_detect_confidence(gid_list)


@register_api('/api/image/note/json/', methods=['GET'])
def get_image_notes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_notes(gid_list)


@register_api('/api/image/name/rowid/json/', methods=['GET'])
def get_image_nids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_nids(gid_list)


@register_api('/api/image/name/uuid/json/', methods=['GET'])
def get_image_name_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_name_uuids(gid_list)


@register_api(
    '/api/image/species/rowid/json/', methods=['GET'], __api_plural_check__=False
)
def get_image_species_rowids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_species_rowids(gid_list)


@register_api(
    '/api/image/species/uuid/json/', methods=['GET'], __api_plural_check__=False
)
def get_image_species_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_species_uuids(gid_list)


@register_api('/api/image/imageset/rowid/json/', methods=['GET'])
def get_image_imgsetids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imgsetids(gid_list)


@register_api('/api/image/imageset/uuid/json/', methods=['GET'])
def get_image_imgset_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imgset_uuids(gid_list)


@register_api('/api/image/imageset/text/json/', methods=['GET'])
def get_image_imagesettext_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imagesettext(gid_list)


@register_api('/api/image/imageset/rowid/json/', methods=['PUT'])
def set_image_imgsetids_json(ibs, image_uuid_list, imageset_rowid_list):
    ibs.web_check_uuids(image_uuid_list, [], [])
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.set_image_imgsetids(gid_list, imageset_rowid_list)


@register_api('/api/image/imageset/uuid/json/', methods=['PUT'])
def set_image_imgset_uuids_json(ibs, image_uuid_list, imageset_uuid_list):
    ibs.web_check_uuids(image_uuid_list, [], [])
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.set_image_imgsetids(gid_list, imageset_rowid_list)


@register_api('/api/image/imageset/text/json/', methods=['PUT'])
def set_image_imagesettext_json(ibs, image_uuid_list, imageset_text_list):
    ibs.web_check_uuids(image_uuid_list, [], [])
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
    return ibs.set_image_imgsetids(gid_list, imageset_rowid_list)


@register_api('/api/image/annot/rowid/json/', methods=['GET'])
def get_image_aids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_aids(gid_list)


@register_api('/api/image/annot/uuid/json/', methods=['GET'])
def get_image_annot_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_annot_uuids(gid_list)


@register_api(
    '/api/image/annot/rowid/species/json/', methods=['GET'], __api_plural_check__=False
)
def get_image_aids_of_species_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_aids_of_species(gid_list, **kwargs)


@register_api(
    '/api/image/annot/uuid/species/json/', methods=['GET'], __api_plural_check__=False
)
def get_image_annot_uuids_of_species_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_annot_uuids_of_species(gid_list, **kwargs)


@register_api('/api/image/num/annot/json/', methods=['GET'])
def get_image_num_annotations_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_num_annotations(gid_list)


@register_api('/api/image/unixtime/json/', methods=['GET'])
def get_image_unixtimes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_unixtime(gid_list)


@register_api('/api/image/timedelta/posix/json/', methods=['GET'])
def get_image_timedelta_posix_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_timedelta_posix(gid_list, **kwargs)


@register_api('/api/image/location/code/json/', methods=['GET'])
def get_image_location_codes_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_location_codes(gid_list, **kwargs)


@register_api('/api/annot/json/', methods=['GET'])
def get_valid_annot_uuids_json(ibs, **kwargs):
    aid_list = ibs.get_valid_aids(**kwargs)
    return ibs.get_annot_uuids(aid_list)


@register_api('/api/annot/json/<uuid>/', methods=['GET'])
def annotation_src_api_json(ibs, uuid=None):
    aid = ibs.get_annot_aids_from_uuid(uuid)
    return ibs.annotation_src_api(aid)


@register_api('/api/annot/rowid/uuid/json/', methods=['GET'])
def get_annot_aids_from_uuid_json(ibs, annot_uuid_list):
    return ibs.get_annot_aids_from_uuid(annot_uuid_list)


@register_api('/api/annot/image/rowid/json/', methods=['GET'])
def get_annot_gids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_gids(aid_list)


@register_api('/api/annot/uuid/hashid/json/', methods=['GET'])
def get_annot_hashid_uuid_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_hashid_uuid(aid_list, **kwargs)


@register_api('/api/annot/exemplar/json/', methods=['POST'])
def set_exemplars_from_quality_and_viewpoint_json(
    ibs, annot_uuid_list, annot_name_list, **kwargs
):
    ibs.web_check_uuids([], annot_uuid_list, [])
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    if annot_name_list is not None:
        # Set names for query annotations
        nid_list = ibs.add_names(annot_name_list)
        ibs.set_annot_name_rowids(aid_list, nid_list)
    new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(aid_list, **kwargs)
    new_annot_uuid_list = ibs.get_annot_uuids(aid_list)
    return new_annot_uuid_list, new_flag_list


@register_api('/api/annot/bbox/json/', methods=['GET'])
def get_annot_bboxes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_bboxes(aid_list)


@register_api('/api/annot/detect/confidence/json/', methods=['GET'])
def get_annot_detect_confidence_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_detect_confidence(aid_list)


@register_api('/api/annot/exemplar/json/', methods=['GET'])
def get_annot_exemplar_flags_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_exemplar_flags(aid_list)


@register_api('/api/annot/theta/json/', methods=['GET'])
def get_annot_thetas_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_thetas(aid_list)


@register_api('/api/annot/vert/json/', methods=['GET'])
def get_annot_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_verts(aid_list)


@register_api('/api/annot/vert/rotated/json/', methods=['GET'])
def get_annot_rotated_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_rotated_verts(aid_list)


@register_api('/api/annot/yaw/json/', methods=['GET'])
def get_annot_yaws_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_yaws(aid_list)


@register_api('/api/annot/viewpoint/json/', methods=['GET'])
def get_annot_viewpoints_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_viewpoints(aid_list)


@register_api('/api/annot/num/vert/json/', methods=['GET'])
def get_annot_num_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_num_verts(aid_list)


@register_api('/api/annot/name/rowid/json/', methods=['GET'])
def get_annot_nids_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_nids(aid_list, **kwargs)


@register_api('/api/annot/name/uuid/json/', methods=['GET'])
def get_annot_name_rowids_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list, **kwargs)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/annot/name/text/json/', methods=['GET'])
def get_annot_name_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_name_texts(aid_list)


@register_api('/api/annot/note/json/', methods=['GET'])
def get_annot_notes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_notes(aid_list)


@register_api('/api/annot/species/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species(aid_list)


@register_api(
    '/api/annot/species/rowid/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_species_rowids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_rowids(aid_list)


@register_api(
    '/api/annot/species/uuid/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_species_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_uuids(aid_list)


@register_api(
    '/api/annot/species/text/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_species_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_texts(aid_list)


@register_api('/api/annot/imageset/rowid/json/', methods=['GET'])
def get_annot_imgsetids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_imgsetids(aid_list)


@register_api('/api/annot/imageset/uuid/json/', methods=['GET'])
def get_annot_imgset_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_imgset_uuids(aid_list)


@register_api('/api/annot/imageset/text/json/', methods=['GET'])
def get_annot_image_set_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_set_texts(aid_list)


@register_api('/api/annot/image/name/json/', methods=['GET'])
def get_annot_image_names_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_names(aid_list)


@register_api('/api/annot/image/unixtime/json/', methods=['GET'])
def get_annot_image_unixtimes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_unixtimes(aid_list)


@register_api('/api/annot/image/gps/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_image_gps_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_gps(aid_list)


@register_api('/api/annot/image/file/path/json/', methods=['GET'])
def get_annot_image_paths_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_paths(aid_list)


@register_api('/api/annot/image/uuid/json/', methods=['GET'])
def get_annot_image_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    image_uuid_list = [
        None if aid is None else ibs.get_annot_image_uuids(aid) for aid in aid_list
    ]
    return image_uuid_list


@register_api('/api/annot/quality/json/', methods=['GET'])
def get_annot_qualities_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_qualities(aid_list, **kwargs)


@register_api('/api/annot/quality/text/json/', methods=['GET'])
def get_annot_quality_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_quality_texts(aid_list)


@register_api('/api/annot/yaw/text/json/', methods=['GET'])
def get_annot_yaw_texts_json(ibs, annot_uuid_list):
    # DEPRICATE
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_yaw_texts(aid_list)


@register_api('/api/annot/viewpoint/text/json/', methods=['GET'])
def get_annot_viewpoint_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_viewpoint_texts(aid_list)


@register_api('/api/annot/sex/json/', methods=['GET'])
def get_annot_sex_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_sex(aid_list, **kwargs)


@register_api('/api/annot/sex/text/json/', methods=['GET'])
def get_annot_sex_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_sex_texts(aid_list)


@register_api('/api/annot/reviewed/json/', methods=['GET'])
def get_annot_reviewed_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_reviewed(aid_list)


@register_api('/api/annot/multiple/json/', methods=['GET'])
def get_annot_multiple_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_multiple(aid_list, **kwargs)


@register_api('/api/annot/interest/json/', methods=['GET'])
def get_annot_interest_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_interest(aid_list, **kwargs)


@register_api('/api/annot/image/contributor/tag/json/', methods=['GET'])
def get_annot_image_contributor_tag_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_contributor_tag(aid_list)


@register_api('/api/annot/age/months/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est(aid_list, **kwargs)


@register_api(
    '/api/annot/age/months/text/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_age_months_est_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_texts(aid_list, **kwargs)


@register_api(
    '/api/annot/age/months/min/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_age_months_est_min_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_min(aid_list, **kwargs)


@register_api(
    '/api/annot/age/months/max/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_age_months_est_max_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_max(aid_list, **kwargs)


@register_api(
    '/api/annot/age/months/min/text/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_age_months_est_min_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_min_texts(aid_list, **kwargs)


@register_api(
    '/api/annot/age/months/max/text/json/', methods=['GET'], __api_plural_check__=False
)
def get_annot_age_months_est_max_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_max_texts(aid_list, **kwargs)


@register_api('/api/annot/bbox/json/', methods=['PUT'])
def set_annot_bboxes_json(ibs, annot_uuid_list, bbox_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_bboxes(aid_list, bbox_list)


@register_api('/api/annot/theta/json/', methods=['PUT'])
def set_annot_thetas_json(ibs, annot_uuid_list, theta_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_thetas(aid_list, theta_list)


@register_api('/api/annot/viewpoint/json/', methods=['PUT'])
def set_annot_viewpoints_json(ibs, annot_uuid_list, viewpoint_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_viewpoints(aid_list, viewpoint_list)


@register_api('/api/annot/quality/text/json/', methods=['PUT'])
def set_annot_quality_texts_json(ibs, annot_uuid_list, quality_text_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_quality_texts(aid_list, quality_text_list)


@register_api('/api/annot/species/json/', methods=['PUT'], __api_plural_check__=False)
def set_annot_species_json(ibs, annot_uuid_list, species_text_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_species(aid_list, species_text_list)


@register_api('/api/annot/multiple/json/', methods=['PUT'])
def set_annot_multiple_json(ibs, annot_uuid_list, flag_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_multiple(aid_list, flag_list)


@register_api('/api/annot/interest/json/', methods=['PUT'])
def set_annot_interest_json(ibs, annot_uuid_list, flag_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_interest(aid_list, flag_list)


@register_api('/api/annot/name/text/json/', methods=['PUT'])
def set_annot_name_texts_json(ibs, annot_uuid_list, name_text_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    nid_list = ibs.get_name_rowids_from_text(name_text_list)
    return ibs.set_annot_name_rowids(aid_list, nid_list)


@register_api(
    '/api/annot/note/json/', methods=['PUT'],
)
def set_annot_note_json(ibs, annot_uuid_list, annot_note_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_notes(aid_list, annot_note_list)


@register_api('/api/annot/tags/json/', methods=['PUT'], __api_plural_check__=False)
def set_annot_tag_text_json(ibs, annot_uuid_list, annot_tags_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.set_annot_tag_text(aid_list, annot_tags_list)


@register_api('/api/name/json/', methods=['GET'])
def get_valid_name_uuids_json(ibs, **kwargs):
    nid_list = ibs.get_valid_nids(**kwargs)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/name/dict/json/', methods=['GET'])
def get_name_nids_with_gids_json(ibs, nid_list=None):
    if nid_list is None:
        nid_list = sorted(ibs.get_valid_nids())
    name_list = ibs.get_name_texts(nid_list)
    gids_list = ibs.get_name_gids(nid_list)

    zipped = list(zip(nid_list, name_list, gids_list))
    combined_dict = {
        name: (ibs.get_name_uuids(nid), ibs.get_image_uuids(gid_list))
        for nid, name, gid_list in zipped
    }
    return combined_dict


@register_api('/api/name/annot/rowid/json/', methods=['GET'])
def get_name_aids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_aids(nid_list, **kwargs)


@register_api('/api/name/annot/uuid/json/', methods=['GET'])
def get_name_annot_uuids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_annot_uuids(nid_list, **kwargs)


@register_api('/api/name/annot/rowid/exemplar/json/', methods=['GET'])
def get_name_exemplar_aids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_exemplar_aids(ibs, nid_list)


@register_api('/api/name/annot/uuid/exemplar/json/', methods=['GET'])
def get_name_exemplar_name_uuids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_exemplar_name_uuids(nid_list, **kwargs)


@register_api('/api/name/image/rowid/json/', methods=['GET'])
def get_name_gids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_gids(ibs, nid_list)


@register_api('/api/name/image/uuid/json/', methods=['GET'])
def get_name_image_uuids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_image_uuids(ibs, nid_list)


@register_api('/api/name/note/json/', methods=['GET'])
def get_name_notes_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_notes(ibs, nid_list)


@register_api('/api/name/num/annot/json/', methods=['GET'])
def get_name_num_annotations_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_num_annotations(ibs, nid_list)


@register_api('/api/name/num/annot/exemplar/json/', methods=['GET'])
def get_name_num_exemplar_annotations_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_num_exemplar_annotations(ibs, nid_list)


@register_api('/api/name/temp/json/', methods=['GET'])
def get_name_temp_flag_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_temp_flag(nid_list, **kwargs)


@register_api('/api/name/alias/text/json/', methods=['GET'], __api_plural_check__=False)
def get_name_alias_texts_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_alias_texts(ibs, nid_list)


@register_api('/api/name/text/json/', methods=['GET'])
def get_name_texts_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_texts(nid_list, **kwargs)


@register_api('/api/name/uuid/text/json/', methods=['GET'])
def get_name_rowids_from_text_json(ibs, name_text_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_text(name_text_list, **kwargs)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/name/rowid/uuid/json/', methods=['GET'])
def get_name_rowids_from_uuid_json(ibs, name_uuid_list, **kwargs):
    return ibs.get_name_rowids_from_uuid(name_uuid_list)


@register_api('/api/name/sex/json/', methods=['GET'])
def get_name_sex_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_sex(nid_list, **kwargs)


@register_api('/api/name/sex/text/json/', methods=['GET'])
def get_name_sex_text_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_sex_text(nid_list, **kwargs)


@register_api(
    '/api/name/age/months/min/json/', methods=['GET'], __api_plural_check__=False
)
def get_name_age_months_est_min_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_age_months_est_min(ibs, nid_list)


@register_api(
    '/api/name/age/months/max/json/', methods=['GET'], __api_plural_check__=False
)
def get_name_age_months_est_max_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_age_months_est_max(ibs, nid_list)


@register_api('/api/name/imageset/rowid/json/', methods=['GET'])
def get_name_imgsetids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_imgsetids(ibs, nid_list)


@register_api('/api/name/imageset/uuid/json/', methods=['GET'])
def get_name_imgset_uuids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_imgset_uuids(ibs, nid_list)


@register_api('/api/species/json/', methods=['GET'], __api_plural_check__=False)
def _get_all_species_rowids_json(ibs, **kwargs):
    species_rowid_list = ibs._get_all_species_rowids(**kwargs)
    return ibs.get_species_uuids(species_rowid_list, **kwargs)


@register_api(
    '/api/species/rowid/text/json/', methods=['GET'], __api_plural_check__=False
)
def get_species_rowids_from_text_json(ibs, species_text_list, **kwargs):
    return ibs.get_species_rowids_from_text(species_text_list, **kwargs)


@register_api(
    '/api/species/rowid/uuid/json/', methods=['GET'], __api_plural_check__=False
)
def get_species_rowids_from_uuids_json(ibs, species_uuid_list):
    return ibs.get_species_rowids_from_uuids(species_uuid_list)


@register_api('/api/species/text/json/', methods=['GET'], __api_plural_check__=False)
def get_species_texts_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_texts(species_rowid_list)


@register_api('/api/species/nice/json/', methods=['GET'], __api_plural_check__=False)
def get_species_nice_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_nice(species_rowid_list)


@register_api('/api/species/code/json/', methods=['GET'], __api_plural_check__=False)
def get_species_codes_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_codes(species_rowid_list)


@register_api('/api/species/note/json/', methods=['GET'], __api_plural_check__=False)
def get_species_notes_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_notes(species_rowid_list)


@register_api('/api/name/text/json/', methods=['PUT'])
def set_name_texts_json(ibs, name_uuid_list, name_text_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.set_name_texts(nid_list, name_text_list, **kwargs)


@register_api('/api/name/note/json/', methods=['PUT'])
def set_name_notes_json(ibs, name_uuid_list, name_note_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.set_name_notes(nid_list, name_note_list, **kwargs)


@register_api('/api/part/bbox/json/', methods=['PUT'])
def set_part_bboxes_json(ibs, part_uuid_list, bbox_list, **kwargs):
    aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    return ibs.set_part_bboxes(aid_list, bbox_list)


@register_api('/api/part/theta/json/', methods=['PUT'])
def set_part_thetas_json(ibs, part_uuid_list, theta_list, **kwargs):
    aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    return ibs.set_part_thetas(aid_list, theta_list)


@register_api('/api/part/viewpoint/json/', methods=['PUT'])
def set_part_viewpoints_json(ibs, part_uuid_list, viewpoint_list, **kwargs):
    aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    return ibs.set_part_viewpoints(aid_list, viewpoint_list)


@register_api('/api/part/quality/text/json/', methods=['PUT'])
def set_part_quality_texts_json(ibs, part_uuid_list, quality_text_list, **kwargs):
    aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    return ibs.set_part_quality_texts(aid_list, quality_text_list)


@register_api('/api/part/type/json/', methods=['PUT'], __api_plural_check__=False)
def set_part_types_json(ibs, part_uuid_list, type_text_list, **kwargs):
    aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    return ibs.set_part_types(aid_list, type_text_list)


# @register_api('/api/part/tags/json/', methods=['PUT'], __api_plural_check__=False)
# def set_part_tag_text_json(ibs, part_uuid_list, part_tags_list, **kwargs):
#     aid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
#     return ibs.set_part_tag_text(aid_list, part_tags_list)


@register_api('/api/match/decision/evidence/json/', methods=['PUT'])
def set_annotmatch_evidence_decision_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_decision_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_evidence_decision(
        annotmatch_rowid_list, match_decision_list
    )


@register_api('/api/match/decision/meta/json/', methods=['PUT'])
def set_annotmatch_meat_decision_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_decision_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_meta_decision(annotmatch_rowid_list, match_decision_list)


@register_api('/api/match/tags/json/', methods=['PUT'], __api_plural_check__=False)
def set_annotmatch_tag_text_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_tags_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_tag_text(annotmatch_rowid_list, match_tags_list)


@register_api('/api/match/confidence/json/', methods=['PUT'])
def set_annotmatch_confidence_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_confidence_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_confidence(annotmatch_rowid_list, match_confidence_list)


@register_api('/api/match/user/json/', methods=['PUT'])
def set_annotmatch_reviewer_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_user_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_reviewer(annotmatch_rowid_list, match_user_list)


@register_api('/api/match/count/json/', methods=['PUT'])
def set_annotmatch_count_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_count_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_count(annotmatch_rowid_list, match_count_list)


@register_api('/api/match/modified/json/', methods=['PUT'])
def set_annotmatch_posixtime_modified_json(
    ibs, match_annot_uuid1_list, match_annot_uuid2_list, match_modified_list, **kwargs
):
    aid1_list = ibs.get_annot_aids_from_uuid(match_annot_uuid1_list)
    aid2_list = ibs.get_annot_aids_from_uuid(match_annot_uuid2_list)
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    return ibs.set_annotmatch_posixtime_modified(
        annotmatch_rowid_list, match_modified_list
    )


@register_api('/api/contributor/rowid/uuid/json/', methods=['GET'])
def get_contributor_rowids_from_uuid_json(ibs, contributor_uuid_list):
    return ibs.get_contributor_rowid_from_uuid(contributor_uuid_list)


@register_api('/api/part/json/', methods=['GET'])
def get_valid_part_uuids_json(ibs, **kwargs):
    part_rowid_list = ibs.get_valid_part_rowids(**kwargs)
    return ibs.get_part_uuids(part_rowid_list)


@register_api('/chaos/imageset/', methods=['GET', 'POST'], __api_plural_check__=False)
def chaos_imageset(ibs):
    """
    REST:
        Method: POST
        URL: /api/image/json/

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be delete from IBEIS
    """
    from random import shuffle, randint

    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    sample = min(len(gid_list) // 2, 50)
    assert sample > 0, 'Cannot create a chaos imageset using an empty database'
    gid_list_ = gid_list[:sample]
    imagetset_name = 'RANDOM_CHAOS_TEST_IMAGESET_%08d' % (randint(0, 99999999))
    imagetset_rowid = ibs.add_imagesets(imagetset_name)
    imagetset_uuid = ibs.get_imageset_uuid(imagetset_rowid)
    ibs.add_image_relationship(gid_list_, [imagetset_rowid] * len(gid_list_))
    return imagetset_name, imagetset_uuid


@register_api('/api/labeler/cnn/json/', methods=['POST'])
def labeler_cnn_json_wrapper(ibs, annot_uuid_list, **kwargs):
    ibs.web_check_uuids([], annot_uuid_list, [])
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.labeler_cnn(aid_list, **kwargs)


@register_api('/api/imageset/json/', methods=['DELETE'])
def delete_imageset_json(ibs, imageset_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/imageset/json/

    Args:
        imageset_uuid_list (list of str) : list of imageset UUIDs to be delete from IBEIS
    """
    imgsetid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    ibs.delete_imagesets(imgsetid_list)
    return True


@register_api('/api/image/json/', methods=['DELETE'])
def delete_images_json(ibs, image_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/image/json/

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be delete from IBEIS
    """
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    ibs.delete_images(gid_list)
    return True


@register_api('/api/annot/json/', methods=['DELETE'])
def delete_annots_json(ibs, annot_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/annot/json/

    Args:
        annot_uuid_list (list of str) : list of annot UUIDs to be delete from IBEIS
    """
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    ibs.delete_annots(aid_list)
    return True


@register_api('/api/species/json/', methods=['DELETE'], __api_plural_check__=False)
def delete_species_json(ibs, species_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/species/json/

    Args:
        species_uuid_list (list of str) : list of species UUIDs to be delete from IBEIS
    """
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    ibs.delete_species(species_rowid_list)
    return True


@register_api('/api/name/json/', methods=['DELETE'])
def delete_name_json(ibs, name_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/name/json/

    Args:
        name_uuid_list (list of str) : list of name UUIDs to be delete from IBEIS
    """
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    ibs.delete_names(nid_list)
    return True


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
